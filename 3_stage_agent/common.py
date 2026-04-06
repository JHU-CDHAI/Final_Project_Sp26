"""Shared utilities for the 3-stage MBA Strategy Agent.

Provides: config loading, LLM factory, Pydantic output models, web search,
context helpers, and per-phase logging/timing infrastructure.
"""

import os
import re
import sys
import json
import time
import operator
import textwrap
from pathlib import Path
from typing import Annotated, Literal

import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

import yaml
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import interrupt

from dotenv import load_dotenv

# ── Load .env from this folder ──
_this_dir = Path(__file__).resolve().parent
load_dotenv(_this_dir / ".env")

# ============================================================================
# CONFIG
# ============================================================================

_config_path = Path(os.getenv(
    "MBA_CONFIG_PATH",
    _this_dir / "config.yaml",
))
with open(_config_path) as _f:
    CFG = yaml.safe_load(_f)

OPENROUTER_BASE_URL = CFG.get("openrouter_base_url", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

AUTO_APPROVE = CFG.get("auto_approve", False)
INPUT_QUERY = CFG["input_query"]

MAX_CONTEXT_CHARS = 300_000

# ============================================================================
# LLM FACTORY
# ============================================================================

def make_llm(model_id: str) -> ChatOpenAI:
    """Create a ChatOpenAI instance routed through OpenRouter."""
    return ChatOpenAI(
        model=model_id,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        max_retries=3,
    )

# ============================================================================
# PYDANTIC OUTPUT MODELS
# ============================================================================

class IntakeOutput(BaseModel):
    user_query: str = Field(description="The core business question")
    country_or_market: str = Field(default="Not specified", description="Country or market mentioned")
    product_idea: str = Field(default="Not specified", description="Product or service idea")
    target_customer: str = Field(default="Not specified", description="Target customer segment")
    budget_range: str = Field(default="Not specified", description="Budget mentioned")
    time_horizon: str = Field(default="Not specified", description="Time horizon mentioned")
    risk_tolerance: str = Field(default="Not specified", description="Risk tolerance mentioned")
    constraints: str = Field(default="None", description="Any constraints mentioned")

class ClarifyOutput(BaseModel):
    problem_framing: str = Field(description="Clear 2-3 sentence problem statement")
    constraints_noted: str = Field(description="Key constraints identified")
    questions: list[str] = Field(default_factory=list, description="Questions for the user about unclear or missing aspects. Empty list if everything is clear.")

class TopicsOutput(BaseModel):
    topics: list[str] = Field(description="List of research topics to investigate")

class FindingSource(BaseModel):
    title: str = ""
    url: str = ""

class Finding(BaseModel):
    claim: str = Field(description="Specific research finding")
    confidence: str = Field(description="high, medium, or low")
    sources: list[FindingSource] = Field(default_factory=list)

class TopicProposalOutput(BaseModel):
    topic: str = Field(description="The topic researched")
    findings: list[Finding] = Field(description="List of findings with confidence")
    summary: str = Field(description="2-3 sentence summary")
    proposal: str = Field(description="Strategic proposal for this topic")
    key_recommendation: str = Field(description="One-line recommendation")
    gap_responses: list[str] = Field(default_factory=list, description="On revision rounds: one response per critic gap, explaining how it was addressed or why it cannot be. Empty list on first round.")

class CriticOutput(BaseModel):
    assessment: str = Field(description="Assessment of the proposal quality")
    gaps: list[str] = Field(description="Identified gaps or weaknesses that can be fixed. Empty list if converged.")
    converged: bool = Field(description="True if proposal is good enough for human review")
    revision_guidance: str = Field(description="If not converged, what to fix next round. Empty string if converged.")
    limitations: list[str] = Field(default_factory=list, description="Remaining limitations that cannot be resolved via web research. Only populated when converged.")

# ============================================================================
# TAVILY WEB SEARCH
# ============================================================================

from tavily import TavilyClient
_tavily_key = os.getenv("TAVILY_API_KEY", "")
if not _tavily_key:
    raise RuntimeError("TAVILY_API_KEY is required. Set it in .env or as an environment variable.")
_tavily = TavilyClient(api_key=_tavily_key)

def web_search(query: str, max_results: int = 3, _node: str = "") -> list[dict]:
    t0 = time.time()
    results = _tavily.search(query=query, max_results=max_results)
    _record("web_search", _node or "web_search", time.time() - t0, query[:80])
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in results.get("results", [])
    ]

# ============================================================================
# TEXT HELPERS
# ============================================================================

def _msg_text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        parts = []
        for block in c:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return " ".join(parts).strip()
    return str(c).strip()

def _truncate(text: str, max_chars: int = MAX_CONTEXT_CHARS, label: str = "text") -> str:
    if len(text) > max_chars:
        print(f"  [WARNING] {label} truncated: {len(text)} -> {max_chars} chars")
        return text[:max_chars]
    return text

def _rebuild_chat_history(history: list[dict]) -> list:
    msgs = []
    for h in history:
        role = h.get("role", "")
        content = h.get("content", "")
        if role == "system":
            msgs.append(SystemMessage(content=content))
        elif role == "human":
            msgs.append(HumanMessage(content=content))
        elif role == "ai":
            msgs.append(AIMessage(content=content))
        else:
            print(f"  [WARNING] Skipping history entry with unknown role: {role}")
    return msgs

def _build_debate_context(debate_history: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not debate_history:
        return ""
    entries = []
    char_count = 0
    for entry in reversed(debate_history):
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        content = _truncate(content, label=f"debate entry ({role})")
        entry_text = f"\n\n[{role.upper()}]:\n{content}"
        if char_count + len(entry_text) > max_chars and entries:
            break
        entries.append(entry_text)
        char_count += len(entry_text)
    entries.reverse()
    if len(entries) < len(debate_history):
        print(f"  [WARNING] debate history truncated: showing last {len(entries)} of {len(debate_history)} entries")
    truncated_note = f" (showing last {len(entries)} of {len(debate_history)} entries)" if len(entries) < len(debate_history) else ""
    context = f"\n\n--- DEBATE HISTORY{truncated_note} (retain and build on this) ---"
    context += "".join(entries)
    context += "\n--- END DEBATE HISTORY ---"
    return context

# ============================================================================
# LOGGING INFRASTRUCTURE
# ============================================================================

_output_dir: Path | None = None

def set_output_dir(path: Path):
    global _output_dir
    _output_dir = path
    (path / "logs").mkdir(parents=True, exist_ok=True)

def _append_log(filename: str, content: str):
    if _output_dir is None:
        return
    with open(_output_dir / "logs" / filename, "a", encoding="utf-8") as f:
        f.write(content)

def _topic_log_filename(idx: int, topic: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', topic.lower())[:50].strip('_')
    return f"topic_{idx+1}_{slug}.txt"

# ── Runtime tracking ──
_timings: list[dict] = []

def _record(category: str, node: str, elapsed: float, detail: str = ""):
    _timings.append({"category": category, "node": node, "elapsed": elapsed, "detail": detail})

def save_timings():
    if not _timings:
        return

    phases = [
        ("Phase 1 — Problem Intake", ["intake", "clarify_problem"]),
        ("Phase 2 — Topic Planning", ["plan_research_topics"]),
        ("Phase 3 — Research & Debate", ["research_topic_", "critic_topic_"]),
        ("Phase 4 — Synthesis & Action", ["synthesizer", "action_plan"]),
    ]

    llm_total = sum(t["elapsed"] for t in _timings if t["category"] == "llm")
    web_total = sum(t["elapsed"] for t in _timings if t["category"] == "web_search")
    llm_count = len([t for t in _timings if t["category"] == "llm"])
    web_count = len([t for t in _timings if t["category"] == "web_search"])

    log = "=== RUNTIME BREAKDOWN ===\n"
    assigned = set()
    for phase_name, prefixes in phases:
        phase_entries = []
        for i, t in enumerate(_timings):
            if i in assigned:
                continue
            if any(t["node"].startswith(p) for p in prefixes):
                phase_entries.append(t)
                assigned.add(i)
        if not phase_entries:
            continue
        phase_total = sum(t["elapsed"] for t in phase_entries)
        log += f"\n  {phase_name} ({phase_total:.1f}s)\n"
        log += f"  {'-' * 50}\n"
        for t in phase_entries:
            tag = "LLM" if t["category"] == "llm" else "Web"
            detail = f" — {t['detail']}" if t["detail"] else ""
            log += f"    [{tag:3s}] {t['elapsed']:6.1f}s  {t['node']}{detail}\n"

    log += f"\n  {'=' * 50}\n"
    log += f"  LLM total:         {llm_total:6.1f}s  ({llm_count} calls)\n"
    log += f"  Web search total:  {web_total:6.1f}s  ({web_count} calls)\n"
    log += f"  Combined:          {llm_total + web_total:6.1f}s\n"
    _append_log("timings.txt", log)
    print(f"\n  Timings saved to logs/timings.txt")

# ============================================================================
# HANDOFF I/O
# ============================================================================

def save_meta(lines: list[str], output_dir: Path):
    """Write meta.txt with key-value info for this stage."""
    path = output_dir / "meta.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Meta saved: {path}")


def save_summary_stage1(handoff: dict, output_dir: Path):
    """Write a human-readable output.md for stage 1."""
    topics = handoff.get("research_topics", [])
    topics_md = "\n".join(f"{i}. {t}" for i, t in enumerate(topics, 1))
    constraints = handoff.get("constraints_noted", "None")

    md = f"# Stage 1 — Problem Intake\n\n"
    md += f"## Business Question\n{handoff.get('user_query', 'N/A')}\n\n"
    md += f"## Problem Framing\n{handoff.get('problem_framing', 'N/A')}\n\n"
    if constraints and constraints != "None":
        md += f"**Constraints:** {constraints}\n\n"
    md += f"## Research Topics\n{topics_md}\n"

    path = output_dir / "output.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Output saved: {path}")


def save_summary_stage2(handoff: dict, output_dir: Path):
    """Write a human-readable output.md for stage 2."""
    approved = handoff.get("approved_topics", [])

    md = "# Stage 2 — Research & Debate\n"

    for i, topic in enumerate(approved, 1):
        rounds = topic.get("debate_rounds", "?")
        converged = topic.get("debate_converged", "?")
        status = "converged" if converged else "not converged"

        md += f"\n---\n\n## Topic {i}: {topic.get('topic', 'N/A')}\n"
        md += f"**Debate:** {rounds} rounds, {status}\n"

        md += f"\n### Summary\n{topic.get('summary', 'N/A')}\n"
        md += f"\n### Proposal\n{topic.get('proposal', 'N/A')}\n"
        md += f"\n### Key Recommendation\n{topic.get('key_recommendation', 'N/A')}\n"

        findings = topic.get("findings", [])
        if findings:
            md += f"\n### Findings\n"
            for f in findings:
                if isinstance(f, dict):
                    conf = f.get("confidence", "?")
                    claim = f.get("claim", "")
                    sources = f.get("sources", [])
                    source_str = ""
                    if sources:
                        src = sources[0]
                        source_str = f" ({src.get('title', '')} — {src.get('url', '')})"
                    md += f"- [{conf}] {claim}{source_str}\n"
                else:
                    md += f"- {f}\n"

        md += f"\n### Critic Assessment\n{topic.get('critic_assessment', 'N/A')}\n"

        limitations = topic.get("limitations", [])
        if limitations:
            md += f"\n### Limitations\n"
            for l in limitations:
                md += f"- {l}\n"

    path = output_dir / "output.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Output saved: {path}")


def save_handoff(data: dict, output_dir: Path):
    """Save state handoff as JSON for the next stage."""
    path = output_dir / "handoff.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Handoff saved: {path}")

def load_handoff(input_dir: Path) -> dict:
    """Load handoff.json from a previous stage's output directory."""
    path = Path(input_dir) / "handoff.json"
    if not path.exists():
        raise FileNotFoundError(f"No handoff.json found in {input_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================================
# PROMPTS (shared across stages)
# ============================================================================

INTAKE_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior business strategy consultant conducting an intake session.
    Your job is to deeply understand a client's business question, frame it
    precisely and plan research topics.

    You retain full memory of the conversation so far. Build on prior exchanges
    rather than starting over.
""")

INTAKE_PARSE_PROMPT = textwrap.dedent("""\
    Parse the user's business question into structured fields: the core question,
    country/market, product idea, target customer, budget, time horizon, risk
    tolerance, and constraints. Extract whatever is mentioned; use 'Not specified'
    for missing fields.
""")

CLARIFY_INSTRUCTION = textwrap.dedent("""\
    Based on our conversation so far, produce:
    1. A clear 2-3 sentence problem statement (approval-ready)
    2. Key constraints identified so far
    3. Questions for the user about unclear or missing aspects.
       You MUST always include at least 2-3 questions — ask about target
       customer, budget, timeline, product specifics, competitive positioning,
       or anything else that would sharpen the research. Never return an
       empty list.
""")

PLAN_TOPICS_INSTRUCTION = textwrap.dedent("""\
    Based on our agreed problem framing, generate exactly {n}
    specific, searchable research topics. Each topic should be concrete enough
    for a researcher to investigate independently.
""")

RESEARCH_PROPOSE_PROMPT = textwrap.dedent("""\
    You are a business research analyst and strategist. For the given topic:
    1. Research it thoroughly using any web results provided
    2. Produce specific findings with confidence levels
    3. Draft a strategic proposal for this topic
    4. Provide a one-line key recommendation

    You have access to the full debate history — all prior proposals, critic
    feedback, and human feedback for this topic. Build on and improve your
    prior work rather than starting from scratch.

    HIGHEST PRIORITY: If there is [HUMAN] feedback in the debate history,
    treat it as the most important input — above all critic feedback. The
    human is the client and decision-maker. Address every point they raised
    FIRST, then handle any remaining critic gaps. Structure your gap_responses
    so human feedback items appear at the top.

    IMPORTANT: If a critic has reviewed your prior proposal, you MUST directly
    address every gap and concern they raised:
    - For each gap the critic identified, write an explicit response in the
      "gap_responses" field. One entry per gap, e.g.:
      "Gap: Lacks pricing data -> Addressed: Added finding on SEK 25-45 price
       range from Euromonitor report" or
      "Gap: Missing competitor analysis -> Cannot resolve: Private competitor
       financials are unavailable via public sources; recommend hiring a
       local market research firm"
    - Add NEW findings and data points, not just restate prior ones
    - If specific data cannot be found via web search, explicitly state this
      as a data gap and recommend how the client could obtain it
    - Do NOT repeat the same findings verbatim across rounds — each revision
      must contain materially new information or explicitly acknowledge limits
    - CRITICAL: If a gap has appeared in 2+ consecutive rounds and your web
      searches keep returning the same results, you MUST mark it as
      "Cannot resolve" in gap_responses. Do NOT claim "Addressed" if you
      are providing the same data you already provided in a prior round.
      Honest acknowledgment of limits is better than pretending old data is new.
""")

TOPIC_CRITIC_PROMPT = textwrap.dedent("""\
    You are a demanding business strategy critic reviewing a per-topic proposal.
    Evaluate the proposal's quality, identify gaps, and decide if it's ready
    for human review.

    You have access to the full debate history — all prior proposals, your own
    prior critiques, and any human feedback. Use this context to track whether
    the researcher has addressed previous concerns and improved over rounds.

    CRITICAL: On the FIRST round (round 0), you must ALWAYS set converged=false.
    Your job is to push the researcher to improve. Identify specific, actionable
    gaps — missing data points, weak evidence, unsupported claims, missing
    competitor analysis, vague recommendations, etc. Be demanding.

    On subsequent rounds (round 1+), set converged=true if the researcher has
    meaningfully addressed your prior gaps and the proposal is now substantive
    and well-supported. Set converged=false if there are still significant gaps
    the researcher can realistically fill with additional web research.

    IMPORTANT convergence rules:
    - If the researcher has explicitly stated that certain data is unavailable
      via web search and recommended primary research methods, accept that —
      do NOT keep requesting the same unfillable gap across rounds
    - CRITICAL: If you are requesting the SAME gap (or substantially similar
      wording) for a 2nd time, you MUST set converged=true. The researcher
      has already tried and cannot find this data. Continuing to ask for it
      wastes rounds. Move the unresolved gap to limitations and converge.
    - Look at the debate history — if you see the same gap appearing in your
      prior critiques, that is your signal to converge, not to ask again.
    - In the first half of the debate (before round {{half_max}}), you MAY raise
      new gaps you missed before — being thorough is more important than being consistent
    - In the second half (round {{half_max}}+), do NOT raise new gaps that weren't
      in your previous critique unless the revision introduced new problems
    - Your job is to improve the proposal, not to demand perfection
    - When converged=true: gaps MUST be empty, revision_guidance MUST be
      empty. Any remaining concerns that can't be resolved via web research
      go in the "limitations" field instead.
    - Only populate the "limitations" field with items the RESEARCHER has
      explicitly stated cannot be resolved via web search or further analysis
      (look for their gap_responses). Do NOT invent limitations on your own.
      If the researcher hasn't flagged anything as unresolvable, return an
      empty limitations list.
""")

SYNTH_PROMPT = textwrap.dedent("""\
    You are a senior business consultant writing a final recommendation report.
    You have approved proposals for multiple research topics. Synthesize them into:
    1. **Executive Summary** (5 bullet points)
    2. **Problem Framing & Assumptions**
    3. **Evidence Snapshot** (by topic, with key findings and citations)
    4. **Strategic Recommendations** (one per topic, with tradeoffs)
    5. **Integrated Strategy & Rationale** (how the pieces fit together)
    6. **Risks & Mitigations**
    7. **Known Limitations & Data Gaps** (from critic reviews — what couldn't be verified)
    8. **References** — list ALL source URLs from the research findings. Format each
       as a numbered entry: [N] Title — URL. These are the actual web sources the
       researcher found. In sections 3 and 4, cite sources using [N] notation that
       corresponds to this reference list.
    Be specific, evidence-based, actionable. Use markdown.
    Each topic entry includes a "limitations" field listing data gaps the critic
    identified as unresolvable via web research. Surface these honestly in section 7
    and factor them into your risk assessment in section 6.
    Topics also have "debate_converged" — if false, the debate hit the max round
    limit without full agreement, so treat those findings with extra caution.
    Each finding has a "sources" list with "title" and "url" — use these for the
    References section. Do NOT invent or hallucinate URLs.

    FORMATTING RULES (the output will be converted to Word DOCX and PDF):
    - Do NOT use markdown tables — they render poorly in Word. Present tabular
      data as bullet-point lists or numbered lists instead.
    - Use headings (#, ##, ###), bold (**), italic (*), and bullet/numbered
      lists for structure. These convert cleanly.
""")

ACTION_PLAN_PROMPT = textwrap.dedent("""\
    You are a business execution planner. Create a 90-day action plan:
    ### Days 1-30: Foundation & Validation (3-5 actions, milestones, KPIs)
    ### Days 31-60: Build & Test (3-5 actions, milestones, KPIs)
    ### Days 61-90: Launch & Measure (3-5 actions, milestones, KPIs)
    ### Data Gaps & Next Steps
    The recommendation report may include known limitations and data gaps from
    the research phase. Incorporate specific actions to address these gaps
    (e.g., commission a market study, run a focus group, hire a local consultant).
    Be specific to the business context. Use markdown.

    FORMATTING RULES (the output will be converted to Word DOCX and PDF):
    - Do NOT use markdown tables — they render poorly in Word. Present tabular
      data as bullet-point lists or numbered lists instead.
    - Use headings (#, ##, ###), bold (**), italic (*), and bullet/numbered
      lists for structure. These convert cleanly.
""")
