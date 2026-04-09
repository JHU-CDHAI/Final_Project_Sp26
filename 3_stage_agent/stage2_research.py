#!/usr/bin/env python3
"""
Stage 2 — Research & Debate

Pipeline:
  START -> [research_and_propose <-> topic_critic -> Gate 3] per topic -> END

Input:  --input <path to stage 1 output folder> (loads handoff.json)
Output: <output_dir>/handoff.json containing approved_topics + context
        for stage 3 to consume.
"""

import json
import time
import operator
from pathlib import Path
from datetime import datetime
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import interrupt

from common import (
    CFG, AUTO_APPROVE, MAX_CONTEXT_CHARS, make_llm,
    TopicProposalOutput, CriticOutput,
    _truncate, _build_debate_context,
    web_search,
    set_output_dir, _append_log, _topic_log_filename, _record, save_timings,
    save_handoff, save_meta, save_summary_stage2, load_handoff,
    RESEARCH_PROPOSE_PROMPT, TOPIC_CRITIC_PROMPT,
)

# ── Stage config ──
_STAGE_CFG = CFG["stage2_research"]
_MODEL_RESEARCHER = _STAGE_CFG["model_researcher"]
_MODEL_CRITIC = _STAGE_CFG["model_critic"]
MAX_WEB_SEARCH_CT = _STAGE_CFG["max_web_search_ct"]
MAX_DEBATE_ROUNDS = _STAGE_CFG["max_debate_rounds"]
MAX_HUMAN_REVISION_ON_PROPOSAL = _STAGE_CFG["max_human_revision_on_proposal"]

llm_researcher = make_llm(_MODEL_RESEARCHER)
llm_critic = make_llm(_MODEL_CRITIC)

print("=" * 80)
print("STAGE 2 — Research & Debate")
print(f"  Researcher: {_MODEL_RESEARCHER}")
print(f"  Critic:     {_MODEL_CRITIC}")
print("=" * 80)

# ============================================================================
# STATE
# ============================================================================

class Stage2State(MessagesState):
    # Carried from stage 1 handoff
    user_query: str
    country_or_market: str
    product_idea: str
    target_customer: str
    budget_range: str
    time_horizon: str
    risk_tolerance: str
    constraints: str
    problem_framing: str
    constraints_noted: str
    research_topics: list[str]
    research_brief: str
    # Per-topic debate state
    current_topic_idx: int
    current_debate_round: int
    current_topic_proposal: str
    current_topic_critique: str
    current_topic_limitations: list[str]
    debate_converged: bool
    debate_history: list[dict]
    # Approved topics (append-only)
    approved_topics: Annotated[list[dict], operator.add]
    # Human feedback
    human_feedback_3: str
    # Control
    proposal_revision_round: int
    status: str

# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def research_and_propose(state: dict) -> dict:
    idx = state["current_topic_idx"]
    topics = state["research_topics"]
    topic = topics[idx]
    debate_round = state.get("current_debate_round", 0)

    print("\n" + "=" * 80)
    print(f"NODE: Research & Propose — Topic {idx+1}/{len(topics)}: {topic}")
    print(f"  Debate round: {debate_round}")
    print("=" * 80)

    search_context = ""
    market = state.get("country_or_market", "")
    critique = state.get("current_topic_critique", "")

    if debate_round == 0 or not critique:
        query = f"{topic} {market}" if market else topic
        try:
            results = web_search(query, max_results=MAX_WEB_SEARCH_CT, _node=f"research_topic_{idx+1}")
        except Exception as e:
            print(f"  [WARNING] Web search failed: {e}")
            results = []
        if results:
            search_context = "\n\nWeb search results:\n" + "\n".join(
                f"- [{r['title']}]({r['url']}): {_truncate(r['content'], label='web search snippet')}" for r in results)
            print(f"  Found {len(results)} web sources (broad search)")
    else:
        gap_queries = []
        for line in critique.split("\n"):
            line = line.strip()
            if line and len(line) > 20:
                gap_query = f"{line[:100]} {market} {topic[:50]}" if market else f"{line[:100]} {topic[:50]}"
                gap_queries.append(gap_query)

        all_results = []
        per_gap = max(1, MAX_WEB_SEARCH_CT // max(len(gap_queries), 1))
        for gq in gap_queries[:MAX_WEB_SEARCH_CT]:
            try:
                results = web_search(gq, max_results=per_gap, _node=f"research_topic_{idx+1}_gap")
                all_results.extend(results)
            except Exception:
                pass
            if len(all_results) >= MAX_WEB_SEARCH_CT:
                break
        all_results = all_results[:MAX_WEB_SEARCH_CT]

        if all_results:
            search_context = "\n\nWeb search results (targeted at critic gaps):\n" + "\n".join(
                f"- [{r['title']}]({r['url']}): {_truncate(r['content'], label='web search snippet')}" for r in all_results)
            print(f"  Found {len(all_results)} web sources (targeted gap search)")

    debate_history = state.get("debate_history", [])
    debate_context = _build_debate_context(debate_history)

    research_brief = state.get("research_brief", "")
    user_msg = f"""Research this topic: {topic}

{research_brief}
{search_context}{debate_context}"""

    try:
        structured = llm_researcher.with_structured_output(TopicProposalOutput)
        t0 = time.time()
        result = structured.invoke([SystemMessage(content=RESEARCH_PROPOSE_PROMPT),
                                    HumanMessage(content=user_msg)])
        _record("llm", f"research_topic_{idx+1}", time.time() - t0, f"round {debate_round+1}")
        proposal_dict = result.model_dump()
        proposal_dict["findings"] = [f.model_dump() if hasattr(f, "model_dump") else f for f in result.findings]
        gap_responses_text = ""
        if result.gap_responses:
            gap_responses_text = (
                f"\n\n**Gap Responses ({len(result.gap_responses)}):**\n"
                + "\n".join(f"- {r}" for r in result.gap_responses)
            )
        proposal_text = (
            f"**Topic: {result.topic}**\n\n"
            f"**Summary:** {result.summary}\n\n"
            f"**Proposal:** {result.proposal}\n\n"
            f"**Key Recommendation:** {result.key_recommendation}\n\n"
            f"**Findings ({len(result.findings)}):**\n"
            + "\n".join(f"- [{f.confidence}] {f.claim}" for f in result.findings)
            + gap_responses_text
        )
    except Exception as e:
        print(f"  [Structured output failed: {e}] Falling back to text")
        t0 = time.time()
        resp = llm_researcher.invoke([SystemMessage(content=RESEARCH_PROPOSE_PROMPT),
                           HumanMessage(content=user_msg)])
        _record("llm", f"research_topic_{idx+1}", time.time() - t0, f"round {debate_round+1} (fallback)")
        proposal_text = resp.content
        proposal_dict = {"topic": topic, "findings": [], "summary": _truncate(resp.content, label="proposal fallback summary"),
                         "proposal": resp.content, "key_recommendation": "See proposal text"}

    print(f"  -> Proposal generated ({len(proposal_text)} chars)")
    print()
    print(f"  Summary: {proposal_dict.get('summary', 'N/A')}")
    print(f"  Key Recommendation: {proposal_dict.get('key_recommendation', 'N/A')}")
    findings_list = proposal_dict.get("findings", [])
    if findings_list:
        print(f"  Findings ({len(findings_list)}):")
        for f in findings_list[:5]:
            claim = f.get("claim", "") if isinstance(f, dict) else str(f)
            conf = f.get("confidence", "?") if isinstance(f, dict) else "?"
            print(f"    [{conf}] {claim}")
    gap_responses_list = proposal_dict.get("gap_responses", [])
    if gap_responses_list:
        print(f"  Gap Responses ({len(gap_responses_list)}):")
        for r in gap_responses_list:
            print(f"    -> {r}")

    new_debate_history = list(debate_history) + [
        {"role": "researcher", "content": proposal_text}
    ]

    log_file = _topic_log_filename(idx, topic)
    log = f"=== RESEARCH & PROPOSE (Round {debate_round + 1}) ===\n\n"
    log += f"Topic: {proposal_dict.get('topic', topic)}\n\n"
    log += f"Summary:\n{proposal_dict.get('summary', 'N/A')}\n\n"
    log += f"Proposal:\n{proposal_dict.get('proposal', 'N/A')}\n\n"
    log += f"Key Recommendation:\n{proposal_dict.get('key_recommendation', 'N/A')}\n\n"
    for fi in proposal_dict.get("findings", []):
        if isinstance(fi, dict):
            log += f"  [{fi.get('confidence', '?')}] {fi.get('claim', '')}\n"
            for s in fi.get("sources", []):
                log += f"       Source: {s.get('title', '')} — {s.get('url', '')}\n"
        else:
            log += f"  - {fi}\n"
    for r in proposal_dict.get("gap_responses", []):
        log += f"  Gap Response: {r}\n"
    _append_log(log_file, log + "\n")

    return {
        "current_topic_proposal": json.dumps(proposal_dict, default=str),
        "debate_converged": False,
        "debate_history": new_debate_history,
        "messages": [AIMessage(content=f"**[Topic {idx+1}/{len(topics)} — Round {debate_round+1}]**\n\n{proposal_text}")],
    }


def topic_critic(state: dict) -> dict:
    idx = state["current_topic_idx"]
    topics = state["research_topics"]
    topic = topics[idx]
    debate_round = state.get("current_debate_round", 0)

    print("\n" + "=" * 80)
    print(f"NODE: Topic Critic — Topic {idx+1}/{len(topics)}: {topic}")
    print(f"  Debate round: {debate_round}")
    print("=" * 80)

    proposal = state.get("current_topic_proposal", "")

    debate_history = state.get("debate_history", [])
    debate_context = _build_debate_context(debate_history)

    research_brief = state.get("research_brief", "")
    user_msg = f"""Topic: {topic}

{research_brief}

Current proposal:
{_truncate(proposal, label="proposal for critic")}

Debate round: {debate_round + 1} of max {MAX_DEBATE_ROUNDS}
{debate_context}"""

    try:
        structured = llm_critic.with_structured_output(CriticOutput)
        critic_prompt = TOPIC_CRITIC_PROMPT.replace("{{half_max}}", str(MAX_DEBATE_ROUNDS // 2))
        t0 = time.time()
        result = structured.invoke([SystemMessage(content=critic_prompt),
                                    HumanMessage(content=user_msg)])
        _record("llm", f"critic_topic_{idx+1}", time.time() - t0, f"round {debate_round+1}")
        assessment = result.assessment
        gaps = result.gaps
        converged = result.converged
        revision_guidance = result.revision_guidance
        limitations = result.limitations
    except Exception as e:
        print(f"  [Structured output failed: {e}] Assuming converged")
        assessment = "Unable to parse critic output; treating as converged."
        gaps = []
        converged = True
        revision_guidance = ""
        limitations = []

    critique_text = f"**Assessment:** {assessment}\n\n**Converged:** {'Yes' if converged else 'No'}"
    if not converged:
        critique_text += f"\n\n**Gaps:** {', '.join(gaps) if gaps else 'None'}"
        if revision_guidance:
            critique_text += f"\n\n**Revision guidance:** {revision_guidance}"
    if converged and limitations:
        critique_text += f"\n\n**Limitations:** {', '.join(limitations)}"

    print(f"  -> Converged: {converged}")
    print(f"  Assessment: {assessment}")
    if not converged:
        print(f"  Gaps: {len(gaps) if gaps else 'None'}")
        for g in gaps:
            print(f"    - {g}")
        print(f"  Guidance: {revision_guidance if revision_guidance else 'None'}")
    if converged and limitations:
        print(f"  Limitations: {len(limitations)}")
        for l in limitations:
            print(f"    - {l}")

    new_debate_history = list(debate_history) + [
        {"role": "critic", "content": critique_text}
    ]

    log_file = _topic_log_filename(idx, topic)
    log = f"=== CRITIC (Round {debate_round + 1}) ===\n\n"
    log += f"Assessment:\n{assessment}\n\n"
    log += f"Converged: {converged}\n\n"
    if gaps:
        log += "Gaps:\n"
        for g in gaps:
            log += f"  - {g}\n"
        log += "\n"
    if revision_guidance:
        log += f"Revision Guidance:\n{revision_guidance}\n\n"
    if limitations:
        log += "Limitations:\n"
        for l in limitations:
            log += f"  - {l}\n"
        log += "\n"
    _append_log(log_file, log)

    if converged:
        stored_critique = f"{assessment}\n\nLimitations: {', '.join(limitations) if limitations else 'None'}"
    else:
        stored_critique = f"{assessment}\n\nGaps: {', '.join(gaps) if gaps else 'None'}\n\nGuidance: {revision_guidance if revision_guidance else 'None'}"

    return {
        "current_topic_critique": stored_critique,
        "current_topic_limitations": limitations,
        "debate_converged": converged,
        "current_debate_round": debate_round + 1,
        "debate_history": new_debate_history,
        "messages": [AIMessage(content=f"**[Critic — Topic {idx+1}/{len(topics)}]**\n\n{critique_text}")],
    }


def route_after_critic(state: dict) -> Literal["human_gate_3", "research_and_propose"]:
    if state.get("debate_converged", False):
        return "human_gate_3"
    if state.get("current_debate_round", 0) >= MAX_DEBATE_ROUNDS:
        print(f"  [Max debate rounds ({MAX_DEBATE_ROUNDS}) reached — moving to human gate]")
        return "human_gate_3"
    return "research_and_propose"


def _build_approved_entry(state: dict, topic: str) -> dict:
    try:
        proposal_data = json.loads(state.get("current_topic_proposal", "{}"))
    except (json.JSONDecodeError, TypeError):
        proposal_data = {"topic": topic, "proposal": state.get("current_topic_proposal", "")}
    return {
        "topic": topic,
        "proposal": proposal_data.get("proposal", ""),
        "key_recommendation": proposal_data.get("key_recommendation", ""),
        "summary": proposal_data.get("summary", ""),
        "findings": proposal_data.get("findings", []),
        "critic_assessment": state.get("current_topic_critique", ""),
        "limitations": state.get("current_topic_limitations", []),
        "debate_rounds": state.get("current_debate_round", 1),
        "debate_converged": state.get("debate_converged", False),
    }


_DEBATE_RESET = {
    "current_debate_round": 0,
    "current_topic_proposal": "",
    "current_topic_critique": "",
    "current_topic_limitations": [],
    "proposal_revision_round": 0,
    "debate_converged": False,
    "debate_history": [],
}


def human_gate_3(state: dict) -> dict:
    idx = state["current_topic_idx"]
    topics = state["research_topics"]
    topic = topics[idx]

    print("\n" + "=" * 80)
    print(f"NODE: Human Gate 3 — Topic {idx+1}/{len(topics)}: {topic}")
    print("=" * 80)

    print()
    proposal_raw = state.get("current_topic_proposal", "")
    try:
        proposal_data = json.loads(proposal_raw)
        proposal_display = (
            f"Summary: {proposal_data.get('summary', 'N/A')}\n\n"
            f"Proposal: {proposal_data.get('proposal', 'N/A')}\n\n"
            f"Key Recommendation: {proposal_data.get('key_recommendation', 'N/A')}"
        )
    except (json.JSONDecodeError, TypeError):
        proposal_display = _truncate(proposal_raw, label="proposal display")

    critique_display = state.get("current_topic_critique", "No critique available")
    debate_rounds = state.get("current_debate_round", 0)
    converged = state.get("debate_converged", False)
    converge_status = "Converged" if converged else f"Not converged (max {MAX_DEBATE_ROUNDS} rounds reached)"

    if AUTO_APPROVE:
        print("  -> Auto-approved")
        feedback = "approved"
    else:
        limitations = state.get("current_topic_limitations", [])
        limitations_note = ""
        if limitations:
            limitations_note = (
                "\n\n--- LIMITATIONS (cannot be resolved via web research) ---\n"
                + "\n".join(f"  - {l}" for l in limitations)
            )

        feedback = interrupt(
            f"Topic {idx+1}/{len(topics)}: '{topic}' ({debate_rounds} debate rounds, {converge_status})\n\n"
            f"--- PROPOSAL ---\n{proposal_display}\n\n"
            f"--- CRITIC ---\n{critique_display}"
            f"{limitations_note}\n\n"
            "---\n"
            "- Press Enter or 'approve' -> accept and move to next topic\n"
            "- Or provide feedback to revise this topic's proposal\n"
            "  (Note: limitations listed above require primary research and cannot be addressed by further web search)"
        )
        feedback = str(feedback).strip()

    is_approved = not feedback or feedback.lower() in (
        "approve", "approved", "ok", "yes", "skip", "looks good", "lgtm"
    )

    log_file = _topic_log_filename(idx, topic)

    if is_approved:
        print(f"  -> Approved topic {idx+1}")
        _append_log(log_file, f"=== HUMAN GATE 3 ===\nFeedback: approved\n\n")
        return {
            "human_feedback_3": "approved",
            "approved_topics": [_build_approved_entry(state, topic)],
            "current_topic_idx": idx + 1,
            **_DEBATE_RESET,
        }
    else:
        revision_round = state.get("proposal_revision_round", 0) + 1
        if revision_round >= MAX_HUMAN_REVISION_ON_PROPOSAL:
            print(f"  ** Max human revisions ({MAX_HUMAN_REVISION_ON_PROPOSAL}) reached. Forcing approve. **")
            _append_log(log_file, f"=== HUMAN GATE 3 ===\nFeedback: {feedback}\n(Max rounds reached — forced approve)\n\n")
            return {
                "human_feedback_3": "approved",
                "approved_topics": [_build_approved_entry(state, topic)],
                "current_topic_idx": idx + 1,
                **_DEBATE_RESET,
            }

        print(f"  -> Revise requested: {feedback[:100]} (revision {revision_round}/{MAX_HUMAN_REVISION_ON_PROPOSAL})")

        human_critique = (
            f"**Assessment:** Human reviewer requested revisions.\n\n"
            f"**Gaps:**\n- {feedback}\n\n"
            f"**Revision guidance:** Address the human reviewer's feedback above."
        )

        debate_history = list(state.get("debate_history", []))
        debate_history.append({"role": "human", "content": human_critique})

        _append_log(log_file, f"=== HUMAN GATE 3 (Revision {revision_round}) ===\nFeedback: {feedback}\n\n")

        return {
            "human_feedback_3": feedback,
            "current_topic_critique": human_critique,
            "proposal_revision_round": revision_round,
            "current_debate_round": 0,
            "debate_converged": False,
            "debate_history": debate_history,
        }


def route_after_gate_3(state: dict) -> Literal["research_and_propose", "__end__"]:
    if state.get("human_feedback_3", "") not in ("approved", ""):
        return "research_and_propose"

    idx = state.get("current_topic_idx", 0)
    total = len(state.get("research_topics", []))
    if idx < total:
        return "research_and_propose"
    return "__end__"


# ============================================================================
# BUILD GRAPH
# ============================================================================

def build_graph(checkpointer=None):
    g = StateGraph(Stage2State)

    g.add_node("research_and_propose", research_and_propose)
    g.add_node("topic_critic", topic_critic)
    g.add_node("human_gate_3", human_gate_3)

    g.add_edge(START, "research_and_propose")
    g.add_edge("research_and_propose", "topic_critic")
    g.add_conditional_edges("topic_critic", route_after_critic, {
        "human_gate_3": "human_gate_3",
        "research_and_propose": "research_and_propose",
    })
    g.add_conditional_edges("human_gate_3", route_after_gate_3, {
        "research_and_propose": "research_and_propose",
        "__end__": END,
    })

    return g.compile(checkpointer=checkpointer)


# ============================================================================
# TERMINAL RUNNER
# ============================================================================

if __name__ == "__main__":
    import argparse
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    import report_export

    parser = argparse.ArgumentParser(description="Stage 2 — Research & Debate")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to stage 1 output folder (contains handoff.json)")
    parser.add_argument("--name", type=str, default=None,
                        help="Suffix appended to the output folder")
    args = parser.parse_args()

    # ── Load stage 1 handoff ──
    handoff = load_handoff(args.input)
    print(f"  Loaded handoff from: {args.input}")
    print(f"  Topics: {handoff['research_topics']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{ts}_{args.name}" if args.name else ts
    output_dir = Path(__file__).resolve().parent / "results" / "stage2_research" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(output_dir)

    report_export.start_log(output_dir)

    t0 = time.time()

    print(f"  Auto-approve: {AUTO_APPROVE}")
    print(f"  Output dir:   {output_dir}")
    print("=" * 80)

    agent = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "stage2-run-1"}}

    # Seed state from handoff
    initial_state = {
        "research_topics": handoff["research_topics"],
        "research_brief": handoff["research_brief"],
        "current_topic_idx": 0,
        "current_debate_round": 0,
        "approved_topics": [],
    }

    result = agent.invoke(initial_state, config)

    while True:
        snapshot = agent.get_state(config)
        if not snapshot.next:
            break
        for task in snapshot.tasks:
            if hasattr(task, 'interrupts'):
                for intr in task.interrupts:
                    print("\n" + "=" * 80)
                    print(intr.value)
                    print("=" * 80)
        feedback = input("\n> ").strip()
        if not feedback:
            feedback = "approved"
        result = agent.invoke(Command(resume=feedback), config)

    elapsed = time.time() - t0

    # ── Save handoff for stage 3 ──
    handoff_out = {
        "research_topics": handoff["research_topics"],
        "research_brief": handoff["research_brief"],
        "approved_topics": result.get("approved_topics", []),
        "config": {
            "model_researcher": _MODEL_RESEARCHER,
            "model_critic": _MODEL_CRITIC,
        },
    }
    save_handoff(handoff_out, output_dir)
    save_summary_stage2(handoff_out, output_dir)
    save_timings()

    # ── Meta ──
    approved = result.get("approved_topics", [])
    topics = handoff["research_topics"]
    save_meta([
        f"Stage:            2 — Research & Debate",
        f"Timestamp:        {datetime.now().isoformat()}",
        f"Input (Stage 1):  {args.input}",
        f"Elapsed:          {elapsed:.1f}s",
        f"",
        f"Models:",
        f"  researcher    {_MODEL_RESEARCHER}",
        f"  critic        {_MODEL_CRITIC}",
        f"",
        f"Settings:",
        f"  Max web search results:     {MAX_WEB_SEARCH_CT}",
        f"  Max debate rounds:          {MAX_DEBATE_ROUNDS}",
        f"  Max proposal revisions:     {MAX_HUMAN_REVISION_ON_PROPOSAL}",
    ], output_dir)

    print("\n" + "=" * 80)
    print("STAGE 2 COMPLETE")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Output:  {output_dir}")
    print(f"\nTo continue:\n  python stage3_synthesis.py --input {output_dir}")
    print("=" * 80)

    report_export.stop_log()
