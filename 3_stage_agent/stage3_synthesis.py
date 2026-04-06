#!/usr/bin/env python3
"""
Stage 3 — Synthesis & Action Plan

Pipeline:
  START -> synthesizer -> action_plan_90d -> [Gate 4] -> END

Input:  --input <path to stage 2 output folder> (loads handoff.json)
Output: Final reports (Markdown, DOCX, meta.txt) + logs
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import interrupt

from common import (
    CFG, AUTO_APPROVE, MAX_CONTEXT_CHARS, make_llm,
    _truncate, _rebuild_chat_history,
    set_output_dir, _append_log, _record, save_timings,
    save_meta, load_handoff,
    SYNTH_PROMPT, ACTION_PLAN_PROMPT,
)

# ── Stage config ──
_STAGE_CFG = CFG["stage3_synthesis"]
_MODEL = _STAGE_CFG["model"]
MAX_HUMAN_REVISION_ON_PLAN = _STAGE_CFG["max_human_revision_on_plan"]

llm_synthesizer = make_llm(_MODEL)

print("=" * 80)
print("STAGE 3 — Synthesis & Action Plan")
print(f"  Model: {_MODEL}")
print("=" * 80)

# ============================================================================
# STATE
# ============================================================================

class Stage3State(MessagesState):
    # Carried from stage 2 handoff
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
    approved_topics: list[dict]
    # Synthesis state
    recommendation: str
    action_plan: str
    action_plan_history: list[dict]
    final_output: str
    # Human feedback
    human_feedback_4: str
    # Control
    plan_revision_round: int
    status: str

# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def synthesizer(state: dict) -> dict:
    print("\n" + "=" * 80)
    print("NODE: Synthesize Recommendation (runs once)")
    print("=" * 80)

    approved = state.get("approved_topics", [])
    print(f"  Synthesizing {len(approved)} approved topics")

    approved_text = _truncate(json.dumps(approved, indent=2, default=str), label="synthesizer approved_topics")
    research_brief = state.get("research_brief", "")

    user_msg = _truncate(f"""{research_brief}

Approved topic proposals:
{approved_text}

Write a comprehensive recommendation report with all 8 sections.
IMPORTANT: Include a References section at the end with all source URLs from the findings. Cite them using [N] notation throughout the report.""", label="synthesizer input")

    # ── Log synthesizer input ──
    log = "=== SYNTHESIZER INPUT ===\n\n"
    log += f"Research Brief:\n{research_brief}\n\n"
    log += f"Approved Topics ({len(approved)}):\n"
    log += "=" * 40 + "\n"
    for entry in approved:
        log += f"\nTopic: {entry.get('topic', 'N/A')}\n"
        log += f"Debate Rounds: {entry.get('debate_rounds', '?')}\n"
        log += f"Converged: {entry.get('debate_converged', '?')}\n\n"
        log += f"Summary:\n{entry.get('summary', 'N/A')}\n\n"
        log += f"Proposal:\n{entry.get('proposal', 'N/A')}\n\n"
        log += f"Key Recommendation:\n{entry.get('key_recommendation', 'N/A')}\n\n"
        for fi in entry.get("findings", []):
            if isinstance(fi, dict):
                log += f"  [{fi.get('confidence', '?')}] {fi.get('claim', '')}\n"
                for s in fi.get("sources", []):
                    log += f"       Source: {s.get('title', '')} — {s.get('url', '')}\n"
            else:
                log += f"  - {fi}\n"
        log += f"\nCritic Assessment:\n{entry.get('critic_assessment', 'N/A')}\n"
        limitations = entry.get("limitations", [])
        if limitations:
            log += "Limitations:\n"
            for l in limitations:
                log += f"  - {l}\n"
        log += "\n" + "-" * 40 + "\n"
    _append_log("synthesizer.txt", log)

    t0 = time.time()
    resp = llm_synthesizer.invoke([SystemMessage(content=SYNTH_PROMPT),
                                    HumanMessage(content=user_msg)])
    _record("llm", "synthesizer", time.time() - t0, "recommendation report")
    recommendation = resp.content

    if not recommendation:
        print("  [WARNING] Empty response, retrying with shorter prompt...")
        t0 = time.time()
        resp = llm_synthesizer.invoke([SystemMessage(content=SYNTH_PROMPT),
                                        HumanMessage(content=f"Problem: {state.get('problem_framing', '')}\nTopics: {_truncate(approved_text, label='synthesizer retry')}\nWrite recommendation.")])
        _record("llm", "synthesizer", time.time() - t0, "recommendation report (retry)")
        recommendation = resp.content

    print(f"  -> Report generated ({len(recommendation)} chars)")

    _append_log("synthesizer.txt",
        f"\n=== SYNTHESIZER OUTPUT ===\n\n"
        f"{recommendation}\n")

    return {
        "recommendation": recommendation,
        "messages": [AIMessage(content=f"**Recommendation Report:**\n\n{recommendation}")],
    }


def action_plan_90d(state: dict) -> dict:
    print("\n" + "=" * 80)
    print("NODE: 90-Day Action Plan")
    print("=" * 80)

    n_topics = len(state.get("approved_topics", []))
    topic_names = [t.get("topic", "?") for t in state.get("approved_topics", [])]

    action_plan_history = list(state.get("action_plan_history", []))

    if not action_plan_history:
        action_plan_history.append({"role": "system", "content": ACTION_PLAN_PROMPT})
        action_plan_history.append({"role": "human", "content":
            f"""Recommendation report:
{state.get('recommendation', 'N/A')}

Market: {state.get('country_or_market', 'N/A')}
Product: {state.get('product_idea', 'N/A')}
Budget: {state.get('budget_range', 'N/A')}
Timeline: {state.get('time_horizon', 'N/A')}
Research topics covered: {', '.join(topic_names)}

Create a 90-day action plan based on the recommendation report above."""})
    else:
        action_plan_history.append({"role": "human", "content":
            "Revise the action plan based on the feedback above."})

    chat_msgs = _rebuild_chat_history(action_plan_history)
    t0 = time.time()
    resp = llm_synthesizer.invoke(chat_msgs)
    plan_round = state.get("plan_revision_round", 0)
    _record("llm", "action_plan", time.time() - t0, f"round {plan_round + 1}")
    plan = resp.content

    action_plan_history.append({"role": "ai", "content": plan})

    final = f"""{'='*60}
FINAL MBA STRATEGY REPORT
{'='*60}

{state.get('recommendation', '[No recommendation]')}

{'='*60}
90-DAY ACTION PLAN
{'='*60}

{plan}

{'='*60}
METADATA
{'='*60}
Model: {_MODEL}
Topics researched: {n_topics}
Topic names: {', '.join(topic_names)}
Web search: Tavily
"""

    print(f"  -> Action plan generated ({len(plan)} chars)")

    plan_round = state.get("plan_revision_round", 0)
    log_file = f"action_plan_{plan_round + 1}.txt"
    log = f"=== ACTION PLAN (Round {plan_round + 1}) ===\n\n"
    if plan_round == 0:
        log += f"Input:\n"
        log += f"  Market: {state.get('country_or_market', 'N/A')}\n"
        log += f"  Product: {state.get('product_idea', 'N/A')}\n"
        log += f"  Budget: {state.get('budget_range', 'N/A')}\n"
        log += f"  Timeline: {state.get('time_horizon', 'N/A')}\n"
        log += f"  Topics: {', '.join(topic_names)}\n\n"
    else:
        hf = state.get("human_feedback_4", "")
        log += f"Human Feedback:\n{hf}\n\n"
    log += f"Output:\n{plan}\n"
    _append_log(log_file, log)

    return {
        "action_plan": plan,
        "action_plan_history": action_plan_history,
        "final_output": final,
        "status": "pending_final_approval",
        "messages": [AIMessage(content=f"**90-Day Action Plan:**\n\n{plan}")],
    }


def human_gate_4(state: dict) -> dict:
    print("\n" + "=" * 80)
    print("NODE: Human Gate 4 — Final Approval")
    print("=" * 80)

    recommendation = state.get("recommendation", "")
    action_plan = state.get("action_plan", "")

    plan_round = state.get("plan_revision_round", 0)
    log_file = f"action_plan_{plan_round + 1}.txt"

    if AUTO_APPROVE:
        print("  -> Auto-approved")
        _append_log(log_file, "\n=== HUMAN GATE 4 ===\nFeedback: auto-approved\n")
        return {"human_feedback_4": "approved", "status": "finalize"}

    feedback = interrupt(
        f"--- RECOMMENDATION REPORT ---\n{recommendation}\n\n"
        f"--- 90-DAY ACTION PLAN ---\n{action_plan}\n\n"
        "---\n"
        "- Press Enter or 'approve' -> finalize and save the report\n"
        "- Or provide feedback to revise the action plan"
    )
    feedback = str(feedback).strip()

    if not feedback or feedback.lower() in ("approve", "approved", "ok", "yes", "skip", "looks good", "lgtm"):
        print("  -> Approved — finalizing")
        _append_log(log_file, "\n=== HUMAN GATE 4 ===\nFeedback: approved\n")
        return {"human_feedback_4": "approved", "status": "finalize"}

    revision_round = plan_round + 1
    if revision_round >= MAX_HUMAN_REVISION_ON_PLAN:
        print(f"  ** Max plan revisions ({MAX_HUMAN_REVISION_ON_PLAN}) reached. Forcing approve. **")
        _append_log(log_file, f"\n=== HUMAN GATE 4 ===\nFeedback: {feedback}\n(Max rounds reached — forced approve)\n")
        return {"human_feedback_4": "approved", "status": "finalize", "plan_revision_round": revision_round}

    print(f"  -> Feedback: {feedback[:100]} (revision {revision_round}/{MAX_HUMAN_REVISION_ON_PLAN})")

    action_plan_history = list(state.get("action_plan_history", []))
    action_plan_history.append({"role": "human", "content": feedback})

    return {"human_feedback_4": feedback, "plan_revision_round": revision_round, "action_plan_history": action_plan_history}


def route_after_gate_4(state: dict) -> Literal["action_plan_90d", "__end__"]:
    if state.get("human_feedback_4") == "approved":
        return "__end__"
    return "action_plan_90d"


# ============================================================================
# BUILD GRAPH
# ============================================================================

def build_graph(checkpointer=None):
    g = StateGraph(Stage3State)

    g.add_node("synthesizer", synthesizer)
    g.add_node("action_plan_90d", action_plan_90d)
    g.add_node("human_gate_4", human_gate_4)

    g.add_edge(START, "synthesizer")
    g.add_edge("synthesizer", "action_plan_90d")
    g.add_edge("action_plan_90d", "human_gate_4")
    g.add_conditional_edges("human_gate_4", route_after_gate_4, {
        "action_plan_90d": "action_plan_90d",
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

    parser = argparse.ArgumentParser(description="Stage 3 — Synthesis & Action Plan")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to stage 2 output folder (contains handoff.json)")
    parser.add_argument("--name", type=str, default=None,
                        help="Suffix appended to the output folder")
    args = parser.parse_args()

    # ── Load stage 2 handoff ──
    handoff = load_handoff(args.input)
    print(f"  Loaded handoff from: {args.input}")
    print(f"  Approved topics: {len(handoff.get('approved_topics', []))}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{ts}_{args.name}" if args.name else ts
    output_dir = Path(__file__).resolve().parent / "results" / "stage3_synthesis" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(output_dir)

    report_export.start_log(output_dir)

    t0 = time.time()

    print(f"  Auto-approve: {AUTO_APPROVE}")
    print(f"  Output dir:   {output_dir}")
    print("=" * 80)

    agent = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "stage3-run-1"}}

    initial_state = {
        "user_query": handoff.get("user_query", ""),
        "country_or_market": handoff.get("country_or_market", ""),
        "product_idea": handoff.get("product_idea", ""),
        "target_customer": handoff.get("target_customer", ""),
        "budget_range": handoff.get("budget_range", ""),
        "time_horizon": handoff.get("time_horizon", ""),
        "risk_tolerance": handoff.get("risk_tolerance", ""),
        "constraints": handoff.get("constraints", ""),
        "problem_framing": handoff.get("problem_framing", ""),
        "constraints_noted": handoff.get("constraints_noted", ""),
        "research_topics": handoff.get("research_topics", []),
        "research_brief": handoff.get("research_brief", ""),
        "approved_topics": handoff["approved_topics"],
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

    # ── Save reports ──
    # Build a config dict compatible with report_export.save_all
    stage_config = handoff.get("config", {})
    export_config = {
        "input_query": stage_config.get("input_query", handoff.get("user_query", "")),
        "agents": {
            "intake":      {"model": stage_config.get("model_intake", "?")},
            "researcher":  {"model": stage_config.get("model_researcher", "?")},
            "critic":      {"model": stage_config.get("model_critic", "?")},
            "synthesizer": {"model": _MODEL},
        },
    }
    report_export.save_all(result, export_config, output_dir, elapsed)
    save_timings()

    # ── Meta ──
    approved = handoff.get("approved_topics", [])
    topic_names = [a.get("topic", "?") for a in approved]
    save_meta([
        f"Stage:            3 — Synthesis & Action Plan",
        f"Timestamp:        {datetime.now().isoformat()}",
        f"Input (Stage 2):  {args.input}",
        f"Elapsed:          {elapsed:.1f}s",
        f"",
        f"Model:",
        f"  synthesizer   {_MODEL}",
        f"",
        f"Settings:",
        f"  Max plan revisions:   {MAX_HUMAN_REVISION_ON_PLAN}",
    ], output_dir)

    print("\n" + "=" * 80)
    print("STAGE 3 COMPLETE")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Output:  {output_dir}")
    print("=" * 80)

    report_export.stop_log()
