#!/usr/bin/env python3
"""
Stage 1 — Problem Intake & Topic Planning

Pipeline:
  START -> intake -> clarify_problem -> [Gate 1] -> plan_research_topics -> [Gate 2] -> END

Output: <output_dir>/handoff.json containing research_brief, topics, and parsed fields
        for stage 2 to consume.
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
    CFG, AGENTS, AUTO_APPROVE, INPUT_QUERY,
    MAX_CLARIFY_ROUNDS, MAX_RESEARCH_TOPICS, MAX_TOPICS_REVISION,
    llm_intake,
    IntakeOutput, ClarifyOutput, TopicsOutput,
    _msg_text, _truncate, _rebuild_chat_history,
    set_output_dir, _append_log, _record, save_timings, save_handoff, save_meta,
    INTAKE_SYSTEM_PROMPT, INTAKE_PARSE_PROMPT, CLARIFY_INSTRUCTION,
    PLAN_TOPICS_INSTRUCTION,
)

print("=" * 80)
print("STAGE 1 — Problem Intake & Topic Planning")
print("=" * 80)

# ============================================================================
# STATE
# ============================================================================

class Stage1State(MessagesState):
    user_query: str
    country_or_market: str
    product_idea: str
    target_customer: str
    budget_range: str
    time_horizon: str
    risk_tolerance: str
    constraints: str
    intake_chat_history: list[dict]
    problem_framing: str
    constraints_noted: str
    questions: list[str]
    research_topics: list[str]
    research_brief: str
    human_feedback_1: str
    human_feedback_2: str
    clarify_round: int
    topics_revision_round: int
    status: str

# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def intake(state: dict) -> dict:
    print("\n" + "=" * 80)
    print("NODE: Intake — Parse Input")
    print("=" * 80)

    user_text = state.get("user_query", "")
    if not user_text:
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                user_text = _msg_text(m)
                break
            if isinstance(m, dict) and m.get("type") == "human":
                user_text = m.get("content", "")
                break
            if hasattr(m, "type") and m.type == "human":
                user_text = _msg_text(m)
                break

    if not user_text:
        print("  -> No input found, skipping")
        return {}

    print(f"  Input: {user_text[:100]}...")

    try:
        structured = llm_intake.with_structured_output(IntakeOutput)
        t0 = time.time()
        result = structured.invoke([
            SystemMessage(content=INTAKE_PARSE_PROMPT),
            HumanMessage(content=user_text)])
        _record("llm", "intake", time.time() - t0, "parse input")
        parsed = {
            "user_query": result.user_query,
            "country_or_market": result.country_or_market,
            "product_idea": result.product_idea,
            "target_customer": result.target_customer,
            "budget_range": result.budget_range,
            "time_horizon": result.time_horizon,
            "risk_tolerance": result.risk_tolerance,
            "constraints": result.constraints,
        }
    except Exception as e:
        print(f"  [Structured output failed: {e}] Using raw text as query")
        parsed = {"user_query": user_text}

    chat_history = [
        {"role": "system", "content": INTAKE_SYSTEM_PROMPT},
        {"role": "human", "content": user_text},
    ]

    log = "=== INTAKE ===\n\n"
    for k, v in parsed.items():
        log += f"{k}: {v}\n"
    _append_log("problem_intake.txt", log + "\n")

    return {**parsed, "intake_chat_history": chat_history}


def clarify_problem(state: dict) -> dict:
    print("\n" + "=" * 80)
    print("NODE: Clarify Problem")
    print("=" * 80)

    history = list(state.get("intake_chat_history", []))

    hf = state.get("human_feedback_1", "")
    if hf and hf != "approved":
        history.append({"role": "human", "content": hf})

    history.append({"role": "human", "content": CLARIFY_INSTRUCTION})

    chat_msgs = _rebuild_chat_history(history)
    try:
        structured = llm_intake.with_structured_output(ClarifyOutput)
        t0 = time.time()
        result = structured.invoke(chat_msgs)
        _record("llm", "clarify_problem", time.time() - t0, "clarify")
        framing = result.problem_framing
        constraints_noted = result.constraints_noted
        questions = result.questions
    except Exception as e:
        print(f"  [Structured output failed: {e}] Falling back to text")
        t0 = time.time()
        resp = llm_intake.invoke(chat_msgs)
        _record("llm", "clarify_problem", time.time() - t0, "clarify (fallback)")
        framing = resp.content
        constraints_noted = ""
        questions = []

    print(f"  -> {len(questions)} questions generated")

    log = "=== CLARIFY PROBLEM ===\n\n"
    log += f"Problem Framing:\n{framing}\n\n"
    log += f"Constraints:\n{constraints_noted}\n\n"
    log += "Questions:\n"
    for i, q in enumerate(questions, 1):
        log += f"  {i}. {q}\n"
    _append_log("problem_intake.txt", log + "\n")

    chat_summary = f"**Problem Framing:**\n{framing}"
    if constraints_noted:
        chat_summary += f"\n\n**Constraints:**\n{constraints_noted}"
    if questions:
        questions_text = "\n".join(f"  {i}. {q}" for i, q in enumerate(questions, 1))
        chat_summary += f"\n\n**Questions for you:**\n{questions_text}"

    ai_response = f"Problem framing: {framing}"
    if constraints_noted:
        ai_response += f"\nConstraints: {constraints_noted}"
    if questions:
        ai_response += f"\nQuestions: {json.dumps(questions)}"
    history.append({"role": "ai", "content": ai_response})

    return {
        "problem_framing": framing,
        "constraints_noted": constraints_noted,
        "questions": questions,
        "intake_chat_history": history,
        "messages": [AIMessage(content=chat_summary)],
    }


def human_gate_1(state: dict) -> dict:
    print("\n" + "=" * 80)
    print("NODE: Human Gate 1 — Problem Framing")
    print("=" * 80)

    framing = state.get("problem_framing", "")
    questions = state.get("questions", [])
    constraints_noted = state.get("constraints_noted", "")

    review_content = f"--- PROBLEM FRAMING ---\n{framing}"
    if constraints_noted:
        review_content += f"\n\n--- CONSTRAINTS ---\n{constraints_noted}"
    if questions:
        questions_text = "\n".join(f"  {i}. {q}" for i, q in enumerate(questions, 1))
        review_content += f"\n\n--- QUESTIONS ---\n{questions_text}"

    review_content += (
        "\n\n---\n"
        "- Press Enter or 'approve' -> proceed to topic planning\n"
        "- Or answer the questions / provide feedback to revise"
    )

    if AUTO_APPROVE:
        print("  -> Auto-approved")
        _append_log("problem_intake.txt", "=== HUMAN GATE 1 ===\nFeedback: auto-approved\n\n")
        return {"human_feedback_1": "approved"}

    feedback = interrupt(review_content)
    feedback = str(feedback).strip()

    if not feedback or feedback.lower() in ("approve", "approved", "ok", "yes", "skip", "looks good", "lgtm"):
        print("  -> Approved")
        _append_log("problem_intake.txt", "=== HUMAN GATE 1 ===\nFeedback: approved\n\n")
        return {"human_feedback_1": "approved"}

    clarify_round = state.get("clarify_round", 0) + 1
    if clarify_round >= MAX_CLARIFY_ROUNDS:
        print(f"  ** Max clarify rounds ({MAX_CLARIFY_ROUNDS}) reached. Forcing approve. **")
        _append_log("problem_intake.txt", f"=== HUMAN GATE 1 ===\nFeedback: {feedback}\n(Max rounds reached — forced approve)\n\n")
        return {"human_feedback_1": "approved", "clarify_round": clarify_round}

    print(f"  -> Feedback: {feedback[:100]} (round {clarify_round}/{MAX_CLARIFY_ROUNDS})")
    _append_log("problem_intake.txt", f"=== HUMAN GATE 1 (Round {clarify_round}) ===\nFeedback: {feedback}\n\n")
    return {"human_feedback_1": feedback, "clarify_round": clarify_round}


def route_after_gate_1(state: dict) -> Literal["clarify_problem", "plan_research_topics"]:
    return "plan_research_topics" if state.get("human_feedback_1") == "approved" else "clarify_problem"


def plan_research_topics(state: dict) -> dict:
    print("\n" + "=" * 80)
    print("NODE: Plan Research Topics")
    print("=" * 80)

    history = list(state.get("intake_chat_history", []))

    hf = state.get("human_feedback_2", "")
    if hf and hf != "approved":
        history.append({"role": "human", "content": f"Revise the research topics based on this feedback: {hf}"})

    history.append({"role": "human", "content": PLAN_TOPICS_INSTRUCTION.format(n=MAX_RESEARCH_TOPICS)})

    chat_msgs = _rebuild_chat_history(history)
    structured = llm_intake.with_structured_output(TopicsOutput)
    t0 = time.time()
    result = structured.invoke(chat_msgs)
    _record("llm", "plan_research_topics", time.time() - t0, "generate topics")
    topics = result.topics[:MAX_RESEARCH_TOPICS]

    print(f"  Topics ({len(topics)}):")
    for i, t in enumerate(topics, 1):
        print(f"    {i}. {t}")

    topics_text = "\n".join(f"  {i}. {t}" for i, t in enumerate(topics, 1))
    research_brief = (
        f"## Problem\n{state.get('problem_framing', '') or state.get('user_query', '')}\n\n"
        f"## Context\n"
        f"- Market: {state.get('country_or_market', 'N/A')}\n"
        f"- Product: {state.get('product_idea', 'N/A')}\n"
        f"- Target customer: {state.get('target_customer', 'N/A')}\n"
        f"- Budget: {state.get('budget_range', 'N/A')}\n"
        f"- Time horizon: {state.get('time_horizon', 'N/A')}\n"
        f"- Risk tolerance: {state.get('risk_tolerance', 'N/A')}\n"
        f"- Constraints: {state.get('constraints', 'None')}\n"
        f"- Constraints noted: {state.get('constraints_noted', 'None')}\n\n"
        f"## Research Topics\n{topics_text}"
    )

    print(f"  Research brief: {len(research_brief)} chars")

    history.append({"role": "ai", "content": f"Research topics:\n{topics_text}"})

    log = "=== PLAN RESEARCH TOPICS ===\n\n"
    log += "Topics:\n"
    for i, t in enumerate(topics, 1):
        log += f"  {i}. {t}\n"
    log += f"\nResearch Brief:\n{research_brief}\n"
    _append_log("topic_planning.txt", log + "\n")

    return {
        "research_topics": topics,
        "research_brief": research_brief,
        "intake_chat_history": history,
        "messages": [AIMessage(content=f"**Research Plan** ({len(topics)} topics):\n{topics_text}")],
    }


def human_gate_2(state: dict) -> dict:
    print("\n" + "=" * 80)
    print("NODE: Human Gate 2 — Research Topics")
    print("=" * 80)

    topics = state.get("research_topics", [])
    topics_text = "\n".join(f"  {i}. {t}" for i, t in enumerate(topics, 1))

    if AUTO_APPROVE:
        print("  -> Auto-approved")
        _append_log("topic_planning.txt", "=== HUMAN GATE 2 ===\nFeedback: auto-approved\n\n")
        return {"human_feedback_2": "approved", "intake_chat_history": []}

    feedback = interrupt(
        f"--- RESEARCH TOPICS ({len(topics)}) ---\n{topics_text}\n\n"
        "---\n"
        "- Press Enter or 'approve' -> start researching these topics\n"
        "- Or provide feedback to change the topics (e.g. add, remove, rephrase)"
    )
    feedback = str(feedback).strip()

    if not feedback or feedback.lower() in ("approve", "approved", "ok", "yes", "skip", "looks good", "lgtm"):
        print("  -> Approved")
        _append_log("topic_planning.txt", "=== HUMAN GATE 2 ===\nFeedback: approved\n\n")
        return {"human_feedback_2": "approved", "intake_chat_history": []}

    revision_round = state.get("topics_revision_round", 0) + 1
    if revision_round >= MAX_TOPICS_REVISION:
        print(f"  ** Max topics revisions ({MAX_TOPICS_REVISION}) reached. Forcing approve. **")
        _append_log("topic_planning.txt", f"=== HUMAN GATE 2 ===\nFeedback: {feedback}\n(Max rounds reached — forced approve)\n\n")
        return {"human_feedback_2": "approved", "topics_revision_round": revision_round, "intake_chat_history": []}

    print(f"  -> Feedback: {feedback[:100]} (revision {revision_round}/{MAX_TOPICS_REVISION})")
    _append_log("topic_planning.txt", f"=== HUMAN GATE 2 (Round {revision_round}) ===\nFeedback: {feedback}\n\n")
    return {"human_feedback_2": feedback, "topics_revision_round": revision_round}


def route_after_gate_2(state: dict) -> Literal["plan_research_topics", "__end__"]:
    if state.get("human_feedback_2") == "approved":
        return "__end__"
    return "plan_research_topics"


# ============================================================================
# BUILD GRAPH
# ============================================================================

def build_graph(checkpointer=None):
    g = StateGraph(Stage1State)

    g.add_node("intake", intake)
    g.add_node("clarify_problem", clarify_problem)
    g.add_node("human_gate_1", human_gate_1)
    g.add_node("plan_research_topics", plan_research_topics)
    g.add_node("human_gate_2", human_gate_2)

    g.add_edge(START, "intake")
    g.add_edge("intake", "clarify_problem")
    g.add_edge("clarify_problem", "human_gate_1")
    g.add_conditional_edges("human_gate_1", route_after_gate_1, {
        "clarify_problem": "clarify_problem",
        "plan_research_topics": "plan_research_topics",
    })
    g.add_edge("plan_research_topics", "human_gate_2")
    g.add_conditional_edges("human_gate_2", route_after_gate_2, {
        "plan_research_topics": "plan_research_topics",
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

    parser = argparse.ArgumentParser(description="Stage 1 — Problem Intake & Topic Planning")
    parser.add_argument("--name", type=str, default=None,
                        help="Suffix appended to the output folder")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{ts}_{args.name}" if args.name else ts
    output_dir = Path(__file__).resolve().parent / "results" / "stage1_intake" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(output_dir)

    report_export.start_log(output_dir)

    t0 = time.time()

    print(f"  Query:        {INPUT_QUERY}")
    print(f"  Auto-approve: {AUTO_APPROVE}")
    print(f"  Output dir:   {output_dir}")
    print()
    for name, cfg in AGENTS.items():
        temp_str = f"  temp={cfg['temperature']}" if "temperature" in cfg else ""
        print(f"    {name:12s}  {cfg['model']:40s}{temp_str}")
    print("=" * 80)

    agent = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "stage1-run-1"}}

    result = agent.invoke({"user_query": INPUT_QUERY}, config)

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

    # ── Save handoff for stage 2 ──
    handoff = {
        "user_query": result.get("user_query", ""),
        "country_or_market": result.get("country_or_market", ""),
        "product_idea": result.get("product_idea", ""),
        "target_customer": result.get("target_customer", ""),
        "budget_range": result.get("budget_range", ""),
        "time_horizon": result.get("time_horizon", ""),
        "risk_tolerance": result.get("risk_tolerance", ""),
        "constraints": result.get("constraints", ""),
        "problem_framing": result.get("problem_framing", ""),
        "constraints_noted": result.get("constraints_noted", ""),
        "research_topics": result.get("research_topics", []),
        "research_brief": result.get("research_brief", ""),
        "config": {
            "agents": AGENTS,
            "input_query": INPUT_QUERY,
        },
    }
    save_handoff(handoff, output_dir)
    save_timings()

    # ── Meta ──
    topics = result.get("research_topics", [])
    save_meta([
        f"Stage:            1 — Problem Intake & Topic Planning",
        f"Timestamp:        {datetime.now().isoformat()}",
        f"Input Query:      {INPUT_QUERY}",
        f"Elapsed:          {elapsed:.1f}s",
        f"",
        f"Model:",
        f"  intake        {AGENTS['intake']['model']}",
        f"",
        f"Settings:",
        f"  Max clarify rounds:    {MAX_CLARIFY_ROUNDS}",
        f"  Max research topics:   {MAX_RESEARCH_TOPICS}",
        f"  Max topics revision:   {MAX_TOPICS_REVISION}",
    ], output_dir)

    print("\n" + "=" * 80)
    print("STAGE 1 COMPLETE")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Output:  {output_dir}")
    print(f"\nTo continue:\n  python stage2_research.py --input {output_dir}")
    print("=" * 80)

    report_export.stop_log()
