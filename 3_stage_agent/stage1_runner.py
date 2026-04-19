"""
stage1_runner.py — Widget-based Q&A runner for Stage 1 (Design 3, Option A).

Pipeline:  intake → clarify_problem → [Gate 1: Problem Framing] → plan_research_topics
                                    → [Gate 2: Research Topics]  → END

Two Output widgets are used:
  history_out — accumulates approved-phase summary cards (never cleared)
  gate_out    — shows the current gate widget (cleared/replaced each gate)

Notebook usage:
    # step5a cell — start the agent
    import stage1_runner
    stage1_runner.start_stage1(mod, CONFIG)

    # step5b cell — always rendered below, updates live
    stage1_runner.show_answer_widget()
"""

import time
import shutil
from datetime import datetime
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display, HTML, Markdown
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import report_export

DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/AI_Essentials_Final_Project")

# ============================================================================
# Module-level session — persists across cells within the same kernel session
# ============================================================================

_session: dict = {}

# ============================================================================
# Internal helpers
# ============================================================================

def _extract_interrupt() -> str | None:
    """Return the pending interrupt message, or None if graph is done.
    Also updates _session['gate_type'] to 'framing' or 'topics'."""
    snapshot = _session["agent"].get_state(_session["lc_config"])
    if not snapshot.next:
        _session["gate_type"] = ""
        return None
    if "human_gate_1" in snapshot.next:
        _session["gate_type"] = "framing"
    elif "human_gate_2" in snapshot.next:
        _session["gate_type"] = "topics"
    for task in snapshot.tasks:
        if hasattr(task, "interrupts"):
            for intr in task.interrupts:
                return str(intr.value)
    return None


def _finish(result: dict):
    """Save all Stage 1 outputs and display the completion banner."""
    from common import save_handoff, save_summary_stage1, save_meta, save_timings

    # Show research-topics summary card via widget callback if registered
    cb = _session.get("_show_phase_complete")
    if cb:
        cb("topics")

    config_dict = _session["config_dict"]
    output_dir  = _session["output_dir"]
    elapsed     = time.time() - _session["t0"]

    handoff = {
        "user_query":        result.get("user_query", ""),
        "country_or_market": result.get("country_or_market", ""),
        "product_idea":      result.get("product_idea", ""),
        "target_customer":   result.get("target_customer", ""),
        "budget_range":      result.get("budget_range", ""),
        "time_horizon":      result.get("time_horizon", ""),
        "risk_tolerance":    result.get("risk_tolerance", ""),
        "constraints":       result.get("constraints", ""),
        "problem_framing":   result.get("problem_framing", ""),
        "constraints_noted": result.get("constraints_noted", ""),
        "research_topics":   result.get("research_topics", []),
        "research_brief":    result.get("research_brief", ""),
        "config": {
            "model_intake": config_dict["stage1_intake"]["model"],
            "input_query":  config_dict["input_query"],
        },
    }
    save_handoff(handoff, output_dir)
    save_summary_stage1(handoff, output_dir)

    _cfg = config_dict["stage1_intake"]
    save_meta([
        "Stage:            1 — Problem Intake & Topic Planning",
        f"Timestamp:        {datetime.now().isoformat()}",
        f"Input Query:      {config_dict['input_query']}",
        f"Elapsed:          {elapsed:.1f}s",
        "",
        "Model:",
        f"  intake        {_cfg['model']}",
        "",
        "Settings:",
        f"  Max clarify rounds:    {_cfg['max_clarify_rounds']}",
        f"  Max research topics:   {_cfg['max_research_topics']}",
        f"  Max topics revision:   {_cfg['max_topics_revision']}",
    ], output_dir)

    save_timings()

    if Path("/content/drive/MyDrive").exists():
        drive_dest = DRIVE_OUTPUT_DIR / "stage1_intake" / output_dir.name
        drive_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output_dir, drive_dest)
        print(f"Drive: {drive_dest}")

    report_export.stop_log()
    _session["done"] = True

    display(HTML("<hr style='border:4px solid #1B2A4A; margin:24px 0'>"))
    display(Markdown("## Stage 1 Complete"))
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Local:   {output_dir.resolve()}")

# ============================================================================
# Public API
# ============================================================================

def start_stage1(mod, config_dict: dict):
    """
    Called from step5a cell. Builds the LangGraph agent, runs it to the first
    interrupt, and stores session state for the Answer cell (step5b) to use.
    """
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(ts)
    output_dir.mkdir(parents=True, exist_ok=True)
    mod.set_output_dir(output_dir)
    print(f"Output dir: {output_dir.resolve()}\n")

    agent     = mod.build_graph(checkpointer=MemorySaver())
    lc_config = {"configurable": {"thread_id": "colab-stage1-1"}}

    report_export.start_log(output_dir)

    _session.clear()
    _session.update(
        agent=agent,
        lc_config=lc_config,
        output_dir=output_dir,
        config_dict=config_dict,
        mod=mod,
        gate_num=0,
        gate_type="",
        done=False,
        t0=time.time(),
    )

    result = agent.invoke({"user_query": mod.INPUT_QUERY}, lc_config)
    _session["last_result"] = result

    pending = _extract_interrupt()  # sets _session["gate_type"]
    if pending:
        _session["gate_num"]        += 1
        _session["pending_question"] = pending
        display(HTML(
            "<div style='border:2px solid #f57c00;padding:10px 14px;border-radius:6px;"
            "background:#fff8e1;margin-top:12px;font-size:15px'>"
            "⬇ <b>The AI has a question for you.</b> "
            "Run the <b>Answer cell below</b> to respond.</div>"
        ))
    else:
        _finish(result)


def submit_answer(answer_text: str) -> bool:
    """
    Called by the Submit / Approve buttons in step5b. Resumes the LangGraph agent
    with the student's answer. Returns True if another gate is pending, False if done.
    """
    feedback = answer_text.strip() or "approved"
    result   = _session["agent"].invoke(Command(resume=feedback), _session["lc_config"])
    _session["last_result"] = result

    pending = _extract_interrupt()  # updates _session["gate_type"]
    if pending:
        _session["gate_num"]        += 1
        _session["pending_question"] = pending
        return True

    _finish(result)
    return False


def show_answer_widget():
    """
    Render the Q&A widget. Call once from the step5b cell.

    Layout:
      history_out — approved-phase summary cards accumulate here
      gate_out    — current gate form (replaced on each gate transition)
    """
    if _session.get("done"):
        display(HTML(
            "<div style='border:2px solid #388e3c;padding:10px;border-radius:6px;"
            "background:#f1f8e9'>✓ Stage 1 is already complete.</div>"
        ))
        return

    if not _session:
        display(HTML(
            "<div style='border:2px solid #e53935;padding:10px;border-radius:6px;"
            "background:#fff3f3'>⚠ Run the cell above (Step 5a) first.</div>"
        ))
        return

    history_out = widgets.Output()
    gate_out    = widgets.Output()
    display(widgets.VBox([history_out, gate_out]))

    # ── Phase-complete summary cards ─────────────────────────────────────────

    def _show_phase_complete(phase: str):
        result = _session.get("last_result", {})
        with history_out:
            if phase == "framing":
                framing     = result.get("problem_framing", "")
                constraints = result.get("constraints_noted", "")
                constraints_html = (
                    f"<div style='margin-top:8px;font-size:13px'>"
                    f"<b>Constraints:</b> {constraints}</div>"
                    if constraints and constraints.lower() not in ("none", "")
                    else ""
                )
                display(HTML(f"""
                    <div style='border:2px solid #388e3c;padding:14px;border-radius:8px;
                                background:#f1f8e9;margin-bottom:12px'>
                      <div style='font-weight:bold;color:#388e3c;font-size:15px;margin-bottom:8px'>
                        ✓ Problem Framing Approved</div>
                      <div style='white-space:pre-wrap;font-size:14px'>{framing}</div>
                      {constraints_html}
                    </div>
                """))

            elif phase == "topics":
                topics = result.get("research_topics", [])
                items  = "".join(
                    f"<li style='margin:4px 0'>{t}</li>" for t in topics
                )
                display(HTML(f"""
                    <div style='border:2px solid #1565c0;padding:14px;border-radius:8px;
                                background:#e3f2fd;margin-bottom:12px'>
                      <div style='font-weight:bold;color:#1565c0;font-size:15px;margin-bottom:8px'>
                        ✓ Research Topics Approved ({len(topics)} topics)</div>
                      <ol style='margin:4px 0;padding-left:20px;font-size:14px'>{items}</ol>
                    </div>
                """))

    # Register so _finish() can call it without importing show_answer_widget
    _session["_show_phase_complete"] = _show_phase_complete

    # ── Gate renderer ────────────────────────────────────────────────────────

    def _render():
        gate_type = _session.get("gate_type", "")
        q         = _session.get("pending_question", "")
        gate_num  = _session.get("gate_num", 1)

        if gate_type == "framing":
            border_color  = "#e65100"
            bg_color      = "#fff8e1"
            gate_label    = "Stage 1 — Problem Framing Review"
            hint          = (
                "The AI has framed your business question. "
                "Provide feedback to refine it, or approve to move to research topic planning."
            )
            submit_label  = "Submit Feedback"
            approve_label = "✓ Approve & Continue"
            approve_tip   = "Approve the problem framing and proceed to topic planning."
        elif gate_type == "topics":
            border_color  = "#1565c0"
            bg_color      = "#e3f2fd"
            gate_label    = "Stage 1 — Research Topics Review"
            hint          = (
                "Review the proposed research topics. "
                "Provide feedback to add, remove, or rephrase, or approve to finalize Stage 1."
            )
            submit_label  = "Submit Feedback"
            approve_label = "✓ Approve & End"
            approve_tip   = "Approve the research topics and finalize Stage 1."
        else:
            border_color  = "#e53935"
            bg_color      = "#fff3f3"
            gate_label    = f"Human Gate {gate_num}"
            hint          = (
                "Type your answer below and click Submit Answer, or "
                "click Approve & End to accept as-is."
            )
            submit_label  = "Submit Answer"
            approve_label = "✓ Approve & End"
            approve_tip   = "Approve as-is and finalize Stage 1."

        answer_input = widgets.Textarea(
            placeholder="Type feedback here, or leave blank and click Approve…",
            layout=widgets.Layout(width="560px", height="80px"),
        )
        restart_btn = widgets.Button(
            description="↺ Re-start",
            button_style="warning",
            layout=widgets.Layout(width="130px", height="36px"),
            tooltip="Wipe this conversation and re-submit your original question from scratch.",
        )
        submit_btn = widgets.Button(
            description=submit_label,
            button_style="danger",
            layout=widgets.Layout(width="170px", height="36px"),
            tooltip="Send your feedback to the AI.",
        )
        approve_btn = widgets.Button(
            description=approve_label,
            button_style="success",
            layout=widgets.Layout(width="190px", height="36px"),
            tooltip=approve_tip,
        )

        def _disable_all():
            restart_btn.disabled = True
            submit_btn.disabled  = True
            approve_btn.disabled = True

        def _on_submit(b):
            _disable_all()
            answer = answer_input.value
            answer_input.value = ""
            with gate_out:
                gate_out.clear_output()
                display(HTML("<i style='color:#888'>Sending your feedback…</i>"))
            more = submit_answer(answer)
            with gate_out:
                gate_out.clear_output()
                if more:
                    _render()

        def _on_approve(b):
            _disable_all()
            prev_gate = _session.get("gate_type", "")
            with gate_out:
                gate_out.clear_output()
                display(HTML("<i style='color:#888'>Approving…</i>"))
            more = submit_answer("approved")
            # After Gate 1 approval, show framing card immediately.
            # Topics card is shown from _finish() via _show_phase_complete callback.
            if prev_gate == "framing":
                _show_phase_complete("framing")
            with gate_out:
                gate_out.clear_output()
                if more:
                    _render()

        def _on_restart(b):
            _disable_all()
            mod         = _session["mod"]
            config_dict = _session["config_dict"]
            with gate_out:
                gate_out.clear_output()
                display(HTML("<i style='color:#888'>Re-starting — please wait…</i>"))
            with history_out:
                history_out.clear_output()
            start_stage1(mod, config_dict)
            # Re-register callback after start_stage1 clears _session
            _session["_show_phase_complete"] = _show_phase_complete
            with gate_out:
                gate_out.clear_output()
                if not _session.get("done"):
                    _render()

        submit_btn.on_click(_on_submit)
        approve_btn.on_click(_on_approve)
        restart_btn.on_click(_on_restart)

        with gate_out:
            gate_out.clear_output()
            display(widgets.VBox([
                widgets.HTML(f"""
                    <div style='border:3px solid {border_color};padding:16px;border-radius:8px;
                                background:{bg_color};margin-bottom:10px'>
                      <span style='font-size:18px;font-weight:bold;color:{border_color}'>
                        ⚠ {gate_label}</span>
                      <div style='margin-top:12px;font-size:14px;white-space:pre-wrap;
                                  border-left:3px solid {border_color};padding-left:12px'>{q}</div>
                    </div>
                    <p style='color:#555;font-size:13px;margin:0 0 6px'>{hint}</p>
                """),
                answer_input,
                widgets.HBox(
                    [restart_btn, submit_btn, approve_btn],
                    layout=widgets.Layout(gap="10px", margin="6px 0 0 0"),
                ),
            ]))

    _render()
