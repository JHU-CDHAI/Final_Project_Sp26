"""
stage1_runner.py — Widget-based Q&A runner for Stage 1 (Design 3, Option A).

Pipeline:  intake → clarify_problem → [Gate 1: Problem Framing] → plan_research_topics
                                    → [Gate 2: Research Topics]  → END

Two separate functions for two separate notebook cells:

  Step 5b:  stage1_runner.show_framing_widget()   — Problem Framing review
  Step 5c:  stage1_runner.show_topics_widget()    — Research Topics review

Each cell:
  - While active:   shows the gate question, a feedback textarea, and 3 buttons.
  - After approval: replaces itself with a finalized-content card (from output.md)
                    and a Re-start button in case the student approved by accident.

Re-start from either cell re-renders both widgets in place.
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
# Module-level state
# ============================================================================

_session: dict = {}

# Widget handles and render callbacks that survive _session.clear() across
# restarts — lets Re-start from either cell update both cells in place.
_widget_refs: dict = {}

# ============================================================================
# Internal helpers
# ============================================================================

def _extract_interrupt() -> str | None:
    """Return pending interrupt text; also updates _session['gate_type']."""
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
    """Save all Stage 1 outputs and show the completion banner."""
    from common import save_handoff, save_summary_stage1, save_meta, save_timings

    _session["approved_topics"] = result.get("research_topics", [])
    _session["done"] = True

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

    display(HTML("<hr style='border:4px solid #1B2A4A; margin:24px 0'>"))
    display(Markdown("## Stage 1 Complete"))
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Local:   {output_dir.resolve()}")

# ============================================================================
# Public API — called from notebook cells
# ============================================================================

def start_stage1(mod, config_dict: dict):
    """Step 5a. Builds the agent and runs to the first interrupt (Gate 1)."""
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
        framing_approved=False,
        t0=time.time(),
    )

    result = agent.invoke({"user_query": mod.INPUT_QUERY}, lc_config)
    _session["last_result"] = result

    pending = _extract_interrupt()
    if pending:
        _session["gate_num"]        += 1
        _session["pending_question"] = pending
        display(HTML(
            "<div style='border:2px solid #e65100;padding:10px 14px;border-radius:6px;"
            "background:#fff8e1;margin-top:12px;font-size:15px'>"
            "⬇ <b>The AI has framed your problem.</b> "
            "Run <b>Step 5b</b> below to review and respond.</div>"
        ))
    else:
        _finish(result)


def submit_answer(answer_text: str) -> bool:
    """Resume the agent. Returns True if another gate is pending, False if done."""
    feedback = answer_text.strip() or "approved"
    result   = _session["agent"].invoke(Command(resume=feedback), _session["lc_config"])
    _session["last_result"] = result

    pending = _extract_interrupt()
    if pending:
        _session["gate_num"]        += 1
        _session["pending_question"] = pending
        return True

    _finish(result)
    return False

# ============================================================================
# Shared widget helpers
# ============================================================================

def _make_restart_btn() -> widgets.Button:
    return widgets.Button(
        description="↺ Re-start",
        button_style="warning",
        layout=widgets.Layout(width="140px", height="36px"),
        tooltip="Wipe this conversation and re-submit your original question from scratch.",
    )


def _do_restart():
    """Called by any Re-start button. Resets state and re-renders both cells."""
    mod         = _session["mod"]
    config_dict = _session["config_dict"]

    # Show "waiting" in topics cell immediately (before session is cleared)
    topics_out = _widget_refs.get("topics_out")
    if topics_out:
        with topics_out:
            topics_out.clear_output()
            display(HTML(
                "<div style='border:2px solid #999;padding:10px;border-radius:6px;"
                "background:#f5f5f5;color:#555'>"
                "Re-starting — complete Problem Framing above first.</div>"
            ))

    start_stage1(mod, config_dict)

    # Re-render framing cell to the active gate
    render_framing = _widget_refs.get("render_framing_gate")
    if render_framing:
        render_framing()

# ============================================================================
# Step 5b — Problem Framing (Gate 1)
# ============================================================================

def show_framing_widget():
    """
    Renders the Problem Framing gate widget.
    - Active:   amber gate card + feedback textarea + 3 buttons
    - Complete: green card with the saved framing text + Re-start button
    """
    out = widgets.Output()
    display(out)
    _widget_refs["framing_out"] = out

    # ── Completion card ──────────────────────────────────────────────────────

    def _render_complete():
        import html as _html
        framing     = _session.get("approved_framing", "")
        constraints = _session.get("approved_constraints", "")
        user_query  = _session.get("config_dict", {}).get("input_query", "")

        copy_text = f"## Business Question\n{user_query}"
        copy_text += f"\n\n## Problem Framing\n{framing}"
        if constraints and constraints.lower() not in ("none", ""):
            copy_text += f"\n\n**Constraints:** {constraints}"

        with out:
            out.clear_output()
            display(widgets.HTML(
                "<div style='border:2px solid #388e3c;padding:12px 16px;border-radius:8px;"
                "background:#f1f8e9'>"
                "<div style='font-weight:bold;color:#388e3c;font-size:16px;margin-bottom:6px'>"
                "✓ Problem Framing Approved</div>"
                "<div style='color:#555;font-size:13px;margin-bottom:10px'>"
                "This content will be saved to <code>output.md</code> once you "
                "approve the research topics in Step 5c. "
                "Select the text below to copy it.</div>"
                "<pre style='background:#fff;border:1px solid #c8e6c9;border-radius:4px;"
                "padding:12px;font-size:13px;white-space:pre-wrap;word-wrap:break-word;"
                "font-family:inherit;margin:0;width:100%;box-sizing:border-box'>"
                f"{_html.escape(copy_text)}</pre></div>"
            ))

    _widget_refs["render_framing_complete"] = _render_complete

    # ── Active gate ──────────────────────────────────────────────────────────

    def _render_gate():
        q = _session.get("pending_question", "")

        answer_input = widgets.Textarea(
            placeholder="Type feedback here, or leave blank and click Approve & Continue…",
            layout=widgets.Layout(width="560px", height="80px"),
        )
        submit_btn  = widgets.Button(
            description="Submit Feedback",
            button_style="primary",
            layout=widgets.Layout(width="170px", height="36px"),
            tooltip="Send your feedback to refine the problem framing.",
        )
        approve_btn = widgets.Button(
            description="✓ Approve & Continue",
            button_style="success",
            layout=widgets.Layout(width="190px", height="36px"),
            tooltip="Approve this framing and proceed to research topic planning.",
        )

        def _disable_all():
            submit_btn.disabled  = True
            approve_btn.disabled = True

        def _on_submit(b):
            _disable_all()
            answer = answer_input.value
            answer_input.value = ""
            with out:
                out.clear_output()
                display(HTML("<i style='color:#888'>Sending your feedback…</i>"))
            more = submit_answer(answer)
            if more and _session.get("gate_type") == "framing":
                _render_gate()
            else:
                _capture_framing_and_complete(more)

        def _on_approve(b):
            _disable_all()
            with out:
                out.clear_output()
                display(HTML("<i style='color:#888'>Approving…</i>"))
            more = submit_answer("approved")
            _capture_framing_and_complete(more)

        submit_btn.on_click(_on_submit)
        approve_btn.on_click(_on_approve)

        with out:
            out.clear_output()
            display(widgets.VBox([
                widgets.HTML(f"""
                    <div style='border:3px solid #388e3c;padding:16px;border-radius:8px;
                                background:#f1f8e9;margin-bottom:10px'>
                      <span style='font-size:18px;font-weight:bold;color:#388e3c'>
                        Stage 1 — Problem Framing Review</span>
                      <div style='margin-top:12px;font-size:14px;white-space:pre-wrap;
                                  border-left:3px solid #388e3c;padding-left:12px'>{q}</div>
                    </div>
                    <p style='color:#555;font-size:13px;margin:0 0 6px'>
                      Review the AI's framing of your business question.
                      Provide feedback to refine it, or approve to proceed to research topic planning.</p>
                """),
                answer_input,
                widgets.HBox(
                    [submit_btn, approve_btn],
                    layout=widgets.Layout(gap="10px", margin="6px 0 0 0"),
                ),
            ]))

    def _capture_framing_and_complete(more: bool):
        """Save framing content into session then show completion card."""
        result = _session.get("last_result", {})
        _session["approved_framing"]    = result.get("problem_framing", "")
        _session["approved_constraints"] = result.get("constraints_noted", "")
        _session["framing_approved"]    = True
        _render_complete()
        # If topics gate is now ready, signal the topics widget
        if more and _session.get("gate_type") == "topics":
            render_topics = _widget_refs.get("render_topics_gate")
            if render_topics:
                render_topics()

    _widget_refs["render_framing_gate"] = _render_gate

    # ── Initial render ───────────────────────────────────────────────────────
    if not _session:
        with out:
            display(HTML(
                "<div style='border:2px solid #e53935;padding:10px;border-radius:6px;"
                "background:#fff3f3'>Run Step 5a first.</div>"
            ))
    elif _session.get("framing_approved"):
        _render_complete()
    elif _session.get("gate_type") == "framing":
        _render_gate()
    else:
        with out:
            display(HTML(
                "<div style='border:2px solid #e53935;padding:10px;border-radius:6px;"
                "background:#fff3f3'>Run Step 5a first.</div>"
            ))

# ============================================================================
# Step 5c — Research Topics (Gate 2)
# ============================================================================

def show_topics_widget():
    """
    Renders the Research Topics gate widget.
    - Waiting:  grey card if Problem Framing not yet approved
    - Active:   blue gate card + feedback textarea + 3 buttons
    - Complete: blue card with the saved topics list + Re-start button
    """
    out = widgets.Output()
    display(out)
    _widget_refs["topics_out"] = out

    # ── Completion card ──────────────────────────────────────────────────────

    def _render_complete():
        import html as _html
        topics      = _session.get("approved_topics", [])
        topics_text = "\n".join(f"{i}. {t}" for i, t in enumerate(topics, 1))

        with out:
            out.clear_output()
            display(widgets.HTML(
                f"<div style='border:2px solid #1565c0;padding:12px 16px;border-radius:8px;"
                f"background:#e3f2fd'>"
                f"<div style='font-weight:bold;color:#1565c0;font-size:16px;margin-bottom:6px'>"
                f"✓ Research Topics Approved ({len(topics)} topics) — saved to output.md</div>"
                f"<div style='color:#555;font-size:13px;margin-bottom:10px'>"
                f"These are the exact research topics that will be used in Stage 2. "
                f"Select the text below to copy it.</div>"
                f"<pre style='background:#fff;border:1px solid #bbdefb;border-radius:4px;"
                f"padding:12px;font-size:13px;white-space:pre-wrap;word-wrap:break-word;"
                f"font-family:inherit;margin:0;width:100%;box-sizing:border-box'>"
                f"{_html.escape(topics_text)}</pre></div>"
            ))

    _widget_refs["render_topics_complete"] = _render_complete

    # ── Active gate ──────────────────────────────────────────────────────────

    def _render_gate():
        q = _session.get("pending_question", "")

        answer_input = widgets.Textarea(
            placeholder="Type feedback to change topics, or leave blank and click Approve & End…",
            layout=widgets.Layout(width="560px", height="80px"),
        )
        submit_btn  = widgets.Button(
            description="Submit Feedback",
            button_style="primary",
            layout=widgets.Layout(width="170px", height="36px"),
            tooltip="Send feedback to revise the research topics.",
        )
        approve_btn = widgets.Button(
            description="✓ Approve & End",
            button_style="success",
            layout=widgets.Layout(width="160px", height="36px"),
            tooltip="Approve the research topics and finalize Stage 1.",
        )

        def _disable_all():
            submit_btn.disabled  = True
            approve_btn.disabled = True

        def _on_submit(b):
            _disable_all()
            answer = answer_input.value
            answer_input.value = ""
            with out:
                out.clear_output()
                display(HTML("<i style='color:#888'>Sending your feedback…</i>"))
            more = submit_answer(answer)
            if more:
                _render_gate()
            else:
                _render_complete()

        def _on_approve(b):
            _disable_all()
            with out:
                out.clear_output()
                display(HTML("<i style='color:#888'>Approving and finalizing…</i>"))
            more = submit_answer("approved")
            if not more:
                _render_complete()

        submit_btn.on_click(_on_submit)
        approve_btn.on_click(_on_approve)

        with out:
            out.clear_output()
            display(widgets.VBox([
                widgets.HTML(f"""
                    <div style='border:3px solid #1565c0;padding:16px;border-radius:8px;
                                background:#e3f2fd;margin-bottom:10px'>
                      <span style='font-size:18px;font-weight:bold;color:#1565c0'>
                        ⚠ Stage 1 — Research Topics Review</span>
                      <div style='margin-top:12px;font-size:14px;white-space:pre-wrap;
                                  border-left:3px solid #1565c0;padding-left:12px'>{q}</div>
                    </div>
                    <p style='color:#555;font-size:13px;margin:0 0 6px'>
                      Review the proposed research topics.
                      Provide feedback to add, remove, or rephrase, or approve to finalize Stage 1.</p>
                """),
                answer_input,
                widgets.HBox(
                    [submit_btn, approve_btn],
                    layout=widgets.Layout(gap="10px", margin="6px 0 0 0"),
                ),
            ]))

    _widget_refs["render_topics_gate"] = _render_gate

    # ── Initial render ───────────────────────────────────────────────────────
    if _session.get("done"):
        _render_complete()
    elif _session.get("gate_type") == "topics":
        _render_gate()
    elif _session.get("framing_approved"):
        with out:
            display(HTML(
                "<div style='border:2px solid #999;padding:10px;border-radius:6px;"
                "background:#f5f5f5;color:#555'>Generating research topics…</div>"
            ))
    elif not _session:
        with out:
            display(HTML(
                "<div style='border:2px solid #999;padding:10px;border-radius:6px;"
                "background:#f5f5f5;color:#555'>Complete Step 5b (Problem Framing) first.</div>"
            ))
    else:
        with out:
            display(HTML(
                "<div style='border:2px solid #999;padding:10px;border-radius:6px;"
                "background:#f5f5f5;color:#555'>Complete the Problem Framing step above first.</div>"
            ))
