"""
local_agent/stage1_runner.py — Widget-based Q&A runner for Stage 1 (Design 3, Option A).

Replaces run_stage.run_stage1()'s input() interrupt loop with an ipywidgets
alert box that students cannot miss. The agent pauses at each LangGraph
interrupt; the Submit button in the Answer cell resumes it.

Notebook usage:
    # step5a cell — start the agent
    import stage1_runner
    stage1_runner.start_stage1(mod, CONFIG)

    # step5b cell — always rendered below, updates live
    stage1_runner.show_answer_widget()

Fallback: if LangGraph cross-cell state bridging proves unreliable,
switch to Option C (batch all questions upfront as a single form).
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
    """Return the pending interrupt message string, or None if graph is done."""
    snapshot = _session["agent"].get_state(_session["lc_config"])
    if not snapshot.next:
        return None
    for task in snapshot.tasks:
        if hasattr(task, "interrupts"):
            for intr in task.interrupts:
                return str(intr.value)
    return None


def _finish(result: dict):
    """Save all Stage 1 outputs and display the completion banner."""
    from common import save_handoff, save_summary_stage1, save_meta, save_timings

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
        done=False,
        t0=time.time(),
    )

    result = agent.invoke({"user_query": mod.INPUT_QUERY}, lc_config)
    _session["last_result"] = result

    pending = _extract_interrupt()
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
    Called by the Submit button in step5b. Resumes the LangGraph agent with the
    student's answer. Returns True if another question is pending, False if done.
    """
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


def show_answer_widget():
    """
    Render the Q&A alert widget. Call once from the step5b cell.
    The widget clears and redraws in-place as each question cycles through.
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

    out = widgets.Output()

    def _render():
        q        = _session.get("pending_question", "")
        gate_num = _session.get("gate_num", 1)

        answer_input = widgets.Textarea(
            placeholder="Type your answer here, or leave blank to approve as-is…",
            layout=widgets.Layout(width="560px", height="80px"),
        )
        submit_btn = widgets.Button(
            description="Submit Answer",
            button_style="danger",
            layout=widgets.Layout(width="170px", height="36px"),
        )

        def _on_submit(b):
            submit_btn.disabled = True
            answer = answer_input.value
            answer_input.value = ""
            with out:
                out.clear_output()
                display(HTML("<i style='color:#888'>Sending your answer…</i>"))
            more = submit_answer(answer)
            with out:
                out.clear_output()
                if more:
                    _render()
                # If not more: _finish() already rendered the completion banner

        submit_btn.on_click(_on_submit)

        with out:
            out.clear_output()
            display(widgets.VBox([
                widgets.HTML(f"""
                    <div style='border:3px solid #e53935;padding:16px;border-radius:8px;
                                background:#fff3f3;margin-bottom:10px'>
                    <span style='font-size:20px;font-weight:bold;color:#e53935'>
                    ⚠ Human Gate {gate_num} — The AI has a question for you:</span>
                    <div style='margin-top:12px;font-size:14px;white-space:pre-wrap;
                                border-left:3px solid #e53935;padding-left:12px'>{q}</div>
                    </div>
                    <p style='color:#555;font-size:13px;margin:0 0 6px'>
                    Type your answer below and click <b>Submit Answer</b>.<br>
                    Leave blank and submit to <b>approve</b> the current framing and continue.</p>
                """),
                answer_input,
                submit_btn,
            ]))

    _render()
    display(out)
