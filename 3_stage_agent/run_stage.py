"""Colab runners for each stage of the 3-stage MBA Strategy Agent."""

import sys
import time
import shutil
import yaml
import importlib
from datetime import datetime
from pathlib import Path
from IPython.display import display, Markdown, HTML
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import report_export

DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/AI_Essentials_Final_Project")


def _write_config(config_dict: dict, repo_dir: str):
    """Write config.yaml to the 3_stage_agent folder."""
    config_path = Path(repo_dir) / "3_stage_agent" / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Config written to {config_path}")


def _ensure_path(repo_dir: str):
    """Add 3_stage_agent to sys.path if not already there."""
    agent_dir = str(Path(repo_dir) / "3_stage_agent")
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)


def _reload_common():
    """Reload common.py so it picks up the freshly written config.yaml."""
    import common
    return importlib.reload(common)


def setup_stage1(repo_dir: str):
    """
    Bootstrap helper — call once after cloning the repo and mounting Drive.
    Adds the agent dir to sys.path, creates my_question.txt if missing,
    and shows the config UI. Returns the config_ui module so the caller
    can pass it straight to load_stage1().
    """
    import config_ui as _cui
    _ensure_path(repo_dir)
    importlib.reload(_cui)
    _cui.create_question_file()
    _cui.show_stage1()
    return _cui


def _run_graph(agent, lc_config, initial_state, output_dir,
               on_gate=None, get_feedback=None, on_node=None):
    """Shared interrupt loop for all stages.

    on_gate:      optional callable(snapshot) called after each gate resume.
    get_feedback: optional callable(interrupt_text) -> str that replaces the
                  default display + input() pair (used for widget-based gates).
    on_node:      optional callable(node_name, state_updates) fired after each
                  node execution — enables live progress updates between gates.
    """
    report_export.start_log(output_dir)

    def _stream(cmd):
        """Run agent stream until next interrupt or completion, firing on_node per node."""
        for chunk in agent.stream(cmd, lc_config, stream_mode="updates"):
            node_name = next((k for k in chunk if not k.startswith("__")), None)
            if on_node and node_name:
                on_node(node_name, chunk.get(node_name, {}))

    try:
        t0 = time.time()
        _stream(initial_state)

        gate_num = 0
        while True:
            snapshot = agent.get_state(lc_config)
            if not snapshot.next:
                break

            for task in snapshot.tasks:
                if hasattr(task, "interrupts"):
                    for intr in task.interrupts:
                        gate_num += 1
                        interrupt_text = str(intr.value)

                        if get_feedback is not None:
                            feedback = get_feedback(interrupt_text) or "approved"
                        else:
                            display(HTML("<hr style='border:3px solid #2E75B6; margin:24px 0 8px 0'>"))
                            display(Markdown(f"### Human Gate {gate_num}"))
                            display(Markdown(interrupt_text))
                            display(HTML("<hr style='border:3px solid #2E75B6; margin:8px 0 16px 0'>"))
                            feedback = input("approve / or type feedback > ").strip() or "approved"

                        _stream(Command(resume=feedback))

            if on_gate:
                on_gate(agent.get_state(lc_config))

        result = agent.get_state(lc_config).values
        elapsed = time.time() - t0
        return result, elapsed

    finally:
        report_export.stop_log()


def _copy_to_drive(output_dir: Path, stage_subdir: str):
    """Copy output to Google Drive if mounted."""
    if Path("/content/drive/MyDrive").exists():
        drive_dest = DRIVE_OUTPUT_DIR / stage_subdir / output_dir.name
        drive_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output_dir, drive_dest)
        print(f"Drive: {drive_dest}")
        return drive_dest
    else:
        print("[SKIP] Google Drive not mounted — Drive save skipped")
        return None


# ============================================================================
# STAGE 1
# ============================================================================

def load_stage1(config_ui, repo_dir: str):
    """Snapshot stage 1 widgets, write config, import module."""
    config_dict = config_ui.get_config_stage1()
    _write_config(config_dict, repo_dir)
    _ensure_path(repo_dir)
    _reload_common()

    if "stage1_intake" in sys.modules:
        mod = importlib.reload(sys.modules["stage1_intake"])
    else:
        import stage1_intake as mod

    return mod, config_dict


def run_stage1(mod, config_dict: dict):
    """Run stage 1 and save outputs."""
    agent = mod.build_graph(checkpointer=MemorySaver())
    lc_config = {"configurable": {"thread_id": "colab-stage1-1"}}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(ts)
    output_dir.mkdir(parents=True, exist_ok=True)
    mod.set_output_dir(output_dir)
    print(f"Output dir: {output_dir.resolve()}\n")

    initial_state = {"user_query": mod.INPUT_QUERY}
    result, elapsed = _run_graph(agent, lc_config, initial_state, output_dir)

    # Save handoff + output.md
    from common import save_handoff, save_summary_stage1, save_meta, save_timings

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
            "model_intake": config_dict["stage1_intake"]["model"],
            "input_query": config_dict["input_query"],
        },
    }
    save_handoff(handoff, output_dir)
    save_summary_stage1(handoff, output_dir)

    _model = config_dict["stage1_intake"]["model"]
    _cfg = config_dict["stage1_intake"]
    topics = result.get("research_topics", [])
    save_meta([
        f"Stage:            1 — Problem Intake & Topic Planning",
        f"Timestamp:        {datetime.now().isoformat()}",
        f"Input Query:      {config_dict['input_query']}",
        f"Elapsed:          {elapsed:.1f}s",
        f"",
        f"Model:",
        f"  intake        {_model}",
        f"",
        f"Settings:",
        f"  Max clarify rounds:    {_cfg['max_clarify_rounds']}",
        f"  Max research topics:   {_cfg['max_research_topics']}",
        f"  Max topics revision:   {_cfg['max_topics_revision']}",
    ], output_dir)

    save_timings()

    drive_dest = _copy_to_drive(output_dir, "stage1_intake")

    display(HTML("<hr style='border:4px solid #1B2A4A; margin:24px 0'>"))
    display(Markdown("## Stage 1 Complete"))
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Local:   {output_dir.resolve()}")
    return result


# ============================================================================
# STAGE 2
# ============================================================================

def load_stage2(config_ui, repo_dir: str):
    """Snapshot stage 2 widgets, write config, import module."""
    config_dict = config_ui.get_config_stage2()
    _write_config(config_dict, repo_dir)
    _ensure_path(repo_dir)
    _reload_common()

    if "stage2_research" in sys.modules:
        mod = importlib.reload(sys.modules["stage2_research"])
    else:
        import stage2_research as mod

    return mod, config_dict


def _parse_topics(text: str) -> list[str]:
    """Parse a research topics text block into a list of topic strings.

    Handles common paste formats:
      - Numbered:  "1. ", "1) ", "(1) ", "1: ", "1 - ", "1 . "
      - Lettered:  "a. ", "a) ", "A. ", "A) "
      - Roman:     "i. ", "ii) ", "III. "
      - Bulleted:  "- ", "* ", "• ", "◦ ", "▪ ", "► ", "→ ", "– ", "— "
      - Markdown:  "## heading", "**bold**", "> blockquote"
      - Separators: "---", "===", "***", "___"
      - Labels:    "Research Topics:", "Topics:" (skipped)
      - Plain lines (one topic per line)
    """
    import re
    topics = []
    for line in text.strip().splitlines():
        # Normalize whitespace (tabs, non-breaking spaces, etc.)
        line = re.sub(r"[\t\u00a0]+", " ", line).strip()
        if not line:
            continue
        # Skip markdown headers
        if re.match(r"^#{1,6}\s", line):
            continue
        # Skip blockquotes
        if re.match(r"^>\s", line):
            continue
        # Skip separator lines (---, ===, ***, ___)
        if re.match(r"^[-=\*_]{3,}\s*$", line):
            continue
        # Skip label-only lines like "Research Topics:" or "Topics:"
        if re.match(r"^(research\s+)?topics?\s*:?\s*$", line, re.IGNORECASE):
            continue
        # Skip common instruction/boilerplate text
        if re.match(r"^(copy|paste|enter|put|the following|below|above)\b", line, re.IGNORECASE):
            continue
        # Strip leading numbering: "1. ", "1) ", "(1) ", "1: ", "1 - "
        line = re.sub(r"^\(?\d+[\.\)\:\-]\)?\s*", "", line)
        # Strip leading letter numbering: "a. ", "a) ", "A. ", "A) "
        line = re.sub(r"^\(?[a-zA-Z][\.\)]\s*", "", line)
        # Strip leading roman numerals: "i. ", "ii) ", "III. "
        line = re.sub(r"^\(?[ivxIVX]+[\.\)]\s*", "", line)
        # Strip leading bullets (common unicode + ascii)
        line = re.sub(r"^[-\*•◦▪►→–—]\s*", "", line)
        # Strip surrounding markdown bold/italic
        line = re.sub(r"^\*{1,2}(.+?)\*{1,2}$", r"\1", line)
        # Strip surrounding quotes
        line = re.sub(r'^["\u201c\u201d\']+(.+?)["\u201c\u201d\']+$', r"\1", line)
        line = line.strip()
        if line:
            topics.append(line)
    return topics


def run_stage2(mod, config_dict: dict, *,
               research_context: str = "", research_topics_text: str = ""):
    """Run stage 2 and save outputs.

    Parameters
    ----------
    research_context : str
        Free-text research context from Stage 1 (problem framing, constraints, etc.).
    research_topics_text : str
        Research topics from Stage 1, one per line (numbered or plain).
    """
    from common import (
        save_handoff, save_summary_stage2,
        save_meta, save_timings,
    )

    if not research_context.strip():
        raise ValueError("Please paste your Stage 1 research context.")
    if not research_topics_text.strip():
        raise ValueError("Please paste your Stage 1 research topics.")

    import ipywidgets as _widgets

    _s2_cfg = config_dict["stage2_research"]
    research_topics = _parse_topics(research_topics_text)
    n = len(research_topics)
    print(f"Research context: {len(research_context.strip())} chars")
    print(f"Topics ({n}):")
    for i, t in enumerate(research_topics, 1):
        print(f"  {i}. {t}")

    # Build the research brief that gets passed to LLM prompts
    topics_text = "\n".join(f"  {i}. {t}" for i, t in enumerate(research_topics, 1))
    research_brief = f"{research_context.strip()}\n\n## Research Topics\n{topics_text}"

    agent = mod.build_graph(checkpointer=MemorySaver())
    lc_config = {"configurable": {"thread_id": "colab-stage2-1"}}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(ts)
    output_dir.mkdir(parents=True, exist_ok=True)
    mod.set_output_dir(output_dir)
    print(f"Output dir: {output_dir.resolve()}\n")

    # ── Progress bar (displayed at top of cell output, updates in-place) ──
    _first = research_topics[0][:60] if research_topics else ""
    _bar = _widgets.IntProgress(
        value=0, min=0, max=n, bar_style="info",
        layout=_widgets.Layout(width="95%", height="22px"),
    )
    _label = _widgets.HTML(
        value=(
            f"<b style='font-size:14px'>Stage 2 — 0 / {n} topics approved &nbsp;|&nbsp; "
            f"Searching for Topic 1: <em>{_first}</em></b>"
        )
    )
    _progress_box = _widgets.VBox(
        [_label, _bar],
        layout=_widgets.Layout(
            border="2px solid #2E75B6", padding="10px 16px",
            border_radius="6px", margin="0 0 12px 0", width="95%",
        ),
    )
    display(_progress_box)

    # Shared animation CSS reused by gate cards and placeholder slots
    import threading
    import html as _html
    import ipywidgets as _w

    _SLOT_CSS = (
        "<style>"
        "@keyframes s2-pulse{0%,100%{opacity:.35}50%{opacity:1}}"
        ".s2-dot{display:inline-block;animation:s2-pulse 1.2s ease-in-out infinite}"
        ".s2-dot:nth-child(2){animation-delay:.2s}"
        ".s2-dot:nth-child(3){animation-delay:.4s}"
        "</style>"
    )
    _SLOT_HTML = (
        _SLOT_CSS +
        "<div style='border:2px solid #90caf9;padding:10px 14px;"
        "border-radius:6px;background:#e3f2fd;color:#1565c0;"
        "font-size:14px;margin:6px 0'>"
        "Preparing research proposal&nbsp;"
        "<span class='s2-dot'>.</span>"
        "<span class='s2-dot'>.</span>"
        "<span class='s2-dot'>.</span>"
        "</div>"
    )

    _current_slot: list = [None]

    def _new_slot():
        slot = _w.Output()
        display(slot)
        with slot:
            display(HTML(_SLOT_HTML))
        _current_slot[0] = slot

    _new_slot()  # placeholder for first research proposal

    # on_node: fires after each graph node completes — updates label with current phase
    _NODE_PHASE = {
        "research_and_propose": "Critic reviewing",
        "topic_critic":         "Awaiting your review of",
        "human_gate_3":         "Searching for",
    }

    def _on_node(node_name, _updates):
        phase = _NODE_PHASE.get(node_name)
        if not phase:
            return
        vals      = agent.get_state(lc_config).values
        approved  = len(vals.get("approved_topics", []))
        topic_idx = vals.get("current_topic_idx", 0)
        _bar.value = approved
        if node_name == "human_gate_3":
            if topic_idx < n:
                nm = research_topics[topic_idx][:60]
                _label.value = (
                    f"<b style='font-size:14px'>Stage 2 — {approved} / {n} topics approved "
                    f"&nbsp;|&nbsp; {phase} Topic {topic_idx + 1}: <em>{nm}</em></b>"
                )
            else:
                _bar.bar_style = "success"
                _label.value = (
                    f"<b style='font-size:14px;color:#388e3c'>"
                    f"✓ All {n} topics approved — finalizing…</b>"
                )
        else:
            capped = min(topic_idx, n - 1)
            nm = research_topics[capped][:60]
            _label.value = (
                f"<b style='font-size:14px'>Stage 2 — {approved} / {n} topics approved "
                f"&nbsp;|&nbsp; {phase} Topic {capped + 1}: <em>{nm}</em></b>"
            )

    def _on_gate(snapshot):
        # Bar value is already kept current by _on_node; this is a safety sync.
        _bar.value = len(snapshot.values.get("approved_topics", []))

    initial_state = {
        "research_topics": research_topics,
        "research_brief": research_brief,
        "current_topic_idx": 0,
        "current_debate_round": 0,
        "approved_topics": [],
    }

    # ── Widget gate (replaces input() with button-based UI) ──────────────────

    def _widget_gate(interrupt_text: str) -> str:
        # Strip the instruction footer ("---\n- Press Enter…") since buttons replace it
        footer_idx = interrupt_text.find("\n---\n- ")
        if footer_idx != -1:
            interrupt_text = interrupt_text[:footer_idx].strip()

        event = threading.Event()
        result_holder = ["approved"]
        gate_out = _current_slot[0]  # reuse the pre-created placeholder slot

        feedback_area = _w.Textarea(
            placeholder="Type feedback here, or leave blank and click Approve…",
            layout=_w.Layout(width="90%", height="80px"),
        )
        submit_btn = _w.Button(
            description="Submit Feedback",
            button_style="primary",
            layout=_w.Layout(width="170px", height="36px"),
        )
        approve_btn = _w.Button(
            description="✓ Approve & Continue",
            button_style="success",
            layout=_w.Layout(width="190px", height="36px"),
        )

        def _on_approve(b):
            submit_btn.disabled = True
            approve_btn.disabled = True
            with gate_out:
                gate_out.clear_output(wait=True)
                display(HTML(
                    _SLOT_CSS +
                    "<div style='border:2px solid #a5d6a7;padding:10px 14px;"
                    "border-radius:6px;background:#f1f8e9;opacity:0.85'>"
                    "<b style='color:#388e3c'>✓ Approved</b>"
                    " &nbsp;—&nbsp; generating next research proposal&nbsp;"
                    "<span class='s2-dot'>.</span>"
                    "<span class='s2-dot'>.</span>"
                    "<span class='s2-dot'>.</span>"
                    "</div>"
                ))
            _new_slot()  # immediately show placeholder for the next topic
            result_holder[0] = "approved"
            event.set()

        def _on_submit(b):
            submit_btn.disabled = True
            approve_btn.disabled = True
            answer = feedback_area.value.strip()
            with gate_out:
                gate_out.clear_output(wait=True)
                display(HTML(
                    _SLOT_CSS +
                    "<div style='border:2px solid #90caf9;padding:10px 14px;"
                    "border-radius:6px;background:#e3f2fd;opacity:0.85'>"
                    "Sending feedback and revising&nbsp;"
                    "<span class='s2-dot'>.</span>"
                    "<span class='s2-dot'>.</span>"
                    "<span class='s2-dot'>.</span>"
                    "</div>"
                ))
            _new_slot()  # immediately show placeholder for the revised proposal
            result_holder[0] = answer
            event.set()

        submit_btn.on_click(_on_submit)
        approve_btn.on_click(_on_approve)

        escaped = _html.escape(interrupt_text)
        with gate_out:
            gate_out.clear_output(wait=True)  # replace "..." placeholder with gate content
            display(_w.VBox([
                _w.HTML(
                    "<div style='border:3px solid #1565c0;padding:16px;border-radius:8px;"
                    "background:#e3f2fd;margin-bottom:10px'>"
                    "<span style='font-size:18px;font-weight:bold;color:#1565c0'>"
                    "Stage 2 — Topic Review</span>"
                    f"<div style='margin-top:12px;font-size:14px;white-space:pre-wrap;"
                    f"border-left:3px solid #1565c0;padding-left:12px'>{escaped}</div>"
                    "</div>"
                    "<p style='color:#555;font-size:13px;margin:0 0 6px'>"
                    "Review the research proposal. Provide feedback to revise it, "
                    "or approve to move to the next topic.</p>"
                ),
                feedback_area,
                _w.HBox(
                    [submit_btn, approve_btn],
                    layout=_w.Layout(gap="10px", margin="6px 0 0 0"),
                ),
            ]))

        event.wait()
        return result_holder[0]

    result, elapsed = _run_graph(
        agent, lc_config, initial_state, output_dir,
        on_gate=_on_gate, get_feedback=_widget_gate, on_node=_on_node,
    )

    # Clear the last placeholder slot (no more topics coming)
    if _current_slot[0] is not None:
        _current_slot[0].clear_output()

    # Final progress update
    approved_ct = len(result.get("approved_topics", []))
    _bar.value = n
    _bar.bar_style = "success"
    _label.value = (
        f"<b style='font-size:14px;color:#388e3c'>"
        f"✓ Stage 2 Complete — {approved_ct} / {n} topics approved &nbsp;|&nbsp; "
        f"Elapsed: {elapsed:.0f}s</b>"
    )

    handoff_out = {
        "research_topics": research_topics,
        "research_brief": research_brief,
        "approved_topics": result.get("approved_topics", []),
        "config": {
            "model_researcher": _s2_cfg["model_researcher"],
            "model_critic": _s2_cfg["model_critic"],
        },
    }
    save_handoff(handoff_out, output_dir)
    save_summary_stage2(handoff_out, output_dir)

    save_meta([
        f"Stage:            2 — Research & Debate",
        f"Timestamp:        {datetime.now().isoformat()}",
        f"Elapsed:          {elapsed:.1f}s",
        f"",
        f"Models:",
        f"  researcher    {_s2_cfg['model_researcher']}",
        f"  critic        {_s2_cfg['model_critic']}",
        f"",
        f"Settings:",
        f"  Max web search results:     {_s2_cfg['max_web_search_ct']}",
        f"  Max debate rounds:          {_s2_cfg['max_debate_rounds']}",
        f"  Max proposal revisions:     {_s2_cfg['max_human_revision_on_proposal']}",
    ], output_dir)

    save_timings()

    drive_dest = _copy_to_drive(output_dir, "stage2_research")

    display(HTML("<hr style='border:4px solid #1B2A4A; margin:24px 0'>"))
    display(Markdown("## Stage 2 Complete"))
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Local:   {output_dir.resolve()}")
    return result


# ============================================================================
# STAGE 3
# ============================================================================

def load_stage3(config_ui, repo_dir: str):
    """Snapshot stage 3 widgets, write config, import module."""
    config_dict = config_ui.get_config_stage3()
    _write_config(config_dict, repo_dir)
    _ensure_path(repo_dir)
    _reload_common()

    if "stage3_synthesis" in sys.modules:
        mod = importlib.reload(sys.modules["stage3_synthesis"])
    else:
        import stage3_synthesis as mod

    return mod, config_dict


def run_stage3(mod, config_dict: dict, handoff_path: str = "", handoff: dict | None = None):
    """Run stage 3 and save final reports.

    Parameters
    ----------
    handoff_path : str
        Path to the uploaded handoff.json file from Stage 2. Used when running
        Stage 3 standalone.
    handoff : dict | None
        In-memory Stage 2 result (the dict returned by ``run_stage2``). Used when
        running Stages 2 and 3 back-to-back in the same notebook. Takes precedence
        over ``handoff_path`` if both are provided.
    """
    import json as _json
    from common import save_meta, save_timings

    if handoff is not None:
        handoff_in = handoff
        handoff_source = "in-memory Stage 2 result"
    elif handoff_path:
        with open(handoff_path, "r", encoding="utf-8") as f:
            handoff_in = _json.load(f)
        handoff_source = handoff_path
    else:
        raise ValueError("Please provide a Stage 2 handoff (either handoff_path or handoff dict).")
    print(f"Loaded handoff from: {handoff_source}")
    print(f"Approved topics: {len(handoff_in.get('approved_topics', []))}")

    agent = mod.build_graph(checkpointer=MemorySaver())
    lc_config = {"configurable": {"thread_id": "colab-stage3-1"}}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(ts)
    output_dir.mkdir(parents=True, exist_ok=True)
    mod.set_output_dir(output_dir)
    print(f"Output dir: {output_dir.resolve()}\n")

    initial_state = {
        "research_topics": handoff_in.get("research_topics", []),
        "research_brief": handoff_in.get("research_brief", ""),
        "approved_topics": handoff_in["approved_topics"],
    }

    result, elapsed = _run_graph(agent, lc_config, initial_state, output_dir)

    # Save reports (md + docx)
    _s3_cfg = config_dict["stage3_synthesis"]
    stage_config = handoff_in.get("config", {})
    export_config = {
        "input_query": handoff_in.get("research_brief", "")[:200],
        "agents": {
            "researcher":  {"model": stage_config.get("model_researcher", "?")},
            "critic":      {"model": stage_config.get("model_critic", "?")},
            "synthesizer": {"model": _s3_cfg["model"]},
        },
    }
    report_export.save_all(result, export_config, output_dir, elapsed)

    save_meta([
        f"Stage:            3 — Synthesis & Action Plan",
        f"Timestamp:        {datetime.now().isoformat()}",
        f"Input (Stage 2):  {handoff_source}",
        f"Elapsed:          {elapsed:.1f}s",
        f"",
        f"Model:",
        f"  synthesizer   {_s3_cfg['model']}",
        f"",
        f"Settings:",
        f"  Max plan revisions:   {_s3_cfg['max_human_revision_on_plan']}",
    ], output_dir)

    save_timings()

    drive_dest = _copy_to_drive(output_dir, "stage3_synthesis")

    display(HTML("<hr style='border:4px solid #1B2A4A; margin:24px 0'>"))
    display(Markdown("## Stage 3 Complete"))
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Local:   {output_dir.resolve()}")
    if drive_dest:
        print(f"Drive:   {drive_dest}")

    return result
