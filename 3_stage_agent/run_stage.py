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


def _run_graph(agent, lc_config, initial_state, output_dir):
    """Shared interrupt loop for all stages."""
    report_export.start_log(output_dir)

    try:
        t0 = time.time()
        result = agent.invoke(initial_state, lc_config)

        gate_num = 0
        while True:
            snapshot = agent.get_state(lc_config)
            if not snapshot.next:
                break

            for task in snapshot.tasks:
                if hasattr(task, "interrupts"):
                    for intr in task.interrupts:
                        gate_num += 1
                        display(HTML("<hr style='border:3px solid #2E75B6; margin:24px 0 8px 0'>"))
                        display(Markdown(f"### Human Gate {gate_num}"))
                        display(Markdown(str(intr.value)))
                        display(HTML("<hr style='border:3px solid #2E75B6; margin:8px 0 16px 0'>"))

                        feedback = input("approve / or type feedback > ").strip()
                        if not feedback:
                            feedback = "approved"
                        result = agent.invoke(Command(resume=feedback), lc_config)

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
      - Numbered:  "1. Topic", "1) Topic", "(1) Topic", "1: Topic"
      - Bulleted:  "- Topic", "* Topic", "• Topic"
      - Markdown headers/bold: "## Research Topics", "**Topic**"
      - Separator lines: "---", "==="
      - Plain lines (one topic per line)
    """
    import re
    topics = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip markdown headers (e.g. "## Research Topics")
        if re.match(r"^#{1,6}\s", line):
            continue
        # Skip separator lines (---, ===, ***)
        if re.match(r"^[-=\*_]{3,}\s*$", line):
            continue
        # Strip leading numbering: "1. ", "1) ", "(1) ", "1: "
        line = re.sub(r"^\(?\d+[\.\)\:]\)?\s*", "", line)
        # Strip leading bullets: "- ", "* ", "• "
        line = re.sub(r"^[-\*•]\s+", "", line)
        # Strip surrounding markdown bold: **Topic** -> Topic
        line = re.sub(r"^\*\*(.+?)\*\*$", r"\1", line)
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

    _s2_cfg = config_dict["stage2_research"]
    research_topics = _parse_topics(research_topics_text)
    print(f"Research context: {len(research_context.strip())} chars")
    print(f"Topics ({len(research_topics)}):")
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

    initial_state = {
        "research_topics": research_topics,
        "research_brief": research_brief,
        "current_topic_idx": 0,
        "current_debate_round": 0,
        "approved_topics": [],
    }

    result, elapsed = _run_graph(agent, lc_config, initial_state, output_dir)

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


def run_stage3(mod, config_dict: dict, handoff_path: str = ""):
    """Run stage 3 and save final reports.

    Parameters
    ----------
    handoff_path : str
        Path to the uploaded handoff.json file from Stage 2.
    """
    import json as _json
    from common import save_meta, save_timings

    if not handoff_path:
        raise ValueError("Please upload your Stage 2 handoff.json file.")
    with open(handoff_path, "r", encoding="utf-8") as f:
        handoff_in = _json.load(f)
    print(f"Loaded handoff from: {handoff_path}")
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
        f"Input (Stage 2):  {handoff_path}",
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
