"""Load config, import agent module, and run the MBA agent interactively."""

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


def load(config_ui, repo_dir: str):
    """Snapshot widget values, write config.yaml, and import the agent module.

    Returns
    -------
    mba : module
        The (re)loaded mba_agent_v4 module.
    config_dict : dict
        The CONFIG dict snapshot from the widgets.
    """
    config_dict = config_ui.get_config()

    config_path = Path(repo_dir) / "MBA-course" / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Config written to {config_path}")

    agent_dir = str(Path(repo_dir) / "MBA-course")
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
        print(f"Agent module path added: {agent_dir}")

    try:
        import mba_agent_v4
        mba = importlib.reload(mba_agent_v4)
    except Exception:
        import mba_agent_v4 as mba

    return mba, config_dict


def run(mba, config_dict: dict):
    """Run the agent, handle human-in-the-loop gates, save reports + log.

    Parameters
    ----------
    mba : module
        The imported mba_agent_v4 module.
    config_dict : dict
        The CONFIG dict from config_ui.get_config().

    Returns
    -------
    result : dict
        Final agent state.
    """
    agent = mba.build_graph(checkpointer=MemorySaver())
    lc_config = {"configurable": {"thread_id": "colab-run-1"}}

    # ── Timestamped output dir ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(ts)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir.resolve()}\n")

    # ── Start logging ──
    report_export.start_log(output_dir)

    try:
        # ── Run ──
        t0 = time.time()
        result = agent.invoke({"user_query": mba.INPUT_QUERY}, lc_config)

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
                        display(Markdown(f"### Human Input {gate_num}"))
                        display(Markdown(str(intr.value)))
                        display(HTML("<hr style='border:3px solid #2E75B6; margin:8px 0 16px 0'>"))

                        feedback = input("approve / or type feedback > ").strip()
                        if not feedback:
                            feedback = "approved"
                        result = agent.invoke(Command(resume=feedback), lc_config)

        elapsed = time.time() - t0

        # ── Save reports ──
        report_export.save_all(result, config_dict, output_dir, elapsed)

        # ── Copy to Google Drive ──
        if Path("/content/drive/MyDrive").exists():
            DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            drive_dest = DRIVE_OUTPUT_DIR / output_dir.name
            shutil.copytree(output_dir, drive_dest)
            print(f"Drive:    {drive_dest}")
        else:
            print("[SKIP] Google Drive not mounted — Drive save skipped")

        display(HTML("<hr style='border:4px solid #1B2A4A; margin:24px 0'>"))
        display(Markdown("## Agent Complete"))
        print(f"Elapsed: {elapsed:.0f}s")

        return result

    finally:
        # Always restore stdout, even if interrupted or crashed
        report_export.stop_log()
