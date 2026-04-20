"""
local_agent/config_ui.py — Local override of the cloned repo's config_ui.py.

Stage 1 changes:
  Design 1: Config persists to stage1_config.yaml; dropdowns pre-fill on rerun.
             A Reset button (next to the form) is the only way to clear saved config.
  Design 2: Business question is read from my_question.txt (Drive-first, cwd fallback).
             File created empty at setup; empty file shows a warning and blocks the LLM.

Stage 2: input form replaced — auto-loads Stage 1 handoff.json instead of copy-paste.
"""

import os
from datetime import datetime
from pathlib import Path

import yaml
import ipywidgets as widgets
from IPython.display import display, HTML

_FLASH_CSS = (
    "<style>"
    "@keyframes s1-sf{0%{background:#a5d6a7}100%{background:transparent}}"
    "@keyframes s1-ef{0%{background:#ef9a9a}100%{background:transparent}}"
    "@keyframes s1-df{0%{background:#a5d6a7}100%{background:#f1f8e9}}"
    ".s1-sf{animation:s1-sf 1.4s ease-out forwards;display:inline-block;"
    "border-radius:4px;padding:1px 6px}"
    ".s1-ef{animation:s1-ef 1.4s ease-out forwards;display:inline-block;"
    "border-radius:4px;padding:1px 6px}"
    ".s1-df{animation:s1-df 1.4s ease-out forwards}"
    "</style>"
)

# ============================================================================
# Shared constants
# ============================================================================

MODEL_OPTIONS = [
    "openai/gpt-5.2",
    "anthropic/claude-opus-4.6",
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
]

_label_layout  = widgets.Layout(width="220px")
_widget_layout = widgets.Layout(width="320px")
_slider_layout = widgets.Layout(width="260px")

# ============================================================================
# Design 2 — File paths (Drive-first, local mimic fallback, cwd last resort)
# ============================================================================

def _project_dir() -> Path:
    """
    Returns the persistent project directory:
      Colab: /content/drive/MyDrive/AI_Essentials_Final_Project  (Drive mounted)
      Local: <project_root>/content/drive/MyDrive/AI_Essentials_Final_Project

    This file lives at <root>/content/Final_Project_Sp26/3_stage_agent/config_ui.py
    so parents[3] == <root> in both environments.
    """
    mydrive = Path("/content/drive/MyDrive")
    if mydrive.exists():
        p = mydrive / "AI_Essentials_Final_Project"
        p.mkdir(parents=True, exist_ok=True)
        return p
    mimic = Path(__file__).parents[3] / "content" / "drive" / "MyDrive" / "AI_Essentials_Final_Project"
    mimic.mkdir(parents=True, exist_ok=True)
    return mimic

def _config_path() -> Path:
    return _project_dir() / "stage1_config.yaml"

def _question_path() -> Path:
    return _project_dir() / "my_question.txt"

def _constraints_path() -> Path:
    return _project_dir() / "my_constraints.txt"

def _topics_path() -> Path:
    return _project_dir() / "my_topics.txt"

# ============================================================================
# Design 1 — YAML helpers
# ============================================================================

def _load_yaml(path: Path) -> dict | None:
    try:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return None

def _save_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

# ============================================================================
# Design 2 — Question file helpers
# ============================================================================

def create_question_file():
    """
    Called during the setup cell (step 3).
    Creates my_question.txt empty if it does not already exist.
    """
    path = _question_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("")
        display(HTML(f"""
            <div style='border:2px solid #1976d2;padding:12px;border-radius:6px;
                        background:#e3f2fd;margin:6px 0'>
            <b style='color:#1976d2'>📄 my_question.txt created</b><br>
            Open it in the file browser (left sidebar → folder icon), type your
            business question, save the file, then re-run Step 5.<br>
            <code style='font-size:12px'>{path}</code>
            </div>
        """))
    else:
        display(HTML(f"""
            <div style='color:#555;font-size:13px;margin:4px 0'>
            ✓ <code>my_question.txt</code> already exists at
            <code>{path}</code>
            </div>
        """))


def _is_instruction_line(line: str) -> bool:
    """
    Returns True if a line is our auto-generated instruction text.
    Matches whether or not the student removed the leading '#'.
    """
    core = line.strip().lstrip("#").strip()
    return core.startswith("⚠ Please enter your business question")


def read_question() -> str:
    """
    Reads my_question.txt and returns the question string.
    Strips comment lines (#) and instruction lines (with or without #).
    Raises ValueError with a visible warning if nothing remains.
    """
    path = _question_path()
    if not path.exists():
        create_question_file()

    raw = path.read_text(encoding="utf-8")
    lines = [
        l for l in raw.splitlines()
        if not l.strip().startswith("#") and not _is_instruction_line(l)
    ]
    question = "\n".join(lines).strip()

    if not question:
        # Write a single reminder line — no "delete this" instruction
        path.write_text(
            "# ⚠ Please enter your business question below this line, then re-run Step 5.\n\n"
        )
        display(HTML("""
            <div style='border:3px solid #e53935;padding:16px;border-radius:8px;
                        background:#fff3f3;margin:8px 0'>
            <b style='font-size:18px;color:#e53935'>⚠ No question found</b><br><br>
            Please enter and save your question in the Step 4 Setup cell, then re-run this cell.<br>
            <b>The AI will not start until a question is provided.</b>
            </div>
        """))
        raise ValueError("my_question.txt is empty — LLM not called.")

    display(HTML(f"""
        <div style='border:2px solid #388e3c;padding:10px;border-radius:6px;
                    background:#f1f8e9;margin:8px 0'>
        <b style='color:#388e3c'>✓ Question loaded from my_question.txt:</b><br>
        <span style='font-size:14px'>{question[:300]}{"…" if len(question) > 300 else ""}</span>
        </div>
    """))
    return question

# ============================================================================
# Stage 1 — widgets & defaults
# ============================================================================

_DEFAULTS = {
    "model":               "openai/gpt-5.2",
    "max_clarify_rounds":  5,
    "max_research_topics": 2,
    "max_topics_revision": 2,
}

_s1_query          = widgets.Textarea(
    placeholder="Type your business question here, or edit my_question.txt directly…",
    layout=widgets.Layout(width="560px", height="100px"),
)
_s1_initial_question: str = ""  # value pre-filled at show_stage1() time; used to detect file edits
_s1_model          = widgets.Dropdown(options=MODEL_OPTIONS, value=_DEFAULTS["model"], layout=_widget_layout)
_s1_max_clarify    = widgets.BoundedIntText(value=_DEFAULTS["max_clarify_rounds"],  min=1, max=10, layout=_slider_layout)
_s1_max_topics     = widgets.BoundedIntText(value=_DEFAULTS["max_research_topics"], min=1, max=5,  layout=_slider_layout)
_s1_max_topics_rev = widgets.BoundedIntText(value=_DEFAULTS["max_topics_revision"], min=1, max=5,  layout=_slider_layout)


def _apply_values(d: dict):
    _s1_model.value          = d.get("model",               _DEFAULTS["model"])
    _s1_max_clarify.value    = d.get("max_clarify_rounds",  _DEFAULTS["max_clarify_rounds"])
    _s1_max_topics.value     = d.get("max_research_topics", _DEFAULTS["max_research_topics"])
    _s1_max_topics_rev.value = d.get("max_topics_revision", _DEFAULTS["max_topics_revision"])


def _labeled(label_text, w, hint=None):
    row = widgets.HBox([widgets.Label(label_text, layout=_label_layout), w])
    if hint:
        hint_w = widgets.HTML(
            f'<p style="color:#888;font-size:12px;margin:0 0 4px 220px;">{hint}</p>'
        )
        return widgets.VBox([row, hint_w])
    return row


_s1_save_btn    = widgets.Button(
    description="Save Question",
    button_style="primary",
    icon="save",
    layout=widgets.Layout(width="160px", margin="6px 0 0 0"),
)
_s1_save_status = widgets.Output()

_s1_form = widgets.VBox([
    widgets.HTML(
        '<p style="color:#000;font-weight:500;font-size:18px;margin:0 0 8px;">'
        'Enter your business question and configure the intake settings.</p>'
        '<p style="color:#555;font-size:13px;margin:0 0 4px;">'
        'Your question is also saved to <code>my_question.txt</code> — '
        'you can edit either the box below or the file directly. Both stay in sync.</p>'
    ),
    widgets.HTML("<h3>Business Question</h3>"),
    _s1_query,
    _s1_save_btn,
    _s1_save_status,
    widgets.HTML("<h3>Model</h3>"),
    _labeled("Intake model:", _s1_model),
    widgets.HTML("<h3>Settings</h3>"),
    _labeled("Max clarify rounds:", _s1_max_clarify,
             "How many times the agent can ask you clarifying questions"),
    _labeled("Max research topics:", _s1_max_topics,
             "Number of topics the agent will come up with to research on"),
    _labeled("Max topic revisions:", _s1_max_topics_rev,
             "How many times you can revise the research topic list"),
])

_status_banner = widgets.Output()


def _save_question_to_file():
    """Write current textarea value to my_question.txt."""
    text = _s1_query.value.strip()
    _question_path().parent.mkdir(parents=True, exist_ok=True)
    _question_path().write_text(text, encoding="utf-8")
    return text


def show_stage1():
    """Display config UI, pre-filling question and dropdowns from saved files."""
    global _s1_initial_question
    # Pre-fill question textarea from txt (strip comment lines)
    qpath = _question_path()
    if qpath.exists():
        raw = qpath.read_text(encoding="utf-8")
        lines = [l for l in raw.splitlines() if not l.strip().startswith("#")]
        _s1_query.value = "\n".join(lines).strip()
    _s1_initial_question = _s1_query.value  # snapshot so get_config_stage1 can detect direct file edits

    # Pre-fill dropdowns from YAML
    saved = _load_yaml(_config_path())
    with _status_banner:
        _status_banner.clear_output()
        if saved:
            _apply_values(saved)
            display(HTML(
                "<div style='border:2px solid #388e3c;padding:8px 12px;border-radius:6px;"
                "background:#f1f8e9;margin-bottom:8px'>"
                f"✓ Loaded your previous settings from "
                f"<code>{_config_path().name}</code></div>"
            ))
        else:
            display(HTML(
                "<div style='border:2px solid #1976d2;padding:8px 12px;border-radius:6px;"
                "background:#e3f2fd;margin-bottom:8px'>"
                "Using default settings — your choices will be saved when you run Step 5.</div>"
            ))

    # Wire Save Question button
    def _on_save(b):
        text = _save_question_to_file()
        ts = datetime.now().strftime("%H:%M:%S")
        with _s1_save_status:
            _s1_save_status.clear_output()
            if text:
                display(HTML(
                    _FLASH_CSS +
                    "<span class='s1-sf' style='color:#388e3c;font-size:13px'>"
                    f"✓ Saved to <code>{_question_path().name}</code> "
                    f"<span style='color:#aaa'>({ts})</span></span>"
                ))
            else:
                display(HTML(
                    _FLASH_CSS +
                    "<span class='s1-ef' style='color:#e53935;font-size:13px'>"
                    f"⚠ Question is empty — please type something first. "
                    f"<span style='color:#aaa'>({ts})</span></span>"
                ))

    _s1_save_btn.on_click(_on_save)

    # Save Config / Reset to Defaults buttons
    save_config_btn = widgets.Button(
        description="Save Config",
        button_style="primary",
        icon="check",
        layout=widgets.Layout(width="150px", margin="12px 0 0 0"),
    )
    reset_btn = widgets.Button(
        description="Reset to Defaults",
        button_style="warning",
        icon="refresh",
        layout=widgets.Layout(width="180px", margin="12px 0 0 6px"),
    )

    def _on_save_config(b):
        cfg = {
            "model":               _s1_model.value,
            "max_clarify_rounds":  _s1_max_clarify.value,
            "max_research_topics": _s1_max_topics.value,
            "max_topics_revision": _s1_max_topics_rev.value,
        }
        _save_yaml(_config_path(), cfg)
        ts = datetime.now().strftime("%H:%M:%S")
        with _status_banner:
            _status_banner.clear_output()
            display(HTML(
                _FLASH_CSS +
                "<div class='s1-df' style='border:2px solid #388e3c;padding:8px 12px;"
                "border-radius:6px;background:#f1f8e9;margin-bottom:8px'>"
                f"✓ Settings saved to <code>{_config_path().name}</code> "
                f"<span style='color:#aaa;font-size:12px'>({ts})</span></div>"
            ))

    def _on_reset(b):
        _config_path().unlink(missing_ok=True)
        _apply_values(_DEFAULTS)
        with _status_banner:
            _status_banner.clear_output()
            display(HTML(
                "<div style='border:2px solid #f57c00;padding:8px 12px;border-radius:6px;"
                "background:#fff8e1;margin-bottom:8px'>"
                "↺ Reset to defaults. Saved config deleted.</div>"
            ))

    save_config_btn.on_click(_on_save_config)
    reset_btn.on_click(_on_reset)
    display(widgets.VBox([
        _status_banner,
        _s1_form,
        widgets.HBox([save_config_btn, reset_btn]),
    ]))


def check_question() -> bool:
    """Show a warning if no question is saved; confirmation if one is ready. Returns True/False."""
    qpath = _question_path()
    raw = qpath.read_text(encoding="utf-8") if qpath.exists() else ""
    lines = [l for l in raw.splitlines()
             if not l.strip().startswith("#") and not _is_instruction_line(l)]
    question = "\n".join(lines).strip()

    if not question:
        display(HTML("""
            <div style='border:3px solid #e53935;padding:16px;border-radius:8px;
                        background:#fff3f3;margin:8px 0'>
            <b style='font-size:16px;color:#e53935'>&#9888; No question found</b><br><br>
            Please enter and save your question in the Step 4 Setup cell above,
            then re-run this cell.
            </div>
        """))
        return False

    display(HTML(
        f"<div style='border:2px solid #388e3c;padding:10px;border-radius:6px;"
        f"background:#f1f8e9;margin:8px 0'>"
        f"<b style='color:#388e3c'>&#10003; Question ready:</b><br>"
        f"<span style='font-size:14px'>{question[:300]}{'&hellip;' if len(question) > 300 else ''}</span>"
        f"</div>"
    ))
    return True


def get_config_stage1() -> dict:
    """Return config dict, syncing textarea → txt and saving settings to YAML."""
    stage1_cfg = {
        "model":               _s1_model.value,
        "max_clarify_rounds":  _s1_max_clarify.value,
        "max_research_topics": _s1_max_topics.value,
        "max_topics_revision": _s1_max_topics_rev.value,
    }
    _save_yaml(_config_path(), stage1_cfg)

    # Determine the question source.
    # If the textarea was changed since setup, it's the primary source → sync to file.
    # If the textarea is unchanged (or empty), the file may have been edited directly
    # → read from file without overwriting it first.
    textarea_val = _s1_query.value.strip()
    if textarea_val and textarea_val != _s1_initial_question:
        # User typed something new in the textarea — show confirmation here
        question = textarea_val
        _save_question_to_file()
        display(HTML(f"""
            <div style='border:2px solid #388e3c;padding:10px;border-radius:6px;
                        background:#f1f8e9;margin:8px 0'>
            <b style='color:#388e3c'>✓ Question confirmed:</b><br>
            <span style='font-size:14px'>{question[:300]}{"…" if len(question) > 300 else ""}</span>
            </div>
        """))
    else:
        # Textarea empty or unchanged — file may have been edited directly.
        # read_question() already displays its own confirmation, so don't add another.
        try:
            question = read_question()   # raises ValueError if still empty
            _s1_query.value = question   # sync textarea ← file
        except ValueError:
            # Mirror the error into the config UI widget (Cell 8) so it appears
            # right below the question box — not just in the agent start cell.
            with _s1_save_status:
                _s1_save_status.clear_output()
                display(HTML("""
                    <div style='border:3px solid #e53935;padding:12px;border-radius:8px;
                                background:#fff3f3;margin:4px 0'>
                    <b style='font-size:15px;color:#e53935'>⚠ No question found</b><br><br>
                    Please enter and save your question in the Step 4 Setup cell, then re-run Cell 5a.
                    </div>
                """))
            raise

    return {
        "input_query":         question,
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "auto_approve":        False,
        "stage1_intake":       stage1_cfg,
        "stage2_research": {
            "model_researcher":              "openai/gpt-5.2",
            "model_critic":                  "openai/gpt-5.2",
            "max_web_search_ct":             10,
            "max_debate_rounds":             2,
            "max_human_revision_on_proposal": 2,
        },
        "stage3_synthesis": {
            "model":                    "anthropic/claude-opus-4.6",
            "max_human_revision_on_plan": 2,
        },
    }


# ============================================================================
# Stage 2 — load Stage 1 output from handoff.json
# ============================================================================

import json as _json

_s2_loaded: dict = {}  # holds parsed handoff data after clicking Load


def _stage1_output_dir() -> Path:
    return _project_dir() / "stage1_intake"


def _latest_handoff() -> "Path | None":
    out_dir = _stage1_output_dir()
    if not out_dir.exists():
        return None
    runs = sorted(
        [d for d in out_dir.iterdir() if d.is_dir() and (d / "handoff.json").exists()],
        key=lambda d: d.name,
        reverse=True,
    )
    return runs[0] / "handoff.json" if runs else None


_s2_path_widget = widgets.Text(
    value="",
    placeholder="Auto-detected, or paste path to your Stage 1 output folder",
    layout=widgets.Layout(width="95%"),
)
_s2_load_status = widgets.Output()


def _on_load(b):
    global _s2_loaded
    with _s2_load_status:
        _s2_load_status.clear_output()
        raw_path = _s2_path_widget.value.strip()
        if raw_path:
            candidate = Path(raw_path)
            if candidate.is_file() and candidate.name == "handoff.json":
                handoff_path = candidate
            elif (candidate / "handoff.json").exists():
                handoff_path = candidate / "handoff.json"
            else:
                display(HTML(
                    f"<div style='border:2px solid #d32f2f;padding:10px;border-radius:6px;"
                    f"background:#fff3f3'>❌ No <code>handoff.json</code> found at:"
                    f"<br><code>{raw_path}</code></div>"
                ))
                return
        else:
            handoff_path = _latest_handoff()
            if not handoff_path:
                display(HTML(
                    f"<div style='border:2px solid #d32f2f;padding:10px;border-radius:6px;"
                    f"background:#fff3f3'>"
                    f"❌ No Stage 1 output found in <code>{_stage1_output_dir()}</code>.<br>"
                    f"Run Stage 1 first, or paste the path to your output folder above.</div>"
                ))
                return

        try:
            data = _json.loads(handoff_path.read_text(encoding="utf-8"))
        except Exception as exc:
            display(HTML(
                f"<div style='color:#d32f2f'>❌ Failed to read <code>{handoff_path}</code>: {exc}</div>"
            ))
            return

        _s2_loaded = data
        _s2_path_widget.value = str(handoff_path.parent)
        topics = data.get("research_topics", [])
        display(HTML(
            "<div style='border:2px solid #388e3c;padding:12px;border-radius:6px;background:#f1f8e9'>"
            f"<b style='color:#388e3c'>✓ Loaded from:</b> <code>{handoff_path}</code><br><br>"
            f"<b>Business Question:</b> {data.get('user_query', '')[:300]}<br><br>"
            f"<b>Research Topics ({len(topics)}):</b><ol>"
            + "".join(f"<li>{t}</li>" for t in topics)
            + "</ol></div>"
        ))


_s2_load_btn = widgets.Button(
    description="Load Stage 1 Output",
    button_style="primary",
    icon="download",
    layout=widgets.Layout(width="200px", margin="8px 0"),
)
_s2_load_btn.on_click(_on_load)

_s2_input_form = widgets.VBox([
    widgets.HTML(
        '<b style="font-size:18px;">Load Stage 1 Output</b>'
        '<p style="color:#555;margin:4px 0 8px;">'
        'Click <b>Load Stage 1 Output</b> to auto-detect your latest Stage 1 run, '
        'or paste the path to a specific Stage 1 output folder above first.</p>'
    ),
    _s2_path_widget,
    _s2_load_btn,
    _s2_load_status,
])


def show_stage2_input():
    latest = _latest_handoff()
    if latest:
        _s2_path_widget.value = str(latest.parent)
    display(_s2_input_form)


def get_stage1_output() -> dict:
    if not _s2_loaded:
        raise ValueError(
            "No Stage 1 output loaded — run step 4 and click 'Load Stage 1 Output' first."
        )
    topics: list = _s2_loaded.get("research_topics", [])
    research_topics_text = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(topics))
    research_context = (
        f"## Business Question\n{_s2_loaded.get('user_query', '')}\n\n"
        f"## Problem Framing\n{_s2_loaded.get('problem_framing', '')}\n\n"
        f"**Constraints:** {_s2_loaded.get('constraints_noted', '')}"
    )
    return {
        "research_context":     research_context,
        "research_topics_text": research_topics_text,
    }


_s2_model_researcher  = widgets.Dropdown(options=MODEL_OPTIONS, value="openai/gpt-5.2", layout=_widget_layout)
_s2_model_critic      = widgets.Dropdown(options=MODEL_OPTIONS, value="openai/gpt-5.2", layout=_widget_layout)
_s2_max_web           = widgets.BoundedIntText(value=10, min=1, max=30, layout=_slider_layout)
_s2_max_debate        = widgets.BoundedIntText(value=2,  min=1, max=10, layout=_slider_layout)
_s2_max_rev_proposal  = widgets.BoundedIntText(value=2,  min=1, max=5,  layout=_slider_layout)

_s2_form = widgets.VBox([
    widgets.HTML(
        '<p style="color:#000;font-weight:500;font-size:18px;margin:0 0 8px;">'
        'Choose AI models and settings for the research &amp; debate phase.</p>'
        '<p style="color:#d32f2f;font-weight:600;font-size:18px;margin:0 0 8px;">'
        'Warning: Re-running this cell will reset to defaults.</p>'
    ),
    widgets.HTML("<h3>Models</h3>"),
    _labeled("Researcher:", _s2_model_researcher),
    _labeled("Critic:",     _s2_model_critic),
    widgets.HTML("<h3>Settings</h3>"),
    _labeled("Max web search results:", _s2_max_web,
             "Tavily search results per query per debate round"),
    _labeled("Max debate rounds:", _s2_max_debate,
             "Researcher vs Critic rounds per topic before human review"),
    _labeled("Max proposal revisions:", _s2_max_rev_proposal,
             "How many times you can send a topic back for revision"),
])


def show_stage2():
    display(_s2_form)


def get_config_stage2() -> dict:
    return {
        "input_query":         "",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "auto_approve":        False,
        "stage1_intake": {
            "model":               "openai/gpt-5.2",
            "max_clarify_rounds":  5,
            "max_research_topics": 2,
            "max_topics_revision": 2,
        },
        "stage2_research": {
            "model_researcher":              _s2_model_researcher.value,
            "model_critic":                  _s2_model_critic.value,
            "max_web_search_ct":             _s2_max_web.value,
            "max_debate_rounds":             _s2_max_debate.value,
            "max_human_revision_on_proposal": _s2_max_rev_proposal.value,
        },
        "stage3_synthesis": {
            "model":                    "anthropic/claude-opus-4.6",
            "max_human_revision_on_plan": 2,
        },
    }


# ============================================================================
# Stage 3 — unchanged from original
# ============================================================================

_s3_model       = widgets.Dropdown(options=MODEL_OPTIONS, value="anthropic/claude-opus-4.6", layout=_widget_layout)
_s3_max_rev_plan = widgets.BoundedIntText(value=2, min=1, max=5, layout=_slider_layout)

_s3_form = widgets.VBox([
    widgets.HTML(
        '<p style="color:#000;font-weight:500;font-size:18px;margin:0 0 8px;">'
        'Choose the synthesizer model and settings.</p>'
        '<p style="color:#d32f2f;font-weight:600;font-size:18px;margin:0 0 8px;">'
        'Warning: Re-running this cell will reset to defaults.</p>'
    ),
    widgets.HTML("<h3>Model</h3>"),
    _labeled("Synthesizer:", _s3_model),
    widgets.HTML("<h3>Settings</h3>"),
    _labeled("Max plan revisions:", _s3_max_rev_plan,
             "How many times you can revise the final action plan"),
])


def show_stage3():
    display(_s3_form)


def get_config_stage3() -> dict:
    return {
        "input_query":         "",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "auto_approve":        False,
        "stage1_intake": {
            "model":               "openai/gpt-5.2",
            "max_clarify_rounds":  5,
            "max_research_topics": 2,
            "max_topics_revision": 2,
        },
        "stage2_research": {
            "model_researcher":              "openai/gpt-5.2",
            "model_critic":                  "openai/gpt-5.2",
            "max_web_search_ct":             10,
            "max_debate_rounds":             2,
            "max_human_revision_on_proposal": 2,
        },
        "stage3_synthesis": {
            "model":                    _s3_model.value,
            "max_human_revision_on_plan": _s3_max_rev_plan.value,
        },
    }
