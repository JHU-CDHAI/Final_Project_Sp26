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

def _s2_config_path() -> Path:
    return _project_dir() / "stage2_config.yaml"

def _s3_config_path() -> Path:
    return _project_dir() / "stage3_config.yaml"

def _question_path() -> Path:
    return _project_dir() / "my_question.txt"

def _constraints_path() -> Path:
    return _project_dir() / "my_constraints.txt"

def _topics_path() -> Path:
    return _project_dir() / "my_topics.txt"

def _research_context_path() -> Path:
    return _project_dir() / "my_research_context.txt"

def _research_topics_path() -> Path:
    return _project_dir() / "my_research_topics.txt"

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
    Called during the Step 4 setup cell.
    Creates my_question.txt empty if it does not already exist.
    """
    path = _question_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("")


def create_stage2_files():
    """Create my_research_context.txt and my_research_topics.txt if they don't exist."""
    for p in [_research_context_path(), _research_topics_path()]:
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("")


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
            "# ⚠ Please enter your business question below this line, then re-run the Step 4 check cell.\n\n"
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
    "max_clarify_rounds":  100,
    "max_research_topics": 5,
    "max_topics_revision": 100,
}

_s1_query          = widgets.Textarea(
    placeholder="Type your business question here, or edit my_question.txt directly…",
    layout=widgets.Layout(width="560px", height="100px"),
)
_s1_initial_question: str = ""  # value pre-filled at show_stage1() time; used to detect file edits
_s1_model          = widgets.Dropdown(options=MODEL_OPTIONS, value=_DEFAULTS["model"], layout=_widget_layout)
_s1_max_clarify    = widgets.IntText(value=_DEFAULTS["max_clarify_rounds"],  layout=_slider_layout)
_s1_max_topics     = widgets.IntText(value=_DEFAULTS["max_research_topics"], layout=_slider_layout)
_s1_max_topics_rev = widgets.IntText(value=_DEFAULTS["max_topics_revision"], layout=_slider_layout)


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
    if saved:
        _apply_values(saved)

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

    config_save_status = widgets.Output()

    def _on_save_config(b):
        cfg = {
            "model":               _s1_model.value,
            "max_clarify_rounds":  _s1_max_clarify.value,
            "max_research_topics": _s1_max_topics.value,
            "max_topics_revision": _s1_max_topics_rev.value,
        }
        _save_yaml(_config_path(), cfg)
        ts = datetime.now().strftime("%H:%M:%S")
        with config_save_status:
            config_save_status.clear_output()
            display(HTML(
                _FLASH_CSS +
                "<span class='s1-sf' style='color:#388e3c;font-size:13px'>"
                f"✓ Saved config "
                f"<span style='color:#aaa'>({ts})</span></span>"
            ))

    def _on_reset(b):
        _config_path().unlink(missing_ok=True)
        _apply_values(_DEFAULTS)
        with config_save_status:
            config_save_status.clear_output()
            display(HTML(
                "<div style='border:2px solid #f57c00;padding:8px 12px;border-radius:6px;"
                "background:#fff8e1;margin-bottom:8px'>"
                "↺ Reset to defaults. Saved config deleted.</div>"
            ))

    save_config_btn.on_click(_on_save_config)
    reset_btn.on_click(_on_reset)
    display(widgets.VBox([
        _s1_form,
        widgets.HBox([save_config_btn, reset_btn]),
        config_save_status,
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
        raise RuntimeError("No question found — please save your question in Step 4 first.")

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
            # Mirror the error into the config UI widget (Step 4) so it appears
            # right below the question box — not just in the agent start cell.
            with _s1_save_status:
                _s1_save_status.clear_output()
                display(HTML("""
                    <div style='border:3px solid #e53935;padding:12px;border-radius:8px;
                                background:#fff3f3;margin:4px 0'>
                    <b style='font-size:15px;color:#e53935'>⚠ No question found</b><br><br>
                    Please enter and save your question in Step 4, then re-run Step 5a.
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
# Stage 2 — Research Context & Topics text-file input
# ============================================================================

_s2_context_area = widgets.Textarea(
    placeholder="Paste your research context here (from Stage 1 output)…",
    layout=widgets.Layout(width="560px", height="140px"),
)
_s2_topics_area = widgets.Textarea(
    placeholder="Paste your research topics here (one per line, or numbered)…",
    layout=widgets.Layout(width="560px", height="120px"),
)
_s2_context_save_status = widgets.Output()
_s2_topics_save_status  = widgets.Output()

_s2_context_save_btn = widgets.Button(
    description="Save Research Context",
    button_style="primary",
    icon="save",
    layout=widgets.Layout(width="210px", margin="6px 0 0 0"),
)
_s2_topics_save_btn = widgets.Button(
    description="Save Research Topics",
    button_style="primary",
    icon="save",
    layout=widgets.Layout(width="210px", margin="6px 0 0 0"),
)

_s2_input_form = widgets.VBox([
    widgets.HTML(
        '<p style="color:#000;font-weight:500;font-size:18px;margin:0 0 8px;">'
        'Enter your Stage 1 research output below.</p>'
        '<p style="color:#555;font-size:13px;margin:0 0 4px;">'
        'Your inputs are saved to <code>my_research_context.txt</code> and '
        '<code>my_research_topics.txt</code> — you can edit either the boxes below '
        'or the files directly. Both stay in sync.</p>'
    ),
    widgets.HTML("<h3>Research Context</h3>"),
    _s2_context_area,
    _s2_context_save_btn,
    _s2_context_save_status,
    widgets.HTML("<h3>Research Topics</h3>"),
    _s2_topics_area,
    _s2_topics_save_btn,
    _s2_topics_save_status,
])


def show_stage2_input():
    """Display Stage 2 input form, pre-filling textareas from saved txt files."""
    create_stage2_files()

    cp = _research_context_path()
    if cp.exists():
        raw = cp.read_text(encoding="utf-8")
        lines = [l for l in raw.splitlines() if not l.strip().startswith("#")]
        _s2_context_area.value = "\n".join(lines).strip()

    tp = _research_topics_path()
    if tp.exists():
        raw = tp.read_text(encoding="utf-8")
        lines = [l for l in raw.splitlines() if not l.strip().startswith("#")]
        _s2_topics_area.value = "\n".join(lines).strip()

    def _on_save_context(b):
        text = _s2_context_area.value.strip()
        _research_context_path().write_text(text, encoding="utf-8")
        ts = datetime.now().strftime("%H:%M:%S")
        with _s2_context_save_status:
            _s2_context_save_status.clear_output()
            if text:
                display(HTML(
                    _FLASH_CSS +
                    "<span class='s1-sf' style='color:#388e3c;font-size:13px'>"
                    f"✓ Saved to <code>{_research_context_path().name}</code> "
                    f"<span style='color:#aaa'>({ts})</span></span>"
                ))
            else:
                display(HTML(
                    _FLASH_CSS +
                    "<span class='s1-ef' style='color:#e53935;font-size:13px'>"
                    f"⚠ Research context is empty — please paste something first. "
                    f"<span style='color:#aaa'>({ts})</span></span>"
                ))

    def _on_save_topics(b):
        text = _s2_topics_area.value.strip()
        _research_topics_path().write_text(text, encoding="utf-8")
        ts = datetime.now().strftime("%H:%M:%S")
        with _s2_topics_save_status:
            _s2_topics_save_status.clear_output()
            if text:
                display(HTML(
                    _FLASH_CSS +
                    "<span class='s1-sf' style='color:#388e3c;font-size:13px'>"
                    f"✓ Saved to <code>{_research_topics_path().name}</code> "
                    f"<span style='color:#aaa'>({ts})</span></span>"
                ))
            else:
                display(HTML(
                    _FLASH_CSS +
                    "<span class='s1-ef' style='color:#e53935;font-size:13px'>"
                    f"⚠ Research topics are empty — please paste something first. "
                    f"<span style='color:#aaa'>({ts})</span></span>"
                ))

    _s2_context_save_btn.on_click(_on_save_context)
    _s2_topics_save_btn.on_click(_on_save_topics)
    display(_s2_input_form)


def get_stage1_output() -> dict:
    """Return research context and topics, reading from textareas (with txt-file fallback)."""
    context = _s2_context_area.value.strip()
    if not context:
        cp = _research_context_path()
        if cp.exists():
            context = cp.read_text(encoding="utf-8").strip()

    topics_text = _s2_topics_area.value.strip()
    if not topics_text:
        tp = _research_topics_path()
        if tp.exists():
            topics_text = tp.read_text(encoding="utf-8").strip()

    if not context:
        raise ValueError(
            "No research context found — enter and save it in Step 4 first."
        )
    if not topics_text:
        raise ValueError(
            "No research topics found — enter and save them in Step 4 first."
        )

    return {
        "research_context":     context,
        "research_topics_text": topics_text,
    }


_S2_DEFAULTS = {
    "model_researcher":               "openai/gpt-5.2",
    "model_critic":                   "openai/gpt-5.2",
    "max_web_search_ct":              10,
    "max_debate_rounds":              2,
    "max_human_revision_on_proposal": 2,
}

_s2_model_researcher  = widgets.Dropdown(options=MODEL_OPTIONS, value=_S2_DEFAULTS["model_researcher"], layout=_widget_layout)
_s2_model_critic      = widgets.Dropdown(options=MODEL_OPTIONS, value=_S2_DEFAULTS["model_critic"],     layout=_widget_layout)
_s2_max_web           = widgets.BoundedIntText(value=_S2_DEFAULTS["max_web_search_ct"],              min=1, max=30, layout=_slider_layout)
_s2_max_debate        = widgets.BoundedIntText(value=_S2_DEFAULTS["max_debate_rounds"],              min=1, max=10, layout=_slider_layout)
_s2_max_rev_proposal  = widgets.BoundedIntText(value=_S2_DEFAULTS["max_human_revision_on_proposal"], min=1, max=5,  layout=_slider_layout)


def _apply_s2_values(d: dict):
    _s2_model_researcher.value = d.get("model_researcher",               _S2_DEFAULTS["model_researcher"])
    _s2_model_critic.value     = d.get("model_critic",                   _S2_DEFAULTS["model_critic"])
    _s2_max_web.value          = d.get("max_web_search_ct",              _S2_DEFAULTS["max_web_search_ct"])
    _s2_max_debate.value       = d.get("max_debate_rounds",              _S2_DEFAULTS["max_debate_rounds"])
    _s2_max_rev_proposal.value = d.get("max_human_revision_on_proposal", _S2_DEFAULTS["max_human_revision_on_proposal"])

_s2_form = widgets.VBox([
    widgets.HTML(
        '<p style="color:#000;font-weight:500;font-size:18px;margin:0 0 8px;">'
        'Choose AI models and settings for the research &amp; debate phase.</p>'
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
    saved = _load_yaml(_s2_config_path())
    if saved:
        _apply_s2_values(saved)

    save_btn = widgets.Button(
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
    status = widgets.Output()

    def _on_save(b):
        cfg = {
            "model_researcher":               _s2_model_researcher.value,
            "model_critic":                   _s2_model_critic.value,
            "max_web_search_ct":              _s2_max_web.value,
            "max_debate_rounds":              _s2_max_debate.value,
            "max_human_revision_on_proposal": _s2_max_rev_proposal.value,
        }
        _save_yaml(_s2_config_path(), cfg)
        ts = datetime.now().strftime("%H:%M:%S")
        with status:
            status.clear_output()
            display(HTML(
                _FLASH_CSS +
                "<span class='s1-sf' style='color:#388e3c;font-size:13px'>"
                f"✓ Saved config "
                f"<span style='color:#aaa'>({ts})</span></span>"
            ))

    def _on_reset(b):
        _s2_config_path().unlink(missing_ok=True)
        _apply_s2_values(_S2_DEFAULTS)
        with status:
            status.clear_output()
            display(HTML(
                "<div style='border:2px solid #f57c00;padding:8px 12px;border-radius:6px;"
                "background:#fff8e1;margin-bottom:8px'>"
                "↺ Reset to defaults. Saved config deleted.</div>"
            ))

    save_btn.on_click(_on_save)
    reset_btn.on_click(_on_reset)
    display(widgets.VBox([_s2_form, widgets.HBox([save_btn, reset_btn]), status]))


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

_S3_DEFAULTS = {
    "model":                    "anthropic/claude-opus-4.6",
    "max_human_revision_on_plan": 2,
}

_s3_model        = widgets.Dropdown(options=MODEL_OPTIONS, value=_S3_DEFAULTS["model"],                    layout=_widget_layout)
_s3_max_rev_plan = widgets.BoundedIntText(value=_S3_DEFAULTS["max_human_revision_on_plan"], min=1, max=5, layout=_slider_layout)


def _apply_s3_values(d: dict):
    _s3_model.value        = d.get("model",                    _S3_DEFAULTS["model"])
    _s3_max_rev_plan.value = d.get("max_human_revision_on_plan", _S3_DEFAULTS["max_human_revision_on_plan"])

_s3_form = widgets.VBox([
    widgets.HTML(
        '<p style="color:#000;font-weight:500;font-size:18px;margin:0 0 8px;">'
        'Choose the synthesizer model and settings.</p>'
    ),
    widgets.HTML("<h3>Model</h3>"),
    _labeled("Synthesizer:", _s3_model),
    widgets.HTML("<h3>Settings</h3>"),
    _labeled("Max plan revisions:", _s3_max_rev_plan,
             "How many times you can revise the final action plan"),
])


def show_stage3():
    saved = _load_yaml(_s3_config_path())
    if saved:
        _apply_s3_values(saved)

    save_btn = widgets.Button(
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
    status = widgets.Output()

    def _on_save(b):
        cfg = {
            "model":                    _s3_model.value,
            "max_human_revision_on_plan": _s3_max_rev_plan.value,
        }
        _save_yaml(_s3_config_path(), cfg)
        ts = datetime.now().strftime("%H:%M:%S")
        with status:
            status.clear_output()
            display(HTML(
                _FLASH_CSS +
                "<span class='s1-sf' style='color:#388e3c;font-size:13px'>"
                f"✓ Saved config "
                f"<span style='color:#aaa'>({ts})</span></span>"
            ))

    def _on_reset(b):
        _s3_config_path().unlink(missing_ok=True)
        _apply_s3_values(_S3_DEFAULTS)
        with status:
            status.clear_output()
            display(HTML(
                "<div style='border:2px solid #f57c00;padding:8px 12px;border-radius:6px;"
                "background:#fff8e1;margin-bottom:8px'>"
                "↺ Reset to defaults. Saved config deleted.</div>"
            ))

    save_btn.on_click(_on_save)
    reset_btn.on_click(_on_reset)
    display(widgets.VBox([_s3_form, widgets.HBox([save_btn, reset_btn]), status]))


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
