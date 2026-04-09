"""Per-stage configuration widgets for the 3-stage MBA Strategy Agent notebooks."""

import ipywidgets as widgets
from IPython.display import display

MODEL_OPTIONS = [
    "openai/gpt-5.2",
    "anthropic/claude-opus-4.6",
    "google/gemini-3-flash-preview",
]

_label_layout = widgets.Layout(width="220px")
_widget_layout = widgets.Layout(width="320px")
_slider_layout = widgets.Layout(width="260px")


def _labeled(label_text, w, hint=None):
    row = widgets.HBox([widgets.Label(label_text, layout=_label_layout), w])
    if hint:
        hint_w = widgets.HTML(
            f'<p style="color:#888;font-size:12px;margin:0 0 4px 220px;">{hint}</p>'
        )
        return widgets.VBox([row, hint_w])
    return row


# ============================================================================
# STAGE 1 — Problem Intake & Topic Planning
# ============================================================================

_s1_query = widgets.Textarea(
    value="Please type your business question here...",
    placeholder="Please type your business question here...",
    layout=widgets.Layout(width="550px", height="80px"),
)
_s1_model = widgets.Dropdown(options=MODEL_OPTIONS, value="openai/gpt-5.2", layout=_widget_layout)
_s1_max_clarify = widgets.BoundedIntText(value=5, min=1, max=10, layout=_slider_layout)
_s1_max_topics = widgets.BoundedIntText(value=2, min=1, max=5, layout=_slider_layout)
_s1_max_topics_rev = widgets.BoundedIntText(value=2, min=1, max=5, layout=_slider_layout)

_s1_form = widgets.VBox([
    widgets.HTML(
        '<p style="color:#000;font-weight:500;font-size:18px;margin:0 0 8px;">'
        'Enter your business question below, then select an AI model and configure '
        'the intake settings.</p>'
        '<p style="color:#d32f2f;font-weight:600;font-size:18px;margin:0 0 8px;">'
        'Warning: Once your settings are finalized, proceed directly to the Run step. '
        "Re-running this cell will reset everything to the default configuration.</p>"
    ),
    widgets.HTML("<h3>Business Question</h3>"),
    _s1_query,
    widgets.HTML("<h3>Model</h3>"),
    _labeled("Intake model:", _s1_model),
    widgets.HTML("<h3>Settings</h3>"),
    _labeled("Max clarify rounds:", _s1_max_clarify,
             "How many times the agent can ask you clarifying questions (Gate 1)"),
    _labeled("Max research topics:", _s1_max_topics,
             "Number of topics the agent will come up with to research on"),
    _labeled("Max topic revisions:", _s1_max_topics_rev,
             "How many times you can revise the research topic list (Gate 2)"),
    widgets.HTML(
        '<p style="color:#d32f2f;font-weight:600;font-size:14px;margin:12px 0 0;">'
        'Warning: Once your settings are finalized, proceed directly to the Run step. '
        'Re-running this cell will reset everything to the default configuration.</p>'
    ),
])


def show_stage1():
    display(_s1_form)


def get_config_stage1() -> dict:
    return {
        "input_query": _s1_query.value,
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "auto_approve": False,
        "stage1_intake": {
            "model": _s1_model.value,
            "max_clarify_rounds": _s1_max_clarify.value,
            "max_research_topics": _s1_max_topics.value,
            "max_topics_revision": _s1_max_topics_rev.value,
        },
        # Placeholders so config.yaml stays complete
        "stage2_research": {
            "model_researcher": "openai/gpt-5.2",
            "model_critic": "openai/gpt-5.2",
            "max_web_search_ct": 10,
            "max_debate_rounds": 2,
            "max_human_revision_on_proposal": 2,
        },
        "stage3_synthesis": {
            "model": "anthropic/claude-opus-4.6",
            "max_human_revision_on_plan": 2,
        },
    }


# ============================================================================
# STAGE 2 — Research & Debate
# ============================================================================

# -- Stage 1 output input (shown in step 2 of the Stage 2 notebook) --
_s2_context = widgets.Textarea(
    value="",
    placeholder="Paste your problem framing and constraints here...",
    layout=widgets.Layout(width="95%", height="160px"),
)
_s2_topics = widgets.Textarea(
    value="",
    placeholder="1. First research topic\n2. Second research topic",
    layout=widgets.Layout(width="95%", height="100px"),
)

_s2_input_form = widgets.VBox([
    widgets.HTML(
        '<b style="font-size:14px;">Research Context</b>'
        '<p style="color:#555;margin:2px 0 6px;">Copy the <b>Problem Framing</b> '
        'and <b>Constraints</b> sections from your Stage 1 output and paste them '
        'into the box below. This gives the research agents background on your '
        'business question.</p>'
    ),
    _s2_context,
    widgets.HTML(
        '<b style="font-size:14px;margin-top:12px;">Research Topics</b>'
        '<p style="color:#555;margin:2px 0 6px;">Copy the <b>Research Topics</b> '
        'list from your Stage 1 output and paste it below. '
        '<b>Put exactly one topic per line.</b>'
        '<br><i>Example:</i></p>'
        '<pre style="color:#888;margin:2px 0 6px;font-size:12px;">'
        '1. Market entry strategies for beverage industry in Sweden\n'
        '2. Consumer preferences and trends in the Swedish beverage market</pre>'
    ),
    _s2_topics,
    widgets.HTML(
        '<p style="color:#d32f2f;font-weight:600;font-size:14px;margin:12px 0 0;">'
        'Warning: Once you have pasted your Stage 1 output, proceed to the next steps. '
        'Re-running this cell will clear both boxes.</p>'
    ),
])


def show_stage2_input():
    display(_s2_input_form)


def get_stage1_output() -> dict:
    return {
        "research_context": _s2_context.value,
        "research_topics_text": _s2_topics.value,
    }


_s2_model_researcher = widgets.Dropdown(options=MODEL_OPTIONS, value="openai/gpt-5.2", layout=_widget_layout)
_s2_model_critic = widgets.Dropdown(options=MODEL_OPTIONS, value="openai/gpt-5.2", layout=_widget_layout)
_s2_max_web = widgets.BoundedIntText(value=10, min=1, max=30, layout=_slider_layout)
_s2_max_debate = widgets.BoundedIntText(value=2, min=1, max=10, layout=_slider_layout)
_s2_max_rev_proposal = widgets.BoundedIntText(value=2, min=1, max=5, layout=_slider_layout)

_s2_form = widgets.VBox([
    widgets.HTML(
        '<p style="color:#000;font-weight:500;font-size:18px;margin:0 0 8px;">'
        'Choose AI models and settings for the research &amp; debate phase.</p>'
        '<p style="color:#d32f2f;font-weight:600;font-size:18px;margin:0 0 8px;">'
        'Warning: Once your settings are finalized, proceed directly to the Run step. '
        "Re-running this cell will reset everything to the default configuration.</p>"
    ),
    widgets.HTML("<h3>Models</h3>"),
    _labeled("Researcher:", _s2_model_researcher),
    _labeled("Critic:", _s2_model_critic),
    widgets.HTML("<h3>Settings</h3>"),
    _labeled("Max web search results:", _s2_max_web,
             "Tavily search results per query per debate round"),
    _labeled("Max debate rounds:", _s2_max_debate,
             "Researcher vs Critic rounds per topic before human review"),
    _labeled("Max proposal revisions:", _s2_max_rev_proposal,
             "How many times you can send a topic back for revision during research &amp; debate, for each topic (Gate 3)"),
    widgets.HTML(
        '<p style="color:#d32f2f;font-weight:600;font-size:14px;margin:12px 0 0;">'
        'Warning: Once your settings are finalized, proceed directly to the Run step. '
        'Re-running this cell will reset everything to the default configuration.</p>'
    ),
])


def show_stage2():
    display(_s2_form)


def get_config_stage2() -> dict:
    return {
        "input_query": "",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "auto_approve": False,
        "stage1_intake": {
            "model": "openai/gpt-5.2",
            "max_clarify_rounds": 5,
            "max_research_topics": 2,
            "max_topics_revision": 2,
        },
        "stage2_research": {
            "model_researcher": _s2_model_researcher.value,
            "model_critic": _s2_model_critic.value,
            "max_web_search_ct": _s2_max_web.value,
            "max_debate_rounds": _s2_max_debate.value,
            "max_human_revision_on_proposal": _s2_max_rev_proposal.value,
        },
        "stage3_synthesis": {
            "model": "anthropic/claude-opus-4.6",
            "max_human_revision_on_plan": 2,
        },
    }


# ============================================================================
# STAGE 3 — Synthesis & Action Plan
# ============================================================================

_s3_model = widgets.Dropdown(options=MODEL_OPTIONS, value="anthropic/claude-opus-4.6", layout=_widget_layout)
_s3_max_rev_plan = widgets.BoundedIntText(value=2, min=1, max=5, layout=_slider_layout)

_s3_form = widgets.VBox([
    widgets.HTML(
        '<p style="color:#000;font-weight:500;font-size:18px;margin:0 0 8px;">'
        'Choose the synthesizer model and settings.</p>'
        '<p style="color:#d32f2f;font-weight:600;font-size:18px;margin:0 0 8px;">'
        'Warning: Once your settings are finalized, proceed directly to the Run step. '
        "Re-running this cell will reset everything to the default configuration.</p>"
    ),
    widgets.HTML("<h3>Model</h3>"),
    _labeled("Synthesizer:", _s3_model),
    widgets.HTML("<h3>Settings</h3>"),
    _labeled("Max plan revisions:", _s3_max_rev_plan,
             "How many times you can revise the final action plan (Gate 4)"),
    widgets.HTML(
        '<p style="color:#d32f2f;font-weight:600;font-size:14px;margin:12px 0 0;">'
        'Warning: Once your settings are finalized, proceed directly to the Run step. '
        'Re-running this cell will reset everything to the default configuration.</p>'
    ),
])


def show_stage3():
    display(_s3_form)


def get_config_stage3() -> dict:
    return {
        "input_query": "",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "auto_approve": False,
        "stage1_intake": {
            "model": "openai/gpt-5.2",
            "max_clarify_rounds": 5,
            "max_research_topics": 2,
            "max_topics_revision": 2,
        },
        "stage2_research": {
            "model_researcher": "openai/gpt-5.2",
            "model_critic": "openai/gpt-5.2",
            "max_web_search_ct": 10,
            "max_debate_rounds": 2,
            "max_human_revision_on_proposal": 2,
        },
        "stage3_synthesis": {
            "model": _s3_model.value,
            "max_human_revision_on_plan": _s3_max_rev_plan.value,
        },
    }
