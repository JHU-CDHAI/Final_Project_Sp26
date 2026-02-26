"""Interactive configuration widgets for the MBA Strategy Agent notebook."""

import ipywidgets as widgets
from IPython.display import display

MODEL_OPTIONS = [
    "openai/gpt-4o",
    "openai/gpt-5.2",
    "openai/o3-mini",
    "google/gemini-3-flash-preview",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.6",
]

# ── Styling ──
_label_layout = widgets.Layout(width="220px")
_widget_layout = widgets.Layout(width="320px")
_slider_layout = widgets.Layout(width="260px")


def _labeled(label_text, w):
    return widgets.HBox([widgets.Label(label_text, layout=_label_layout), w])


# ── Widgets ──
query_input = widgets.Textarea(
    value="How should I start a beverage business in Sweden?",
    placeholder="Enter your business question here...",
    layout=widgets.Layout(width="550px", height="80px"),
)

model_intake = widgets.Dropdown(options=MODEL_OPTIONS, value="openai/gpt-4o", layout=_widget_layout)
model_researcher = widgets.Dropdown(options=MODEL_OPTIONS, value="openai/gpt-4o", layout=_widget_layout)
model_critic = widgets.Dropdown(options=MODEL_OPTIONS, value="openai/gpt-4o", layout=_widget_layout)
model_synthesizer = widgets.Dropdown(options=MODEL_OPTIONS, value="anthropic/claude-opus-4.6", layout=_widget_layout)

max_clarify = widgets.BoundedIntText(value=5, min=1, max=10, layout=_slider_layout)
max_topics = widgets.BoundedIntText(value=2, min=1, max=5, layout=_slider_layout)
max_web = widgets.BoundedIntText(value=10, min=1, max=30, layout=_slider_layout)
max_debate = widgets.BoundedIntText(value=2, min=1, max=10, layout=_slider_layout)
max_rev_proposal = widgets.BoundedIntText(value=2, min=1, max=5, layout=_slider_layout)
max_rev_plan = widgets.BoundedIntText(value=2, min=1, max=5, layout=_slider_layout)

form = widgets.VBox([
    widgets.HTML(
        '<p style="color:#d32f2f;font-weight:600;font-size:24px;margin:0 0 8px;">'
        '⚠ Do not re-run this cell after changing values.</p>'
        '<p style="color:#000;font-weight:600;font-size:18px;margin:0 0 8px;">'
        'Adjust values in the widget.</p>'
    ),
    widgets.HTML("<h3>Business Question</h3>"),
    query_input,

    widgets.HTML("<h3>Agent Models</h3>"),
    _labeled("Intake:", model_intake),
    _labeled("Researcher:", model_researcher),
    _labeled("Critic:", model_critic),
    _labeled("Synthesizer:", model_synthesizer),

    widgets.HTML("<h3>Settings</h3>"),
    _labeled("Max clarify rounds:", max_clarify),
    _labeled("Max research topics:", max_topics),
    _labeled("Max web search results:", max_web),
    _labeled("Max debate rounds:", max_debate),
    _labeled("Max proposal revisions:", max_rev_proposal),
    _labeled("Max plan revisions:", max_rev_plan),
])


def show():
    """Display the config form."""
    display(form)


def get_config() -> dict:
    """Read current widget values into a CONFIG dict."""
    return {
        "input_query": query_input.value,
        "agents": {
            "intake":      {"model": model_intake.value},
            "researcher":  {"model": model_researcher.value},
            "critic":      {"model": model_critic.value},
            "synthesizer": {"model": model_synthesizer.value},
        },
        "max_clarify_rounds": max_clarify.value,
        "max_research_topics": max_topics.value,
        "max_web_search_ct": max_web.value,
        "max_debate_rounds": max_debate.value,
        "max_human_revision_on_proposal": max_rev_proposal.value,
        "max_human_revision_on_plan": max_rev_plan.value,
    }
