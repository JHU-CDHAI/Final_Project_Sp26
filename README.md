## Quick Start

### 1. Set up the environment

```bash
cd Final_Project_Sp26

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your API keys

```bash
cp agent/.env.example agent/.env
```

Edit `agent/.env` and fill in your keys:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
TAVILY_API_KEY=tvly-your-key-here
```

- **OpenRouter** — get a key at [openrouter.ai](https://openrouter.ai)
- **Tavily** — sign up for free at [tavily.com](https://tavily.com) (1,000 credits/month on the free plan)

### 3. (Optional) Edit the configuration

Open `agent/config.yaml` to change the business question, swap models, or tune agent behavior:

```yaml
input_query: "How should I start a beverage business in Sweden?"

agents:
  intake:       { model: "openai/gpt-4o" }
  researcher:   { model: "openai/gpt-4o" }
  critic:       { model: "openai/gpt-4o" }
  synthesizer:  { model: "openai/gpt-4o" }
```

Available models:

| Provider | Models |
|----------|--------|
| OpenAI | `openai/gpt-4o`, `openai/gpt-5.2`, `openai/o3-mini` |
| Google | `google/gemini-3-flash-preview` |
| Anthropic | `anthropic/claude-sonnet-4.5`, `anthropic/claude-opus-4.6` |

### 4. Run the agent

```bash
cd agent
../.venv/bin/python mba_agent_v4.py
```

The agent will walk through its pipeline and pause at three human gates. At each `>` prompt:
- Press **Enter** to approve and continue
- Type feedback to ask the agent to revise

### 5. View results

Reports are saved to `agent/results/<timestamp>/`:

| File | Contents |
|------|----------|
| `report_<ts>.md` | Strategy report in Markdown |
| `report_<ts>.docx` | Strategy report in Word |
| `meta.yaml` | Run metadata (models, elapsed time) |
| `log.txt` | Full console log |

## Google Colab

This project can also run as an interactive notebook on Google Colab. See `agent/Final_Project_Multi_Agent.ipynb` for the guided walkthrough with a widget-based config UI and Google Drive export.

## License

MIT
