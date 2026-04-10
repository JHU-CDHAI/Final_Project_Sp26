# Colab Notebook Activity Ideas

Ideas for descriptive instructions and student activities to embed in the Stage 1/2/3 notebooks.

---
## How to get your OpenRouter API key
WE NEED TO INSTRUCT THEM TO USE THEIR NAME OR SOMETHING AS IDENTIFIER WHEN CREATING THE KEY.

## Stage 1: Problem Intake & Topic Planning

### Core Activity: Model Comparison on Problem Framing

Ask students to run Stage 1 **twice** (or three times) using a different model each run
(GPT-5.2, Claude Opus 4.6, Gemini 3 Flash) with the **same business question and settings**.

**What to compare and write up:**

| Dimension | Where to find it | What to look for |
|---|---|---|
| **Problem framing quality** | `output.md` | Did the model correctly identify the core problem? Were the clarifying questions useful or generic? |
| **Research topics** | `output.md` | Are the topics specific and actionable, or vague? Did different models surface different angles? |
| **Response time** | `meta.txt` / `logs/timings.txt` | Wall-clock time per gate, LLM call time vs. total |
| **Billing cost** | OpenRouter dashboard | Token counts (input/output), estimated cost per run |
| **Interaction feel** | Student's own experience | Did the model ask too many clarifying questions? Too few? Did "press Enter to approve" feel earned? |

**Suggested reflection prompt:**
> Which model produced the most useful problem framing for your question? Were the results comparable? What trade-offs would matter if you were choosing a model for a real consulting engagement?

---

## Stage 2: Research & Debate

### Core Activity: Agent Interaction Behavior Comparison

Students pick their best Stage 1 output (or a provided reference output) and run Stage 2
with **different model pairings** for Researcher vs. Critic. For example:

- Run A: GPT-5.2 (Researcher) + GPT-5.2 (Critic)
- Run B: Claude Opus (Researcher) + Gemini Flash (Critic)
- Run C: Gemini Flash (Researcher) + Claude Opus (Critic)

**What to compare and write up:**

| Dimension | Where to find it | What to look for |
|---|---|---|
| **Debate dynamics** | `logs/topic_*.txt` | Does the Critic push back substantively or just agree? Do debates converge or stall? How many rounds did it take? |
| **Research depth** | `output.md` per-topic findings | Are sources cited? Are confidence levels reasonable? Does the Researcher actually use web search results or hallucinate? |
| **Quality of key recommendations** | `output.md` | Are recommendations specific and evidence-backed, or generic advice? |
| **Cross-model friction** | Student's own experience | Do mixed-model pairings produce more interesting debate than same-model? |

**Suggested reflection prompt:**
> Did using different models for the Researcher and Critic produce a more rigorous debate than using the same model for both? What patterns did you notice in how each model plays the "Critic" role -- does one model challenge assumptions more aggressively? 


---

## Stage 3: Synthesis & Action Plan

### Core Activity: Final Report Quality Comparison

Students run Stage 3 using at least **two different models** on the same Stage 2 handoff.

**What to compare and write up:**

| Dimension | Where to find it | What to look for |
|---|---|---|
| **Report structure & clarity** | `output.md` / `.docx` | Is the executive summary actually useful? Is the 90-day plan concrete? |
| **Evidence integration** | `output.md` | Does the synthesizer connect findings back to the research? Are citations preserved? |
| **Actionability of the plan** | `output.md` — 90-day action plan | Are milestones realistic? Are KPIs measurable? Would a real manager be able to act on this? |
| **Cost & speed** | OpenRouter dashboard and logs | Total tokens and cost for synthesis |

---
