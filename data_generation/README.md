# Data generation

This module creates the structured gold data and the natural-language tutoring
dialogues used by the experiment. Run commands from the repository root.

## Setup

Activate the project environment and create a local `.env` file with an OpenAI
API key. The model and endpoint can be overridden when needed.

```bash
source .venv/bin/activate
```

```dotenv
OPENAI_API_KEY=...
DIALOGUE_LLM_MODEL=gpt-4.1-nano
LLM_BASE_URL=https://api.openai.com/v1
```

## Commands

```bash
python -m data_generation.generate plan
python -m data_generation.generate gold
python -m data_generation.generate dialogues
```

- `plan` prints the expected dataset scale and does not write files or call an API.
- `gold` creates the LLM-independent experimental baseline. It writes the gold
  CSV files and resets `manifest.json`; use it only when intentionally creating
  a new data version.
- `dialogues` reads the existing gold data and renders one five-task tutoring
  session per LLM call. It does not modify the gold CSV files.

## Outputs

`data/` contains the versioned experimental data:

- `questions.csv`, `learner_profiles.csv`, and `splits.csv`
- `gold_interactions.csv` and `gold_mastery.csv`
- `raw_dialogues.jsonl`
- `manifest.json`

`cache/` stores LLM responses so an interrupted dialogue run can be resumed.
Keep the cache while resuming a run.
Before marking a dialogue run complete, the generator verifies task count,
turn order, question fidelity, gold initial answers, correction closure, and
teacher confirmation.
