# MSc experiment

This repository contains the synthetic fraction-tutoring data, the two main
prediction pipelines, the gold-label diagnostic condition, and the evaluation
outputs reported in the dissertation.

## Experiment structure

| Dissertation component | Code | Purpose |
|---|---|---|
| Pipeline 1: Pure-LLM | `experiment/pure_llm/` | Reads the visible dialogue history and directly predicts eight probabilities on Days 10 and 20. |
| Pipeline 2: LLM-SimpleKT | `experiment/llm_simplekt/` | Classifies each initial answer, builds time-ordered records, and uses SimpleKT to predict the same probabilities. |
| gold-SimpleKT diagnostic | `experiment/llm_simplekt/` with gold labels | Repeats the SimpleKT condition with simulator labels to measure the error added by LLM classification. |
| Common evaluation | `experiment/evaluation/` | Checks coverage and calculates learner-level MAE, RMSE, and paired confidence intervals. |

The two main pipelines use the same 48 test learners, the same dialogue
histories, the same Day 10 and Day 20 checkpoints, and the same eight fraction
concepts. The gold-SimpleKT run is diagnostic and is not a third main pipeline.

## Main directories

- `data_generation/` creates the structured gold data and dialogue dataset.
- `data_generation/data/` contains the fixed dataset used by the formal run.
- `experiment/` contains both main pipelines and the common evaluator.
- `outputs/` contains the formal results reported in the dissertation.
- `optional_pure_llm_sensitivity/` contains the supplementary A-B-D Pure-LLM experiment.

Data generation is separate from the main pipeline. Running `run.py` never
regenerates or changes the files in `data_generation/data/`.

## Experimental-design materials

The following files support inspection of the experimental design:

- `data_generation/data/questions.csv` fixes the 24 questions and their
  question–concept mapping.
- `data_generation/data/learner_profiles.csv` stores the simulated learner
  parameters used to generate each trajectory. These hidden parameters are not
  given to either prediction pipeline.
- `data_generation/data/splits.csv` fixes the learner-level 144/48/48 training,
  validation, and test split.
- `data_generation/data/gold_interactions.csv` stores task order and simulator
  initial-answer labels fixed before dialogue wording is generated.
- `data_generation/data/gold_mastery.csv` stores F1–F8 reference probabilities
  on Days 10 and 20.
- `data_generation/data/manifest.json` records data scale and the model and
  settings used to generate `raw_dialogues.jsonl`.
- `experiment/settings.py` fixes shared checkpoints, concepts, test questions,
  random seeds, bootstrap samples, LLM settings, and SimpleKT hyperparameters.
- `experiment/evaluation/evaluate.py` defines the expected
  learner–day–concept key set, requires complete predictions before error
  calculation, aggregates within learners, and performs learner-level paired
  comparison.

Together, these materials record the conditions fixed before evaluation and
the field that changes in each diagnostic comparison. Pure-LLM and
LLM-SimpleKT use the same 48 test learners, visible histories, checkpoints,
fixed test questions, reference probabilities, and evaluator. gold-SimpleKT
keeps the SimpleKT model, learner split, hyperparameters, and prediction targets
unchanged and replaces only the LLM initial-answer labels with simulator labels.

## Setup

Create a new environment on each machine and install the pinned project
dependencies. Do not copy `.venv` between computers.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add an OpenAI API key to `.env`. The reported dialogue generation uses
`gpt-4.1-nano`; initial-answer classification and formal Pure-LLM prediction use
`gpt-4.1-mini`. These settings are separate so changing one stage does not
silently change the other.

## Run the formal pipeline

The dataset is already present. Run stages separately when resuming work:

```bash
python run.py extract
python run.py llm-simplekt
python run.py gold-simplekt
python run.py pure-llm
python run.py evaluate
```

`python run.py all` runs the same stages in order. The LLM stages may use cached
responses when matching cache entries exist.

## Formal outputs

The principal prediction files are:

- `outputs/pure_llm_predictions.csv`
- `outputs/llm_simplekt_predictions.csv`
- `outputs/gold_simplekt_predictions.csv`
- `outputs/classification_metrics.csv`
- `outputs/prediction_coverage.csv`
- `outputs/state_metrics.csv`
- `outputs/paired_state_metrics.csv`

Supplementary A-B-D results are stored only under
`outputs/optional_pure_llm_sensitivity/`.

The repository records the experiment configuration and the outputs reported
in the dissertation so that they can be inspected. Trained SimpleKT weights
and per-epoch training histories are not included, so rerunning training is not
claimed to reproduce an identical model checkpoint.
