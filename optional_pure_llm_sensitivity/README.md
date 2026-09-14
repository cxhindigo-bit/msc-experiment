# Optional Pure-LLM sensitivity experiment

This directory contains the optional A–B–D comparison. It does not change the
main pipeline or overwrite its outputs.

| Condition | Model | Prompt | Status |
|---|---|---|---|
| A: `original-mini` | `gpt-4.1-mini` | original | reads the completed main result |
| B: `improved-mini` | `gpt-4.1-mini` | improved evidence instructions | new run |
| D: `improved-full` | `gpt-4.1` | improved evidence instructions | new run |

The improved prompt tells the model to use first answers as the main evidence
and not to treat a correction after a teacher hint as an independent correct
answer. It does not include simulator probabilities or gold answers.

Run a small check first:

Set the project `.env` to use the OpenAI endpoint and your own API key:

```text
LLM_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-key
```

Then run a small check:

```bash
python -m optional_pure_llm_sensitivity.run --condition improved-mini --smoke 2
```

Run both complete new conditions:

```bash
python -m optional_pure_llm_sensitivity.run --all
python -m optional_pure_llm_sensitivity.evaluate
```

New results are written only under:

```text
outputs/optional_pure_llm_sensitivity/
```

If the analysis is not retained in the dissertation, this directory and its
output directory can be removed without changing the completed main experiment.
