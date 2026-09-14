"""Shared paths and fixed experiment settings."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data_generation" / "data"
OUTPUT_DIR = PROJECT_ROOT / os.getenv("PIPELINE_OUTPUT_DIR", "outputs")
CACHE_DIR = OUTPUT_DIR / "cache"

# Design item: Prediction concepts
# Current setting: F1–F8; eight probabilities per checkpoint and 16 per learner.
CONCEPTS = [f"F{i}" for i in range(1, 9)]
CONCEPT_NAMES = {
    "F1": "fraction meaning and representation",
    "F2": "equivalent fractions",
    "F3": "comparing fractions with the same denominator",
    "F4": "comparing fractions with different denominators",
    "F5": "finding and using common denominators",
    "F6": "adding fractions with the same denominator",
    "F7": "adding fractions with different denominators",
    "F8": "simplifying fractions",
}
# Design item: Fixed test question
# Current setting: Use the first question in the bank for each concept.
PROBE_QUESTIONS = {
    "F1": "Q0001", "F2": "Q0004", "F3": "Q0007", "F4": "Q0010",
    "F5": "Q0013", "F6": "Q0016", "F7": "Q0019", "F8": "Q0022",
}
# Design item: History boundaries
# Current setting: Day 10 uses 20 tasks; Day 20 uses all 40 tasks.
CHECKPOINTS = [10, 20]

# Design item: Experiment random seed
# Current setting: 20260720.
SEED = 20260720
# Design item: Bootstrap confidence interval
# Current setting: 2,000 learner-level resamples; use the 2.5th and 97.5th percentiles.
BOOTSTRAP_SAMPLES = 2000

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# Design item: Shared formal LLM conditions
# Current setting: gpt-4.1-mini, temperature 0.0, and at most three attempts per session or checkpoint.
LLM_MODEL = os.getenv("PIPELINE_LLM_MODEL", "gpt-4.1-mini").strip()
LLM_THINKING = os.getenv("PIPELINE_LLM_THINKING", "not_applicable").strip()
LLM_TEMPERATURE = 0.0
MAX_LLM_ATTEMPTS = 3

# Design item: SimpleKT hyperparameters
# Current setting: Size 64, learning rate 0.001, batch 32, dropout 0.1, two blocks, four heads, 30 epochs, patience five.
SIMPLEKT = {
    "embedding_size": 64,
    "learning_rate": 0.001,
    "batch_size": 32,
    "dropout": 0.1,
    "max_epochs": 30,
    "patience": 5,
    "n_blocks": 2,
    "attention_heads": 4,
}
