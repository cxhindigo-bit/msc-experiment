from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"

# Fixed seed for reproducible data generation.
SEED = 20260720

# 240 learners, eight sessions each, five tasks per session.
N_LEARNERS = 240
SESSION_DAYS = [1, 3, 6, 8, 11, 13, 16, 18]
TASKS_PER_SESSION = 5

# OpenAI dialogue-generation settings.
DIALOGUE_TEMPERATURE = 0.7

# Leave room for a full session while limiting excessive output.
DIALOGUE_MAX_TOKENS = 1600

# OpenAI Chat Completions generation does not use a provider-specific thinking setting.
DIALOGUE_THINKING = "not_applicable"

# The program writes the gold initial answer; the LLM generates the remaining dialogue.
DIALOGUE_INITIAL_ANSWER_MODE = "deterministic_injection"

# Maximum dialogue-generation attempts per session.
MAX_DIALOGUE_ATTEMPTS = 3

# Prediction days use the first 20 and all 40 tasks, respectively.
CHECKPOINTS = [10, 20]

# The eight fraction concepts predicted separately in the experiment.
CONCEPTS = [f"F{i}" for i in range(1, 9)]

# Each learner keeps the same rates across all tasks.
LEARNING_RATE_RANGE = (0.18, 0.30)
FORGETTING_RATE_RANGE = (0.008, 0.022)

# Initial correct-answer probability range.
INITIAL_MASTERY = (0.10, 0.35)

# Incorrect responses receive 20% of the correct-response learning gain.
INCORRECT_LEARNING_FACTOR = 0.20

# Keep generated probabilities away from 0 and 1.
MASTERY_MIN = 0.05
MASTERY_MAX = 0.95
