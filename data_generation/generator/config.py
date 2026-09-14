from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"

# Design item: Master random seed
# Current setting: 20260720.
SEED = 20260720

# Design item: Dataset scale
# Current setting: 240 simulated learners.
N_LEARNERS = 240
# Design item: Session schedule
# Current setting: Days 1, 3, 6, 8, 11, 13, 16, and 18; five tasks per session and 40 tasks per learner.
SESSION_DAYS = [1, 3, 6, 8, 11, 13, 16, 18]
TASKS_PER_SESSION = 5

# Design item: Dialogue-generation request
# Current setting: Temperature 0.7, maximum 1,600 output tokens, and at most three attempts per session.
DIALOGUE_TEMPERATURE = 0.7
DIALOGUE_MAX_TOKENS = 1600
DIALOGUE_THINKING = "not_applicable"

# Design item: Initial-answer source
# Current setting: The program writes the preset gold answer into the second utterance.
DIALOGUE_INITIAL_ANSWER_MODE = "deterministic_injection"

MAX_DIALOGUE_ATTEMPTS = 3

# Design item: Checkpoint dates
# Current setting: Days 10 and 20, each two days after the latest session.
CHECKPOINTS = [10, 20]

# Design item: Concept set
# Current setting: F1–F8, giving eight probabilities per checkpoint and 16 per learner.
CONCEPTS = [f"F{i}" for i in range(1, 9)]

# Design item: Learning rate
# Current setting: Sample once per learner from U(0.18, 0.30), then keep it fixed.
LEARNING_RATE_RANGE = (0.18, 0.30)
# Design item: Forgetting rate
# Current setting: Sample once per learner from U(0.008, 0.022), then keep it fixed.
FORGETTING_RATE_RANGE = (0.008, 0.022)

# Design item: Initial concept probability
# Current setting: Sample each learner–concept value from U(0.10, 0.35).
INITIAL_MASTERY = (0.10, 0.35)

# Design item: Learning after an incorrect initial answer
# Current setting: Use 20% of the learning gain applied after a correct answer.
INCORRECT_LEARNING_FACTOR = 0.20

# Design item: Probability bounds
# Current setting: Clip simulator probabilities to 0.05–0.95.
MASTERY_MIN = 0.05
MASTERY_MAX = 0.95
