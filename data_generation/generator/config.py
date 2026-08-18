from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"

# 固定随机种子，使数据可以重复生成
# Fixed random seed for reproducible data generation.
SEED = 20260720

# 实验规模：240 名学习者，每人八次会话，每次五项任务
# Experiment scale: 240 learners, eight sessions each, five tasks per session.
N_LEARNERS = 240
SESSION_DAYS = [1, 3, 6, 8, 11, 13, 16, 18]
TASKS_PER_SESSION = 5

# LM Studio 的官方 Chat Completions 和结构化输出示例均使用 0.7。本项目采用
# 该值生成具有适度措辞变化的对话，但不把它解释为经过验证的最佳值。
# LM Studio's official Chat Completions and structured-output examples use 0.7.
# This project adopts it for moderate wording variation, not as a validated optimum.
# https://github.com/lmstudio-ai/docs/blob/b02d17517b73c51f520cd5129855cdf30e0728f7/1_developer/3_openai-compat/structured-output.md
DIALOGUE_TEMPERATURE = 0.7

# 两个预测日期分别使用前 20 项和全部 40 项任务历史
# The two prediction days use the first 20 and all 40 tasks respectively.
CHECKPOINTS = [10, 20]

# F1-F8 是实验中分别预测的八个分数概念
# F1-F8 are the eight fraction concepts predicted separately in the experiment.
CONCEPTS = [f"F{i}" for i in range(1, 9)]

# 每名学习者只抽取一次学习率和遗忘率，并在全部任务中保持不变
# Each learner draws these rates once and keeps them for all tasks.
LEARNING_RATE_RANGE = (0.18, 0.30)
FORGETTING_RATE_RANGE = (0.008, 0.022)

# 每个概念在第一次会话前的初始正确作答概率范围
# Initial correct-answer probability range before the first session.
INITIAL_MASTERY = (0.10, 0.35)

# 学习者首次答错后仍可能通过接触题目和教师反馈学到一点；0.20 表示此时
# 只获得“答对学习增量”的 20%，不是把正确作答概率直接增加 0.20。
# 例如 m=0.30、学习率=0.24 时，答对增加 0.168，答错只增加 0.0336。
# A learner may still learn from the task and teacher feedback after an incorrect
# initial answer. The gain is 20% of the correct-response gain, not a direct 0.20 increase.
INCORRECT_LEARNING_FACTOR = 0.20

# 避免生成绝对必错或绝对必对的状态
# Prevent generated states from becoming certainly wrong or certainly correct.
MASTERY_MIN = 0.05
MASTERY_MAX = 0.95
