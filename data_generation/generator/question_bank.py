"""The 24 fraction questions used by the mastery experiment."""


QUESTIONS = [
    {"id": "Q0001", "concept": "F1", "text": "What does the fraction 3/8 mean?", "correct_answer": "three out of eight equal parts", "incorrect_answer": "eight out of three parts", "feedback_hint": "Ask which number gives all equal parts and which gives the selected parts."},
    {"id": "Q0002", "concept": "F1", "text": "A bar has 6 equal parts and 2 are shaded. What fraction is shaded?", "correct_answer": "2/6", "incorrect_answer": "6/2", "feedback_hint": "Ask which number counts all parts and which counts the shaded parts."},
    {"id": "Q0003", "concept": "F1", "text": "In 4/7, what does the denominator tell us?", "correct_answer": "seven equal parts altogether", "incorrect_answer": "seven selected pieces", "feedback_hint": "Ask whether the denominator counts selected pieces or all equal pieces."},
    {"id": "Q0004", "concept": "F2", "text": "Give a fraction equivalent to 1/2.", "correct_answer": "2/4", "incorrect_answer": "2/2", "feedback_hint": "Ask what must happen to both numerator and denominator."},
    {"id": "Q0005", "concept": "F2", "text": "Are 2/3 and 4/6 equivalent?", "correct_answer": "yes", "incorrect_answer": "no", "feedback_hint": "Ask whether both parts of 2/3 can be multiplied by the same number."},
    {"id": "Q0006", "concept": "F2", "text": "Complete: 3/5 = ?/10.", "correct_answer": "6", "incorrect_answer": "8", "feedback_hint": "Ask how 5 changed to 10 and apply the same change to 3."},
    {"id": "Q0007", "concept": "F3", "text": "Which is larger, 3/8 or 5/8?", "correct_answer": "5/8", "incorrect_answer": "3/8", "feedback_hint": "Point out that the denominators match and ask which numerator is larger."},
    {"id": "Q0008", "concept": "F3", "text": "Put 1/7, 4/7, and 6/7 in ascending order.", "correct_answer": "1/7, 4/7, 6/7", "incorrect_answer": "6/7, 4/7, 1/7", "feedback_hint": "Ask the student to compare the numerators because the denominators match."},
    {"id": "Q0009", "concept": "F3", "text": "Compare 2/9 and 7/9 using <, >, or =.", "correct_answer": "2/9 < 7/9", "incorrect_answer": "2/9 > 7/9", "feedback_hint": "Ask which numerator is smaller when the denominators match."},
    {"id": "Q0010", "concept": "F4", "text": "Which is larger, 2/3 or 3/5?", "correct_answer": "2/3", "incorrect_answer": "3/5", "feedback_hint": "Ask the student to rewrite both fractions with denominator 15."},
    {"id": "Q0011", "concept": "F4", "text": "Compare 3/4 and 5/8.", "correct_answer": "3/4", "incorrect_answer": "5/8", "feedback_hint": "Ask the student to rewrite 3/4 in eighths."},
    {"id": "Q0012", "concept": "F4", "text": "Put 1/2 and 4/7 in ascending order.", "correct_answer": "1/2, 4/7", "incorrect_answer": "4/7, 1/2", "feedback_hint": "Ask the student to express both fractions in fourteenths."},
    {"id": "Q0013", "concept": "F5", "text": "Find a common denominator for 1/3 and 1/4.", "correct_answer": "12", "incorrect_answer": "7", "feedback_hint": "Ask for a number that both 3 and 4 divide into."},
    {"id": "Q0014", "concept": "F5", "text": "Rewrite 2/5 and 1/2 using a common denominator.", "correct_answer": "4/10 and 5/10", "incorrect_answer": "2/7 and 1/7", "feedback_hint": "Ask what common multiple 2 and 5 share."},
    {"id": "Q0015", "concept": "F5", "text": "Give a common denominator for 3/4 and 1/6.", "correct_answer": "12", "incorrect_answer": "10", "feedback_hint": "Ask the student to list multiples of 4 and 6."},
    {"id": "Q0016", "concept": "F6", "text": "Calculate 2/7 + 3/7.", "correct_answer": "5/7", "incorrect_answer": "5/14", "feedback_hint": "Ask whether the denominator changes when it already matches."},
    {"id": "Q0017", "concept": "F6", "text": "Calculate 1/9 + 5/9.", "correct_answer": "6/9", "incorrect_answer": "6/18", "feedback_hint": "Point out that the pieces already have the same size."},
    {"id": "Q0018", "concept": "F6", "text": "What is 3/10 + 4/10?", "correct_answer": "7/10", "incorrect_answer": "7/20", "feedback_hint": "Ask whether tenths become twentieths when they are combined."},
    {"id": "Q0019", "concept": "F7", "text": "Calculate 1/2 + 1/3.", "correct_answer": "5/6", "incorrect_answer": "2/5", "feedback_hint": "Ask for a common denominator for 2 and 3."},
    {"id": "Q0020", "concept": "F7", "text": "Calculate 2/5 + 1/4.", "correct_answer": "13/20", "incorrect_answer": "3/9", "feedback_hint": "Ask for a common denominator for 5 and 4."},
    {"id": "Q0021", "concept": "F7", "text": "What is 3/4 + 1/6?", "correct_answer": "11/12", "incorrect_answer": "4/10", "feedback_hint": "Ask for a common multiple of 4 and 6."},
    {"id": "Q0022", "concept": "F8", "text": "Simplify 4/8.", "correct_answer": "1/2", "incorrect_answer": "1/8", "feedback_hint": "Ask for a common factor that divides both 4 and 8."},
    {"id": "Q0023", "concept": "F8", "text": "Write 6/9 in simplest form.", "correct_answer": "2/3", "incorrect_answer": "2/9", "feedback_hint": "Ask which factor divides both 6 and 9."},
    {"id": "Q0024", "concept": "F8", "text": "Simplify 10/15.", "correct_answer": "2/3", "incorrect_answer": "2/15", "feedback_hint": "Ask for a common factor of 10 and 15."},
]


QUESTIONS_BY_CONCEPT = {
    concept: [question for question in QUESTIONS if question["concept"] == concept]
    for concept in {question["concept"] for question in QUESTIONS}
}
