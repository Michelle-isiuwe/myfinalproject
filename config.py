"""
config.py
=========
Central configuration for the Explainable Student Academic Performance
Prediction System.

All paths, hyperparameters, feature lists, and category mappings are
defined here so the rest of the codebase can stay clean and modular.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"
UPLOADS_DIR = BASE_DIR / "uploads"
DB_DIR = BASE_DIR / "database"
ASSETS_DIR = BASE_DIR / "assets"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, UPLOADS_DIR, DB_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "fyp_real_response_sheet.xlsx"
DB_PATH = DB_DIR / "eduprediction.db"

# ---------------------------------------------------------------------------
# Target & risk labels
# ---------------------------------------------------------------------------
CGPA_COL = "Current CGPA Range (5.0 scale)"
TARGET_COL = "risk_level"

RISK_CLASSES = [
    "Below 1.50",
    "1.50 – 2.49",
    "2.50 – 3.49",
    "3.50 – 4.49",
    "4.50 – 5.00",
]

RISK_COLORS = {
    "Below 1.50":    "#EF4444",
    "1.50 – 2.49":    "#F59E0B",
    "2.50 – 3.49":    "#3B82F6",
    "3.50 – 4.49":    "#10B981",
    "4.50 – 5.00":    "#0EA5A4",
}

# Map current CGPA buckets to explicit 5-band labels
CGPA_TO_RISK = {
    "Below 1.50":  "Below 1.50",
    "1.50 – 2.49": "1.50 – 2.49",
    "2.50 – 3.49": "2.50 – 3.49",
    "3.50 – 4.49": "3.50 – 4.49",
    "4.50 – 5.00": "4.50 – 5.00",
}

# ---------------------------------------------------------------------------
# Degree-class display names (UI only — the model still predicts CGPA bands)
# ---------------------------------------------------------------------------
# The survey collected CGPA ranges, so the model's target is the range itself.
# These names are shown alongside the range in the UI for readability.
CGPA_BAND_TO_CLASS = {
    "4.50 – 5.00": "First Class",
    "3.50 – 4.49": "Second Class Upper",
    "2.50 – 3.49": "Second Class Lower",
    "1.50 – 2.49": "Third Class",
    "Below 1.50":  "Pass",
}

# Combined label for display, e.g. "4.50 – 5.00 (First Class)"
CGPA_BAND_DISPLAY = {
    band: f"{band} ({cls})" for band, cls in CGPA_BAND_TO_CLASS.items()
}

# ---------------------------------------------------------------------------
# Target-leakage columns (NEVER fed into the model)
# ---------------------------------------------------------------------------
LEAKAGE_COLS = [
    CGPA_COL,                                            # Current CGPA itself
    "Previous Semester CGPA Range   (5.0 scale)",        # Previous CGPA
    "How would you rate your academic performance overall?",  # self-rated perf
    "What is your average Test/Continuous Assessment score?", # CA score
]

# ---------------------------------------------------------------------------
# Predictor features (everything else from the survey)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "Gender",
    "Age Range",
    "Level of Study",
    "Field of Study",
    "Do you usually complete and submit assignments on time?",
    "How often do you prepare for tests/exams in advance?",
    "How many hours do you study per day on average?",
    "Do you have a personal study timetable?",
    "How would you rate your concentration during study?",
    "How often do you follow your study timetable?",
    "How well do you understand what is taught in class?",
    "Do you engage in group study",
    "What is your average class attendance rate?",
    "How actively do you participate in class?",
    "Do you attend tutorials/practical sessions regularly?",
    "Do you have a part-time job? If yes, how many hours do you work per week?",
    "How would you rate your financial situation?",
    "How conducive is your study environment?",
    "Do you have regular access to the internet?",
    "What devices do you use for studying?",
    "How often do you use online learning tools (e.g., YouTube, Coursera, AI)?",
    "How many hours do you spend on social media daily?",
    "How many hours do you sleep daily?",
    "How often do you feel academically stressed?",
]

# Short readable aliases (used in the UI to keep things tidy)
FEATURE_ALIASES = {
    "Gender": "Gender",
    "Age Range": "Age",
    "Level of Study": "Level",
    "Field of Study": "Field",
    "Do you usually complete and submit assignments on time?": "Assignment Submission",
    "How often do you prepare for tests/exams in advance?": "Test Preparation",
    "How many hours do you study per day on average?": "Daily Study Hours",
    "Do you have a personal study timetable?": "Has Timetable",
    "How would you rate your concentration during study?": "Concentration",
    "How often do you follow your study timetable?": "Follows Timetable",
    "How well do you understand what is taught in class?": "Class Understanding",
    "Do you engage in group study": "Group Study",
    "What is your average class attendance rate?": "Class Attendance",
    "How actively do you participate in class?": "Class Participation",
    "Do you attend tutorials/practical sessions regularly?": "Tutorial Attendance",
    "Do you have a part-time job? If yes, how many hours do you work per week?": "Part-time Job",
    "How would you rate your financial situation?": "Financial Situation",
    "How conducive is your study environment?": "Study Environment",
    "Do you have regular access to the internet?": "Internet Access",
    "What devices do you use for studying?": "Devices Used",
    "How often do you use online learning tools (e.g., YouTube, Coursera, AI)?": "Online Tools Usage",
    "How many hours do you spend on social media daily?": "Social Media Hours",
    "How many hours do you sleep daily?": "Sleep Hours",
    "How often do you feel academically stressed?": "Stress Level",
}

# ---------------------------------------------------------------------------
# Ordinal feature ordering — for ordinal label encoding where order matters
# ---------------------------------------------------------------------------
ORDINAL_ORDERINGS = {
    "Age Range": ["Below 18", "18–21", "22–25", "26 and above"],
    "Level of Study": ["100 Level", "200 Level", "300 Level", "400 Level", "500 Level"],
    "Do you usually complete and submit assignments on time?":
        ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "How often do you prepare for tests/exams in advance?":
        ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "How many hours do you study per day on average?":
        ["Less than 1 hour", "1–2 hours", "3–4 hours", "More than 4 hours"],
    "How would you rate your concentration during study?":
        ["Very low", "low", "Moderate", "High", "Very High"],
    "How often do you follow your study timetable?":
        ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "How well do you understand what is taught in class?":
        ["Very poorly", "Poorly", "Moderately", "Well", "Very Well"],
    "Do you engage in group study":
        ["Never", "Rarely", "Occasionally", "Frequently"],
    "What is your average class attendance rate?":
        ["Below 50%", "50–69%", "70–89%", "90–100%"],
    "How actively do you participate in class?":
        ["Not Active", "Passive", "Moderately Active", "Active", "Very Active"],
    "Do you attend tutorials/practical sessions regularly?":
        ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "Do you have a part-time job? If yes, how many hours do you work per week?":
        ["No, I do not have a part-time job", "Yes, 1–10 hours per week",
         "Yes, 11–20 hours per week", "Yes, above 20 hours per week"],
    "How would you rate your financial situation?":
        ["Very Challenging", "Challenging", "Moderate", "Stable", "Very Stable"],
    "How conducive is your study environment?":
        ["Very Poor", "Poor", "Moderate", "Conducive", "Very Conducive"],
    "How often do you use online learning tools (e.g., YouTube, Coursera, AI)?":
        ["Never", "Rarely", "Sometimes", "Often", "Very Often"],
    "How many hours do you spend on social media daily?":
        ["Less than 1 hour", "1–3 hours", "4–6 hours", "More than 6 hours"],
    "How many hours do you sleep daily?":
        ["Less than 4 hours", "4–6 hours", "7–8 hours", "More than 8 hours"],
    "How often do you feel academically stressed?":
        ["Never", "Rarely", "Sometimes", "Often", "Very Often"],
}

# Pure categoricals (no natural order) → one-hot encoded
NOMINAL_COLS = [
    "Gender",
    "Field of Study",
    "Do you have a personal study timetable?",
    "Do you have regular access to the internet?",
    "What devices do you use for studying?",
]

# Fixed response sets for nominal fields (UI + training consistency)
NOMINAL_OPTIONS = {
    "Do you have a personal study timetable?": ["No", "Yes"],
}

# ---------------------------------------------------------------------------
# Model hyper-parameters  (matched to the v2 pipeline that produced ~80%)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_FOLDS = 5  # OOF folds used inside the Meta-Ensemble

RF_PARAMS = dict(
    n_estimators=300,
    min_samples_split=4,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    verbosity=0,
    n_jobs=-1,
)

LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

LR_PARAMS = dict(max_iter=2000, random_state=RANDOM_STATE)
META_LR_PARAMS = dict(max_iter=3000, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# Resampling strategies & ensembles available in the UI
# ---------------------------------------------------------------------------
RESAMPLING_STRATEGIES = ["None (Original)", "SMOTE+Tomek"]

ENSEMBLE_STRATEGIES = [
    "Meta-Ensemble Pipeline",
]