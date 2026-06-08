-- ============================================================
-- EduPrediction SQLite Schema
-- Explainable Student Academic Performance Prediction System
-- ============================================================

-- Application users (students, educators, administrators)
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    full_name     TEXT,
    email         TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('admin', 'educator', 'student')),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login    TIMESTAMP
);

-- Profile fields collected once at student registration
CREATE TABLE IF NOT EXISTS student_profiles (
    profile_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER UNIQUE NOT NULL,
    gender          TEXT,
    age_range       TEXT,
    level_of_study  TEXT,
    field_of_study  TEXT,
    internet_access TEXT,
    devices_used    TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Student records (created when an educator runs a prediction for a student)
CREATE TABLE IF NOT EXISTS students (
    student_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    matric_no     TEXT,
    full_name     TEXT,
    level         TEXT,
    field         TEXT,
    created_by    INTEGER,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(user_id)
);

-- Every prediction the system has ever made
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      INTEGER,
    user_id         INTEGER NOT NULL,
    model_used      TEXT NOT NULL,
    resampling      TEXT,
    predicted_risk  TEXT NOT NULL,
    confidence      REAL,
    prob_below_150  REAL,                  -- "Below 1.50"
    prob_150_249    REAL,                  -- "1.50 – 2.49"
    prob_250_349    REAL,                  -- "2.50 – 3.49"
    prob_350_449    REAL,                  -- "3.50 – 4.49"
    prob_450_500    REAL,                  -- "4.50 – 5.00"
    input_payload   TEXT,                  -- JSON dump of feature values
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (user_id)    REFERENCES users(user_id)
);

-- History of which model an educator activated for production use
CREATE TABLE IF NOT EXISTS model_selection_history (
    selection_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL,
    model_choice   TEXT NOT NULL,
    resampling     TEXT,
    accuracy       REAL,
    f1_score       REAL,
    roc_auc        REAL,
    notes          TEXT,
    selected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Saved (bookmarked) predictions for follow-up
CREATE TABLE IF NOT EXISTS saved_predictions (
    saved_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    user_id       INTEGER NOT NULL,
    note          TEXT,
    saved_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
    FOREIGN KEY (user_id)       REFERENCES users(user_id)
);

-- Educator suggestions for students (account-level; optional prediction link)
CREATE TABLE IF NOT EXISTS educator_feedback (
    feedback_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    student_user_id    INTEGER NOT NULL,
    educator_user_id   INTEGER NOT NULL,
    prediction_id      INTEGER,
    message            TEXT NOT NULL,
    read_at            TIMESTAMP,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_user_id)  REFERENCES users(user_id),
    FOREIGN KEY (educator_user_id) REFERENCES users(user_id),
    FOREIGN KEY (prediction_id)    REFERENCES predictions(prediction_id)
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_feedback_student    ON educator_feedback(student_user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_user      ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_student   ON predictions(student_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created   ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_users_username        ON users(username);
CREATE INDEX IF NOT EXISTS idx_student_profiles_user ON student_profiles(user_id);