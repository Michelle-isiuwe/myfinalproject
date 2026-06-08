"""
database/db_manager.py
======================
Thin SQLite wrapper. Handles initialization, user CRUD, prediction
logging, model-selection logging and history retrieval.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import bcrypt
import pandas as pd

from config import DB_PATH


SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_PREDICTION_PROB_COLS = {
    "Below 1.50": "prob_below_150",
    "1.50 – 2.49": "prob_150_249",
    "2.50 – 3.49": "prob_250_349",
    "3.50 – 4.49": "prob_350_449",
    "4.50 – 5.00": "prob_450_500",
}


class DBManager:
    """SQLite manager — kept intentionally simple and dependency-light."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    # ------------------------------------------------------------------ #
    # Low-level helpers
    # ------------------------------------------------------------------ #
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        """Apply schema.sql, run migrations, and seed a default admin."""
        with self._conn() as conn:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
            self._migrate(conn)
            cur = conn.execute("SELECT COUNT(*) AS n FROM users")
            if cur.fetchone()["n"] == 0:
                self._seed_default_users(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Apply in-place schema migrations for existing databases."""
        # --- Migrate users role constraint to include 'student' ---
        schema_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchone()
        if schema_row and "'student'" not in schema_row[0]:
            conn.executescript("""
                PRAGMA foreign_keys = OFF;
                DROP TABLE IF EXISTS users_new;
                CREATE TABLE users_new (
                    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    username      TEXT UNIQUE NOT NULL,
                    full_name     TEXT,
                    email         TEXT UNIQUE,
                    password_hash TEXT NOT NULL,
                    role          TEXT NOT NULL CHECK (role IN ('admin', 'educator', 'student')),
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login    TIMESTAMP
                );
                INSERT INTO users_new SELECT * FROM users;
                DROP TABLE users;
                ALTER TABLE users_new RENAME TO users;
                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                PRAGMA foreign_keys = ON;
            """)

        conn.execute("""
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
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_student "
            "ON educator_feedback(student_user_id)"
        )
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(educator_feedback)").fetchall()
        }
        if "read_at" not in cols:
            conn.execute(
                "ALTER TABLE educator_feedback ADD COLUMN read_at TIMESTAMP"
            )

    def _seed_default_users(self, conn: sqlite3.Connection) -> None:
        """Seed the default admin account so the system is usable out of the box."""
        defaults = [
            ("admin", "System Administrator", "admin@edupredict.local", "admin123", "admin"),
        ]
        for username, name, email, pw, role in defaults:
            pw_hash = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
            conn.execute(
                "INSERT INTO users (username, full_name, email, password_hash, role) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, name, email, pw_hash, role),
            )

    # ------------------------------------------------------------------ #
    # User / Auth
    # ------------------------------------------------------------------ #
    def create_user(self, username: str, full_name: str, email: str,
                    password: str, role: str = "student") -> Tuple[bool, str]:
        if role not in ("admin", "educator", "student"):
            return False, "Invalid role."
        if len(password) < 6:
            return False, "Password must be at least 6 characters."
        pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO users (username, full_name, email, password_hash, role) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (username, full_name, email, pw_hash, role),
                )
            return True, "User created successfully."
        except sqlite3.IntegrityError as e:
            return False, f"Username or email already exists. ({e})"

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if row is None:
                return None
            if not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
                return None
            conn.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?",
                (row["user_id"],),
            )
            return dict(row)

    def list_users(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT user_id, username, full_name, email, role, "
                "created_at, last_login FROM users ORDER BY user_id",
                conn,
            )

    def reset_user_password(self, user_id: int, new_password: str) -> Tuple[bool, str]:
        if len(new_password) < 6:
            return False, "Password must be at least 6 characters."
        pw_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE users SET password_hash = ? WHERE user_id = ?",
                (pw_hash, user_id),
            )
            if cur.rowcount == 0:
                return False, "User not found."
        return True, "Password updated successfully."

    def delete_user(self, user_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM educator_feedback "
                "WHERE student_user_id = ? OR educator_user_id = ?",
                (user_id, user_id),
            )
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

    # ------------------------------------------------------------------ #
    # Student profiles (stored once at registration)
    # ------------------------------------------------------------------ #
    def create_student_profile(self, user_id: int, gender: str, age_range: str,
                                level_of_study: str, field_of_study: str,
                                internet_access: str, devices_used: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO student_profiles
                   (user_id, gender, age_range, level_of_study, field_of_study,
                    internet_access, devices_used)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, gender, age_range, level_of_study, field_of_study,
                 internet_access, devices_used),
            )

    def get_student_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM student_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_last_academic_inputs(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Return the input payload of the student's latest prediction (for academic pre-fill)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT input_payload FROM predictions WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (user_id,),
            ).fetchone()
            if row and row["input_payload"]:
                return json.loads(row["input_payload"])
            return None

    # ------------------------------------------------------------------ #
    # Educator student management helpers
    # ------------------------------------------------------------------ #
    def list_students_with_latest_prediction(self) -> pd.DataFrame:
        """All student users with their profile and latest prediction (for educator dashboard)."""
        with self._conn() as conn:
            return pd.read_sql_query(
                """SELECT u.user_id, u.username, u.full_name,
                          sp.level_of_study, sp.field_of_study,
                          lp.predicted_risk, lp.confidence, lp.model_used,
                          lp.created_at AS predicted_at
                   FROM users u
                   LEFT JOIN student_profiles sp ON u.user_id = sp.user_id
                   LEFT JOIN (
                       SELECT user_id, predicted_risk, confidence, model_used, created_at,
                              ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn
                       FROM predictions
                   ) lp ON lp.user_id = u.user_id AND lp.rn = 1
                   WHERE u.role = 'student'
                   ORDER BY u.created_at DESC""",
                conn,
            )

    def get_student_predictions(self, student_user_id: int,
                                limit: int = 50) -> pd.DataFrame:
        """All predictions for a specific student."""
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM predictions WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                conn, params=(student_user_id, limit),
            )

    def get_latest_student_prediction(self, student_user_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (student_user_id,),
            ).fetchone()
            return dict(row) if row else None

    def prediction_row_to_view(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Shape a DB prediction row like session ``last_prediction`` for the UI."""
        payload = json.loads(row["input_payload"]) if row.get("input_payload") else {}
        probs = {
            band: float(row.get(col) or 0.0)
            for band, col in _PREDICTION_PROB_COLS.items()
        }
        return {
            "id": row["prediction_id"],
            "payload": payload,
            "result": {
                "predicted_band": row["predicted_risk"],
                "confidence": float(row["confidence"]),
                "probabilities": probs,
            },
            "model": row["model_used"],
            "timestamp": row.get("created_at"),
        }

    def get_latest_prediction_view(self, student_user_id: int) -> Optional[Dict[str, Any]]:
        row = self.get_latest_student_prediction(student_user_id)
        return self.prediction_row_to_view(row) if row else None

    # ------------------------------------------------------------------ #
    # Educator feedback for students
    # ------------------------------------------------------------------ #
    def add_educator_feedback(
        self,
        student_user_id: int,
        educator_user_id: int,
        message: str,
        prediction_id: Optional[int] = None,
    ) -> int:
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be empty.")
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO educator_feedback
                   (student_user_id, educator_user_id, prediction_id, message)
                   VALUES (?, ?, ?, ?)""",
                (student_user_id, educator_user_id, prediction_id, message),
            )
            return cur.lastrowid

    def get_educator_feedback_for_student(
        self, student_user_id: int, limit: int = 50
    ) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                """SELECT f.feedback_id, f.message, f.created_at, f.prediction_id,
                          u.full_name AS educator_name, u.username AS educator_username,
                          p.created_at AS prediction_at, p.predicted_risk
                   FROM educator_feedback f
                   JOIN users u ON f.educator_user_id = u.user_id
                   LEFT JOIN predictions p ON f.prediction_id = p.prediction_id
                   WHERE f.student_user_id = ?
                   ORDER BY f.created_at DESC
                   LIMIT ?""",
                conn,
                params=(student_user_id, limit),
            )

    def count_unread_feedback(self, student_user_id: int) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM educator_feedback "
                "WHERE student_user_id = ? AND read_at IS NULL",
                (student_user_id,),
            ).fetchone()
            return int(row["n"])

    def mark_feedback_read(self, student_user_id: int) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE educator_feedback SET read_at = CURRENT_TIMESTAMP "
                "WHERE student_user_id = ? AND read_at IS NULL",
                (student_user_id,),
            )
            return cur.rowcount

    # ------------------------------------------------------------------ #
    # Legacy student records (created by educator during old workflow)
    # ------------------------------------------------------------------ #
    def create_student(self, matric_no: str, full_name: str, level: str,
                       field: str, created_by: int) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO students (matric_no, full_name, level, field, created_by) "
                "VALUES (?, ?, ?, ?, ?)",
                (matric_no, full_name, level, field, created_by),
            )
            return cur.lastrowid

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #
    def log_prediction(
        self,
        user_id: int,
        model_used: str,
        resampling: str,
        predicted_risk: str,
        confidence: float,
        probs: Dict[str, float],
        input_payload: Dict[str, Any],
        student_id: Optional[int] = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO predictions
                   (student_id, user_id, model_used, resampling, predicted_risk,
                    confidence, prob_below_150, prob_150_249, prob_250_349,
                    prob_350_449, prob_450_500, input_payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    student_id, user_id, model_used, resampling, predicted_risk,
                    float(confidence),
                    float(probs.get("Below 1.50", 0.0)),
                    float(probs.get("1.50 – 2.49", 0.0)),
                    float(probs.get("2.50 – 3.49", 0.0)),
                    float(probs.get("3.50 – 4.49", 0.0)),
                    float(probs.get("4.50 – 5.00", 0.0)),
                    json.dumps(input_payload, default=str),
                ),
            )
            return cur.lastrowid

    def get_predictions(self, user_id: Optional[int] = None,
                        limit: int = 500) -> pd.DataFrame:
        with self._conn() as conn:
            if user_id is None:
                q = ("SELECT p.*, u.username FROM predictions p "
                     "LEFT JOIN users u ON p.user_id = u.user_id "
                     "ORDER BY created_at DESC LIMIT ?")
                return pd.read_sql_query(q, conn, params=(limit,))
            q = ("SELECT p.*, u.username FROM predictions p "
                 "LEFT JOIN users u ON p.user_id = u.user_id "
                 "WHERE p.user_id = ? ORDER BY created_at DESC LIMIT ?")
            return pd.read_sql_query(q, conn, params=(user_id, limit))

    def clear_predictions(self, user_id: Optional[int] = None) -> int:
        with self._conn() as conn:
            if user_id is None:
                cur = conn.execute("DELETE FROM predictions")
            else:
                cur = conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
            return cur.rowcount

    # ------------------------------------------------------------------ #
    # Model selection history
    # ------------------------------------------------------------------ #
    def log_model_selection(self, user_id: int, model_choice: str,
                             resampling: str, accuracy: float, f1: float,
                             roc_auc: float, notes: str = "") -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO model_selection_history
                   (user_id, model_choice, resampling, accuracy, f1_score, roc_auc, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, model_choice, resampling, float(accuracy),
                 float(f1), float(roc_auc), notes),
            )
            return cur.lastrowid

    def get_model_selection_history(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT m.*, u.username FROM model_selection_history m "
                "LEFT JOIN users u ON m.user_id = u.user_id "
                "ORDER BY selected_at DESC",
                conn,
            )

    # ------------------------------------------------------------------ #
    # Saved predictions
    # ------------------------------------------------------------------ #
    def save_prediction(self, prediction_id: int, user_id: int, note: str = "") -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO saved_predictions (prediction_id, user_id, note) "
                "VALUES (?, ?, ?)",
                (prediction_id, user_id, note),
            )
            return cur.lastrowid

    def get_saved_predictions(self, user_id: int) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                """SELECT s.*, p.predicted_risk, p.confidence, p.model_used,
                          p.created_at AS predicted_at
                   FROM saved_predictions s
                   JOIN predictions p ON s.prediction_id = p.prediction_id
                   WHERE s.user_id = ?
                   ORDER BY s.saved_at DESC""",
                conn, params=(user_id,),
            )

    def clear_saved_predictions(self, user_id: Optional[int] = None) -> int:
        with self._conn() as conn:
            if user_id is None:
                cur = conn.execute("DELETE FROM saved_predictions")
            else:
                cur = conn.execute("DELETE FROM saved_predictions WHERE user_id = ?", (user_id,))
            return cur.rowcount

    # ------------------------------------------------------------------ #
    # Aggregate statistics
    # ------------------------------------------------------------------ #
    def get_statistics(self) -> Dict[str, Any]:
        """Return high-level counts used by the admin dashboard."""
        with self._conn() as conn:
            total_users = conn.execute(
                "SELECT COUNT(*) AS n FROM users").fetchone()["n"]
            total_predictions = conn.execute(
                "SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
            total_students = conn.execute(
                "SELECT COUNT(*) AS n FROM users WHERE role = 'student'").fetchone()["n"]
            total_saved = conn.execute(
                "SELECT COUNT(*) AS n FROM saved_predictions").fetchone()["n"]

            rows = conn.execute(
                "SELECT predicted_risk, COUNT(*) AS n "
                "FROM predictions GROUP BY predicted_risk"
            ).fetchall()
            risk_breakdown = {r["predicted_risk"]: r["n"] for r in rows}

        return {
            "total_users": total_users,
            "total_predictions": total_predictions,
            "total_students": total_students,
            "total_saved_predictions": total_saved,
            "risk_breakdown": risk_breakdown,
        }


# Singleton-style helper for the Streamlit app
_db_instance: Optional[DBManager] = None


def get_db() -> DBManager:
    global _db_instance
    if _db_instance is None:
        _db_instance = DBManager()
    return _db_instance