
import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "smartguard_v2.db"


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database():
    with get_connection() as conn:
        conn.execute("PRAGMA journal_mode = WAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS doorlock_settings (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                password_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS registered_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_label TEXT NOT NULL UNIQUE,
                face_data_path TEXT,
                is_active INTEGER NOT NULL DEFAULT 1
                    CHECK(is_active IN (0, 1)),
                registered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                method TEXT NOT NULL,
                success INTEGER NOT NULL
                    CHECK(success IN (0, 1)),
                source TEXT NOT NULL,
                detail TEXT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id)
                    REFERENCES registered_users(id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                media_path TEXT,
                access_log_id INTEGER,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(access_log_id)
                    REFERENCES access_logs(id)
            )
        """)

    print("SQLite DB 생성 완료:", DB_PATH)


def set_doorlock_password(password):
    if not password or len(password) < 4:
        raise ValueError("비밀번호는 최소 4자리입니다.")

    hashed = generate_password_hash(password)

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO doorlock_settings (id, password_hash)
            VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET
                password_hash = excluded.password_hash,
                updated_at = CURRENT_TIMESTAMP
        """, (hashed,))


def register_user(name, face_label, face_data_path=None):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO registered_users
            (name, face_label, face_data_path)
            VALUES (?, ?, ?)
        """, (name, face_label, face_data_path))


def find_user_by_face(face_label):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT id, name, face_label
            FROM registered_users
            WHERE face_label = ?
              AND is_active = 1
        """, (face_label,)).fetchone()

    return dict(row) if row else None


def add_access_log(
    method, success, source, user_id=None, detail=""
):
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO access_logs
            (user_id, method, success, source, detail)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, method, int(success), source, detail))
        return cursor.lastrowid


def add_security_event(
    event_type, message, media_path=None, access_log_id=None
):
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO security_events
            (event_type, message, media_path, access_log_id)
            VALUES (?, ?, ?, ?)
        """, (event_type, message, media_path, access_log_id))
        return cursor.lastrowid


def show_tables():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()

    for row in rows:
        print("-", row["name"])


if __name__ == "__main__":
    init_database()
    show_tables()
