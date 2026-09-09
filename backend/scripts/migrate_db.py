"""
One-time migration script: adds the v2 columns (gradcam_available,
weather_source, fusion_method) to an existing v1 `predictions` table
without losing any existing rows.

WHY THIS EXISTS: SQLAlchemy's `Base.metadata.create_all()` (used in
app/database.py::init_db) only CREATES tables that don't exist yet — it
does not ALTER existing tables to add new columns. Anyone who already
has a `storage/app.db` from the v1 release of this project needs to run
this once after upgrading, or simply delete `storage/app.db` and let it
be recreated empty (fine for a fresh demo, not fine if you want to keep
existing prediction history).

Usage:
    cd backend
    python -m scripts.migrate_db      # if run as a package, or:
    python migrate_db.py              # if run directly from backend/

Safe to run multiple times — it checks for each column's existence
before adding it.
"""
import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings  # noqa: E402


def migrate():
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    if not os.path.exists(db_path):
        print(f"No existing database found at {db_path} — nothing to migrate. "
              "A fresh v2 schema will be created automatically on next app startup.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cur.fetchall()}

    new_columns = {
        "gradcam_available": "BOOLEAN DEFAULT 0 NOT NULL",
        "weather_source": "VARCHAR DEFAULT 'manual' NOT NULL",
        "fusion_method": "VARCHAR DEFAULT 'late_fusion' NOT NULL",
    }

    added = []
    for col_name, col_def in new_columns.items():
        if col_name not in existing_columns:
            cur.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_def}")
            added.append(col_name)

    conn.commit()
    conn.close()

    if added:
        print(f"Migration complete. Added columns: {added}")
    else:
        print("Database already up to date — no migration needed.")


if __name__ == "__main__":
    migrate()
