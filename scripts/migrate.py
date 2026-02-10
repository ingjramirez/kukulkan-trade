"""Run pending SQL migrations against the SQLite database.

Tracks applied migrations in a `schema_migrations` table.

Usage:
    python scripts/migrate.py                     # default: data/kukulkan.db
    python scripts/migrate.py --db /path/to.db    # custom path
    python scripts/migrate.py --dry-run           # print pending, don't apply
"""

import argparse
import glob
import os
import sqlite3
import sys

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "migrations")


def ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT NOT NULL PRIMARY KEY,
            applied_at DATETIME NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def get_applied(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {r[0] for r in rows}


def get_pending(applied: set[str]) -> list[str]:
    pattern = os.path.join(MIGRATIONS_DIR, "*.sql")
    files = sorted(glob.glob(pattern))
    pending = []
    for f in files:
        name = os.path.basename(f)
        if name not in applied:
            pending.append(f)
    return pending


def apply_migration(conn: sqlite3.Connection, path: str) -> None:
    name = os.path.basename(path)
    with open(path) as f:
        sql = f.read()
    conn.executescript(sql)
    conn.execute(
        "INSERT INTO schema_migrations (version) VALUES (?)", (name,)
    )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "--db", default="data/kukulkan.db", help="Path to SQLite database"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show pending migrations without applying"
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    ensure_migrations_table(conn)
    applied = get_applied(conn)
    pending = get_pending(applied)

    if not pending:
        print("No pending migrations.")
        return

    print(f"Pending migrations: {len(pending)}")
    for p in pending:
        name = os.path.basename(p)
        if args.dry_run:
            print(f"  [pending] {name}")
        else:
            print(f"  Applying {name}...", end=" ")
            try:
                apply_migration(conn, p)
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")
                conn.close()
                sys.exit(1)

    conn.close()
    if args.dry_run:
        print("(dry-run — no changes applied)")
    else:
        print("All migrations applied.")


if __name__ == "__main__":
    main()
