"""Run pending SQL migrations against SQLite or PostgreSQL.

Tracks applied migrations in a `schema_migrations` table.

Usage:
    python scripts/migrate.py                                       # default: data/kukulkan.db (SQLite)
    python scripts/migrate.py --db /path/to.db                      # custom SQLite path
    python scripts/migrate.py --db postgresql://user:pass@host/db    # PostgreSQL
    python scripts/migrate.py --dry-run                              # print pending, don't apply
"""

import argparse
import glob
import os
import sqlite3
import sys

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "migrations")
PG_MIGRATIONS_DIR = os.path.join(MIGRATIONS_DIR, "pg")


# ── SQLite helpers ───────────────────────────────────────────────────────────


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
    conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (name,))
    conn.commit()


def run_sqlite(db_path: str, dry_run: bool) -> None:
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    ensure_migrations_table(conn)
    applied = get_applied(conn)
    pending = get_pending(applied)

    if not pending:
        print("No pending migrations.")
        conn.close()
        return

    print(f"Pending migrations: {len(pending)}")
    for p in pending:
        name = os.path.basename(p)
        if dry_run:
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
    if dry_run:
        print("(dry-run — no changes applied)")
    else:
        print("All migrations applied.")


# ── PostgreSQL helpers ───────────────────────────────────────────────────────


def _pg_dsn(url: str) -> str:
    """Convert SQLAlchemy-style URL to psycopg2 DSN if needed.

    Accepts both:
      postgresql://user:pass@host/db
      postgresql+asyncpg://user:pass@host/db
    Returns a plain postgresql:// URL usable by psycopg2.
    """
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg2://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix) :]
    return url


def run_pg(db_url: str, dry_run: bool) -> None:
    try:
        import psycopg2
    except ImportError:
        print("psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    dsn = _pg_dsn(db_url)
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    cur = conn.cursor()

    # Ensure migrations table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT NOT NULL PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    conn.commit()

    # Get applied
    cur.execute("SELECT version FROM schema_migrations")
    applied = {r[0] for r in cur.fetchall()}

    # Get pending PG migrations
    if not os.path.isdir(PG_MIGRATIONS_DIR):
        print(f"PG migrations directory not found: {PG_MIGRATIONS_DIR}")
        conn.close()
        sys.exit(1)

    pattern = os.path.join(PG_MIGRATIONS_DIR, "*.sql")
    files = sorted(glob.glob(pattern))
    pending = [(f, os.path.basename(f)) for f in files if os.path.basename(f) not in applied]

    if not pending:
        print("No pending PG migrations.")
        conn.close()
        return

    print(f"Pending PG migrations: {len(pending)}")
    for path, name in pending:
        if dry_run:
            print(f"  [pending] {name}")
        else:
            print(f"  Applying {name}...", end=" ")
            try:
                with open(path) as f:
                    sql = f.read()
                cur.execute(sql)
                conn.commit()
                print("OK")
            except Exception as e:
                conn.rollback()
                print(f"FAILED: {e}")
                conn.close()
                sys.exit(1)

    conn.close()
    if dry_run:
        print("(dry-run — no changes applied)")
    else:
        print("All PG migrations applied.")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "--db",
        default="data/kukulkan.db",
        help="SQLite path or PostgreSQL URL (postgresql://user:pass@host/db)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show pending migrations without applying")
    args = parser.parse_args()

    if args.db.startswith("postgresql"):
        run_pg(args.db, args.dry_run)
    else:
        run_sqlite(args.db, args.dry_run)


if __name__ == "__main__":
    main()
