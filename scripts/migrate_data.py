"""One-time data migration: SQLite → PostgreSQL.

Reads all rows from a SQLite database and inserts them into PostgreSQL.
Processes tables in FK dependency order (tenants first).
Idempotent: skips rows that already exist (based on primary key).

Usage:
    python scripts/migrate_data.py \
        --sqlite data/kukulkan.db \
        --pg postgresql://kukulkan:PASS@localhost:5432/kukulkan
"""

import argparse
import sys

import sqlalchemy as sa
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

# Table order respecting FK dependencies (tenants first)
TABLE_ORDER = [
    "tenants",
    "portfolios",
    "positions",
    "trades",
    "daily_snapshots",
    "momentum_rankings",
    "agent_decisions",
    "market_data",
    "technical_indicators",
    "macro_data",
    "news_log",
    "discovered_tickers",
    "agent_memory",
    "trailing_stops",
    "earnings_calendar",
    "watchlist",
    "intraday_snapshots",
    "sentinel_actions",
    "tool_call_logs",
    "agent_conversations",
    "posture_history",
    "playbook_snapshots",
    "conviction_calibration",
    "agent_budget_log",
    "improvement_snapshots",
    "parameter_changelog",
    "sentiment_indicators",
    "ticker_signals",
]

BATCH_SIZE = 1000


def _pg_dsn(url: str) -> str:
    """Normalize SQLAlchemy-style PG URL to plain postgresql://."""
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg2://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix) :]
    return url


def migrate_table(
    src_engine: sa.engine.Engine,
    dst_engine: sa.engine.Engine,
    table_name: str,
) -> int:
    """Copy all rows from src to dst for a single table. Returns row count."""
    src_inspector = inspect(src_engine)
    if table_name not in src_inspector.get_table_names():
        return 0

    # Reflect the table from source
    src_meta = sa.MetaData()
    src_table = sa.Table(table_name, src_meta, autoload_with=src_engine)

    # Read all rows from SQLite
    with Session(src_engine) as src_session:
        rows = src_session.execute(sa.select(src_table)).fetchall()

    if not rows:
        return 0

    columns = [c.name for c in src_table.columns]
    pk_cols = _get_pk_columns(dst_engine, table_name)
    inserted = 0

    with Session(dst_engine) as dst_session:
        batch = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            batch.append(row_dict)

            if len(batch) >= BATCH_SIZE:
                _insert_batch(dst_session, table_name, columns, batch, pk_cols)
                inserted += len(batch)
                batch = []

        if batch:
            _insert_batch(dst_session, table_name, columns, batch, pk_cols)
            inserted += len(batch)

        dst_session.commit()

    return inserted


def _get_pk_columns(engine: sa.engine.Engine, table_name: str) -> list[str]:
    """Get primary key column names for a table."""
    inspector = inspect(engine)
    pk = inspector.get_pk_constraint(table_name)
    return pk["constrained_columns"] if pk else ["id"]


def _insert_batch(
    session: Session, table_name: str, columns: list[str], batch: list[dict], pk_cols: list[str]
) -> None:
    """Insert a batch of rows. On PK conflict, update all non-PK columns (upsert)."""
    cols = ", ".join(columns)
    placeholders = ", ".join(f":{c}" for c in columns)
    non_pk = [c for c in columns if c not in pk_cols]
    if non_pk:
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_pk)
        conflict_cols = ", ".join(pk_cols)
        sql = text(
            f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {updates}"
        )
    else:
        sql = text(f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders}) ON CONFLICT DO NOTHING")
    session.execute(sql, batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate data from SQLite to PostgreSQL")
    parser.add_argument("--sqlite", required=True, help="Path to SQLite database file")
    parser.add_argument("--pg", required=True, help="PostgreSQL URL (postgresql://user:pass@host/db)")
    args = parser.parse_args()

    src_url = f"sqlite:///{args.sqlite}"
    dst_url = _pg_dsn(args.pg)

    src_engine = create_engine(src_url)
    dst_engine = create_engine(dst_url)

    # Verify connectivity
    try:
        with dst_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        print(f"Cannot connect to PostgreSQL: {e}")
        sys.exit(1)

    print(f"Source: {args.sqlite}")
    print(f"Dest:   {dst_url}")
    print()

    total = 0
    for table_name in TABLE_ORDER:
        count = migrate_table(src_engine, dst_engine, table_name)
        if count > 0:
            print(f"  {table_name}: {count} rows")
            total += count
        else:
            print(f"  {table_name}: (empty or not found)")

    # Reset PG sequences to max(id) + 1 for SERIAL columns
    print("\nResetting sequences...")
    with dst_engine.connect() as conn:
        for table_name in TABLE_ORDER:
            try:
                result = conn.execute(text(f"SELECT MAX(id) FROM {table_name}"))
                max_id = result.scalar()
                if max_id is not None:
                    seq_name = f"{table_name}_id_seq"
                    conn.execute(text(f"SELECT setval('{seq_name}', {max_id})"))
            except Exception:
                pass  # Table may not have an id column or sequence
        conn.commit()

    print(f"\nDone. {total} total rows migrated.")

    src_engine.dispose()
    dst_engine.dispose()


if __name__ == "__main__":
    main()
