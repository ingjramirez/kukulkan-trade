"""One-time: remove duplicate reconciled trade entries."""

import asyncio

from sqlalchemy import text

from src.storage.database import Database


async def fix():
    db = Database(url="sqlite+aiosqlite:///data/kukulkan.db")
    await db.init_db()

    async with db.session() as s:
        result = await s.execute(
            text(
                "SELECT id, ticker, side, shares, price, reason, executed_at "
                "FROM trades WHERE reason LIKE '%reconciled%' ORDER BY id"
            )
        )
        rows = result.fetchall()
        print(f"Found {len(rows)} reconciled trades:")
        for r in rows:
            print(f"  id={r[0]} {r[2]} {r[3]}x {r[1]} @ {r[4]} - {r[6]}")

        # Keep the first 4, delete the rest (duplicates)
        if len(rows) > 4:
            dup_ids = [r[0] for r in rows[4:]]
            id_list = ",".join(str(i) for i in dup_ids)
            await s.execute(text(f"DELETE FROM trades WHERE id IN ({id_list})"))
            await s.commit()
            print(f"\nDeleted {len(dup_ids)} duplicate trades: {dup_ids}")
        else:
            print("\nNo duplicates to delete")

    await db.close()


if __name__ == "__main__":
    asyncio.run(fix())
