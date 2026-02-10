"""One-time: set Portfolio A cash to correct value.

A = Alpaca equity - B total value, so the books balance.
"""

import asyncio

from src.storage.database import Database


async def fix():
    db = Database(url="sqlite+aiosqlite:///data/kukulkan.db")
    await db.init_db()

    correct_a_cash = 22030.73
    await db.upsert_portfolio("A", cash=correct_a_cash, total_value=correct_a_cash)

    port_a = await db.get_portfolio("A")
    port_b = await db.get_portfolio("B")
    combined = port_a.total_value + port_b.total_value
    print(f"Portfolio A: cash=${port_a.cash:.2f}, total=${port_a.total_value:.2f}")
    print(f"Portfolio B: cash=${port_b.cash:.2f}, total=${port_b.total_value:.2f}")
    print(f"Combined: ${combined:.2f}")
    await db.close()


if __name__ == "__main__":
    asyncio.run(fix())
