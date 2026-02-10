"""One-time: redistribute cash between portfolios to fix accounting leak.

B absorbed A's cash due to the naive 1/3:2/3 sync bug. Redistribute so
each portfolio holds its correct share of Alpaca equity (A=1/3, B=2/3).
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.client import TradingClient
from src.storage.database import Database


async def fix():
    db = Database(url="sqlite+aiosqlite:///data/kukulkan.db")
    await db.init_db()

    client = TradingClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=True,
    )

    # Get Alpaca truth
    account = client.get_account()
    equity = float(account.equity)
    print(f"Alpaca equity: ${equity:.2f}")

    # Target allocations
    a_target = round(equity / 3, 2)
    b_target = round(equity * 2 / 3, 2)
    print(f"Target A: ${a_target:.2f}")
    print(f"Target B: ${b_target:.2f}")

    # B's positions value (at cost from DB)
    positions_b = await db.get_positions("B")
    # Use market value from Alpaca
    alpaca_positions = client.get_all_positions()
    market_prices = {p.symbol: float(p.current_price) for p in alpaca_positions}

    b_pos_value = sum(
        p.shares * market_prices.get(p.ticker, p.avg_price)
        for p in positions_b
    )
    print(f"B positions (market value): ${b_pos_value:.2f}")

    # A has no positions → A cash = A target
    a_cash = a_target
    # B cash = B target - B positions
    b_cash = round(b_target - b_pos_value, 2)

    print(f"\nNew A cash: ${a_cash:.2f} (was ${(await db.get_portfolio('A')).cash:.2f})")
    print(f"New B cash: ${b_cash:.2f} (was ${(await db.get_portfolio('B')).cash:.2f})")

    await db.upsert_portfolio("A", cash=a_cash, total_value=a_target)
    await db.upsert_portfolio("B", cash=b_cash, total_value=b_target)

    # Verify
    port_a = await db.get_portfolio("A")
    port_b = await db.get_portfolio("B")
    print(f"\nFinal A: ${port_a.total_value:.2f}")
    print(f"Final B: ${port_b.total_value:.2f}")
    print(f"Combined: ${port_a.total_value + port_b.total_value:.2f}")
    print(f"Alpaca: ${equity:.2f}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(fix())
