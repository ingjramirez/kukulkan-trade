"""One-time backfill: insert today's missing trades and sync positions from Alpaca.

Run on server: cd /opt/kukulkan-trade && source .venv/bin/activate && python scripts/backfill_trades.py
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from src.storage.database import Database  # noqa: E402


async def backfill():
    db = Database(url="sqlite+aiosqlite:///data/kukulkan.db")
    await db.init_db()

    # The 4 missing trades from today's morning session (actual Alpaca fill prices)
    trades = [
        ("B", "XLE", "SELL", 111, 53.33973, "AI: Energy overexposure trim (reconciled)"),
        ("B", "XOM", "SELL", 27, 149.88963, "AI: Energy overexposure trim (reconciled)"),
        ("B", "GLD", "SELL", 9, 463.27, "AI: Gold profit-taking (reconciled)"),
        ("B", "XLP", "BUY", 23, 87.76, "AI: Defensive rotation (reconciled)"),
    ]

    for portfolio, ticker, side, shares, price, reason in trades:
        await db.log_trade(
            portfolio=portfolio,
            ticker=ticker,
            side=side,
            shares=shares,
            price=price,
            reason=reason,
        )
        print(f"Logged: {side} {shares}x {ticker} @ ${price:.2f}")

    # Now sync positions from Alpaca
    from alpaca.trading.client import TradingClient

    client = TradingClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=True,
    )

    # Get Alpaca positions (source of truth)
    positions = client.get_all_positions()
    alpaca_map: dict[str, tuple[float, float]] = {}
    for pos in positions:
        qty = float(pos.qty)
        if qty > 0:
            alpaca_map[pos.symbol] = (qty, float(pos.avg_entry_price))
            print(f"Alpaca: {pos.symbol} = {qty} shares @ ${float(pos.avg_entry_price):.2f}")
        else:
            print(f"Alpaca SHORT (skip): {pos.symbol} = {qty}")

    # Fix DB positions to match Alpaca
    for pname in ("A", "B"):
        db_positions = await db.get_positions(pname)
        for p in db_positions:
            if p.ticker in alpaca_map:
                a_qty, a_price = alpaca_map[p.ticker]
                if abs(p.shares - a_qty) > 0.01:
                    await db.upsert_position(pname, p.ticker, a_qty, a_price)
                    print(f"Fixed {pname}/{p.ticker}: {p.shares} -> {a_qty}")
                else:
                    print(f"OK    {pname}/{p.ticker}: {p.shares} (matches)")
                del alpaca_map[p.ticker]
            else:
                await db.upsert_position(pname, p.ticker, 0, p.avg_price)
                print(f"Zeroed {pname}/{p.ticker}: {p.shares} -> 0")

    # New Alpaca positions not in DB — assign to B
    for ticker, (qty, price) in alpaca_map.items():
        await db.upsert_position("B", ticker, qty, price)
        print(f"Added B/{ticker}: 0 -> {qty}")

    # Sync cash from Alpaca account
    account = client.get_account()
    alpaca_cash = float(account.cash)
    print(f"\nAlpaca total cash: ${alpaca_cash:.2f}")
    for pname, ratio in [("A", 1 / 3), ("B", 2 / 3)]:
        portfolio = await db.get_portfolio(pname)
        if portfolio:
            new_cash = round(alpaca_cash * ratio, 2)
            await db.upsert_portfolio(pname, cash=new_cash, total_value=portfolio.total_value)
            print(f"{pname} cash: ${portfolio.cash:.2f} -> ${new_cash:.2f}")

    await db.close()
    print("\nBackfill complete!")


if __name__ == "__main__":
    asyncio.run(backfill())
