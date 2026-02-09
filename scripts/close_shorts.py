"""One-time: close all unintended short positions on Alpaca paper account.

Run on server: cd /opt/kukulkan-trade && source .venv/bin/activate && python scripts/close_shorts.py
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def main():
    client = TradingClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=True,
    )

    positions = client.get_all_positions()
    shorts = [(p.symbol, abs(float(p.qty))) for p in positions if float(p.qty) < 0]

    if not shorts:
        print("No short positions found. Nothing to do.")
        return

    print(f"Found {len(shorts)} short positions to close:\n")
    for symbol, qty in shorts:
        print(f"  {symbol}: -{qty:.0f} shares")

    print()
    for symbol, qty in shorts:
        try:
            order = client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=int(qty),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=f"kk-close-short-{symbol}-{int(time.time())}",
                )
            )
            print(f"BUY {int(qty)}x {symbol} — order {order.id} ({order.status})")
        except Exception as e:
            print(f"FAILED {symbol}: {e}")

    # Wait for fills
    print("\nWaiting 10s for fills...")
    time.sleep(10)

    # Verify
    positions = client.get_all_positions()
    remaining_shorts = [p for p in positions if float(p.qty) < 0]
    if remaining_shorts:
        print(f"\nWARNING: {len(remaining_shorts)} shorts still open:")
        for p in remaining_shorts:
            print(f"  {p.symbol}: {p.qty}")
    else:
        print("\nAll shorts closed successfully!")

    account = client.get_account()
    print(f"\nAccount equity: ${float(account.equity):,.2f}")
    print(f"Account cash:   ${float(account.cash):,.2f}")

    print("\nRemaining positions:")
    positions = client.get_all_positions()
    for p in positions:
        print(f"  {p.symbol}: {p.qty} shares @ ${float(p.avg_entry_price):.2f}")


if __name__ == "__main__":
    main()
