# bitso_btc_websocket.py
# pip install websockets

import asyncio
import json
from decimal import Decimal, InvalidOperation
from collections import deque

import websockets

WS_URL = "wss://ws.bitso.com"
BOOK = "btc_usd"
PRINT_EVERY_SEC = 2
RECENT_TRADES = 10


def to_decimal(x):
    try:
        return Decimal(str(x))
    except (InvalidOperation, TypeError, ValueError):
        return None


class TopBookView:
    def __init__(self):
        self.bids = []   # [(price, amount), ...]
        self.asks = []   # [(price, amount), ...]
        self.trades = deque(maxlen=RECENT_TRADES)

    def apply_orders(self, msg):
        payload = msg.get("payload", {})

        # Bitso orders payload shape:
        # {
        #   "bids": [...],
        #   "asks": [...]
        # }
        bids_raw = payload.get("bids", [])
        asks_raw = payload.get("asks", [])

        bids = []
        asks = []

        for row in bids_raw:
            if not isinstance(row, dict):
                continue
            price = to_decimal(row.get("r"))
            amount = to_decimal(row.get("a"))
            if price is None or amount is None:
                continue
            bids.append((price, amount))

        for row in asks_raw:
            if not isinstance(row, dict):
                continue
            price = to_decimal(row.get("r"))
            amount = to_decimal(row.get("a"))
            if price is None or amount is None:
                continue
            asks.append((price, amount))

        self.bids = sorted(bids, key=lambda x: x[0], reverse=True)[:10]
        self.asks = sorted(asks, key=lambda x: x[0])[:10]

    def apply_trades(self, msg):
        payload = msg.get("payload", [])
        if not isinstance(payload, list):
            return

        for tr in payload:
            if not isinstance(tr, dict):
                continue

            taker_side = tr.get("t")
            if taker_side == 0:
                side = "buy"
            elif taker_side == 1:
                side = "sell"
            else:
                side = "unknown"

            self.trades.append({
                "id": tr.get("i"),
                "price": tr.get("r"),
                "amount": tr.get("a"),
                "value": tr.get("v"),
                "side": side,
                "ts": tr.get("x"),
            })

    def best_bid(self):
        return self.bids[0][0] if self.bids else None

    def best_ask(self):
        return self.asks[0][0] if self.asks else None


async def printer(state: TopBookView):
    while True:
        print("\n" + "=" * 100)
        print(f"BOOK: {BOOK}")

        bid = state.best_bid()
        ask = state.best_ask()

        if bid is not None and ask is not None:
            spread = ask - bid
            mid = (ask + bid) / Decimal("2")
            print(f"Best Bid: {bid} | Best Ask: {ask} | Spread: {spread} | Mid: {mid}")
        else:
            print("Waiting for book data...")

        print("\nTOP 10 BIDS")
        for px, amt in state.bids:
            print(f"{str(px):>16} | {amt}")

        print("\nTOP 10 ASKS")
        for px, amt in state.asks:
            print(f"{str(px):>16} | {amt}")

        print("\nRECENT TRADES")
        if not state.trades:
            print("No trades yet.")
        else:
            for tr in list(state.trades)[-10:]:
                print(
                    f"id={tr['id']} side={tr['side']} "
                    f"price={tr['price']} amount={tr['amount']} value={tr['value']} ts={tr['ts']}"
                )

        await asyncio.sleep(PRINT_EVERY_SEC)


async def main():
    state = TopBookView()

    while True:
        try:
            print(f"Connecting to {WS_URL} ...")
            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=20,
                max_size=2**20,
            ) as ws:
                print("Connected.")

                subs = [
                    {"action": "subscribe", "book": BOOK, "type": "orders"},
                    {"action": "subscribe", "book": BOOK, "type": "trades"},
                ]

                for sub in subs:
                    await ws.send(json.dumps(sub))
                    print(f"Subscribed: {sub}")

                print_task = asyncio.create_task(printer(state))

                try:
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            print(f"[WARN] Could not decode message: {raw}")
                            continue

                        if not isinstance(msg, dict):
                            print(f"[INFO] Non-dict message: {msg}")
                            continue

                        # subscription ack
                        if msg.get("action") == "subscribe":
                            print(f"[ACK] {msg}")
                            continue

                        # keepalive
                        if msg.get("type") == "ka":
                            continue

                        msg_type = msg.get("type")

                        if msg_type == "orders":
                            state.apply_orders(msg)
                        elif msg_type == "trades":
                            state.apply_trades(msg)
                        else:
                            print(f"[INFO] Other message: {msg}")

                finally:
                    print_task.cancel()
                    try:
                        await print_task
                    except asyncio.CancelledError:
                        pass

        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            print("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())