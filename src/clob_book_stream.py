"""Maintain Polymarket CLOB asks from full books and absolute level updates."""

import json
from threading import Lock
from time import monotonic

from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

from .inplay_scan import MAX_AGE, as_num

MARKET_SOCKET = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class ClobBookStream:
    def __init__(self):
        self.lock = Lock()
        self.books = {}
        self.minimums = {}
        self.revision = 0

    def observe_market_terms(self, books):
        with self.lock:
            for token, book in books.items():
                minimum = as_num(book.get("min_order_size"))
                if minimum is not None and minimum > 0:
                    self.minimums[token] = minimum

    def snapshot(self, tokens):
        with self.lock:
            return {token: {**self.books[token], "min_order_size": self.minimums.get(token)}
                    for token in tokens if token in self.books}

    def apply(self, message, subscribed):
        if not isinstance(message, dict):
            return
        kind = message.get("event_type")
        stamp = as_num(message.get("timestamp"))
        with self.lock:
            changed = False
            if kind == "market_resolved":
                for token in message.get("assets_ids") or []:
                    if token in subscribed:
                        self.books[token] = {"timestamp": stamp, "asks": [], "closed": True}
                        changed = True
            elif stamp is not None and kind == "book":
                token = message.get("asset_id")
                previous = self.books.get(token)
                if (token in subscribed and isinstance(message.get("asks"), list)
                        and (previous is None or not previous.get("closed") and stamp >= previous["timestamp"])):
                    self.books[token] = {"timestamp": stamp, "asks": message["asks"]}
                    minimum = as_num(message.get("min_order_size"))
                    if minimum is not None and minimum > 0:
                        self.minimums[token] = minimum
                    changed = True
            elif stamp is not None and kind == "price_change":
                changes = {}
                for level in message.get("price_changes") or []:
                    token = level.get("asset_id")
                    previous = self.books.get(token)
                    price = as_num(level.get("price"))
                    size = as_num(level.get("size"))
                    if (token in subscribed and previous and not previous.get("closed")
                            and stamp >= previous["timestamp"] and level.get("side") == "SELL"
                            and price is not None and 0 < price < 1 and size is not None and size >= 0):
                        if token not in changes:
                            changes[token] = {as_num(row["price"]): row for row in previous["asks"]}
                        asks = changes[token]
                        if size == 0:
                            asks.pop(price, None)
                        else:
                            asks[price] = {"price": price, "size": size}
                for token, asks in changes.items():
                    self.books[token] = {"timestamp": stamp, "asks": list(asks.values())}
                    changed = True
            if changed:
                self.revision += 1

    def follow(self, stop, wanted):
        while not stop.is_set():
            tokens = wanted()
            if not tokens:
                stop.wait(0.5)
                continue
            try:
                with connect(MARKET_SOCKET, open_timeout=8, close_timeout=1, max_size=8 * 1024 * 1024) as socket:
                    socket.send(json.dumps({"assets_ids": sorted(tokens), "type": "market",
                                            "custom_feature_enabled": True}))
                    updated = last_message = last_ping = monotonic()
                    while not stop.is_set():
                        reference = monotonic()
                        if reference - last_message > MAX_AGE:
                            break
                        if reference - last_ping >= 10:
                            socket.send("PING")
                            last_ping = reference
                        if reference - updated >= 1:
                            desired = wanted()
                            for operation, delta in (("unsubscribe", tokens - desired), ("subscribe", desired - tokens)):
                                if delta:
                                    socket.send(json.dumps({"assets_ids": sorted(delta), "operation": operation,
                                                            "custom_feature_enabled": True}))
                            with self.lock:
                                for token in tokens - desired:
                                    self.books.pop(token, None)
                                    self.minimums.pop(token, None)
                            tokens, updated = desired, reference
                            if not tokens:
                                break
                        try:
                            frame = socket.recv(timeout=0.25)
                        except TimeoutError:
                            continue
                        last_message = monotonic()
                        if frame == "PONG":
                            continue
                        data = json.loads(frame)
                        for message in data if isinstance(data, list) else [data]:
                            self.apply(message, tokens)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, WebSocketException):
                pass
            finally:
                with self.lock:
                    self.books = {token: book for token, book in self.books.items() if book.get("closed")}
                    self.revision += 1
            stop.wait(2)
