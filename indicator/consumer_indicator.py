# indicator/consumer_indicator.py
import os
import json
import time
from collections import deque

import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TICKS_TOPIC = os.getenv("KAFKA_TICKS_TOPIC", "ticks")       # FIXED
IND_TOPIC = os.getenv("KAFKA_IND_TOPIC", "indicators")      # FIXED
SYMBOL = os.getenv("SYMBOL", "RELIANCE.NS")

# ---- Kafka Consumer ----
consumer = KafkaConsumer(
    TICKS_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",          # IMPORTANT CHANGE
    enable_auto_commit=True
)

# ---- Kafka Producer ----
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Rolling window of last prices
window = deque(maxlen=200)

def compute_indicators(series: pd.Series):
    out = {}
    out["sma_5"] = series.tail(5).mean()
    out["sma_20"] = series.tail(20).mean()
    out["ema_12"] = series.ewm(span=12, adjust=False).mean().iloc[-1]
    out["ema_26"] = series.ewm(span=26, adjust=False).mean().iloc[-1]

    delta = series.diff().dropna()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()

    rs = (up / (down + 1e-9)).iloc[-1] if not up.empty else None
    out["rsi_14"] = 100 - (100 / (1 + rs)) if rs is not None else None

    out["volatility_20"] = series.pct_change().rolling(20).std().iloc[-1]
    return out

if __name__ == "__main__":
    print(f"📡 Indicator consumer listening on topic '{TICKS_TOPIC}'...")

    while True:
        msg_pack = consumer.poll(timeout_ms=1000)
        if not msg_pack:
            continue

        for _, msgs in msg_pack.items():
            for msg in msgs:
                data = msg.value

                quote = data.get("quote", {})
                price = quote.get("price")


                if price is None:
                    continue

                window.append(price)

                if len(window) < 20:
                    continue  # need enough history

                series = pd.Series(list(window))
                indicators = compute_indicators(series)

                payload = {
                    "symbol": SYMBOL,
                    "ts": int(time.time()),
                    "price": price,
                    "indicators": indicators
                }

                producer.send(IND_TOPIC, payload)
                producer.flush()

                print("✓ published indicators:", payload)
