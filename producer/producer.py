import os
import time
import json
import yfinance as yf
from kafka import KafkaProducer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("KAFKA_TICKS_TOPIC", "ticks")
SYMBOL = os.getenv("SYMBOL", "RELIANCE.NS")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))  # seconds

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    api_version=(0, 10),
)

def fetch_latest_quote(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d", interval="1m")

    if data.empty:
        raise Exception("No data returned from Yahoo Finance")

    last = data.iloc[-1]

    return {
        "price": float(last["Close"]),
        "open": float(last["Open"]),
        "high": float(last["High"]),
        "low": float(last["Low"]),
        "volume": int(last["Volume"]),
    }

if __name__ == "__main__":
    print(f"✅ Producing Yahoo Finance ticks for {SYMBOL} → topic '{TOPIC}'")

    while True:
        try:
            quote = fetch_latest_quote(SYMBOL)

            msg = {
                "symbol": SYMBOL,
                "ts": int(time.time()),
                "quote": quote,
            }

            producer.send(TOPIC, msg)
            producer.flush()

            print("→ sent:", msg)

        except Exception as e:
            print("❌ Error:", e)

        time.sleep(POLL_INTERVAL)
