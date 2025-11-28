# agent/agent.py
import os
import json
import time
import math
import pickle
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

# ----------- CONFIG -----------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
IND_TOPIC = os.getenv("IND_TOPIC", "indicators")
ACTION_TOPIC = os.getenv("ACTION_TOPIC", "actions")
SYMBOL = os.getenv("SYMBOL", "RELIANCE.NS")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")  # optional ML model
VOL_THRESHOLD = float(os.getenv("VOL_THRESHOLD", "0.02"))  # e.g., 2% std dev threshold
SIDEWAYS_PCT = float(os.getenv("SIDEWAYS_PCT", "0.005"))  # 0.5% difference for sideways
MIN_CONF_WAIT = 0.45   # if ml_prob in (0.45,0.55) => uncertain
MAX_POSITION_UNITS = int(os.getenv("MAX_POSITION_UNITS", "50"))
BASE_CASH = float(os.getenv("BASE_CASH", "10000.0"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.03"))  # optional stop-loss 3%
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.06"))  # take profit 6%
# -------------------------------

# Load model if present
ml_model = None
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            ml_model = pickle.load(f)
        print("ML model loaded from", MODEL_PATH)
    else:
        print("⚠ No ML model found at", MODEL_PATH, "- running rule-only mode.")
except Exception as e:
    print("⚠ Failed loading ML model:", e)
    ml_model = None

# Kafka consumer & producer
consumer = KafkaConsumer(
    IND_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True
)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Portfolio state
cash = BASE_CASH
position = 0
entry_price = None
last_action_ts = 0

def safe_get(d, key, default=None):
    v = d.get(key, default)
    try:
        return float(v) if v is not None else default
    except:
        return default

def ml_predict(indicators):
    """Return (label, prob) or (None, None) if model missing or error."""
    if ml_model is None:
        return None, None
    try:
        features = [
            safe_get(indicators, "sma_5"),
            safe_get(indicators, "sma_20"),
            safe_get(indicators, "ema_12"),
            safe_get(indicators, "ema_26"),
            safe_get(indicators, "rsi_14"),
            safe_get(indicators, "volatility_20")
        ]
        # If any feature missing, skip ML
        if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in features):
            return None, None
        X = np.array(features).reshape(1, -1)
        prob = float(ml_model.predict_proba(X)[0][1])  # prob of uptrend
        label = 1 if prob >= 0.5 else 0
        return label, prob
    except Exception as e:
        print("ML prediction error:", e)
        return None, None

def rule_decision(indicators):
    """Simple EMA crossover + RSI rules."""
    ema12 = safe_get(indicators, "ema_12")
    ema26 = safe_get(indicators, "ema_26")
    rsi = safe_get(indicators, "rsi_14", 50)

    if ema12 is None or ema26 is None:
        return "hold"

    # Classic rule-based signals
    if ema12 > ema26 and rsi < 70:
        return "buy"
    if ema12 < ema26 and rsi > 30:
        return "sell"
    return "hold"

def detect_conflict(indicators, rule_dec):
    """Return True if indicators conflict (e.g., EMA says buy but RSI suggests sell)."""
    ema12 = safe_get(indicators, "ema_12")
    ema26 = safe_get(indicators, "ema_26")
    rsi = safe_get(indicators, "rsi_14", 50)

    # conflict if EMA crossover says buy but RSI indicates overbought (sell), or vice versa
    if ema12 is None or ema26 is None:
        return False
    ema_signal = "buy" if ema12 > ema26 else "sell"
    if ema_signal != rule_dec and ((ema_signal == "buy" and rsi > 65) or (ema_signal == "sell" and rsi < 35)):
        return True
    return False

def detect_sideways(indicators):
    """Detect sideways market (SMA5 close to SMA20 within SIDEWAYS_PCT)."""
    sma5 = safe_get(indicators, "sma_5")
    sma20 = safe_get(indicators, "sma_20")
    if sma5 is None or sma20 is None:
        return False
    if abs(sma5 - sma20) / max(1e-9, sma20) < SIDEWAYS_PCT:
        return True
    return False

def decide_final(indicators):
    """
    Hybrid decision combining rule, ML, volatility, conflict, and market regime.
    Returns: action, reason, wait_days, ml_label, ml_prob, confidence_for_sizing
    """
    rule = rule_decision(indicators)
    ml_label, ml_prob = ml_predict(indicators)

    volatility = safe_get(indicators, "volatility_20", 0.0)
    conflict = detect_conflict(indicators, rule)
    sideways = detect_sideways(indicators)

    reason_parts = []
    wait_days = 0

    # Determine uncertainty
    uncertain_ml = False
    if ml_prob is None:
        uncertain_ml = True  # treat as less confident when no ML
    else:
        if MIN_CONF_WAIT < ml_prob < (1 - MIN_CONF_WAIT):
            uncertain_ml = True

    # Base decision: try to align rule + ML when possible
    action = rule

    # If ML exists and agrees strongly, prefer ML if high confidence
    if ml_label is not None and ml_prob is not None:
        if ml_prob >= 0.7:
            action = "buy"
            reason_parts.append(f"ML strong up (p={ml_prob:.2f})")
        elif ml_prob <= 0.3:
            action = "sell"
            reason_parts.append(f"ML strong down (p={ml_prob:.2f})")
        else:
            # medium confidence -> follow rule but mark uncertainty
            reason_parts.append(f"ML weak (p={ml_prob:.2f})")

    # Now evaluate WAIT conditions (use ALL conditions)
    if conflict:
        reason_parts.append("Indicator conflict")
        wait_days = max(wait_days, 2)
    if uncertain_ml:
        reason_parts.append("ML uncertainty")
        wait_days = max(wait_days, 1)
    if volatility and volatility > VOL_THRESHOLD:
        reason_parts.append(f"High volatility ({volatility:.3f})")
        wait_days = max(wait_days, 2)
    if sideways:
        reason_parts.append("Sideways market")
        wait_days = max(wait_days, 1)

    # If wait_days > 0 decide to WAIT rather than act immediately
    if wait_days > 0:
        final_action = "wait"
        reason = "; ".join(reason_parts) if reason_parts else "Wait condition"
        confidence_for_sizing = ml_prob if ml_prob is not None else 0.5
        return final_action, reason, wait_days, ml_label, ml_prob, confidence_for_sizing

    # No wait: finalize BUY/SELL/HOLD
    final_action = action
    reason = "; ".join(reason_parts) if reason_parts else "Rule/ML agreement"
    confidence_for_sizing = ml_prob if ml_prob is not None else 0.5
    return final_action, reason, wait_days, ml_label, ml_prob, confidence_for_sizing

def compute_order_size(confidence, price):
    """Compute units to trade based on confidence and available cash."""
    global cash
    # Map confidence [0.0,1.0] to units [1, MAX_POSITION_UNITS]
    units = 1
    if confidence is None:
        confidence = 0.5
    # scale: 0.5->~5 units, 1.0->MAX_POSITION_UNITS
    units = int(max(1, min(MAX_POSITION_UNITS, math.ceil(confidence * MAX_POSITION_UNITS))))
    # Respect cash
    max_affordable = int(cash // max(1e-6, price))
    units = min(units, max_affordable)
    return units

print(f" Hybrid Agent started. Listening to '{IND_TOPIC}', publishing to '{ACTION_TOPIC}'...")

try:
    for msg in consumer:
        rec = msg.value
        price = safe_get(rec, "price")
        inds = rec.get("indicators", {})

        # Sanity check
        if price is None or inds is None:
            print("Skipping message with missing data:", rec)
            continue

        action, reason, wait_days, ml_label, ml_prob, confidence = decide_final(inds)

        event = {
            "symbol": SYMBOL,
            "price": price,
            "action": action,
            "reason": reason,
            "wait_days": int(wait_days),
            "rule": rule_decision(inds),
            "ml_label": int(ml_label) if ml_label is not None else None,
            "ml_prob": float(ml_prob) if ml_prob is not None else None,
            "ts": int(time.time())
        }

        # Execution logic

        if action == "wait":
            # Publish wait recommendation (no trade)
            print(f"WAIT for {wait_days} day(s): {reason} | price={price:.2f}")
            # Optionally include suggested re-check timestamp
            event["suggest_recheck_ts"] = int(time.time()) + wait_days * 24 * 3600
            producer.send(ACTION_TOPIC, event)

        elif action == "buy":
            size = compute_order_size(confidence, price)
            if size <= 0:
                print("✖ Not enough cash to buy any unit.")
                event["qty"] = 0
                event["cash"] = cash
                producer.send(ACTION_TOPIC, event)
            else:
                # buy size units
                cash -= size * price
                position += size
                entry_price = price if entry_price is None else entry_price
                event["qty"] = size
                event["cash"] = cash
                print(f" BUY {size} @ {price:.2f} | cash={cash:.2f} | ml_prob={ml_prob}")
                producer.send(ACTION_TOPIC, event)

        elif action == "sell":
            if position <= 0:
                print("✖ No position to sell.")
                event["qty"] = 0
                event["cash"] = cash
                producer.send(ACTION_TOPIC, event)
            else:
                # simple: sell all or size based on confidence
                size = position  # sell full
                cash += size * price
                event["qty"] = size
                event["cash"] = cash
                print(f" SELL {size} @ {price:.2f} | cash={cash:.2f} | ml_prob={ml_prob}")
                position = 0
                entry_price = None
                producer.send(ACTION_TOPIC, event)

        else:  # hold
            event["qty"] = 0
            event["cash"] = cash
            print(f" HOLD @ {price:.2f} | reason: {reason}")
            producer.send(ACTION_TOPIC, event)

        producer.flush()
        last_action_ts = int(time.time())

except KeyboardInterrupt:
    print("\n Agent stopped by user.")
finally:
    consumer.close()
