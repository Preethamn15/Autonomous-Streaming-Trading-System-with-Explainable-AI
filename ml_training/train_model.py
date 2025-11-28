# ml_training/train_model.py
import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

MODEL_OUTPUT_PATH = os.path.join("agent", "model.pkl")
SYMBOL = "RELIANCE.NS"   # change if you want
START_DATE = "2022-01-01"
END_DATE = "2025-01-01"

def compute_indicators(df):
    """Compute SMA, EMA, RSI, Volatility exactly like indicator engine."""
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # RSI 14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Volatility 20 (std dev of returns)
    df["volatility_20"] = df["Close"].pct_change().rolling(20).std()

    return df

def prepare_features(df):
    """Prepare ML dataset from indicators."""
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    features = df[["sma_5", "sma_20", "ema_12", "ema_26", "rsi_14", "volatility_20"]]
    target = df["target"]

    return features, target

def train_model():
    print("📥 Downloading historical data...")
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)

    if df.empty:
        print("❌ No historical data found! Check ticker or date range.")
        return

    print("📊 Computing indicators...")
    df = compute_indicators(df)

    print("🧪 Preparing training data...")
    X, y = prepare_features(df)

    print("🔀 Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("🤖 Training Logistic Regression model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("📈 Evaluating model...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model Accuracy: {acc:.2f}")

    print(f"💾 Saving model to {MODEL_OUTPUT_PATH} ...")
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)

    print("🎉 Model training completed successfully!")

if __name__ == "__main__":
    train_model()
