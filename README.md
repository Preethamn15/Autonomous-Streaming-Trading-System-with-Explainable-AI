# Autonomous Stream Trading with Explainable AI (TheoryEL)

This project implements a fully autonomous, real-time trading decision platform using
Machine Learning, Streaming Data Pipelines, Technical Indicators, and Explainable AI (XAI).
It consumes live market data, generates indicators, makes ML-based BUY/SELL/WAIT decisions,
and displays them in a live dashboard.

The system is built as a distributed, containerized microservice architecture using Kafka,
Python, Random Forest, SHAP explainability, and a Web Dashboard (Dash + Plotly).

---

## Key Features

- **Real-time data ingestion** using Kafka producers  
- **Streaming indicator engine** computing RSI, SMA, EMA, Volatility  
- **ML-powered hybrid trading agent** using Random Forest  
- **Explainable AI** (SHAP values) for decisions  
- **Interactive dashboard** showing:
  - Live prices
  - Indicators
  - Agent decision
  - Why BUY / SELL / WAIT explanation  
- **Docker-based microservice orchestration**

---

## System Architecture

The platform is divided into 5 major components:

1. **Producer**  
   Fetches live market price every second and pushes it to Kafka topic `price_stream`.

2. **Indicator Engine**  
   Consumes live prices → computes technical indicators → sends to `indicator_stream`.

3. **ML Agent**  
   Loads trained Random Forest model → predicts BUY / SELL / WAIT  
   + generates explanations.

4. **Dashboard (Dash/Plotly)**  
   Real-time dashboard displaying price, indicators, and ML agent output.

5. **Model Training Module**  
   Offline training script that generates `model.pkl`.

---

# Hybrid Trading Decision Agent

An autonomous trading agent that makes real-time BUY/SELL/HOLD/WAIT decisions using a **Hybrid Rule + Machine Learning (Random Forest)** approach.

## 🔹 Features
- Real-time decision-making (Kafka streaming)
- Hybrid logic: EMA, SMA, RSI, Volatility + Random Forest
- Safe WAIT mode for high-risk conditions (conflict, volatility, sideways, ML uncertainty)
- Explainable decisions (XAI): reasons, ML probability, rule signals
- Dynamic position sizing based on confidence
- Tracks cash, position, entry price, P/L

## 🔹 Indicators Used
- **EMA 12/26** – trend crossover (buy/sell)
- **SMA 5/20** – trend + sideways detection
- **RSI 14** – overbought/oversold momentum
- **Volatility 20** – risk & instability filter

## 🔹 ML Model
- Random Forest classifier  
- Features: SMA5, SMA20, EMA12, EMA26, RSI14, Volatility20  
- Outputs: trend label + confidence

## 🔹 Decision Flow
1. Read indicators from Kafka  
2. Apply rule-based signals  
3. Run ML prediction  
4. Check conflicts, volatility, sideways  
5. Final decision → BUY/SELL/HOLD/WAIT  
6. Send result to Kafka dashboard  

## 🔹 Why Hybrid?
- Rules provide stability  
- ML captures non-linear patterns  
- WAIT logic prevents false trades  
- XAI ensures transparency  

---

# 📸 System Visuals & UI Showcase

This section presents all project images in one clean, organized gallery, making the documentation visually appealing and easy to understand.

---

## Architecture Diagram

<p align="center">
  <img src="architecture.png" width="750">
</p>

---

## Dashboard Overview

### 🔷 Live Price & Indicators
<p align="center">
  <img src="dashboard_price.png" width="800">
</p>

---

### Agent Decision Panel
<p align="center">
  <img src="dashboard_decision.png" width="750">
</p>

---

### ML Confidence Gauge
<p align="center">
  <img src="dashboard_confidence.png" width="750">
</p>

---

### SHAP Explainability (Local Waterfall)
<p align="center">
  <img src="dashboard_shap.png" width="750">
</p>

---

### Indicator Panels (RSI & Volatility)
<p align="center">
  <img src="dashboard_rsi_vol.png" width="800">
</p>

---

### Action Distribution (Donut / Pie)
<p align="center">
  <img src="dashboard_action_distribution.png" width="750">
</p>

---

### Recent Actions Log
<p align="center">
  <img src="dashboard_recent_actions.png" width="800">
</p>

---


# 🔥 HOW TO RUN THIS PROJECT ON YOUR LAPTOP  
**(Using Your Exact Commands — Windows + PowerShell)**

This is the official way to run your entire system.

---

# **STEP 1 — Start Kafka & Zookeeper**

Open Terminal (VS Code or PowerShell): docker-compose up -d

This starts:

- Zookeeper  
- Kafka Broker  

---

# STEP 2 — Start Producer (Terminal 1)

$env:FINNHUB_API_KEY="API_KEY_REPLACE_WITH_YOURS"
$env:SYMBOL="RELIANCE.NS"
$env:KAFKA_BOOTSTRAP="localhost:9092"

.\venv\Scripts\Activate.ps1

python producer/producer.py


---

# STEP 3 — Start Indicator Consumer (Terminal 2)

$env:KAFKA_BOOTSTRAP="localhost:9092"
$env:KAFKA_TICKS_TOPIC="ticks"
$env:KAFKA_IND_TOPIC="indicators"
$env:SYMBOL="RELIANCE.NS"

.\venv\Scripts\Activate.ps1

python indicator/consumer_indicator.py


---

# **STEP 4 — Start ML Agent (Terminal 3)**
$env:KAFKA_BOOTSTRAP="localhost:9092"

$env:IND_TOPIC="indicators"

$env:SYMBOL="RELIANCE.NS"

.\venv\Scripts\Activate.ps1
python agent/agent.py


---

# **STEP 5 — Start Dashboard (Terminal 4)**
.\venv\Scripts\Activate.ps1
cd dashboard
python app.py


Dashboard will open at: http://localhost:8050


---



# Training the Model
**open terminal:**
cd ml_training
python train_model.py

This outputs:model.pkl

**Copy this to:** agent/model.pkl

---

# How the Pipeline Flows

Producer → Kafka (ticks) → Indicator Engine → Kafka (indicators) → ML Agent → Kafka (actions) → Dashboard

Real-time decisions:

✔ BUY  
✔ SELL  
✔ WAIT (used during conflict or ML uncertainty)

---

# Explainability (XAI)

**The agent provides:**
- Indicator contributions  
- SHAP values  
- Reasons like:

  - “RSI negatively pushing BUY (overbought)”  
  - “SMA cross-over supports BUY”  
  - “Volatility high → wait”






