# ============================================================================
#  AlgoPulse Dashboard — FINAL VERSION WITH:
#  - Local SHAP Waterfall
#  - Global SHAP Waterfall
#  - Global SHAP Heatmap (last 100 points)
#  - Static 92% Model Confidence
#  - Clean Layout (A1 placement)
# ============================================================================

import os
import time
import threading
import pickle
import math
from collections import deque, defaultdict

import json
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
import shap

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# ---------------- CONFIG ----------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TICKS_TOPIC = os.getenv("KAFKA_TICKS_TOPIC", "ticks")
IND_TOPIC = os.getenv("KAFKA_IND_TOPIC", "indicators")
ACTIONS_TOPIC = os.getenv("KAFKA_ACTION_TOPIC", "actions")
SYMBOL = os.getenv("SYMBOL", "RELIANCE.NS")

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent", "model.pkl")
MAX_POINTS = 500
POLL_INTERVAL_SECONDS = 1.0

FEATURE_NAMES = ["sma_5", "sma_20", "ema_12", "ema_26", "rsi_14", "volatility_20"]

# Appearance
BG = "#0f1724"
ACCENT_GREEN = "#00cc96"
ACCENT_RED = "#ff4d4d"
ACCENT_YELLOW = "#ffcc00"

# ---------------- IN-MEMORY STORE ----------------
ticks_deque = deque(maxlen=MAX_POINTS)
inds_deque = deque(maxlen=MAX_POINTS)
actions_deque = deque(maxlen=MAX_POINTS)
shap_history = deque(maxlen=MAX_POINTS)

# ---------------- MODEL + SHAP LOADER ----------------
ml_model = None
shap_explainer = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            ml_model = pickle.load(f)

        try:
            shap_explainer = shap.TreeExplainer(ml_model)
        except:
            shap_explainer = None

        print("✅ Model & SHAP loaded")

except:
    print("⚠ Model load error")

# ---------------- SAFE JSON PARSER ----------------
def json_loader(m):
    try:
        if isinstance(m, bytes):
            return json.loads(m.decode("utf-8"))
        if isinstance(m, str):
            return json.loads(m)
        return m
    except:
        return None

# ---------------- KAFKA CONSUMER THREAD ----------------
def start_consumer(topic, store):
    def worker():
        consumer = None
        for attempt in range(8):
            try:
                consumer = KafkaConsumer(
                    topic,
                    bootstrap_servers=KAFKA_BOOTSTRAP,
                    auto_offset_reset="latest",
                    value_deserializer=json_loader,
                    consumer_timeout_ms=1000
                )
                print(f"Connected to {topic}")
                break
            except:
                time.sleep(1)

        if consumer is None:
            print(f"❌ Could not connect: {topic}")
            return

        while True:
            try:
                msg_pack = consumer.poll(timeout_ms=500)
                for _, msgs in msg_pack.items():
                    for m in msgs:
                        if m.value:
                            store.append(m.value)

                            # SHAP
                            if topic == IND_TOPIC and shap_explainer:
                                ind = m.value.get("indicators", {})
                                features = [ind.get(f) for f in FEATURE_NAMES]

                                if None not in features:
                                    try:
                                        X = np.array(features).reshape(1, -1)
                                        shap_vals = shap_explainer.shap_values(X)
                                        vals = shap_vals[-1][0] if isinstance(shap_vals, list) else shap_vals[0]

                                        shap_history.append({
                                            "ts": m.value.get("ts"),
                                            "shap": dict(zip(FEATURE_NAMES, vals))
                                        })
                                    except:
                                        pass
            except Exception as e:
                print("Consumer error:", e)
                time.sleep(1)

    threading.Thread(target=worker, daemon=True).start()

# Start consumers
start_consumer(TICKS_TOPIC, ticks_deque)
start_consumer(IND_TOPIC, inds_deque)
start_consumer(ACTIONS_TOPIC, actions_deque)

# ---------------- DASH APP ----------------
app = dash.Dash(__name__, title="Autonomous Stream Trading with Explainable Signal Intelligence")
server = app.server

# UI helper card
def card(title, child, height=None):
    return html.Div(
        className="card",
        style={"height": height} if height else {},
        children=[
            html.Div(title, className="card-title"),
            child
        ]
    )

# ---------------- LAYOUT ----------------
app.layout = html.Div(
    className="main-container",
    children=[

        # HEADER
        html.Div(
            className="header",
            children=[
                html.H2("Autonomous Stream Trading with Explainable Signal Intelligence"),
                html.Div([
                    dcc.Input(id="symbol-input", value=SYMBOL, className="symbol-input"),
                    html.Button("Set", id="symbol-set", className="symbol-btn")
                ])
            ]
        ),

        # TOP SECTION
        html.Div(
            className="row",
            children=[

                # LEFT CHARTS
                html.Div(
                    className="col col-2-3",
                    children=[
                        card("Live Price & Indicators", dcc.Graph(id="price-chart"), "380px"),

                        html.Div(
                            className="row",
                            children=[
                                html.Div(card("RSI (14)", dcc.Graph(id="rsi-chart"), "180px"), className="col col-1-2"),
                                html.Div(card("Volatility (20)", dcc.Graph(id="vol-chart"), "180px"), className="col col-1-2"),
                            ]
                        )
                    ]
                ),

                # RIGHT PANEL
                html.Div(
                    className="col col-1-3",
                    children=[
                        card("Agent Decision", html.Div(id="agent-panel"), "300px"),
                        card("ML Confidence", dcc.Graph(id="ml-gauge"), "170px"),
                        card("Local SHAP (Waterfall)", dcc.Graph(id="shap-local"), "260px")
                    ]
                ),
            ]
        ),

        # ACTIONS
        html.Div(
            className="row",
            children=[
                html.Div(card("Recent Actions", html.Div(id="actions-table", className="actions-box"), height="350px"), className="col col-1-2"),
                html.Div(card("Action Distribution", dcc.Graph(id="action-dist"), "230px"), className="col col-1-2"),
            ]
        ),

        # GLOBAL SHAP SUMMARY
        html.Div(card("Global SHAP Summary (Waterfall)", dcc.Graph(id="shap-summary"), "350px")),

        # GLOBAL SHAP HEATMAP — NEW CARD (A1)
        html.Div(card("Global SHAP Heatmap (Last 100 Points)", dcc.Graph(id="shap-heatmap"), "400px")),

        dcc.Interval(id="live-interval", interval=2000)
    ]
)

# ---------------- CALLBACKS ----------------

@app.callback(
    Output("symbol-input", "value"),
    Input("symbol-set", "n_clicks"),
    Input("symbol-input", "value")
)
def update_symbol(n, v):
    global SYMBOL
    SYMBOL = v
    return v

# ---- PRICE, RSI, VOL ----
@app.callback(
    Output("price-chart", "figure"),
    Output("rsi-chart", "figure"),
    Output("vol-chart", "figure"),
    Input("live-interval", "n_intervals")
)
def update_price(n):

    df = pd.DataFrame(list(inds_deque))
    if df.empty:
        return go.Figure(), go.Figure(), go.Figure()

    df["datetime"] = (
        pd.to_datetime(df["ts"], unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Asia/Kolkata")
    )

    df = df.sort_values("datetime")

    # PRICE
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["price"], mode="lines",
        line=dict(color=ACCENT_GREEN)
    ))

    if "indicators" in df.columns:
        ind = pd.json_normalize(df["indicators"])
        for col in ["sma_5", "sma_20", "ema_12", "ema_26"]:
            if col in ind.columns:
                price_fig.add_trace(go.Scatter(
                    x=df["datetime"], y=ind[col], mode="lines",
                    line=dict(dash="dot"), name=col
                ))

    price_fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,font=dict(color="white"), xaxis=dict(color="white"), yaxis=dict(color="white"),)

    # RSI
    rsi_fig = go.Figure()
    if "indicators" in df.columns:
        ind = pd.json_normalize(df["indicators"])
        if "rsi_14" in ind:
            rsi_fig.add_trace(go.Scatter(x=df["datetime"], y=ind["rsi_14"], mode="lines", line=dict(color="yellow")))
            rsi_fig.add_hline(70, line_dash="dash", line_color=ACCENT_RED)
            rsi_fig.add_hline(30, line_dash="dash", line_color=ACCENT_GREEN)

    rsi_fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,font=dict(color="white"), xaxis=dict(color="white"), yaxis=dict(color="white"),)

    # VOL
    vol_fig = go.Figure()
    if "indicators" in df.columns:
        ind = pd.json_normalize(df["indicators"])
        if "volatility_20" in ind:
            vol_fig.add_trace(go.Bar(x=df["datetime"], y=ind["volatility_20"], marker_color="#9bd1ff"))

    vol_fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="white"), xaxis=dict(color="white"), yaxis=dict(color="white"),)

    return price_fig, rsi_fig, vol_fig
# ---------------- WHY-NOT EXPLANATIONS (FULLY SAFE VERSION) ----------------
def generate_why_not(latest, shap_vals):
    action = latest.get("action", "wait").lower()

    # Always create all keys first to avoid KeyError
    reasons = {
        "why_not_buy": [],
        "why_not_sell": [],
        "why_not_wait": []
    }

    rsi = shap_vals.get("rsi_14", 0)
    vol = shap_vals.get("volatility_20", 0)
    ema12 = shap_vals.get("ema_12", 0)
    ema26 = shap_vals.get("ema_26", 0)

    # ---------------- WHY NOT BUY ----------------
    if action != "buy":
        if rsi > 0:
            reasons["why_not_buy"].append("RSI contributing negatively toward BUY (overbought).")
        if vol < 0:
            reasons["why_not_buy"].append("High volatility made BUY risky.")
        if ema12 < ema26:
            reasons["why_not_buy"].append("Short-term EMA below long-term EMA → bearish crossover.")

        if not reasons["why_not_buy"]:
            reasons["why_not_buy"].append("Model confidence did not justify BUY.")

    # ---------------- WHY NOT SELL ----------------
    if action != "sell":
        if rsi < 0:
            reasons["why_not_sell"].append("RSI contributing positively → SELL not ideal.")
        if ema12 > ema26:
            reasons["why_not_sell"].append("Bullish EMA crossover prevented SELL.")
        if vol > 0.1:
            reasons["why_not_sell"].append("Low volatility → no strong SELL signal.")

        if not reasons["why_not_sell"]:
            reasons["why_not_sell"].append("Model confidence did not justify SELL.")

    # ---------------- WHY NOT WAIT ----------------
    if action != "wait":
        if abs(rsi) < 0.01:
            reasons["why_not_wait"].append("RSI neutral; WAIT could also be valid.")
        if vol < 0.02:
            reasons["why_not_wait"].append("Low volatility suggests WAIT.")

        if not reasons["why_not_wait"]:
            reasons["why_not_wait"].append("WAIT was not chosen because other signals were stronger.")

    return reasons


# ---- ACTION + SHAP ----
@app.callback(
    Output("agent-panel", "children"),
    Output("ml-gauge", "figure"),
    Output("shap-local", "figure"),
    Output("actions-table", "children"),
    Output("action-dist", "figure"),
    Output("shap-summary", "figure"),
    Output("shap-heatmap", "figure"),   # <-- if you added heatmap
    Input("live-interval", "n_intervals")
)

def update_actions(n):
    why_not = {
        "why_not_buy": ["Not enough data"],
        "why_not_sell": ["Not enough data"],
        "why_not_wait": ["Not enough data"]
    }


    # ---------------- ACTION TABLE ----------------
    acts = list(actions_deque)[-15:][::-1]
    rows = []
    dist = defaultdict(int)

    for a in acts:
        act = a.get("action", "")
        dist[act] += 1

        rows.append(
            html.Div(className="action-row", children=[
                html.Span(act.upper(), className=f"tag tag-{act}"),
                html.Span(f"{a.get('symbol')} @ {a.get('price',0):.2f}"),
                html.Div(a.get("reason",""), className="reason")
            ])
        )

    latest = acts[0] if acts else None
    # ---- UPDATE WHY-NOT BEFORE BUILDING PANEL ----
    if shap_history and latest:
        last = shap_history[-1]
        vals = last["shap"]
        why_not = generate_why_not(latest, vals)


    # ---------------- AGENT PANEL ----------------
    # ---------------- AGENT PANEL ----------------
    if latest:
        agent_ui = [
            html.Div([
                html.Div([
                    "Latest Action: ",
                    html.Span(latest["action"].upper(), className=f"tag tag-{latest['action']}")
                ]),
                html.Div(f"Symbol: {latest.get('symbol')}"),
                html.Div(f"Price: {latest.get('price',0):.2f}"),
            ], className="agent-section"),

            html.Div("Why NOT BUY:", className="why-title"),
            html.Ul([html.Li(w) for w in why_not["why_not_buy"]], className="agent-panel-list"),

            html.Div("Why NOT SELL:", className="why-title"),
            html.Ul([html.Li(w) for w in why_not["why_not_sell"]], className="agent-panel-list"),

            html.Div("Why NOT WAIT:", className="why-title"),
            html.Ul([html.Li(w) for w in why_not["why_not_wait"]], className="agent-panel-list"),
        ]

    else:
        agent_ui = "No actions yet"


    # ---------------- STATIC GAUGE 92% ----------------
    # ---------------- STATIC GAUGE 92% (Clean Dark Theme) ----------------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=92,
        number={
            "suffix": "%",
            "font": {"size": 60, "color": "white"}
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "rgba(0,0,0,0)",   # hide tick labels
                "ticks": ""                    # remove ticks
            },
            "bar": {"color": ACCENT_GREEN, "thickness": 0.25},
            "bgcolor": BG,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": ACCENT_RED},
                {"range": [30, 70], "color": ACCENT_YELLOW},
                {"range": [70, 100], "color": ACCENT_GREEN}
            ]
        }
    ))

    gauge.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=10, r=10, t=10, b=10),
    )


    # ---------------- LOCAL SHAP (Waterfall) ----------------
    
    # ---------------- LOCAL SHAP (Waterfall) ----------------
    if shap_history:
        last = shap_history[-1]
        vals = last["shap"]
        items = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)
        names = [i[0] for i in items]
        values = [i[1] for i in items]
        colors = ["#00cc96" if v > 0 else "#ff6b6b" for v in values]

        shap_local_fig = go.Figure()

        shap_local_fig.add_trace(go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color=colors, line=dict(color="white", width=1.2)),
        ))

        shap_local_fig.update_layout(
            title="Local SHAP Contribution (Last Tick)",
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(color="white"),
            margin=dict(l=120, r=40, t=40, b=40),
            xaxis=dict(
                zeroline=True,
                zerolinecolor="white",
                zerolinewidth=1.5,
                gridcolor="#23344a"
            ),
        )
    else:
        shap_local_fig = go.Figure()



    # ---------------- ACTION DISTRIBUTION PIE ----------------
    dist_fig = go.Figure(go.Pie(labels=list(dist.keys()), values=list(dist.values()), hole=0.5))
    dist_fig.update_layout(paper_bgcolor=BG)

    # ---------------- GLOBAL SHAP WATERFALL ----------------
    # ---------------- GLOBAL SHAP SUMMARY DOT PLOT ----------------
    if shap_history:
        df = pd.DataFrame([{**s["shap"]} for s in shap_history])

        mean_vals = df.mean().sort_values()
        features = mean_vals.index.tolist()

        # Each SHAP row flattened for scatter
        xs = []
        ys = []
        cs = []

        for i, f in enumerate(features):
            for v in df[f].values:
                xs.append(v)
                ys.append(f)
                cs.append(v)

        shap_summary_fig = go.Figure()

        shap_summary_fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                color=cs,
                colorscale="Viridis",
                size=7,
                line=dict(color="white", width=0.4),
                colorbar=dict(title="SHAP Value"),
            )
        ))

        shap_summary_fig.update_layout(
            title="Global SHAP Summary (Dot Plot)",
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(color="white"),
            margin=dict(l=120, r=40, t=40, b=40),
            xaxis=dict(
                title="SHAP Value Impact",
                zeroline=True,
                zerolinecolor="white",
                zerolinewidth=1.5,
                gridcolor="#23344a",
            ),
            yaxis=dict(
                title="Features",
                categoryorder="array",
                categoryarray=features
            )
        )
    else:
        shap_summary_fig = go.Figure()


    # ---------------- GLOBAL SHAP HEATMAP (Last 100 Points) ----------------
    # ---------------- GLOBAL SHAP HEATMAP (Improved Professional Version) ----------------
# ---------------- GLOBAL SHAP HEATMAP (Improved Professional Version) ----------------
    if shap_history:
        heatmap_data = list(shap_history)[-100:]

        df_hm = pd.DataFrame([{**s["shap"]} for s in heatmap_data])

        # If data is too small, pad rows to avoid blocky graph
        if df_hm.shape[0] < 20:
            df_hm = df_hm.reindex(range(20), method='pad')

        heatmap_fig = go.Figure(data=go.Heatmap(
            z=df_hm.values,
            x=df_hm.columns,
            y=list(range(len(df_hm))),
            colorscale="Viridis",
            zsmooth="best",
            colorbar=dict(
                title=dict(
                    text="SHAP Value",
                    font=dict(color="white")
                ),
                tickfont=dict(color="white")
            ),
        ))

        heatmap_fig.update_layout(
            title="Global SHAP Heatmap (Last 100 Points)",
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(color="white"),
            margin=dict(t=50, l=60, r=40, b=40),
            xaxis=dict(
                tickangle=0,
                showgrid=False,
                tickfont=dict(size=12, color="white")
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(size=10, color="white"),
                title="Time (Recent → Older)"
            )
        )
    else:
        heatmap_fig = go.Figure()



    return agent_ui, gauge, shap_local_fig, rows, dist_fig, shap_summary_fig, heatmap_fig



# ---------------- RUN ----------------
if __name__ == "__main__":
    print("🚀 Dashboard: http://127.0.0.1:8050")
    app.run(host="0.0.0.0", port=8050, debug=False)
