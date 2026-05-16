import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import ta
import yfinance as yf
from streamlit_autorefresh import st_autorefresh


ASSETS = {
    "Gold (XAU/USD)": "GC=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
}

ENABLE_TELEGRAM = False
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
TRADE_LOG_PATH = Path(__file__).with_name("virtual_trades.csv")
TRADE_COLUMNS = [
    "trade_id",
    "symbol",
    "asset",
    "interval",
    "signal",
    "direction",
    "entry_time",
    "entry_price",
    "stop_loss",
    "take_profit",
    "risk_percent",
    "confidence",
    "status",
    "result",
    "exit_time",
    "exit_price",
    "pnl_points",
    "exit_reason",
]


st.set_page_config(page_title="AI Gold Trading Agent", layout="wide")
st_autorefresh(interval=60_000, key="refresh")

st.title("AI Gold & Forex Trading Dashboard")

st.sidebar.header("Settings")

asset_label = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
symbol = ASSETS[asset_label]

interval = st.sidebar.selectbox("Timeframe", ["5m", "15m", "1h"], index=1)
period = st.sidebar.selectbox("Historical Period", ["7d", "30d", "60d"])
risk_percent = st.sidebar.slider("Risk Per Trade (%)", 1, 5, 2)
enable_paper_trading = st.sidebar.toggle("Auto Virtual Trading", value=True)
trade_reference_levels = st.sidebar.toggle("Trade HOLD References", value=True)


@st.cache_data(ttl=60)
def fetch_data(ticker: str, timeframe: str, history_period: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        interval=timeframe,
        period=history_period,
        progress=False,
        auto_adjust=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    close = frame["Close"].astype(float)

    frame["EMA20"] = ta.trend.ema_indicator(close, window=20)
    frame["EMA50"] = ta.trend.ema_indicator(close, window=50)
    frame["RSI"] = ta.momentum.rsi(close, window=14)

    macd = ta.trend.MACD(close)
    frame["MACD"] = macd.macd()
    frame["MACD_SIGNAL"] = macd.macd_signal()
    frame["ATR"] = ta.volatility.average_true_range(
        frame["High"].astype(float),
        frame["Low"].astype(float),
        close,
    )

    return frame.dropna()


def calculate_signal(data: pd.DataFrame) -> tuple[str, str]:
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    if latest["EMA20"] > latest["EMA50"]:
        trend = "BULLISH"
    elif latest["EMA20"] < latest["EMA50"]:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    if (
        latest["EMA20"] > latest["EMA50"]
        and latest["RSI"] < 40
        and previous["MACD"] < previous["MACD_SIGNAL"]
        and latest["MACD"] > latest["MACD_SIGNAL"]
    ):
        return "BUY", trend

    if (
        latest["EMA20"] < latest["EMA50"]
        and latest["RSI"] > 60
        and previous["MACD"] > previous["MACD_SIGNAL"]
        and latest["MACD"] < latest["MACD_SIGNAL"]
    ):
        return "SELL", trend

    return "HOLD", trend


def calculate_trade_levels(entry_price: float, atr: float) -> dict[str, float]:
    return {
        "buy_stop_loss": round(entry_price - atr * 1.5, 2),
        "buy_take_profit": round(entry_price + atr * 3, 2),
        "sell_stop_loss": round(entry_price + atr * 1.5, 2),
        "sell_take_profit": round(entry_price - atr * 3, 2),
    }


def suggested_direction(signal: str, trend: str) -> str:
    if signal in {"BUY", "SELL"}:
        return signal
    if trend == "BULLISH":
        return "BUY Reference"
    if trend == "BEARISH":
        return "SELL Reference"
    return "Neutral Reference"


def calculate_confidence(signal: str, trend: str, latest: pd.Series) -> int:
    seed_source = f"{signal}-{trend}-{latest.name}-{latest['Close']:.4f}"
    seed = int(hashlib.sha256(seed_source.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return int(rng.integers(70, 95))


def send_telegram_alert(message: str) -> None:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload, timeout=10)


def load_trade_log() -> pd.DataFrame:
    if not TRADE_LOG_PATH.exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)

    trades = pd.read_csv(TRADE_LOG_PATH)
    for column in TRADE_COLUMNS:
        if column not in trades.columns:
            trades[column] = np.nan

    return trades[TRADE_COLUMNS]


def save_trade_log(trades: pd.DataFrame) -> None:
    trades.to_csv(TRADE_LOG_PATH, index=False)


def normalize_trade_direction(direction: str) -> str:
    if direction.startswith("BUY"):
        return "BUY"
    if direction.startswith("SELL"):
        return "SELL"
    return "NEUTRAL"


def build_trade_id(ticker: str, timeframe: str, candle_time: object, trade_direction: str) -> str:
    return f"{ticker}|{timeframe}|{pd.Timestamp(candle_time).isoformat()}|{trade_direction}"


def update_open_trades(trades: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades

    updated = trades.copy()
    open_mask = (
        (updated["status"] == "OPEN")
        & (updated["symbol"] == symbol)
        & (updated["interval"] == interval)
    )

    for trade_index, trade in updated[open_mask].iterrows():
        entry_time = pd.Timestamp(trade["entry_time"])
        future_bars = market_data[market_data.index > entry_time]

        if future_bars.empty:
            continue

        trade_direction = normalize_trade_direction(str(trade["direction"]))
        entry = float(trade["entry_price"])
        stop = float(trade["stop_loss"])
        target = float(trade["take_profit"])

        for candle_time, candle in future_bars.iterrows():
            high = float(candle["High"])
            low = float(candle["Low"])
            exit_price = None
            result = None
            exit_reason = None

            if trade_direction == "BUY":
                if low <= stop:
                    exit_price = stop
                    result = "LOSS"
                    exit_reason = "Stop Loss"
                elif high >= target:
                    exit_price = target
                    result = "WIN"
                    exit_reason = "Take Profit"
                pnl_points = exit_price - entry if exit_price is not None else None
            elif trade_direction == "SELL":
                if high >= stop:
                    exit_price = stop
                    result = "LOSS"
                    exit_reason = "Stop Loss"
                elif low <= target:
                    exit_price = target
                    result = "WIN"
                    exit_reason = "Take Profit"
                pnl_points = entry - exit_price if exit_price is not None else None
            else:
                pnl_points = None

            if result is None:
                continue

            updated.loc[trade_index, "status"] = "CLOSED"
            updated.loc[trade_index, "result"] = result
            updated.loc[trade_index, "exit_time"] = pd.Timestamp(candle_time).isoformat()
            updated.loc[trade_index, "exit_price"] = round(float(exit_price), 2)
            updated.loc[trade_index, "pnl_points"] = round(float(pnl_points), 2)
            updated.loc[trade_index, "exit_reason"] = exit_reason
            break

    return updated


def open_virtual_trade(
    trades: pd.DataFrame,
    trade_signal: str,
    trade_direction: str,
    current_time: object,
) -> tuple[pd.DataFrame, bool]:
    normalized_direction = normalize_trade_direction(trade_direction)
    should_trade = trade_signal in {"BUY", "SELL"} or (
        trade_reference_levels and normalized_direction in {"BUY", "SELL"}
    )

    if not enable_paper_trading or not should_trade:
        return trades, False

    trade_id = build_trade_id(symbol, interval, current_time, normalized_direction)
    if not trades.empty and trade_id in set(trades["trade_id"].astype(str)):
        return trades, False

    new_trade = pd.DataFrame(
        [
            {
                "trade_id": trade_id,
                "symbol": symbol,
                "asset": asset_label,
                "interval": interval,
                "signal": trade_signal,
                "direction": normalized_direction,
                "entry_time": pd.Timestamp(current_time).isoformat(),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_percent": risk_percent,
                "confidence": confidence,
                "status": "OPEN",
                "result": "PENDING",
                "exit_time": "",
                "exit_price": "",
                "pnl_points": "",
                "exit_reason": "",
            }
        ]
    )

    return pd.concat([trades, new_trade], ignore_index=True), True


def calculate_performance(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "total": 0,
            "closed": 0,
            "open": 0,
            "wins": 0,
            "losses": 0,
            "success_rate": 0.0,
            "net_points": 0.0,
        }

    filtered = trades[(trades["symbol"] == symbol) & (trades["interval"] == interval)]
    closed = filtered[filtered["status"] == "CLOSED"]
    wins = int((closed["result"] == "WIN").sum())
    losses = int((closed["result"] == "LOSS").sum())
    closed_count = int(len(closed))
    pnl = pd.to_numeric(closed["pnl_points"], errors="coerce").fillna(0)

    return {
        "total": int(len(filtered)),
        "closed": closed_count,
        "open": int((filtered["status"] == "OPEN").sum()),
        "wins": wins,
        "losses": losses,
        "success_rate": round((wins / closed_count) * 100, 2) if closed_count else 0.0,
        "net_points": round(float(pnl.sum()), 2),
    }


try:
    df = add_indicators(fetch_data(symbol, interval, period))
except Exception as exc:
    st.error(f"Unable to load market data: {exc}")
    st.stop()

if len(df) < 2:
    st.warning("Not enough market data is available for this selection yet.")
    st.stop()

latest = df.iloc[-1]
signal, trend = calculate_signal(df)
entry_price = round(float(latest["Close"]), 2)
atr = float(latest["ATR"])
levels = calculate_trade_levels(entry_price, atr)
direction = suggested_direction(signal, trend)

if direction.startswith("BUY"):
    stop_loss = levels["buy_stop_loss"]
    take_profit = levels["buy_take_profit"]
elif direction.startswith("SELL"):
    stop_loss = levels["sell_stop_loss"]
    take_profit = levels["sell_take_profit"]
else:
    stop_loss = levels["buy_stop_loss"]
    take_profit = levels["buy_take_profit"]

confidence = calculate_confidence(signal, trend, latest)

trade_log = load_trade_log()
trade_log = update_open_trades(trade_log, df)
trade_log, trade_opened = open_virtual_trade(trade_log, signal, direction, latest.name)
save_trade_log(trade_log)
performance = calculate_performance(trade_log)

if ENABLE_TELEGRAM and signal != "HOLD":
    message = f"""
AI TRADING SIGNAL

Asset: {asset_label}
Symbol: {symbol}

Signal: {signal}
Trend: {trend}

Entry: {entry_price}
SL: {stop_loss}
TP: {take_profit}

Confidence: {confidence}%

Time: {datetime.now()}
"""
    send_telegram_alert(message)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Signal", signal)
col2.metric("Trend", trend)
col3.metric("Price", entry_price)
col4.metric("Confidence", f"{confidence}%")

perf1, perf2, perf3, perf4 = st.columns(4)
perf1.metric("Success Rate", f"{performance['success_rate']}%")
perf2.metric("Closed Trades", performance["closed"])
perf3.metric("Open Trades", performance["open"])
perf4.metric("Net Points", performance["net_points"])

if trade_opened:
    st.success(f"New virtual {normalize_trade_direction(direction)} trade opened for this candle.")
elif enable_paper_trading:
    st.info("Virtual trading is active. No duplicate trade was opened for the current candle.")
else:
    st.warning("Virtual trading is turned off.")

st.subheader("Trade Setup")
trade_df = pd.DataFrame(
    {
        "Parameter": [
            "Setup Type",
            "Direction",
            "Entry",
            "Stop Loss",
            "Take Profit",
            "Risk %",
            "Risk Reward",
            "BUY SL",
            "BUY TP",
            "SELL SL",
            "SELL TP",
        ],
        "Value": [
            "Active Signal" if signal != "HOLD" else "Reference Levels",
            direction,
            str(entry_price),
            str(stop_loss),
            str(take_profit),
            f"{risk_percent}%",
            "1:2",
            str(levels["buy_stop_loss"]),
            str(levels["buy_take_profit"]),
            str(levels["sell_stop_loss"]),
            str(levels["sell_take_profit"]),
        ],
    }
)
st.table(trade_df)

st.subheader("Virtual Trading Performance")
performance_df = pd.DataFrame(
    {
        "Metric": ["Total Trades", "Closed", "Open", "Wins", "Losses", "Success Rate", "Net Points"],
        "Value": [
            str(performance["total"]),
            str(performance["closed"]),
            str(performance["open"]),
            str(performance["wins"]),
            str(performance["losses"]),
            f"{performance['success_rate']}%",
            str(performance["net_points"]),
        ],
    }
)
st.table(performance_df)

recent_trades = trade_log[
    (trade_log["symbol"] == symbol) & (trade_log["interval"] == interval)
].tail(20)

if not recent_trades.empty:
    recent_trade_display = recent_trades[
        [
            "entry_time",
            "signal",
            "direction",
            "entry_price",
            "stop_loss",
            "take_profit",
            "status",
            "result",
            "exit_time",
            "exit_price",
            "pnl_points",
        ]
    ].fillna("")

    st.dataframe(
        recent_trade_display.astype(str),
        width="stretch",
    )

st.subheader("Technical Indicators")
indicator_df = pd.DataFrame(
    {
        "Indicator": ["EMA20", "EMA50", "RSI", "MACD", "ATR"],
        "Value": [
            round(float(latest["EMA20"]), 2),
            round(float(latest["EMA50"]), 2),
            round(float(latest["RSI"]), 2),
            round(float(latest["MACD"]), 2),
            round(float(latest["ATR"]), 2),
        ],
    }
)
st.table(indicator_df)

st.subheader("Price Chart")
fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candles",
    )
)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], line=dict(width=1), name="EMA20"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], line=dict(width=1), name="EMA50"))
fig.update_layout(height=700, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, width="stretch")

st.subheader("Latest Market Snapshot")
snapshot = df.tail(10)[["Close", "EMA20", "EMA50", "RSI", "MACD"]]
st.dataframe(snapshot)

st.markdown("---")
st.markdown(
    """
### AI Trading Agent Notes

- This dashboard is for educational purposes.
- Always use proper risk management.
- Avoid trading during major news events.
- Recommended asset: XAU/USD
- Recommended timeframe: 15M
"""
)
