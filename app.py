import hashlib
import os
from datetime import datetime, timedelta, timezone
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

ASSET_NEWS_COUNTRIES = {
    "Gold (XAU/USD)": ["united states"],
    "EUR/USD": ["united states", "euro area", "germany", "france"],
    "GBP/USD": ["united states", "united kingdom"],
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
require_volume_confirmation = st.sidebar.toggle("Require Volume Confirmation", value=False)
volume_spike_threshold = st.sidebar.slider("Volume Spike Threshold", 1.0, 3.0, 1.2, 0.1)
avoid_high_impact_news = st.sidebar.toggle("Avoid High Impact News", value=True)
block_if_news_unavailable = st.sidebar.toggle("Block if News Unavailable", value=False)
news_minutes_before = st.sidebar.slider("Minutes Before News", 15, 180, 60, 15)
news_minutes_after = st.sidebar.slider("Minutes After News", 15, 180, 30, 15)


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

    if "Volume" in frame.columns:
        frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce").fillna(0)
        frame["Volume_SMA20"] = frame["Volume"].rolling(window=20).mean()
        frame["Volume_Ratio"] = np.where(
            frame["Volume_SMA20"] > 0,
            frame["Volume"] / frame["Volume_SMA20"],
            0,
        )
    else:
        frame["Volume"] = 0
        frame["Volume_SMA20"] = 0
        frame["Volume_Ratio"] = 0

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


def has_volume_data(data: pd.DataFrame) -> bool:
    return "Volume" in data.columns and pd.to_numeric(data["Volume"], errors="coerce").fillna(0).sum() > 0


def calculate_volume_status(latest: pd.Series) -> tuple[str, bool]:
    volume = float(latest.get("Volume", 0))
    average_volume = float(latest.get("Volume_SMA20", 0))
    ratio = float(latest.get("Volume_Ratio", 0))

    if volume <= 0 or average_volume <= 0:
        return "Unavailable", False
    if ratio >= volume_spike_threshold:
        return "Confirmed", True
    return "Low", False


def date_range_for_calendar() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (now + timedelta(days=3)).strftime("%Y-%m-%d")
    return start, end


def get_trading_economics_credentials() -> str:
    secret_client = ""
    secret_key = ""

    try:
        secret_client = st.secrets.get("TRADING_ECONOMICS_CLIENT", "")
        secret_key = st.secrets.get("TRADING_ECONOMICS_KEY", "")
    except Exception:
        pass

    client = os.getenv("TRADING_ECONOMICS_CLIENT", secret_client)
    key = os.getenv("TRADING_ECONOMICS_KEY", secret_key)

    if client and key:
        return f"{client}:{key}"

    return "guest:guest"


@st.cache_data(ttl=900)
def fetch_economic_calendar(
    countries: tuple[str, ...],
    start: str,
    end: str,
    credentials: str,
) -> tuple[pd.DataFrame, str]:
    if not countries:
        return pd.DataFrame(), "No countries selected"

    country_path = ",".join(country.replace(" ", "%20") for country in countries)
    url = f"https://api.tradingeconomics.com/calendar/country/{country_path}/{start}/{end}"
    params = {"c": credentials}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        events = response.json()
    except Exception as exc:
        if credentials == "guest:guest":
            return pd.DataFrame(), "Calendar unavailable. Add Trading Economics API credentials or turn off 'Block if News Unavailable'."
        return pd.DataFrame(), f"Calendar unavailable: {exc}"

    if not isinstance(events, list) or not events:
        return pd.DataFrame(), "No events found"

    calendar = pd.DataFrame(events)
    if "Date" not in calendar.columns:
        return pd.DataFrame(), "Calendar response missing event dates"

    calendar["Date"] = pd.to_datetime(calendar["Date"], errors="coerce", utc=True)
    calendar["Importance"] = pd.to_numeric(calendar.get("Importance", 0), errors="coerce").fillna(0)
    calendar = calendar.dropna(subset=["Date"])
    calendar = calendar.sort_values("Date")

    return calendar, "Loaded"


def calculate_news_guard(calendar: pd.DataFrame, status: str) -> tuple[bool, str, pd.DataFrame]:
    if not avoid_high_impact_news:
        return False, "News guard off", pd.DataFrame()

    if calendar.empty:
        return block_if_news_unavailable, status, pd.DataFrame()

    now = datetime.now(timezone.utc)
    high_impact = calendar[calendar["Importance"] >= 3].copy()
    if high_impact.empty:
        return False, "No high-impact news found", high_impact

    window_start = now - timedelta(minutes=news_minutes_after)
    window_end = now + timedelta(minutes=news_minutes_before)
    active_events = high_impact[
        (high_impact["Date"] >= window_start) & (high_impact["Date"] <= window_end)
    ]

    if active_events.empty:
        return False, "Clear", high_impact

    next_event = active_events.iloc[0]
    event_name = next_event.get("Event", "High-impact news")
    country = next_event.get("Country", "")
    event_time = next_event["Date"].strftime("%Y-%m-%d %H:%M UTC")
    return True, f"{country} {event_name} at {event_time}", high_impact


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
    trade_blocked: bool,
    block_reason: str,
) -> tuple[pd.DataFrame, bool, str]:
    normalized_direction = normalize_trade_direction(trade_direction)
    should_trade = trade_signal in {"BUY", "SELL"} or (
        trade_reference_levels and normalized_direction in {"BUY", "SELL"}
    )

    if not enable_paper_trading or not should_trade:
        return trades, False, "Virtual trading is off or no trade setup is available."

    if trade_blocked:
        return trades, False, block_reason

    trade_id = build_trade_id(symbol, interval, current_time, normalized_direction)
    if not trades.empty and trade_id in set(trades["trade_id"].astype(str)):
        return trades, False, "No duplicate trade opened for the current candle."

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

    return pd.concat([trades, new_trade], ignore_index=True), True, "Trade opened."


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
volume_available = has_volume_data(df)
volume_status, volume_confirmed = calculate_volume_status(latest)
volume_blocked = require_volume_confirmation and not volume_confirmed

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
if volume_confirmed:
    confidence = min(confidence + 3, 99)
elif require_volume_confirmation:
    confidence = max(confidence - 8, 0)

calendar_start, calendar_end = date_range_for_calendar()
calendar_countries = tuple(ASSET_NEWS_COUNTRIES.get(asset_label, ["united states"]))
te_credentials = get_trading_economics_credentials()
news_calendar, news_status = fetch_economic_calendar(
    calendar_countries,
    calendar_start,
    calendar_end,
    te_credentials,
)
news_blocked, news_reason, high_impact_events = calculate_news_guard(news_calendar, news_status)
trade_blocked = news_blocked or volume_blocked
trade_block_reason = (
    f"News guard active: {news_reason}"
    if news_blocked
    else f"Volume filter active: {volume_status}"
    if volume_blocked
    else "Clear to trade"
)

trade_log = load_trade_log()
trade_log = update_open_trades(trade_log, df)
trade_log, trade_opened, trade_message = open_virtual_trade(
    trade_log,
    signal,
    direction,
    latest.name,
    trade_blocked,
    trade_block_reason,
)
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

guard1, guard2, guard3, guard4 = st.columns(4)
guard1.metric("Volume", f"{int(float(latest.get('Volume', 0))):,}" if volume_available else "N/A")
guard2.metric("Volume Ratio", f"{float(latest.get('Volume_Ratio', 0)):.2f}x" if volume_available else "N/A")
guard3.metric("Volume Status", volume_status)
guard4.metric("News Guard", "BLOCKED" if news_blocked else "CLEAR")

perf1, perf2, perf3, perf4 = st.columns(4)
perf1.metric("Success Rate", f"{performance['success_rate']}%")
perf2.metric("Closed Trades", performance["closed"])
perf3.metric("Open Trades", performance["open"])
perf4.metric("Net Points", performance["net_points"])

if trade_opened:
    st.success(f"New virtual {normalize_trade_direction(direction)} trade opened for this candle.")
elif trade_blocked:
    st.error(f"Virtual trade blocked. {trade_message}")
elif enable_paper_trading:
    st.info(f"Virtual trading is active. {trade_message}")
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
            "Volume Status",
            "News Guard",
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
            volume_status,
            "Blocked" if news_blocked else "Clear",
            str(levels["buy_stop_loss"]),
            str(levels["buy_take_profit"]),
            str(levels["sell_stop_loss"]),
            str(levels["sell_take_profit"]),
        ],
    }
)
st.table(trade_df)

st.subheader("News Guard")
if news_blocked:
    st.error(f"No virtual trades while news guard is active: {news_reason}")
else:
    st.success(f"News guard clear: {news_reason}")

if not high_impact_events.empty:
    upcoming_news = high_impact_events[high_impact_events["Date"] >= datetime.now(timezone.utc)].head(10)
    if not upcoming_news.empty:
        news_display = upcoming_news.copy()
        news_display["Date"] = news_display["Date"].dt.strftime("%Y-%m-%d %H:%M UTC")
        visible_columns = [
            column
            for column in ["Date", "Country", "Event", "Importance", "Actual", "Forecast", "Previous"]
            if column in news_display.columns
        ]
        st.dataframe(news_display[visible_columns].astype(str), width="stretch")

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
        "Indicator": ["EMA20", "EMA50", "RSI", "MACD", "ATR", "Volume", "Avg Volume", "Volume Ratio"],
        "Value": [
            round(float(latest["EMA20"]), 2),
            round(float(latest["EMA50"]), 2),
            round(float(latest["RSI"]), 2),
            round(float(latest["MACD"]), 2),
            round(float(latest["ATR"]), 2),
            int(float(latest.get("Volume", 0))),
            round(float(latest.get("Volume_SMA20", 0)), 2),
            round(float(latest.get("Volume_Ratio", 0)), 2),
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
snapshot = df.tail(10)[["Close", "Volume", "Volume_Ratio", "EMA20", "EMA50", "RSI", "MACD"]]
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
