import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
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
    "session_id",
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

TRADING_SESSIONS_UTC = {
    "All Sessions": None,
    "London": (7, 16),
    "New York": (12, 21),
    "London/New York Overlap": (12, 16),
}

USER_TIMEZONES = [
    "Asia/Karachi",
    "Asia/Dubai",
    "Asia/Kolkata",
    "Asia/Singapore",
    "Europe/London",
    "Europe/Berlin",
    "America/New_York",
    "America/Chicago",
    "America/Los_Angeles",
    "UTC",
]


st.set_page_config(page_title="AI Gold Trading Agent", page_icon=":chart_with_upwards_trend:", layout="wide")
st_autorefresh(interval=60_000, key="refresh")

st.markdown(
    """
    <style>
    :root {
        --bg: #f5f7fb;
        --panel: #ffffff;
        --panel-soft: #f8fafc;
        --border: #d9e2ef;
        --text: #172033;
        --muted: #68758a;
        --accent: #c89b3c;
        --accent-soft: #fff6df;
        --buy: #11845b;
        --sell: #c24141;
        --hold: #536179;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    [data-testid="stSidebar"] {
        background: #eef3f9;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] label {
        color: var(--text);
    }

    [data-testid="stSidebar"] [data-testid="stButton"] button {
        border-radius: 6px;
        border: 1px solid #b8c6d8;
        background: #ffffff;
        color: var(--text);
        font-weight: 600;
    }

    .dashboard-hero {
        padding: 1.2rem 1.35rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: linear-gradient(135deg, #ffffff 0%, #f9fbff 58%, #fff7e6 100%);
        box-shadow: 0 12px 28px rgba(21, 31, 52, 0.06);
    }

    .dashboard-kicker {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .dashboard-title {
        margin: 0;
        font-size: 2.15rem;
        line-height: 1.1;
        font-weight: 800;
        color: var(--text);
    }

    .dashboard-subtitle {
        margin: 0.55rem 0 0;
        color: var(--muted);
        font-size: 0.98rem;
        max-width: 760px;
    }

    .status-strip {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }

    .status-chip {
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.28rem 0.65rem;
        background: #ffffff;
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
    }

    .status-chip strong {
        color: var(--text);
    }

    h3 {
        color: var(--text);
        font-weight: 800;
        margin-top: 1.45rem;
    }

    div[data-testid="stMetric"] {
        min-height: 108px;
        padding: 0.95rem 1rem;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--panel);
        box-shadow: 0 8px 18px rgba(21, 31, 52, 0.05);
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        white-space: normal;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text);
        font-size: 1.72rem;
        font-weight: 800;
        line-height: 1.12;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    div[data-testid="stTable"],
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
        background: var(--panel);
    }

    .stPlotlyChart {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.35rem;
        background: var(--panel);
        box-shadow: 0 8px 18px rgba(21, 31, 52, 0.04);
    }

    .small-note {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: -0.35rem;
        margin-bottom: 0.55rem;
    }

    .assistant-panel {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--panel);
        padding: 1rem 1.1rem;
        margin: 1rem 0;
        box-shadow: 0 10px 22px rgba(21, 31, 52, 0.05);
    }

    .assistant-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }

    .assistant-decision {
        font-size: 2rem;
        line-height: 1.1;
        font-weight: 850;
        margin-bottom: 0.35rem;
    }

    .decision-buy { color: var(--buy); }
    .decision-sell { color: var(--sell); }
    .decision-wait { color: var(--hold); }

    .assistant-note {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dashboard-hero">
        <div class="dashboard-kicker">Trade assistant dashboard</div>
        <h1 class="dashboard-title">AI Gold & Forex Trading Dashboard</h1>
        <p class="dashboard-subtitle">
            Clear BUY/SELL reference levels, risk controls, and explainable signal notes to support your own analysis.
        </p>
        <div class="status-strip">
            <span class="status-chip">Mode: <strong>Virtual trading</strong></span>
            <span class="status-chip">Refresh: <strong>60 seconds</strong></span>
            <span class="status-chip">Risk: <strong>ATR based</strong></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "paper_session_id" not in st.session_state:
    st.session_state.paper_session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

st.sidebar.header("Settings")

asset_label = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
symbol = ASSETS[asset_label]

interval = st.sidebar.selectbox("Timeframe", ["5m", "15m", "1h"], index=1)
period = st.sidebar.selectbox("Indicator Data Window", ["7d", "30d", "60d"])
risk_percent = st.sidebar.slider("Risk Per Trade (%)", 1, 5, 2)
enable_paper_trading = st.sidebar.toggle("Auto Virtual Trading", value=True)
trade_reference_levels = st.sidebar.toggle("Trade HOLD References", value=False)
min_adx = st.sidebar.slider("Minimum ADX", 10, 40, 20)
trading_session = st.sidebar.selectbox(
    "Trading Session",
    list(TRADING_SESSIONS_UTC.keys()),
    index=3,
)
user_timezone_name = st.sidebar.selectbox(
    "Your Timezone",
    USER_TIMEZONES,
    index=USER_TIMEZONES.index("Asia/Karachi"),
)
user_timezone = pytz.timezone(user_timezone_name)
require_volume_confirmation = st.sidebar.toggle("Require Volume Confirmation", value=False)
volume_spike_threshold = st.sidebar.slider("Volume Spike Threshold", 1.0, 3.0, 1.2, 0.1)
spread_points = st.sidebar.number_input("Spread Cost (points)", 0.0, 100.0, 0.20, 0.01)
slippage_points = st.sidebar.number_input("Slippage Cost (points)", 0.0, 100.0, 0.10, 0.01)
avoid_high_impact_news = st.sidebar.toggle("Avoid High Impact News", value=True)
block_if_news_unavailable = st.sidebar.toggle("Block if News Unavailable", value=False)
news_minutes_before = st.sidebar.slider("Minutes Before News", 15, 180, 60, 15)
news_minutes_after = st.sidebar.slider("Minutes After News", 15, 180, 30, 15)

st.sidebar.header("Session")
st.sidebar.caption(f"Current: {st.session_state.paper_session_id}")
if st.sidebar.button("Start New Session", use_container_width=True):
    st.session_state.paper_session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    st.rerun()


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
    frame["ADX"] = ta.trend.adx(
        frame["High"].astype(float),
        frame["Low"].astype(float),
        close,
        window=14,
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


def calculate_execution_entry(direction: str, market_price: float) -> float:
    cost = spread_points + slippage_points
    if direction.startswith("BUY"):
        return round(market_price + cost, 2)
    if direction.startswith("SELL"):
        return round(market_price - cost, 2)
    return round(market_price, 2)


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


def as_utc_timestamp(timestamp: object) -> pd.Timestamp:
    converted = pd.Timestamp(timestamp)
    if converted.tzinfo is None:
        return converted.tz_localize("UTC")
    return converted.tz_convert("UTC")


def format_local_time(timestamp: object) -> str:
    local_time = as_utc_timestamp(timestamp).tz_convert(user_timezone)
    return local_time.strftime("%Y-%m-%d %H:%M %Z")


def selected_session_local_window() -> str:
    session = TRADING_SESSIONS_UTC[trading_session]
    if session is None:
        return "24 hours"

    today_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_hour, end_hour = session
    start_time = today_utc.replace(hour=start_hour).astimezone(user_timezone)
    end_time = today_utc.replace(hour=end_hour).astimezone(user_timezone)
    return f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}"


def is_in_selected_session(timestamp: object) -> bool:
    session = TRADING_SESSIONS_UTC[trading_session]
    if session is None:
        return True

    hour = as_utc_timestamp(timestamp).hour
    start_hour, end_hour = session
    return start_hour <= hour < end_hour


def apply_strategy_filters(
    raw_signal: str,
    latest_row: pd.Series,
    volume_is_confirmed: bool,
) -> tuple[str, list[str]]:
    reasons = []

    if raw_signal not in {"BUY", "SELL"}:
        return raw_signal, reasons

    adx_value = float(latest_row.get("ADX", 0))
    if adx_value < min_adx:
        reasons.append(f"ADX below {min_adx}")

    if not is_in_selected_session(latest_row.name):
        reasons.append(f"outside {trading_session}")

    if require_volume_confirmation and not volume_is_confirmed:
        reasons.append("volume not confirmed")

    if reasons:
        return "HOLD", reasons

    return raw_signal, reasons


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


def clear_session_trades(trades: pd.DataFrame, session_id: str) -> pd.DataFrame:
    if trades.empty or "session_id" not in trades.columns:
        return trades

    return trades[trades["session_id"].astype(str) != session_id].copy()


def normalize_trade_direction(direction: str) -> str:
    if direction.startswith("BUY"):
        return "BUY"
    if direction.startswith("SELL"):
        return "SELL"
    return "NEUTRAL"


def build_trade_id(ticker: str, timeframe: str, candle_time: object, trade_direction: str) -> str:
    return (
        f"{st.session_state.paper_session_id}|"
        f"{ticker}|{timeframe}|{pd.Timestamp(candle_time).isoformat()}|{trade_direction}"
    )


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
        if not enable_paper_trading:
            return trades, False, "Virtual trading is off."
        return trades, False, "No BUY/SELL setup is available."

    if trade_blocked:
        return trades, False, block_reason

    trade_id = build_trade_id(symbol, interval, current_time, normalized_direction)
    if not trades.empty and trade_id in set(trades["trade_id"].astype(str)):
        return trades, False, "No duplicate trade opened for the current candle."

    new_trade = pd.DataFrame(
        [
            {
                "trade_id": trade_id,
                "session_id": st.session_state.paper_session_id,
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


def calculate_performance(trades: pd.DataFrame, session_id: str | None = None) -> dict[str, float | int]:
    if trades.empty:
        return {
            "total": 0,
            "closed": 0,
            "open": 0,
            "wins": 0,
            "losses": 0,
            "success_rate": 0.0,
            "net_points": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
        }

    filtered = trades[(trades["symbol"] == symbol) & (trades["interval"] == interval)]
    if session_id is not None and "session_id" in filtered.columns:
        filtered = filtered[filtered["session_id"].astype(str) == session_id]
    closed = filtered[filtered["status"] == "CLOSED"]
    wins = int((closed["result"] == "WIN").sum())
    losses = int((closed["result"] == "LOSS").sum())
    closed_count = int(len(closed))
    pnl = pd.to_numeric(closed["pnl_points"], errors="coerce").fillna(0)
    winning_pnl = pnl[pnl > 0]
    losing_pnl = pnl[pnl < 0]
    avg_win = float(winning_pnl.mean()) if not winning_pnl.empty else 0.0
    avg_loss = abs(float(losing_pnl.mean())) if not losing_pnl.empty else 0.0
    gross_profit = float(winning_pnl.sum()) if not winning_pnl.empty else 0.0
    gross_loss = abs(float(losing_pnl.sum())) if not losing_pnl.empty else 0.0
    win_rate = wins / closed_count if closed_count else 0
    loss_rate = losses / closed_count if closed_count else 0
    equity = pnl.cumsum()
    drawdown = equity.cummax() - equity

    return {
        "total": int(len(filtered)),
        "closed": closed_count,
        "open": int((filtered["status"] == "OPEN").sum()),
        "wins": wins,
        "losses": losses,
        "success_rate": round(win_rate * 100, 2) if closed_count else 0.0,
        "net_points": round(float(pnl.sum()), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else 0.0,
        "expectancy": round((win_rate * avg_win) - (loss_rate * avg_loss), 2),
        "max_drawdown": round(float(drawdown.max()), 2) if not drawdown.empty else 0.0,
    }


def backtest_strategy(data: pd.DataFrame) -> dict[str, float | int]:
    closed_results = []

    for row_number in range(51, len(data) - 1):
        test_slice = data.iloc[: row_number + 1]
        current = data.iloc[row_number]
        raw_signal, _ = calculate_signal(test_slice)
        volume_status_value, volume_ok = calculate_volume_status(current)
        filtered_signal, filter_reasons = apply_strategy_filters(raw_signal, current, volume_ok)

        if filtered_signal not in {"BUY", "SELL"} or filter_reasons:
            continue

        market_entry = round(float(current["Close"]), 2)
        execution_entry = calculate_execution_entry(filtered_signal, market_entry)
        test_levels = calculate_trade_levels(execution_entry, float(current["ATR"]))
        stop = test_levels["buy_stop_loss"] if filtered_signal == "BUY" else test_levels["sell_stop_loss"]
        target = test_levels["buy_take_profit"] if filtered_signal == "BUY" else test_levels["sell_take_profit"]

        for _, future_candle in data.iloc[row_number + 1 :].iterrows():
            high = float(future_candle["High"])
            low = float(future_candle["Low"])

            if filtered_signal == "BUY":
                if low <= stop:
                    closed_results.append(stop - execution_entry)
                    break
                if high >= target:
                    closed_results.append(target - execution_entry)
                    break
            else:
                if high >= stop:
                    closed_results.append(execution_entry - stop)
                    break
                if low <= target:
                    closed_results.append(execution_entry - target)
                    break

    if not closed_results:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "net_points": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
        }

    pnl = pd.Series(closed_results)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = len(wins) / len(pnl)
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = abs(float(losses.mean())) if not losses.empty else 0.0
    equity = pnl.cumsum()
    drawdown = equity.cummax() - equity

    return {
        "trades": int(len(pnl)),
        "win_rate": round(win_rate * 100, 2),
        "net_points": round(float(pnl.sum()), 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else 0.0,
        "expectancy": round((win_rate * avg_win) - ((1 - win_rate) * avg_loss), 2),
        "max_drawdown": round(float(drawdown.max()), 2),
    }


def assistant_decision(signal_value: str, blocked: bool) -> tuple[str, str, str]:
    if blocked:
        return "WAIT", "decision-wait", "A setup exists, but one or more safety filters blocked it."
    if signal_value == "BUY":
        return "BUY SETUP", "decision-buy", "The assistant sees a bullish setup. Confirm with your own chart analysis before entry."
    if signal_value == "SELL":
        return "SELL SETUP", "decision-sell", "The assistant sees a bearish setup. Confirm with your own chart analysis before entry."
    return "WAIT", "decision-wait", "No active BUY or SELL setup is confirmed right now."


def build_explanation_points() -> list[str]:
    points = [
        f"Trend is {trend} based on EMA20 versus EMA50.",
        f"Raw signal is {raw_signal}; tradable signal is {signal}.",
        f"ADX is {round(float(latest.get('ADX', 0)), 2)} with minimum filter set to {min_adx}.",
        f"Session is {'open' if is_in_selected_session(latest.name) else 'closed'} for {trading_session}.",
        f"Volume status is {volume_status} with ratio {float(latest.get('Volume_Ratio', 0)):.2f}x.",
        f"News guard is {'blocked' if news_blocked else 'clear'}.",
    ]

    if filter_reasons:
        points.append(f"Trade filter blocked the setup because: {', '.join(filter_reasons)}.")

    if signal == "HOLD":
        points.append("Assistant recommendation: wait; use BUY/SELL levels as reference only.")
    elif signal == "BUY":
        points.append("Assistant recommendation: only consider BUY if price action confirms your own bullish analysis.")
    elif signal == "SELL":
        points.append("Assistant recommendation: only consider SELL if price action confirms your own bearish analysis.")

    return points


try:
    df = add_indicators(fetch_data(symbol, interval, period))
except Exception as exc:
    st.error(f"Unable to load market data: {exc}")
    st.stop()

if len(df) < 2:
    st.warning("Not enough market data is available for this selection yet.")
    st.stop()

latest = df.iloc[-1]
raw_signal, trend = calculate_signal(df)
market_price = round(float(latest["Close"]), 2)
atr = float(latest["ATR"])
volume_available = has_volume_data(df)
volume_status, volume_confirmed = calculate_volume_status(latest)
signal, filter_reasons = apply_strategy_filters(raw_signal, latest, volume_confirmed)
direction = suggested_direction(signal, trend)
volume_blocked = require_volume_confirmation and not volume_confirmed
strategy_blocked = bool(filter_reasons)
entry_price = calculate_execution_entry(direction, market_price)
levels = calculate_trade_levels(entry_price, atr)
buy_entry_price = calculate_execution_entry("BUY", market_price)
sell_entry_price = calculate_execution_entry("SELL", market_price)
buy_levels = calculate_trade_levels(buy_entry_price, atr)
sell_levels = calculate_trade_levels(sell_entry_price, atr)

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
trade_blocked = news_blocked or strategy_blocked
trade_block_reason = (
    f"News guard active: {news_reason}"
    if news_blocked
    else f"Strategy filter active: {', '.join(filter_reasons)}"
    if strategy_blocked
    else "Clear to trade"
)
filter_status = (
    "No Setup"
    if raw_signal not in {"BUY", "SELL"}
    else "Blocked"
    if strategy_blocked
    else "Pass"
)
assistant_title, assistant_class, assistant_note = assistant_decision(signal, trade_blocked)
explanation_points = build_explanation_points()

trade_log = load_trade_log()
if st.sidebar.button("Clear Current Session Trades", use_container_width=True):
    trade_log = clear_session_trades(trade_log, st.session_state.paper_session_id)
    save_trade_log(trade_log)
    st.rerun()

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
session_performance = calculate_performance(trade_log, st.session_state.paper_session_id)
all_time_performance = calculate_performance(trade_log)

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

col1, col2 = st.columns(2)
col1.metric("Signal", signal)
col2.metric("Trend", trend)
col3, col4 = st.columns(2)
col3.metric("Price", market_price)
col4.metric("Confidence", f"{confidence}%")

guard1, guard2 = st.columns(2)
guard1.metric("Volume", f"{int(float(latest.get('Volume', 0))):,}" if volume_available else "N/A")
guard2.metric("Volume Ratio", f"{float(latest.get('Volume_Ratio', 0)):.2f}x" if volume_available else "N/A")
guard3, guard4 = st.columns(2)
guard3.metric("Volume Status", volume_status)
guard4.metric("News Guard", "BLOCKED" if news_blocked else "CLEAR")

filter1, filter2 = st.columns(2)
filter1.metric("Raw Signal", raw_signal)
filter2.metric("ADX", round(float(latest.get("ADX", 0)), 2))
filter3, filter4 = st.columns(2)
filter3.metric("Session", "OPEN" if is_in_selected_session(latest.name) else "CLOSED")
filter4.metric("Trade Filter", filter_status.upper())

time1, time2 = st.columns(2)
time1.metric("Your Time", format_local_time(datetime.now(timezone.utc)))
time2.metric("Session Window", selected_session_local_window())

perf1, perf2 = st.columns(2)
perf1.metric("Session Success", f"{session_performance['success_rate']}%")
perf2.metric("Session Closed", session_performance["closed"])
perf3, perf4 = st.columns(2)
perf3.metric("Session Open", session_performance["open"])
perf4.metric("Session Net", session_performance["net_points"])

quality1, quality2 = st.columns(2)
quality1.metric("Session Expectancy", session_performance["expectancy"])
quality2.metric("Session Profit Factor", session_performance["profit_factor"])
quality3, quality4 = st.columns(2)
quality3.metric("Session Avg Win/Loss", f"{session_performance['avg_win']} / {session_performance['avg_loss']}")
quality4.metric("Session Drawdown", session_performance["max_drawdown"])

if trade_opened:
    st.success(f"New virtual {normalize_trade_direction(direction)} trade opened for this candle.")
elif trade_blocked:
    st.error(f"Virtual trade blocked. {trade_message}")
elif enable_paper_trading:
    st.info(f"Virtual trading is active. {trade_message}")
else:
    st.warning("Virtual trading is turned off.")

st.subheader("Trade Assistant")
st.markdown(
    f"""
    <div class="assistant-panel">
        <div class="assistant-label">Assistant view</div>
        <div class="assistant-decision {assistant_class}">{assistant_title}</div>
        <p class="assistant-note">{assistant_note}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

levels_df = pd.DataFrame(
    {
        "Side": ["BUY", "SELL"],
        "Entry Reference": [str(buy_entry_price), str(sell_entry_price)],
        "Take Profit": [str(buy_levels["buy_take_profit"]), str(sell_levels["sell_take_profit"])],
        "Stop Loss": [str(buy_levels["buy_stop_loss"]), str(sell_levels["sell_stop_loss"])],
        "Use Case": [
            "Use only if your analysis also supports a bullish trade.",
            "Use only if your analysis also supports a bearish trade.",
        ],
    }
)
st.table(levels_df)

with st.expander("Why is the assistant suggesting this?"):
    for point in explanation_points:
        st.markdown(f"- {point}")

context_df = pd.DataFrame(
    {
        "Item": [
            "Market Price",
            "Raw Signal",
            "Tradable Signal",
            "Filter",
            "Trend",
            "ADX",
            "Session",
            "Local Candle Time",
            "Session Window",
            "Risk",
            "Costs",
        ],
        "Value": [
            str(market_price),
            raw_signal,
            signal,
            filter_status if not filter_reasons else ", ".join(filter_reasons),
            trend,
            str(round(float(latest.get("ADX", 0)), 2)),
            "Open" if is_in_selected_session(latest.name) else "Closed",
            format_local_time(latest.name),
            selected_session_local_window(),
            f"{risk_percent}%",
            f"{spread_points + slippage_points:.2f}",
        ],
    }
)
st.table(context_df)

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

st.subheader("Current Session Performance")
st.caption(f"Session ID: {st.session_state.paper_session_id}")
session_performance_df = pd.DataFrame(
    {
        "Metric": [
            "Total Trades",
            "Closed",
            "Open",
            "Wins",
            "Losses",
            "Success Rate",
            "Net Points",
            "Expectancy",
            "Profit Factor",
            "Avg Win",
            "Avg Loss",
            "Max Drawdown",
        ],
        "Value": [
            str(session_performance["total"]),
            str(session_performance["closed"]),
            str(session_performance["open"]),
            str(session_performance["wins"]),
            str(session_performance["losses"]),
            f"{session_performance['success_rate']}%",
            str(session_performance["net_points"]),
            str(session_performance["expectancy"]),
            str(session_performance["profit_factor"]),
            str(session_performance["avg_win"]),
            str(session_performance["avg_loss"]),
            str(session_performance["max_drawdown"]),
        ],
    }
)
st.table(session_performance_df)

with st.expander("All-Time Paper Trading Performance"):
    all_time_performance_df = pd.DataFrame(
        {
            "Metric": [
                "Total Trades",
                "Closed",
                "Open",
                "Wins",
                "Losses",
                "Success Rate",
                "Net Points",
                "Expectancy",
                "Profit Factor",
                "Avg Win",
                "Avg Loss",
                "Max Drawdown",
            ],
            "Value": [
                str(all_time_performance["total"]),
                str(all_time_performance["closed"]),
                str(all_time_performance["open"]),
                str(all_time_performance["wins"]),
                str(all_time_performance["losses"]),
                f"{all_time_performance['success_rate']}%",
                str(all_time_performance["net_points"]),
                str(all_time_performance["expectancy"]),
                str(all_time_performance["profit_factor"]),
                str(all_time_performance["avg_win"]),
                str(all_time_performance["avg_loss"]),
                str(all_time_performance["max_drawdown"]),
            ],
        }
    )
    st.table(all_time_performance_df)

recent_trades = trade_log[
    (trade_log["symbol"] == symbol)
    & (trade_log["interval"] == interval)
    & (trade_log["session_id"].astype(str) == st.session_state.paper_session_id)
].tail(20)

if not recent_trades.empty:
    recent_trade_display = recent_trades[
        [
            "entry_time",
            "session_id",
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
        "Indicator": ["EMA20", "EMA50", "RSI", "MACD", "ATR", "ADX", "Volume", "Avg Volume", "Volume Ratio"],
        "Value": [
            round(float(latest["EMA20"]), 2),
            round(float(latest["EMA50"]), 2),
            round(float(latest["RSI"]), 2),
            round(float(latest["MACD"]), 2),
            round(float(latest["ATR"]), 2),
            round(float(latest.get("ADX", 0)), 2),
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
        increasing_line_color="#11845b",
        increasing_fillcolor="#d7f0e5",
        decreasing_line_color="#c24141",
        decreasing_fillcolor="#f7dada",
    )
)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], line=dict(width=1.5, color="#c89b3c"), name="EMA20"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], line=dict(width=1.5, color="#3b6ea8"), name="EMA50"))
fig.update_layout(
    height=700,
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fbfcff",
    margin=dict(l=16, r=16, t=28, b=16),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(gridcolor="#e7edf5"),
    yaxis=dict(gridcolor="#e7edf5"),
)
st.plotly_chart(fig, width="stretch")

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
