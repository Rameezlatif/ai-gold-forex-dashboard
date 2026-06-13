# AI Gold & Forex Trading Dashboard

Streamlit dashboard for educational gold and forex signal testing. It fetches market data from Yahoo Finance, calculates technical indicators, generates BUY/SELL/HOLD signals, and tracks virtual paper trades to measure bot performance over time.

## Features

- XAU/USD, EUR/USD, and GBP/USD market data
- EMA20, EMA50, RSI, MACD, and ATR indicators
- Entry, stop loss, and take profit levels
- Clear BUY TP, BUY SL, SELL TP, and SELL SL assistant table
- Explainable assistant notes for why a setup is BUY, SELL, or WAIT
- Automatic virtual paper trading
- Success rate, open trades, closed trades, and net points tracking
- Volume confirmation and volume spike tracking
- High-impact economic-news guard for paper trades
- ADX trend-strength filter
- London/New York trading-session filter
- User timezone selector for local session timing
- Spread and slippage simulation
- Expectancy, profit factor, average win/loss, and drawdown metrics
- Recent virtual trade history
- Optional Telegram alert hook

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Windows Desktop-Style Launch

On Windows, double-click:

```text
run_windows_app.bat
```

The launcher installs requirements if needed, starts Streamlit, and opens the app at `http://localhost:8501`.

For a true installable desktop app (`.exe` or installer), wrap this project with PyInstaller, Electron, or Tauri on a Windows machine. The Streamlit app can be packaged, but the build should be done on Windows.

## Notes

This project is for educational and paper-trading use only. It does not place real broker trades.

For the economic-news guard, add Trading Economics credentials in Streamlit secrets or environment variables:

```toml
TRADING_ECONOMICS_CLIENT = "your_client"
TRADING_ECONOMICS_KEY = "your_key"
```

If no credentials are configured, the app can block virtual trades when `Block if News Unavailable` is enabled. By default, that setting is off so paper-trading tests can continue without calendar credentials.
