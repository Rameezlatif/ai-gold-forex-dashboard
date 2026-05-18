# AI Gold & Forex Trading Dashboard

Streamlit dashboard for educational gold and forex signal testing. It fetches market data from Yahoo Finance, calculates technical indicators, generates BUY/SELL/HOLD signals, and tracks virtual paper trades to measure bot performance over time.

## Features

- XAU/USD, EUR/USD, and GBP/USD market data
- EMA20, EMA50, RSI, MACD, and ATR indicators
- Entry, stop loss, and take profit levels
- Automatic virtual paper trading
- Success rate, open trades, closed trades, and net points tracking
- Volume confirmation and volume spike tracking
- High-impact economic-news guard for paper trades
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

## Notes

This project is for educational and paper-trading use only. It does not place real broker trades.

For the economic-news guard, add Trading Economics credentials in Streamlit secrets or environment variables:

```toml
TRADING_ECONOMICS_CLIENT = "your_client"
TRADING_ECONOMICS_KEY = "your_key"
```

If no credentials are configured, the app can block virtual trades when `Block if News Unavailable` is enabled. By default, that setting is off so paper-trading tests can continue without calendar credentials.
