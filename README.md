# NEPSE Signal Indicator
A Streamlit-powered financial analysis tool to visualize and score NEPSE (Nepal Stock Exchange) stocks using technical indicators and candlestick patterns.

## Features

- Select and analyze NEPSE-listed stocks via a simple Streamlit UI
- Calculate technical indicators like:
  - EMA (Exponential Moving Average)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Supertrend
  - Ichimoku Cloud
- Detect candlestick patterns like:
  - Doji
  - Engulfing
  - Hammer
  - Marubozu
  - Tweezer
- Score indicators and candlesticks to generate a simple bullish/bearish signal
- Visualizations of price movement and indicators

---

## What are Technical Indicators?

Technical indicators are mathematical calculations based on price, volume, or open interest of a security. They help traders make predictions about future price movements.

In this app, we calculate:

- **EMA**: Shows average price over a period, weighted more on recent prices
- **MACD**: Shows the relationship between two EMAs, used for identifying trend changes
- **RSI**: Indicates if a stock is overbought or oversold
- **Supertrend**: Trend-following indicator
- **Ichimoku Cloud**: Identifies support/resistance and trend direction

---

##  Candlestick Patterns

These patterns help identify market sentiment based on daily price action. This app detects:

- **Doji**: Indecision in the market
- **Engulfing**: Reversal signal
- **Hammer**: Potential bottom reversal
- **Marubozu**: Strong momentum
- **Tweezer Top/Bottom**: Possible reversal

---indicators includes: EMA, MACD, RSI, Supertrend, Ichimoku cloud and different type of candlestick indication like momentum candle, marubozu candles etc

Companies are: AKPL, ALICL, BHL, CHL, HLI, Hurja, Nabil, NFS, NLIC, Nyadi

<pre>
Directory structure
__pycache__/
cleaned_dataset/
Cleaning/
├── __pycache__/
├── clean.py
├── cleanNsave.py
├── data_scrape.ipynb
datasets/
model/
└── nepse_model.joblib
reqd_module/
├── __init__.py
├── lstm_model.py
├── signal_calculator.py
├── technical_indicators.py
main.py
</pre>

#### Note: this is just a probabilistic indication
