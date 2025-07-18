# NEPSE_signal_indicator
Indicates the market signal for 10 different companies listed in using the LSTM model and different technical indicators NEPSE

Technical indicators includes: EMA, MACD, RSI, Supertrend, Ichimoku cloud and different type of candlestick indication like momentum candle, marubozu candles etc

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
