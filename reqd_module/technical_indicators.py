import pandas as pd
import numpy as np

def calculate_technical_indicators(df):
    # Simple Moving Averages
    df['sma_50'] = df['close_price'].rolling(window=50).mean()
    df['sma_200'] = df['close_price'].rolling(window=200).mean()  # Corrected window from 50 to 200

    # Exponential Moving Averages
    df['ema_12'] = df['close_price'].ewm(span=12, adjust=False).mean()
    df['ema_20'] = df['close_price'].ewm(span=20, adjust=False).mean()
    df['ema_26'] = df['close_price'].ewm(span=26, adjust=False).mean()
    df['ema_50'] = df['close_price'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close_price'].ewm(span=200, adjust=False).mean()
    df['ema_score'] = 0
    df['ema_score'] += (df['ema_20'] > df['ema_50']).astype(int)
    df['ema_score'] += (df['ema_50'] > df['ema_200']).astype(int)
    df['ema_score'] += (df['close_price'] > df['ema_20']).astype(int)

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_buy_score'] = 0
    df['macd_buy_score'] += ((df['macd'] > df['macd_signal']) & (df['close_price'] < df['ema_200'])).astype(int)

    # RSI
    window_length = 14
    delta = df['close_price'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window_length, min_periods=window_length).mean()
    avg_loss = loss.rolling(window=window_length, min_periods=window_length).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_score'] = 0
    df['rsi_score'] += (df['rsi'] < 30).astype(int)
    df['rsi_score'] += (df['rsi'] > 70).astype(int) * -1

    # Supertrend
    df = calculate_supertrend(df)

    # Ichimoku
    df = calculate_ichimoku(df)
    df = ichimoku_score(df)

    return df

def calculate_supertrend(df, period=10, multiplier=3):
    df = df.copy()
    df['hl'] = df['high_price'] - df['low_price']
    df['hc'] = abs(df['high_price'] - df['close_price'].shift())
    df['lc'] = abs(df['low_price'] - df['close_price'].shift())
    df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    hl2 = (df['high_price'] + df['low_price']) / 2
    df['upper_band'] = hl2 + multiplier * df['atr']
    df['lower_band'] = hl2 - multiplier * df['atr']
    df['supertrend'] = np.nan
    in_uptrend = True
    for current in range(1, len(df)):
        previous = current - 1
        if df['close_price'].iloc[current] > df['upper_band'].iloc[previous]:
            in_uptrend = True
        elif df['close_price'].iloc[current] < df['lower_band'].iloc[previous]:
            in_uptrend = False
        else:
            if in_uptrend and df['lower_band'].iloc[current] < df['lower_band'].iloc[previous]:
                df.loc[df.index[current], 'lower_band'] = df['lower_band'].iloc[previous]
            if not in_uptrend and df['upper_band'].iloc[current] > df['upper_band'].iloc[previous]:
                df.loc[df.index[current], 'upper_band'] = df['upper_band'].iloc[previous]
        df.loc[df.index[current], 'supertrend'] = (
            df['lower_band'].iloc[current] if in_uptrend else df['upper_band'].iloc[current]
        )
    df.drop(['hl', 'hc', 'lc', 'tr'], axis=1, inplace=True)
    df['supertrend_score'] = 0
    df['supertrend_score'] += is_bullish_supertrend(df).astype(int)
    df['supertrend_score'] -= is_bearish_supertrend(df).astype(int)
    return df

def is_bullish_supertrend(df):
    return df['close_price'] > df['supertrend']

def is_bearish_supertrend(df):
    return df['close_price'] < df['supertrend']

def calculate_ichimoku(df):
    high_9 = df['high_price'].rolling(window=9).max()
    low_9 = df['low_price'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2
    high_26 = df['high_price'].rolling(window=26).max()
    low_26 = df['low_price'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    high_52 = df['high_price'].rolling(window=52).max()
    low_52 = df['low_price'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    df['chikou_span'] = df['close_price'].shift(-26)
    return df

def ichimoku_score(df):
    df = df.copy()
    score = []
    for idx, row in df.iterrows():
        s = 0
        if (row['close_price'] > row['senkou_span_a']) and (row['close_price'] > row['senkou_span_b']):
            s += 2
        elif (row['close_price'] > min(row['senkou_span_a'], row['senkou_span_b'])) and (row['close_price'] < max(row['senkou_span_a'], row['senkou_span_b'])):
            s += 1
        else:
            s -= 2
        if row['tenkan_sen'] > row['kijun_sen']:
            s += 1
        else:
            s -= 1
        if row['chikou_span'] > row['close_price']:
            s += 1
        else:
            s -= 1
        score.append(s)
    df['ichimoku_score'] = score
    return df