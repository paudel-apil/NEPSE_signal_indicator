import pandas as pd
import numpy as np

def is_bullish_engulfing(df):
    prev_red = df['close_price'].shift(1) < df['open_price'].shift(1)
    curr_green = df['close_price'] > df['open_price']
    engulf = (df['open_price'] < df['close_price'].shift(1)) & (df['close_price'] > df['open_price'].shift(1))
    return prev_red & curr_green & engulf

def is_bearish_engulfing(df):
    prev_green = df['close_price'].shift(1) > df['open_price'].shift(1)
    curr_red = df['close_price'] < df['open_price']
    engulf = (df['open_price'] > df['close_price'].shift(1)) & (df['close_price'] < df['open_price'].shift(1))
    return prev_green & curr_red & engulf

def is_bullish_momentum(df, threshold=0.7):
    body = df['close_price'] - df['open_price']
    candle_range = df['high_price'] - df['low_price']
    body_ratio = body / candle_range
    return (body > 0) & (body_ratio > threshold)

def is_bearish_momentum(df, threshold=0.7):
    body = df['open_price'] - df['close_price']
    candle_range = df['high_price'] - df['low_price']
    body_ratio = body / candle_range
    return (body > 0) & (body_ratio > threshold)

def is_bullish_marubozu(df, threshold=0.05):
    upper_wick = df['high_price'] - df[['close_price', 'open_price']].max(axis=1)
    lower_wick = df[['close_price', 'open_price']].min(axis=1) - df['low_price']
    candle_range = df['high_price'] - df['low_price']
    return (upper_wick / candle_range < threshold) & (lower_wick / candle_range < threshold) & (df['close_price'] > df['open_price'])

def is_bearish_marubozu(df, threshold=0.05):
    upper_wick = df['high_price'] - df[['close_price', 'open_price']].max(axis=1)
    lower_wick = df[['close_price', 'open_price']].min(axis=1) - df['low_price']
    candle_range = df['high_price'] - df['low_price']
    return (upper_wick / candle_range < threshold) & (lower_wick / candle_range < threshold) & (df['close_price'] < df['open_price'])

def is_doji(df, threshold=0.1):
    body = abs(df['close_price'] - df['open_price'])
    candle_range = df['high_price'] - df['low_price']
    return (body / candle_range) < threshold

def is_bullish_doji(df, threshold=0.1):
    return is_doji(df, threshold) & (df['close_price'] > df['open_price'])

def is_bearish_doji(df, threshold=0.1):
    return is_doji(df, threshold) & (df['close_price'] < df['open_price'])

def is_bullish_hammer(df, wick_ratio=2):
    body = abs(df['close_price'] - df['open_price'])
    lower_wick = df[['close_price', 'open_price']].min(axis=1) - df['low_price']
    upper_wick = df['high_price'] - df[['close_price', 'open_price']].max(axis=1)
    return (lower_wick > wick_ratio * body) & (upper_wick < body)

def is_bearish_hammer(df, wick_ratio=2):
    body = abs(df['close_price'] - df['open_price'])
    lower_wick = df[['close_price', 'open_price']].min(axis=1) - df['low_price']
    upper_wick = df['high_price'] - df[['close_price', 'open_price']].max(axis=1)
    return (lower_wick > wick_ratio * body) & (upper_wick < body)

def is_tweezer_top(df, tolerance=0.05):
    return abs(df['high_price'] - df['high_price'].shift(1)) < (tolerance * df['high_price'])

def is_tweezer_bottom(df, tolerance=0.05):
    return abs(df['low_price'] - df['low_price'].shift(1)) < (tolerance * df['low_price'])

def calculate_candle_scores(df):
    df['candle_score'] = 0
    df['candle_score'] += is_bullish_momentum(df).astype(int)
    df['candle_score'] += is_bullish_doji(df).astype(int)
    df['candle_score'] += is_bullish_hammer(df).astype(int)
    df['candle_score'] += is_bullish_engulfing(df).astype(int)
    df['candle_score'] += is_bullish_marubozu(df).astype(int)
    df['candle_score'] += is_tweezer_bottom(df).astype(int)
    df['candle_score'] -= is_bearish_momentum(df).astype(int)
    df['candle_score'] -= is_bearish_doji(df).astype(int)
    df['candle_score'] -= is_bearish_hammer(df).astype(int)
    df['candle_score'] -= is_bearish_engulfing(df).astype(int)
    df['candle_score'] -= is_bearish_marubozu(df).astype(int)
    df['candle_score'] -= is_tweezer_top(df).astype(int)
    return df

def combine_scores_and_predict(df, deep_scores):
    tech_weight = 0.7
    deep_weight = 0.3
    # Assign deep_scores only to the last 20% of the DataFrame with nullable integer type
    test_size = int(0.2 * len(df))
    padded_scores = np.pad(deep_scores, (0, len(df) - len(deep_scores)), 'constant', constant_values=0)
    df['deep_score'] = pd.Series(padded_scores, index=df.index, dtype='Int64')  # Use nullable Int64
    df['total_score'] = (df['ema_score'] + df['macd_buy_score'] + df['supertrend_score'] + df['rsi_score'] + df['ichimoku_score'])
    df['final_score'] = (df['total_score'] * tech_weight + df['deep_score'] * deep_weight)
    def classify_signal(score):
        if score >= 3:
            return "Strong Bullish"
        elif score >= 1:
            return "Moderate Bullish"
        elif score > -1:
            return "Neutral"
        elif score > -3:
            return "Moderate Bearish"
        else:
            return "Strong Bearish"
    df['final_signal'] = df['final_score'].apply(classify_signal)
    return df['final_signal'].iloc[-1]  # Return the most recent prediction