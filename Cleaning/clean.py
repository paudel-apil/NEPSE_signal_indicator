import pandas as pd
import numpy as np

def cleanse_df(df):
  df['date'] = pd.to_datetime(df['date'], dayfirst = True)

  df['volume'] = df['volume'].apply(lambda x: float(x.replace(",", "")) if isinstance(x, str) else float(x))
  df['turnover'] = df['turnover'].apply(lambda x: float(x.replace(",", "")) if isinstance(x, str) else float(x))
  df['open_price'] = df['open_price'].apply(lambda x: float(x.replace(",", "")) if isinstance(x, str) else float(x))
  df['close_price'] = df['close_price'].apply(lambda x: float(x.replace(",", "")) if isinstance(x, str) else float(x))
  df['high_price'] = df['high_price'].apply(lambda x: float(x.replace(",", "")) if isinstance(x, str) else float(x))
  df['low_price'] = df['low_price'].apply(lambda x: float(x.replace(",", "")) if isinstance(x, str) else float(x))


  df['open_price'] = df['open_price'].astype(np.float32)
  df['close_price'] = df['close_price'].astype(np.float32)
  df['high_price'] = df['high_price'].astype(np.float32)
  df['low_price'] = df['low_price'].astype(np.float32)
  df['volume'] = df['volume'].astype(np.float32)
  df['turnover'] = df['turnover'].astype(np.float32)

  df = df.sort_values('date')
  df.set_index('date', inplace = True)

  return df