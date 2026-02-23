import pandas as pd
import numpy as np
import math

# Synthetic OHLC data resembling futures around 83000
np.random.seed(42)
base = 83000.0
# create 120 periods
n = 120
vol = 200.0
closes = base + np.cumsum(np.random.randn(n) * vol/4)
highs = closes + np.abs(np.random.randn(n) * vol/2)
lows = closes - np.abs(np.random.randn(n) * vol/2)

df = pd.DataFrame({
    'High': highs,
    'Low': lows,
    'Close': closes
})

def compute_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr

atr = compute_atr(df, 14)
df['ATR'] = atr

df['ATR_MA50'] = df['ATR'].rolling(50, min_periods=10).mean()
df['ATR_RATIO'] = (df['ATR'] / df['ATR_MA50']).replace([np.inf, -np.inf], np.nan).fillna(1.0)

print('Last Close:', df['Close'].iloc[-1])
print('Last ATR:', df['ATR'].iloc[-1])
print('Last ATR_MA50:', df['ATR_MA50'].iloc[-1])
print('Last ATR_RATIO:', df['ATR_RATIO'].iloc[-1])

# Show sample tail
print('\nTail:')
print(df[['Close','ATR','ATR_MA50','ATR_RATIO']].tail(10))
