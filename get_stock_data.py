import yfinance as yf
import pandas as pd
import numpy as np
import os

# 股票列表
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',]  # 科技
tickers += ['JPM', 'BAC', 'C', 'WFC', 'GS',]            # 金融
tickers += ['JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',]         # 医药
tickers += ['XOM', 'CVX', 'COP', 'SLB', 'BKR',]          # 能源
tickers += ['DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',]      # 消费
tickers += ['CAT', 'DE', 'MMM', 'GE', 'HON']            # 工业


num_features_to_keep = 9
start_date = '2021-01-01'
end_date = '2023-01-01'


def get_stock_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date, proxy="http://127.0.0.1:7890")
    # 日期
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day

    close = data['Close'].shift(1)

    # 移动平均线
    data['MA5'] = close.rolling(window=5).mean()
    data['MA10'] = close.rolling(window=10).mean()
    data['MA20'] = close.rolling(window=20).mean()

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    data.loc[data.index >= RSI.index[0], 'RSI'] = RSI

    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    MACD = exp1 - exp2
    data.loc[data.index >= MACD.index[0], 'MACD'] = MACD

    # VWAP
    data['VWAP'] = (close * data['Volume']).cumsum() / data['Volume'].cumsum()

    # Bollinger Bands
    period = 20
    data['SMA'] = close.rolling(window=period).mean()
    data['Std_dev'] = close.rolling(window=period).std()
    data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
    data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']

    # 相对大盘的表现
    benchmark_data = yf.download('SPY', start=start_date, end=end_date)['Close']
    data['Relative_Performance'] = (close / benchmark_data.values) * 100

    # 价格变化率
    data['ROC'] = (close.pct_change(periods=1)) * 100

    # 平均变化率
    high_low_range = data['High'] - data['Low']
    high_close_range = abs(data['High'].shift(1) - close.shift(1))
    low_close_range = abs(data['Low'].shift(1) - close.shift(1))
    true_range = pd.concat([high_low_range, high_close_range, low_close_range], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    data['Close_yes'] = data['Close'].shift(1)
    data['Open_yes'] = data['Open'].shift(1)
    data['High_yes'] = data['High'].shift(1)
    data['Low_yes'] = data['Low'].shift(1)

    data = data.dropna()
    return data


# 获取并保存所有股票数据
for ticker in tickers:
    stock_data = get_stock_data(ticker)
    stock_data.to_csv(f'data/{ticker}.csv')  # 保存到 data 文件夹