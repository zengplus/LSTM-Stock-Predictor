import os
import pandas as pd
import numpy as np
import yfinance as yf

# 股票分类列表
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',        # 科技
    'JPM', 'BAC', 'C', 'WFC', 'GS',                # 金融
    'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',            # 医药
    'XOM', 'CVX', 'COP', 'SLB', 'BKR',             # 能源
    'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',         # 消费
    'CAT', 'DE', 'MMM', 'GE', 'HON'                # 工业
]

# 设置日期范围和保留特征数量
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
NUM_FEATURES_TO_KEEP = 9

# 定义函数以获取和处理股票数据
def get_stock_data(ticker):
    # 下载股票数据
    data = yf.download(ticker, start=START_DATE, end=END_DATE, proxy="http://127.0.0.1:7890")
    
    # 提取日期相关信息
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    
    # 计算移动平均线
    data['MA5'] = data['Close'].shift(1).rolling(window=5).mean()
    data['MA10'] = data['Close'].shift(1).rolling(window=10).mean()
    data['MA20'] = data['Close'].shift(1).rolling(window=20).mean()

    # RSI计算
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD指标
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

    # VWAP计算
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

    # 布林带
    period = 20
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['Std_dev'] = data['Close'].rolling(window=period).std()
    data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
    data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']

    # 相对大盘表现
    benchmark_data = yf.download('SPY', start=START_DATE, end=END_DATE)['Close']
    data['Relative_Performance'] = (data['Close'] / benchmark_data.values) * 100

    # 价格变化率（ROC）
    data['ROC'] = data['Close'].pct_change(periods=1) * 100

    # 平均真实范围（ATR）
    high_low_range = data['High'] - data['Low']
    high_close_range = abs(data['High'] - data['Close'].shift(1))
    low_close_range = abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low_range, high_close_range, low_close_range], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    # 前一天数据
    data[['Close_yes', 'Open_yes', 'High_yes', 'Low_yes']] = data[['Close', 'Open', 'High', 'Low']].shift(1)

    # 删除缺失值
    data = data.dropna()
    return data

# 获取并保存所有股票数据
data_folder = 'data'
os.makedirs(data_folder, exist_ok=True)  # 创建数据文件夹（若不存在）
for ticker in tickers:
    stock_data = get_stock_data(ticker)
    stock_data.to_csv(f'{data_folder}/{ticker}.csv')  # 保存到 data 文件夹

# 后处理CSV文件，删除第二行和第三行并重命名列
def clean_csv_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # 删除第二行和第三行
            df = df.drop([0, 1]).reset_index(drop=True)
            
            # 重命名'Price'列为'Date'（假设需要更改的列名为 'Price'）
            df = df.rename(columns={'Price': 'Date'})
            
            # 保存修改后的文件
            df.to_csv(file_path, index=False)
    print("所有文件处理完成！")

# 运行清理CSV文件的函数
clean_csv_files(data_folder)
