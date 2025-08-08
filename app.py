import streamlit as st # 创建Web应用streamlit程序
import os # 操作系统相关功能
import pandas as pd # 数据处理和分析
import numpy as np # 数值计算
import yfinance as yf # 从Yahoo Finance下载股票数据
import warnings # 警告控制
import torch # PyTorch深度学习框架
import torch.nn as nn # PyTorch神经网络模块
import torch.optim as optim # PyTorch优化器模块
from torch.utils.data import DataLoader, TensorDataset # PyTorch数据加载和数据集工具
from sklearn.preprocessing import MinMaxScaler # 数据归一化的Scikit-learn工具
import pickle # Python对象序列化和反序列化
import time # 时间相关功能
import matplotlib.pyplot as plt # 绘图的库
import seaborn as sns # 基于Matplotlib的统计数据可视化库
import random # 生成随机数
from tqdm import tqdm # 显示进度条
import shutil # 文件操作，如复制和删除
import json # JSON数据处理
import gc # 垃圾回收接口


# 设置随机种子函数
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # 如果使用GPU
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True # 确保CUDA操作确定性
        torch.backends.cudnn.benchmark = False # 禁用基准优化

# 默认设置固定种子
set_seed(42)
# 随机种子影响下面两个参数
    # 模型初始化​​ (1.5.5):
        # 种子决定LSTM层初始参数
        # ±2%的最终性能差异主要源于此
        # 从而影响最终的预测准确率
    # ​进化策略​​ (2.1):
        # 种子决定策略探索方向
        # 造成±15%的交易绩效波动
        # 从而影响最终的交易策略收益率

warnings.filterwarnings('ignore')  # 忽略警告
sns.set()  # 设置seaborn默认样式

# ====================== 性能优化组件 ======================
class PerformanceOptimizer:
    """内存和计算优化器"""
    def __init__(self):
        self.memory_threshold = 4 * 1024**3  # 4GB内存阈值
    
    def check_memory(self):
        """简化内存检查"""
        try:
            if torch.cuda.is_available(): # 获取GPU可用内存
                free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                if free_mem < 2:
                    return True
            return False
        except:
            return False
    
    def optimize_data_loading(self, tickers):
        """优化数据加载策略"""
        if len(tickers) > 5:
            st.info("检测到大量股票，启用分块处理模式")
            return self.chunk_processing(tickers)
        return tickers
    
    def chunk_processing(self, tickers, chunk_size=3):
        """分块处理股票数据"""
        results = []
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            results.extend(chunk)
        return results

# ====================== 量化分析模块 ======================
class QuantitativeAnalyzer:
    """高级量化分析工具"""
    def __init__(self, returns, risk_free_rate=0.02):
        self.returns = returns # 收益率数据
        self.risk_free_rate = risk_free_rate # 无风险利率
    
    def sharpe_ratio(self):
        """计算夏普比率"""
        excess_returns = self.returns - self.risk_free_rate
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-7) # 避免除零
    
    def sortino_ratio(self):
        """计算索提诺比率"""
        downside_returns = self.returns[self.returns < 0] # 只考虑负收益
        if len(downside_returns) == 0:
            return 0
        downside_std = np.std(downside_returns)
        return (np.mean(self.returns) - self.risk_free_rate) / (downside_std + 1e-7)
    
    def max_drawdown(self):
        """计算最大回撤"""
        cumulative = np.cumprod(1 + self.returns)  # 累积收益
        peak = np.maximum.accumulate(cumulative) # 累积峰值
        drawdown = (peak - cumulative) / peak # 回撤计算
        return np.max(drawdown) # 最大回撤
    
    def value_at_risk(self, confidence=0.95):
        """计算在险价值(VaR)"""
        return np.percentile(self.returns, 100 * (1 - confidence))
    
    def expected_shortfall(self, confidence=0.95):
        """计算预期短缺(ES)"""
        var = self.value_at_risk(confidence)
        return np.mean(self.returns[self.returns <= var])
    
    def performance_report(self):
        """生成完整绩效报告"""
        return {
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Max Drawdown': self.max_drawdown(),
            'VaR (95%)': self.value_at_risk(),
            'Expected Shortfall (95%)': self.expected_shortfall()
        }

# ====================== 结果存储系统 ======================
class ResultStorage:
    """结构化结果存储系统"""
    def __init__(self, base_dir='results'):
        self.base_dir = base_dir
        self.session_id = f"session_{int(time.time())}"
        self.full_path = os.path.join(base_dir, self.session_id) # 使用时间戳创建唯一会话ID
        self.create_directories()
    
    def create_directories(self):
        """创建标准目录结构"""
        os.makedirs(self.full_path, exist_ok=True)
        os.makedirs(os.path.join(self.full_path, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(self.full_path, 'trading'), exist_ok=True)
        os.makedirs(os.path.join(self.full_path, 'feature_importance'), exist_ok=True)
        os.makedirs(os.path.join(self.full_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.full_path, 'analytics'), exist_ok=True)
        os.makedirs(os.path.join(self.full_path, 'scalers'), exist_ok=True)  # 保存归一化器
    
    def save_prediction(self, ticker, df):
        """保存预测结果到CSV"""
        path = os.path.join(self.full_path, 'predictions', f'{ticker}_predictions.csv')
        df.to_csv(path, index=False)
    
    def save_trading_record(self, ticker, df):
        """保存交易记录到CSV"""
        path = os.path.join(self.full_path, 'trading', f'{ticker}_trades.csv')
        df.to_csv(path, index=False)
    
    def save_feature_importance(self, ticker, df):
        """保存特征重要性结果"""
        path = os.path.join(self.full_path, 'feature_importance', f'{ticker}_features.csv')
        df.to_csv(path, index=False)
    
    def save_model(self, ticker, model):
        """保存模型"""
        path = os.path.join(self.full_path, 'models', f'{ticker}_model.pth')
        torch.save(model.state_dict(), path)
    
    def save_scaler(self, ticker, scaler_X, scaler_y):
        """保存归一化器"""
        scaler_path = os.path.join(self.full_path, 'scalers', f'{ticker}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
    
    def load_scaler(self, ticker):
        """加载归一化器"""
        scaler_path = os.path.join(self.full_path, 'scalers', f'{ticker}_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                return scalers['scaler_X'], scalers['scaler_y']
        return None, None
    
    def save_performance_report(self, ticker, report):
        """保存绩效报告"""
        path = os.path.join(self.full_path, 'analytics', f'{ticker}_performance.json')
        with open(path, 'w') as f:
            json.dump(report, f, indent=4)
    
    def save_session_summary(self, summary):
        """保存会话总结"""
        path = os.path.join(self.full_path, 'session_summary.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=4)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 技术指标计算函数
def calculate_technical_indicators(data, start_date=None, end_date=None):
    """
    计算技术指标，用于增强LSTM模型的输入特征
    
    技术指标包括:
    - 移动平均线(MA5, MA10, MA20)
    - 相对强弱指数(RSI)
    - 移动平均收敛发散(MACD)
    - 成交量加权平均价格(VWAP)
    - 布林带(Bollinger Bands)
    - 相对大盘表现
    - 价格变化率(ROC)
    - 平均真实波幅(ATR)
    """
    # 添加日期特征
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    
    # 移动平均线
    data['MA5'] = data['Close'].shift(1).rolling(window=5, min_periods=1).mean()
    data['MA10'] = data['Close'].shift(1).rolling(window=10, min_periods=1).mean()
    data['MA20'] = data['Close'].shift(1).rolling(window=20, min_periods=1).mean()
    
    # RSI指标
    close_prev = data['Close'].shift(1)
    delta = data['Close'] - close_prev  # 当日收盘价 - 前一日收盘价
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-7)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD指标
    close_prev = data['Close'].shift(1)
    exp1 = close_prev.ewm(span=12, adjust=False).mean()
    exp2 = close_prev.ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    
    # VWAP指标
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / (data['Volume'].cumsum() + 1e-7)
    
    # 布林带
    period = 20
    close_prev = data['Close'].shift(1)
    data['SMA'] = close_prev.rolling(window=period, min_periods=1).mean()
    # 标准差应该基于价格变化
    price_changes = close_prev.diff()
    data['Std_dev'] = price_changes.rolling(window=period, min_periods=1).std()
    data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
    data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']
    
    # 相对大盘表现
    if start_date and end_date:
        try:
            benchmark_data = yf.download('SPY', start=start_date, end=end_date)['Close']
            data['Relative_Performance'] = (data['Close'] / benchmark_data.values) * 100
        except:
            data['Relative_Performance'] = np.nan
    
    # ROC指标
    close_prev = data['Close'].shift(1)
    data['ROC'] = ((data['Close'] - close_prev) / close_prev) * 100
    
    # ATR指标
    high_low_range = data['High'] - data['Low']
    high_close_prev = abs(data['High'] - data['Close'].shift(1))
    low_close_prev = abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low_range, high_close_prev, low_close_prev], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
    data[['Close_yes', 'Open_yes', 'High_yes', 'Low_yes']] = data[['Close', 'Open', 'High', 'Low']].shift(2) # 前一天价格数据
    
    # 删除包含NaN值的行
    data = data.dropna()
    
    return data

# ====================== 数据获取模块 ======================
# akshare数据源
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    st.warning("AKShare not available. Install with: pip install akshare")

# 导入AKShare数据
def get_akshare_enhanced_data(ticker, start_date, end_date):
    """使用01.py逻辑的增强版AKShare数据获取"""
    try:
        from datetime import timedelta
        
        # 扩展日期范围确保技术指标计算
        extended_start_date = start_date - timedelta(days=365)
        
        def convert_to_standard_format(df, ticker, start_date, end_date):
            """统一数据格式"""
            column_lower_mapping = {col.lower(): col for col in df.columns}
            
            column_mappings = {
                'trade_date': 'Date',
                'ts_code': 'Ticker',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume',
                'volume': 'Volume',
                'date': 'Date'
            }
            
            for src, dest in column_mappings.items():
                src_lower = src.lower()
                if src_lower in column_lower_mapping:
                    df.rename(columns={column_lower_mapping[src_lower]: dest}, inplace=True)
            
            if 'Date' not in df.columns:
                date_candidates = ['trade_date', 'date', 'datetime']
                for candidate in date_candidates:
                    if candidate in df.columns:
                        df.rename(columns={candidate: 'Date'}, inplace=True)
                        break
            
            if 'Date' in df.columns:
                date_formats = ['%Y%m%d', '%Y-%m-%d']
                for fmt in date_formats:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce')
                        df = df.dropna(subset=['Date'])
                        if not df.empty:
                            break
                    except:
                        continue
            
            if 'Date' in df.columns:
                if df['Date'].dtype != 'datetime64[ns]':
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.sort_values('Date', ascending=True)
                df = df.set_index('Date')
            
            df['Ticker'] = ticker
            
            if not df.empty and start_date and end_date:
                try:
                    df = df.loc[start_date:end_date]
                except:
                    pass
            
            if 'Volume' not in df.columns:
                volume_candidates = ['Volume', 'volume', 'vol', '成交量']
                for candidate in volume_candidates:
                    if candidate in df.columns:
                        df['Volume'] = df[candidate]
                        break
            
            required_columns = ['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = np.nan
            
            return df[required_columns]
        
        def get_akshare_raw_data(ticker, start_date, end_date):
            """从AKShare获取原始股票数据"""
            try:
                if ticker.endswith(('.SH', '.SZ')):
                    stock_code = ticker.split('.')[0]
                    start_date_int = int(start_date.strftime('%Y%m%d'))
                    end_date_int = int(end_date.strftime('%Y%m%d'))
                    
                    df = ak.stock_zh_a_hist(
                        symbol=stock_code, 
                        period="daily", 
                        start_date=start_date_int, 
                        end_date=end_date_int, 
                        adjust="qfq"
                    )
                    if '日期' not in df.columns and 'date' in df.columns:
                        df = df.rename(columns={"date": "日期"})
                    df = df.rename(columns={
                        "日期": "Date", "开盘": "Open", "收盘": "Close",
                        "最高": "High", "最低": "Low", "成交量": "Volume"
                    })
                
                elif ticker.endswith('.HK'):
                    stock_code = ticker.split('.')[0]
                    start_date_int = int(start_date.strftime('%Y%m%d'))
                    end_date_int = int(end_date.strftime('%Y%m%d'))
                    
                    try:
                        df = ak.stock_hk_hist(
                            symbol=stock_code, 
                            period="daily", 
                            start_date=start_date_int, 
                            end_date=end_date_int, 
                            adjust="qfq"
                        )
                    except:
                        # 如果港股API失败，尝试使用美股API
                        df = ak.stock_us_daily(symbol=stock_code, adjust="qfq")
                    
                    if '日期' not in df.columns and 'date' in df.columns:
                        df = df.rename(columns={"date": "日期"})
                    df = df.rename(columns={
                        "日期": "Date", "开盘": "Open", "收盘": "Close",
                        "最高": "High", "最低": "Low", "成交量": "Volume"
                    })
                
                else:
                    df = ak.stock_us_daily(symbol=ticker, adjust="qfq")
                    if 'date' in df.columns:
                        df = df.rename(columns={"date": "Date"})
                    df = df.rename(columns={
                        "open": "Open", "close": "Close",
                        "high": "High", "low": "Low", "volume": "Volume"
                    })
                
                if df.empty:
                    raise ValueError(f"AKShare未返回 {ticker} 的数据")
                
                return convert_to_standard_format(df, ticker, start_date, end_date)
            
            except Exception as e:
                raise Exception(f"AKShare数据获取失败 {ticker}: {str(e)}")
        
        # 获取数据
        data = get_akshare_raw_data(ticker, extended_start_date, end_date)
        
        if data.empty:
            raise ValueError(f"未获取到 {ticker} 数据")
        
        # 计算技术指标
        data = calculate_technical_indicators_enhanced(data)
        
        # 返回请求时间范围
        return data.loc[start_date:end_date]
    
    except Exception as e:
        raise Exception(f"增强版AKShare错误 {ticker}: {str(e)}")

def calculate_technical_indicators_enhanced(data):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # === 1. 添加日期特征 ===
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    
    if 'Close' not in data.columns:
        raise ValueError("缺少'Close'列，无法计算技术指标")
    
    # 移动平均线 - 使用前一日收盘价计算当日MA
    for window in [5, 10, 20, 50, 200]:
        try:
            # 使用shift(1)确保使用前一日数据计算当日指标
            data[f'MA{window}'] = data['Close'].shift(1).rolling(
                window=window, min_periods=1
            ).mean()
        except Exception as e:
            data[f'MA{window}'] = np.nan
    
    # === 3. RSI计算 ===
    try:
        # 使用前一日收盘价计算涨跌幅，避免未来数据
        close_prev = data['Close'].shift(1)
        delta = data['Close'] - close_prev  # 当日收盘价 - 前一日收盘价
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # 使用前一日数据计算RSI
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss.replace(0, 1e-7))
        data['RSI'] = 100 - (100 / (1 + rs))
    except Exception as e:
        data['RSI'] = np.nan
    
    # === 4. 使用前一日收盘价 ===
    try:
        # 使用前一日收盘价计算EMA
        close_prev = data['Close'].shift(1)
        exp1 = close_prev.ewm(span=12, adjust=False).mean()
        exp2 = close_prev.ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    except Exception as e:
        data['MACD'] = data['Signal_Line'] = data['MACD_Histogram'] = np.nan
    
    # === 5. VWAP计算 ===
    try:
        # VWAP基于当日数据
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum().replace(0, 1e-7)
    except Exception as e:
        data['VWAP'] = np.nan
    
    # === 6. ROC计算  ===
    try:
        # 使用前一日收盘价计算变化率
        close_prev = data['Close'].shift(1)
        data['ROC'] = ((data['Close'] - close_prev) / close_prev) * 100
    except Exception as e:
        data['ROC'] = np.nan
    
    # === 7. ATR计算 ===
    try:
        # ATR基于当日高低价和前一日收盘价
        high_low_range = data['High'] - data['Low']
        high_close_prev = abs(data['High'] - data['Close'].shift(1))
        low_close_prev = abs(data['Low'] - data['Close'].shift(1))
        true_range = pd.concat([high_low_range, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
    except Exception as e:
        data['ATR'] = np.nan
    
    # === 8. 布林带计算 ===
    try:
        period = 20
        # 使用前一日收盘价计算布林带
        close_prev = data['Close'].shift(1)
        data['SMA'] = close_prev.rolling(window=period, min_periods=1).mean()
        # 标准差应该基于价格变化，而不是价格本身
        price_changes = close_prev.diff()
        data['Std_dev'] = price_changes.rolling(window=period, min_periods=1).std()
        data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
        data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']
    except Exception as e:
        data['Upper_band'] = data['Lower_band'] = data['SMA'] = data['Std_dev'] = np.nan
    
    # === 9. 前一天价格数据 ===
    try:
        # 前一天价格应该是shift(2)，因为shift(1)是当日价格
        data[['Close_prev', 'Open_prev', 'High_prev', 'Low_prev']] = data[
            ['Close', 'Open', 'High', 'Low']
        ].shift(2)  # 修复：使用shift(2)而不是shift(1)
    except Exception as e:
        pass
    
    return data

# 其实是没招了，原本计划是两个一起写的，后面写的跑不出来

# 从Yahoo Finance获取股票原始数据
def get_stock_data_from_yahoo(ticker, start_date, end_date):
    """从Yahoo Finance获取股票数据"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        return data
    except Exception as e:
        raise Exception(f"Yahoo Finance error for {ticker}: {str(e)}")

# 统一的数据获取函数
def get_stock_data(ticker, start_date, end_date, data_source='yahoo', tushare_api_key=None):
    """根据选择的数据源获取股票数据"""
    try:
        if data_source == 'yahoo':
            data = get_stock_data_from_yahoo(ticker, start_date, end_date)
        elif data_source == 'akshare':
            data = get_akshare_enhanced_data(ticker, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        # 对于非增强版数据源，应用标准技术指标计算
        if data_source != 'akshare':
            data = calculate_technical_indicators(data, start_date, end_date)
        
        return data
    except Exception as e:
        raise Exception(f"Data retrieval error for {ticker}: {str(e)}")

# 清理并格式化CSV文件
def clean_csv_files(file_path):
    df = pd.read_csv(file_path)            
    df = df.drop([0, 1]).reset_index(drop=True)            
    df = df.rename(columns={'Price': 'Date'})            
    df.to_csv(file_path, index=False)

# 下载并预处理多只股票数据
def download_and_preprocess_data(tickers, start_date, end_date, lang='en', data_source='yahoo'):
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True) # 创建数据目录
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        try:
            if lang == 'zh':
                status_text.text(f"处理 {ticker} 中... ({i+1}/{len(tickers)})")
            else:
                status_text.text(f"Processing {ticker}... ({i+1}/{len(tickers)})")
                
            stock_data = get_stock_data(ticker, start_date, end_date, data_source)
            stock_data.to_csv(f'{data_folder}/{ticker}.csv')
            clean_csv_files(f'{data_folder}/{ticker}.csv')
            progress_bar.progress((i+1)/len(tickers))
            
            # 手动清理内存
            del stock_data
            gc.collect() # 强制垃圾回收
            
        except Exception as e:
            st.error(f"Error processing {ticker}: {str(e)}")
    
    if lang == 'zh':
        status_text.text("所有股票数据处理完成！")
        st.success("数据准备阶段完成！")
    else:
        status_text.text("All stock data downloaded and preprocessed!")
        st.success("Data preparation phase completed!")

# ====================== 特征工程与数据清洗 ======================
# 提取特征工程数据
def format_feature(data):
    # 基础特征
    base_features = [
        'Volume', 'Year', 'Month', 'Day', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
        'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR'
    ]
    
    # 前一天价格特征（兼容不同命名）
    prev_price_features = []
    if 'Close_yes' in data.columns:
        prev_price_features.extend(['Close_yes', 'Open_yes', 'High_yes', 'Low_yes'])
    elif 'Close_prev' in data.columns:
        prev_price_features.extend(['Close_prev', 'Open_prev', 'High_prev', 'Low_prev'])
    
    # 增强版AKShare特有特征
    enhanced_features = []
    if 'MACD_Histogram' in data.columns:
        enhanced_features.append('MACD_Histogram')
    if 'Signal_Line' in data.columns:
        enhanced_features.append('Signal_Line')
    if 'ROC' in data.columns:
        enhanced_features.append('ROC')
    if 'MA50' in data.columns:
        enhanced_features.append('MA50')
    if 'MA200' in data.columns:
        enhanced_features.append('MA200')
    if 'Range' in data.columns:
        enhanced_features.append('Range')
    if 'Mid_Price' in data.columns:
        enhanced_features.append('Mid_Price')
    
    # 合并所有可用特征
    all_features = base_features + prev_price_features + enhanced_features
    
    # 只使用存在的特征
    available_features = [f for f in all_features if f in data.columns]
    
    X = data[available_features].iloc[1:]
    y = data['Close'].pct_change().iloc[1:]
    return X, y

# ====================== 1、LSTM模型数据获取 ======================
# ====================== 1.1、LSTM模型时间序列数据转换 ======================
# 创建时间序列窗口数据
def prepare_data(data, n_steps):
    """
    将数据准备为时间序列格式，用于LSTM输入
    
    参数:
        data: 输入数据
        n_steps: 时间步长（历史窗口大小）
    
    返回:
        X: 3D数组 [样本数, 时间步长, 特征数]
        y: 对应目标值
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps]) # 输入序列
        y.append(data[i + n_steps]) # 输出值
    return np.array(X), np.array(y)

# ====================== 1.2、LSTM模型模型结构定义 ======================
# LSTM模型结构定义
class LSTMModel(nn.Module):
    """
    LSTM模型类 - 用于时间序列预测
    输入参数:
        input_size: 输入特征数量
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        output_size: 输出大小
        dropout: 防止过拟合的dropout率
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size) # 权重初始化受随机种子影响
        # nn.Linear层的权重初始化使用PyTorch默认随机初始化
        # 不同种子 → 不同的初始权重值 → 影响模型收敛速度和最终性能

    def forward(self, x):
        """前向传播"""
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0)) # LSTM前向传播
        out = self.fc(out[:, -1, :]) # 只取最后一个时间步的输出
        return out

# ====================== 1.3、数据获取与验证 ======================
# 从本地文件获取预处理后的股票数据
def get_stock_data_for_prediction(ticker, data_dir='data'):
    file_path = os.path.join(data_dir, f'{ticker}.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for {ticker} not found at {file_path}")
        
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    
    if data.empty:
        raise ValueError(f"Data file for {ticker} is empty.")
    
    if len(data.columns) < 5:
        raise ValueError(f"Data file for {ticker} has insufficient columns.")
    
    return data

# ====================== 1.4、可视化模块 ======================
# 不重要
# 可视化预测结果
def visualize_predictions(ticker, data, test_indices, predictions, actual_prices, save_dir, lang='en', tomorrow_pred=None):
    test_indices = test_indices[:len(predictions)]
    actual_prices = actual_prices[:len(predictions)]
    # 计算评估指标
    mse = np.mean((predictions - actual_prices) ** 2)
    rmse = np.sqrt(mse) # 均方根误差
    mae = np.mean(np.abs(predictions - actual_prices))
    accuracy = 1 - np.mean(np.abs(predictions - actual_prices) / actual_prices)
    
    metrics = {'rmse': rmse, 'mae': mae, 'accuracy': accuracy}
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # ====================== 绘制实际价格和预测价格 ======================
    ax.plot(test_indices, actual_prices, label='Actual Price', color='blue')
    
    # 绘制预测价格，为连接绿点做准备
    if tomorrow_pred:
        date, price = tomorrow_pred
        # 扩展预测折线到明日预测点
        extended_indices = np.append(test_indices, date)
        extended_predictions = np.append(predictions, price)
        ax.plot(extended_indices, extended_predictions, label='Predicted Price', color='red', linestyle='--')
    else:
        ax.plot(test_indices, predictions, label='Predicted Price', color='red', linestyle='--')
    
    # ====================== 添加明日预测点 ======================
    if tomorrow_pred:
        date, price = tomorrow_pred
        # 使用绿色圆点标记明日预测（覆盖在折线上）
        ax.plot(date, price, 'go', markersize=10, label='Tomorrow Prediction')
        
        # 添加价格标注
        ax.text(
            date + pd.Timedelta(days=2),  # 向右偏移2天
            price, 
            f'{price:.2f}',
            fontsize=12, 
            color='green',
            verticalalignment='center',
            horizontalalignment='left'  # 左对齐，文本从点右侧开始
        )
    
    # ====================== 图表样式设置 ======================
    if lang == 'zh':
        ax.set_title(f'{ticker} 股票价格预测', fontsize=16)
        ax.set_xlabel('日期', fontsize=14)
        ax.set_ylabel('价格', fontsize=14)
        metric_text = f'均方根误差: {rmse:.4f}\n平均绝对误差: {mae:.4f}\n准确率: {accuracy:.2%}'
    else:
        ax.set_title(f'{ticker} Stock Price Prediction', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Price', fontsize=14)
        metric_text = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nAccuracy: {accuracy:.2%}'
    
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # 放大指标文本字号
    ax.text(
        0.02, 0.95, metric_text, 
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)  # 添加半透明背景
    )
    
    # 自动调整X轴范围
    if tomorrow_pred:
        # 包含明日预测点
        ax.set_xlim(min(test_indices) if len(test_indices) > 0 else date - pd.Timedelta(days=10), 
                   date + pd.Timedelta(days=3))
    
    st.pyplot(fig) # 在Streamlit中显示图表
    
    return metrics

# ====================== 1.5、LSTM训练与预测核心模块 ======================
# LSTM模型训练与预测全流程
def train_and_predict_lstm(ticker, data, X, y, n_steps=60, num_epochs=50, batch_size=32, learning_rate=0.001, lang='en', storage=None):
    """
    LSTM模型完整训练与预测流程
    步骤:
    1. 数据验证与归一化
    2. 创建时间序列窗口
    3. 数据集划分
    4. 数据加载器创建
    5. 模型初始化
    6. 定义损失函数和优化器
    7. 训练循环
    8. 模型预测
    9. 可视化结果（不重要）
    10. 特征重要性分析（不重要）
    11. 模型保存（不重要）
    """
    
 # ========== 1.5.1 数据验证与归一化 ==========
    # 验证数据有效性
    if len(y) == 0:
        raise ValueError(f"No valid data found for {ticker}.")
    
    # 检查数据点是否足够
    if len(y) < n_steps:
        raise ValueError(f"Insufficient data points for {ticker}.")
    
    # 创建归一化器
    scaler_y = MinMaxScaler()
    scaler_X = MinMaxScaler()
    
    # 对特征和目标变量进行归一化
    #归一化作用：提升模型收敛速度、避免数值差异对模型的影响、提高模型稳定性、统一特征分布
    scaler_y.fit(y.values.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    X_scaled = scaler_X.fit_transform(X)

 # ========== 1.5.2 创建时间序列窗口 ==========
    # 将数据转换为LSTM需要的序列格式
    X_seq, y_seq = prepare_data(X_scaled, n_steps)
    # 对齐目标变量
    y_seq = y_scaled[n_steps-1:-1]

 # ========== 1.5.3 数据集划分 ==========
    # 按80/20比例划分训练集和验证集
    train_per = 0.8
    split_index = int(train_per * len(X_seq))
    
    # 训练集
    X_train = X_seq[:split_index]
    y_train = y_seq[:split_index]
    
    # 验证集（注意时间序列连续性处理）
    X_val = X_seq[split_index-n_steps+1:]
    y_val = y_seq[split_index-n_steps+1:]

 # ========== 1.5.4 数据加载器创建 ==========
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

 # ========== 1.5.5 模型初始化 ==========
    # 创建LSTM模型实例
    model = LSTMModel(
        input_size=X_train.shape[2],  # 输入特征维度
        hidden_size=50,               # LSTM隐藏单元数
        num_layers=2,                 # LSTM层数
        output_size=1,                # 输出维度
        dropout=0.2                   # Dropout比例
    ).to(device)

 # ========== 1.5.6 定义损失函数和优化器 ==========
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器
    # 学习率调度器（每50轮衰减为0.1倍）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 初始化训练跟踪变量
    train_losses = []
    val_losses = []
    
    # 设置进度显示
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # 准备损失图表
    if lang == 'zh':
        chart_title = ['训练损失', '验证损失']
    else:
        chart_title = ['Training Loss', 'Validation Loss']
        
    chart = st.line_chart(pd.DataFrame(columns=chart_title))

 # ========== 1.5.7 训练循环 ==========
    for epoch in range(num_epochs):
        # ---------- 训练阶段 ----------
        model.train()
        epoch_train_loss = 0
        
        # 遍历训练批次
        for inputs, targets in train_loader:
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 参数更新
            optimizer.step()
            
            # 累计损失
            epoch_train_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---------- 验证阶段 ----------
        model.eval()
        epoch_val_loss = 0
        
        # 禁用梯度计算
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                epoch_val_loss += val_loss.item()
        
        # 计算平均验证损失
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # ---------- 更新进度和图表 ----------
        # 显示训练进度
        if lang == 'zh':
            progress_text.text(f"轮次 {epoch+1}/{num_epochs} - 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")
        else:
            progress_text.text(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
        
        # 更新进度条
        progress_bar.progress((epoch+1)/num_epochs)
        
        # 更新损失图表
        loss_data = pd.DataFrame({
            chart_title[0]: [avg_train_loss],
            chart_title[1]: [avg_val_loss]
        }, index=[epoch])
        chart.add_rows(loss_data)
        
        # 更新学习率
        scheduler.step()

 # ========== 1.5.8 模型预测 ==========
    model.eval()
    predictions = []
    test_indices = []
    predict_percentages = []
    actual_percentages = []

    # 禁用梯度计算
    with torch.no_grad():
        # 对验证集进行预测
        for i in range(1 + split_index, len(X_scaled) + 1):
            # 准备输入数据
            x_input = torch.tensor(
                X_scaled[i - n_steps:i].reshape(1, n_steps, X_train.shape[2]), 
                dtype=torch.float32
            ).to(device)
            
            # 模型预测
            y_pred = model(x_input)
            
            # 反归一化
            y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
            
            # 计算预测价格
            predicted_price = (1 + y_pred[0][0]) * data['Close'].iloc[i - 2]
            predictions.append(predicted_price)
            
            # 记录索引和百分比
            test_indices.append(data.index[i - 1])
            predict_percentages.append(y_pred[0][0] * 100)
            actual_percentages.append(y[i - 1] * 100)

    # 使用最后n_steps个数据点预测明日价格
    last_window = X_scaled[-n_steps:].reshape(1, n_steps, X_train.shape[2])
    last_window_tensor = torch.tensor(last_window, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        model.eval()
        y_pred_tomorrow = model(last_window_tensor)
        y_pred_tomorrow = scaler_y.inverse_transform(y_pred_tomorrow.cpu().numpy().reshape(-1, 1))
    
    # 计算明日预测价格
    last_known_price = data['Close'].iloc[-1]
    predicted_price_tomorrow = (1 + y_pred_tomorrow[0][0]) * last_known_price
    
    # 获取明日日期（假设是最后一个交易日的下一个工作日）
    last_date = data.index[-1]
    tomorrow_date = last_date + pd.tseries.offsets.BDay(1)

 # ========== 1.5.9 可视化累积收益 ==========
    # 绘制累积收益图
    fig, ax = plt.subplots(figsize=(14, 7))
    actual_earnings = np.cumsum(actual_percentages)
    predicted_earnings = np.cumsum(predict_percentages)
    
    # 设置图表语言
    ax.plot(test_indices, actual_earnings, label='Actual Cumulative Return', color='blue')
    ax.plot(test_indices, predicted_earnings, label='Predicted Cumulative Return', color='red')
    ax.set_title(f'{ticker} Cumulative Return')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    
    # 添加图例和网格
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ========== 1.7.10 创建预测结果数据框 ==========
    # 创建预测结果数据框
    df = pd.DataFrame({
        'Date': test_indices,
        'Actual_Price': data['Close'].loc[test_indices].values[:len(predictions)],
        'Predicted_Price': predictions
    })
    
    # 添加明日预测行
    tomorrow_row = pd.DataFrame({
        'Date': [tomorrow_date],
        'Actual_Price': [np.nan],
        'Predicted_Price': [predicted_price_tomorrow]
    })
    df = pd.concat([df, tomorrow_row], ignore_index=True)
    
    # 显示最后10条预测结果（包含明日预测）
    st.dataframe(df)
    
    # ========== 1.7.11 可视化预测结果 ==========
    # 绘制预测结果与实际价格对比图
    metrics = visualize_predictions(
        ticker, 
        data, 
        test_indices, 
        predictions, 
        data['Close'].loc[test_indices].values[:len(predictions)], 
        '', 
        lang,
        tomorrow_pred=(tomorrow_date, predicted_price_tomorrow)  # 新增参数
    )

    # ========== 1.7.12 特征重要性分析 ==========
    def analyze_feature_importance():
        """分析特征对预测结果的影响程度"""
        model.eval()
        feature_count = X_train.shape[2]
        importances = np.zeros(feature_count)
        features = list(X.columns)
        
        # 基准预测（所有特征正常）
        base_input = torch.tensor(
            X_scaled[-n_steps:].reshape(1, n_steps, X_train.shape[2]), 
            dtype=torch.float32
        ).to(device)
        base_output = model(base_input)
        base_value = scaler_y.inverse_transform(base_output.cpu().detach().numpy().reshape(-1, 1))[0][0]
        
        # 分析每个特征的重要性
        for i in range(feature_count):
            # 复制数据并将当前特征置零
            modified_data = X_scaled[-n_steps:].copy()
            modified_data[:, i] = 0
            
            # 使用修改后的数据进行预测
            modified_input = torch.tensor(
                modified_data.reshape(1, n_steps, feature_count), 
                dtype=torch.float32
            ).to(device)
            modified_output = model(modified_input)
            modified_value = scaler_y.inverse_transform(modified_output.cpu().detach().numpy().reshape(-1, 1))[0][0]
            
            # 计算特征重要性（预测值变化）
            importances[i] = abs(modified_value - base_value)
        
        # 归一化重要性
        normalized = importances / (importances.sum() + 1e-7) * 100
        return pd.DataFrame({'特征': features, '重要性': normalized})
    
    # 执行特征重要性分析
    feature_importance = analyze_feature_importance()
    
    # ========== 1.7.13 可视化特征重要性 ==========
    # 创建特征重要性图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.set_title(f'{ticker} Feature Importance Analysis')
    ax.set_xlabel('Importance (%)')
    
    # 提取前10个重要特征
    top_features = feature_importance.sort_values('重要性', ascending=False).head(10)
    sns.barplot(x='重要性', y='特征', data=top_features, palette='viridis', ax=ax)
    st.pyplot(fig)
    
    # 保存特征重要性结果
    if storage:
        storage.save_feature_importance(ticker, feature_importance)
    
    # ========== 1.7.14 模型保存 ==========
    if storage:
        storage.save_model(ticker, model)
        storage.save_scaler(ticker, scaler_X, scaler_y)  # 新增：保存归一化器
    
    # ========== 1.7.15 返回模型和归一化器 ==========
    return metrics, model, scaler_X, scaler_y

# ====================== 1.6、批量预测执行模块 ======================
# 批量执行多只股票的预测
def run_prediction(tickers, num_epochs=50, lang='en', storage=None):
    prediction_metrics = {}
    # 模型缓存字典
    model_cache = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker_name in enumerate(tickers):
        try:
            # 根据语言设置状态文本
            if lang == 'zh':
                status_text.text(f"预测 {ticker_name} 中... ({i+1}/{len(tickers)})")
            else:
                status_text.text(f"Predicting {ticker_name}... ({i+1}/{len(tickers)})")
                
            stock_data = get_stock_data_for_prediction(ticker_name)
            stock_features = format_feature(stock_data)
            metrics, model, scaler_X, scaler_y = train_and_predict_lstm(  # 修改：接收返回的模型和归一化器
                ticker_name, stock_data, *stock_features, num_epochs=num_epochs, lang=lang, storage=storage
            )
            
            # 缓存模型和归一化器
            model_cache[ticker_name] = {
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y
            }
            
            prediction_metrics[ticker_name] = metrics
            progress_bar.progress((i+1)/len(tickers))
            
        except Exception as e:
            st.error(f"Error processing {ticker_name}: {str(e)}")
    
    # 将模型缓存保存到session state
    st.session_state.model_cache = model_cache
    
    if prediction_metrics:
        metrics_df = pd.DataFrame(prediction_metrics).T
        if lang == 'zh':
            metrics_df = metrics_df.rename(columns={
                'rmse': '均方根误差',
                'mae': '平均绝对误差',
                'accuracy': '准确率'
            })
        st.dataframe(metrics_df)
    else:
        st.warning("No stocks successfully predicted")
        return {}

    if prediction_metrics:
        if lang == 'zh':
            summary = {
                '平均准确率': np.mean([m['accuracy'] for m in prediction_metrics.values()]),
                '最佳股票': max(prediction_metrics.items(), key=lambda x: x[1]['accuracy'])[0],
                '最差股票': min(prediction_metrics.items(), key=lambda x: x[1]['accuracy'])[0],
                '平均均方根误差': metrics_df['均方根误差'].mean(),
                '平均平均绝对误差': metrics_df['平均绝对误差'].mean()
            }
        else:
            summary = {
                'Average Accuracy': np.mean([m['accuracy'] for m in prediction_metrics.values()]),
                'Best Stock': max(prediction_metrics.items(), key=lambda x: x[1]['accuracy'])[0],
                'Worst Stock': min(prediction_metrics.items(), key=lambda x: x[1]['accuracy'])[0],
                'Average RMSE': metrics_df['rmse'].mean(),
                'Average MAE': metrics_df['mae'].mean()
            }
    else:
        summary = {}
    
    # 根据语言设置汇总标题
    st.subheader("预测汇总:" if lang == 'zh' else "Prediction Summary:")
    if summary:
        for key, value in summary.items():
            st.text(f"{key}: {value}")
    else:
        st.text("没有可用的预测结果" if lang == 'zh' else "No prediction results available")
    
    return prediction_metrics

# ====================== 2、强化学习交易策略 ======================
# ====================== 2.1 深度进化策略 ======================
class Deep_Evolution_Strategy:
    def __init__(self, weights, reward_function, population_size, sigma, learning_rate):
        """
        初始化深度进化策略优化器
        
        参数:
        weights: 当前模型权重
        reward_function: 评估权重性能的奖励函数
        population_size: 种群大小（扰动样本数量）
        sigma: 扰动强度（噪声标准差）
        learning_rate: 权重更新学习率
        """
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

 # 生成扰动后的权重种群
    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            # 对每个权重添加高斯噪声扰动
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

 # 获取当前最优权重
    def get_weights(self):
        return self.weights

 # 执行进化策略训练
    def train(self, epoch=100, print_every=1, lang='en'):
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            # 步骤1: 生成随机扰动种群
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    # 为每个权重矩阵生成随机扰动
                    x.append(np.random.randn(*w.shape))# 扰动生成受随机种子影响
                    # np.random.randn生成的高斯噪声扰动
                    # 不同种子 → 不同的进化方向 → 交易策略优化路径差异
                population.append(x)


            # 步骤2: 评估每个扰动个体的性能
            for k in range(self.population_size):
                # 获取扰动后的权重
                weights_population = self._get_weight_from_population(self.weights, population[k])
                # 使用奖励函数评估性能
                rewards[k] = self.reward_function(weights_population)
            # 步骤3: 标准化奖励（适应度缩放）
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            # 步骤4: 更新权重（基于适应度的加权平均）
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T
                )
            # 打印训练进度
            if (i + 1) % print_every == 0:
                if lang == 'zh':
                    st.text(f'迭代 {i + 1}. 奖励: {self.reward_function(self.weights):.6f}')
                else:
                    st.text(f'Iteration {i + 1}. Reward: {self.reward_function(self.weights):.6f}')
        # 显示总训练时间  
        if lang == 'zh':
            st.text(f'训练耗时: {time.time() - lasttime:.2f} 秒')
        else:
            st.text(f'Training time: {time.time() - lasttime:.2f} seconds')

# ====================== 2.2 交易策略模型======================
class Model:
    def __init__(self, input_size, layer_size, output_size):
        """
        初始化交易策略神经网络模型
        
        参数:
        input_size: 输入特征维度（时间窗口大小）
        layer_size: 隐藏层神经元数量
        output_size: 输出维度（3种动作：持有/买入/卖出）
        """
        # 初始化权重：输入层到隐藏层，隐藏层到输出层，以及隐藏层偏置
        self.weights = [
            np.random.randn(input_size, layer_size), # 输入层到隐藏层权重
            np.random.randn(layer_size, output_size), # 隐藏层到输出层权重
            np.random.randn(1, layer_size), # 隐藏层偏置
        ]

 # 前向传播预测动作概率
    def predict(self, inputs):
        # 计算隐藏层输出：inputs * W1 + bias
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        # 计算输出层：隐藏层输出 * W2
        decision = np.dot(feed, self.weights[1])
        return decision

 # 获取当前模型权重
    def get_weights(self):
        return self.weights
 #设置模型权重
    def set_weights(self, weights):
        self.weights = weights

# ====================== 2.3 交易代理 ======================
class Agent:
    # 进化策略默认参数
    POPULATION_SIZE = 15  # 种群大小
    SIGMA = 0.1          # 扰动强度
    LEARNING_RATE = 0.03  # 学习率

    def __init__(self, model, window_size, trend, skip, initial_money, ticker, lang='en'):
        """
        初始化交易代理
        
        参数:
        model: 交易策略模型
        window_size: 时间窗口大小（用于状态表示）
        trend: 股票价格序列
        skip: 跳过的天数（交易频率）
        initial_money: 初始资金
        ticker: 股票代码
        lang: 语言设置
        """
        self.model = model
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend # 价格序列
        self.skip = skip # 跳过步长
        self.initial_money = initial_money # 初始资金
        self.ticker = ticker # 股票代码
        self.lang = lang
        # 创建深度进化策略优化器
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward, # 奖励函数
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

 # 根据当前状态选择动作（0=持有, 1=买入, 2=卖出
    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))
        return np.argmax(decision[0])

 # 获取时间点t的状态表示（价格变化序列）
    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        # 获取价格窗口（如果数据不足开头部分用第一个价格填充）
        block = self.trend[d: t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0: t + 1]
        # 计算连续价格变化
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res])
 # 评估权重性能的奖励函数（基于模拟交易）
    def get_reward(self, weights):
        initial_money = self.initial_money
        starting_money = initial_money
        self.model.weights = weights  # 设置模型权重
        state = self.get_state(0) # 初始状态
        inventory = [] # 持仓记录
        
        # 模拟交易过程
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state) # 选择动作
            next_state = self.get_state(t + 1) # 获取下一状态

            # 有足够资金执行买入
            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t]) # 记录买入价格
                starting_money -= self.trend[t] # 扣除资金

            # 有持仓执行卖出动作
            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0) # 取出最早的买入记录
                starting_money += self.trend[t] # 增加资金

            state = next_state # 更新状态

        # 计算最终回报率
        return ((starting_money - initial_money) / initial_money) * 100

 # 训练交易策略模型
    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every=checkpoint, lang=self.lang)

 # 使用训练好的模型执行交易
    def buy(self, storage=None):
        initial_money = self.initial_money
        state = self.get_state(0)
        starting_money = initial_money
        states_sell = [] # 卖出点记录
        states_buy = [] # 买入点记录
        inventory = [] # 当前持仓
        transaction_history = [] # 交易历史
        daily_returns = []  # 每日收益率（用于风险分析）

        # 遍历价格序列执行交易
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state) # 选择动作
            next_state = self.get_state(t + 1)  # 获取下一状态

            # 执行买入
            if action == 1 and initial_money >= self.trend[t]:
                inventory.append(self.trend[t]) # 记录买入价格
                initial_money -= self.trend[t] # 扣除资金
                states_buy.append(t) # 记录买入时间点
                
                # 记录交易历史
                if self.lang == 'zh':
                    transaction_history.append({
                        '日期': t,
                        '操作': '买入',
                        '价格': self.trend[t],
                        '投资回报': 0,
                        '总余额': initial_money
                    })
                else:
                    transaction_history.append({
                        'Day': t,
                        'Action': 'buy',
                        'Price': self.trend[t],
                        'Return': 0,
                        'Total Balance': initial_money
                    })

            # 执行卖出
            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)  # 取出最早的买入记录
                initial_money += self.trend[t] # 增加资金
                states_sell.append(t) # 记录卖出时间点
                try:
                    # 计算本次交易的收益率
                    invest = ((self.trend[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0 # 避免除零错误
                    
                # 记录交易历史
                if self.lang == 'zh':
                    transaction_history.append({
                        '日期': t,
                        '操作': '卖出',
                        '价格': self.trend[t],
                        '投资回报': invest,
                        '总余额': initial_money
                    })
                else:
                    transaction_history.append({
                        'Day': t,
                        'Action': 'sell',
                        'Price': self.trend[t],
                        'Return': invest,
                        'Total Balance': initial_money
                    })

            # 计算每日收益率
            if t > 0:
                # 计算前一日总价值
                prev_value = starting_money + sum([self.trend[t] - p for p in inventory])
                # 计算当前总价值
                current_value = initial_money + sum([self.trend[t] - p for p in inventory])
                # 计算日收益率
                daily_return = (current_value - prev_value) / prev_value
                daily_returns.append(daily_return)

            state = next_state # 更新状态

        # 显示交易记录表格
        df_transaction = pd.DataFrame(transaction_history)
        st.dataframe(df_transaction)
        
        # 保存交易记录
        if storage:
            storage.save_trading_record(self.ticker, df_transaction)

        # 计算总体投资回报
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        
        # ====================== 风险调整绩效报告 ======================
        if daily_returns:
            # 使用量化分析器计算风险指标
            analyzer = QuantitativeAnalyzer(np.array(daily_returns))
            performance_report = analyzer.performance_report()
            
            # 显示绩效报告
            if self.lang == 'zh':
                st.subheader("风险调整绩效报告:")
                report_df = pd.DataFrame({
                    '夏普比率': [performance_report['Sharpe Ratio']],
                    '索提诺比率': [performance_report['Sortino Ratio']],
                    '最大回撤': [performance_report['Max Drawdown']],
                    '在险价值 (95%)': [performance_report['VaR (95%)']],
                    '预期短缺 (95%)': [performance_report['Expected Shortfall (95%)']]
                })
            else:
                st.subheader("Risk-Adjusted Performance Report:")
                report_df = pd.DataFrame(performance_report, index=[0])
                
            st.dataframe(report_df)
            
            # 保存绩效报告
            if storage:
                storage.save_performance_report(self.ticker, performance_report)

        # 创建交易策略可视化图表
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # ====================== 图六 ======================
        # 使用图表标签
        if self.lang == 'zh':
            ax.plot(self.trend, label='Stock Price', color='blue')
            ax.plot(states_buy, [self.trend[i] for i in states_buy], '^', markersize=10, color='green', label='Buy')
            ax.plot(states_sell, [self.trend[i] for i in states_sell], 'v', markersize=10, color='red', label='Sell')
            ax.set_title(f'{self.ticker} Trading Strategy - Total Gain: ${total_gains:.2f} ({invest:.2f}%)')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price')
        else:
            ax.plot(self.trend, label='Stock Price', color='blue')
            ax.plot(states_buy, [self.trend[i] for i in states_buy], '^', markersize=10, color='green', label='Buy')
            ax.plot(states_sell, [self.trend[i] for i in states_sell], 'v', markersize=10, color='red', label='Sell')
            ax.set_title(f'{self.ticker} Trading Strategy - Total Gain: ${total_gains:.2f} ({invest:.2f}%)')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price')
            
        ax.legend() # 显示图例
        ax.grid(True) # 显示网格
        st.pyplot(fig) # 在Streamlit中显示图表

        # 返回交易结果
        return states_buy, states_sell, total_gains, invest

# ====================== 交易策略执行 ======================
# ====================== 单只股票策略处理 ======================
def process_stock(ticker, window_size=30, initial_money=10000, iterations=100, lang='en', storage=None):
    try:
        # 根据语言设置子标题
        if lang == 'zh':
            st.subheader(f"处理 {ticker} 交易策略")
        else:
            st.subheader(f"Processing {ticker} Trading Strategy")
            
        df = get_stock_data_for_prediction(ticker)
        close = df['Close'].values.tolist()

        skip = 1
        model = Model(input_size=window_size, layer_size=500, output_size=3)
        agent = Agent(model=model, window_size=window_size, trend=close, 
                     skip=skip, initial_money=initial_money, ticker=ticker, lang=lang)
        
        # 根据语言设置训练文本
        if lang == 'zh':
            st.text(f"训练 {ticker} 交易代理...")
        else:
            st.text(f"Training {ticker} trading agent...")
            
        agent.fit(iterations=iterations, checkpoint=10)

        states_buy, states_sell, total_gains, invest = agent.buy(storage)
        
        # 根据语言设置成功信息
        if lang == 'zh':
            st.success(f"{ticker} 交易完成! 总收益: ${total_gains:.2f} ({invest:.2f}%)")
        else:
            st.success(f"{ticker} trading completed! Total Gain: ${total_gains:.2f} ({invest:.2f}%)")
        
        # 根据语言设置返回结果
        if lang == 'zh':
            return {
                '股票代码': ticker,
                '总收益': total_gains,
                '投资回报率': invest,
                '买入次数': len(states_buy),
                '卖出次数': len(states_sell)
            }
        else:
            return {
                'Ticker': ticker,
                'Total Gain': total_gains,
                'ROI (%)': invest,
                'Buy Actions': len(states_buy),
                'Sell Actions': len(states_sell)
            }
        
    except Exception as e:
        st.error(f"Error processing {ticker}: {e}")
        return None

# ====================== 批量交易执行 ======================
def run_trading(tickers, initial_money=10000, iterations=100, lang='en', storage=None):
    trading_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        # 根据语言设置状态文本
        if lang == 'zh':
            status_text.text(f"执行 {ticker} 交易策略... ({i+1}/{len(tickers)})")
        else:
            status_text.text(f"Executing {ticker} trading strategy... ({i+1}/{len(tickers)})")
            
        result = process_stock(ticker, initial_money=initial_money, iterations=iterations, lang=lang, storage=storage)
        if result:
            trading_results.append(result)
        progress_bar.progress((i+1)/len(tickers))
    
    if trading_results:
        trading_df = pd.DataFrame(trading_results)
        if lang == 'zh':
            trading_df = trading_df.rename(columns={
                '股票代码': '股票代码',
                '总收益': '总收益',
                '投资回报率': '投资回报率(%)',
                '买入次数': '买入次数',
                '卖出次数': '卖出次数'
            })
        st.dataframe(trading_df)
    else:
        st.warning("No trading strategies successfully executed")
    
    if trading_results and lang == 'zh':
        st.subheader("交易汇总:")
        st.text(f"交易股票数量: {len(trading_results)}")
        st.text(f"平均回报率: {trading_df['投资回报率(%)'].mean():.2f}%")
        st.text(f"表现最佳: {trading_df.loc[trading_df['投资回报率(%)'].idxmax()]['股票代码']} "
                f"({trading_df['投资回报率(%)'].max():.2f}%)")
        st.text(f"表现最差: {trading_df.loc[trading_df['投资回报率(%)'].idxmin()]['股票代码']} "
                f"({trading_df['投资回报率(%)'].min():.2f}%)")
    elif trading_results and lang == 'en':
        st.subheader("Trading Summary:")
        st.text(f"Stocks traded: {len(trading_results)}")
        st.text(f"Average ROI: {trading_df['ROI (%)'].mean():.2f}%")
        st.text(f"Best performer: {trading_df.loc[trading_df['ROI (%)'].idxmax()]['Ticker']} "
                f"({trading_df['ROI (%)'].max():.2f}%)")
        st.text(f"Worst performer: {trading_df.loc[trading_df['ROI (%)'].idxmin()]['Ticker']} "
                f"({trading_df['ROI (%)'].min():.2f}%)")
    else:
        if lang == 'zh':
            st.text("没有可用的交易结果")
        else:
            st.text("No trading results available")
    
    return trading_results


# ====================== 明日价格预测 ======================
def get_tomorrow_prediction(ticker, lang='en'):
    try:
        # 获取股票数据
        df = get_stock_data_for_prediction(ticker)
        actual_price = df['Close'].iloc[-1]
        
        # 尝试从缓存加载模型和归一化器
        if 'model_cache' in st.session_state and ticker in st.session_state.model_cache:
            model_info = st.session_state.model_cache[ticker]
            model = model_info['model']
            scaler_X = model_info['scaler_X']
            scaler_y = model_info['scaler_y']
            
            if lang == 'zh':
                st.info(f"使用已训练的模型预测 {ticker} 明日价格")
            else:
                st.info(f"Using trained model to predict tomorrow's price for {ticker}")
        else:
            # 没有缓存模型，使用备用方案
            if lang == 'zh':
                st.warning(f"未找到 {ticker} 的已训练模型，使用简单移动平均预测")
            else:
                st.warning(f"No trained model found for {ticker}, using simple moving average")
            
            # 使用简单移动平均作为备用方案
            predicted_price = df['Close'].rolling(window=5).mean().iloc[-1]
            predicted_change = (predicted_price - actual_price) / actual_price * 100
            
            if lang == 'zh':
                if predicted_change > 3:
                    recommendation = "强烈买入"
                elif predicted_change > 1:
                    recommendation = "买入"
                elif predicted_change < -3:
                    recommendation = "强烈卖出"
                elif predicted_change < -1:
                    recommendation = "卖出"
                else:
                    recommendation = "持有"
                
                return {
                    '股票代码': ticker,
                    '当前价格': actual_price,
                    '明日预测价格': predicted_price,
                    '预测涨跌幅': f"{predicted_change:.2f}%",
                    '投资建议': recommendation
                }
            else:
                if predicted_change > 3:
                    recommendation = "Strong Buy"
                elif predicted_change > 1:
                    recommendation = "Buy"
                elif predicted_change < -3:
                    recommendation = "Strong Sell"
                elif predicted_change < -1:
                    recommendation = "Sell"
                else:
                    recommendation = "Hold"
                
                return {
                    'Ticker': ticker,
                    'Current Price': actual_price,
                    'Predicted Price': predicted_price,
                    'Predicted Change': f"{predicted_change:.2f}%",
                    'Recommendation': recommendation
                }
        
        # 准备特征
        X, _ = format_feature(df)
        
        # 归一化特征
        X_scaled = scaler_X.transform(X)
        
        # 准备时间序列数据
        n_steps = 60
        # 使用最后n_steps个数据点
        last_window = X_scaled[-n_steps:].reshape(1, n_steps, X_scaled.shape[1])
        last_window_tensor = torch.tensor(last_window, dtype=torch.float32).to(device)
        
        # 使用模型预测
        with torch.no_grad():
            model.eval()
            y_pred = model(last_window_tensor)
            y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
        
        # 计算预测价格
        predicted_price = (1 + y_pred[0][0]) * df['Close'].iloc[-1]
        predicted_change = (predicted_price - actual_price) / actual_price * 100
        
        # 生成投资建议
        if lang == 'zh':
            if predicted_change > 3:
                recommendation = "强烈买入"
            elif predicted_change > 1:
                recommendation = "买入"
            elif predicted_change < -3:
                recommendation = "强烈卖出"
            elif predicted_change < -1:
                recommendation = "卖出"
            else:
                recommendation = "持有"
            
            return {
                '股票代码': ticker,
                '当前价格': actual_price,
                '明日预测价格': predicted_price,
                '预测涨跌幅': f"{predicted_change:.2f}%",
                '投资建议': recommendation
            }
        else:
            if predicted_change > 3:
                recommendation = "Strong Buy"
            elif predicted_change > 1:
                recommendation = "Buy"
            elif predicted_change < -3:
                recommendation = "Strong Sell"
            elif predicted_change < -1:
                recommendation = "Sell"
            else:
                recommendation = "Hold"
            
            return {
                'Ticker': ticker,
                'Current Price': actual_price,
                'Predicted Price': predicted_price,
                'Predicted Change': f"{predicted_change:.2f}%",
                'Recommendation': recommendation
            }
    except Exception as e:
        if lang == 'zh':
            st.error(f"获取 {ticker} 预测时出错: {str(e)}")
        else:
            st.error(f"Error getting prediction for {ticker}: {str(e)}")
        return None

# 翻译文本
TEXTS = {
    "en": {
        "title": "📈 Stock Prediction and Trading Strategy System",
        "subtitle": "Predict stock prices and generate trading strategies using deep learning models",
        "sidebar_header": "Parameter Settings",
        "ticker_label": "Stock Tickers (comma separated)",
        "ticker_default": "AAPL,MSFT,GOOGL",
        "start_date": "Start Date",
        "end_date": "End Date",
        "initial_money": "Initial Capital ($)",
        "num_epochs": "Training Epochs",
        "iterations": "Strategy Iterations",
        "language": "Language",
        "instructions": "**Instructions:**",
        "instructions_list": [
            "1. Enter stock tickers (comma separated)",
            "2. Set date range",
            "3. Adjust other parameters",
            "4. Click Run to start analysis"
        ],
        "run_button": "Run Full Analysis",
        "run_prediction_button": "Run Prediction Only",
        "error_no_tickers": "Please enter at least one stock ticker",
        "data_prep": "Preparing data...",
        "training_model": "Training prediction models...",
        "generating_strategy": "Generating trading strategies...",
        "prediction_results": "Stock Prediction Results",
        "trading_results": "Trading Strategy Analysis",
        "tomorrow_prediction": "Tomorrow's Price Prediction & Recommendations",
        "stock_code": "Ticker",
        "current_price": "Current Price",
        "predicted_price": "Predicted Price",
        "predicted_change": "Predicted Change",
        "recommendation": "Recommendation",
        "analysis_complete": "Analysis completed!",
        "tickers": "Tickers",
        "total_return": "Total Return",
        "roi": "ROI (%)",
        "buy_count": "Buy Actions",
        "sell_count": "Sell Actions",
        "avg_return": "Average ROI:",
        "best_performer": "Best Performer:",
        "worst_performer": "Worst Performer:",
        "download_results": "Download Full Results",
        "advanced_options": "Advanced Options",
        "enable_incremental_training": "Enable Incremental Training",
        "enable_multi_timeframe": "Enable Multi-Timeframe Analysis",
        "backtest_timeframe": "Backtest Timeframe",
        "daily": "Daily",
        "weekly": "Weekly",
        "monthly": "Monthly",
        "randomness_control": "Randomness Control",
        "enable_randomness": "Enable Randomness",
        "random_seed": "Random Seed",
        "randomness_explanation": "**Randomness Explanation:**\n- Enabled: Results vary each run, more realistic\n- Disabled: Fixed results for reproducibility",
        "data_prep_complete": "Data preparation phase completed!",
        "feature_importance_analysis": "Feature Importance Analysis",
        "importance_percent": "Importance (%)",
        "rmse": "Root Mean Squared Error",
        "mae": "Mean Absolute Error",
        "accuracy": "Accuracy",
        "ticker": "Ticker",
        "total_gain": "Total Gain",
        "roi_percent": "ROI (%)",
        "buy_actions": "Buy Actions",
        "sell_actions": "Sell Actions",
        "stocks_traded": "Stocks traded",
        "average_roi": "Average ROI",
        "best_performer": "Best performer",
        "worst_performer": "Worst performer",
        "data_source": "Data Source",
        "data_source_yahoo": "Yahoo Finance",
        "data_source_akshare": "AKShare"
    },
    "zh": {
        "title": "📈 股票预测与交易策略系统",
        "subtitle": "使用深度学习模型预测股票价格并生成交易策略",
        "sidebar_header": "参数设置",
        "ticker_label": "股票代码（用逗号分隔）",
        "ticker_default": "AAPL,MSFT,GOOGL",
        "start_date": "开始日期",
        "end_date": "结束日期",
        "initial_money": "初始资金 ($)",
        "num_epochs": "训练轮数",
        "iterations": "策略迭代次数",
        "language": "语言",
        "instructions": "**使用说明:**",
        "instructions_list": [
            "1. 输入股票代码（多个用逗号分隔）",
            "2. 设置日期范围",
            "3. 调整其他参数",
            "4. 点击运行按钮开始分析"
        ],
        "run_button": "运行完整分析",
        "run_prediction_button": "仅获取明日预测",
        "error_no_tickers": "请输入至少一个股票代码",
        "data_prep": "数据准备中...",
        "training_model": "训练预测模型中...",
        "generating_strategy": "生成交易策略中...",
        "prediction_results": "股票预测结果",
        "trading_results": "交易策略分析",
        "tomorrow_prediction": "明日价格预测及投资建议",
        "stock_code": "股票代码",
        "current_price": "当前价格",
        "predicted_price": "预测价格",
        "predicted_change": "预测涨跌幅",
        "recommendation": "投资建议",
        "analysis_complete": "分析完成！",
        "tickers": "股票代码",
        "total_return": "总收益",
        "roi": "投资回报率(%)",
        "buy_count": "买入次数",
        "sell_count": "卖出次数",
        "avg_return": "平均回报率:",
        "best_performer": "表现最佳:",
        "worst_performer": "表现最差:",
        "download_results": "下载完整结果",
        "advanced_options": "高级选项",
        "enable_incremental_training": "启用增量训练",
        "enable_multi_timeframe": "启用多时间框架分析",
        "backtest_timeframe": "回测时间框架",
        "daily": "日线",
        "weekly": "周线",
        "monthly": "月线",
        "randomness_control": "随机性控制",
        "enable_randomness": "启用随机性",
        "random_seed": "随机种子",
        "randomness_explanation": "**随机性说明:**\n- 启用：每次运行结果不同，更贴近现实\n- 禁用：固定结果，便于复现",
        "data_prep_complete": "数据准备阶段完成！",
        "feature_importance_analysis": "特征重要性分析",
        "importance_percent": "重要性 (%)",
        "rmse": "均方根误差",
        "mae": "平均绝对误差",
        "accuracy": "准确率",
        "ticker": "股票代码",
        "total_gain": "总收益",
        "roi_percent": "投资回报率 (%)",
        "buy_actions": "买入次数",
        "sell_actions": "卖出次数",
        "stocks_traded": "交易股票数量",
        "average_roi": "平均回报率",
        "best_performer": "表现最佳",
        "worst_performer": "表现最差",
        "data_source": "数据源",
        "data_source_yahoo": "Yahoo Finance",
        "data_source_akshare": "AKShare"
    }
}

# 获取翻译文本的函数
def tr(key):
    lang = st.session_state.get('lang', 'en')
    return TEXTS[lang].get(key, key)

# 主界面布局函数
def create_header():
    st.title(tr("title"))
    st.markdown(f"### {tr('subtitle')}")
    st.markdown("---")

# 创建侧边栏
def create_sidebar():
    with st.sidebar:
        st.header(tr("sidebar_header"))
        
        # 语言切换选择器
        lang_options = {"English": "en", "中文": "zh"}
        default_lang = st.session_state.get('lang', 'en')
        default_index = 0 if default_lang == 'en' else 1
        selected_lang = st.selectbox(
            tr("language"), 
            options=list(lang_options.keys()), 
            index=default_index,
            key="lang_select"
        )
        st.session_state.lang = lang_options[selected_lang]
        
        # 数据源选择
        st.subheader(tr("data_source"))
        data_source_options = {
            tr("data_source_yahoo"): "yahoo",
            tr("data_source_akshare"): "akshare"
        }
        
        # 根据可用性过滤选项
        available_options = {}
        if True:  # Yahoo Finance总是可用
            available_options[tr("data_source_yahoo")] = "yahoo"
        if AKSHARE_AVAILABLE:
            available_options[tr("data_source_akshare")] = "akshare"
        
        selected_data_source = st.selectbox(
            tr("data_source"),
            options=list(available_options.keys()),
            key="data_source_select"
        )
        data_source = available_options[selected_data_source]
        

        
        # 随机性控制选项
        st.subheader(tr("randomness_control"))
        enable_randomness = st.checkbox(
            tr("enable_randomness"),
            value=False,
            help=tr("randomness_explanation")
        )
        
        if not enable_randomness:
            seed_value = st.number_input(
                tr("random_seed"),
                value=42,
                help="固定此值可确保结果可复现"
            )
            set_seed(seed_value)
        else:
            # 使用时间作为种子
            set_seed(int(time.time()))
        
        # 计算默认日期范围（当前时间往前一年）
        current_date = pd.to_datetime("today")
        default_end_date = current_date
        default_start_date = current_date - pd.DateOffset(years=1)
        
        # 股票和日期参数
        tickers_input = st.text_input(
            tr("ticker_label"), 
            tr("ticker_default"),
            key="tickers"
        )
        start_date = st.date_input(
            tr("start_date"), 
            default_start_date,
            key="start_date"
        )
        end_date = st.date_input(
            tr("end_date"), 
            default_end_date,
            key="end_date"
        )
        initial_money = st.number_input(
            tr("initial_money"), 
            10000, 1000000, 10000,
            key="initial_money"
        )
        num_epochs = st.slider(
            tr("num_epochs"), 
            10, 200, 50,
            key="num_epochs"
        )
        iterations = st.slider(
            tr("iterations"), 
            10, 200, 50,
            key="iterations"
        )
        
        # 高级选项
        with st.expander(tr("advanced_options")):
            incremental_training = st.checkbox(tr("enable_incremental_training"), value=True)
            multi_timeframe = st.checkbox(tr("enable_multi_timeframe"), value=False)
            backtest_timeframe = st.selectbox(tr("backtest_timeframe"), 
                                            [tr("daily"), tr("weekly"), tr("monthly")])
        
        st.markdown("---")
        st.markdown(tr("instructions"))
        for instruction in tr("instructions_list"):
            st.markdown(instruction)
    
    return tickers_input, start_date, end_date, initial_money, num_epochs, iterations, enable_randomness, data_source

# Streamlit主应用
def main():
    st.set_page_config(
        page_title=tr("title"),
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化语言设置
    if 'lang' not in st.session_state:
        st.session_state.lang = 'en'
    
    # 初始化结果存储系统
    if 'storage' not in st.session_state:
        st.session_state.storage = ResultStorage()
    
    create_header()
    tickers_input, start_date, end_date, initial_money, num_epochs, iterations, enable_randomness, data_source = create_sidebar()
    
    # 显示随机性状态
    if enable_randomness:
        if st.session_state.lang == 'zh':
            randomness_status = "随机模式 (结果不可复现)"
        else:
            randomness_status = "Random Mode (results not reproducible)"
    else:
        if st.session_state.lang == 'zh':
            randomness_status = "固定种子模式 (结果可复现)"
        else:
            randomness_status = "Fixed Seed Mode (results reproducible)"
    
    if st.session_state.lang == 'zh':
        st.info(f"**随机性状态:** {randomness_status}")
    else:
        st.info(f"**Randomness Status:** {randomness_status}")
    
    # 处理股票代码输入
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    # 创建操作按钮
    if st.button(tr("run_button"), use_container_width=True, key="run_full_analysis"):
        run_full_analysis(tickers, start_date, end_date, initial_money, num_epochs, iterations, data_source)
    
    # 明日预测按钮
    if st.button(tr("run_prediction_button"), use_container_width=True, key="run_prediction_only"):
        run_prediction_only(tickers)
    
    # ====================== 性能监控面板 ======================
    with st.expander("系统性能监控" if st.session_state.lang == 'zh' else "System Performance Monitor"):
        if st.button("检查系统资源" if st.session_state.lang == 'zh' else "Check System Resources"):
            optimizer = PerformanceOptimizer()
            low_memory = optimizer.check_memory()
            
            if low_memory:
                if st.session_state.lang == 'zh':
                    st.warning("⚠️ 系统内存不足 - 已启用分块处理模式")
                else:
                    st.warning("⚠️ Low memory - Enabling chunk processing mode")
            else:
                if st.session_state.lang == 'zh':
                    st.success("✅ 系统资源充足")
                else:
                    st.success("✅ System resources sufficient")
            
            if torch.cuda.is_available():
                if st.session_state.lang == 'zh':
                    st.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
                    st.info(f"GPU内存: {torch.cuda.mem_get_info()[1]/1024**3:.2f} GB")
                else:
                    st.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
                    st.info(f"GPU Memory: {torch.cuda.mem_get_info()[1]/1024**3:.2f} GB")
            else:
                if st.session_state.lang == 'zh':
                    st.warning("未检测到GPU - 使用CPU进行计算")
                else:
                    st.warning("No GPU detected - Using CPU for computation")
    
    # 显示当前语言状态
    st.sidebar.markdown("---")
    st.sidebar.info(f"Current language: {'English' if st.session_state.lang == 'en' else '中文'}")

def run_full_analysis(tickers, start_date, end_date, initial_money, num_epochs, iterations, data_source='yahoo'):
    if not tickers:
        st.error(tr("error_no_tickers"))
        return
    
    # 使用性能优化器
    with st.spinner("优化资源分配中..." if st.session_state.lang == 'zh' else "Optimizing resource allocation..."):
        optimizer = PerformanceOptimizer()
        optimized_tickers = optimizer.optimize_data_loading(tickers)
    
    with st.spinner(tr("data_prep")):
        download_and_preprocess_data(optimized_tickers, start_date, end_date, st.session_state.lang, data_source)
    
    st.header(tr("prediction_results"))
    with st.spinner(tr("training_model")):
        prediction_metrics = run_prediction(optimized_tickers, 
                                          num_epochs=num_epochs, 
                                          lang=st.session_state.lang,
                                          storage=st.session_state.storage)
    
    
    st.header(tr("trading_results"))
    with st.spinner(tr("generating_strategy")):
        trading_results = run_trading(optimized_tickers, 
                                     initial_money=initial_money, 
                                     iterations=iterations, 
                                     lang=st.session_state.lang,
                                     storage=st.session_state.storage)
    
    display_tomorrow_predictions(optimized_tickers, lang=st.session_state.lang)
    
    # 保存会话摘要
    summary = {
        'tickers': optimized_tickers,
        'start_date': str(start_date),
        'end_date': str(end_date),
        'initial_money': initial_money,
        'num_epochs': num_epochs,
        'iterations': iterations,
        'data_source': data_source,
        'timestamp': str(pd.Timestamp.now())
    }
    st.session_state.storage.save_session_summary(summary)
    
    # 提供结果下载
    st.success(tr("analysis_complete"))
    zip_path = shutil.make_archive(st.session_state.storage.full_path, 'zip', st.session_state.storage.full_path)
    with open(zip_path, "rb") as f:
        st.download_button(tr("download_results"), f, file_name=f"results_{int(time.time())}.zip")

def run_prediction_only(tickers):
    if not tickers:
        st.error(tr("error_no_tickers"))
        return
    
    display_tomorrow_predictions(tickers, lang=st.session_state.lang)
    

def display_tomorrow_predictions(tickers, lang='en'):
    st.header(tr("tomorrow_prediction"))
    st.markdown("---")
    
    # 创建列布局
    num_cols = 5
    cols = st.columns(num_cols)
    
    # 根据语言设置列标题
    if lang == 'zh':
        headers = ["股票代码", "当前价格", "预测价格", "预测涨跌幅", "投资建议"]
    else:
        headers = ["Ticker", "Current Price", "Predicted Price", "Predicted Change", "Recommendation"]
    
    # 添加标题行
    for i, header in enumerate(headers):
        cols[i % num_cols].markdown(f"**{header}**")
    
    # 获取并显示每只股票的预测
    for ticker in tickers:
        prediction = get_tomorrow_prediction(ticker, lang=lang)
        if prediction:
            # 创建一个新的列组
            cols = st.columns(num_cols)
            
            # 根据语言显示不同的键
            if lang == 'zh':
                cols[0].text(prediction['股票代码'])
                cols[1].text(f"${prediction['当前价格']:.2f}")
                cols[2].text(f"${prediction['明日预测价格']:.2f}")
            else:
                cols[0].text(prediction['Ticker'])
                cols[1].text(f"${prediction['Current Price']:.2f}")
                cols[2].text(f"${prediction['Predicted Price']:.2f}")
            
            # 处理涨跌幅显示（带颜色）
            if lang == 'zh':
                change_text = prediction['预测涨跌幅']
                rec_text = prediction['投资建议']
            else:
                change_text = prediction['Predicted Change']
                rec_text = prediction['Recommendation']
            
            # 解析涨跌幅
            change_value = float(change_text.replace('%', '').strip())
            
            # 设置涨跌幅颜色
            change_color = "red" if change_value < 0 else "green"
            change_display = f"<span style='color:{change_color}'>{change_text}</span>"
            cols[3].markdown(change_display, unsafe_allow_html=True)
            
            # 设置建议颜色
            rec_color_mapping = {
                "买入": "green", "Buy": "green", 
                "卖出": "red", "Sell": "red",
                "强烈买入": "darkgreen", "Strong Buy": "darkgreen",
                "强烈卖出": "darkred", "Strong Sell": "darkred",
                "持有": "black", "Hold": "black"
            }
            rec_color = rec_color_mapping.get(rec_text, "black")
            rec_display = f"<span style='color:{rec_color}'>{rec_text}</span>"
            cols[4].markdown(rec_display, unsafe_allow_html=True)
    
    st.markdown("---")

if __name__ == "__main__":
    main()