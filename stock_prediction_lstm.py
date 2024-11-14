import os
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置PyTorch设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 股票列表
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',       # 科技
    'JPM', 'BAC', 'C', 'WFC', 'GS',                # 金融
    'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',            # 医药
    'XOM', 'CVX', 'COP', 'SLB', 'BKR',             # 能源
    'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',         # 消费
    'CAT', 'DE', 'MMM', 'GE', 'HON'                # 工业
]

# 配置参数
NUM_FEATURES_TO_KEEP = 9
N_STEPS = 60
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'

# 读取单个股票数据
def get_stock_data(ticker):
    file_path = os.path.join('data', f'{ticker}.csv')
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return data

# 读取所有股票数据
stock_data = {ticker: get_stock_data(ticker) for ticker in tickers}

# 格式化特征数据
def format_feature(data):
    features = [
        'Volume', 'Year', 'Month', 'Day', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
        'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR',
        'Close_yes', 'Open_yes', 'High_yes', 'Low_yes'
    ]
    X = data[features].iloc[1:]
    y = data['Close'].pct_change().iloc[1:]
    return X, y

# 为所有股票准备特征数据
stock_features = {ticker: format_feature(data) for ticker, data in stock_data.items()}

# 准备数据函数，用于时间序列
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# 定义LSTM模型类，增加Dropout层以减少过拟合
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
def visualize_predictions(ticker, data, predict_result, test_indices, predictions, actual_percentages):
    """
    可视化LSTM预测结果
    
    Args:
        ticker: 股票代码
        data: 原始股票数据
        predict_result: 预测变化百分比的字典
        test_indices: 测试集日期索引
        predictions: 预测价格列表
        actual_percentages: 实际变化百分比列表
    """
    # 获取测试集的实际价格和预测价格
    actual_prices = data['Close'].loc[test_indices].values
    predicted_prices = np.array(predictions)
    
    # 创建图形
    plt.figure(figsize=(15, 7))
    
    # 绘制实际价格
    plt.plot(test_indices, actual_prices, 
             label='Actual Price', 
             color='blue', 
             linewidth=2, 
             alpha=0.7)
    
    # 绘制预测价格
    plt.plot(test_indices, predicted_prices, 
             label='LSTM Prediction', 
             color='red', 
             linewidth=2, 
             linestyle='--', 
             alpha=0.7)
    
    # 计算预测准确度指标
    mse = np.mean((predicted_prices - actual_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted_prices - actual_prices))
    
    # 设置图形标题和标签
    plt.title(f'{ticker} Stock Price Prediction\nRMSE: {rmse:.2f}, MAE: {mae:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加预测准确度的文字说明
    accuracy = 1 - np.mean(np.abs(predicted_prices - actual_prices) / actual_prices)
    plt.text(0.02, 0.95, 
             f'Prediction Accuracy: {accuracy*100:.2f}%', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('pic/predictions', exist_ok=True)
    plt.savefig(f'pic/predictions/{ticker}_prediction.png')
    plt.close()
    
    return {
        'rmse': rmse,
        'mae': mae,
        'accuracy': accuracy
    }

# 使用LSTM模型进行训练和预测
def train_and_predict_lstm(ticker, data, X, y, n_steps=60, num_epochs=500, batch_size=32, learning_rate=0.001):
    # 数据归一化处理
    scaler_y = MinMaxScaler()
    scaler_X = MinMaxScaler()
    scaler_y.fit(y.values.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    X_scaled = scaler_X.fit_transform(X)

    X_train, y_train = prepare_data(X_scaled, n_steps)
    y_train = y_scaled[n_steps-1:-1]

    # 划分训练和验证数据
    train_per = 0.8
    split_index = int(train_per * len(X_train))
    X_val = X_train[split_index-n_steps+1:]
    y_val = y_train[split_index-n_steps+1:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size=X_train.shape[2], hidden_size=50, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 学习率调度器

    # 记录训练和验证的损失
    train_losses = []
    val_losses = []

    # 使用 tqdm 显示进度条
    with tqdm(total=num_epochs, desc=f"Training {ticker}", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            # 训练模式
            model.train()
            epoch_train_loss = 0
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            # 记录每个epoch的训练损失
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证模式
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    epoch_val_loss += val_loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # 更新 tqdm 显示的信息
            pbar.set_postfix({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})
            pbar.update(1)
            scheduler.step()

    # 绘制训练和验证的损失变化图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {ticker}')
    plt.legend()
    plt.grid(True)
    os.makedirs('pic/loss', exist_ok=True)
    plt.savefig(f'pic/loss/{ticker}_loss.png')
    plt.close()

    # 累积收益计算部分
    model.eval()
    predictions = []
    test_indices = []
    predict_percentages = []
    actual_percentages = []

    with torch.no_grad():
        for i in range(1 + split_index, len(X_scaled) + 1):
            x_input = torch.tensor(X_scaled[i - n_steps:i].reshape(1, n_steps, X_train.shape[2]), dtype=torch.float32).to(device)
            y_pred = model(x_input)
            y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
            predictions.append((1 + y_pred[0][0]) * data['Close'].iloc[i - 2])
            test_indices.append(data.index[i - 1])  # 确保使用完整日期索引
            predict_percentages.append(y_pred[0][0] * 100)
            actual_percentages.append(y[i - 1] * 100)

    cumulative_naive_percentage = np.cumsum(actual_percentages)
    cumulative_lstm_percentage = np.cumsum(
        [a if p > 0 else 0 for p, a in zip(predict_percentages, actual_percentages)]
    )

    # 绘制累积收益率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(test_indices, cumulative_naive_percentage, marker='o', markersize=3, linestyle='-', color='blue',
             label='Naive Strategy')
    plt.plot(test_indices, cumulative_lstm_percentage, marker='o', markersize=3, linestyle='-', color='orange',
             label='LSTM Strategy')
    plt.title(f'Cumulative Earnings Percentages for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs('pic/earnings', exist_ok=True)
    plt.savefig(f'pic/earnings/{ticker}_cumulative.png')
    plt.close()

    # 返回预测结果
    predict_result = {str(date): pred / 100 for date, pred in zip(test_indices, predict_percentages)}
    return predict_result, test_indices, predictions, actual_percentages

# 为每个股票进行预测并保存结果
all_predictions_lstm = {}
prediction_metrics = {}

def save_predictions_with_indices(ticker, test_indices, predictions):
    # 合并 `test_indices` 和 `predictions` 为一个 DataFrame
    df = pd.DataFrame({
        'Date': test_indices,
        'Prediction': predictions
    })

    # 保存 DataFrame 到文件
    file_path = os.path.join('predictions', f'{ticker}_predictions.pkl')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(df, file)

    print(f'Saved predictions for {ticker} to {file_path}')


for ticker in tickers:
    print(f"\nProcessing {ticker}")
    data = stock_data[ticker]
    X, y = stock_features[ticker]
    
    # 训练LSTM并获取预测
    predict_result, test_indices, predictions, actual_percentages = train_and_predict_lstm(ticker, data, X, y)
    all_predictions_lstm[ticker] = predict_result
    
    # 可视化预测结果
    metrics = visualize_predictions(ticker, data, predict_result, test_indices, predictions, actual_percentages)
    prediction_metrics[ticker] = metrics
    
    # 保存预测结果
    file_path = os.path.join('predictions', f'{ticker}_predictions.pkl')
    with open(file_path, 'wb') as file:
        save_predictions_with_indices(ticker, test_indices, predictions)

# 保存预测指标
folder_path = 'output'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
else:
    print(f"Folder '{folder_path}' already exists.")

metrics_df = pd.DataFrame(prediction_metrics).T
metrics_df.to_csv('output/prediction_metrics.csv')
print("\nPrediction metrics summary:")
print(metrics_df.describe())

# 可视化所有股票的预测准确度对比
plt.figure(figsize=(15, 6))
accuracies = [metrics['accuracy'] * 100 for metrics in prediction_metrics.values()]
plt.bar(prediction_metrics.keys(), accuracies)
plt.title('Prediction Accuracy Across Stocks')
plt.xlabel('Stock')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pic/predictions/accuracy_comparison.png')
plt.close()

# 添加汇总报告
summary = {
    'Average Accuracy': np.mean(accuracies),
    'Best Stock': max(prediction_metrics.items(), key=lambda x: x[1]['accuracy'])[0],
    'Worst Stock': min(prediction_metrics.items(), key=lambda x: x[1]['accuracy'])[0],
    'Average RMSE': metrics_df['rmse'].mean(),
    'Average MAE': metrics_df['mae'].mean()
}

with open('output/prediction_summary.txt', 'w') as f:
    for key, value in summary.items():
        f.write(f'{key}: {value}\n')

print("\nPrediction Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")