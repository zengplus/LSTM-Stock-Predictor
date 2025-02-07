import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
from visualization import (
    plot_stock_prediction,
    plot_training_loss,
    plot_cumulative_earnings,
    plot_accuracy_comparison
)

warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def get_stock_data(ticker, data_dir='data'):
    file_path = os.path.join(data_dir, f'{ticker}.csv')
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return data

def format_feature(data):
    features = [
        'Volume', 'Year', 'Month', 'Day', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
        'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR',
        'Close_yes', 'Open_yes', 'High_yes', 'Low_yes'
    ]
    X = data[features].iloc[1:]
    y = data['Close'].pct_change().iloc[1:]
    return X, y

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

def visualize_predictions(ticker, data, predict_result, test_indices, predictions, actual_percentages, save_dir):
    actual_prices = data['Close'].loc[test_indices].values
    predicted_prices = np.array(predictions)
    
    mse = np.mean((predicted_prices - actual_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted_prices - actual_prices))
    accuracy = 1 - np.mean(np.abs(predicted_prices - actual_prices) / actual_prices)
    
    metrics = {'rmse': rmse, 'mae': mae, 'accuracy': accuracy}
    plot_stock_prediction(ticker, test_indices, actual_prices, predicted_prices, metrics, save_dir)
    
    return metrics

def train_and_predict_lstm(ticker, data, X, y, save_dir, n_steps=60, num_epochs=500, batch_size=32, learning_rate=0.001):
    # 数据归一化和准备部分
    scaler_y = MinMaxScaler()
    scaler_X = MinMaxScaler()
    scaler_y.fit(y.values.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    X_scaled = scaler_X.fit_transform(X)

    X_train, y_train = prepare_data(X_scaled, n_steps)
    y_train = y_scaled[n_steps-1:-1]

    train_per = 0.8
    split_index = int(train_per * len(X_train))
    X_val = X_train[split_index-n_steps+1:]
    y_val = y_train[split_index-n_steps+1:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]

    # PyTorch数据准备
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    train_losses = []
    val_losses = []

    with tqdm(total=num_epochs, desc=f"Training {ticker}", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            # 训练和验证循环
            model.train()
            epoch_train_loss = 0
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    epoch_val_loss += val_loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            pbar.set_postfix({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})
            pbar.update(1)
            scheduler.step()

    # 使用可视化工具绘制损失曲线
    plot_training_loss(ticker, train_losses, val_losses, save_dir)

    # 预测
    model.eval()
    predictions = []
    test_indices = []
    predict_percentages = []
    actual_percentages = []

    with torch.no_grad():
        for i in range(1 + split_index, len(X_scaled) + 1):
            x_input = torch.tensor(X_scaled[i - n_steps:i].reshape(1, n_steps, X_train.shape[2]), 
                                 dtype=torch.float32).to(device)
            y_pred = model(x_input)
            y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
            predictions.append((1 + y_pred[0][0]) * data['Close'].iloc[i - 2])
            test_indices.append(data.index[i - 1])
            predict_percentages.append(y_pred[0][0] * 100)
            actual_percentages.append(y[i - 1] * 100)

    # 使用可视化工具绘制累积收益率曲线
    plot_cumulative_earnings(ticker, test_indices, actual_percentages, predict_percentages, save_dir)

    predict_result = {str(date): pred / 100 for date, pred in zip(test_indices, predict_percentages)}
    return predict_result, test_indices, predictions, actual_percentages

def save_predictions_with_indices(ticker, test_indices, predictions, save_dir):
    df = pd.DataFrame({
        'Date': test_indices,
        'Prediction': predictions
    })

    file_path = os.path.join(save_dir, 'predictions', f'{ticker}_predictions.pkl')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(df, file)

    print(f'Saved predictions for {ticker} to {file_path}')

def predict(ticker_name, stock_data, stock_features, save_dir, epochs=500, batch_size=32, learning_rate=0.001):
    all_predictions_lstm = {}
    prediction_metrics = {}

    print(f"\nProcessing {ticker_name}")
    data = stock_data
    X, y = stock_features
    
    predict_result, test_indices, predictions, actual_percentages = train_and_predict_lstm(
        ticker_name, data, X, y, save_dir, num_epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    all_predictions_lstm[ticker_name] = predict_result
    
    metrics = visualize_predictions(ticker_name, data, predict_result, test_indices, predictions, actual_percentages, save_dir)
    prediction_metrics[ticker_name] = metrics
    
    save_predictions_with_indices(ticker_name, test_indices, predictions, save_dir)

    # 保存预测指标
    os.makedirs(os.path.join(save_dir, 'output'), exist_ok=True)
    metrics_df = pd.DataFrame(prediction_metrics).T
    metrics_df.to_csv(os.path.join(save_dir, 'output', f'{ticker_name}_prediction_metrics.csv'))
    print("\nPrediction metrics summary:")
    print(metrics_df.describe())

    # 使用可视化工具绘制准确度对比图
    plot_accuracy_comparison(prediction_metrics, save_dir)

    # 生成汇总报告
    summary = {
        'Average Accuracy': np.mean([m['accuracy'] * 100 for m in prediction_metrics.values()]),
        'Best Stock': max(prediction_metrics.items(), key=lambda x: x[1]['accuracy'])[0],
        'Worst Stock': min(prediction_metrics.items(), key=lambda x: x[1]['accuracy'])[0],
        'Average RMSE': metrics_df['rmse'].mean(),
        'Average MAE': metrics_df['mae'].mean()
    }

    # 保存汇总报告
    with open(os.path.join(save_dir, 'output', f'{ticker_name}_prediction_summary.txt'), 'w') as f:
        for key, value in summary.items():
            f.write(f'{key}: {value}\n')

    print("\nPrediction Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return metrics

if __name__ == "__main__":
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',       # 科技
        'JPM', 'BAC', 'C', 'WFC', 'GS',                # 金融
        'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',            # 医药
        'XOM', 'CVX', 'COP', 'SLB', 'BKR',             # 能源
        'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',         # 消费
        'CAT', 'DE', 'MMM', 'GE', 'HON'                # 工业
    ]

    save_dir = 'results'  # 设置保存目录
    for ticker_name in tickers:
        stock_data = get_stock_data(ticker_name)
        stock_features = format_feature(stock_data)
        predict(
            ticker_name=ticker_name,
            stock_data=stock_data,
            stock_features=stock_features,
            save_dir=save_dir
        )