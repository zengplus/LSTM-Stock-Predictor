import os
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Stock list
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech
           'JPM', 'BAC', 'C', 'WFC', 'GS',  # Finance
           'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',  # Pharma
           'XOM', 'CVX', 'COP', 'SLB', 'BKR',  # Energy
           'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',  # Consumer
           'CAT', 'DE', 'MMM', 'GE', 'HON']  # Industrial

num_features_to_keep = 9
n_steps = 60
start_date = '2021-01-01'
end_date = '2023-01-01'

# Load stock data from stock_data directory
def get_stock_data(ticker):
    file_path = os.path.join('data', f'{ticker}.csv')
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return data

# Load all stock data
stock_data = {ticker: get_stock_data(ticker) for ticker in tickers}

# Format features for the model
def format_feature(data):
    features = ['Volume', 'Year', 'Month', 'Day', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
                'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR',
                'Close_yes', 'Open_yes', 'High_yes', 'Low_yes']
    X = data[features].iloc[1:]
    y = data['Close'].pct_change().iloc[1:]
    return X, y

# Prepare features for all stocks
stock_features = {ticker: format_feature(data) for ticker, data in stock_data.items()}

# LSTM model class
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Train and predict using LSTM
def train_and_predict_lstm(ticker, data, X, y, n_steps=60, num_epochs=500, batch_size=32, learning_rate=0.001):
    scaler_y = MinMaxScaler()
    scaler_X = MinMaxScaler()
    
    scaler_y.fit(y.values.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    X_scaled = scaler_X.fit_transform(X)

    X_train, y_train = prepare_data(X_scaled, n_steps)
    y_train = y_scaled[n_steps-1:-1]

    train_per = 0.8
    split_index = int(train_per * len(X_train))
    X_test = X_train[split_index-n_steps+1:]
    y_test = y_train[split_index-n_steps+1:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    model = LSTMModel(input_size=X_train.shape[2], hidden_size=50, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    model.eval()
    predictions = []
    test_indices = []
    predict_percentages = []
    actual_percentages = []

    with torch.no_grad():
        for i in range(1+split_index, len(X_scaled)+1):
            x_input = torch.tensor(X_scaled[i - n_steps:i].reshape(1, n_steps, X_train.shape[2]),
                                   dtype=torch.float32).to(device)
            y_pred = model(x_input)
            y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
            predictions.append((1 + y_pred[0][0]) * data['Close'].iloc[i - 2])
            test_indices.append(y.index[i-1])
            predict_percentages.append(y_pred[0][0] * 100)
            actual_percentages.append(y[i-1] * 100)

    delta = [p - a for p, a in zip(predict_percentages, actual_percentages)]
    result = pd.DataFrame({
        'predict_percentages(%)': predict_percentages,
        'actual_percentages(%)': actual_percentages,
        'delta(%)': delta
    })
    print(result)

    # Plot cumulative earnings
    cumulative_naive_percentage = np.cumsum(actual_percentages)
    cumulative_lstm_percentage = np.cumsum([a if p > 0 else 0 for p, a in zip(predict_percentages, actual_percentages)])
    plt.figure(figsize=(10, 6))
    plt.plot(test_indices, cumulative_naive_percentage, marker='o', markersize=3, linestyle='-', color='blue',
             label='Naive Strategy')
    plt.plot(test_indices, cumulative_lstm_percentage, marker='o', markersize=3, linestyle='-', color='orange',
             label='LSTM Strategy')
    plt.title(f'Daily Earnings Percentages for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs('pic', exist_ok=True)
    plt.savefig(f'pic/{ticker}.png')
    plt.close()

    predict_result = {str(date): pred / 100 for date, pred in zip(test_indices, predict_percentages)}
    return predict_result


# Predict for all stocks
all_predictions_lstm = {}
for ticker in tickers:
    data = stock_data[ticker]
    X, y = stock_features[ticker]
    predict_result = train_and_predict_lstm(ticker, data, X, y)
    all_predictions_lstm[ticker] = predict_result

# Save predictions
os.makedirs('../backend/predictions/LSTM', exist_ok=True)
for ticker, predictions in all_predictions_lstm.items():
    file_path = os.path.join('../backend/predictions/LSTM', f'{ticker}_predictions.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(predictions, file)
        print(f'Saved predictions for {ticker} to {file_path}')
