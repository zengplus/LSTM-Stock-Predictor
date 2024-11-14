import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pickle
sns.set()

class Deep_Evolution_Strategy:

    inputs = None

    def __init__(
        self, weights, reward_function, population_size, sigma, learning_rate
    ):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights

    def train(self, epoch = 100, print_every = 1):
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.weights, population[k]
                )
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T
                )
            if (i + 1) % print_every == 0:
                print(
                    'iter %d. reward: %f'
                    % (i + 1, self.reward_function(self.weights))
                )
        print('time taken to train:', time.time() - lasttime, 'seconds')


class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(1, layer_size),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

def load_predictions(ticker):
    """
    加载预测文件并打印调试信息
    """
    file_path = f'predictions/{ticker}_predictions.pkl'
    print(f"\nAttempting to load predictions from: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            predictions = pickle.load(f)
        print(f"Successfully loaded predictions for {ticker}")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Sample prediction dates: {list(predictions.keys())[:5]}")
        print(f"Sample prediction values: {list(predictions.values())[:5]}\n")
        return predictions
    except FileNotFoundError:
        print(f"Warning: No prediction file found at {file_path}")
        print("Make sure the predictions file exists and the path is correct")
        return {}
    except Exception as e:
        print(f"Error loading predictions: {str(e)}")
        return {}

# PyTorch版本的交易模型
class Trading_Model(nn.Module):
    def __init__(self, input_size, layer_size, output_size):
        super(Trading_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, layer_size)
        self.fc2 = nn.Linear(layer_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def predict(self, inputs):
        self.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(inputs).to(self.device)
            return self.forward(inputs).cpu().numpy()

    def get_weights(self):
        return [p.data.cpu().numpy() for p in self.parameters()]

    def set_weights(self, weights):
        for p, w in zip(self.parameters(), weights):
            p.data = torch.FloatTensor(w).to(self.device)

class Enhanced_Agent:
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, window_size, trend, skip, initial_money, ticker, dates):
        self.model = model
        self.window_size = window_size
        self.trend = trend
        self.skip = skip
        self.initial_money = initial_money
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.dates = dates
        self.predictions = load_predictions(ticker)
        
        self.init_evolution_strategy()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _calculate_reward(self, t, action):
        """
        计算单个交易动作的奖励
        """
        if t >= len(self.trend) - 1:
            return 0
        
        current_price = self.trend[t]
        next_price = self.trend[t + 1]
        price_change = (next_price - current_price) / current_price
        
        # 基础奖励
        reward = 0
        
        # 获取预测值
        prediction = self.get_prediction(t)
        predicted_direction = 1 if prediction > 0 else -1
        actual_direction = 1 if price_change > 0 else -1
        
        if action == 1:  # 买入
            # 基础奖励：价格变动的百分比
            reward = price_change * 100
            
            # 如果预测方向正确，给予额外奖励
            if predicted_direction == actual_direction:
                reward *= 1.2
            
            # 对追涨杀跌的行为进行惩罚
            if price_change > 0.02:  # 如果价格已经上涨超过2%
                reward *= 0.8
                
        elif action == 2:  # 卖出
            # 卖出时的基础奖励是价格变动的反向
            reward = -price_change * 100
            
            # 如果预测方向正确，给予额外奖励
            if predicted_direction == actual_direction:
                reward *= 1.2
            
            # 对追跌杀涨的行为进行惩罚
            if price_change < -0.02:  # 如果价格已经下跌超过2%
                reward *= 0.8
                
        else:  # 持有
            # 持有时给予小的奖励/惩罚
            reward = price_change * 20  # 降低持有的奖励/惩罚幅度
            
            # 如果预测趋势向好但选择不买，或预测趋势向下但选择不卖，给予小惩罚
            if (predicted_direction == 1 and actual_direction == 1) or \
               (predicted_direction == -1 and actual_direction == -1):
                reward *= 0.9
        
        # 考虑市场波动性
        window = self.trend[max(0, t-self.window_size):t+1]
        if len(window) > 1:
            avg_volatility = np.mean([abs(window[i+1] - window[i])/window[i] for i in range(len(window)-1)])
            if avg_volatility > 0.01:  # 如果平均波动率超过1%
                reward *= 0.8
        
        return np.clip(reward, -100, 100)

    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d:t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0:t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])

        prediction = self.get_prediction(t)
        window = self.trend[max(0, t-self.window_size):t+1]
        
        momentum = (window[-1] - window[0]) / window[0] if len(window) > 1 else 0
        volatility = np.std(window) if len(window) > 1 else 0
        
        gains = sum(max(window[i+1] - window[i], 0) for i in range(len(window)-1))
        losses = sum(max(window[i] - window[i+1], 0) for i in range(len(window)-1))
        rsi = gains / (gains + losses + 1e-9) * 100 if (gains + losses) > 0 else 50
        
        return np.concatenate([
            np.array(res),
            [prediction],
            [momentum],
            [volatility],
            [rsi]
        ]).reshape(1, -1)

    def get_prediction(self, t):
        """
        获取特定时间点的预测值
        """
        try:
            date = self.dates[t] if t < len(self.dates) else self.dates[-1]
            date_str = str(date)
            return self.predictions.get(date_str, 0)
        except (IndexError, KeyError):
            return 0

    def train(self, epochs=100, batch_size=32):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            state = self.get_state(0)
            total_reward = 0
            self.model.train()

            for t in range(0, len(self.trend) - 1, self.skip):
                action = self.act(state)
                next_state = self.get_state(t + 1)
                reward = self._calculate_reward(t, action)
                
                self.memory.append((state.reshape(-1), action, reward, next_state.reshape(-1)))
                
                if len(self.memory) >= batch_size:
                    batch = random.sample(self.memory, batch_size)
                    
                    states = np.array([item[0] for item in batch])
                    actions = np.array([item[1] for item in batch])
                    rewards = np.array([item[2] for item in batch])
                    next_states = np.array([item[3] for item in batch])
                    
                    states = torch.FloatTensor(states).to(self.device)
                    actions = torch.LongTensor(actions).to(self.device)
                    rewards = torch.FloatTensor(rewards).to(self.device)
                    next_states = torch.FloatTensor(next_states).to(self.device)
                    
                    q_values = self.model(states)
                    current_q = q_values.gather(1, actions.unsqueeze(1))
                    
                    with torch.no_grad():
                        next_q_values = self.model(next_states)
                        max_next_q = next_q_values.max(1)[0]
                        target_q = rewards + self.gamma * max_next_q
                    
                    loss = criterion(current_q.squeeze(), target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                total_reward += reward
                state = next_state
                
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Total Reward: {total_reward:.2f}')

    def buy(self):
        initial_money = self.initial_money
        working_money = initial_money
        states_buy = []
        states_sell = []
        inventory = []
        state = self.get_state(0)
        
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)
            current_price = self.trend[t]
            
            if action == 1 and working_money >= current_price:
                inventory.append({'price': current_price, 'time': t})
                working_money -= current_price
                states_buy.append(t)
                print(f'第{t}天: 以{current_price:.2f}的价格买入1单位, 当前余额{working_money:.2f}')
            
            elif action == 2 and inventory:
                buy_info = inventory.pop(0)
                bought_price = buy_info['price']
                hold_time = t - buy_info['time']
                
                working_money += current_price
                states_sell.append(t)
                profit_percent = (current_price - bought_price) / bought_price * 100
                print(f'第{t}天: 以{current_price:.2f}的价格卖出1单位, '
                      f'收益率{profit_percent:.2f}%, 当前余额{working_money:.2f}, '
                      f'持有时间{hold_time}天')
            
            state = next_state

        total_return = ((working_money - initial_money) / initial_money) * 100
        
        return states_buy, states_sell, working_money - initial_money, total_return

def main():
    # 读取数据
    ticker = 'GOOGL'  # 或其他股票代码
    df = pd.read_csv(f'data/{ticker}.csv')
    df.set_index('Date', inplace=True)
    
    # 获取日期列表
    dates = df.index.tolist()
    close_prices = df.Close.values
    
    # 参数设置
    window_size = 30
    skip = 1
    initial_money = 10000
    
    # 初始化交易模型
    trading_model = Trading_Model(input_size=window_size+4, layer_size=500, output_size=3)
    
    # 初始化代理
    agent = Enhanced_Agent(
        model=trading_model,
        window_size=window_size,
        trend=close_prices.tolist(),
        skip=skip,
        initial_money=initial_money,
        ticker=ticker,
        dates=dates  # 传入日期列表
    )
    
    # 训练代理
    print("Training trading agent...")
    agent.train(epochs=100)
    
    # 执行交易并评估结果
    print("Executing trades...")
    states_buy, states_sell, total_gains, invest = agent.buy()
    
    # 可视化结果
    plt.figure(figsize=(20, 10))
    plt.plot(close_prices, label='Stock Price', color='blue', alpha=0.7)
    plt.scatter(states_buy, [close_prices[i] for i in states_buy], color='green', marker='^', 
               label='Buy', alpha=1, s=100)
    plt.scatter(states_sell, [close_prices[i] for i in states_sell], color='red', marker='v', 
               label='Sell', alpha=1, s=100)
    plt.title('Trading Signals', fontsize=15)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('pic/trades/GOOGL_trades.png')
    
    # 打印交易结果
    print(f'\n交易结果汇总:')
    print(f'总收益: ${total_gains:.2f}')
    print(f'投资回报率: {invest:.2f}%')
    print(f'总交易次数: {len(states_buy)}')
    print(f'胜率: {sum(1 for i, j in zip(states_buy, states_sell) if close_prices[j] > close_prices[i]) / len(states_buy):.2%}')

if __name__ == "__main__":
    main()