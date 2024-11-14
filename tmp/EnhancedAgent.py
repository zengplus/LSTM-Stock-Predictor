import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import deque
import random
import pickle

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
class Enhanced_Agent:
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, window_size, trend, skip, initial_money, lstm_predictions):
        self.model = model
        self.window_size = window_size
        self.trend = trend
        self.skip = skip
        self.initial_money = initial_money
        self.lstm_predictions = lstm_predictions
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_evolution_strategy()

    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d:t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0:t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])

        lstm_pred = self.lstm_predictions[t] if t < len(self.lstm_predictions) else self.lstm_predictions[-1]
        window = self.trend[max(0, t-self.window_size):t+1]
        short_window = self.trend[max(0, t-5):t+1]
        
        # 计算多个时间周期的动量
        momentum_short = (window[-1] - short_window[0]) / short_window[0] if len(short_window) > 1 else 0
        momentum_long = (window[-1] - window[0]) / window[0] if len(window) > 1 else 0
        
        # 计算波动率
        volatility = np.std(window) if len(window) > 1 else 0
        
        # 计算RSI
        gains = sum(max(window[i+1] - window[i], 0) for i in range(len(window)-1))
        losses = sum(max(window[i] - window[i+1], 0) for i in range(len(window)-1))
        rsi = gains / (gains + losses + 1e-9) * 100 if (gains + losses) > 0 else 50
        
        # 计算移动平均
        ma5 = np.mean(short_window) if len(short_window) > 0 else window[-1]
        ma20 = np.mean(window[-20:]) if len(window) >= 20 else window[-1]
        
        return np.concatenate([
            np.array(res),
            [lstm_pred],
            [momentum_short],
            [momentum_long],
            [volatility],
            [rsi],
            [window[-1]/ma5 - 1],  # 价格相对MA5的偏离度
            [window[-1]/ma20 - 1],  # 价格相对MA20的偏离度
            [1 if lstm_pred > window[-1] else -1]  # LSTM预测方向
        ]).reshape(1, -1)

    def _calculate_reward(self, t, action):
        if t >= len(self.trend) - 1:
            return 0
        
        current_price = self.trend[t]
        next_price = self.trend[t + 1]
        price_change = (next_price - current_price) / current_price
        
        reward = 0
        lstm_prediction = self.lstm_predictions[t] if t < len(self.lstm_predictions) else self.lstm_predictions[-1]
        price_momentum = (current_price - self.trend[max(0, t-5)]) / self.trend[max(0, t-5)]
        
        if action == 1:  # Buy
            # Only reward buying when price is expected to increase and has positive momentum
            if lstm_prediction > current_price and price_momentum > 0:
                reward = price_change * 150
            else:
                reward = price_change * -100
                
        elif action == 2:  # Sell
            # Only reward selling when price is expected to decrease and has negative momentum
            if lstm_prediction < current_price and price_momentum < 0:
                reward = -price_change * 150
            else:
                reward = -price_change * -100
                
        else:  # Hold
            # Small positive reward for holding during uptrend
            reward = price_change * 30
        
        return np.clip(reward, -100, 100)

    def get_reward(self, weights):
        initial_money = self.initial_money
        starting_money = initial_money
        self.model.set_weights(weights)
        state = self.get_state(0)
        inventory = []
        total_trades = 0
        profitable_trades = 0

        max_position_size = initial_money * 0.2
        stop_loss = -0.05
        take_profit = 0.1

        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)
            current_price = self.trend[t]
            
            # 计算当前持仓市值
            portfolio_value = sum([current_price for pos in inventory])
            total_value = starting_money + portfolio_value
            
            # 检查止损/止盈
            for pos in inventory[:]:
                profit_pct = (current_price - pos) / pos
                if profit_pct <= stop_loss or profit_pct >= take_profit:
                    starting_money += current_price
                    inventory.remove(pos)
                    total_trades += 1
                    if profit_pct > 0:
                        profitable_trades += 1
            
            position_size = min(max_position_size, starting_money * 0.2)
            if action == 1 and starting_money >= current_price and len(inventory) < 5:
                if position_size >= current_price:
                    inventory.append(current_price)
                    starting_money -= current_price
                    total_trades += 1
                
            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                starting_money += current_price
                total_trades += 1
                if current_price > bought_price:
                    profitable_trades += 1

            state = next_state
            
            # 使用总价值判断止损
            if total_value < initial_money * 0.7:
                return -100

        # 最终收益计算包括未平仓头寸的当前市值
        final_portfolio_value = starting_money + sum([self.trend[-1] for _ in inventory])
        final_return = ((final_portfolio_value - initial_money) / initial_money) * 100
        
        holding_cost = len(inventory) * 0.001 * initial_money
        trade_efficiency = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        
        reward = final_return + trade_efficiency * 0.2 - holding_cost
        
        if total_trades > len(self.trend) * 0.1:
            reward *= 0.8
        
        return np.clip(reward, -100, 100)

    def init_evolution_strategy(self):
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(3)

        with torch.no_grad():
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def train(self, epochs=500, batch_size=32):
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
        
        max_position_size = initial_money * 0.2
        stop_loss = -0.05
        take_profit = 0.1
        
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)
            current_price = self.trend[t]
            
            # 计算当前持仓市值
            portfolio_value = sum([current_price for pos in inventory])
            total_value = working_money + portfolio_value
            
            # 检查止损/止盈
            for pos in inventory[:]:
                profit_pct = (current_price - pos['price']) / pos['price']
                if profit_pct <= stop_loss or profit_pct >= take_profit:
                    working_money += current_price
                    states_sell.append(t)
                    inventory.remove(pos)
                    print(f'第{t}天: 触发{'止损' if profit_pct <= stop_loss else '止盈'}, '
                        f'以{current_price:.2f}的价格卖出, 总市值: {total_value:.2f}')
            
            position_size = min(max_position_size, working_money * 0.2)
            if action == 1 and working_money >= current_price and len(inventory) < 5:
                if position_size >= current_price:
                    inventory.append({'price': current_price, 'time': t})
                    working_money -= current_price
                    states_buy.append(t)
                    print(f'第{t}天: 以{current_price:.2f}的价格买入1单位, '
                        f'当前余额{working_money:.2f}, 总市值: {total_value:.2f}')
                    
            elif action == 2 and inventory:
                pos = inventory.pop(0)
                working_money += current_price
                states_sell.append(t)
                profit_percent = (current_price - pos['price']) / pos['price'] * 100
                print(f'第{t}天: 以{current_price:.2f}的价格卖出1单位, '
                    f'收益率{profit_percent:.2f}%, 总市值: {total_value:.2f}')
            
            state = next_state
            
            # 使用总市值判断止损
            if total_value < initial_money * 0.7:
                print(f"触发总体止损，清空所有持仓, 当前总市值: {total_value:.2f}")
                for pos in inventory:
                    working_money += current_price
                    states_sell.append(t)
                break

        # 最终收益计算包括未平仓头寸的当前市值
        final_portfolio_value = working_money + sum([self.trend[-1] for _ in inventory])
        total_return = ((final_portfolio_value - initial_money) / initial_money) * 100
        
        return states_buy, states_sell, final_portfolio_value - initial_money, total_return
def main():
    df = pd.read_csv('data/GOOGL.csv')
    close_prices = df.Close.values
    
    with open('predictions/GOOGL_predictions.pkl', 'rb') as file:
        lstm_predictions = pickle.load(file)
    
    lstm_pred_list = [value for value in lstm_predictions.values()]
    
    window_size = 30
    skip = 1
    initial_money = 10000
    
    # Calculate correct input size: window_size + all additional features
    input_size = window_size + 8  # window_size + [lstm_pred, momentum_short, momentum_long, volatility, rsi, ma5_diff, ma20_diff, pred_direction]
    
    trading_model = Trading_Model(input_size=input_size, layer_size=500, output_size=3)
    
    agent = Enhanced_Agent(
        model=trading_model,
        window_size=window_size,
        trend=close_prices.tolist(),
        skip=skip,
        initial_money=initial_money,
        lstm_predictions=lstm_pred_list
    )
    
    print("Training trading agent...")
    agent.train(epochs=500)
    
    print("Executing trades...")
    states_buy, states_sell, total_gains, invest = agent.buy()
    
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
    plt.show()
    
    print(f'\n交易结果汇总:')
    print(f'总收益: ${total_gains:.2f}')
    print(f'投资回报率: {invest:.2f}%')
    print(f'总交易次数: {len(states_buy)}')
    print(f'胜率: {sum(1 for i, j in zip(states_buy, states_sell) if close_prices[j] > close_prices[i]) / len(states_buy):.2%}')

if __name__ == "__main__":
    main()