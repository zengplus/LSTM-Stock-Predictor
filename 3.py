import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle
from tqdm import tqdm

class TradingEnvironment:
    def __init__(self, prices, predictions, initial_balance=100000, max_trades=10):
        self.prices = prices
        self.predictions = predictions
        self.initial_balance = initial_balance
        self.max_trades = max_trades
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        return self._get_state()
    
    def _get_state(self):
        if self.current_step >= len(self.prices):
            return None
            
        price = self.prices[self.current_step]
        
        # 获取LSTM预测值
        date = str(self.current_step)
        pred_change = self.predictions.get(date, 0)
        
        # 计算价格动量
        momentum = 0
        if self.current_step > 5:
            momentum = (price - self.prices[self.current_step-5]) / self.prices[self.current_step-5]
        
        # 计算移动平均
        ma5 = np.mean(self.prices[max(0, self.current_step-5):self.current_step+1])
        ma10 = np.mean(self.prices[max(0, self.current_step-10):self.current_step+1])
        
        # 计算波动性
        volatility = 0
        if self.current_step > 10:
            volatility = np.std(self.prices[self.current_step-10:self.current_step+1]) / price
        
        # 归一化后的状态值
        state = np.array([
            self.position / self.max_trades,  # 归一化持仓
            self.balance / self.initial_balance,  # 归一化余额
            pred_change,  # LSTM预测的变化率
            momentum,  # 价格动量
            (price - ma5) / ma5,  # 相对于5日均线的位置
            (price - ma10) / ma10,  # 相对于10日均线的位置
            volatility,  # 波动率
            price / self.prices[0] - 1  # 相对于初始价格的涨幅
        ])
        
        return state
    
    def step(self, action):
        if self.current_step >= len(self.prices) - 1:
            return self._get_state(), 0, True
            
        current_price = self.prices[self.current_step]
        self.current_step += 1
        next_price = self.prices[self.current_step]
        
        # 记录当前投资组合价值
        portfolio_value_before = self.balance + self.position * current_price
        
        # 根据可用资金确定交易数量
        max_shares = self.balance // current_price
        trade_shares = min(1, max_shares) if max_shares > 0 else 0
        
        reward = 0
        if action == 1:  # 买入
            if self.balance >= current_price * trade_shares and self.position < self.max_trades:
                self.position += trade_shares
                self.balance -= current_price * trade_shares
                self.trades.append((self.current_step, 'buy', current_price, trade_shares))
                
        elif action == 2:  # 卖出
            if self.position > 0:
                sell_shares = min(1, self.position)
                self.position -= sell_shares
                self.balance += current_price * sell_shares
                self.trades.append((self.current_step, 'sell', current_price, sell_shares))
        
        # 计算新的投资组合价值
        portfolio_value_after = self.balance + self.position * next_price
        
        # 计算奖励
        immediate_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        
        # 设计更激进的奖励机制
        if action != 0:  # 如果进行了交易
            if immediate_return > 0:
                reward = immediate_return * 100  # 放大正收益
            else:
                reward = immediate_return * 50  # 减小负收益惩罚
        else:  # 如果没有交易
            # 根据LSTM预测来惩罚错过的机会
            pred_change = self.predictions.get(str(self.current_step), 0)
            if abs(pred_change) > 0.01:  # 如果预测变化显著但没有行动
                reward = -0.1
        
        # 记录投资组合价值
        self.portfolio_values.append(portfolio_value_after)
        
        # 在回合结束时给予额外奖励
        done = self.current_step >= len(self.prices) - 1
        if done:
            total_return = (portfolio_value_after - self.initial_balance) / self.initial_balance
            reward += total_return * 100  # 放大最终收益的奖励
        
        return self._get_state(), reward, done

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size=3):
        super(DQNAgent, self).__init__()
        # 使用更大的网络
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        
    def forward(self, x):
        return self.network(x)

class TradingAgent:
    def __init__(self, state_size, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.agent = DQNAgent(state_size).to(self.device)
        self.target_agent = DQNAgent(state_size).to(self.device)
        self.target_agent.load_state_dict(self.agent.state_dict())
        
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=100000)
        
        # 修改超参数
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update = 5
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, 2)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.agent(state_tensor)
        return q_values.argmax().item()
    
    def train(self, env, episodes=500, max_steps=None):
        rewards_history = []
        portfolio_values = []
        best_return = -np.inf
        
        for episode in tqdm(range(episodes), desc="Training"):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = self.act(state)
                next_state, reward, done = env.step(action)
                
                if next_state is not None:
                    self.memory.append((state, action, reward, next_state, done))
                    state = next_state
                    total_reward += reward
                    steps += 1
                
                if len(self.memory) >= self.batch_size:
                    self._train_step(self.batch_size)
                
                if done or (max_steps and steps >= max_steps):
                    break
            
            if episode % self.target_update == 0:
                self.target_agent.load_state_dict(self.agent.state_dict())
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            rewards_history.append(total_reward)
            portfolio_values.append(env.portfolio_values[-1])
            
            # 保存最佳模型
            current_return = (env.portfolio_values[-1] - env.initial_balance) / env.initial_balance
            if current_return > best_return:
                best_return = current_return
                torch.save(self.agent.state_dict(), 'best_model.pth')
            
            if episode % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(f'Episode {episode}, Average Reward: {avg_reward:.2f}, '
                      f'Return: {current_return*100:.2f}%, Epsilon: {self.epsilon:.2f}')
        
        # 加载最佳模型用于测试
        self.agent.load_state_dict(torch.load('best_model.pth'))
        return rewards_history, portfolio_values

def visualize_trades(prices, trades, portfolio_values, ticker):
    plt.figure(figsize=(15, 10))
    
    # 绘制价格和交易点
    plt.subplot(2, 1, 1)
    plt.plot(prices, label='Stock Price', color='blue', alpha=0.6)
    
    buy_points = [t[0] for t in trades if t[1] == 'buy']
    sell_points = [t[0] for t in trades if t[1] == 'sell']
    
    plt.plot(buy_points, [prices[i] for i in buy_points], '^', 
             color='green', label='Buy', markersize=10)
    plt.plot(sell_points, [prices[i] for i in sell_points], 'v', 
             color='red', label='Sell', markersize=10)
    
    plt.title(f'{ticker} Trading Signals')
    plt.legend()
    plt.grid(True)
    
    # 绘制投资组合价值
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_values, label='Portfolio Value', color='orange')
    plt.title('Portfolio Value Over Time')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('pic/trades', exist_ok=True)
    plt.savefig(f'pic/trades/{ticker}_trades.png')
    plt.close()

def main():
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'JPM', 'BAC', 'C', 'WFC', 'GS',
        'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',
        'XOM', 'CVX', 'COP', 'SLB', 'BKR',
        'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',
        'CAT', 'DE', 'MMM', 'GE', 'HON'
    ]
    
    results = {}
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}")
        
        # 加载数据
        data = pd.read_csv(f'data/{ticker}.csv')
        prices = data['Close'].values
        
        with open(f'predictions/{ticker}_predictions.pkl', 'rb') as f:
            predictions = pickle.load(f)
        
        # 创建环境和代理
        env = TradingEnvironment(prices, predictions)
        agent = TradingAgent(state_size=8)  # 更新状态空间大小
        
        # 训练
        rewards_history, portfolio_values = agent.train(env, episodes=500)
        
        # 测试
        state = env.reset()
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            state, _, done = env.step(action)
        
        # 可视化和保存结果
        visualize_trades(prices, env.trades, env.portfolio_values, ticker)
        
        final_value = env.portfolio_values[-1]
        total_return = (final_value - env.initial_balance) / env.initial_balance * 100
        
        results[ticker] = {
            'final_value': final_value,
            'return': total_return,
            'n_trades': len(env.trades)
        }
    
    results_df = pd.DataFrame(results).T
    results_df.to_csv('trading_results.csv')
    print("\nTrading Results:")
    print(results_df)

if __name__ == "__main__":
    main()