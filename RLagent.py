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
    def __init__(self, prices, predictions, initial_balance=100000, max_trades=5):
        """
        交易环境
        
        Args:
            prices: 股票价格序列
            predictions: LSTM预测的价格变化率
            initial_balance: 初始资金
            max_trades: 每个时间步最大交易数量
        """
        self.prices = prices
        self.predictions = predictions
        self.initial_balance = initial_balance
        self.max_trades = max_trades
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0  # 当前持仓量
        self.current_step = 0
        self.trades = []
        self.portfolio_values = []
        return self._get_state()
    
    def _get_state(self):
        """
        状态包含:
        - 当前持仓量
        - 当前余额
        - LSTM预测的价格变化
        - 当前价格相对于过去N天的变化率
        - 技术指标 (MA5, MA10等)
        """
        if self.current_step >= len(self.prices):
            return None
            
        price = self.prices[self.current_step]
        
        # 获取LSTM预测值
        date = str(self.current_step)
        pred_change = self.predictions.get(date, 0)
        
        # 计算价格变化率
        price_change = 0
        if self.current_step > 0:
            price_change = (price - self.prices[self.current_step-1]) / self.prices[self.current_step-1]
        
        # 计算移动平均
        ma5 = np.mean(self.prices[max(0, self.current_step-5):self.current_step+1])
        ma10 = np.mean(self.prices[max(0, self.current_step-10):self.current_step+1])
        
        # 归一化处理
        normalized_balance = self.balance / self.initial_balance
        normalized_position = self.position / self.max_trades
        normalized_price = price / self.prices[0]
        
        state = np.array([
            normalized_position,
            normalized_balance,
            pred_change,
            price_change,
            (price - ma5) / ma5,
            (price - ma10) / ma10,
            normalized_price
        ])
        
        return state
    
    def step(self, action):
        """
        执行交易行为
        action: [0: 持有, 1: 买入, 2: 卖出]
        """
        if self.current_step >= len(self.prices) - 1:
            return self._get_state(), 0, True
            
        current_price = self.prices[self.current_step]
        self.current_step += 1
        next_price = self.prices[self.current_step]
        
        # 记录当前投资组合价值
        portfolio_value_before = self.balance + self.position * current_price
        
        reward = 0
        trade_amount = 1  # 每次交易1个单位
        
        if action == 1:  # 买入
            max_can_buy = min(self.max_trades - self.position, self.balance // current_price)
            if max_can_buy > 0:
                self.position += trade_amount
                self.balance -= current_price * trade_amount
                self.trades.append((self.current_step, 'buy', current_price, trade_amount))
                
        elif action == 2:  # 卖出
            if self.position > 0:
                self.position -= trade_amount
                self.balance += current_price * trade_amount
                self.trades.append((self.current_step, 'sell', current_price, trade_amount))
        
        # 计算新的投资组合价值和奖励
        portfolio_value_after = self.balance + self.position * next_price
        profit = portfolio_value_after - portfolio_value_before
        
        # 设计奖励函数
        reward = profit / portfolio_value_before * 100  # 收益率作为基础奖励
        
        # 添加额外的奖励/惩罚
        if action != 0:  # 交易成本惩罚
            reward -= 0.1
        
        if self.position < 0 or self.balance < 0:  # 非法操作惩罚
            reward = -10
            
        # 记录投资组合价值
        self.portfolio_values.append(portfolio_value_after)
        
        done = self.current_step >= len(self.prices) - 1
        if done:
            total_return = (portfolio_value_after - self.initial_balance) / self.initial_balance * 100
            reward += total_return  # 在episode结束时给予总收益奖励
            
        return self._get_state(), reward, done

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size=3):
        super(DQNAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
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
        
        self.optimizer = optim.Adam(self.agent.parameters())
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=100000)
        
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, 2)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.agent(state_tensor)
        return q_values.argmax().item()
    
    def train(self, env, episodes=1000, max_steps=None):
        rewards_history = []
        portfolio_values = []
        
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
                    batch = random.sample(self.memory, self.batch_size)
                    self._train_step(batch)
                
                if done or (max_steps and steps >= max_steps):
                    break
            
            # 更新目标网络
            if episode % self.target_update == 0:
                self.target_agent.load_state_dict(self.agent.state_dict())
            
            # 衰减epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            rewards_history.append(total_reward)
            portfolio_values.append(env.portfolio_values[-1])
            
            if episode % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(f'Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}')
        
        return rewards_history, portfolio_values
    
    def _train_step(self, batch):
        states = torch.FloatTensor([x[0] for x in batch]).to(self.device)
        actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_states = torch.FloatTensor([x[3] for x in batch]).to(self.device)
        dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)
        
        current_q = self.agent(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_agent(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
    # 股票列表
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',       # 科技
        'JPM', 'BAC', 'C', 'WFC', 'GS',                # 金融
        'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',            # 医药
        'XOM', 'CVX', 'COP', 'SLB', 'BKR',             # 能源
        'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',         # 消费
        'CAT', 'DE', 'MMM', 'GE', 'HON'                # 工业
    ]
    
    results = {}
    
    for ticker in tickers:
        print(f"\nTraining agent for {ticker}")
        
        # 加载数据和LSTM预测结果
        data = pd.read_csv(f'data/{ticker}.csv')
        prices = data['Close'].values
        
        with open(f'predictions/{ticker}_predictions.pkl', 'rb') as f:
            predictions = pickle.load(f)
        
        # 创建环境和代理
        env = TradingEnvironment(prices, predictions)
        agent = TradingAgent(state_size=7)  # 7个特征的状态空间
        
        # 训练代理
        rewards_history, portfolio_values = agent.train(env, episodes=200)
        
        # 测试和可视化
        state = env.reset()
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            state, _, done = env.step(action)
        
        # 可视化结果
        visualize_trades(prices, env.trades, env.portfolio_values, ticker)
        
        # 记录结果
        final_value = env.portfolio_values[-1]
        total_return = (final_value - env.initial_balance) / env.initial_balance * 100
        
        results[ticker] = {
            'final_value': final_value,
            'return': total_return,
            'n_trades': len(env.trades)
        }
        
        # 保存模型
        os.makedirs('models', exist_ok=True)
        torch.save(agent.agent.state_dict(), f'models/{ticker}_model.pth')
    
    # 保存和显示结果
    results_df = pd.DataFrame(results).T
    results_df.to_csv('trading_results.csv')
    print("\nTrading Results:")
    print(results_df)

if __name__ == "__main__":
    main()