import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import logging
import warnings
import pickle
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_agent.log'),
        logging.StreamHandler()
    ]
)

class DQNAgent(nn.Module):
    def __init__(self, state_size: int, hidden_sizes: List[int] = [128, 64], dropout: float = 0.2):
        super(DQNAgent, self).__init__()
        
        layers = []
        prev_size = state_size
        
        # 构建动态层
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_size, 3))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> tuple:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
        
    def __len__(self) -> int:
        return len(self.buffer)

class TradingEnvironment:
    def __init__(self, prices: np.ndarray, predictions: np.ndarray, 
                 initial_money: float = 10000, transaction_fee: float = 0.001):
        self.prices = np.array(prices)
        self.predictions = np.array(predictions)
        self.initial_money = initial_money
        self.transaction_fee = transaction_fee  # 交易费用
        self.reset()
        
    def reset(self) -> np.ndarray:
        self.money = self.initial_money
        self.shares = 0
        self.current_step = 0
        self.trades = []
        self.portfolio_values = [self.initial_money]
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        try:
            if self.current_step < len(self.prices):
                price = self.prices[self.current_step]
                pred = self.predictions[self.current_step]
                
                # 计算更多技术指标
                price_sma_10 = np.mean(self.prices[max(0, self.current_step-10):self.current_step+1])
                price_sma_30 = np.mean(self.prices[max(0, self.current_step-30):self.current_step+1])
                
                # 计算波动率 (20天)
                if self.current_step >= 20:
                    volatility = np.std(self.prices[self.current_step-20:self.current_step+1]) / price
                else:
                    volatility = 0
                
                return np.array([
                    self.shares / 100,  # 归一化持仓量
                    self.money / self.initial_money,  # 归一化现金
                    pred,  # LSTM预测
                    (price / price_sma_10) - 1,  # 相对于10日均线
                    (price / price_sma_30) - 1,  # 相对于30日均线
                    volatility,  # 波动率
                    price / self.prices[0] - 1  # 相对于初始价格的变化率
                ], dtype=np.float32)
            return None
        except Exception as e:
            logging.error(f"Error in _get_state: {str(e)}")
            return None
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        try:
            if self.current_step >= len(self.prices) - 1:
                return self._get_state(), 0, True
                
            current_price = self.prices[self.current_step]
            self.current_step += 1
            next_price = self.prices[self.current_step]
            
            initial_portfolio_value = self.money + self.shares * current_price
            
            # 执行交易（考虑交易费用）
            if action == 1 and self.money >= current_price:  # 买入
                max_shares = self.money // current_price
                shares_to_buy = max_shares // 2  # 只使用一半资金
                fee = shares_to_buy * current_price * self.transaction_fee
                
                if self.money >= (shares_to_buy * current_price + fee):
                    self.shares += shares_to_buy
                    self.money -= (shares_to_buy * current_price + fee)
                    self.trades.append((self.current_step, 1))
                    
            elif action == 2 and self.shares > 0:  # 卖出
                fee = self.shares * current_price * self.transaction_fee
                self.money += self.shares * current_price - fee
                self.shares = 0
                self.trades.append((self.current_step, 2))
                
            new_portfolio_value = self.money + self.shares * next_price
            self.portfolio_values.append(new_portfolio_value)
            
            # 计算奖励（夏普比率）
            if len(self.portfolio_values) > 1:
                returns = np.diff(self.portfolio_values[-2:]) / self.portfolio_values[-2]
                reward = returns * 100
                
                # 添加风险调整
                if len(self.portfolio_values) > 30:
                    returns_30d = np.diff(self.portfolio_values[-31:]) / self.portfolio_values[-31:-1]
                    sharpe_ratio = np.mean(returns_30d) / (np.std(returns_30d) + 1e-6)
                    reward += sharpe_ratio
            else:
                reward = 0
                
            done = self.current_step >= len(self.prices) - 1
            
            if done:
                total_return = ((new_portfolio_value - self.initial_money) / self.initial_money) * 100
                reward = total_return
                
            return self._get_state(), reward, done
            
        except Exception as e:
            logging.error(f"Error in step: {str(e)}")
            return self._get_state(), 0, True

class TradingAgent:
    def __init__(self, state_size: int = 7, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.agent = DQNAgent(state_size).to(self.device)
        self.target_agent = DQNAgent(state_size).to(self.device)
        self.target_agent.load_state_dict(self.agent.state_dict())
        
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
    def update_target_network(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        
    def train(self, env: TradingEnvironment, episodes: int = 1000, batch_size: int = 64,
              gamma: float = 0.99, epsilon_start: float = 1.0, epsilon_end: float = 0.01,
              epsilon_decay: float = 0.995, memory_size: int = 100000,
              update_target_freq: int = 10) -> List[float]:
        
        memory = ReplayBuffer(memory_size)
        epsilon = epsilon_start
        best_reward = float('-inf')
        rewards_history = []
        
        try:
            for episode in tqdm(range(episodes), desc="Training DQN"):
                state = env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    # Epsilon-greedy action selection
                    if np.random.random() < epsilon:
                        action = np.random.randint(0, 3)
                    else:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            action = self.agent(state_tensor).argmax().item()
                    
                    next_state, reward, done = env.step(action)
                    total_reward += reward
                    
                    # 存储经验
                    memory.push(state, action, reward, next_state, done)
                    state = next_state
                    
                    # 经验回放
                    if len(memory) >= batch_size:
                        states, actions, rewards, next_states, dones = memory.sample(batch_size)
                        
                        states = torch.FloatTensor(states).to(self.device)
                        next_states = torch.FloatTensor(next_states).to(self.device)
                        actions = torch.LongTensor(actions).to(self.device)
                        rewards = torch.FloatTensor(rewards).to(self.device)
                        dones = torch.FloatTensor(dones).to(self.device)
                        
                        # Double DQN
                        current_q = self.agent(states).gather(1, actions.unsqueeze(1))
                        
                        with torch.no_grad():
                            next_actions = self.agent(next_states).argmax(1, keepdim=True)
                            next_q = self.target_agent(next_states).gather(1, next_actions)
                            target_q = rewards.unsqueeze(1) + gamma * next_q * (1 - dones.unsqueeze(1))
                        
                        # Huber损失
                        loss = F.smooth_l1_loss(current_q, target_q)
                        
                        # 梯度裁剪
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
                        self.optimizer.step()
                
                rewards_history.append(total_reward)
                
                # 更新目标网络
                if episode % update_target_freq == 0:
                    self.update_target_network()
                
                # 更新epsilon
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                
                # 更新学习率
                self.scheduler.step(total_reward)
                
                # 保存最佳模型
                if total_reward > best_reward:
                    best_reward = total_reward
                    torch.save({
                        'model_state_dict': self.agent.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'episode': episode,
                        'best_reward': best_reward
                    }, 'best_model.pth')
                
                if episode % 10 == 0:
                    avg_reward = np.mean(rewards_history[-10:])
                    logging.info(f'Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}')
                    
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            
        return rewards_history
    
    def trade(self, env: TradingEnvironment) -> Tuple[List, float]:
        """使用训练好的模型进行交易"""
        try:
            self.agent.eval()
            state = env.reset()
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.agent(state_tensor).argmax().item()
                state, _, done = env.step(action)
            
            return env.trades, env.money + env.shares * env.prices[-1]
            
        except Exception as e:
            logging.error(f"Error during trading: {str(e)}")
            return [], env.initial_money

def plot_trades(prices: np.ndarray, trades: List, initial_money: float,
               final_value: float, ticker: str):
    """绘制交易结果"""
    try:
        plt.figure(figsize=(15, 7))
        plt.plot(prices, label='Stock Price', color='blue', alpha=0.6)
        
        # 标记买卖点
        buy_points = [t[0] for t in trades if t[1] == 1]
        sell_points = [t[0] for t in trades if t[1] == 2]
        
        plt.plot(buy_points, prices[buy_points], '^', markersize=10,
                color='green', label='Buy')
        plt.plot(sell_points, prices[sell_points], 'v', markersize=10,
                color='red', label='Sell')
        
        # 计算收益率和夏普比率
        returns = ((final_value - initial_money) / initial_money) * 100
        daily_returns = np.diff(prices) / prices[:-1]
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
        
        plt.title(f'{ticker} Trading Results\n' +
                 f'Initial: ${initial_money:,.2f}, Final: ${final_value:,.2f}\n' +
                 f'Return: {returns:.2f}%, Sharpe Ratio: {sharpe_ratio:.2f}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        os.makedirs('pic/trades', exist_ok=True)
        plt.savefig(f'pic/trades/{ticker}_trades.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logging.error(f"Error in plot_trades: {str(e)}")

def main():
    # 配置随机种子以确保可重复性
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # 股票列表
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
        try:
            logging.info(f"\nTraining agent for {ticker}")
            
            # 加载股票数据
            stock_data = pd.read_csv(f'data/{ticker}.csv')
            prices = stock_data['Close'].values
            
            # 数据预处理和归一化
            prices = np.array(prices, dtype=np.float32)
            
            # 加载LSTM预测结果
            with open(f'predictions/{ticker}_predictions.pkl', 'rb') as f:
                predictions = pickle.load(f)
            
            pred_values = np.array([predictions[date] for date in predictions.keys()], dtype=np.float32)
            
            # 创建环境和代理
            env = TradingEnvironment(
                prices=prices[-len(pred_values):],
                predictions=pred_values,
                transaction_fee=0.001  # 0.1% 交易费用
            )
            
            agent = TradingAgent(state_size=7)  # 更新状态空间大小
            
            # 训练代理
            rewards = agent.train(
                env,
                episodes=200,
                batch_size=64,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.995,
                memory_size=100000,
                update_target_freq=10
            )
            
            # 进行实际交易测试
            trades, final_value = agent.trade(env)
            
            # 绘制交易结果
            plot_trades(
                prices[-len(pred_values):],
                trades,
                env.initial_money,
                final_value,
                ticker
            )
            
            # 计算额外的指标
            returns = ((final_value - env.initial_money) / env.initial_money) * 100
            portfolio_values = np.array(env.portfolio_values)
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
            max_drawdown = np.min(portfolio_values / np.maximum.accumulate(portfolio_values) - 1)
            
            # 记录结果
            results[ticker] = {
                'final_value': final_value,
                'return': returns,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'n_trades': len(trades),
                'avg_trade_return': returns / (len(trades) if trades else 1)
            }
            
            # 保存训练好的模型
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': agent.agent.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'final_reward': final_value - env.initial_money,
                'hyperparameters': {
                    'state_size': 7,
                    'hidden_sizes': [128, 64],
                    'dropout': 0.2
                }
            }, f'models/{ticker}_model.pth')
            
        except Exception as e:
            logging.error(f"Error processing {ticker}: {str(e)}")
            continue
    
    # 输出所有股票的交易结果
    try:
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('return', ascending=False)
        
        print("\nTrading Results Summary:")
        print(results_df)
        
        # 保存详细结果
        results_df.to_csv('trading_results.csv')
        
        # 创建结果可视化
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        results_df['return'].plot(kind='bar')
        plt.title('Returns by Stock')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        results_df['sharpe_ratio'].plot(kind='bar')
        plt.title('Sharpe Ratios')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        results_df['max_drawdown'].plot(kind='bar')
        plt.title('Maximum Drawdowns')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        results_df['n_trades'].plot(kind='bar')
        plt.title('Number of Trades')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('trading_summary.png')
        plt.close()
        
        # 计算整体统计数据
        summary_stats = {
            'Average Return': results_df['return'].mean(),
            'Median Return': results_df['return'].median(),
            'Best Return': results_df['return'].max(),
            'Worst Return': results_df['return'].min(),
            'Average Sharpe': results_df['sharpe_ratio'].mean(),
            'Average Max Drawdown': results_df['max_drawdown'].mean(),
            'Total Trades': results_df['n_trades'].sum()
        }
        
        print("\nOverall Performance Summary:")
        for metric, value in summary_stats.items():
            print(f"{metric}: {value:.2f}")
            
    except Exception as e:
        logging.error(f"Error in results analysis: {str(e)}")

if __name__ == "__main__":
    main()