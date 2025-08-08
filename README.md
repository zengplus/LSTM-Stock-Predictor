# LSTM-Stock-Predictor

[English](README_EN.md) | 中文

> **Note**: 基于 [stock_trading](https://github.com/MilleXi/stock_trading) 项目的基础上学习。

> **Important**: 本项目仅作为LSTM技术的学习实践结果，不构成任何形式的投资建议，其预测结果亦不具备参考准确性。

---

基于LSTM预测和强化学习的股票预测系统。该系统结合了深度学习的预测能力和强化学习的决策能力，可以自动进行股票价格预测和交易决策。

## 项目特点

- 使用LSTM进行股票价格走势预测
- 采用深度进化策略DES进行交易决策
- 完整的数据处理和特征工程
- 可视化界面展示预测和交易结果
- 支持批量处理多支股票
- 多数据源支持（Yahoo Finance、AKShare）
- 多语言界面支持（中文/英文）

## 主要功能

### 1. 股票数据获取与预处理
- 支持多种数据源：Yahoo Finance 和 AKShare
- 自动下载并预处理股票历史数据
- 计算多种技术指标（移动平均线、RSI、MACD、布林带等）
- 修复了技术指标计算中的未来数据泄露问题

### 2. LSTM价格预测
- 使用长短期记忆网络（LSTM）模型预测股票价格
- 提供明日价格预测功能
- 可视化预测结果与实际价格对比
- 生成投资建议（买入/卖出/持有）

### 3. 强化学习交易策略
- 基于深度进化策略的强化学习模型
- 生成买入/卖出信号
- 模拟交易并计算投资回报率
- 风险调整绩效分析（夏普比率、索提诺比率、最大回撤等）

### 4. 可视化分析
- 价格预测与实际走势对比图
- 交易信号可视化（买入/卖出点标记）
- 特征重要性分析
- 累积收益曲线

### 5. 多语言支持
- 支持中英文界面切换
- 所有分析结果和图表根据语言设置自动适配

## 技术栈

- **深度学习框架**：PyTorch
- **数据处理**：Pandas, NumPy
- **数据获取**：yfinance, AKShare
- **可视化**：Matplotlib, Seaborn, Streamlit
- **优化算法**：深度进化策略DES
- **界面**：Streamlit

## 安装指南

### 前置要求
- Python 3.7+
- pip

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/zengplus/LSTM-Stock-Predictor.git
cd LSTM-Stock-Predictor
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明

### 运行应用
```bash
streamlit run app.py
```

### 界面操作指南

1. **侧边栏参数设置**：
   - 输入股票代码（多个代码用逗号分隔）
   - 设置日期范围
   - 调整初始资金、训练轮数等参数
   - 选择数据源（Yahoo Finance 或 AKShare）
   - 设置语言（中文/英文）

2. **功能按钮**：
   - **运行完整分析**：执行数据获取、模型训练、预测和交易策略生成
   - **仅获取明日预测**：快速获取明日价格预测和投资建议

3. **结果展示**：
   - 股票预测结果（RMSE、MAE、准确率）
   - 交易策略分析（总收益、投资回报率）
   - 明日价格预测及投资建议
   - 可视化图表（价格走势、交易信号等）

#### 首页示例
![Home Page](images/home_page.png)

#### 结果页示例
![Results Page](images/results_page.png)

## 项目结构

```
LSTM-Stock-Predictor/
├── results/                # 分析结果存储目录
│   └── session_{timestamp}/ # 以时间戳命名的会话目录
│       ├── analytics/        # 分析报告
│       ├── feature_importance/ # 特征重要性分析结果
│       ├── models/           # 保存的模型
│       ├── predictions/      # 预测结果
│       ├── scalers/          # 归一化器
│       ├── trading/          # 交易记录
│       └── session_summary.json # 会话总结
├── data/                   # 股票数据存储目录
├── images/                 # 项目文档图片目录
├── app.py                  # 主应用程序
├── requirements.txt        # 依赖列表
├── README.md               # 中文项目文档
├── README_EN.md            # 英文项目文档
└── LICENSE                 # 许可证文件
```

## 许可证

本项目采用 MIT 许可证 - 详情请查看 LICENSE 文件。

---

**重要提示**：股票市场投资有风险，本工具提供的预测和建议仅供参考，不构成投资建议。实际投资决策需谨慎。本项目仅用于学习和研究目的。