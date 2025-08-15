# LSTM-Stock-Predictor

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-green.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

English | [中文](README.md)

> [!NOTE]
> **Developed through learning based on the [stock_trading](https://github.com/MilleXi/stock_trading) project.**

> [!IMPORTANT]
> **This project is solely a result of learning and practicing LSTM technology, and does not constitute any form of investment advice. Its prediction results are also not accurate for reference.**

---

## Project Overview

A stock prediction system based on LSTM prediction and reinforcement learning. This system combines the predictive capabilities of deep learning with the decision-making capabilities of reinforcement learning to automatically perform stock price prediction and trading decisions.

## Installation Guide

### Prerequisites
- Python 3.7+
- pip

### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/zengplus/LSTM-Stock-Predictor.git
cd LSTM-Stock-Predictor
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage Instructions

### Launch Application
```bash
streamlit run app.py
```

### Interface Operation Guide

#### 1. Sidebar Parameter Settings
- Enter stock codes (multiple codes separated by commas)
- Set date range
- Adjust initial capital, training rounds, and other parameters
- Select data source (Yahoo Finance or AKShare)
- Set language (Chinese/English)

#### 2. Function Buttons
- **Run Full Analysis**: Execute data acquisition, model training, prediction, and trading strategy generation
- **Get Tomorrow's Prediction Only**: Quickly obtain tomorrow's price prediction and investment recommendations

#### 3. Result Display
- Stock prediction results (RMSE, MAE, accuracy)
- Trading strategy analysis (total returns, investment return rate)
- Tomorrow's price prediction and investment recommendations
- Visualization charts (price trends, trading signals, etc.)

### Interface Preview

#### Home Page Example
![Home Page](images/home_page.png)

#### Results Page Example
![Results Page](images/results_page.png)

## Project Structure

```
LSTM-Stock-Predictor/
├── results/                # Analysis results storage directory
│   └── session_{timestamp}/ # Session directory named with timestamp
│       ├── analytics/        # Analysis reports
│       ├── feature_importance/ # Feature importance analysis results
│       ├── models/           # Saved models
│       ├── predictions/      # Prediction results
│       ├── scalers/          # Normalizers
│       ├── trading/          # Trading records
│       └── session_summary.json # Session summary
├── data/                   # Stock data storage directory
├── images/                 # Project documentation images directory
├── app.py                  # Main application
├── requirements.txt        # Dependencies list
├── README.md               # Chinese Project documentation
├── README_EN.md            # English Project documentation
└── LICENSE                 # License file
```

---

**Important Notice**: Stock market investment involves risks. The predictions and recommendations provided by this tool are for reference only and do not constitute investment advice. Actual investment decisions should be made with caution. This project is for learning and research purposes only. 