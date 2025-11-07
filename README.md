# US30m Automated Trading Bot

## 🚀 Advanced AI-Powered Trading System

Professional trading bot for US30m index using **Markov Chains** and **Neural Networks** with multi-timeframe analysis (5-minute entry, 1-hour confirmation).

## 📁 Project Files

- **`us30m_trading_bot.py`** - Main trading bot implementation
- **`quick_start.py`** - Easy-to-use startup script
- **`backtest.py`** - Backtesting module for strategy validation
- **`config.py`** - Configuration and parameters
- **`requirements.txt`** - Python dependencies
- **`US30M_TRADING_BOT_README.md`** - Comprehensive documentation

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Bot
```bash
python quick_start.py
```

### 3. Backtest (Optional)
```bash
python backtest.py
```

## 🧠 AI Models

### Markov Chain Model (HMM)
- 3-state Hidden Markov Model for market regime detection
- Identifies: ranging, trending, and volatile market conditions
- Uses volatility, RSI, ATR, ADX, and volume features

### Neural Network (LSTM + Attention)
- CNN feature extraction layers
- LSTM for temporal pattern recognition
- Attention mechanism for timeframe weighting
- Separate models for 5-minute and 1-hour analysis
- Outputs: BUY, HOLD, SELL with confidence scores

## 📊 Strategy Overview

**Entry Timeframe**: 5-minute chart (precise entry points)
**Confirmation Timeframe**: 1-hour chart (trend validation)
**Signal Requirement**: Both timeframes must agree with ≥65% confidence

## 🛡️ Risk Management

- **Max Risk per Trade**: 2% of account balance
- **Daily Loss Limit**: 5% maximum drawdown
- **Stop Loss**: 2x ATR from entry
- **Take Profit**: 4x ATR from entry (2:1 R:R ratio)
- **Position Sizing**: Dynamic based on account size and stop distance

## 📈 Technical Indicators

- SMA (20, 50), EMA (12, 26)
- RSI (14), MACD
- Bollinger Bands
- ATR, ADX
- Volume analysis

## 🔐 MT5 Credentials

```python
Login: 222566231
Password: 1234Narra#
Server: Exness-MT5Real30
Symbol: US30m
```

## 📝 Features

✅ Real-time market monitoring
✅ Automated trade execution
✅ Multi-timeframe signal confirmation
✅ Advanced AI-based predictions
✅ Risk-managed position sizing
✅ Stop loss and take profit automation
✅ Comprehensive logging
✅ Backtesting capability
✅ Daily loss protection

## ⚠️ Important Notes

- The bot checks for trading opportunities every 5 minutes
- First run will train models (30-60 minutes)
- Subsequent runs load pre-trained models (instant start)
- All trades are logged to `us30m_trading_bot.log`
- Models are saved for reuse: `markov_model.pkl`, `nn_model_5m_model.h5`, `nn_model_1h_model.h5`

## 📚 Documentation

For detailed information, see **`US30M_TRADING_BOT_README.md`**

## ⚠️ Disclaimer

Trading involves significant risk. This bot is provided for educational purposes. Always test thoroughly on a demo account before live trading. Past performance does not guarantee future results.

---

**Version**: 1.0
**Created**: November 2025
**Strategy**: Multi-Timeframe ML Trading System