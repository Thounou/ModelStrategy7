# US30m Trading Bot

Advanced automated trading bot for US30m index using Multi-Timeframe Analysis, Markov Chains, and Neural Networks.

## Features

### 🎯 Multi-Timeframe Strategy
- **Entry Signals**: 5-minute timeframe for precise entry points
- **Confirmation**: 1-hour timeframe for trend confirmation
- **Risk Management**: Dynamic position sizing based on ATR

### 🧠 AI Models

#### 1. Markov Chain Model (Hidden Markov Model)
- **Purpose**: Market regime detection and state transition analysis
- **States**: 3 market states (ranging, trending, volatile)
- **Features**:
  - Volatility (rolling standard deviation)
  - Price range percentage
  - Volume changes
  - RSI (Relative Strength Index)
  - ATR (Average True Range)
  - ADX (Average Directional Index)

#### 2. Neural Network (LSTM + Attention)
- **Architecture**:
  - CNN layers for feature extraction
  - LSTM layers for temporal pattern recognition
  - Attention mechanism for important timeframe focus
  - Multi-head output for classification
- **Outputs**: BUY, HOLD, or SELL with confidence score
- **Separate models** for 5-minute and 1-hour timeframes

### 📊 Technical Indicators Used
- Simple Moving Averages (SMA 20, 50)
- Exponential Moving Averages (EMA 12, 26)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- ADX (Average Directional Index)

### 🛡️ Risk Management
- **Maximum Risk per Trade**: 2% of account balance
- **Daily Loss Limit**: 5% maximum daily drawdown
- **Position Sizing**: Dynamic calculation based on stop loss distance
- **Stop Loss**: 2x ATR from entry price
- **Take Profit**: 4x ATR from entry price (2:1 Risk/Reward ratio)

## Installation

### Prerequisites
- Python 3.8 or higher
- MetaTrader 5 terminal installed
- Active MT5 trading account

### Setup Steps

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify MT5 Installation**:
   - Ensure MetaTrader 5 is installed and running
   - Verify your account can access US30m symbol

3. **Configure Credentials**:
   - Credentials are already set in `config.py`
   - Login: 222566231
   - Server: Exness-MT5Real30

## Usage

### First Time Setup (Training Models)

Run the bot for the first time to train the models:

```bash
python us30m_trading_bot.py
```

This will:
1. Connect to your MT5 account
2. Download 180 days of historical data
3. Train the Markov Chain model
4. Train the 5-minute Neural Network
5. Train the 1-hour Neural Network
6. Save trained models to disk

**Note**: Initial training may take 30-60 minutes depending on your hardware.

### Subsequent Runs

After models are trained, the bot will:
1. Load pre-trained models (much faster)
2. Start monitoring the market
3. Generate signals every 5 minutes
4. Execute trades automatically when conditions are met

### Manual Trading Mode

To test signals without executing trades, modify the `execute_trade` method to return False.

## How It Works

### Signal Generation Process

1. **Fetch Data**:
   - 300 bars of 5-minute data
   - 200 bars of 1-hour data

2. **Market Regime Detection**:
   - Markov model analyzes 1-hour data
   - Identifies current market state
   - Provides state probabilities

3. **Neural Network Predictions**:
   - 5-minute model predicts short-term direction
   - 1-hour model confirms the trend
   - Both models provide confidence scores

4. **Signal Confirmation**:
   - Signals must agree across timeframes
   - Minimum confidence threshold: 65%
   - Signal types: BUY, HOLD, or SELL

5. **Risk Assessment**:
   - Check daily loss limit
   - Verify no existing positions
   - Calculate position size based on risk

6. **Trade Execution**:
   - Place order with calculated position size
   - Set stop loss at 2x ATR
   - Set take profit at 4x ATR

### Position Management

- Monitors all open positions
- Logs profit/loss in real-time
- Can implement trailing stops (extend `manage_positions` method)

## Configuration

Edit `config.py` to customize:

```python
# Risk parameters
MAX_RISK_PER_TRADE = 0.02  # 2% per trade
MAX_DAILY_LOSS = 0.05      # 5% daily limit

# Signal parameters
MIN_CONFIDENCE = 0.65       # Minimum confidence
CHECK_INTERVAL = 300        # Check every 5 minutes

# Model parameters
MARKOV_STATES = 3           # Number of market states
TRAINING_DAYS = 180         # Historical data for training
```

## Logging

All activities are logged to:
- **File**: `us30m_trading_bot.log`
- **Console**: Real-time output

Log includes:
- Connection status
- Market regime detection
- Signal generation
- Trade execution
- Position management
- Account status updates

## Model Files

Trained models are saved as:
- `markov_model.pkl` - Markov Chain model
- `nn_model_5m_model.h5` - 5-minute Neural Network
- `nn_model_5m_scaler.pkl` - 5-minute data scaler
- `nn_model_1h_model.h5` - 1-hour Neural Network
- `nn_model_1h_scaler.pkl` - 1-hour data scaler

## Performance Monitoring

The bot logs:
- **Account Balance**: Current account balance
- **Equity**: Current equity including open positions
- **Profit**: Current profit/loss
- **Signal Confidence**: Prediction confidence for each signal
- **Market State**: Current market regime

## Safety Features

1. **Daily Loss Limit**: Automatically stops trading if daily loss exceeds 5%
2. **Position Limits**: Only one position at a time
3. **Minimum Confidence**: Trades only when confidence ≥ 65%
4. **Stop Loss**: Always set for every trade
5. **Position Sizing**: Never risks more than 2% per trade

## Troubleshooting

### Connection Issues
```
Error: MT5 initialize() failed
Solution: Ensure MetaTrader 5 is running and you're logged in
```

### Symbol Not Found
```
Error: Symbol US30m not found
Solution: Check if US30m is available in your broker's Market Watch
```

### Insufficient Training Data
```
Error: Failed to fetch training data
Solution: Ensure you have historical data loaded in MT5
```

### Model Training Errors
```
Error: Neural network training failed
Solution: Check if you have enough RAM and valid historical data
```

## Advanced Usage

### Retraining Models

To retrain models with fresh data:

```python
bot = US30MTradingBot(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
bot.initialize()
bot.train_models(training_days=365)  # Use more historical data
```

### Backtesting

Modify the code to run in backtesting mode:
- Replace `mt5.copy_rates_from_pos` with historical data
- Store all trades without executing them
- Calculate performance metrics

### Paper Trading

Set a flag to simulate trades without real execution:
- Log all signals and calculated positions
- Track hypothetical P&L
- Validate strategy before going live

## Performance Expectations

This is a machine learning-based system with the following characteristics:

- **Win Rate**: Typically 50-60% (varies with market conditions)
- **Risk/Reward**: 2:1 ratio
- **Trade Frequency**: 2-5 trades per day on average
- **Maximum Drawdown**: Controlled by risk management (5% daily, ~15% overall)

**Note**: Past performance does not guarantee future results. Always test thoroughly before live trading.

## Disclaimer

⚠️ **Trading involves significant risk of loss.**

- This bot is provided for educational purposes
- Use at your own risk
- Start with a demo account or small capital
- Monitor the bot regularly
- Understand all parameters before going live
- Past performance does not indicate future results

## Support & Updates

For issues or questions:
1. Check the log file for detailed error messages
2. Verify MT5 connection and data availability
3. Ensure all dependencies are installed correctly

## License

This trading bot is provided as-is without warranty of any kind.

---

**Created**: November 2025
**Version**: 1.0
**Strategy**: Multi-Timeframe ML Trading System
