"""
Configuration file for US30m Trading Bot
"""

# MT5 Credentials
MT5_LOGIN = 222566231
MT5_PASSWORD = "1234Narra#"
MT5_SERVER = "Exness-MT5Real30"

# Trading Parameters
SYMBOL = "US30m"
TIMEFRAME_ENTRY = "M5"  # 5-minute for entry signals
TIMEFRAME_CONFIRMATION = "H1"  # 1-hour for confirmation

# Model Parameters
MARKOV_STATES = 3  # Number of market states
NN_SEQUENCE_LENGTH_5M = 60  # 60 bars (5 hours of 5-min data)
NN_SEQUENCE_LENGTH_1H = 100  # 100 bars (100 hours of data)
NN_FEATURES = 15

# Training Parameters
TRAINING_DAYS = 180  # Days of historical data for training
NN_EPOCHS = 30
NN_BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Risk Management
MAX_RISK_PER_TRADE = 0.02  # 2% risk per trade
MAX_DAILY_LOSS = 0.05  # 5% maximum daily loss
RISK_REWARD_RATIO = 2.0  # Target 2:1 risk/reward
STOP_LOSS_ATR_MULTIPLE = 2.0  # Stop loss at 2x ATR
TAKE_PROFIT_ATR_MULTIPLE = 4.0  # Take profit at 4x ATR

# Signal Parameters
MIN_CONFIDENCE = 0.65  # Minimum confidence for trade execution
CHECK_INTERVAL = 300  # Check for signals every 5 minutes (in seconds)

# Position Sizing
MIN_POSITION_SIZE = 0.01  # Minimum lot size
MAX_POSITION_SIZE = 1.0  # Maximum lot size

# Logging
LOG_FILE = "us30m_trading_bot.log"
LOG_LEVEL = "INFO"
