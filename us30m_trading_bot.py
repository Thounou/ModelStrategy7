"""
US30m Trading Bot with 5-minute entry and 1-hour confirmation
Using Markov Chains and Neural Networks
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import ta
from datetime import datetime, timedelta
import logging
import time
import pickle
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('us30m_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MT5Connection:
    """Handle MetaTrader 5 connection and operations"""
    
    def __init__(self, login: int, password: str, server: str):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        
    def connect(self):
        """Initialize connection to MT5"""
        if not mt5.initialize():
            logger.error(f"MT5 initialize() failed, error code: {mt5.last_error()}")
            return False
        
        # Login to MT5
        authorized = mt5.login(self.login, password=self.password, server=self.server)
        if not authorized:
            logger.error(f"Failed to connect to account #{self.login}, error code: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        self.connected = True
        logger.info(f"Connected to MT5 account #{self.login}")
        return True
    
    def disconnect(self):
        """Close MT5 connection"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def get_historical_data(self, symbol: str, timeframe, bars: int):
        """Fetch historical data from MT5"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None:
            logger.error(f"Failed to get data for {symbol}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def get_account_info(self):
        """Get account information"""
        if not self.connected:
            return None
        
        account_info = mt5.account_info()
        if account_info is None:
            return None
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'profit': account_info.profit
        }
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   sl: float = 0, tp: float = 0, comment: str = ""):
        """Place a trading order"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}")
                return None
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None
        
        # Determine order type and price
        if order_type.upper() == "BUY":
            trade_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif order_type.upper() == "SELL":
            trade_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            logger.error(f"Invalid order type: {order_type}")
            return None
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": trade_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment}")
            return None
        
        logger.info(f"Order placed: {order_type} {volume} lots of {symbol} at {price}")
        return result
    
    def close_position(self, position):
        """Close an open position"""
        if not self.connected:
            return None
        
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return None
        
        # Determine close price and order type
        if position.type == mt5.ORDER_TYPE_BUY:
            trade_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            trade_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": trade_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result
    
    def get_open_positions(self, symbol: str = None):
        """Get open positions"""
        if not self.connected:
            return []
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        return positions if positions is not None else []


class MarkovChainModel:
    """Markov Chain for pattern recognition and state transitions"""
    
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, df):
        """Extract features for Markov model"""
        features = []
        
        # Returns
        returns = df['close'].pct_change()
        
        # Volatility
        volatility = returns.rolling(window=20).std()
        
        # Range
        range_pct = (df['high'] - df['low']) / df['close']
        
        # Volume change
        volume_change = df['tick_volume'].pct_change()
        
        # RSI
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # ATR
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        atr_pct = atr / df['close']
        
        # ADX (trend strength)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        
        # Combine features
        features_df = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'range_pct': range_pct,
            'volume_change': volume_change,
            'rsi': rsi,
            'atr_pct': atr_pct,
            'adx': adx
        })
        
        return features_df.dropna()
    
    def train(self, df):
        """Train Markov model on historical data"""
        features_df = self.extract_features(df)
        features_scaled = self.scaler.fit_transform(features_df)
        
        self.model.fit(features_scaled)
        self.is_trained = True
        logger.info(f"Markov model trained with {self.n_states} states")
        
    def predict_state(self, df):
        """Predict current market state"""
        if not self.is_trained:
            logger.warning("Model not trained yet")
            return None
        
        features_df = self.extract_features(df)
        features_scaled = self.scaler.transform(features_df)
        
        # Predict states
        states = self.model.predict(features_scaled)
        
        # Get probabilities
        probabilities = self.model.predict_proba(features_scaled)
        
        return states[-1], probabilities[-1]
    
    def get_transition_matrix(self):
        """Get state transition matrix"""
        if self.is_trained:
            return self.model.transmat_
        return None
    
    def save_model(self, filepath):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'is_trained': self.is_trained}, f)
        logger.info(f"Markov model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = data['is_trained']
            logger.info(f"Markov model loaded from {filepath}")
            return True
        return False


class NeuralNetworkModel:
    """Neural Network with LSTM and Attention for price prediction"""
    
    def __init__(self, sequence_length=60, n_features=15):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_model(self):
        """Build LSTM + Attention model"""
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # CNN for feature extraction
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        
        # LSTM layers
        lstm_out = LSTM(128, return_sequences=True, dropout=0.2)(x)
        lstm_out = LSTM(64, return_sequences=True, dropout=0.2)(lstm_out)
        
        # Attention mechanism
        attention = Attention()([lstm_out, lstm_out])
        attention_flat = Flatten()(attention)
        
        # Dense layers
        dense = Dense(32, activation='relu')(attention_flat)
        dense = Dropout(0.2)(dense)
        
        # Output layer: 3 classes (BUY, HOLD, SELL)
        outputs = Dense(3, activation='softmax')(dense)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Neural network model built")
        return self.model
    
    def prepare_features(self, df):
        """Prepare features for neural network"""
        features = pd.DataFrame()
        
        # Price features
        features['close'] = df['close']
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df['tick_volume']
        
        # Technical indicators
        features['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        features['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        features['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        features['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # RSI
        features['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        features['bb_high'] = bollinger.bollinger_hband()
        features['bb_low'] = bollinger.bollinger_lband()
        features['bb_mid'] = bollinger.bollinger_mavg()
        
        # ATR
        features['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        return features.dropna()
    
    def create_sequences(self, data, labels=None):
        """Create sequences for LSTM input"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            if labels is not None:
                y.append(labels[i + self.sequence_length])
        
        return np.array(X), np.array(y) if labels is not None else None
    
    def prepare_labels(self, df, future_bars=5):
        """Prepare labels for training (BUY, HOLD, SELL)"""
        future_return = (df['close'].shift(-future_bars) - df['close']) / df['close']
        
        labels = np.zeros((len(df), 3))
        
        # BUY: future return > 0.2%
        labels[future_return > 0.002, 0] = 1
        
        # HOLD: -0.2% < future return < 0.2%
        labels[(future_return >= -0.002) & (future_return <= 0.002), 1] = 1
        
        # SELL: future return < -0.2%
        labels[future_return < -0.002, 2] = 1
        
        return labels
    
    def train(self, df, epochs=50, batch_size=32, validation_split=0.2):
        """Train neural network"""
        if self.model is None:
            self.build_model()
        
        # Prepare features and labels
        features_df = self.prepare_features(df)
        features_scaled = self.scaler.fit_transform(features_df)
        
        labels = self.prepare_labels(df)
        labels = labels[:len(features_scaled)]
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, labels)
        
        # Remove samples with all-zero labels
        valid_samples = y.sum(axis=1) > 0
        X = X[valid_samples]
        y = y[valid_samples]
        
        logger.info(f"Training neural network with {len(X)} samples")
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("Neural network training completed")
        
        return history
    
    def predict(self, df):
        """Predict trading signal"""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained yet")
            return None, None
        
        # Prepare features
        features_df = self.prepare_features(df)
        features_scaled = self.scaler.transform(features_df)
        
        # Create sequence
        X, _ = self.create_sequences(features_scaled)
        
        if len(X) == 0:
            return None, None
        
        # Predict
        predictions = self.model.predict(X[-1:], verbose=0)
        
        # Get signal and confidence
        signal_idx = np.argmax(predictions[0])
        confidence = predictions[0][signal_idx]
        
        # Map to signal: 0=BUY, 1=HOLD, 2=SELL
        signal_map = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}
        signal = signal_map[signal_idx]
        
        return signal, confidence
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model is not None:
            self.model.save(f"{filepath}_model.h5")
            with open(f"{filepath}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Neural network model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        if os.path.exists(f"{filepath}_model.h5"):
            self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
            with open(f"{filepath}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            logger.info(f"Neural network model loaded from {filepath}")
            return True
        return False


class RiskManager:
    """Risk management and position sizing"""
    
    def __init__(self, max_risk_per_trade=0.02, max_daily_loss=0.05):
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade
        self.max_daily_loss = max_daily_loss  # 5% daily loss limit
        self.daily_pnl = 0
        self.last_reset = datetime.now().date()
        
    def calculate_position_size(self, account_balance, entry_price, stop_loss_price):
        """Calculate position size based on risk"""
        if stop_loss_price == 0 or entry_price == 0:
            return 0
        
        risk_amount = account_balance * self.max_risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        
        # Round to 2 decimal places (lots)
        position_size = round(position_size, 2)
        
        return max(0.01, position_size)  # Minimum 0.01 lots
    
    def check_daily_loss_limit(self, account_balance, initial_balance):
        """Check if daily loss limit is reached"""
        # Reset daily PnL at start of new day
        if datetime.now().date() > self.last_reset:
            self.daily_pnl = 0
            self.last_reset = datetime.now().date()
        
        current_loss = (account_balance - initial_balance) / initial_balance
        
        if current_loss < -self.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {current_loss:.2%}")
            return False
        
        return True
    
    def calculate_stop_loss_take_profit(self, entry_price, signal, atr, risk_reward_ratio=2.0):
        """Calculate stop loss and take profit levels"""
        if signal == 'BUY':
            stop_loss = entry_price - (2 * atr)
            take_profit = entry_price + (risk_reward_ratio * 2 * atr)
        elif signal == 'SELL':
            stop_loss = entry_price + (2 * atr)
            take_profit = entry_price - (risk_reward_ratio * 2 * atr)
        else:
            return 0, 0
        
        return stop_loss, take_profit


class US30MTradingBot:
    """Main trading bot for US30m"""
    
    def __init__(self, mt5_login, mt5_password, mt5_server):
        self.symbol = "US30m"
        self.mt5 = MT5Connection(mt5_login, mt5_password, mt5_server)
        self.markov_model = MarkovChainModel(n_states=3)
        self.nn_model_5m = NeuralNetworkModel(sequence_length=60)
        self.nn_model_1h = NeuralNetworkModel(sequence_length=100)
        self.risk_manager = RiskManager()
        
        self.initial_balance = None
        self.is_running = False
        
    def initialize(self):
        """Initialize bot and connect to MT5"""
        logger.info("Initializing US30m Trading Bot...")
        
        if not self.mt5.connect():
            logger.error("Failed to connect to MT5")
            return False
        
        # Get initial balance
        account_info = self.mt5.get_account_info()
        if account_info:
            self.initial_balance = account_info['balance']
            logger.info(f"Account balance: ${self.initial_balance:.2f}")
        
        logger.info("Bot initialized successfully")
        return True
    
    def train_models(self, training_days=180):
        """Train Markov and Neural Network models"""
        logger.info("Training models...")
        
        # Fetch training data
        logger.info("Fetching 1-hour data for training...")
        df_1h = self.mt5.get_historical_data(self.symbol, mt5.TIMEFRAME_H1, training_days * 24)
        
        logger.info("Fetching 5-minute data for training...")
        df_5m = self.mt5.get_historical_data(self.symbol, mt5.TIMEFRAME_M5, training_days * 288)
        
        if df_1h is None or df_5m is None:
            logger.error("Failed to fetch training data")
            return False
        
        # Train Markov model on 1-hour data
        logger.info("Training Markov Chain model...")
        self.markov_model.train(df_1h)
        self.markov_model.save_model('markov_model.pkl')
        
        # Train neural networks
        logger.info("Training 5-minute Neural Network...")
        self.nn_model_5m.train(df_5m, epochs=30)
        self.nn_model_5m.save_model('nn_model_5m')
        
        logger.info("Training 1-hour Neural Network...")
        self.nn_model_1h.train(df_1h, epochs=30)
        self.nn_model_1h.save_model('nn_model_1h')
        
        logger.info("All models trained successfully")
        return True
    
    def load_models(self):
        """Load pre-trained models"""
        logger.info("Loading pre-trained models...")
        
        markov_loaded = self.markov_model.load_model('markov_model.pkl')
        nn_5m_loaded = self.nn_model_5m.load_model('nn_model_5m')
        nn_1h_loaded = self.nn_model_1h.load_model('nn_model_1h')
        
        if markov_loaded and nn_5m_loaded and nn_1h_loaded:
            logger.info("All models loaded successfully")
            return True
        else:
            logger.warning("Some models could not be loaded, training required")
            return False
    
    def generate_signal(self):
        """Generate trading signal with multi-timeframe confirmation"""
        # Fetch recent data
        df_5m = self.mt5.get_historical_data(self.symbol, mt5.TIMEFRAME_M5, 300)
        df_1h = self.mt5.get_historical_data(self.symbol, mt5.TIMEFRAME_H1, 200)
        
        if df_5m is None or df_1h is None:
            logger.error("Failed to fetch data for signal generation")
            return None, 0
        
        # Get Markov state (market regime)
        markov_state, state_probs = self.markov_model.predict_state(df_1h)
        logger.info(f"Market state: {markov_state}, Probabilities: {state_probs}")
        
        # Get neural network predictions
        signal_5m, confidence_5m = self.nn_model_5m.predict(df_5m)
        signal_1h, confidence_1h = self.nn_model_1h.predict(df_1h)
        
        logger.info(f"5-min signal: {signal_5m} (confidence: {confidence_5m:.2f})")
        logger.info(f"1-hour signal: {signal_1h} (confidence: {confidence_1h:.2f})")
        
        # Multi-timeframe confirmation
        if signal_5m == signal_1h and signal_5m != 'HOLD':
            # Both timeframes agree
            avg_confidence = (confidence_5m + confidence_1h) / 2
            
            # Require minimum confidence
            if avg_confidence >= 0.65:
                logger.info(f"Signal confirmed: {signal_5m} with confidence {avg_confidence:.2f}")
                return signal_5m, avg_confidence
        
        logger.info("No confirmed signal")
        return None, 0
    
    def execute_trade(self, signal, confidence):
        """Execute trade based on signal"""
        # Check daily loss limit
        account_info = self.mt5.get_account_info()
        if not account_info:
            logger.error("Failed to get account info")
            return False
        
        if not self.risk_manager.check_daily_loss_limit(account_info['balance'], self.initial_balance):
            logger.warning("Daily loss limit reached, skipping trade")
            return False
        
        # Check if already in position
        positions = self.mt5.get_open_positions(self.symbol)
        if len(positions) > 0:
            logger.info("Already in position, skipping new trade")
            return False
        
        # Get current price and ATR for stop loss calculation
        df_5m = self.mt5.get_historical_data(self.symbol, mt5.TIMEFRAME_M5, 100)
        if df_5m is None:
            return False
        
        atr = ta.volatility.AverageTrueRange(df_5m['high'], df_5m['low'], df_5m['close']).average_true_range().iloc[-1]
        current_price = df_5m['close'].iloc[-1]
        
        # Calculate stop loss and take profit
        sl, tp = self.risk_manager.calculate_stop_loss_take_profit(current_price, signal, atr)
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            account_info['balance'], current_price, sl
        )
        
        logger.info(f"Executing {signal} order:")
        logger.info(f"  Price: {current_price:.2f}")
        logger.info(f"  Position size: {position_size} lots")
        logger.info(f"  Stop Loss: {sl:.2f}")
        logger.info(f"  Take Profit: {tp:.2f}")
        logger.info(f"  Confidence: {confidence:.2f}")
        
        # Place order
        order_type = "BUY" if signal == "BUY" else "SELL"
        result = self.mt5.place_order(
            self.symbol, 
            order_type, 
            position_size,
            sl=sl,
            tp=tp,
            comment=f"ML_{signal}_{confidence:.2f}"
        )
        
        if result:
            logger.info(f"Trade executed successfully: {result.order}")
            return True
        else:
            logger.error("Trade execution failed")
            return False
    
    def manage_positions(self):
        """Manage open positions"""
        positions = self.mt5.get_open_positions(self.symbol)
        
        if len(positions) == 0:
            return
        
        logger.info(f"Managing {len(positions)} open position(s)")
        
        for position in positions:
            # Get current price
            df_5m = self.mt5.get_historical_data(self.symbol, mt5.TIMEFRAME_M5, 50)
            if df_5m is None:
                continue
            
            current_price = df_5m['close'].iloc[-1]
            
            # Calculate profit/loss
            if position.type == mt5.ORDER_TYPE_BUY:
                pnl_pct = (current_price - position.price_open) / position.price_open
            else:
                pnl_pct = (position.price_open - current_price) / position.price_open
            
            logger.info(f"Position #{position.ticket}: P/L = {pnl_pct:.2%}")
            
            # Check for trailing stop or early exit conditions
            # (can be extended with more sophisticated exit logic)
    
    def run(self, check_interval=300):
        """Main trading loop"""
        logger.info("Starting trading bot...")
        self.is_running = True
        
        while self.is_running:
            try:
                logger.info("=" * 60)
                logger.info(f"Checking for trading opportunities at {datetime.now()}")
                
                # Manage existing positions
                self.manage_positions()
                
                # Generate signal
                signal, confidence = self.generate_signal()
                
                # Execute trade if signal is strong
                if signal:
                    self.execute_trade(signal, confidence)
                
                # Display account status
                account_info = self.mt5.get_account_info()
                if account_info:
                    logger.info(f"Account Status:")
                    logger.info(f"  Balance: ${account_info['balance']:.2f}")
                    logger.info(f"  Equity: ${account_info['equity']:.2f}")
                    logger.info(f"  Profit: ${account_info['profit']:.2f}")
                
                # Wait before next check
                logger.info(f"Waiting {check_interval} seconds until next check...")
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Stopping bot...")
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)
        
        self.shutdown()
    
    def shutdown(self):
        """Shutdown bot and close connections"""
        logger.info("Shutting down trading bot...")
        self.is_running = False
        self.mt5.disconnect()
        logger.info("Bot shutdown complete")


def main():
    """Main entry point"""
    # MT5 Credentials
    MT5_LOGIN = 222566231
    MT5_PASSWORD = "1234Narra#"
    MT5_SERVER = "Exness-MT5Real30"
    
    # Create bot instance
    bot = US30MTradingBot(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
    
    # Initialize bot
    if not bot.initialize():
        logger.error("Failed to initialize bot")
        return
    
    # Try to load pre-trained models, or train new ones
    if not bot.load_models():
        logger.info("Training models (this may take a while)...")
        if not bot.train_models(training_days=180):
            logger.error("Failed to train models")
            bot.shutdown()
            return
    
    # Start trading
    try:
        bot.run(check_interval=300)  # Check every 5 minutes
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        bot.shutdown()


if __name__ == "__main__":
    main()
