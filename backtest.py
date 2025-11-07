"""
Backtesting module for US30m Trading Bot
Test the strategy on historical data before live trading
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from us30m_trading_bot import (
    MT5Connection, MarkovChainModel, NeuralNetworkModel, RiskManager
)
from config import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtest:
    """Backtesting engine for trading strategy"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.equity_curve = []
        self.positions = []
        
    def run_backtest(self, df_5m, df_1h, markov_model, nn_model_5m, nn_model_1h, 
                     risk_manager, start_idx=1000):
        """
        Run backtest on historical data
        
        Args:
            df_5m: 5-minute historical data
            df_1h: 1-hour historical data
            markov_model: Trained Markov model
            nn_model_5m: Trained 5-minute neural network
            nn_model_1h: Trained 1-hour neural network
            risk_manager: Risk management instance
            start_idx: Starting index for backtest (skip training data)
        """
        logger.info(f"Starting backtest with ${self.initial_capital:.2f}")
        
        # Align timeframes
        df_5m = df_5m.copy()
        df_1h = df_1h.copy()
        
        current_position = None
        
        # Iterate through historical data
        for i in range(start_idx, len(df_5m) - 1):
            # Get data up to current point
            data_5m = df_5m.iloc[:i+1].copy()
            
            # Get corresponding 1-hour data
            current_time = data_5m.iloc[-1]['time']
            data_1h = df_1h[df_1h['time'] <= current_time].copy()
            
            if len(data_1h) < 100:
                continue
            
            # Get current price
            current_price = data_5m.iloc[-1]['close']
            current_time = data_5m.iloc[-1]['time']
            
            # Manage existing position
            if current_position is not None:
                # Check stop loss and take profit
                if current_position['type'] == 'BUY':
                    pnl = (current_price - current_position['entry_price']) * current_position['size']
                    
                    if current_price <= current_position['sl']:
                        # Stop loss hit
                        self.close_position(current_position, current_price, current_time, 'SL Hit')
                        current_position = None
                    elif current_price >= current_position['tp']:
                        # Take profit hit
                        self.close_position(current_position, current_price, current_time, 'TP Hit')
                        current_position = None
                else:  # SELL
                    pnl = (current_position['entry_price'] - current_price) * current_position['size']
                    
                    if current_price >= current_position['sl']:
                        # Stop loss hit
                        self.close_position(current_position, current_price, current_time, 'SL Hit')
                        current_position = None
                    elif current_price <= current_position['tp']:
                        # Take profit hit
                        self.close_position(current_position, current_price, current_time, 'TP Hit')
                        current_position = None
            
            # Generate new signal if no position
            if current_position is None:
                signal, confidence = self.generate_signal(
                    data_5m, data_1h, markov_model, nn_model_5m, nn_model_1h
                )
                
                if signal and confidence >= MIN_CONFIDENCE:
                    # Calculate position parameters
                    import ta as ta_lib
                    atr = ta_lib.volatility.AverageTrueRange(
                        data_5m['high'], data_5m['low'], data_5m['close']
                    ).average_true_range().iloc[-1]
                    
                    if signal == 'BUY':
                        sl = current_price - (STOP_LOSS_ATR_MULTIPLE * atr)
                        tp = current_price + (TAKE_PROFIT_ATR_MULTIPLE * atr)
                    else:  # SELL
                        sl = current_price + (STOP_LOSS_ATR_MULTIPLE * atr)
                        tp = current_price - (TAKE_PROFIT_ATR_MULTIPLE * atr)
                    
                    # Calculate position size
                    position_size = risk_manager.calculate_position_size(
                        self.capital, current_price, sl
                    )
                    
                    # Open position
                    current_position = {
                        'type': signal,
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'sl': sl,
                        'tp': tp,
                        'size': position_size,
                        'confidence': confidence
                    }
                    
                    logger.info(f"[{current_time}] Opening {signal} at {current_price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}")
            
            # Track equity
            if current_position:
                if current_position['type'] == 'BUY':
                    unrealized_pnl = (current_price - current_position['entry_price']) * current_position['size']
                else:
                    unrealized_pnl = (current_position['entry_price'] - current_price) * current_position['size']
                
                equity = self.capital + unrealized_pnl
            else:
                equity = self.capital
            
            self.equity_curve.append({
                'time': current_time,
                'equity': equity,
                'capital': self.capital
            })
        
        # Close any remaining position
        if current_position is not None:
            final_price = df_5m.iloc[-1]['close']
            final_time = df_5m.iloc[-1]['time']
            self.close_position(current_position, final_price, final_time, 'End of backtest')
        
        # Generate report
        self.generate_report()
    
    def generate_signal(self, data_5m, data_1h, markov_model, nn_model_5m, nn_model_1h):
        """Generate trading signal"""
        try:
            # Get Markov state
            state, state_probs = markov_model.predict_state(data_1h)
            
            # Get neural network predictions
            signal_5m, confidence_5m = nn_model_5m.predict(data_5m)
            signal_1h, confidence_1h = nn_model_1h.predict(data_1h)
            
            # Multi-timeframe confirmation
            if signal_5m == signal_1h and signal_5m != 'HOLD':
                avg_confidence = (confidence_5m + confidence_1h) / 2
                return signal_5m, avg_confidence
            
            return None, 0
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None, 0
    
    def close_position(self, position, exit_price, exit_time, reason):
        """Close a position and record trade"""
        if position['type'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - exit_price) * position['size']
        
        self.capital += pnl
        
        trade = {
            'type': position['type'],
            'entry_price': position['entry_price'],
            'entry_time': position['entry_time'],
            'exit_price': exit_price,
            'exit_time': exit_time,
            'pnl': pnl,
            'pnl_pct': (pnl / (position['entry_price'] * position['size'])) * 100,
            'size': position['size'],
            'confidence': position['confidence'],
            'reason': reason
        }
        
        self.trades.append(trade)
        
        logger.info(f"[{exit_time}] Closed {position['type']} at {exit_price:.2f}, P/L: ${pnl:.2f} ({reason})")
    
    def generate_report(self):
        """Generate backtest performance report"""
        if len(self.trades) == 0:
            logger.warning("No trades executed during backtest")
            return
        
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        
        # Overall metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0]) if losing_trades > 0 else 0
        
        profit_factor = abs(sum(t['pnl'] for t in self.trades if t['pnl'] > 0) / 
                           sum(t['pnl'] for t in self.trades if t['pnl'] < 0)) if losing_trades > 0 else 0
        
        final_capital = self.capital
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate max drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        print(f"\nInitial Capital:    ${self.initial_capital:,.2f}")
        print(f"Final Capital:      ${final_capital:,.2f}")
        print(f"Total Return:       {total_return:.2f}%")
        print(f"Total P/L:          ${total_pnl:,.2f}")
        print()
        print(f"Total Trades:       {total_trades}")
        print(f"Winning Trades:     {winning_trades}")
        print(f"Losing Trades:      {losing_trades}")
        print(f"Win Rate:           {win_rate:.2f}%")
        print()
        print(f"Average Win:        ${avg_win:.2f}")
        print(f"Average Loss:       ${avg_loss:.2f}")
        print(f"Profit Factor:      {profit_factor:.2f}")
        print(f"Max Drawdown:       {max_drawdown:.2f}%")
        print()
        
        # Best and worst trades
        best_trade = max(self.trades, key=lambda x: x['pnl'])
        worst_trade = min(self.trades, key=lambda x: x['pnl'])
        
        print(f"Best Trade:         ${best_trade['pnl']:.2f} ({best_trade['type']})")
        print(f"Worst Trade:        ${worst_trade['pnl']:.2f} ({worst_trade['type']})")
        print()
        print("=" * 70)
        
        # Plot equity curve
        self.plot_equity_curve()
    
    def plot_equity_curve(self):
        """Plot equity curve"""
        equity_df = pd.DataFrame(self.equity_curve)
        
        plt.figure(figsize=(14, 7))
        
        plt.subplot(2, 1, 1)
        plt.plot(equity_df['time'], equity_df['equity'], label='Equity', linewidth=2)
        plt.plot(equity_df['time'], equity_df['capital'], label='Capital (Realized)', linewidth=2, alpha=0.7)
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        
        plt.subplot(2, 1, 2)
        plt.fill_between(equity_df['time'], equity_df['drawdown'], 0, color='red', alpha=0.3)
        plt.plot(equity_df['time'], equity_df['drawdown'], color='red', linewidth=2)
        plt.title('Drawdown', fontsize=14, fontweight='bold')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150)
        print("\nEquity curve saved to: backtest_results.png")
        plt.show()


def main():
    """Run backtest"""
    print("US30m Trading Bot - Backtesting")
    print("=" * 70)
    print()
    
    # Connect to MT5
    print("Connecting to MT5...")
    mt5_conn = MT5Connection(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
    if not mt5_conn.connect():
        print("Failed to connect to MT5")
        return
    
    # Fetch historical data
    print("Fetching historical data...")
    df_5m = mt5_conn.get_historical_data(SYMBOL, mt5.TIMEFRAME_M5, 365 * 288)  # 1 year
    df_1h = mt5_conn.get_historical_data(SYMBOL, mt5.TIMEFRAME_H1, 365 * 24)
    
    if df_5m is None or df_1h is None:
        print("Failed to fetch data")
        mt5_conn.disconnect()
        return
    
    print(f"Fetched {len(df_5m)} 5-minute bars and {len(df_1h)} 1-hour bars")
    
    # Load or train models
    print("\nLoading models...")
    markov_model = MarkovChainModel()
    nn_model_5m = NeuralNetworkModel()
    nn_model_1h = NeuralNetworkModel(sequence_length=100)
    
    if not (markov_model.load_model('markov_model.pkl') and 
            nn_model_5m.load_model('nn_model_5m') and
            nn_model_1h.load_model('nn_model_1h')):
        print("Models not found. Training...")
        
        # Train models on first 80% of data
        train_split = int(len(df_5m) * 0.8)
        
        print("Training Markov model...")
        markov_model.train(df_1h[:train_split])
        markov_model.save_model('markov_model.pkl')
        
        print("Training 5-minute neural network...")
        nn_model_5m.train(df_5m[:train_split], epochs=30)
        nn_model_5m.save_model('nn_model_5m')
        
        print("Training 1-hour neural network...")
        nn_model_1h.train(df_1h[:train_split], epochs=30)
        nn_model_1h.save_model('nn_model_1h')
    
    print("Models loaded successfully")
    
    # Create risk manager
    risk_manager = RiskManager(
        max_risk_per_trade=MAX_RISK_PER_TRADE,
        max_daily_loss=MAX_DAILY_LOSS
    )
    
    # Run backtest
    print("\nRunning backtest...")
    print("This may take a few minutes...")
    print()
    
    backtest = Backtest(initial_capital=10000)
    
    # Test on last 20% of data
    test_start = int(len(df_5m) * 0.8)
    
    backtest.run_backtest(
        df_5m[test_start:],
        df_1h,
        markov_model,
        nn_model_5m,
        nn_model_1h,
        risk_manager,
        start_idx=1000
    )
    
    mt5_conn.disconnect()


if __name__ == "__main__":
    main()
