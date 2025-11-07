"""
Quick start script for US30m Trading Bot
Run this to quickly start the trading bot with default settings
"""

import sys
import os
from us30m_trading_bot import US30MTradingBot
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
import logging

def main():
    print("=" * 70)
    print("US30m TRADING BOT - QUICK START")
    print("=" * 70)
    print()
    print("This bot will:")
    print("  ✓ Connect to your MT5 account")
    print("  ✓ Load or train AI models (Markov Chain + Neural Networks)")
    print("  ✓ Monitor US30m market every 5 minutes")
    print("  ✓ Execute trades with proper risk management")
    print()
    print("Risk Parameters:")
    print("  • Max risk per trade: 2%")
    print("  • Daily loss limit: 5%")
    print("  • Stop loss: 2x ATR")
    print("  • Take profit: 4x ATR (2:1 R:R)")
    print()
    print("=" * 70)
    print()
    
    # Confirm start
    response = input("Start the trading bot? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Bot start cancelled.")
        sys.exit(0)
    
    print()
    print("Initializing bot...")
    print()
    
    # Create bot instance
    bot = US30MTradingBot(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
    
    # Initialize
    if not bot.initialize():
        print("ERROR: Failed to initialize bot. Check your MT5 connection.")
        sys.exit(1)
    
    # Load or train models
    print()
    if not bot.load_models():
        print("No pre-trained models found.")
        print()
        train = input("Train new models? This will take 30-60 minutes (yes/no): ").strip().lower()
        
        if train in ['yes', 'y']:
            print()
            print("Training models with 180 days of historical data...")
            print("This may take a while. Please wait...")
            print()
            
            if not bot.train_models(training_days=180):
                print("ERROR: Failed to train models.")
                bot.shutdown()
                sys.exit(1)
            
            print()
            print("Models trained successfully!")
        else:
            print("Cannot run bot without trained models.")
            bot.shutdown()
            sys.exit(0)
    
    # Start trading
    print()
    print("=" * 70)
    print("BOT IS NOW RUNNING")
    print("=" * 70)
    print()
    print("The bot will check for trading opportunities every 5 minutes.")
    print("Press Ctrl+C to stop the bot.")
    print()
    
    try:
        bot.run(check_interval=300)
    except KeyboardInterrupt:
        print()
        print("Bot stopped by user.")
    finally:
        bot.shutdown()
        print()
        print("Bot shutdown complete. Goodbye!")


if __name__ == "__main__":
    main()
