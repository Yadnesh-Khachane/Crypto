"""
main.py - Smart RL Trading Bot - 24/7 Runner
Shark Exchange | ₹1,500 Capital | 80x Leverage
"""

import sys
import os
import signal
import time
from datetime import datetime
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

from data_feed import DataFeed
from trader import SmartTrader
from gui import TradingGUI
from config import *

class BotRunner:
    """24/7 Bot Runner with auto-save and monitoring"""
    
    def __init__(self):
        self.data_feed = None
        self.trader = None
        self.gui = None
        self.start_time = datetime.now()
        self.last_save_time = datetime.now()
        self.save_interval = 300  # Save every 5 minutes
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutdown signal received...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Cleanup and save before exit"""
        print("💾 Saving RL model...")
        if self.trader:
            self.trader.rl_agent.save()
        
        runtime = datetime.now() - self.start_time
        print(f"⏱️ Total runtime: {runtime}")
        print("👋 Bot shutting down. Goodbye!")
    
    def run(self):
        """Start the bot"""
        print("=" * 70)
        print("🤖 SMART RL TRADING BOT - 24/7 RUNNER")
        print("=" * 70)
        print(f"📅 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Capital: ${INITIAL_CAPITAL_USD:.2f} (₹{INITIAL_CAPITAL_INR})")
        print(f"📊 Leverage: {LEVERAGE}x")
        print(f"📈 Position: {BASE_BTC_SIZE} BTC (${POSITION_VALUE_USD:.2f})")
        print(f"🧠 RL: DQN with Prioritized Experience Replay")
        print(f"📡 Data: Shark Exchange WebSocket")
        print(f"💾 Auto-save: Every {self.save_interval}s")
        print("=" * 70)
        
        # Initialize data feed
        print("\n🔌 Initializing data feed...")
        self.data_feed = DataFeed(use_live=True)
        
        # Wait for initial data
        print("⏳ Waiting for price data...")
        timeout = 30
        start_wait = time.time()
        while self.data_feed.get_current_price() == 65000:
            if time.time() - start_wait > timeout:
                print("⚠️ Live data timeout, using synthetic fallback")
                break
            time.sleep(1)
        
        print(f"✅ Initial price: ${self.data_feed.get_current_price():,.2f}")
        
        # Initialize trader
        print("\n🤖 Initializing RL trader...")
        self.trader = SmartTrader(
            initial_capital=INITIAL_CAPITAL_USD,
            leverage=LEVERAGE,
            use_rl=True
        )
        
        # Load existing model if available
        model_path = os.path.join(MODEL_DIR, MODEL_FILE)
        if os.path.exists(model_path):
            print("📂 Loading saved RL model...")
            success = self.trader.rl_agent.load(model_path)
            if success:
                print(f"   Epsilon: {self.trader.rl_agent.epsilon:.4f}")
                print(f"   Training steps: {self.trader.rl_agent.train_step}")
        
        # Start GUI
        print("\n🖥️ Starting GUI...")
        app = QApplication(sys.argv)
        self.gui = TradingGUI(self.trader, self.data_feed)
        self.gui.show()
        
        # Setup auto-save timer
        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self.auto_save)
        self.save_timer.start(self.save_interval * 1000)
        
        # Print startup message
        print("\n" + "=" * 70)
        print("✅ BOT IS RUNNING!")
        print("=" * 70)
        print("📊 Monitor the GUI for real-time updates")
        print("🧠 RL agent will learn from each trade")
        print("💾 Models auto-saved every 5 minutes")
        print("🖱️ Use mouse wheel to scroll trade log")
        print("⏸ Press 'Pause Trading' to stop taking new positions")
        print("=" * 70)
        
        # Run application
        exit_code = app.exec_()
        
        # Cleanup
        self.cleanup()
        sys.exit(exit_code)
    
    def auto_save(self):
        """Auto-save RL model periodically"""
        if self.trader and self.trader.rl_agent.train_step > 0:
            self.trader.rl_agent.save()
            self.last_save_time = datetime.now()
            
            # Log auto-save
            stats = self.trader.get_stats()
            print(f"💾 Auto-saved | Trades: {stats['total_trades']} | "
                  f"P&L: ${stats['total_pnl']:+.4f} | ε: {stats['rl_epsilon']:.4f}")


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        sys.exit(1)
    
    # Run bot
    runner = BotRunner()
    
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
        runner.cleanup()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        runner.cleanup()