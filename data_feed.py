"""
data_feed.py - Shark Exchange Data Feed with Order Book
Fetches real-time data via WebSocket with REST API fallback
"""

import numpy as np
import threading
import time
import json
import socketio
import requests
from datetime import datetime
from collections import deque

class DataFeed:
    def __init__(self, use_live=True):
        self.use_live = use_live
        self.current_price = 65000
        self.price_history = deque(maxlen=10000)
        self.volume_history = deque(maxlen=10000)
        self.timestamp_history = deque(maxlen=10000)
        
        # Order book
        self.bids = []
        self.asks = []
        self.spread = 0
        self.order_book_imbalance = 0
        
        # Funding rate
        self.funding_rate = 0.0001
        
        # Connection status
        self.is_connected = False
        self.update_count = 0
        self.last_update_time = None
        
        # WebSocket client
        self.sio = socketio.Client(logger=False, engineio_logger=False)
        self.setup_handlers()
        
        if use_live:
            self.connect_websocket()
        else:
            self.generate_synthetic_data()
    
    def setup_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.sio.event
        def connect():
            print("✅ Connected to Shark Exchange")
            self.is_connected = True
            
            # Subscribe to multiple streams
            subscriptions = [
                "btcusdt@markPrice",
                "btcusdt@ticker",
                "btcusdt@depth20",
                "btcusdt@marketInfo"
            ]
            
            for sub in subscriptions:
                self.sio.emit('subscribe', {"params": [sub]})
                print(f"📡 Subscribed: {sub}")
        
        @self.sio.event
        def disconnect():
            print("🔌 Disconnected from Shark Exchange")
            self.is_connected = False
        
        @self.sio.on('markPriceUpdate')
        def on_mark_price(data):
            try:
                if isinstance(data, list):
                    data = data[0]
                price = float(data.get('p', 0))
                if price > 0:
                    self.update_price(price)
            except Exception:
                pass
        
        @self.sio.on('24hrTicker')
        def on_ticker(data):
            try:
                price = float(data.get('c', 0))
                volume = float(data.get('v', 0))
                if price > 0:
                    self.update_price(price, volume)
                
                # Update funding rate if available
                if 'fr' in data:
                    self.funding_rate = float(data['fr'])
            except Exception:
                pass
        
        @self.sio.on('depthUpdate')
        def on_depth(data):
            try:
                self.bids = [[float(b[0]), float(b[1])] for b in data.get('b', [])[:10]]
                self.asks = [[float(a[0]), float(a[1])] for a in data.get('a', [])[:10]]
                self._calculate_order_book()
            except Exception:
                pass
    
    def _calculate_order_book(self):
        """Calculate order book metrics"""
        if self.bids and self.asks:
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            self.spread = (best_ask - best_bid) / best_bid
            
            total_bid_vol = sum(b[1] for b in self.bids[:5])
            total_ask_vol = sum(a[1] for a in self.asks[:5])
            
            if total_bid_vol + total_ask_vol > 0:
                self.order_book_imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
    
    def connect_websocket(self):
        """Connect to Shark Exchange WebSocket in background thread"""
        print("🌐 Connecting to Shark Exchange...")
        
        def connect_thread():
            try:
                self.sio.connect(
                    'https://fawss.sharkexchange.in',
                    transports=['websocket'],
                    socketio_path='socket.io',
                    headers={
                        'Origin': 'https://sharkexchange.in',
                        'User-Agent': 'Mozilla/5.0'
                    }
                )
                self.sio.wait()
            except Exception as e:
                print(f"❌ WebSocket failed: {e}")
                self.generate_synthetic_data()
        
        thread = threading.Thread(target=connect_thread, daemon=True)
        thread.start()
    
    def fetch_funding_rate(self):
        """Fetch funding rate from REST API"""
        try:
            url = "https://api.sharkexchange.in/api/v1/funding/rate?symbol=BTCUSDT"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.funding_rate = float(data.get('fundingRate', self.funding_rate))
        except Exception:
            pass
    
    def update_price(self, price, volume=None):
        """Update current price and history"""
        if price > 0:
            self.current_price = price
            self.price_history.append(price)
            self.timestamp_history.append(datetime.now())
            self.update_count += 1
            self.last_update_time = datetime.now()
            
            if volume:
                self.volume_history.append(volume)
            else:
                self.volume_history.append(np.random.uniform(500, 2000))
    
    def generate_synthetic_data(self):
        """Generate realistic synthetic data"""
        print("🔧 Generating synthetic data...")
        np.random.seed(42)
        
        n = 10000
        price = 65000
        prices = []
        volumes = []
        
        for i in range(n):
            # Realistic BTC movement
            mu = 0.00005
            sigma = 0.0003
            
            # Regime switching
            if np.random.random() < 0.01:  # 1% chance of regime change
                sigma *= np.random.uniform(0.5, 3.0)
            
            returns = np.random.normal(mu, sigma)
            
            # Mean reversion
            if price > 66000:
                returns -= 0.0002
            elif price < 64000:
                returns += 0.0002
            
            price *= (1 + returns)
            price = max(30000, min(100000, price))
            prices.append(price)
            
            # Volume with bursts
            if np.random.random() < 0.05:  # 5% chance of volume spike
                vol = np.random.uniform(2000, 5000)
            else:
                vol = np.random.uniform(500, 1500)
            volumes.append(vol)
        
        self.price_history = deque(prices, maxlen=10000)
        self.volume_history = deque(volumes, maxlen=10000)
        self.current_price = prices[-1]
        
        # Generate synthetic order book
        self.bids = [[self.current_price * (1 - i*0.0001), np.random.uniform(1, 10)] for i in range(10)]
        self.asks = [[self.current_price * (1 + i*0.0001), np.random.uniform(1, 10)] for i in range(10)]
        self._calculate_order_book()
        
        # Periodically update funding rate
        def update_funding():
            while True:
                self.funding_rate = np.random.uniform(-0.001, 0.001)
                time.sleep(3600)  # Every hour
        
        thread = threading.Thread(target=update_funding, daemon=True)
        thread.start()
    
    def get_current_price(self):
        """Get latest price"""
        return self.current_price
    
    def get_order_book(self):
        """Get current order book data"""
        return {
            'bids': self.bids[:5] if self.bids else [],
            'asks': self.asks[:5] if self.asks else [],
            'spread': self.spread,
            'imbalance': self.order_book_imbalance
        }
    
    def get_price_history(self, n=100):
        """Get last n prices"""
        prices = list(self.price_history)
        return prices[-n:] if len(prices) >= n else prices
    
    def get_volume_history(self, n=100):
        """Get last n volumes"""
        volumes = list(self.volume_history)
        return volumes[-n:] if len(volumes) >= n else volumes
    
    def get_connection_status(self):
        """Get connection status string"""
        if self.is_connected:
            return f"🟢 Live | ${self.current_price:,.2f} | Updates: {self.update_count}"
        return f"🔴 Synthetic | ${self.current_price:,.2f}"
    
    def restart_connection(self):
        """Restart WebSocket connection"""
        try:
            self.sio.disconnect()
        except:
            pass
        time.sleep(2)
        self.connect_websocket()