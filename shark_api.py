"""
shark_api.py - Shark Exchange API Integration
Uses your API keys for authenticated endpoints
"""

import os
import hmac
import hashlib
import time
import json
import requests
from config import *

class SharkExchangeAPI:
    """Shark Exchange REST API Client"""
    
    def __init__(self, api_key=None, secret_key=None):
        self.api_key = api_key or SHARK_API_KEY
        self.secret_key = secret_key or SHARK_SECRET_KEY
        self.base_url = "https://api.sharkexchange.in"
        
        if not self.api_key or not self.secret_key:
            print("⚠️ API keys not found! Using public endpoints only.")
            print("   Set SHARK_API_KEY and SHARK_SECRET_KEY in config.py")
    
    def _generate_signature(self, params):
        """Generate HMAC SHA256 signature"""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method, endpoint, params=None, signed=False):
        """Make API request"""
        url = f"{self.base_url}{endpoint}"
        headers = {
            'X-API-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        if signed and self.secret_key:
            params = params or {}
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"❌ API Request failed: {e}")
            return None
    
    def get_account_balance(self):
        """Get account balance"""
        if not self.api_key:
            return {'USDT': {'available': 18.00, 'locked': 0}}
        
        result = self._request('GET', '/api/v1/account/balances', signed=True)
        if result:
            return result.get('balances', {})
        return None
    
    def get_positions(self):
        """Get current positions"""
        if not self.api_key:
            return []
        
        result = self._request('GET', '/api/v1/positions', signed=True)
        if result:
            return result.get('positions', [])
        return []
    
    def place_order(self, symbol, side, order_type, quantity, price=None, leverage=80):
        """Place an order"""
        if not self.api_key:
            print("⚠️ Cannot place order without API keys")
            return None
        
        params = {
            'symbol': symbol,
            'side': side.upper(),  # BUY or SELL
            'type': order_type.upper(),  # MARKET or LIMIT
            'quantity': quantity,
            'leverage': leverage
        }
        
        if price and order_type.upper() == 'LIMIT':
            params['price'] = price
        
        result = self._request('POST', '/api/v1/order', params=params, signed=True)
        if result:
            print(f"✅ Order placed: {result}")
        return result
    
    def cancel_order(self, order_id, symbol='BTCUSDT'):
        """Cancel an order"""
        if not self.api_key:
            return None
        
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        result = self._request('DELETE', '/api/v1/order', params=params, signed=True)
        return result
    
    def get_order_status(self, order_id, symbol='BTCUSDT'):
        """Get order status"""
        if not self.api_key:
            return None
        
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        result = self._request('GET', '/api/v1/order', params=params, signed=True)
        return result
    
    def set_leverage(self, symbol='BTCUSDT', leverage=80):
        """Set leverage for symbol"""
        if not self.api_key:
            return True  # Paper trading
        
        params = {
            'symbol': symbol,
            'leverage': leverage
        }
        
        result = self._request('POST', '/api/v1/leverage', params=params, signed=True)
        if result:
            print(f"✅ Leverage set to {leverage}x for {symbol}")
        return result
    
    def get_ticker(self, symbol='BTCUSDT'):
        """Get ticker price"""
        params = {'symbol': symbol}
        result = self._request('GET', '/api/v1/ticker', params=params)
        if result:
            return float(result.get('lastPrice', 0))
        return None
    
    def get_order_book(self, symbol='BTCUSDT', limit=10):
        """Get order book"""
        params = {'symbol': symbol, 'limit': limit}
        result = self._request('GET', '/api/v1/depth', params=params)
        if result:
            return {
                'bids': result.get('bids', []),
                'asks': result.get('asks', [])
            }
        return None
    
    def get_funding_rate(self, symbol='BTCUSDT'):
        """Get current funding rate"""
        params = {'symbol': symbol}
        result = self._request('GET', '/api/v1/funding/rate', params=params)
        if result:
            return float(result.get('fundingRate', 0.0001))
        return 0.0001