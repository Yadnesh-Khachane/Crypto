"""
strategies.py - Professional crypto trading strategies
Used by major BTC traders. All signal generators for RL state.
"""

import numpy as np
import pandas as pd
from collections import deque
from scipy import stats

class StrategyEngine:
    """Engine that generates trading signals from multiple strategies"""
    
    def __init__(self, max_history=500):
        self.max_history = max_history
        
        # Price data
        self.close_prices = deque(maxlen=max_history)
        self.high_prices = deque(maxlen=max_history)
        self.low_prices = deque(maxlen=max_history)
        self.open_prices = deque(maxlen=max_history)
        self.volumes = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        
        # Order book
        self.bid_volumes = deque(maxlen=100)
        self.ask_volumes = deque(maxlen=100)
        self.bid_prices = deque(maxlen=100)
        self.ask_prices = deque(maxlen=100)
        
        # Computed indicators
        self.vwap = 0
        self.atr = 0
        self.adx = 0
        self.spread = 0
        
    def update(self, price, high=None, low=None, open_price=None, 
               volume=None, timestamp=None, bid_data=None, ask_data=None):
        """Update with new candle/tick data"""
        self.close_prices.append(price)
        self.high_prices.append(high if high else price)
        self.low_prices.append(low if low else price)
        self.open_prices.append(open_price if open_price else price)
        self.volumes.append(volume if volume else 1000)
        self.timestamps.append(timestamp)
        
        # Update order book
        if bid_data:
            self.bid_volumes.append(bid_data.get('volume', 0))
            self.bid_prices.append(bid_data.get('price', price))
        if ask_data:
            self.ask_volumes.append(ask_data.get('volume', 0))
            self.ask_prices.append(ask_data.get('price', price))
        
        # Compute indicators
        self._compute_indicators()
    
    def _compute_indicators(self):
        """Compute all technical indicators"""
        if len(self.close_prices) < 20:
            return
        
        prices = np.array(self.close_prices)
        highs = np.array(self.high_prices)
        lows = np.array(self.low_prices)
        volumes = np.array(self.volumes)
        
        # VWAP
        if len(volumes) > 0:
            self.vwap = np.sum(prices * volumes) / np.sum(volumes)
        
        # ATR (Average True Range)
        if len(highs) >= 14:
            tr = np.maximum(
                highs[-14:] - lows[-14:],
                np.abs(highs[-14:] - np.roll(prices[-14:], 1)),
                np.abs(lows[-14:] - np.roll(prices[-14:], 1))
            )
            self.atr = np.mean(tr)
        
        # ADX (Trend Strength)
        if len(highs) >= 28:
            self.adx = self._calculate_adx(highs[-28:], lows[-28:], prices[-28:])
        
        # Spread
        if len(self.bid_prices) > 0 and len(self.ask_prices) > 0:
            self.spread = (self.ask_prices[-1] - self.bid_prices[-1]) / self.bid_prices[-1]
    
    def _calculate_adx(self, high, low, close, period=14):
        """Calculate Average Directional Index - SAFE VERSION"""
        try:
            if len(high) < period + 1:
                return 0
            
            plus_dm = np.zeros(len(high))
            minus_dm = np.zeros(len(high))
            tr = np.zeros(len(high))
            
            for i in range(1, len(high)):
                tr[i] = max(
                    high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1])
                )
                
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
            
            # Smooth with Wilder's method
            atr_val = np.mean(tr[-period:])
            if atr_val == 0 or np.isnan(atr_val):
                return 0
            
            plus_di = 100 * np.mean(plus_dm[-period:]) / atr_val
            minus_di = 100 * np.mean(minus_dm[-period:]) / atr_val
            
            if np.isnan(plus_di) or np.isnan(minus_di):
                return 0
            
            di_sum = plus_di + minus_di
            if di_sum == 0:
                return 0
            
            dx = 100 * abs(plus_di - minus_di) / di_sum
            
            if np.isnan(dx):
                return 0
            
            return float(dx)
        except Exception as e:
            return 0
    
    # ============================================================
    # STRATEGY 1: EMA CROSSOVER WITH VOLUME CONFIRMATION
    # ============================================================
    def ema_crossover_signal(self):
        """EMA 9/21/50 crossover with volume confirmation"""
        if len(self.close_prices) < 50:
            return {'signal': 0, 'confidence': 0, 'direction': 'neutral'}
        
        prices = pd.Series(self.close_prices)
        ema_9 = prices.ewm(span=9).mean().iloc[-1]
        ema_21 = prices.ewm(span=21).mean().iloc[-1]
        ema_50 = prices.ewm(span=50).mean().iloc[-1]
        
        # Trend alignment
        bullish_alignment = ema_9 > ema_21 > ema_50
        bearish_alignment = ema_9 < ema_21 < ema_50
        
        # Volume confirmation
        vol_sma = pd.Series(self.volumes).rolling(20).mean().iloc[-1]
        vol_surge = self.volumes[-1] / vol_sma if vol_sma > 0 else 1
        
        confidence = 0
        direction = 'neutral'
        
        if bullish_alignment and vol_surge > 1.2:
            # Price pullback to EMA-9 (good entry)
            pullback = (prices.iloc[-1] - ema_9) / ema_9
            if -0.005 < pullback < 0.002:
                confidence = 0.7
                direction = 'long'
        elif bearish_alignment and vol_surge > 1.2:
            pullback = (prices.iloc[-1] - ema_9) / ema_9
            if -0.002 < pullback < 0.005:
                confidence = 0.7
                direction = 'short'
        
        return {
            'signal': 1 if confidence > 0 else 0,
            'confidence': confidence,
            'direction': direction,
            'ema_9': ema_9,
            'ema_21': ema_21,
            'ema_50': ema_50,
            'vol_surge': vol_surge
        }
    
    # ============================================================
    # STRATEGY 2: RSI DIVERGENCE DETECTION
    # ============================================================
    def rsi_divergence_signal(self, period=14):
        """RSI divergence with price action"""
        if len(self.close_prices) < period + 10:
            return {'signal': 0, 'confidence': 0}
        
        prices = np.array(self.close_prices)
        rsi = self._calculate_rsi(prices, period)
        current_rsi = rsi[-1]
        
        confidence = 0
        direction = 'neutral'
        
        # Oversold bounce (bullish)
        if current_rsi < 30:
            # Check for bullish divergence
            price_low_1 = min(prices[-10:-5])
            price_low_2 = min(prices[-5:])
            rsi_low_1 = min(rsi[-10:-5])
            rsi_low_2 = min(rsi[-5:])
            
            if price_low_2 < price_low_1 and rsi_low_2 > rsi_low_1:
                confidence = 0.8
                direction = 'long'
            elif current_rsi < 25:
                confidence = 0.6
                direction = 'long'
        
        # Overbought rejection (bearish)
        elif current_rsi > 70:
            price_high_1 = max(prices[-10:-5])
            price_high_2 = max(prices[-5:])
            rsi_high_1 = max(rsi[-10:-5])
            rsi_high_2 = max(rsi[-5:])
            
            if price_high_2 > price_high_1 and rsi_high_2 < rsi_high_1:
                confidence = 0.8
                direction = 'short'
            elif current_rsi > 75:
                confidence = 0.6
                direction = 'short'
        
        return {
            'signal': 1 if confidence > 0 else 0,
            'confidence': confidence,
            'direction': direction,
            'rsi': current_rsi
        }
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return np.ones(len(prices)) * 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Create full RSI array
        rsi_full = np.zeros(len(prices))
        rsi_full[-1] = rsi
        
        for i in range(len(prices) - 2, -1, -1):
            if i >= period:
                gain = gains[i]
                loss = losses[i]
                avg_gain = (avg_gain * (period - 1) + gain) / period
                avg_loss = (avg_loss * (period - 1) + loss) / period
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi_full[i] = 100 - (100 / (1 + rs))
        
        return rsi_full
    
    # ============================================================
    # STRATEGY 3: BOLLINGER BAND SQUEEZE
    # ============================================================
    def bollinger_squeeze_signal(self, period=20, std_dev=2.0):
        """Bollinger Band squeeze breakout"""
        if len(self.close_prices) < period + 10:
            return {'signal': 0, 'confidence': 0}
        
        prices = np.array(self.close_prices)
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + std_dev * std
        lower_band = sma - std_dev * std
        bandwidth = (upper_band - lower_band) / sma
        
        # Check for squeeze (low bandwidth historically)
        historical_bandwidths = []
        for i in range(10):
            hist_prices = prices[-period-i:-i] if i > 0 else prices[-period:]
            hist_sma = np.mean(hist_prices)
            hist_std = np.std(hist_prices)
            hist_bw = (2 * std_dev * hist_std) / hist_sma
            historical_bandwidths.append(hist_bw)
        
        avg_bandwidth = np.mean(historical_bandwidths)
        
        confidence = 0
        direction = 'neutral'
        
        # Squeeze detected (low volatility, expect breakout)
        if bandwidth < avg_bandwidth * 0.8:
            # Determine breakout direction
            momentum = np.mean(np.diff(prices[-5:]))
            
            if momentum > 0 and prices[-1] > sma:
                confidence = 0.65
                direction = 'long'
            elif momentum < 0 and prices[-1] < sma:
                confidence = 0.65
                direction = 'short'
        
        return {
            'signal': 1 if confidence > 0 else 0,
            'confidence': confidence,
            'direction': direction,
            'bandwidth': bandwidth,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'sma': sma
        }
    
    # ============================================================
    # STRATEGY 4: VOLUME PROFILE & MARKET PROFILE
    # ============================================================
    def volume_profile_signal(self):
        """Volume-weighted price levels"""
        if len(self.close_prices) < 50:
            return {'signal': 0, 'confidence': 0}
        
        prices = np.array(self.close_prices)
        volumes = np.array(self.volumes)
        
        # Find high volume nodes (price levels with most volume)
        price_bins = np.linspace(min(prices[-50:]), max(prices[-50:]), 20)
        volume_profile = np.zeros(len(price_bins) - 1)
        
        for i in range(len(prices[-50:]) - 1):
            for j in range(len(price_bins) - 1):
                if price_bins[j] <= prices[i] < price_bins[j+1]:
                    volume_profile[j] += volumes[i]
        
        # POC (Point of Control) - price with highest volume
        poc_idx = np.argmax(volume_profile)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
        
        # Value Area (70% of volume)
        total_vol = np.sum(volume_profile)
        cumulative = 0
        value_area_low = poc_price
        value_area_high = poc_price
        
        for i in range(poc_idx, -1, -1):
            cumulative += volume_profile[i]
            if cumulative >= total_vol * 0.35:
                value_area_low = price_bins[i]
                break
        
        cumulative = 0
        for i in range(poc_idx, len(volume_profile)):
            cumulative += volume_profile[i]
            if cumulative >= total_vol * 0.35:
                value_area_high = price_bins[i+1]
                break
        
        current_price = prices[-1]
        confidence = 0
        direction = 'neutral'
        
        # Price below value area (buy opportunity)
        if current_price < value_area_low:
            confidence = 0.7
            direction = 'long'
        # Price above value area (sell opportunity)
        elif current_price > value_area_high:
            confidence = 0.7
            direction = 'short'
        # Price at POC in value area
        elif abs(current_price - poc_price) / poc_price < 0.002:
            confidence = 0.5
            direction = 'long' if current_price > poc_price else 'short'
        
        return {
            'signal': 1 if confidence > 0 else 0,
            'confidence': confidence,
            'direction': direction,
            'poc_price': poc_price,
            'value_area_low': value_area_low,
            'value_area_high': value_area_high
        }
    
    # ============================================================
    # STRATEGY 5: ORDER FLOW IMBALANCE
    # ============================================================
    def order_flow_signal(self):
        """Order book imbalance analysis"""
        if len(self.bid_volumes) < 10 or len(self.ask_volumes) < 10:
            return {'signal': 0, 'confidence': 0}
        
        total_bid_vol = np.sum(self.bid_volumes)
        total_ask_vol = np.sum(self.ask_volumes)
        
        if total_bid_vol + total_ask_vol == 0:
            return {'signal': 0, 'confidence': 0}
        
        imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        
        confidence = 0
        direction = 'neutral'
        
        # Strong buying pressure
        if imbalance > 0.3:
            confidence = min(0.8, abs(imbalance) * 1.5)
            direction = 'long'
        # Strong selling pressure
        elif imbalance < -0.3:
            confidence = min(0.8, abs(imbalance) * 1.5)
            direction = 'short'
        
        return {
            'signal': 1 if confidence > 0 else 0,
            'confidence': confidence,
            'direction': direction,
            'imbalance': imbalance
        }
    
    # ============================================================
    # STRATEGY 6: SUPPORT & RESISTANCE
    # ============================================================
    def support_resistance_signal(self):
        """Key level identification"""
        if len(self.close_prices) < 100:
            return {'signal': 0, 'confidence': 0}
        
        prices = np.array(self.close_prices)
        highs = np.array(self.high_prices)
        lows = np.array(self.low_prices)
        
        # Find swing highs and lows
        resistance_levels = []
        support_levels = []
        
        for i in range(5, len(prices) - 5):
            # Swing high
            if highs[i] == max(highs[i-5:i+6]):
                resistance_levels.append(highs[i])
            # Swing low
            if lows[i] == min(lows[i-5:i+6]):
                support_levels.append(lows[i])
        
        if not resistance_levels or not support_levels:
            return {'signal': 0, 'confidence': 0}
        
        # Cluster nearby levels
        resistance_clusters = self._cluster_levels(resistance_levels)
        support_clusters = self._cluster_levels(support_levels)
        
        current_price = prices[-1]
        confidence = 0
        direction = 'neutral'
        nearest_support = None
        nearest_resistance = None
        
        # Find nearest levels
        for level in support_clusters:
            if level < current_price:
                if nearest_support is None or level > nearest_support:
                    nearest_support = level
        
        for level in resistance_clusters:
            if level > current_price:
                if nearest_resistance is None or level < nearest_resistance:
                    nearest_resistance = level
        
        # Bounce off support
        if nearest_support:
            distance_to_support = (current_price - nearest_support) / current_price
            if 0 < distance_to_support < 0.01:  # Within 1% of support
                # Check for bullish reversal patterns
                if prices[-1] > prices[-2] and prices[-2] < prices[-3]:
                    confidence = 0.75
                    direction = 'long'
        
        # Rejection at resistance
        if nearest_resistance:
            distance_to_resistance = (nearest_resistance - current_price) / current_price
            if 0 < distance_to_resistance < 0.01:
                if prices[-1] < prices[-2] and prices[-2] > prices[-3]:
                    confidence = 0.75
                    direction = 'short'
        
        return {
            'signal': 1 if confidence > 0 else 0,
            'confidence': confidence,
            'direction': direction,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }
    
    def _cluster_levels(self, levels, threshold=0.002):
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        sorted_levels = sorted(levels)
        clusters = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if (level - clusters[-1]) / clusters[-1] > threshold:
                clusters.append(level)
        
        return clusters
    
    # ============================================================
    # STRATEGY 7: FUNDING RATE ARBITRAGE
    # ============================================================
    def funding_rate_signal(self, funding_rate=None):
        """Trade based on funding rate extremes"""
        if funding_rate is None:
            funding_rate = 0.0001  # Default 0.01%
        
        confidence = 0
        direction = 'neutral'
        
        # Extremely negative funding (shorts paying longs)
        if funding_rate < -0.001:  # -0.1%
            confidence = 0.6
            direction = 'long'
        # Extremely positive funding (longs paying shorts)
        elif funding_rate > 0.001:  # 0.1%
            confidence = 0.6
            direction = 'short'
        
        return {
            'signal': 1 if confidence > 0 else 0,
            'confidence': confidence,
            'direction': direction,
            'funding_rate': funding_rate
        }
    
    # ============================================================
    # STRATEGY 8: ICHIMOKU CLOUD
    # ============================================================
    def ichimoku_signal(self):
        """Ichimoku Cloud analysis"""
        if len(self.close_prices) < 52:
            return {'signal': 0, 'confidence': 0}
        
        prices = np.array(self.close_prices)
        highs = np.array(self.high_prices)
        lows = np.array(self.low_prices)
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan = (max(highs[-9:]) + min(lows[-9:])) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun = (max(highs[-26:]) + min(lows[-26:])) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 (plotted 26 periods ahead)
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        senkou_b = (max(highs[-52:]) + min(lows[-52:])) / 2
        
        current_price = prices[-1]
        confidence = 0
        direction = 'neutral'
        
        # Price above cloud (bullish)
        if current_price > max(senkou_a, senkou_b) and tenkan > kijun:
            confidence = 0.7
            direction = 'long'
        # Price below cloud (bearish)
        elif current_price < min(senkou_a, senkou_b) and tenkan < kijun:
            confidence = 0.7
            direction = 'short'
        # TK cross inside cloud
        elif abs(tenkan - kijun) / kijun < 0.001:
            if current_price > senkou_a:
                confidence = 0.5
                direction = 'long'
            else:
                confidence = 0.5
                direction = 'short'
        
        return {
            'signal': 1 if confidence > 0 else 0,
            'confidence': confidence,
            'direction': direction,
            'tenkan': tenkan,
            'kijun': kijun,
            'cloud_top': max(senkou_a, senkou_b),
            'cloud_bottom': min(senkou_a, senkou_b)
        }
    
    # ============================================================
    # STRATEGY 9: VWAP DEVIATION
    # ============================================================
    def vwap_deviation_signal(self):
        """Trade VWAP mean reversion"""
        if len(self.close_prices) < 50:
            return {'signal': 0, 'confidence': 0}
        
        current_price = self.close_prices[-1]
        deviation = (current_price - self.vwap) / self.vwap
        
        confidence = 0
        direction = 'neutral'
        
        # Significant deviation from VWAP
        if deviation < -0.02:  # 2% below VWAP
            confidence = 0.7
            direction = 'long'
        elif deviation > 0.02:  # 2% above VWAP
            confidence = 0.7
            direction = 'short'
        elif abs(deviation) < 0.002:  # At VWAP
            # Check momentum for direction
            momentum = np.mean(np.diff(list(self.close_prices)[-5:]))
            if momentum > 0:
                confidence = 0.5
                direction = 'long'
            else:
                confidence = 0.5
                direction = 'short'
        
        return {
            'signal': 1 if confidence > 0 else 0,
            'confidence': confidence,
            'direction': direction,
            'vwap': self.vwap,
            'deviation': deviation
        }
    
    # ============================================================
    # STRATEGY 10: MARKET STRUCTURE (HH/HL, LH/LL)
    # ============================================================
    def market_structure_signal(self):
        """Higher Highs/Lows or Lower Highs/Lows"""
        if len(self.close_prices) < 30:
            return {'signal': 0, 'confidence': 0}
        
        prices = np.array(self.close_prices)
        
        # Find recent swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(3, len(prices) - 3):
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
               prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                swing_highs.append(prices[i])
            if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
               prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                swing_lows.append(prices[i])
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'signal': 0, 'confidence': 0}
        
        # Check structure
        last_two_highs = swing_highs[-2:]
        last_two_lows = swing_lows[-2:]
        
        confidence = 0
        direction = 'neutral'
        
        # Bullish structure (HH, HL)
        if last_two_highs[-1] > last_two_highs[-2] and last_two_lows[-1] > last_two_lows[-2]:
            confidence = 0.8
            direction = 'long'
        # Bearish structure (LH, LL)
        elif last_two_highs[-1] < last_two_highs[-2] and last_two_lows[-1] < last_two_lows[-2]:
            confidence = 0.8
            direction = 'short'
        
        return {
            'signal': 1 if confidence > 0 else 0,
            'confidence': confidence,
            'direction': direction
        }
    
    # ============================================================
    # COMBINE ALL SIGNALS
    # ============================================================
    def get_combined_signal(self, funding_rate=None, order_book=None):
        """Get weighted signal from all strategies - SAFE VERSION"""
        try:
            if funding_rate is None:
                funding_rate = 0.0001
            if order_book is None:
                order_book = {}
            
            signals = {}
            
            # Try each strategy individually, use empty dict on failure
            for name, func in [
                ('ema_crossover', self.ema_crossover_signal),
                ('rsi_divergence', self.rsi_divergence_signal),
                ('bollinger_squeeze', self.bollinger_squeeze_signal),
                ('volume_profile', self.volume_profile_signal),
                ('order_flow', self.order_flow_signal),
                ('support_resistance', self.support_resistance_signal),
                ('funding_rate', lambda: self.funding_rate_signal(funding_rate)),
                ('ichimoku', self.ichimoku_signal),
                ('vwap_deviation', self.vwap_deviation_signal),
                ('market_structure', self.market_structure_signal)
            ]:
                try:
                    signals[name] = func()
                except Exception:
                    signals[name] = {'signal': 0, 'confidence': 0, 'direction': 'neutral'}
            
            # Count active signals per direction
            long_signals = []
            short_signals = []
            total_confidence = 0
            
            for name, signal in signals.items():
                conf = signal.get('confidence', 0)
                if conf is None or np.isnan(conf):
                    conf = 0
                
                if conf > 0:
                    direction = signal.get('direction', 'neutral')
                    if direction == 'long':
                        long_signals.append(conf)
                    elif direction == 'short':
                        short_signals.append(conf)
                    total_confidence += conf
            
            long_score = len(long_signals)
            short_score = len(short_signals)
            long_confidence = np.mean(long_signals) if long_signals else 0
            short_confidence = np.mean(short_signals) if short_signals else 0
            
            # Get ADX safely
            adx_val = self.adx if hasattr(self, 'adx') and self.adx is not None and not np.isnan(self.adx) else 0
            atr_val = self.atr if hasattr(self, 'atr') and self.atr is not None and not np.isnan(self.atr) else 0
            spread_val = self.spread if hasattr(self, 'spread') and self.spread is not None and not np.isnan(self.spread) else 0
            vwap_val = self.vwap if hasattr(self, 'vwap') and self.vwap is not None and not np.isnan(self.vwap) else 0
            
            # Determine final signal
            result = {
                'all_signals': signals,
                'long_count': long_score,
                'short_count': short_score,
                'long_confidence': float(long_confidence),
                'short_confidence': float(short_confidence),
                'total_confidence': float(total_confidence),
                'atr': float(atr_val),
                'adx': float(adx_val),
                'spread': float(spread_val),
                'vwap': float(vwap_val)
            }
            
            if long_score >= 3 and long_confidence > 0.5:
                result['signal'] = 1
                result['confidence'] = float(long_confidence)
                result['direction'] = 'long'
            elif short_score >= 3 and short_confidence > 0.5:
                result['signal'] = 2
                result['confidence'] = float(short_confidence)
                result['direction'] = 'short'
            else:
                result['signal'] = 0
                result['confidence'] = 0.0
                result['direction'] = 'neutral'
            
            return result
            
        except Exception as e:
            # Return safe defaults on any error
            return {
                'signal': 0,
                'confidence': 0.0,
                'direction': 'neutral',
                'long_count': 0,
                'short_count': 0,
                'long_confidence': 0.0,
                'short_confidence': 0.0,
                'total_confidence': 0.0,
                'atr': 0.0,
                'adx': 0.0,
                'spread': 0.0,
                'vwap': 0.0,
                'all_signals': {}
            }