"""
trader.py - Smart RL Trader with Percentage-Based Auto Exit
FINAL SETTINGS: 40% TP | 10% SL | 30x Leverage | 1:4 Risk/Reward
IMPROVED REWARD: Scaled profits + HOLD rewards + Patience bonus
"""

import numpy as np
from collections import deque
from datetime import datetime, timedelta
import time
from config import *
from strategies import StrategyEngine

class SmartTrader:
    def __init__(self, initial_capital=INITIAL_CAPITAL_USD, leverage=LEVERAGE, use_rl=True):
        # ===== CAPITAL MANAGEMENT =====
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.margin_used = 0
        self.total_equity = initial_capital
        self.equity_history = deque(maxlen=10000)
        self.peak_equity = initial_capital
        
        # ===== POSITION PARAMETERS =====
        self.leverage = leverage
        self.base_btc_size = BASE_BTC_SIZE
        self.position_value = POSITION_VALUE_USD
        self.margin_per_trade = self.position_value / self.leverage
        
        # ===== 🎯 PERCENTAGE-BASED EXITS (FINAL) =====
        self.tp_pnl_percent = TAKE_PROFIT_PNL_PERCENT    # 40% P&L on margin
        self.sl_pnl_percent = STOP_LOSS_PNL_PERCENT      # 10% P&L on margin
        self.trailing_activate_percent = TRAILING_ACTIVATE_PERCENT  # 25% P&L
        self.trailing_distance_percent = TRAILING_DISTANCE_PERCENT  # 8%
        
        # Corresponding price moves
        self.tp_price_pct = TAKE_PROFIT_PRICE_PCT        # 1.33% at 30x
        self.sl_price_pct = STOP_LOSS_PRICE_PCT          # 0.33% at 30x
        
        # ===== POSITIONS =====
        self.positions = []
        self.closed_trades = []
        self.max_positions = MAX_POSITIONS
        
        # ===== FEES =====
        self.taker_fee_rate = TAKER_FEE * (1 + GST_RATE)
        self.maker_fee_rate = MAKER_FEE * (1 + GST_RATE)
        self.total_fees_paid = 0
        
        # ===== ENTRY FILTERS =====
        self.min_confidence = MIN_CONFIDENCE_SCORE
        self.min_signals = MIN_SIGNAL_STRENGTH
        self.cooldown_seconds = COOLDOWN_SECONDS
        self.last_trade_time = None
        
        # ===== WARMUP PERIOD =====
        self.startup_delay = 30
        self.warmup_delay = 60
        self.start_time = datetime.now()
        self.is_warmed_up = False
        self.initial_warmup_complete = False
        
        # ===== PERFORMANCE TRACKING =====
        self.wins = 0
        self.losses = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        
        # ===== DAILY LIMITS =====
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 20
        self.daily_loss_limit = -3.00
        self.trading_day = datetime.now().date()
        
        # ===== STRATEGY ENGINE =====
        self.strategy_engine = StrategyEngine()
        
        # ===== PRICE DATA =====
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.current_price = 65000.0
        self.last_volume = 1000.0
        
        # ===== RL AGENT =====
        self.use_rl = use_rl
        if use_rl:
            from rl_agent import DQNAgent
            self.rl_agent = DQNAgent(
                state_size=RL_STATE_SIZE,
                action_size=RL_ACTION_SIZE,
                hidden_layers=[128, 64],
                learning_rate=RL_LEARNING_RATE,
                gamma=RL_GAMMA
            )
        
        # ===== TRAINING STATE =====
        self.last_state = None
        self.last_action = None
        self.step_count = 0
        self.training_interval = RL_TRAINING_INTERVAL
        
        # ===== TRADE LOG =====
        self.trade_log = deque(maxlen=200)
        
        # ===== CONSECUTIVE LOSSES =====
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        
        # ===== 🆕 IMPROVED REWARD TRACKING =====
        self.hold_count = 0           # Count consecutive HOLD actions
        self.last_hold_reward_time = datetime.now()  # Track when we last gave HOLD reward
        self.avoided_bad_trades = 0   # Count times we avoided a bad trade
        self.total_rewards_given = 0  # Track total rewards
        
        print("=" * 60)
        print("🤖 SMART RL TRADER INITIALIZED")
        print("=" * 60)
        print(f"💰 Capital: ${self.capital:.4f} (₹{self.capital*83:.0f})")
        print(f"📊 Leverage: {self.leverage}x")
        print(f"📈 Position: {self.base_btc_size} BTC")
        print(f"💵 Margin/Trade: ${self.margin_per_trade:.4f}")
        print(f"🎯 TP: {self.tp_pnl_percent}% P&L ({self.tp_price_pct*100:.2f}% price move)")
        print(f"🛑 SL: {self.sl_pnl_percent}% P&L ({self.sl_price_pct*100:.2f}% price move)")
        print(f"📊 R:R Ratio: {self.tp_pnl_percent/self.sl_pnl_percent:.1f}:1")
        print(f"🔒 Trailing: Activates at {self.trailing_activate_percent}% P&L")
        print(f"🧠 Reward: Scaled + HOLD bonus + Patience bonus")
        if use_rl:
            print(f"🧠 RL Epsilon: {self.rl_agent.epsilon:.4f}")
        print("=" * 60)
    
    def calculate_position_metrics(self, entry_price, direction='long'):
        """Calculate all position metrics including fees"""
        if entry_price is None or entry_price <= 0:
            entry_price = 65000.0
        
        btc_size = self.base_btc_size
        position_value = btc_size * entry_price
        margin_required = position_value / self.leverage
        entry_fee = position_value * self.taker_fee_rate
        
        if direction == 'long':
            take_profit_price = entry_price * (1 + self.tp_price_pct)
            stop_loss_price = entry_price * (1 - self.sl_price_pct)
            liquidation_price = entry_price * (1 - 0.95/self.leverage)
            gross_profit_at_tp = (take_profit_price - entry_price) * btc_size
        else:
            take_profit_price = entry_price * (1 - self.tp_price_pct)
            stop_loss_price = entry_price * (1 + self.sl_price_pct)
            liquidation_price = entry_price * (1 + 0.95/self.leverage)
            gross_profit_at_tp = (entry_price - take_profit_price) * btc_size
        
        exit_fee_at_tp = (btc_size * take_profit_price) * self.taker_fee_rate
        total_fees = entry_fee + exit_fee_at_tp
        net_profit = gross_profit_at_tp - total_fees
        
        return {
            'btc_size': btc_size,
            'position_value': position_value,
            'margin_required': margin_required,
            'entry_fee': entry_fee,
            'take_profit_price': take_profit_price,
            'stop_loss_price': stop_loss_price,
            'liquidation_price': liquidation_price,
            'net_profit_estimate': net_profit,
            'total_fees_estimate': total_fees
        }
    
    def get_state_features(self, price, funding_rate=None, order_book=None):
        """Extract features for RL state"""
        if price is None or price <= 0:
            price = 65000.0
        if funding_rate is None:
            funding_rate = 0.0001
        if order_book is None:
            order_book = {}
        
        volume = 1000.0
        if len(self.volume_history) > 0:
            volume = self.volume_history[-1]
        self.last_volume = volume
        
        try:
            self.strategy_engine.update(price=price, volume=volume)
        except:
            pass
        
        try:
            signals = self.strategy_engine.get_combined_signal(funding_rate, order_book)
        except:
            signals = {
                'long_count': 0, 'short_count': 0,
                'long_confidence': 0.0, 'short_confidence': 0.0,
                'total_confidence': 0.0, 'direction': 'neutral',
                'adx': 0.0, 'atr': 0.0, 'spread': 0.0
            }
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.current_price = price
        
        prices = list(self.price_history)
        
        if len(prices) < 20:
            return np.zeros(RL_STATE_SIZE, dtype=np.float32)
        
        try:
            returns_1m = (prices[-1] - prices[-2]) / max(prices[-2], 1) if len(prices) >= 2 else 0.0
            returns_5m = (prices[-1] - prices[-5]) / max(prices[-5], 1) if len(prices) >= 5 else 0.0
            returns_15m = (prices[-1] - prices[-15]) / max(prices[-15], 1) if len(prices) >= 15 else 0.0
        except:
            returns_1m = returns_5m = returns_15m = 0.0
        
        try:
            if len(prices) >= 21:
                rets = [(prices[i] - prices[i-1]) / max(prices[i-1], 1) for i in range(-20, 0)]
                volatility = np.std(rets) if len(rets) > 0 else 0.0
            else:
                volatility = 0.0
        except:
            volatility = 0.0
        
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else price
        
        open_positions = len([p for p in self.positions if p.get('is_open', False)])
        position_utilization = open_positions / max(self.max_positions, 1)
        
        unrealized_pnl = sum(p.get('pnl', 0) for p in self.positions if p.get('is_open', False))
        
        capital_ratio = self.capital / max(self.initial_capital, 0.01)
        win_rate = self.wins / max(1, self.total_trades)
        
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds() / 3600.0
        else:
            time_since_last = 24.0
        
        loss_streak = self.consecutive_losses / max(self.max_consecutive_losses, 1)
        daily_trade_ratio = self.daily_trades / max(self.max_daily_trades, 1)
        
        features = np.array([
            np.clip(float(returns_1m) * 100, -10, 10),
            np.clip(float(returns_5m) * 100, -10, 10),
            np.clip(float(returns_15m) * 100, -10, 10),
            np.clip(float(volatility) * 1000, -10, 10),
            np.clip((price - sma_20) / max(sma_20, 1) * 100, -10, 10),
            float(signals.get('long_count', 0)) / 10.0,
            float(signals.get('short_count', 0)) / 10.0,
            float(signals.get('long_confidence', 0)),
            float(signals.get('short_confidence', 0)),
            float(signals.get('adx', 0)) / 100.0,
            float(position_utilization),
            float(unrealized_pnl),
            float(capital_ratio),
            float(win_rate),
            float(self.total_pnl) * 10.0,
            float(time_since_last) / 24.0,
            self.margin_used / max(self.initial_capital, 0.01),
            1.0 if funding_rate > 0.001 else -1.0 if funding_rate < -0.001 else 0.0,
            abs(funding_rate) * 10000.0,
            float(loss_streak),
            float(daily_trade_ratio),
            float(self.daily_pnl),
            float(self.total_fees),
            float(len(self.price_history)) / 1000.0,
            float(volatility) / max(self.tp_price_pct, 0.0001)
        ], dtype=np.float32)
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.clip(features, -10, 10)
        
        return features
    
    def calculate_pnl(self, position, current_price):
        """Calculate current PnL including fees"""
        if not position.get('is_open', False) or current_price is None or current_price <= 0:
            return 0.0
        
        btc_size = position.get('btc_size', self.base_btc_size)
        entry_price = position.get('entry_price', current_price)
        
        if position['type'] == 'LONG':
            gross_pnl = (current_price - entry_price) * btc_size
        else:
            gross_pnl = (entry_price - current_price) * btc_size
        
        exit_fee_estimate = (btc_size * current_price) * self.taker_fee_rate
        net_pnl = gross_pnl - position.get('entry_fee', 0) - exit_fee_estimate
        
        return net_pnl
    
    def calculate_pnl_percent(self, position, current_price):
        """Calculate P&L as percentage of margin"""
        net_pnl = self.calculate_pnl(position, current_price)
        margin = position.get('margin_used', 0.01)
        if margin <= 0:
            return 0.0
        return (net_pnl / margin) * 100
    
    def should_enter_trade(self, action, confidence, signal_count, price):
        """Check if we should enter a trade"""
        if price is None or price <= 0:
            return False
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if elapsed < self.startup_delay:
            return False
        
        if not self.initial_warmup_complete:
            self.initial_warmup_complete = True
            print(f"✅ Warmup complete - Trading enabled!")
        
        if elapsed < self.warmup_delay:
            if confidence < 0.8 or signal_count < 5:
                return False
        
        if self.last_trade_time:
            if (datetime.now() - self.last_trade_time).total_seconds() < self.cooldown_seconds:
                return False
        
        if len([p for p in self.positions if p.get('is_open', False)]) >= self.max_positions:
            return False
        
        if (self.capital - self.margin_used) < self.margin_per_trade * 1.1:
            return False
        
        if confidence < self.min_confidence:
            return False
        
        if signal_count < self.min_signals:
            return False
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False
        
        current_day = datetime.now().date()
        if current_day != self.trading_day:
            self.trading_day = current_day
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.consecutive_losses = 0
        
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        if self.daily_pnl < self.daily_loss_limit:
            return False
        
        return True
    
    def execute_action(self, action, price, funding_rate=None, order_book=None):
        """Execute trading action"""
        if price is None or price <= 0:
            price = self.current_price if self.current_price > 0 else 65000.0
        if funding_rate is None:
            funding_rate = 0.0001
        if order_book is None:
            order_book = {}
        
        self.current_price = price
        self.step_count += 1
        
        volume = self.last_volume if hasattr(self, 'last_volume') else 1000.0
        
        try:
            self.strategy_engine.update(price=price, volume=volume)
        except:
            pass
        
        try:
            signals = self.strategy_engine.get_combined_signal(funding_rate, order_book)
        except:
            signals = {'long_count': 0, 'short_count': 0, 'long_confidence': 0.0, 'short_confidence': 0.0}
        
        try:
            current_state = self.get_state_features(price, funding_rate, order_book)
        except:
            current_state = np.zeros(RL_STATE_SIZE, dtype=np.float32)
        
        # ===== 🆕 REWARD FOR HOLDING & PATIENCE =====
        if self.last_state is not None and self.last_action is not None:
            try:
                reward = self.calculate_reward()
                self.rl_agent.store_experience(
                    self.last_state, self.last_action, reward, current_state, False
                )
            except:
                pass
        
        # Train periodically
        if self.step_count % self.training_interval == 0:
            try:
                if len(self.rl_agent.memory) > RL_BATCH_SIZE * 2:
                    self.rl_agent.train_batch(RL_BATCH_SIZE)
            except:
                pass
        
        # Check exits
        self.check_exits(price)
        
        # ===== 🆕 HANDLE HOLD ACTION REWARD =====
        result = None
        if action == 0:  # HOLD
            self.hold_count += 1
            # Give small reward for holding when conditions are bad
            # This teaches RL to be patient
            if self.step_count % 50 == 0:  # Every ~100 seconds
                # Check if market conditions are bad
                long_conf = signals.get('long_confidence', 0)
                short_conf = signals.get('short_confidence', 0)
                
                if max(long_conf, short_conf) < self.min_confidence:
                    # Good job holding in bad conditions!
                    self.avoided_bad_trades += 1
                    if self.avoided_bad_trades % 5 == 0:
                        print(f"🧠 Patience reward: Avoided {self.avoided_bad_trades} bad trades")
        else:
            # Reset hold counter when we take action
            self.hold_count = 0
        
        # Open position if signal
        if action in [1, 2]:
            direction = 'long' if action == 1 else 'short'
            confidence = signals.get('long_confidence' if action == 1 else 'short_confidence', 0)
            signal_count = signals.get('long_count' if action == 1 else 'short_count', 0)
            
            if self.should_enter_trade(action, confidence, signal_count, price):
                try:
                    result = self.open_position(price, direction, signals)
                    self.avoided_bad_trades = 0  # Reset counter on entry
                except Exception as e:
                    print(f"⚠️ Open error: {e}")
        
        self.last_state = current_state
        self.last_action = action
        
        return result
    
    def open_position(self, price, direction='long', signals=None):
        """Open a new position"""
        if signals is None:
            signals = {}
        
        metrics = self.calculate_position_metrics(price, direction)
        total_cost = metrics['margin_required'] + metrics['entry_fee']
        
        if total_cost > self.capital:
            print(f"⚠️ Insufficient capital: Need ${total_cost:.4f}, Have ${self.capital:.4f}")
            return None
        
        self.capital -= total_cost
        self.margin_used += metrics['margin_required']
        self.total_fees_paid += metrics['entry_fee']
        self.total_fees += metrics['entry_fee']
        self.daily_trades += 1
        
        position = {
            'id': self.total_trades + 1,
            'type': 'LONG' if direction == 'long' else 'SHORT',
            'entry_price': price,
            'btc_size': metrics['btc_size'],
            'position_value': metrics['position_value'],
            'margin_used': metrics['margin_required'],
            'entry_fee': metrics['entry_fee'],
            'take_profit_price': metrics['take_profit_price'],
            'stop_loss_price': metrics['stop_loss_price'],
            'liquidation_price': metrics['liquidation_price'],
            'entry_time': datetime.now(),
            'is_open': True,
            'current_price': price,
            'pnl': -metrics['entry_fee'],
            'pnl_percent': 0.0,
            'peak_pnl_percent': 0.0,
            'peak_price': price,
            'trailing_activated': False,
            'trailing_peak_pnl': 0.0,
            'signals': signals,
            'exit_price': None,
            'exit_time': None,
            'exit_reason': None
        }
        
        self.positions.append(position)
        self.total_trades += 1
        self.last_trade_time = datetime.now()
        
        self.trade_log.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'type': 'OPEN',
            'direction': position['type'],
            'entry': price,
            'btc_size': metrics['btc_size'],
            'margin': metrics['margin_required'],
            'fee': metrics['entry_fee'],
            'tp_price': metrics['take_profit_price'],
            'sl_price': metrics['stop_loss_price'],
            'capital': self.capital
        })
        
        emoji = "📈" if direction == 'long' else "📉"
        print(f"\n{emoji} OPEN {position['type']} #{position['id']} @ ${price:,.2f}")
        print(f"   Margin: ${metrics['margin_required']:.4f} | Fee: ${metrics['entry_fee']:.4f}")
        print(f"   TP: ${metrics['take_profit_price']:,.2f} ({self.tp_pnl_percent}% P&L)")
        print(f"   SL: ${metrics['stop_loss_price']:,.2f} ({self.sl_pnl_percent}% P&L)")
        print(f"   Net Profit Est: ${metrics['net_profit_estimate']:.4f}")
        print(f"   Capital Left: ${self.capital:.4f}\n")
        
        return position
    
    def check_exits(self, current_price):
        """Check and execute exits based on P&L percentage"""
        if current_price is None or current_price <= 0:
            return
        
        self.current_price = current_price
        
        for position in self.positions[:]:
            if not position.get('is_open', False):
                continue
            
            position['current_price'] = current_price
            
            net_pnl = self.calculate_pnl(position, current_price)
            position['pnl'] = net_pnl
            
            margin = position.get('margin_used', 0.01)
            pnl_percent = (net_pnl / margin) * 100 if margin > 0 else 0
            position['pnl_percent'] = pnl_percent
            
            if pnl_percent > position.get('peak_pnl_percent', 0):
                position['peak_pnl_percent'] = pnl_percent
            
            if position['type'] == 'LONG' and current_price > position.get('peak_price', 0):
                position['peak_price'] = current_price
            elif position['type'] == 'SHORT' and current_price < position.get('peak_price', float('inf')):
                position['peak_price'] = current_price
            
            should_close = False
            close_reason = None
            
            # CHECK 1: TAKE PROFIT
            if pnl_percent >= self.tp_pnl_percent:
                should_close = True
                close_reason = f'TP_{self.tp_pnl_percent}%'
                print(f"   🎯 TAKE PROFIT! {pnl_percent:.1f}% P&L (Target: {self.tp_pnl_percent}%)")
            
            # CHECK 2: TRAILING STOP
            if not should_close:
                if pnl_percent >= self.trailing_activate_percent and not position.get('trailing_activated'):
                    position['trailing_activated'] = True
                    position['trailing_peak_pnl'] = pnl_percent
                    print(f"   🔒 Trailing ACTIVATED at {pnl_percent:.1f}% P&L")
                
                if position.get('trailing_activated'):
                    if pnl_percent > position.get('trailing_peak_pnl', 0):
                        position['trailing_peak_pnl'] = pnl_percent
                    
                    peak = position.get('trailing_peak_pnl', 0)
                    if pnl_percent <= peak - self.trailing_distance_percent:
                        should_close = True
                        close_reason = f'TRAILING_{peak:.0f}%'
                        print(f"   🔒 TRAILING STOP! {pnl_percent:.1f}% from peak {peak:.1f}%")
            
            # CHECK 3: STOP LOSS
            if not should_close:
                if pnl_percent <= -self.sl_pnl_percent:
                    should_close = True
                    close_reason = f'SL_{self.sl_pnl_percent}%'
                    print(f"   🛑 STOP LOSS! {pnl_percent:.1f}% (Limit: -{self.sl_pnl_percent}%)")
            
            # CHECK 4: LIQUIDATION
            if not should_close:
                if position['type'] == 'LONG' and current_price <= position['liquidation_price']:
                    should_close = True
                    close_reason = 'LIQUIDATION'
                elif position['type'] == 'SHORT' and current_price >= position['liquidation_price']:
                    should_close = True
                    close_reason = 'LIQUIDATION'
            
            if should_close:
                self.close_position(position, current_price, close_reason)
        
        unrealized_pnl = sum(p.get('pnl', 0) for p in self.positions if p.get('is_open', False))
        self.total_equity = self.capital + self.margin_used + unrealized_pnl
        if self.total_equity > self.peak_equity:
            self.peak_equity = self.total_equity
        self.equity_history.append(self.total_equity)
    
    def close_position(self, position, exit_price, reason):
        """Close a position and return capital"""
        position['is_open'] = False
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['exit_reason'] = reason
        
        exit_fee = (position['btc_size'] * exit_price) * self.taker_fee_rate
        
        if position['type'] == 'LONG':
            gross_pnl = (exit_price - position['entry_price']) * position['btc_size']
        else:
            gross_pnl = (position['entry_price'] - exit_price) * position['btc_size']
        
        total_fees = position['entry_fee'] + exit_fee
        net_pnl = gross_pnl - total_fees
        
        position['net_pnl'] = net_pnl
        position['exit_fee'] = exit_fee
        position['total_fees'] = total_fees
        
        hold_seconds = (position['exit_time'] - position['entry_time']).total_seconds()
        position['hold_time'] = hold_seconds
        
        pnl_pct = (net_pnl / position['margin_used']) * 100 if position['margin_used'] > 0 else 0
        
        returned_capital = position['margin_used'] + net_pnl
        self.capital += returned_capital
        self.margin_used -= position['margin_used']
        self.total_fees_paid += exit_fee
        self.total_fees += exit_fee
        self.total_pnl += net_pnl
        self.daily_pnl += net_pnl
        
        if net_pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
        
        self.closed_trades.append(position)
        
        emoji = "✅" if net_pnl > 0 else "❌"
        hold_str = f"{hold_seconds:.0f}s" if hold_seconds < 60 else f"{hold_seconds/60:.1f}m"
        
        self.trade_log.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'type': 'CLOSE',
            'direction': position['type'],
            'entry': position['entry_price'],
            'exit': exit_price,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'fees': total_fees,
            'reason': reason,
            'capital': self.capital,
            'hold_time': hold_str
        })
        
        print(f"\n{emoji} CLOSE #{position['id']} | {reason}")
        print(f"   ${position['entry_price']:,.2f} → ${exit_price:,.2f}")
        print(f"   Gross: ${gross_pnl:+.4f} | Fees: ${total_fees:.4f}")
        print(f"   Net P&L: ${net_pnl:+.4f} ({pnl_pct:+.1f}% on margin)")
        print(f"   Hold: {hold_str}")
        print(f"   Capital: ${self.capital:.4f}")
        print(f"   Win Rate: {self.wins}/{self.total_trades} ({self.wins/max(1,self.total_trades)*100:.1f}%)")
        if self.consecutive_losses > 0:
            print(f"   ⚠️ Consecutive Losses: {self.consecutive_losses}\n")
        else:
            print()
    
    def calculate_reward(self):
        """
        🆕 IMPROVED REWARD FUNCTION
        1. Scaled rewards (no cap) - bigger profits = bigger rewards
        2. HOLD rewards - teaches patience
        3. Patience bonus - rewards avoiding bad trades
        """
        reward = 0.0
        
        # ===== 1. SCALED REWARD FROM CLOSED TRADES =====
        if self.closed_trades:
            last_trade = self.closed_trades[-1]
            net_pnl = last_trade.get('net_pnl', 0)
            pnl_pct = (net_pnl / max(last_trade.get('margin_used', 0.01), 0.0001)) * 100
            
            if net_pnl > 0:
                # 🆕 SCALED: Bigger profit = bigger reward (NO CAP!)
                reward += pnl_pct * 0.15  # 40% P&L = +6.0, 80% = +12.0, 200% = +30.0
                reward += 1.0  # Base win bonus
                
                # Extra bonus for hitting exact TP (optimal exit)
                if 'TP' in str(last_trade.get('exit_reason', '')):
                    reward += 3.0  # Perfect exit bonus
                
                # Bonus for successful trailing (protected profits)
                if 'TRAILING' in str(last_trade.get('exit_reason', '')):
                    peak = last_trade.get('peak_pnl_percent', 0)
                    final = pnl_pct
                    protection = final / max(peak, 1)  # How much profit did we keep?
                    reward += 1.0 + protection  # Better protection = more reward
            else:
                # 🆕 SCALED LOSS: Bigger loss = bigger penalty
                reward += pnl_pct * 0.05  # -10% = -0.5, -50% = -2.5, -100% = -5.0
                reward -= 0.5  # Base loss penalty
                
                # Heavy penalty for liquidation
                if 'LIQUIDATION' in str(last_trade.get('exit_reason', '')):
                    reward -= 10.0  # Never liquidate!
                
                # SL penalty (less severe than liquidation)
                if 'SL' in str(last_trade.get('exit_reason', '')):
                    reward -= 1.0
            
            # Streak penalty
            if self.consecutive_losses > 1:
                reward -= self.consecutive_losses * 0.5
        
        # ===== 2. HOLD REWARD - TEACHES PATIENCE =====
        # Reward for holding when no good opportunities exist
        if self.last_action == 0:  # HOLD action
            self.hold_count += 1
            
            # Every ~60 seconds of holding, give small reward
            if self.hold_count > 0 and self.hold_count % 30 == 0:
                # Check if market conditions justify holding
                open_positions = len([p for p in self.positions if p.get('is_open', False)])
                
                if open_positions == 0:
                    # No positions and holding = waiting for good entry
                    reward += 0.05  # Small patience reward
                elif open_positions > 0:
                    # Holding with open positions = letting trades work
                    reward += 0.02  # Even smaller, but still positive
        else:
            self.hold_count = 0  # Reset on non-HOLD action
        
        # ===== 3. PATIENCE BONUS - REWARD AVOIDING BAD TRADES =====
        # When we chose HOLD while signals were weak (good decision!)
        if self.avoided_bad_trades > 0 and self.avoided_bad_trades % 10 == 0:
            reward += 0.5  # Bonus for consistently avoiding bad trades
        
        # ===== 4. CAPITAL PRESERVATION BONUS =====
        # Small ongoing reward for keeping capital safe
        if self.capital >= self.initial_capital * 0.95:  # Within 5% of starting
            reward += 0.01  # Tiny reward for capital preservation
        
        self.total_rewards_given += reward
        
        # Clip to prevent extreme values
        reward = np.clip(reward, -20.0, 30.0)
        
        return float(reward)
    
    def get_stats(self):
        """Get comprehensive trading statistics"""
        open_positions = len([p for p in self.positions if p.get('is_open', False)])
        unrealized_pnl = sum(p.get('pnl', 0) for p in self.positions if p.get('is_open', False))
        win_rate = self.wins / max(1, self.total_trades) * 100
        
        if self.closed_trades:
            avg_profit = np.mean([t.get('net_pnl', 0) for t in self.closed_trades])
            avg_fee = np.mean([t.get('total_fees', 0) for t in self.closed_trades])
            avg_hold = np.mean([t.get('hold_time', 0) for t in self.closed_trades])
            avg_pnl_pct = np.mean([
                (t.get('net_pnl', 0) / max(t.get('margin_used', 0.01), 0.0001)) * 100 
                for t in self.closed_trades
            ])
        else:
            avg_profit = 0.0
            avg_fee = 0.0
            avg_hold = 0.0
            avg_pnl_pct = 0.0
        
        return {
            'capital': self.capital,
            'equity': self.total_equity,
            'margin_used': self.margin_used,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'open_positions': open_positions,
            'unrealized_pnl': unrealized_pnl,
            'total_fees': self.total_fees,
            'avg_profit': avg_profit,
            'avg_fee': avg_fee,
            'avg_hold_time': avg_hold,
            'avg_pnl_pct': avg_pnl_pct,
            'rl_epsilon': self.rl_agent.epsilon if self.use_rl else 0,
            'rl_train_steps': self.rl_agent.train_step if self.use_rl else 0,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'avoided_bad_trades': self.avoided_bad_trades,
            'total_rewards': self.total_rewards_given
        }
    
    def get_open_positions(self):
        """Get open positions with P&L info"""
        positions_list = []
        for pos in self.positions:
            if not pos.get('is_open', False):
                continue
            
            pnl_pct = pos.get('pnl_percent', 0)
            peak_pct = pos.get('peak_pnl_percent', 0)
            
            positions_list.append({
                'id': pos.get('id', 0),
                'type': pos.get('type', ''),
                'entry_price': pos.get('entry_price', 0),
                'current_price': pos.get('current_price', 0),
                'btc_size': pos.get('btc_size', 0),
                'margin_used': pos.get('margin_used', 0),
                'pnl': pos.get('pnl', 0),
                'pnl_pct': pnl_pct,
                'peak_pnl_pct': peak_pct,
                'entry_fee': pos.get('entry_fee', 0),
                'take_profit_price': pos.get('take_profit_price', 0),
                'stop_loss_price': pos.get('stop_loss_price', 0),
                'liquidation_price': pos.get('liquidation_price', 0),
                'duration': str(datetime.now() - pos.get('entry_time', datetime.now())).split('.')[0],
                'trailing_activated': pos.get('trailing_activated', False),
                'exit_reason': None
            })
        
        return positions_list
    
    def get_trade_log(self):
        """Get formatted trade log"""
        return list(self.trade_log)[-50:]
    
    def get_equity_curve(self):
        """Get equity history"""
        return list(self.equity_history)
    
    def reset(self):
        """Reset trader state"""
        if self.use_rl and self.rl_agent.train_step > 0:
            try:
                self.rl_agent.save()
                print("💾 Model saved before reset")
            except:
                pass
        self.__init__(self.initial_capital, self.leverage, self.use_rl)
        print("🔄 Trader reset complete")