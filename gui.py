"""
gui.py - Professional Trading GUI with Confidence Display
Shows strategy confidence and signal counts
"""

import sys
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class TradingGUI(QMainWindow):
    def __init__(self, trader, data_feed):
        super().__init__()
        self.trader = trader
        self.data_feed = data_feed
        self.is_running = True
        self.bot_active = True
        
        self.init_ui()
        
        # Update timers
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)
        self.display_timer.start(100)
        
        self.trading_timer = QTimer()
        self.trading_timer.timeout.connect(self.execute_trading_cycle)
        self.trading_timer.start(2000)
        
        self.step_count = 0
        
        # Track last signals for display
        self.last_signals = {
            'long_count': 0,
            'short_count': 0,
            'long_confidence': 0.0,
            'short_confidence': 0.0,
            'direction': 'neutral'
        }
    
    def init_ui(self):
        self.setWindowTitle("🤖 Smart RL Trading Bot - Shark Exchange (₹1,500 | 30x)")
        self.setGeometry(50, 50, 1400, 900)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #0d1117; }
            QLabel { color: #c9d1d9; font-family: 'Consolas', monospace; }
            QGroupBox {
                color: #58a6ff;
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: bold;
                font-size: 13px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QTableWidget {
                background-color: #0d1117;
                color: #c9d1d9;
                gridline-color: #21262d;
                border: 1px solid #30363d;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section {
                background-color: #161b22;
                color: #58a6ff;
                padding: 6px;
                border: 1px solid #30363d;
                font-weight: bold;
            }
            QPushButton {
                background-color: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #30363d; border-color: #58a6ff; }
            QPushButton#pause_btn { background-color: #d2991d; color: #0d1117; }
            QPushButton#pause_btn:hover { background-color: #e3b341; }
            QPushButton#save_btn { background-color: #238636; color: #ffffff; }
            QPushButton#save_btn:hover { background-color: #2ea043; }
            QPushButton#reset_btn { background-color: #da3633; color: #ffffff; }
            QPushButton#reset_btn:hover { background-color: #f85149; }
            QScrollBar:vertical {
                background-color: #161b22;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #30363d;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background-color: #58a6ff; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QProgressBar {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 4px;
                text-align: center;
                color: #c9d1d9;
                font-weight: bold;
            }
            QProgressBar::chunk {
                border-radius: 3px;
            }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # ===== PRICE HEADER =====
        price_header = QWidget()
        price_layout = QHBoxLayout(price_header)
        price_layout.setContentsMargins(0, 0, 0, 0)
        
        self.price_label = QLabel("$65,000.00")
        self.price_label.setStyleSheet("color: #58a6ff; font-size: 36px; font-weight: bold;")
        price_layout.addWidget(self.price_label)
        
        self.status_label = QLabel("🔴 Connecting...")
        self.status_label.setStyleSheet("color: #f85149; font-size: 12px; padding: 6px;")
        price_layout.addWidget(self.status_label)
        
        self.funding_label = QLabel("Funding: 0.0100%")
        self.funding_label.setStyleSheet("color: #8b949e; font-size: 12px; padding: 6px;")
        price_layout.addWidget(self.funding_label)
        
        self.leverage_label = QLabel("🔧 30x")
        self.leverage_label.setStyleSheet("color: #58a6ff; font-size: 12px; padding: 6px; font-weight: bold;")
        price_layout.addWidget(self.leverage_label)
        
        price_layout.addStretch()
        layout.addWidget(price_header)
        
        # ===== CONFIDENCE BAR (NEW!) =====
        confidence_widget = QWidget()
        confidence_layout = QHBoxLayout(confidence_widget)
        confidence_layout.setContentsMargins(0, 0, 0, 0)
        confidence_layout.setSpacing(8)
        
        # Long confidence
        long_group = QGroupBox("📈 Long Signals")
        long_layout = QVBoxLayout(long_group)
        self.long_confidence_bar = QProgressBar()
        self.long_confidence_bar.setMaximum(100)
        self.long_confidence_bar.setValue(0)
        self.long_confidence_bar.setFormat("0% | 0 signals")
        self.long_confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: #3fb950; }")
        long_layout.addWidget(self.long_confidence_bar)
        confidence_layout.addWidget(long_group)
        
        # Direction indicator
        direction_group = QGroupBox("🎯 Direction")
        direction_layout = QVBoxLayout(direction_group)
        self.direction_label = QLabel("NEUTRAL")
        self.direction_label.setStyleSheet("color: #8b949e; font-size: 20px; font-weight: bold;")
        self.direction_label.setAlignment(Qt.AlignCenter)
        direction_layout.addWidget(self.direction_label)
        confidence_layout.addWidget(direction_group)
        
        # Short confidence
        short_group = QGroupBox("📉 Short Signals")
        short_layout = QVBoxLayout(short_group)
        self.short_confidence_bar = QProgressBar()
        self.short_confidence_bar.setMaximum(100)
        self.short_confidence_bar.setValue(0)
        self.short_confidence_bar.setFormat("0% | 0 signals")
        self.short_confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: #f85149; }")
        short_layout.addWidget(self.short_confidence_bar)
        confidence_layout.addWidget(short_group)
        
        layout.addWidget(confidence_widget)
        
        # ===== STATS ROW =====
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setSpacing(6)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        
        self.stat_cards = {}
        stat_items = [
            ("💰 Capital", "capital", "$18.00"),
            ("💵 Margin", "margin", "$0.00"),
            ("📊 P&L", "pnl", "$0.00"),
            ("📈 Win Rate", "winrate", "0%"),
            ("🔄 Trades", "trades", "0"),
            ("📉 Open", "open", "0"),
            ("🧠 RL ε", "epsilon", "1.000"),
            ("⚡ Avg P&L%", "avgpnlpct", "0%")
        ]
        
        for name, key, default in stat_items:
            card = QGroupBox(name)
            card_layout = QVBoxLayout(card)
            value = QLabel(default)
            value.setStyleSheet("color: #58a6ff; font-size: 16px; font-weight: bold;")
            value.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(value)
            card.setLayout(card_layout)
            stats_layout.addWidget(card)
            self.stat_cards[key] = value
        
        layout.addWidget(stats_widget)
        
        # ===== OPEN POSITIONS TABLE =====
        positions_group = QGroupBox("📊 Open Positions (Live P&L with Fees)")
        positions_layout = QVBoxLayout(positions_group)
        
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(9)
        self.positions_table.setHorizontalHeaderLabels([
            "ID", "Type", "Entry", "Current", "Size (BTC)", 
            "P&L ($)", "P&L %", "Duration", "Trailing"
        ])
        
        self.positions_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.positions_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        self.positions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.positions_table.setAlternatingRowColors(True)
        
        positions_layout.addWidget(self.positions_table)
        layout.addWidget(positions_group)
        
        # ===== TRADE LOG =====
        log_group = QGroupBox("📝 Trade History (Scrollable)")
        log_layout = QVBoxLayout(log_group)
        
        self.trade_log_table = QTableWidget()
        self.trade_log_table.setColumnCount(8)
        self.trade_log_table.setHorizontalHeaderLabels([
            "Time", "Action", "Direction", "Entry", "Exit", 
            "P&L ($)", "P&L %", "Reason"
        ])
        
        self.trade_log_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.trade_log_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.trade_log_table.horizontalHeader().setStretchLastSection(True)
        self.trade_log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.trade_log_table.setAlternatingRowColors(True)
        
        log_layout.addWidget(self.trade_log_table)
        layout.addWidget(log_group)
        
        # ===== RL STATUS BAR =====
        rl_widget = QWidget()
        rl_layout = QHBoxLayout(rl_widget)
        rl_layout.setContentsMargins(0, 0, 0, 0)
        
        self.rl_status_label = QLabel("🧠 RL: Waiting for training... | Strategy: Analyzing market...")
        self.rl_status_label.setStyleSheet("color: #d2991d; font-size: 12px; padding: 4px;")
        rl_layout.addWidget(self.rl_status_label)
        
        rl_layout.addStretch()
        layout.addWidget(rl_widget)
        
        # ===== CONTROLS =====
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addStretch()
        
        self.pause_btn = QPushButton("⏸ Pause Trading")
        self.pause_btn.setObjectName("pause_btn")
        self.pause_btn.clicked.connect(self.toggle_trading)
        controls_layout.addWidget(self.pause_btn)
        
        self.save_btn = QPushButton("💾 Save Model")
        self.save_btn.setObjectName("save_btn")
        self.save_btn.clicked.connect(self.save_model)
        controls_layout.addWidget(self.save_btn)
        
        self.reset_btn = QPushButton("🔄 Reset Bot")
        self.reset_btn.setObjectName("reset_btn")
        self.reset_btn.clicked.connect(self.reset_bot)
        controls_layout.addWidget(self.reset_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.status_bar.showMessage("🤖 Bot Ready | 30x Leverage | 40% TP / 10% SL | 1:4 R:R")
    
    def update_confidence_bars(self):
        """Update confidence display bars"""
        signals = self.last_signals
        
        long_count = signals.get('long_count', 0)
        short_count = signals.get('short_count', 0)
        long_conf = signals.get('long_confidence', 0) * 100
        short_conf = signals.get('short_confidence', 0) * 100
        direction = signals.get('direction', 'neutral')
        
        # Update long bar
        self.long_confidence_bar.setValue(int(long_conf))
        self.long_confidence_bar.setFormat(f"{long_conf:.0f}% | {long_count} signals")
        
        # Update short bar
        self.short_confidence_bar.setValue(int(short_conf))
        self.short_confidence_bar.setFormat(f"{short_conf:.0f}% | {short_count} signals")
        
        # Update direction label
        if direction == 'long':
            self.direction_label.setText("🟢 LONG")
            self.direction_label.setStyleSheet("color: #3fb950; font-size: 20px; font-weight: bold;")
        elif direction == 'short':
            self.direction_label.setText("🔴 SHORT")
            self.direction_label.setStyleSheet("color: #f85149; font-size: 20px; font-weight: bold;")
        else:
            self.direction_label.setText("⚪ NEUTRAL")
            self.direction_label.setStyleSheet("color: #8b949e; font-size: 20px; font-weight: bold;")
    
    def update_display(self):
        """Update all display elements"""
        try:
            # Price
            price = self.data_feed.get_current_price()
            if price:
                self.price_label.setText(f"${price:,.2f}")
            
            # Connection status
            status = self.data_feed.get_connection_status()
            if "Live" in status:
                self.status_label.setText("🟢 Live")
                self.status_label.setStyleSheet("color: #3fb950; font-size: 12px; padding: 6px;")
            else:
                self.status_label.setText("🔴 Sim")
                self.status_label.setStyleSheet("color: #f85149; font-size: 12px; padding: 6px;")
            
            # Funding rate
            fr = self.data_feed.funding_rate
            self.funding_label.setText(f"Funding: {fr*100:.4f}%")
            
            # Update confidence bars
            self.update_confidence_bars()
            
            # Stats
            stats = self.trader.get_stats()
            
            self.stat_cards['capital'].setText(f"${stats['capital']:.4f}")
            self.stat_cards['margin'].setText(f"${stats['margin_used']:.4f}")
            
            pnl = stats['total_pnl']
            pnl_text = f"${pnl:+.4f}"
            self.stat_cards['pnl'].setText(pnl_text)
            if pnl > 0:
                self.stat_cards['pnl'].setStyleSheet("color: #3fb950; font-size: 16px; font-weight: bold;")
            elif pnl < 0:
                self.stat_cards['pnl'].setStyleSheet("color: #f85149; font-size: 16px; font-weight: bold;")
            
            self.stat_cards['winrate'].setText(f"{stats['win_rate']:.1f}%")
            self.stat_cards['trades'].setText(str(stats['total_trades']))
            self.stat_cards['open'].setText(str(stats['open_positions']))
            self.stat_cards['epsilon'].setText(f"{stats['rl_epsilon']:.4f}")
            self.stat_cards['avgpnlpct'].setText(f"{stats['avg_pnl_pct']:+.1f}%")
            
            # Update tables
            self.update_positions_table()
            self.update_trade_log_table()
            
        except Exception as e:
            if self.step_count % 100 == 0:
                print(f"⚠️ Display error: {e}")
    
    def update_positions_table(self):
        """Update open positions table"""
        try:
            positions = self.trader.get_open_positions()
            self.positions_table.setRowCount(len(positions))
            
            for row, pos in enumerate(positions):
                self.positions_table.setItem(row, 0, QTableWidgetItem(str(pos.get('id', ''))))
                
                type_item = QTableWidgetItem(pos.get('type', ''))
                if pos.get('type') == 'LONG':
                    type_item.setForeground(QColor('#3fb950'))
                else:
                    type_item.setForeground(QColor('#f85149'))
                self.positions_table.setItem(row, 1, type_item)
                
                self.positions_table.setItem(row, 2, QTableWidgetItem(f"${pos.get('entry_price', 0):,.2f}"))
                self.positions_table.setItem(row, 3, QTableWidgetItem(f"${pos.get('current_price', 0):,.2f}"))
                self.positions_table.setItem(row, 4, QTableWidgetItem(f"{pos.get('btc_size', 0):.4f}"))
                
                pnl = pos.get('pnl', 0)
                pnl_item = QTableWidgetItem(f"${pnl:+.4f}")
                if pnl > 0:
                    pnl_item.setForeground(QColor('#3fb950'))
                elif pnl < 0:
                    pnl_item.setForeground(QColor('#f85149'))
                self.positions_table.setItem(row, 5, pnl_item)
                
                pnl_pct = pos.get('pnl_pct', 0)
                pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.1f}%")
                if pnl_pct > 0:
                    pnl_pct_item.setForeground(QColor('#3fb950'))
                elif pnl_pct < 0:
                    pnl_pct_item.setForeground(QColor('#f85149'))
                self.positions_table.setItem(row, 6, pnl_pct_item)
                
                self.positions_table.setItem(row, 7, QTableWidgetItem(pos.get('duration', '')))
                
                trailing = "🔒 ON" if pos.get('trailing_activated') else "⏸ OFF"
                trail_item = QTableWidgetItem(trailing)
                if pos.get('trailing_activated'):
                    trail_item.setForeground(QColor('#3fb950'))
                self.positions_table.setItem(row, 8, trail_item)
            
            if positions:
                self.positions_table.scrollToBottom()
                
        except Exception as e:
            if self.step_count % 100 == 0:
                print(f"⚠️ Positions error: {e}")
    
    def update_trade_log_table(self):
        """Update trade log table"""
        try:
            log = self.trader.get_trade_log()
            if not log:
                return
            
            current_rows = self.trade_log_table.rowCount()
            
            for i, entry in enumerate(log[current_rows:]):
                row = current_rows + i
                self.trade_log_table.insertRow(row)
                
                self.trade_log_table.setItem(row, 0, QTableWidgetItem(entry.get('time', '')))
                
                action = entry.get('type', '')
                action_item = QTableWidgetItem(action)
                if action == 'OPEN':
                    action_item.setForeground(QColor('#d2991d'))
                self.trade_log_table.setItem(row, 1, action_item)
                
                direction = entry.get('direction', '')
                dir_item = QTableWidgetItem(direction)
                if direction == 'LONG':
                    dir_item.setForeground(QColor('#3fb950'))
                elif direction == 'SHORT':
                    dir_item.setForeground(QColor('#f85149'))
                self.trade_log_table.setItem(row, 2, dir_item)
                
                self.trade_log_table.setItem(row, 3, QTableWidgetItem(f"${entry.get('entry', 0):,.2f}"))
                
                exit_price = entry.get('exit', None)
                if exit_price:
                    self.trade_log_table.setItem(row, 4, QTableWidgetItem(f"${exit_price:,.2f}"))
                else:
                    self.trade_log_table.setItem(row, 4, QTableWidgetItem("-"))
                
                pnl = entry.get('pnl', 0)
                pnl_item = QTableWidgetItem(f"${pnl:+.4f}")
                if pnl > 0:
                    pnl_item.setForeground(QColor('#3fb950'))
                elif pnl < 0:
                    pnl_item.setForeground(QColor('#f85149'))
                self.trade_log_table.setItem(row, 5, pnl_item)
                
                pnl_pct = entry.get('pnl_pct', 0)
                pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.1f}%")
                if pnl_pct > 0:
                    pnl_pct_item.setForeground(QColor('#3fb950'))
                elif pnl_pct < 0:
                    pnl_pct_item.setForeground(QColor('#f85149'))
                self.trade_log_table.setItem(row, 6, pnl_pct_item)
                
                reason = entry.get('reason', '')
                reason_item = QTableWidgetItem(reason)
                if 'TP' in str(reason):
                    reason_item.setForeground(QColor('#3fb950'))
                elif 'SL' in str(reason) or 'LIQUIDATION' in str(reason):
                    reason_item.setForeground(QColor('#f85149'))
                elif 'TRAILING' in str(reason):
                    reason_item.setForeground(QColor('#d2991d'))
                self.trade_log_table.setItem(row, 7, reason_item)
            
            if log:
                self.trade_log_table.scrollToBottom()
            
            while self.trade_log_table.rowCount() > 200:
                self.trade_log_table.removeRow(0)
                
        except Exception as e:
            if self.step_count % 100 == 0:
                print(f"⚠️ Log error: {e}")
    
    def execute_trading_cycle(self):
        """Execute one trading cycle"""
        if not self.bot_active or not self.is_running:
            return
        
        try:
            self.step_count += 1
            price = self.data_feed.get_current_price()
            
            if price is None or price <= 0:
                return
            
            funding_rate = self.data_feed.funding_rate or 0.0001
            order_book = self.data_feed.get_order_book() or {}
            
            # Update strategy engine
            try:
                self.trader.strategy_engine.update(price=price)
            except:
                pass
            
            # Get strategy signals for display
            try:
                signals = self.trader.strategy_engine.get_combined_signal(funding_rate, order_book)
                self.last_signals = signals
            except:
                pass
            
            # Get RL action
            try:
                state = self.trader.get_state_features(price, funding_rate, order_book)
                action = self.trader.rl_agent.get_action(state, training=True)
            except:
                action = 0
            
            # Execute action
            try:
                self.trader.execute_action(action, price, funding_rate, order_book)
            except:
                pass
            
            # Update status
            try:
                stats = self.trader.get_stats()
                action_names = {0: "HOLD ⏸", 1: "LONG 📈", 2: "SHORT 📉"}
                
                long_conf = self.last_signals.get('long_confidence', 0) * 100
                short_conf = self.last_signals.get('short_confidence', 0) * 100
                long_count = self.last_signals.get('long_count', 0)
                short_count = self.last_signals.get('short_count', 0)
                
                self.rl_status_label.setText(
                    f"🧠 Action: {action_names.get(action, '???')} | "
                    f"ε: {stats.get('rl_epsilon', 0):.3f} | "
                    f"Trades: {stats.get('total_trades', 0)} | "
                    f"Long: {long_count} signals ({long_conf:.0f}%) | "
                    f"Short: {short_count} signals ({short_conf:.0f}%)"
                )
                
                self.status_bar.showMessage(
                    f"Capital: ${stats.get('capital', 0):.4f} | "
                    f"P&L: ${stats.get('total_pnl', 0):+.4f} | "
                    f"Win Rate: {stats.get('win_rate', 0):.1f}%"
                )
            except:
                pass
            
        except Exception as e:
            if self.step_count % 100 == 0:
                print(f"⚠️ Trading cycle error: {e}")
    
    def toggle_trading(self):
        self.bot_active = not self.bot_active
        if self.bot_active:
            self.pause_btn.setText("⏸ Pause Trading")
            self.status_bar.showMessage("🤖 Trading Active")
        else:
            self.pause_btn.setText("▶ Resume Trading")
            self.status_bar.showMessage("⏸ Trading Paused")
    
    def save_model(self):
        try:
            self.trader.rl_agent.save()
            self.status_bar.showMessage("💾 Model Saved Successfully")
        except:
            self.status_bar.showMessage("⚠️ Save Failed")
    
    def reset_bot(self):
        reply = QMessageBox.question(
            self, 'Reset Bot',
            'Are you sure you want to reset?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.trader.reset()
            self.trade_log_table.setRowCount(0)
            self.positions_table.setRowCount(0)
            self.status_bar.showMessage("🔄 Bot Reset")
    
    def closeEvent(self, event):
        try:
            self.trader.rl_agent.save()
        except:
            pass
        event.accept()