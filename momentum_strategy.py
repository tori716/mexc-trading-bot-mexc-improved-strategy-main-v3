#!/usr/bin/env python3
"""
モメンタム戦略 - 調査報告書準拠実装
期待利益率: 年間25-45%（勝率60-75%）
トレンドフォロー型の強い方向性投資戦略
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import sys
import os
import math
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 既存の統一システムを継承
from unified_system_windows import (
    WindowsDataSource, ExecutionMode, TradeSignal, TradeResult,
    WindowsUnifiedSystemFactory, AnnualBacktestSystem
)

# Windows環境用のエンコーディング設定
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('momentum_strategy.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class MomentumType(Enum):
    """モメンタムタイプ"""
    PRICE_MOMENTUM = "price_momentum"      # 価格モメンタム
    TECHNICAL_MOMENTUM = "technical_momentum"  # テクニカルモメンタム
    VOLUME_MOMENTUM = "volume_momentum"    # 出来高モメンタム
    CROSS_ASSET_MOMENTUM = "cross_asset_momentum"  # 資産間モメンタム

@dataclass
class MomentumSignal:
    """モメンタムシグナル定義"""
    symbol: str
    momentum_type: MomentumType
    strength: float  # モメンタム強度（0.0-1.0）
    direction: str   # "BUY" or "SELL"
    confidence: float  # 信頼度（0.0-1.0）
    timeframe: str   # 時間軸
    entry_price: float
    target_price: float
    stop_loss_price: float
    timestamp: datetime
    duration_estimate: float  # 期待保有時間（時間）

@dataclass
class MomentumPosition:
    """モメンタムポジション"""
    signal: MomentumSignal
    entry_time: datetime
    entry_price: float
    quantity: float
    unrealized_pnl: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    is_active: bool = True

class MomentumStrategy:
    """モメンタム戦略（調査報告書準拠）"""
    
    def __init__(self, config: Dict, data_source: WindowsDataSource, execution_mode: ExecutionMode):
        self.config = config
        self.data_source = data_source
        self.execution_mode = execution_mode
        self.logger = logging.getLogger(__name__)
        
        # モメンタム管理
        self.active_positions: Dict[str, MomentumPosition] = {}
        self.trade_history = []
        self.momentum_history = {}  # モメンタム履歴
        
        # 調査報告書準拠の設定
        self.momentum_config = {
            # モメンタム検出設定
            "PRICE_MOMENTUM_PERIOD": 14,        # 価格モメンタム期間
            "MOMENTUM_THRESHOLD": 2.0,          # モメンタム閾値（標準偏差倍数）
            "MIN_MOMENTUM_STRENGTH": 0.6,       # 最小モメンタム強度
            "TREND_CONFIRMATION_PERIOD": 5,     # トレンド確認期間
            
            # テクニカル指標設定
            "RSI_PERIOD": 14,                   # RSI期間
            "RSI_OVERSOLD": 30,                 # RSI売られすぎ
            "RSI_OVERBOUGHT": 70,               # RSI買われすぎ
            "MACD_FAST": 12,                    # MACD高速EMA
            "MACD_SLOW": 26,                    # MACD低速EMA
            "MACD_SIGNAL": 9,                   # MACDシグナル
            
            # 移動平均設定
            "EMA_SHORT": 8,                     # 短期EMA
            "EMA_MEDIUM": 21,                   # 中期EMA
            "EMA_LONG": 50,                     # 長期EMA
            
            # エントリー・エグジット設定
            "ENTRY_CONFIRMATION_BARS": 2,       # エントリー確認バー数
            "TAKE_PROFIT_MULTIPLIER": 2.5,      # 利確倍数（ATR基準）
            "STOP_LOSS_MULTIPLIER": 1.5,        # 損切り倍数（ATR基準）
            "TRAILING_STOP_ACTIVATION": 1.5,    # トレーリングストップ発動利益
            "TRAILING_STOP_DISTANCE": 1.0,      # トレーリングストップ距離
            
            # ポジション管理
            "MAX_POSITIONS": 6,                 # 最大ポジション数
            "POSITION_SIZE_PCT": 15.0,          # ポジションサイズ（資金の15%）
            "MIN_VOLUME_RATIO": 1.2,            # 最小出来高倍率
            "MAX_CORRELATION": 0.7,             # 最大相関係数
            
            # 時間管理
            "MAX_HOLD_HOURS": 72,               # 最大保有時間（3日）
            "MIN_HOLD_MINUTES": 30,             # 最小保有時間（30分）
            "REBALANCE_INTERVAL_HOURS": 4,      # リバランス間隔
        }
    
    async def analyze_momentum_signals(self, symbols: List[str]) -> List[MomentumSignal]:
        """モメンタムシグナル分析"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        for symbol in symbols:
            try:
                # 複数時間軸のデータ取得
                ohlcv_1h = await self.data_source.get_ohlcv(symbol, "60m", 100)
                ohlcv_4h = await self.data_source.get_ohlcv(symbol, "240m", 50)
                
                if not ohlcv_1h or not ohlcv_4h:
                    continue
                
                # 価格モメンタム分析
                price_signals = self._analyze_price_momentum(symbol, ohlcv_1h, ohlcv_4h)
                signals.extend(price_signals)
                
                # テクニカルモメンタム分析
                technical_signals = self._analyze_technical_momentum(symbol, ohlcv_1h)
                signals.extend(technical_signals)
                
                # 出来高モメンタム分析
                volume_signals = self._analyze_volume_momentum(symbol, ohlcv_1h)
                signals.extend(volume_signals)
                
            except Exception as e:
                self.logger.warning(f"モメンタム分析エラー {symbol}: {e}")
                continue
        
        # シグナルフィルタリング・ランキング
        filtered_signals = self._filter_and_rank_signals(signals)
        
        if filtered_signals:
            self.logger.info(f"🚀 モメンタムシグナル検出: {len(filtered_signals)}件")
            for i, signal in enumerate(filtered_signals[:3]):
                self.logger.info(f"   {i+1}. {signal.symbol} {signal.direction} "
                               f"強度{signal.strength:.2f} 信頼度{signal.confidence:.2f}")
        
        return filtered_signals
    
    def _analyze_price_momentum(self, symbol: str, ohlcv_1h: List, ohlcv_4h: List) -> List[MomentumSignal]:
        """価格モメンタム分析"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        try:
            df_1h = pd.DataFrame(ohlcv_1h)
            df_4h = pd.DataFrame(ohlcv_4h)
            
            if len(df_1h) < 50 or len(df_4h) < 25:
                return signals
            
            # 価格変化率計算（複数期間）
            periods = [5, 10, 14, 20]
            momentum_scores = []
            
            for period in periods:
                if len(df_1h) > period:
                    price_change = (df_1h['close'].iloc[-1] - df_1h['close'].iloc[-period-1]) / df_1h['close'].iloc[-period-1] * 100
                    momentum_scores.append(abs(price_change))
            
            if not momentum_scores:
                return signals
            
            # モメンタム強度計算
            avg_momentum = sum(momentum_scores) / len(momentum_scores)
            momentum_strength = min(avg_momentum / 10.0, 1.0)  # 10%変化で強度1.0
            
            # トレンド方向性判定
            current_price = df_1h['close'].iloc[-1]
            ema_short = df_1h['close'].ewm(span=self.momentum_config["EMA_SHORT"]).mean().iloc[-1]
            ema_medium = df_1h['close'].ewm(span=self.momentum_config["EMA_MEDIUM"]).mean().iloc[-1]
            ema_long = df_1h['close'].ewm(span=self.momentum_config["EMA_LONG"]).mean().iloc[-1]
            
            # EMAの並び順でトレンド判定
            if ema_short > ema_medium > ema_long and current_price > ema_short:
                direction = "BUY"
                confidence = 0.8
            elif ema_short < ema_medium < ema_long and current_price < ema_short:
                direction = "SELL"
                confidence = 0.8
            else:
                direction = "BUY" if current_price > ema_medium else "SELL"
                confidence = 0.6
            
            # モメンタム閾値チェック
            if momentum_strength >= self.momentum_config["MIN_MOMENTUM_STRENGTH"]:
                
                # ATR計算（リスク管理用）
                atr = self._calculate_atr(df_1h, 14)
                
                # ターゲット・ストップロス価格計算
                if direction == "BUY":
                    target_price = current_price * (1 + (atr * self.momentum_config["TAKE_PROFIT_MULTIPLIER"] / current_price))
                    stop_loss_price = current_price * (1 - (atr * self.momentum_config["STOP_LOSS_MULTIPLIER"] / current_price))
                else:
                    target_price = current_price * (1 - (atr * self.momentum_config["TAKE_PROFIT_MULTIPLIER"] / current_price))
                    stop_loss_price = current_price * (1 + (atr * self.momentum_config["STOP_LOSS_MULTIPLIER"] / current_price))
                
                signal = MomentumSignal(
                    symbol=symbol,
                    momentum_type=MomentumType.PRICE_MOMENTUM,
                    strength=momentum_strength,
                    direction=direction,
                    confidence=confidence,
                    timeframe="1h",
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss_price=stop_loss_price,
                    timestamp=current_time,
                    duration_estimate=24  # 24時間程度の保有期待
                )
                signals.append(signal)
        
        except Exception as e:
            self.logger.warning(f"価格モメンタム分析エラー {symbol}: {e}")
        
        return signals
    
    def _analyze_technical_momentum(self, symbol: str, ohlcv_data: List) -> List[MomentumSignal]:
        """テクニカルモメンタム分析"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        try:
            df = pd.DataFrame(ohlcv_data)
            if len(df) < 50:
                return signals
            
            # RSI計算
            rsi = self._calculate_rsi(df['close'], self.momentum_config["RSI_PERIOD"])
            current_rsi = rsi.iloc[-1]
            
            # MACD計算
            macd_line, macd_signal, macd_histogram = self._calculate_macd(
                df['close'], 
                self.momentum_config["MACD_FAST"], 
                self.momentum_config["MACD_SLOW"], 
                self.momentum_config["MACD_SIGNAL"]
            )
            
            current_macd = macd_line.iloc[-1]
            current_signal = macd_signal.iloc[-1]
            current_histogram = macd_histogram.iloc[-1]
            
            # ボリンジャーバンド計算
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'], 20, 2)
            current_price = df['close'].iloc[-1]
            
            # テクニカル条件判定
            technical_strength = 0.0
            direction = None
            confidence = 0.0
            
            # MACD強気シグナル
            if current_macd > current_signal and macd_histogram.iloc[-1] > macd_histogram.iloc[-2]:
                technical_strength += 0.3
                direction = "BUY"
                confidence += 0.25
            
            # MACD弱気シグナル
            elif current_macd < current_signal and macd_histogram.iloc[-1] < macd_histogram.iloc[-2]:
                technical_strength += 0.3
                direction = "SELL"
                confidence += 0.25
            
            # RSIモメンタム
            if 30 < current_rsi < 70:  # 中立域でのモメンタム
                if current_rsi > 55:
                    technical_strength += 0.2
                    if direction != "SELL":
                        direction = "BUY"
                    confidence += 0.15
                elif current_rsi < 45:
                    technical_strength += 0.2
                    if direction != "BUY":
                        direction = "SELL"
                    confidence += 0.15
            
            # ボリンジャーバンドブレイクアウト
            if current_price > bb_upper.iloc[-1]:
                technical_strength += 0.3
                direction = "BUY"
                confidence += 0.3
            elif current_price < bb_lower.iloc[-1]:
                technical_strength += 0.3
                direction = "SELL"
                confidence += 0.3
            
            # 移動平均クロス
            ema_8 = df['close'].ewm(span=8).mean()
            ema_21 = df['close'].ewm(span=21).mean()
            
            if ema_8.iloc[-1] > ema_21.iloc[-1] and ema_8.iloc[-2] <= ema_21.iloc[-2]:
                technical_strength += 0.2
                direction = "BUY"
                confidence += 0.2
            elif ema_8.iloc[-1] < ema_21.iloc[-1] and ema_8.iloc[-2] >= ema_21.iloc[-2]:
                technical_strength += 0.2
                direction = "SELL"
                confidence += 0.2
            
            # シグナル生成
            if technical_strength >= 0.5 and direction and confidence >= 0.4:
                
                # ATR計算
                atr = self._calculate_atr(df, 14)
                
                # ターゲット・ストップロス価格計算
                if direction == "BUY":
                    target_price = current_price * (1 + (atr * 2.0 / current_price))
                    stop_loss_price = current_price * (1 - (atr * 1.2 / current_price))
                else:
                    target_price = current_price * (1 - (atr * 2.0 / current_price))
                    stop_loss_price = current_price * (1 + (atr * 1.2 / current_price))
                
                signal = MomentumSignal(
                    symbol=symbol,
                    momentum_type=MomentumType.TECHNICAL_MOMENTUM,
                    strength=technical_strength,
                    direction=direction,
                    confidence=confidence,
                    timeframe="1h",
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss_price=stop_loss_price,
                    timestamp=current_time,
                    duration_estimate=12  # 12時間程度の保有期待
                )
                signals.append(signal)
        
        except Exception as e:
            self.logger.warning(f"テクニカルモメンタム分析エラー {symbol}: {e}")
        
        return signals
    
    def _analyze_volume_momentum(self, symbol: str, ohlcv_data: List) -> List[MomentumSignal]:
        """出来高モメンタム分析"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        try:
            df = pd.DataFrame(ohlcv_data)
            if len(df) < 30:
                return signals
            
            # 出来高分析
            current_volume = df['volume'].iloc[-1]
            avg_volume_20 = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # 出来高加重平均価格（VWAP）
            vwap = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            current_price = df['close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            
            # 価格・出来高相関
            price_changes = df['close'].pct_change().dropna()
            volume_changes = df['volume'].pct_change().dropna()
            
            if len(price_changes) >= 10 and len(volume_changes) >= 10:
                correlation = price_changes.tail(10).corr(volume_changes.tail(10))
                
                # 出来高モメンタム条件
                volume_strength = 0.0
                direction = None
                confidence = 0.0
                
                # 大量出来高ブレイクアウト
                if volume_ratio > self.momentum_config["MIN_VOLUME_RATIO"]:
                    volume_strength += 0.4
                    confidence += 0.3
                    
                    # VWAP基準方向判定
                    if current_price > current_vwap * 1.002:  # 0.2%以上上
                        direction = "BUY"
                        confidence += 0.2
                    elif current_price < current_vwap * 0.998:  # 0.2%以上下
                        direction = "SELL"
                        confidence += 0.2
                
                # 価格・出来高の正相関（トレンド継続）
                if not pd.isna(correlation) and abs(correlation) > 0.5:
                    volume_strength += 0.3
                    confidence += 0.2
                    
                    if correlation > 0 and current_price > current_vwap:
                        direction = "BUY"
                    elif correlation > 0 and current_price < current_vwap:
                        direction = "SELL"
                
                # 累積出来高指数（OBV風）
                obv = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
                obv_sma = obv.rolling(window=10).mean()
                
                if obv.iloc[-1] > obv_sma.iloc[-1] and obv.iloc[-2] <= obv_sma.iloc[-2]:
                    volume_strength += 0.3
                    direction = "BUY"
                    confidence += 0.25
                elif obv.iloc[-1] < obv_sma.iloc[-1] and obv.iloc[-2] >= obv_sma.iloc[-2]:
                    volume_strength += 0.3
                    direction = "SELL"
                    confidence += 0.25
                
                # シグナル生成
                if volume_strength >= 0.6 and direction and confidence >= 0.5:
                    
                    # ATR計算
                    atr = self._calculate_atr(df, 14)
                    
                    # ターゲット・ストップロス価格計算
                    if direction == "BUY":
                        target_price = current_price * (1 + (atr * 2.2 / current_price))
                        stop_loss_price = current_price * (1 - (atr * 1.3 / current_price))
                    else:
                        target_price = current_price * (1 - (atr * 2.2 / current_price))
                        stop_loss_price = current_price * (1 + (atr * 1.3 / current_price))
                    
                    signal = MomentumSignal(
                        symbol=symbol,
                        momentum_type=MomentumType.VOLUME_MOMENTUM,
                        strength=volume_strength,
                        direction=direction,
                        confidence=confidence,
                        timeframe="1h",
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss_price=stop_loss_price,
                        timestamp=current_time,
                        duration_estimate=8  # 8時間程度の保有期待
                    )
                    signals.append(signal)
        
        except Exception as e:
            self.logger.warning(f"出来高モメンタム分析エラー {symbol}: {e}")
        
        return signals
    
    def _filter_and_rank_signals(self, signals: List[MomentumSignal]) -> List[MomentumSignal]:
        """シグナルフィルタリング・ランキング"""
        
        # フィルタリング
        filtered = []
        for signal in signals:
            # 最小強度チェック
            if signal.strength < self.momentum_config["MIN_MOMENTUM_STRENGTH"]:
                continue
            
            # 最小信頼度チェック
            if signal.confidence < 0.4:
                continue
            
            # リスクリワード比チェック
            if signal.direction == "BUY":
                risk = signal.entry_price - signal.stop_loss_price
                reward = signal.target_price - signal.entry_price
            else:
                risk = signal.stop_loss_price - signal.entry_price
                reward = signal.entry_price - signal.target_price
            
            if risk <= 0 or reward <= 0 or (reward / risk) < 1.2:
                continue
            
            filtered.append(signal)
        
        # ランキング（強度 × 信頼度でソート）
        filtered.sort(key=lambda x: x.strength * x.confidence, reverse=True)
        
        return filtered
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD計算"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンド計算"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """ATR計算"""
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        return atr if not pd.isna(atr) else df['close'].iloc[-1] * 0.02  # 2%をデフォルト
    
    async def execute_momentum_trade(self, signal: MomentumSignal) -> bool:
        """モメンタム取引実行"""
        
        try:
            current_price = await self.data_source.get_current_price(signal.symbol)
            current_time = self.data_source.get_current_time()
            
            # ポジションサイズ計算
            risk_per_trade = 1000 * (self.momentum_config["POSITION_SIZE_PCT"] / 100)  # 15%
            if signal.direction == "BUY":
                risk_per_share = signal.entry_price - signal.stop_loss_price
            else:
                risk_per_share = signal.stop_loss_price - signal.entry_price
            
            if risk_per_share <= 0:
                return False
            
            quantity = risk_per_trade / risk_per_share
            
            # 最小取引量チェック
            if quantity * current_price < 50:  # 最小$50
                quantity = 50 / current_price
            
            # エントリー実行
            order_result = await self.data_source.place_order(
                symbol=signal.symbol,
                side=signal.direction,
                order_type="MARKET",
                quantity=quantity
            )
            
            if order_result.get("status") == "FILLED":
                # ポジション記録
                position = MomentumPosition(
                    signal=signal,
                    entry_time=current_time,
                    entry_price=current_price,
                    quantity=quantity
                )
                
                self.active_positions[signal.symbol] = position
                
                self.logger.info(f"🚀 モメンタム取引実行: {signal.momentum_type.value}")
                self.logger.info(f"   {signal.symbol} {signal.direction} ${current_price:.4f}")
                self.logger.info(f"   強度{signal.strength:.2f} 信頼度{signal.confidence:.2f}")
                
                return True
            else:
                self.logger.error(f"❌ モメンタム取引失敗: {signal.symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ モメンタム取引実行エラー: {str(e)}")
            return False
    
    async def manage_momentum_positions(self) -> List[TradeResult]:
        """モメンタムポジション管理"""
        
        trades = []
        current_time = self.data_source.get_current_time()
        positions_to_close = []
        
        for symbol, position in self.active_positions.items():
            if not position.is_active:
                continue
            
            try:
                current_price = await self.data_source.get_current_price(symbol)
                signal = position.signal
                
                # 損益計算
                if signal.direction == "BUY":
                    unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    profit_pct = (current_price - position.entry_price) / position.entry_price * 100
                else:
                    unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    profit_pct = (position.entry_price - current_price) / position.entry_price * 100
                
                position.unrealized_pnl = unrealized_pnl
                position.max_profit = max(position.max_profit, unrealized_pnl)
                position.max_loss = min(position.max_loss, unrealized_pnl)
                
                # 利確条件チェック
                if signal.direction == "BUY" and current_price >= signal.target_price:
                    trade = await self._close_momentum_position(position, current_price, current_time, "利確達成")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                elif signal.direction == "SELL" and current_price <= signal.target_price:
                    trade = await self._close_momentum_position(position, current_price, current_time, "利確達成")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                
                # ストップロス条件チェック
                elif signal.direction == "BUY" and current_price <= signal.stop_loss_price:
                    trade = await self._close_momentum_position(position, current_price, current_time, "ストップロス")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                elif signal.direction == "SELL" and current_price >= signal.stop_loss_price:
                    trade = await self._close_momentum_position(position, current_price, current_time, "ストップロス")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                
                # トレーリングストップ
                elif position.max_profit > 0 and position.max_profit >= position.entry_price * position.quantity * 0.05:  # 5%利益でトレーリング開始
                    trailing_stop_price = None
                    
                    if signal.direction == "BUY":
                        trailing_stop_price = current_price * (1 - self.momentum_config["TRAILING_STOP_DISTANCE"] / 100)
                        if current_price <= trailing_stop_price:
                            trade = await self._close_momentum_position(position, current_price, current_time, "トレーリングストップ")
                    else:
                        trailing_stop_price = current_price * (1 + self.momentum_config["TRAILING_STOP_DISTANCE"] / 100)
                        if current_price >= trailing_stop_price:
                            trade = await self._close_momentum_position(position, current_price, current_time, "トレーリングストップ")
                    
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                
                # 時間切れチェック
                hold_hours = (current_time - position.entry_time).total_seconds() / 3600
                if hold_hours >= self.momentum_config["MAX_HOLD_HOURS"]:
                    trade = await self._close_momentum_position(position, current_price, current_time, "時間切れ")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                
            except Exception as e:
                self.logger.warning(f"ポジション管理エラー {symbol}: {e}")
                continue
        
        # ポジションクローズ
        for symbol in positions_to_close:
            del self.active_positions[symbol]
        
        return trades
    
    async def _close_momentum_position(self, position: MomentumPosition, current_price: float,
                                     current_time: datetime, exit_reason: str) -> Optional[TradeResult]:
        """モメンタムポジション決済"""
        
        try:
            signal = position.signal
            
            # 反対売買実行
            side = "SELL" if signal.direction == "BUY" else "BUY"
            order_result = await self.data_source.place_order(
                symbol=signal.symbol,
                side=side,
                order_type="MARKET",
                quantity=position.quantity
            )
            
            if order_result.get("status") == "FILLED":
                # 損益計算
                if signal.direction == "BUY":
                    profit_loss = (current_price - position.entry_price) * position.quantity
                else:
                    profit_loss = (position.entry_price - current_price) * position.quantity
                
                profit_pct = (profit_loss / (position.entry_price * position.quantity)) * 100
                hold_hours = (current_time - position.entry_time).total_seconds() / 3600
                
                trade = TradeResult(
                    symbol=signal.symbol,
                    entry_time=position.entry_time,
                    exit_time=current_time,
                    side=signal.direction,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    quantity=position.quantity,
                    profit_loss=profit_loss,
                    profit_pct=profit_pct,
                    hold_hours=hold_hours,
                    exit_reason=f"モメンタム_{exit_reason}"
                )
                
                self.trade_history.append(trade)
                
                self.logger.info(f"🚀 モメンタム決済: {signal.symbol} {exit_reason}")
                self.logger.info(f"   エントリー: ${position.entry_price:.4f} → 決済: ${current_price:.4f}")
                self.logger.info(f"   利益: ${profit_loss:.2f} ({profit_pct:+.2f}%) 保有{hold_hours:.1f}時間")
                
                return trade
            else:
                self.logger.error(f"❌ モメンタム決済失敗: {signal.symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"モメンタム決済エラー: {str(e)}")
            return None

class MomentumBacktestSystem(AnnualBacktestSystem):
    """モメンタムバックテストシステム"""
    
    def __init__(self, config: Dict, historical_data: Dict[str, pd.DataFrame], 
                 symbols: List[str], start_date: datetime, end_date: datetime):
        super().__init__(config, historical_data, symbols, start_date, end_date)
        
        self.momentum_strategy = MomentumStrategy(
            config, WindowsDataSource(ExecutionMode.BACKTEST, historical_data), ExecutionMode.BACKTEST
        )
        
        # モメンタム専用設定
        self.enhanced_config.update({
            "STRATEGY_NAME": "モメンタム戦略",
            "EXPECTED_ANNUAL_RETURN": 35.0,  # 25-45%の中央値
            "MAX_POSITIONS": 6,              # 最大ポジション数
            "REBALANCE_INTERVAL": 4,         # 4時間ごとチェック
        })
    
    async def _execute_annual_backtest(self):
        """モメンタム年間バックテスト実行"""
        
        capital = self.enhanced_config["INITIAL_CAPITAL"]
        
        # 時間ステップ（4時間ごと）
        timestamps = list(self.historical_data[self.symbols[0]].index[::4])
        
        for i, timestamp in enumerate(timestamps):
            try:
                # データソース時刻設定
                self.momentum_strategy.data_source.set_current_time(timestamp)
                current_portfolio_value = capital
                
                # 既存モメンタムポジション管理
                trades = await self.momentum_strategy.manage_momentum_positions()
                for trade in trades:
                    capital += trade.profit_loss + (trade.entry_price * trade.quantity)
                    self.trades.append(trade)
                    current_portfolio_value += trade.profit_loss
                
                # 新規モメンタムシグナル検索
                active_positions = len(self.momentum_strategy.active_positions)
                if active_positions < self.enhanced_config["MAX_POSITIONS"]:
                    
                    signals = await self.momentum_strategy.analyze_momentum_signals(self.symbols)
                    
                    for signal in signals[:3]:  # TOP3実行
                        if active_positions >= self.enhanced_config["MAX_POSITIONS"]:
                            break
                        
                        # 重複チェック
                        if signal.symbol not in self.momentum_strategy.active_positions:
                            required_capital = signal.entry_price * 100  # 最小$100
                            
                            if capital > required_capital:
                                success = await self.momentum_strategy.execute_momentum_trade(signal)
                                if success:
                                    capital -= required_capital
                                    active_positions += 1
                                    self.logger.info(f"🚀 {signal.symbol} モメンタム開始: {signal.momentum_type.value}")
                
                # ポートフォリオ価値計算
                momentum_investment = sum([
                    pos.entry_price * pos.quantity for pos in self.momentum_strategy.active_positions.values()
                ])
                portfolio_value = capital + momentum_investment
                
                # 日次記録
                if i % 6 == 0:  # 24時間ごと
                    self.daily_portfolio.append({
                        'timestamp': timestamp,
                        'portfolio_value': portfolio_value,
                        'cash': capital,
                        'positions': len(self.momentum_strategy.active_positions),
                        'return_pct': ((portfolio_value - self.enhanced_config["INITIAL_CAPITAL"]) / 
                                     self.enhanced_config["INITIAL_CAPITAL"]) * 100
                    })
                
                # 進捗表示
                if i % 168 == 0:  # 週次
                    progress = (i / len(timestamps)) * 100
                    weeks = i // 42
                    active_positions = len(self.momentum_strategy.active_positions)
                    self.logger.info(f"  進捗: {progress:.1f}% ({weeks}週経過) アクティブポジション:{active_positions}")
                
            except Exception as e:
                self.logger.warning(f"タイムスタンプ {timestamp} でエラー: {str(e)}")
                continue

# メイン実行関数
async def run_momentum_backtest():
    """モメンタム戦略バックテスト実行"""
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 モメンタム戦略 1年間バックテスト開始")
    
    # モメンタム用設定
    config = {
        "STRATEGY_TYPE": "MOMENTUM",
        "PRICE_MOMENTUM_PERIOD": 14,
        "MOMENTUM_THRESHOLD": 2.0,
        "MIN_MOMENTUM_STRENGTH": 0.6,
        "RSI_PERIOD": 14,
        "MACD_FAST": 12,
        "MACD_SLOW": 26,
        "MACD_SIGNAL": 9,
        "TAKE_PROFIT_MULTIPLIER": 2.5,
        "STOP_LOSS_MULTIPLIER": 1.5,
        "MAX_POSITIONS": 6,
        "POSITION_SIZE_PCT": 15.0
    }
    
    # 1年間バックテストシステム作成
    logger.info("📊 モメンタムバックテストシステム作成中...")
    annual_system = await WindowsUnifiedSystemFactory.create_annual_backtest_system(
        config, use_real_data=True
    )
    
    # モメンタムシステムに変換
    momentum_system = MomentumBacktestSystem(
        config, annual_system.historical_data, annual_system.symbols,
        annual_system.start_date, annual_system.end_date
    )
    
    # バックテスト実行
    logger.info("🚀 モメンタムバックテスト実行中...")
    results = await momentum_system.run_annual_comprehensive_backtest()
    
    # 結果表示
    print("\n" + "="*80)
    print("🚀 モメンタム戦略 1年間バックテスト完了")
    print("📊 調査報告書準拠実装（期待年利25-45%）")
    print("="*80)
    
    perf = results['performance_metrics']
    print(f"\n📈 パフォーマンス結果:")
    print(f"   戦略タイプ: モメンタム（トレンドフォロー）")
    print(f"   総取引数: {perf['total_trades']}")
    print(f"   勝率: {perf['win_rate']:.1f}%")
    print(f"   総リターン: {perf['total_return']:+.1f}%")
    print(f"   最大ドローダウン: {perf['max_drawdown']:.1f}%")
    print(f"   シャープレシオ: {perf['sharpe_ratio']:.2f}")
    print(f"   プロフィットファクター: {perf['profit_factor']:.2f}")
    
    # 目標達成評価
    target_monthly = 10.0  # 月10%目標
    target_annual = target_monthly * 12  # 年120%
    achievement_rate = (perf['total_return'] / target_annual) * 100
    
    print(f"\n🎯 目標達成度:")
    print(f"   月10%目標 (年120%) vs 実績年{perf['total_return']:+.1f}%")
    print(f"   達成率: {achievement_rate:.1f}%")
    
    if perf['total_return'] >= 25.0:
        print("✅ 調査報告書期待値（年25-45%）達成")
    else:
        print("❌ 調査報告書期待値未達成")
    
    # 全戦略比較
    print(f"\n📊 戦略比較:")
    print(f"   グリッド取引: +0.2% (勝率100%, 取引53)")
    print(f"   DCA Bot: +0.0% (勝率100%, 取引1)")
    print(f"   アービトラージ: -0.2% (勝率49.3%, 取引505)")
    print(f"   モメンタム: {perf['total_return']:+.1f}% (勝率{perf['win_rate']:.1f}%, 取引{perf['total_trades']})")
    
    return results

async def main():
    """メイン実行"""
    await run_momentum_backtest()

if __name__ == "__main__":
    asyncio.run(main())