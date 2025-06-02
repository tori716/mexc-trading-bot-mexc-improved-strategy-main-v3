#!/usr/bin/env python3
"""
スキャルピング改善版 - 取引機会劇的増加戦略
改善点: シグナル強度0.6→0.4, 緊急度0.7→0.5, 5分足採用, 利確0.3%→0.5%
目標: +0.0% → +15-25%, 取引数0→200-500
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
        logging.FileHandler('scalping_improved.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ScalpingSignalType(Enum):
    """スキャルピングシグナルタイプ"""
    MOMENTUM_BREAKOUT = "momentum_breakout"        # モメンタムブレイクアウト
    MEAN_REVERSION_QUICK = "mean_reversion_quick"  # 短期平均回帰
    VOLATILITY_EXPANSION = "volatility_expansion"  # ボラティリティ拡大
    ORDER_FLOW = "order_flow"                      # 注文フロー（簡易版）

class MarketMicrostructure(Enum):
    """市場微細構造"""
    HIGH_LIQUIDITY = "high_liquidity"      # 高流動性
    NORMAL_LIQUIDITY = "normal_liquidity"  # 通常流動性
    LOW_LIQUIDITY = "low_liquidity"        # 低流動性
    VOLATILE = "volatile"                  # 高ボラティリティ

@dataclass
class ScalpingSignal:
    """スキャルピングシグナル定義"""
    symbol: str
    signal_type: ScalpingSignalType
    direction: str   # "BUY" or "SELL"
    urgency: float   # 緊急度（0.0-1.0）
    strength: float  # シグナル強度（0.0-1.0）
    entry_price: float
    take_profit_price: float
    stop_loss_price: float
    expected_profit_pct: float  # 期待利益率
    risk_pct: float            # リスク率
    max_hold_seconds: int      # 最大保有秒数
    market_condition: MarketMicrostructure
    timestamp: datetime

@dataclass
class ScalpingPosition:
    """スキャルピングポジション"""
    signal: ScalpingSignal
    entry_time: datetime
    entry_price: float
    quantity: float
    filled_price: float        # 実際の約定価格
    slippage_pct: float       # スリッページ
    seconds_held: int = 0
    peak_profit_pct: float = 0.0     # 最高利益率
    max_drawdown_pct: float = 0.0    # 最大ドローダウン
    is_active: bool = True

class ImprovedScalpingStrategy:
    """スキャルピング改善版戦略"""
    
    def __init__(self, config: Dict, data_source: WindowsDataSource, execution_mode: ExecutionMode):
        self.config = config
        self.data_source = data_source
        self.execution_mode = execution_mode
        self.logger = logging.getLogger(__name__)
        
        # スキャルピング管理
        self.active_positions: Dict[str, ScalpingPosition] = {}
        self.trade_history = []
        self.last_scan_time = None
        self.market_conditions = {}  # 銘柄別市場状態
        
        # 🚀 改善版設定（大幅緩和）
        self.scalping_config = {
            # 基本設定
            "SCAN_INTERVAL_SECONDS": 60,        # 30秒 → 60秒（5分足対応）
            "MAX_POSITIONS": 10,                # 8 → 10（積極化）
            "POSITION_SIZE_PCT": 8.0,           # 5.0% → 8.0%（積極化）
            "MIN_TRADE_SIZE_USD": 20,           # 25 → 20（下限引き下げ）
            
            # 利益・損失設定
            "TARGET_PROFIT_PCT": 0.5,           # 🎯 0.3% → 0.5%（現実的利確）
            "STOP_LOSS_PCT": 0.6,               # 0.4% → 0.6%（余裕確保）
            "BREAKEVEN_STOP_PCT": 0.15,         # 0.1% → 0.15%（余裕確保）
            "TRAILING_STOP_PCT": 0.2,           # 0.15% → 0.2%（余裕確保）
            
            # 時間管理
            "MAX_HOLD_SECONDS": 600,            # 300秒 → 600秒（10分、余裕確保）
            "MIN_HOLD_SECONDS": 30,             # 10秒 → 30秒（安定化）
            "FORCE_EXIT_SECONDS": 1200,         # 600秒 → 1200秒（20分）
            
            # テクニカル指標（緩和）
            "EMA_ULTRA_SHORT": 5,               # 3 → 5（安定化）
            "EMA_SHORT": 12,                    # 8 → 12（安定化）
            "EMA_MEDIUM": 26,                   # 21 → 26（安定化）
            "RSI_PERIOD": 14,                   # 9 → 14（標準化）
            "RSI_OVERBOUGHT": 70,               # 75 → 70（感度向上）
            "RSI_OVERSOLD": 30,                 # 25 → 30（感度向上）
            
            # ボラティリティ・流動性（大幅緩和）
            "MIN_VOLATILITY": 0.001,            # 🎯 0.003 → 0.001（大幅緩和）
            "MAX_VOLATILITY": 0.08,             # 0.05 → 0.08（緩和）
            "MIN_VOLUME_RATIO": 1.0,            # 🎯 1.5 → 1.0（大幅緩和）
            "SPREAD_THRESHOLD": 0.005,          # 0.002 → 0.005（緩和）
            
            # シグナル強度（大幅緩和）
            "MIN_SIGNAL_STRENGTH": 0.4,         # 🎯 0.6 → 0.4（大幅緩和）
            "MIN_URGENCY": 0.5,                 # 🎯 0.7 → 0.5（大幅緩和）
            "CONFLUENCE_BONUS": 0.3,            # 0.2 → 0.3（複数シグナル重視）
            
            # リスク管理
            "MAX_DAILY_LOSS_PCT": 15.0,         # 10.0% → 15.0%（余裕確保）
            "MAX_CONSECUTIVE_LOSSES": 8,        # 5 → 8（余裕確保）
            "SLIPPAGE_BUFFER": 0.08,            # 0.05% → 0.08%（余裕確保）
            "COMMISSION_PCT": 0.02,             # 手数料据え置き
        }
    
    async def scan_scalping_opportunities(self, symbols: List[str]) -> List[ScalpingSignal]:
        """スキャルピング機会スキャン（改善版）"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        # 高頻度スキャンの時間制御（緩和）
        if (self.last_scan_time and 
            (current_time - self.last_scan_time).total_seconds() < self.scalping_config["SCAN_INTERVAL_SECONDS"]):
            return signals
        
        self.last_scan_time = current_time
        
        for symbol in symbols:
            try:
                # 🚀 5分足データ採用（ノイズ削減）
                ohlcv_5m = await self.data_source.get_ohlcv(symbol, "5m", 100)
                ohlcv_15m = await self.data_source.get_ohlcv(symbol, "15m", 50)
                
                if not ohlcv_5m or not ohlcv_15m:
                    continue
                
                current_price = await self.data_source.get_current_price(symbol)
                
                # 市場微細構造分析（緩和版）
                market_condition = self._analyze_market_microstructure(symbol, ohlcv_5m)
                
                # 🚀 低流動性でも取引許可（条件緩和）
                # if market_condition == MarketMicrostructure.LOW_LIQUIDITY:
                #     continue
                
                # モメンタムブレイクアウト分析
                momentum_signals = self._analyze_momentum_breakout(symbol, ohlcv_5m, ohlcv_15m, current_price, market_condition)
                signals.extend(momentum_signals)
                
                # 短期平均回帰分析
                reversion_signals = self._analyze_quick_mean_reversion(symbol, ohlcv_5m, current_price, market_condition)
                signals.extend(reversion_signals)
                
                # ボラティリティ拡大分析
                volatility_signals = self._analyze_volatility_expansion(symbol, ohlcv_5m, current_price, market_condition)
                signals.extend(volatility_signals)
                
                # 注文フロー分析（簡易版）
                flow_signals = self._analyze_order_flow(symbol, ohlcv_5m, current_price, market_condition)
                signals.extend(flow_signals)
                
            except Exception as e:
                self.logger.warning(f"スキャルピング分析エラー {symbol}: {e}")
                continue
        
        # シグナルフィルタリング・ランキング（緩和版）
        filtered_signals = self._filter_and_rank_scalping_signals(signals)
        
        if filtered_signals:
            self.logger.info(f"⚡ スキャルピング改善版シグナル検出: {len(filtered_signals)}件")
            for i, signal in enumerate(filtered_signals[:5]):
                self.logger.info(f"   {i+1}. {signal.symbol} {signal.direction} "
                               f"{signal.signal_type.value} 利益{signal.expected_profit_pct:.2f}%")
        
        return filtered_signals
    
    def _analyze_market_microstructure(self, symbol: str, ohlcv_5m: List) -> MarketMicrostructure:
        """市場微細構造分析（緩和版）"""
        
        try:
            df = pd.DataFrame(ohlcv_5m)
            if len(df) < 10:  # 20 → 10（緩和）
                return MarketMicrostructure.NORMAL_LIQUIDITY  # デフォルト値変更
            
            # 出来高分析（緩和）
            recent_volume = df['volume'].tail(5).mean()  # 10 → 5
            avg_volume = df['volume'].tail(20).mean()    # 50 → 20
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # ボラティリティ分析
            returns = df['close'].pct_change().dropna()
            volatility = returns.tail(10).std()  # 20 → 10
            
            # スプレッド推定（高値-安値）
            recent_spreads = ((df['high'] - df['low']) / df['close']).tail(5)  # 10 → 5
            avg_spread = recent_spreads.mean()
            
            # 🚀 流動性判定（大幅緩和）
            if volume_ratio < 0.3 or avg_spread > self.scalping_config["SPREAD_THRESHOLD"] * 3:  # より厳しい条件のみ除外
                return MarketMicrostructure.LOW_LIQUIDITY
            elif volume_ratio > 2.0 and avg_spread < self.scalping_config["SPREAD_THRESHOLD"]:  # 条件緩和
                return MarketMicrostructure.HIGH_LIQUIDITY
            elif volatility > self.scalping_config["MAX_VOLATILITY"]:
                return MarketMicrostructure.VOLATILE
            else:
                return MarketMicrostructure.NORMAL_LIQUIDITY
                
        except Exception as e:
            self.logger.warning(f"市場微細構造分析エラー {symbol}: {e}")
            return MarketMicrostructure.NORMAL_LIQUIDITY  # エラー時もデフォルト許可
    
    def _analyze_momentum_breakout(self, symbol: str, ohlcv_5m: List, ohlcv_15m: List, 
                                 current_price: float, market_condition: MarketMicrostructure) -> List[ScalpingSignal]:
        """モメンタムブレイクアウト分析（改善版）"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        try:
            df_5m = pd.DataFrame(ohlcv_5m)
            df_15m = pd.DataFrame(ohlcv_15m)
            
            if len(df_5m) < 20 or len(df_15m) < 10:  # 要件緩和
                return signals
            
            # EMA計算（改善版パラメータ）
            ema_5 = df_5m['close'].ewm(span=self.scalping_config["EMA_ULTRA_SHORT"]).mean()
            ema_12 = df_5m['close'].ewm(span=self.scalping_config["EMA_SHORT"]).mean()
            ema_26 = df_5m['close'].ewm(span=self.scalping_config["EMA_MEDIUM"]).mean()
            
            # 現在値
            current_ema_5 = ema_5.iloc[-1]
            current_ema_12 = ema_12.iloc[-1]
            current_ema_26 = ema_26.iloc[-1]
            
            # 15分足でのトレンド確認（緩和）
            df_15m_ema_12 = df_15m['close'].ewm(span=12).mean().iloc[-1]
            
            # ブレイクアウト条件（緩和）
            breakout_strength = 0.0
            direction = None
            
            # 🚀 強気ブレイクアウト（条件緩和）
            if (current_price > current_ema_5 and current_ema_5 > current_ema_12 and
                current_price > df_15m_ema_12 * 0.999):  # 1.001 → 0.999（緩和）
                
                breakout_strength = min((current_price - current_ema_12) / current_ema_12 * 100 / 0.3, 1.0)  # 0.5 → 0.3（感度向上）
                direction = "BUY"
                
            # 🚀 弱気ブレイクアウト（条件緩和）
            elif (current_price < current_ema_5 and current_ema_5 < current_ema_12 and
                  current_price < df_15m_ema_12 * 1.001):  # 0.999 → 1.001（緩和）
                
                breakout_strength = min((current_ema_12 - current_price) / current_ema_12 * 100 / 0.3, 1.0)  # 0.5 → 0.3（感度向上）
                direction = "SELL"
            
            if direction and breakout_strength >= self.scalping_config["MIN_SIGNAL_STRENGTH"]:
                
                # 価格設定（改善版）
                if direction == "BUY":
                    take_profit_price = current_price * (1 + self.scalping_config["TARGET_PROFIT_PCT"] / 100)
                    stop_loss_price = current_price * (1 - self.scalping_config["STOP_LOSS_PCT"] / 100)
                else:
                    take_profit_price = current_price * (1 - self.scalping_config["TARGET_PROFIT_PCT"] / 100)
                    stop_loss_price = current_price * (1 + self.scalping_config["STOP_LOSS_PCT"] / 100)
                
                # 緊急度計算（ブレイクアウトは高緊急度）
                urgency = min(breakout_strength + 0.2, 1.0)  # 0.3 → 0.2（緩和）
                
                signal = ScalpingSignal(
                    symbol=symbol,
                    signal_type=ScalpingSignalType.MOMENTUM_BREAKOUT,
                    direction=direction,
                    urgency=urgency,
                    strength=breakout_strength,
                    entry_price=current_price,
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                    expected_profit_pct=self.scalping_config["TARGET_PROFIT_PCT"],
                    risk_pct=self.scalping_config["STOP_LOSS_PCT"],
                    max_hold_seconds=self.scalping_config["MAX_HOLD_SECONDS"],
                    market_condition=market_condition,
                    timestamp=current_time
                )
                signals.append(signal)
        
        except Exception as e:
            self.logger.warning(f"モメンタムブレイクアウト分析エラー {symbol}: {e}")
        
        return signals
    
    def _analyze_quick_mean_reversion(self, symbol: str, ohlcv_5m: List, current_price: float,
                                    market_condition: MarketMicrostructure) -> List[ScalpingSignal]:
        """短期平均回帰分析（改善版）"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        try:
            df = pd.DataFrame(ohlcv_5m)
            if len(df) < 15:  # 20 → 15（緩和）
                return signals
            
            # 短期移動平均
            sma_8 = df['close'].rolling(window=8).mean().iloc[-1]   # 5 → 8（安定化）
            sma_20 = df['close'].rolling(window=20).mean().iloc[-1] # 10 → 20（安定化）
            
            # 乖離計算
            deviation_from_sma8 = (current_price - sma_8) / sma_8 * 100
            deviation_from_sma20 = (current_price - sma_20) / sma_20 * 100
            
            # RSI（改善版）
            rsi = self._calculate_rsi(df['close'], self.scalping_config["RSI_PERIOD"])
            current_rsi = rsi.iloc[-1] if not rsi.isna().iloc[-1] else 50
            
            # 平均回帰条件（緩和）
            reversion_strength = 0.0
            direction = None
            
            # 🚀 買われすぎからの回帰（条件緩和）
            if (current_rsi > self.scalping_config["RSI_OVERBOUGHT"] and
                deviation_from_sma8 > 0.2):  # 0.3% → 0.2%（感度向上）
                
                reversion_strength = min((current_rsi - 50) / 50 + abs(deviation_from_sma8) / 1.5, 1.0)  # 2 → 1.5（感度向上）
                direction = "SELL"
                
            # 🚀 売られすぎからの回帰（条件緩和）
            elif (current_rsi < self.scalping_config["RSI_OVERSOLD"] and
                  deviation_from_sma8 < -0.2):  # -0.3% → -0.2%（感度向上）
                
                reversion_strength = min((50 - current_rsi) / 50 + abs(deviation_from_sma8) / 1.5, 1.0)  # 2 → 1.5（感度向上）
                direction = "BUY"
            
            if direction and reversion_strength >= self.scalping_config["MIN_SIGNAL_STRENGTH"]:
                
                # 価格設定（改善版）
                if direction == "BUY":
                    take_profit_price = current_price * (1 + self.scalping_config["TARGET_PROFIT_PCT"] / 100)
                    stop_loss_price = current_price * (1 - self.scalping_config["STOP_LOSS_PCT"] / 100)
                else:
                    take_profit_price = current_price * (1 - self.scalping_config["TARGET_PROFIT_PCT"] / 100)
                    stop_loss_price = current_price * (1 + self.scalping_config["STOP_LOSS_PCT"] / 100)
                
                # 緊急度（平均回帰は中程度）
                urgency = min(reversion_strength + 0.05, 0.9)  # 0.1 → 0.05（緩和）
                
                signal = ScalpingSignal(
                    symbol=symbol,
                    signal_type=ScalpingSignalType.MEAN_REVERSION_QUICK,
                    direction=direction,
                    urgency=urgency,
                    strength=reversion_strength,
                    entry_price=current_price,
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                    expected_profit_pct=self.scalping_config["TARGET_PROFIT_PCT"],
                    risk_pct=self.scalping_config["STOP_LOSS_PCT"],
                    max_hold_seconds=self.scalping_config["MAX_HOLD_SECONDS"] // 2,  # 短めの保有
                    market_condition=market_condition,
                    timestamp=current_time
                )
                signals.append(signal)
        
        except Exception as e:
            self.logger.warning(f"短期平均回帰分析エラー {symbol}: {e}")
        
        return signals
    
    def _analyze_volatility_expansion(self, symbol: str, ohlcv_5m: List, current_price: float,
                                    market_condition: MarketMicrostructure) -> List[ScalpingSignal]:
        """ボラティリティ拡大分析（改善版）"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        try:
            df = pd.DataFrame(ohlcv_5m)
            if len(df) < 15:  # 20 → 15（緩和）
                return signals
            
            # ボラティリティ計算
            returns = df['close'].pct_change().dropna()
            current_volatility = returns.tail(3).std()  # 5 → 3（短期化）
            avg_volatility = returns.tail(15).std()     # 20 → 15（短期化）
            
            # 出来高急増
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(15).mean()  # 20 → 15（短期化）
            volume_surge = current_volume / avg_volume if avg_volume > 0 else 1
            
            # ATR（短期）
            atr = self._calculate_atr(df, 5)
            recent_range = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
            
            # 🚀 ボラティリティ拡大条件（大幅緩和）
            expansion_strength = 0.0
            direction = None
            
            if (current_volatility > avg_volatility * 1.5 and  # 2倍 → 1.5倍（緩和）
                volume_surge > self.scalping_config["MIN_VOLUME_RATIO"] and
                recent_range > atr * 1.2):  # 1.5 → 1.2（緩和）
                
                expansion_strength = min((current_volatility / avg_volatility - 1) * 0.6 + 
                                       (volume_surge - 1) * 0.4, 1.0)  # より寛大な計算
                
                # 方向性判定（直近の動き）
                recent_change = (current_price - df['close'].iloc[-3]) / df['close'].iloc[-3] * 100  # -5 → -3（短期化）
                
                if recent_change > 0.15:  # 0.2% → 0.15%（感度向上）
                    direction = "BUY"  # 上昇継続
                elif recent_change < -0.15:  # -0.2% → -0.15%（感度向上）
                    direction = "SELL"  # 下落継続
            
            if direction and expansion_strength >= self.scalping_config["MIN_SIGNAL_STRENGTH"]:
                
                # 価格設定（ボラティリティに応じて調整）
                volatility_multiplier = min(current_volatility / 0.005, 2.0)  # 0.01 → 0.005（感度向上）
                
                if direction == "BUY":
                    take_profit_price = current_price * (1 + self.scalping_config["TARGET_PROFIT_PCT"] / 100 * volatility_multiplier)
                    stop_loss_price = current_price * (1 - self.scalping_config["STOP_LOSS_PCT"] / 100 * volatility_multiplier)
                else:
                    take_profit_price = current_price * (1 - self.scalping_config["TARGET_PROFIT_PCT"] / 100 * volatility_multiplier)
                    stop_loss_price = current_price * (1 + self.scalping_config["STOP_LOSS_PCT"] / 100 * volatility_multiplier)
                
                urgency = min(expansion_strength + 0.15, 1.0)  # 0.2 → 0.15（緩和）
                
                signal = ScalpingSignal(
                    symbol=symbol,
                    signal_type=ScalpingSignalType.VOLATILITY_EXPANSION,
                    direction=direction,
                    urgency=urgency,
                    strength=expansion_strength,
                    entry_price=current_price,
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                    expected_profit_pct=self.scalping_config["TARGET_PROFIT_PCT"] * volatility_multiplier,
                    risk_pct=self.scalping_config["STOP_LOSS_PCT"] * volatility_multiplier,
                    max_hold_seconds=self.scalping_config["MAX_HOLD_SECONDS"] // 2,  # 短期
                    market_condition=market_condition,
                    timestamp=current_time
                )
                signals.append(signal)
        
        except Exception as e:
            self.logger.warning(f"ボラティリティ拡大分析エラー {symbol}: {e}")
        
        return signals
    
    def _analyze_order_flow(self, symbol: str, ohlcv_5m: List, current_price: float,
                          market_condition: MarketMicrostructure) -> List[ScalpingSignal]:
        """注文フロー分析（改善版）"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        try:
            df = pd.DataFrame(ohlcv_5m)
            if len(df) < 8:  # 10 → 8（緩和）
                return signals
            
            # OBV（On Balance Volume）簡易版
            obv = []
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.append(df['volume'].iloc[i])
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.append(-df['volume'].iloc[i])
                else:
                    obv.append(0)
            
            obv_series = pd.Series(obv)
            obv_sma = obv_series.rolling(window=3).mean()  # 5 → 3（短期化）
            
            # 価格・出来高の関係分析
            price_changes = df['close'].diff().dropna()
            volume_weighted_price = (df['volume'] * df['close']).rolling(window=3).sum() / df['volume'].rolling(window=3).sum()  # 5 → 3
            
            current_vwap = volume_weighted_price.iloc[-1]
            
            # フロー強度計算（緩和）
            flow_strength = 0.0
            direction = None
            
            if len(obv_sma) >= 3:  # 5 → 3（緩和）
                recent_obv = obv_sma.tail(2).mean()  # 3 → 2（短期化）
                
                # 🚀 上昇フロー（条件緩和）
                if (recent_obv > 0 and current_price > current_vwap * 0.9995):  # 1.001 → 0.9995（大幅緩和）
                    flow_strength = min(abs(recent_obv) / df['volume'].tail(3).mean() * 0.7, 1.0)  # 0.5 → 0.7（感度向上）
                    direction = "BUY"
                    
                # 🚀 下降フロー（条件緩和）
                elif (recent_obv < 0 and current_price < current_vwap * 1.0005):  # 0.999 → 1.0005（大幅緩和）
                    flow_strength = min(abs(recent_obv) / df['volume'].tail(3).mean() * 0.7, 1.0)  # 0.5 → 0.7（感度向上）
                    direction = "SELL"
            
            if direction and flow_strength >= self.scalping_config["MIN_SIGNAL_STRENGTH"]:
                
                # 価格設定
                if direction == "BUY":
                    take_profit_price = current_price * (1 + self.scalping_config["TARGET_PROFIT_PCT"] / 100)
                    stop_loss_price = current_price * (1 - self.scalping_config["STOP_LOSS_PCT"] / 100)
                else:
                    take_profit_price = current_price * (1 - self.scalping_config["TARGET_PROFIT_PCT"] / 100)
                    stop_loss_price = current_price * (1 + self.scalping_config["STOP_LOSS_PCT"] / 100)
                
                urgency = min(flow_strength + 0.1, 0.95)  # 0.15 → 0.1（緩和）
                
                signal = ScalpingSignal(
                    symbol=symbol,
                    signal_type=ScalpingSignalType.ORDER_FLOW,
                    direction=direction,
                    urgency=urgency,
                    strength=flow_strength,
                    entry_price=current_price,
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                    expected_profit_pct=self.scalping_config["TARGET_PROFIT_PCT"],
                    risk_pct=self.scalping_config["STOP_LOSS_PCT"],
                    max_hold_seconds=self.scalping_config["MAX_HOLD_SECONDS"],
                    market_condition=market_condition,
                    timestamp=current_time
                )
                signals.append(signal)
        
        except Exception as e:
            self.logger.warning(f"注文フロー分析エラー {symbol}: {e}")
        
        return signals
    
    def _filter_and_rank_scalping_signals(self, signals: List[ScalpingSignal]) -> List[ScalpingSignal]:
        """スキャルピングシグナルフィルタリング・ランキング（改善版）"""
        
        # フィルタリング（大幅緩和）
        filtered = []
        for signal in signals:
            # 最小強度・緊急度チェック（緩和済み）
            if signal.strength < self.scalping_config["MIN_SIGNAL_STRENGTH"]:
                continue
            if signal.urgency < self.scalping_config["MIN_URGENCY"]:
                continue
            
            # 🚀 リスクリワード比チェック（大幅緩和）
            risk_reward = signal.expected_profit_pct / signal.risk_pct
            if risk_reward < 0.3:  # 0.5 → 0.3（大幅緩和）
                continue
            
            # 🚀 市場条件チェック（緩和）
            # 低流動性でも取引許可
            # if signal.market_condition == MarketMicrostructure.LOW_LIQUIDITY:
            #     continue
            
            filtered.append(signal)
        
        # ランキング（緊急度 × 強度でソート）
        filtered.sort(key=lambda x: x.urgency * x.strength, reverse=True)
        
        return filtered
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """ATR計算"""
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        return atr if not pd.isna(atr) else df['close'].iloc[-1] * 0.01
    
    async def execute_scalping_trade(self, signal: ScalpingSignal) -> bool:
        """スキャルピング取引実行（改善版）"""
        
        try:
            current_price = await self.data_source.get_current_price(signal.symbol)
            current_time = self.data_source.get_current_time()
            
            # 🚀 スリッページ確認（緩和）
            price_drift = abs(current_price - signal.entry_price) / signal.entry_price * 100
            if price_drift > self.scalping_config["SLIPPAGE_BUFFER"]:
                self.logger.warning(f"スリッページ過大 {signal.symbol}: {price_drift:.3f}%")
                return False
            
            # ポジションサイズ計算（積極化）
            risk_per_trade = 1000 * (self.scalping_config["POSITION_SIZE_PCT"] / 100)  # 8%
            trade_amount = min(risk_per_trade, self.scalping_config["MIN_TRADE_SIZE_USD"] * 3)  # 積極化
            quantity = trade_amount / current_price
            
            # エントリー実行
            order_result = await self.data_source.place_order(
                symbol=signal.symbol,
                side=signal.direction,
                order_type="MARKET",
                quantity=quantity
            )
            
            if order_result.get("status") == "FILLED":
                # 実際の約定価格（スリッページ計算）
                filled_price = order_result.get("price", current_price)
                slippage_pct = abs(filled_price - signal.entry_price) / signal.entry_price * 100
                
                # ポジション記録
                position = ScalpingPosition(
                    signal=signal,
                    entry_time=current_time,
                    entry_price=signal.entry_price,
                    quantity=quantity,
                    filled_price=filled_price,
                    slippage_pct=slippage_pct
                )
                
                self.active_positions[signal.symbol] = position
                
                self.logger.info(f"⚡ スキャルピング改善版実行: {signal.signal_type.value}")
                self.logger.info(f"   {signal.symbol} {signal.direction} ${current_price:.4f}")
                self.logger.info(f"   目標利益{signal.expected_profit_pct:.2f}% 緊急度{signal.urgency:.2f}")
                
                return True
            else:
                self.logger.error(f"❌ スキャルピング実行失敗: {signal.symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ スキャルピング実行エラー: {str(e)}")
            return False
    
    async def manage_scalping_positions(self) -> List[TradeResult]:
        """スキャルピングポジション管理（改善版）"""
        
        trades = []
        current_time = self.data_source.get_current_time()
        positions_to_close = []
        
        for symbol, position in self.active_positions.items():
            if not position.is_active:
                continue
            
            try:
                current_price = await self.data_source.get_current_price(symbol)
                signal = position.signal
                
                # 経過時間計算
                seconds_elapsed = (current_time - position.entry_time).total_seconds()
                position.seconds_held = int(seconds_elapsed)
                
                # 損益計算
                if signal.direction == "BUY":
                    unrealized_pnl = (current_price - position.filled_price) * position.quantity
                    profit_pct = (current_price - position.filled_price) / position.filled_price * 100
                else:
                    unrealized_pnl = (position.filled_price - current_price) * position.quantity
                    profit_pct = (position.filled_price - current_price) / position.filled_price * 100
                
                # 最高利益・最大ドローダウン更新
                position.peak_profit_pct = max(position.peak_profit_pct, profit_pct)
                if profit_pct < 0:
                    position.max_drawdown_pct = min(position.max_drawdown_pct, profit_pct)
                
                # 利確条件チェック
                if signal.direction == "BUY" and current_price >= signal.take_profit_price:
                    trade = await self._close_scalping_position(position, current_price, current_time, "利確達成")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                elif signal.direction == "SELL" and current_price <= signal.take_profit_price:
                    trade = await self._close_scalping_position(position, current_price, current_time, "利確達成")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                
                # ストップロス条件チェック
                elif signal.direction == "BUY" and current_price <= signal.stop_loss_price:
                    trade = await self._close_scalping_position(position, current_price, current_time, "ストップロス")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                elif signal.direction == "SELL" and current_price >= signal.stop_loss_price:
                    trade = await self._close_scalping_position(position, current_price, current_time, "ストップロス")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                
                # ブレイクイーブンストップ
                elif (position.peak_profit_pct >= self.scalping_config["BREAKEVEN_STOP_PCT"] and
                      profit_pct <= 0):
                    trade = await self._close_scalping_position(position, current_price, current_time, "ブレイクイーブン")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                
                # トレーリングストップ
                elif (position.peak_profit_pct >= self.scalping_config["TRAILING_STOP_PCT"] and
                      profit_pct <= position.peak_profit_pct - self.scalping_config["TRAILING_STOP_PCT"]):
                    trade = await self._close_scalping_position(position, current_price, current_time, "トレーリング")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                
                # 時間切れチェック
                elif seconds_elapsed >= signal.max_hold_seconds:
                    trade = await self._close_scalping_position(position, current_price, current_time, "時間切れ")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                
                # 強制決済チェック
                elif seconds_elapsed >= self.scalping_config["FORCE_EXIT_SECONDS"]:
                    trade = await self._close_scalping_position(position, current_price, current_time, "強制決済")
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
    
    async def _close_scalping_position(self, position: ScalpingPosition, current_price: float,
                                     current_time: datetime, exit_reason: str) -> Optional[TradeResult]:
        """スキャルピングポジション決済"""
        
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
                # 実際の決済価格
                exit_filled_price = order_result.get("price", current_price)
                
                # 損益計算（手数料込み）
                if signal.direction == "BUY":
                    profit_loss = (exit_filled_price - position.filled_price) * position.quantity
                else:
                    profit_loss = (position.filled_price - exit_filled_price) * position.quantity
                
                # 手数料差し引き
                trade_value = position.filled_price * position.quantity
                commission = trade_value * (self.scalping_config["COMMISSION_PCT"] / 100) * 2  # 往復
                profit_loss -= commission
                
                profit_pct = (profit_loss / trade_value) * 100
                hold_seconds = (current_time - position.entry_time).total_seconds()
                
                trade = TradeResult(
                    symbol=signal.symbol,
                    entry_time=position.entry_time,
                    exit_time=current_time,
                    side=signal.direction,
                    entry_price=position.filled_price,
                    exit_price=exit_filled_price,
                    quantity=position.quantity,
                    profit_loss=profit_loss,
                    profit_pct=profit_pct,
                    hold_hours=hold_seconds / 3600,
                    exit_reason=f"スキャルピング改善_{exit_reason}"
                )
                
                self.trade_history.append(trade)
                
                self.logger.info(f"⚡ スキャルピング改善版決済: {signal.symbol} {exit_reason}")
                self.logger.info(f"   エントリー: ${position.filled_price:.4f} → 決済: ${exit_filled_price:.4f}")
                self.logger.info(f"   利益: ${profit_loss:.2f} ({profit_pct:+.3f}%) 保有{hold_seconds:.0f}秒")
                
                return trade
            else:
                self.logger.error(f"❌ スキャルピング決済失敗: {signal.symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"スキャルピング決済エラー: {str(e)}")
            return None

class ImprovedScalpingBacktestSystem(AnnualBacktestSystem):
    """スキャルピング改善版バックテストシステム"""
    
    def __init__(self, config: Dict, historical_data: Dict[str, pd.DataFrame], 
                 symbols: List[str], start_date: datetime, end_date: datetime):
        super().__init__(config, historical_data, symbols, start_date, end_date)
        
        self.scalping_strategy = ImprovedScalpingStrategy(
            config, WindowsDataSource(ExecutionMode.BACKTEST, historical_data), ExecutionMode.BACKTEST
        )
        
        # スキャルピング改善版専用設定
        self.enhanced_config.update({
            "STRATEGY_NAME": "スキャルピング改善版戦略",
            "EXPECTED_ANNUAL_RETURN": 20.0,  # 0% → 20% 目標
            "MAX_POSITIONS": 10,             # 最大ポジション数（積極化）
            "SCAN_INTERVAL": 1,              # 1時間ごとスキャン
        })
    
    async def _execute_annual_backtest(self):
        """スキャルピング改善版年間バックテスト実行"""
        
        capital = self.enhanced_config["INITIAL_CAPITAL"]
        
        # 時間ステップ（1時間ごと）
        timestamps = list(self.historical_data[self.symbols[0]].index[::1])
        
        for i, timestamp in enumerate(timestamps):
            try:
                # データソース時刻設定
                self.scalping_strategy.data_source.set_current_time(timestamp)
                current_portfolio_value = capital
                
                # 既存スキャルピングポジション管理（高頻度）
                trades = await self.scalping_strategy.manage_scalping_positions()
                for trade in trades:
                    capital += trade.profit_loss + (trade.entry_price * trade.quantity)
                    self.trades.append(trade)
                    current_portfolio_value += trade.profit_loss
                
                # 新規スキャルピングシグナル検索（積極的）
                active_positions = len(self.scalping_strategy.active_positions)
                if active_positions < self.enhanced_config["MAX_POSITIONS"]:
                    
                    signals = await self.scalping_strategy.scan_scalping_opportunities(self.symbols)
                    
                    for signal in signals[:8]:  # TOP8実行（非常に積極的）
                        if active_positions >= self.enhanced_config["MAX_POSITIONS"]:
                            break
                        
                        # 重複チェック
                        if signal.symbol not in self.scalping_strategy.active_positions:
                            required_capital = signal.entry_price * 20  # 最小$20
                            
                            if capital > required_capital:
                                success = await self.scalping_strategy.execute_scalping_trade(signal)
                                if success:
                                    capital -= required_capital
                                    active_positions += 1
                                    self.logger.info(f"⚡ {signal.symbol} スキャルピング改善版開始: {signal.signal_type.value}")
                
                # ポートフォリオ価値計算
                scalping_investment = sum([
                    pos.filled_price * pos.quantity for pos in self.scalping_strategy.active_positions.values()
                ])
                portfolio_value = capital + scalping_investment
                
                # 日次記録
                if i % 24 == 0:  # 24時間ごと
                    self.daily_portfolio.append({
                        'timestamp': timestamp,
                        'portfolio_value': portfolio_value,
                        'cash': capital,
                        'positions': len(self.scalping_strategy.active_positions),
                        'return_pct': ((portfolio_value - self.enhanced_config["INITIAL_CAPITAL"]) / 
                                     self.enhanced_config["INITIAL_CAPITAL"]) * 100
                    })
                
                # 進捗表示
                if i % 168 == 0:  # 週次
                    progress = (i / len(timestamps)) * 100
                    weeks = i // 168
                    active_positions = len(self.scalping_strategy.active_positions)
                    total_trades = len(self.trades)
                    self.logger.info(f"  進捗: {progress:.1f}% ({weeks}週経過) アクティブ:{active_positions} 取引数:{total_trades}")
                
            except Exception as e:
                self.logger.warning(f"タイムスタンプ {timestamp} でエラー: {str(e)}")
                continue

# メイン実行関数
async def run_improved_scalping_backtest():
    """スキャルピング改善版戦略バックテスト実行"""
    
    logger = logging.getLogger(__name__)
    logger.info("⚡ スキャルピング改善版戦略 1年間バックテスト開始")
    
    # スキャルピング改善版用設定
    config = {
        "STRATEGY_TYPE": "IMPROVED_SCALPING",
        "SCAN_INTERVAL_SECONDS": 60,
        "MAX_POSITIONS": 10,
        "POSITION_SIZE_PCT": 8.0,
        "TARGET_PROFIT_PCT": 0.5,      # 🎯 主要改善点
        "STOP_LOSS_PCT": 0.6,
        "MAX_HOLD_SECONDS": 600,
        "MIN_SIGNAL_STRENGTH": 0.4,    # 🎯 主要改善点
        "MIN_URGENCY": 0.5,            # 🎯 主要改善点
        "MIN_VOLUME_RATIO": 1.0,       # 🎯 主要改善点
        "COMMISSION_PCT": 0.02
    }
    
    # 1年間バックテストシステム作成
    logger.info("📊 スキャルピング改善版バックテストシステム作成中...")
    annual_system = await WindowsUnifiedSystemFactory.create_annual_backtest_system(
        config, use_real_data=True
    )
    
    # スキャルピング改善版システムに変換
    scalping_improved_system = ImprovedScalpingBacktestSystem(
        config, annual_system.historical_data, annual_system.symbols,
        annual_system.start_date, annual_system.end_date
    )
    
    # バックテスト実行
    logger.info("⚡ スキャルピング改善版バックテスト実行中...")
    results = await scalping_improved_system.run_annual_comprehensive_backtest()
    
    # 結果表示
    print("\n" + "="*80)
    print("⚡ スキャルピング改善版戦略 1年間バックテスト完了")
    print("🚀 改善実装: シグナル強度0.6→0.4, 緊急度0.7→0.5, 5分足採用, 利確0.3%→0.5%")
    print("="*80)
    
    perf = results['performance_metrics']
    print(f"\n📈 パフォーマンス結果:")
    print(f"   戦略タイプ: スキャルピング改善版（超高頻度取引）")
    print(f"   総取引数: {perf['total_trades']}")
    print(f"   勝率: {perf['win_rate']:.1f}%")
    print(f"   総リターン: {perf['total_return']:+.1f}%")
    print(f"   最大ドローダウン: {perf['max_drawdown']:.1f}%")
    print(f"   シャープレシオ: {perf['sharpe_ratio']:.2f}")
    print(f"   プロフィットファクター: {perf['profit_factor']:.2f}")
    
    # 改善効果分析
    original_return = 0.0
    original_trades = 0
    improvement_return = perf['total_return'] - original_return
    improvement_trades = perf['total_trades'] - original_trades
    
    print(f"\n🚀 改善効果分析:")
    print(f"   改善前: +{original_return:.1f}% (取引{original_trades}回)")
    print(f"   改善後: {perf['total_return']:+.1f}% (取引{perf['total_trades']}回)")
    print(f"   リターン改善: {improvement_return:+.1f}%")
    print(f"   取引数改善: +{improvement_trades}回")
    
    # 目標達成評価
    target_monthly = 10.0  # 月10%目標
    target_annual = target_monthly * 12  # 年120%
    achievement_rate = (perf['total_return'] / target_annual) * 100
    
    print(f"\n🎯 目標達成度:")
    print(f"   月10%目標 (年120%) vs 実績年{perf['total_return']:+.1f}%")
    print(f"   達成率: {achievement_rate:.1f}%")
    
    if perf['total_return'] >= 15.0:
        print("✅ 改善目標（年15-25%）達成")
    elif perf['total_return'] > 0:
        print("⚠️ 改善目標未達成だがプラス転換")
    else:
        print("❌ 改善目標未達成")
    
    # 取引頻度分析
    if perf['total_trades'] > 0:
        daily_trades = perf['total_trades'] / 365
        print(f"\n📈 取引頻度分析:")
        print(f"   1日平均取引数: {daily_trades:.1f}")
        print(f"   改善目標: 200-500取引/年 → {perf['total_trades']}取引/年")
        
        if daily_trades >= 1.0:
            print("✅ 高頻度取引改善成功")
        elif daily_trades >= 0.3:
            print("⚠️ 高頻度取引改善中（やや低め）")
        else:
            print("❌ 高頻度取引改善不足")
    
    return results

async def main():
    """メイン実行"""
    await run_improved_scalping_backtest()

if __name__ == "__main__":
    asyncio.run(main())