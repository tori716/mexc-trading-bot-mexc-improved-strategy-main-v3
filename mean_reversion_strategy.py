#!/usr/bin/env python3
"""
平均回帰戦略 - 調査報告書準拠実装
期待利益率: 年間12-22%（勝率65-80%）
統計的回帰を利用したレンジ相場特化戦略
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
        logging.FileHandler('mean_reversion_strategy.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ReversionType(Enum):
    """平均回帰タイプ"""
    BOLLINGER_REVERSION = "bollinger_reversion"      # ボリンジャーバンド回帰
    MA_REVERSION = "ma_reversion"                    # 移動平均回帰
    RSI_REVERSION = "rsi_reversion"                  # RSI極値回帰
    STATISTICAL_REVERSION = "statistical_reversion"  # 統計的回帰

class MarketState(Enum):
    """市場状態"""
    RANGING = "ranging"      # レンジ相場
    TRENDING_UP = "trending_up"    # 上昇トレンド
    TRENDING_DOWN = "trending_down"  # 下降トレンド
    UNKNOWN = "unknown"      # 不明

@dataclass
class ReversionSignal:
    """平均回帰シグナル定義"""
    symbol: str
    reversion_type: ReversionType
    direction: str   # "BUY" or "SELL"
    strength: float  # 回帰強度（0.0-1.0）
    confidence: float  # 信頼度（0.0-1.0）
    current_price: float
    target_price: float  # 回帰目標価格
    stop_loss_price: float
    deviation_pct: float  # 現在の乖離率
    reversion_probability: float  # 回帰確率
    timestamp: datetime
    expected_duration_hours: float  # 期待保有時間

@dataclass
class ReversionPosition:
    """平均回帰ポジション"""
    signal: ReversionSignal
    entry_time: datetime
    entry_price: float
    quantity: float
    target_reached: bool = False
    max_favorable_move: float = 0.0  # 最大有利方向移動
    max_adverse_move: float = 0.0    # 最大不利方向移動
    is_active: bool = True

class MeanReversionStrategy:
    """平均回帰戦略（調査報告書準拠）"""
    
    def __init__(self, config: Dict, data_source: WindowsDataSource, execution_mode: ExecutionMode):
        self.config = config
        self.data_source = data_source
        self.execution_mode = execution_mode
        self.logger = logging.getLogger(__name__)
        
        # 平均回帰管理
        self.active_positions: Dict[str, ReversionPosition] = {}
        self.trade_history = []
        self.market_state_history = {}  # 市場状態履歴
        
        # 調査報告書準拠の設定（保守的）
        self.reversion_config = {
            # 統計的回帰設定
            "BB_PERIOD": 20,                    # ボリンジャーバンド期間
            "BB_STD_DEV": 2.0,                  # ボリンジャーバンド標準偏差
            "MIN_DEVIATION_PCT": 1.5,           # 最小乖離率（1.5%）
            "MAX_DEVIATION_PCT": 8.0,           # 最大乖離率（8%）
            "REVERSION_THRESHOLD": 0.7,         # 回帰確率閾値
            
            # 移動平均設定
            "MA_SHORT": 10,                     # 短期移動平均
            "MA_MEDIUM": 20,                    # 中期移動平均
            "MA_LONG": 50,                      # 長期移動平均
            "MA_DEVIATION_THRESHOLD": 2.5,      # 移動平均乖離閾値（2.5%）
            
            # RSI設定
            "RSI_PERIOD": 14,                   # RSI期間
            "RSI_OVERSOLD": 25,                 # RSI売られすぎ（より極端）
            "RSI_OVERBOUGHT": 75,               # RSI買われすぎ（より極端）
            "RSI_EXTREME_OVERSOLD": 20,         # RSI極端売られすぎ
            "RSI_EXTREME_OVERBOUGHT": 80,       # RSI極端買われすぎ
            
            # レンジ相場判定
            "TREND_THRESHOLD": 0.05,            # トレンド閾値（5%）
            "RANGE_CONFIRMATION_PERIOD": 30,    # レンジ確認期間
            "MIN_RANGE_VOLATILITY": 0.01,       # 最小レンジボラティリティ
            "MAX_RANGE_VOLATILITY": 0.08,       # 最大レンジボラティリティ
            
            # エントリー・エグジット設定
            "TARGET_PROFIT_PCT": 2.0,           # 目標利益率（2%、保守的）
            "STOP_LOSS_PCT": 3.0,               # ストップロス率（3%）
            "PARTIAL_PROFIT_PCT": 1.0,          # 部分利確率（1%）
            "REVERSION_TARGET_PCT": 0.8,        # 回帰目標達成率（80%）
            
            # ポジション管理
            "MAX_POSITIONS": 4,                 # 最大ポジション数（保守的）
            "POSITION_SIZE_PCT": 8.0,           # ポジションサイズ（8%、小さめ）
            "MIN_VOLUME_RATIO": 1.1,            # 最小出来高倍率
            "MAX_CORRELATION": 0.6,             # 最大相関係数
            
            # 時間管理
            "MAX_HOLD_HOURS": 48,               # 最大保有時間（2日）
            "MIN_HOLD_MINUTES": 15,             # 最小保有時間（15分）
            "REVERSION_WINDOW_HOURS": 12,       # 回帰期待時間
        }
    
    async def analyze_reversion_signals(self, symbols: List[str]) -> List[ReversionSignal]:
        """平均回帰シグナル分析"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        for symbol in symbols:
            try:
                # OHLCV データ取得
                ohlcv_data = await self.data_source.get_ohlcv(symbol, "60m", 100)
                
                if not ohlcv_data or len(ohlcv_data) < 60:
                    continue
                
                df = pd.DataFrame(ohlcv_data)
                current_price = await self.data_source.get_current_price(symbol)
                
                # 市場状態判定
                market_state = self._determine_market_state(df)
                
                # レンジ相場でのみ平均回帰戦略を適用
                if market_state != MarketState.RANGING:
                    continue
                
                # ボリンジャーバンド回帰分析
                bb_signals = self._analyze_bollinger_reversion(symbol, df, current_price)
                signals.extend(bb_signals)
                
                # 移動平均回帰分析
                ma_signals = self._analyze_ma_reversion(symbol, df, current_price)
                signals.extend(ma_signals)
                
                # RSI極値回帰分析
                rsi_signals = self._analyze_rsi_reversion(symbol, df, current_price)
                signals.extend(rsi_signals)
                
                # 統計的回帰分析
                stat_signals = self._analyze_statistical_reversion(symbol, df, current_price)
                signals.extend(stat_signals)
                
            except Exception as e:
                self.logger.warning(f"平均回帰分析エラー {symbol}: {e}")
                continue
        
        # シグナルフィルタリング・ランキング
        filtered_signals = self._filter_and_rank_reversion_signals(signals)
        
        if filtered_signals:
            self.logger.info(f"📈 平均回帰シグナル検出: {len(filtered_signals)}件")
            for i, signal in enumerate(filtered_signals[:3]):
                self.logger.info(f"   {i+1}. {signal.symbol} {signal.direction} "
                               f"乖離{signal.deviation_pct:.1f}% 回帰確率{signal.reversion_probability:.1f}")
        
        return filtered_signals
    
    def _determine_market_state(self, df: pd.DataFrame) -> MarketState:
        """市場状態判定"""
        
        try:
            if len(df) < self.reversion_config["RANGE_CONFIRMATION_PERIOD"]:
                return MarketState.UNKNOWN
            
            # 価格データ
            prices = df['close'].tail(self.reversion_config["RANGE_CONFIRMATION_PERIOD"])
            
            # トレンド分析（簡単な線形回帰）
            x = np.array(range(len(prices)))
            y = np.array(prices)
            n = len(x)
            
            # 線形回帰の傾き計算
            if n > 1:
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                trend_slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
                trend_pct = (trend_slope * len(prices)) / prices.iloc[0] * 100
            else:
                trend_pct = 0
            
            # ボラティリティ分析
            returns = prices.pct_change().dropna()
            volatility = returns.std()
            
            # 価格レンジ分析
            price_range = (prices.max() - prices.min()) / prices.mean()
            
            # 市場状態判定
            if abs(trend_pct) > self.reversion_config["TREND_THRESHOLD"]:
                if trend_pct > 0:
                    return MarketState.TRENDING_UP
                else:
                    return MarketState.TRENDING_DOWN
            elif (self.reversion_config["MIN_RANGE_VOLATILITY"] <= volatility <= 
                  self.reversion_config["MAX_RANGE_VOLATILITY"] and 
                  price_range < 0.15):  # 15%以内のレンジ
                return MarketState.RANGING
            else:
                return MarketState.UNKNOWN
                
        except Exception as e:
            self.logger.warning(f"市場状態判定エラー: {e}")
            return MarketState.UNKNOWN
    
    def _analyze_bollinger_reversion(self, symbol: str, df: pd.DataFrame, current_price: float) -> List[ReversionSignal]:
        """ボリンジャーバンド回帰分析"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        try:
            # ボリンジャーバンド計算
            bb_period = self.reversion_config["BB_PERIOD"]
            bb_std = self.reversion_config["BB_STD_DEV"]
            
            sma = df['close'].rolling(window=bb_period).mean()
            std = df['close'].rolling(window=bb_period).std()
            bb_upper = sma + (std * bb_std)
            bb_lower = sma - (std * bb_std)
            bb_middle = sma
            
            current_sma = bb_middle.iloc[-1]
            current_upper = bb_upper.iloc[-1]
            current_lower = bb_lower.iloc[-1]
            
            if pd.isna(current_sma) or pd.isna(current_upper) or pd.isna(current_lower):
                return signals
            
            # バンド乖離計算
            if current_price > current_upper:
                # 上バンド突破（売りシグナル）
                deviation_pct = ((current_price - current_upper) / current_upper) * 100
                direction = "SELL"
                target_price = current_sma  # 中央線回帰
                stop_loss_price = current_price * (1 + self.reversion_config["STOP_LOSS_PCT"] / 100)
                
            elif current_price < current_lower:
                # 下バンド突破（買いシグナル）
                deviation_pct = ((current_lower - current_price) / current_lower) * 100
                direction = "BUY"
                target_price = current_sma  # 中央線回帰
                stop_loss_price = current_price * (1 - self.reversion_config["STOP_LOSS_PCT"] / 100)
                
            else:
                return signals  # バンド内では取引しない
            
            # 乖離が範囲内かチェック
            min_dev = self.reversion_config["MIN_DEVIATION_PCT"]
            max_dev = self.reversion_config["MAX_DEVIATION_PCT"]
            
            if min_dev <= deviation_pct <= max_dev:
                
                # 回帰確率計算（統計的）
                reversion_probability = self._calculate_reversion_probability(df, current_price, current_sma)
                
                if reversion_probability >= self.reversion_config["REVERSION_THRESHOLD"]:
                    
                    # 強度計算
                    strength = min(deviation_pct / max_dev, 1.0)
                    confidence = reversion_probability
                    
                    signal = ReversionSignal(
                        symbol=symbol,
                        reversion_type=ReversionType.BOLLINGER_REVERSION,
                        direction=direction,
                        strength=strength,
                        confidence=confidence,
                        current_price=current_price,
                        target_price=target_price,
                        stop_loss_price=stop_loss_price,
                        deviation_pct=deviation_pct,
                        reversion_probability=reversion_probability,
                        timestamp=current_time,
                        expected_duration_hours=self.reversion_config["REVERSION_WINDOW_HOURS"]
                    )
                    signals.append(signal)
        
        except Exception as e:
            self.logger.warning(f"ボリンジャーバンド回帰分析エラー {symbol}: {e}")
        
        return signals
    
    def _analyze_ma_reversion(self, symbol: str, df: pd.DataFrame, current_price: float) -> List[ReversionSignal]:
        """移動平均回帰分析"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        try:
            # 複数期間移動平均計算
            ma_short = df['close'].rolling(window=self.reversion_config["MA_SHORT"]).mean().iloc[-1]
            ma_medium = df['close'].rolling(window=self.reversion_config["MA_MEDIUM"]).mean().iloc[-1]
            ma_long = df['close'].rolling(window=self.reversion_config["MA_LONG"]).mean().iloc[-1]
            
            if pd.isna(ma_medium):
                return signals
            
            # 中期移動平均からの乖離分析
            deviation_pct = abs((current_price - ma_medium) / ma_medium) * 100
            threshold = self.reversion_config["MA_DEVIATION_THRESHOLD"]
            
            if deviation_pct >= threshold:
                
                if current_price > ma_medium:
                    direction = "SELL"
                    target_price = ma_medium
                    stop_loss_price = current_price * (1 + self.reversion_config["STOP_LOSS_PCT"] / 100)
                else:
                    direction = "BUY"
                    target_price = ma_medium
                    stop_loss_price = current_price * (1 - self.reversion_config["STOP_LOSS_PCT"] / 100)
                
                # 移動平均の並び順で追加フィルタ
                ma_alignment_valid = False
                if direction == "BUY" and ma_short < ma_medium:  # 短期が中期を下回る
                    ma_alignment_valid = True
                elif direction == "SELL" and ma_short > ma_medium:  # 短期が中期を上回る
                    ma_alignment_valid = True
                
                if ma_alignment_valid:
                    
                    # 回帰確率計算
                    reversion_probability = self._calculate_reversion_probability(df, current_price, ma_medium)
                    
                    if reversion_probability >= self.reversion_config["REVERSION_THRESHOLD"]:
                        
                        strength = min(deviation_pct / (threshold * 2), 1.0)
                        confidence = reversion_probability * 0.9  # MA回帰は少し保守的
                        
                        signal = ReversionSignal(
                            symbol=symbol,
                            reversion_type=ReversionType.MA_REVERSION,
                            direction=direction,
                            strength=strength,
                            confidence=confidence,
                            current_price=current_price,
                            target_price=target_price,
                            stop_loss_price=stop_loss_price,
                            deviation_pct=deviation_pct,
                            reversion_probability=reversion_probability,
                            timestamp=current_time,
                            expected_duration_hours=self.reversion_config["REVERSION_WINDOW_HOURS"]
                        )
                        signals.append(signal)
        
        except Exception as e:
            self.logger.warning(f"移動平均回帰分析エラー {symbol}: {e}")
        
        return signals
    
    def _analyze_rsi_reversion(self, symbol: str, df: pd.DataFrame, current_price: float) -> List[ReversionSignal]:
        """RSI極値回帰分析"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        try:
            # RSI計算
            rsi = self._calculate_rsi(df['close'], self.reversion_config["RSI_PERIOD"])
            current_rsi = rsi.iloc[-1]
            
            if pd.isna(current_rsi):
                return signals
            
            # RSI極値判定
            direction = None
            strength = 0.0
            
            if current_rsi <= self.reversion_config["RSI_EXTREME_OVERSOLD"]:
                direction = "BUY"
                strength = (self.reversion_config["RSI_OVERSOLD"] - current_rsi) / self.reversion_config["RSI_OVERSOLD"]
            elif current_rsi >= self.reversion_config["RSI_EXTREME_OVERBOUGHT"]:
                direction = "SELL"
                strength = (current_rsi - self.reversion_config["RSI_OVERBOUGHT"]) / (100 - self.reversion_config["RSI_OVERBOUGHT"])
            
            if direction and strength > 0:
                
                # 目標価格設定（RSIの50付近を目標）
                ma_20 = df['close'].rolling(window=20).mean().iloc[-1]
                if not pd.isna(ma_20):
                    target_price = ma_20
                else:
                    target_price = current_price * (0.98 if direction == "BUY" else 1.02)
                
                if direction == "BUY":
                    stop_loss_price = current_price * (1 - self.reversion_config["STOP_LOSS_PCT"] / 100)
                    deviation_pct = (50 - current_rsi) / 50 * 100  # RSI 50からの乖離
                else:
                    stop_loss_price = current_price * (1 + self.reversion_config["STOP_LOSS_PCT"] / 100)
                    deviation_pct = (current_rsi - 50) / 50 * 100  # RSI 50からの乖離
                
                # 回帰確率計算（RSI用）
                reversion_probability = self._calculate_rsi_reversion_probability(rsi, current_rsi)
                
                if reversion_probability >= self.reversion_config["REVERSION_THRESHOLD"]:
                    
                    confidence = reversion_probability * strength
                    
                    signal = ReversionSignal(
                        symbol=symbol,
                        reversion_type=ReversionType.RSI_REVERSION,
                        direction=direction,
                        strength=strength,
                        confidence=confidence,
                        current_price=current_price,
                        target_price=target_price,
                        stop_loss_price=stop_loss_price,
                        deviation_pct=abs(deviation_pct),
                        reversion_probability=reversion_probability,
                        timestamp=current_time,
                        expected_duration_hours=self.reversion_config["REVERSION_WINDOW_HOURS"] * 0.8  # RSIは少し早め
                    )
                    signals.append(signal)
        
        except Exception as e:
            self.logger.warning(f"RSI回帰分析エラー {symbol}: {e}")
        
        return signals
    
    def _analyze_statistical_reversion(self, symbol: str, df: pd.DataFrame, current_price: float) -> List[ReversionSignal]:
        """統計的回帰分析"""
        
        signals = []
        current_time = self.data_source.get_current_time()
        
        try:
            # 統計的指標計算
            lookback = 50
            if len(df) < lookback:
                return signals
            
            prices = df['close'].tail(lookback)
            mean_price = prices.mean()
            std_price = prices.std()
            
            if std_price == 0:
                return signals
            
            # Z-スコア計算
            z_score = (current_price - mean_price) / std_price
            
            # 統計的極値判定（±2標準偏差以上）
            if abs(z_score) >= 2.0:
                
                if z_score > 2.0:
                    direction = "SELL"
                    target_price = mean_price
                    stop_loss_price = current_price * (1 + self.reversion_config["STOP_LOSS_PCT"] / 100)
                else:
                    direction = "BUY"
                    target_price = mean_price
                    stop_loss_price = current_price * (1 - self.reversion_config["STOP_LOSS_PCT"] / 100)
                
                deviation_pct = abs((current_price - mean_price) / mean_price) * 100
                strength = min(abs(z_score) / 4.0, 1.0)  # 4標準偏差で最大強度
                
                # 統計的回帰確率計算
                reversion_probability = self._calculate_statistical_reversion_probability(z_score)
                
                if reversion_probability >= self.reversion_config["REVERSION_THRESHOLD"]:
                    
                    confidence = reversion_probability
                    
                    signal = ReversionSignal(
                        symbol=symbol,
                        reversion_type=ReversionType.STATISTICAL_REVERSION,
                        direction=direction,
                        strength=strength,
                        confidence=confidence,
                        current_price=current_price,
                        target_price=target_price,
                        stop_loss_price=stop_loss_price,
                        deviation_pct=deviation_pct,
                        reversion_probability=reversion_probability,
                        timestamp=current_time,
                        expected_duration_hours=self.reversion_config["REVERSION_WINDOW_HOURS"]
                    )
                    signals.append(signal)
        
        except Exception as e:
            self.logger.warning(f"統計的回帰分析エラー {symbol}: {e}")
        
        return signals
    
    def _calculate_reversion_probability(self, df: pd.DataFrame, current_price: float, target_price: float) -> float:
        """回帰確率計算"""
        
        try:
            # 過去の類似状況での回帰実績を分析
            lookback = 100
            if len(df) < lookback:
                return 0.5  # デフォルト50%
            
            prices = df['close'].tail(lookback)
            target_distance = abs(current_price - target_price) / target_price
            
            # 類似の乖離状況を検索
            reversion_count = 0
            total_cases = 0
            
            for i in range(10, len(prices) - 10):
                price_at_i = prices.iloc[i]
                target_at_i = prices.iloc[i-10:i].mean()  # 過去10期間平均を目標とする
                distance_at_i = abs(price_at_i - target_at_i) / target_at_i
                
                # 類似の乖離状況かチェック
                if abs(distance_at_i - target_distance) < target_distance * 0.3:  # 30%以内の類似度
                    total_cases += 1
                    
                    # 次の10期間で目標に向かって回帰したかチェック
                    future_prices = prices.iloc[i+1:i+11]
                    if len(future_prices) >= 5:
                        # 少なくとも50%目標に近づいたか
                        closest_price = future_prices.iloc[np.argmin(np.abs(future_prices - target_at_i))]
                        if abs(closest_price - target_at_i) < abs(price_at_i - target_at_i) * 0.5:
                            reversion_count += 1
            
            if total_cases >= 5:
                probability = reversion_count / total_cases
                return min(max(probability, 0.3), 0.95)  # 30%-95%の範囲
            else:
                return 0.7  # デフォルト70%
                
        except Exception as e:
            self.logger.warning(f"回帰確率計算エラー: {e}")
            return 0.7
    
    def _calculate_rsi_reversion_probability(self, rsi_series: pd.Series, current_rsi: float) -> float:
        """RSI回帰確率計算"""
        
        try:
            # RSI極値からの回帰実績分析
            reversion_count = 0
            total_cases = 0
            
            for i in range(10, len(rsi_series) - 10):
                rsi_at_i = rsi_series.iloc[i]
                
                # 極値判定（現在と類似）
                is_extreme = False
                if current_rsi <= 25 and rsi_at_i <= 25:
                    is_extreme = True
                elif current_rsi >= 75 and rsi_at_i >= 75:
                    is_extreme = True
                
                if is_extreme:
                    total_cases += 1
                    
                    # 次の期間でRSI 50に向かって回帰したかチェック
                    future_rsi = rsi_series.iloc[i+1:i+11]
                    if len(future_rsi) >= 5:
                        if current_rsi <= 25:
                            # 売られすぎからの回復
                            if future_rsi.max() > 40:
                                reversion_count += 1
                        else:
                            # 買われすぎからの調整
                            if future_rsi.min() < 60:
                                reversion_count += 1
            
            if total_cases >= 3:
                probability = reversion_count / total_cases
                return min(max(probability, 0.4), 0.9)
            else:
                return 0.75  # RSIは一般的に回帰しやすい
                
        except Exception as e:
            return 0.75
    
    def _calculate_statistical_reversion_probability(self, z_score: float) -> float:
        """統計的回帰確率計算"""
        
        # 正規分布の性質を利用
        # ±2標準偏差: 95%の確率で平均に戻る傾向
        # ±3標準偏差: 99%以上の確率で異常値
        
        abs_z = abs(z_score)
        
        if abs_z >= 3.0:
            return 0.95
        elif abs_z >= 2.5:
            return 0.9
        elif abs_z >= 2.0:
            return 0.8
        else:
            return 0.7
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _filter_and_rank_reversion_signals(self, signals: List[ReversionSignal]) -> List[ReversionSignal]:
        """平均回帰シグナルフィルタリング・ランキング"""
        
        # フィルタリング
        filtered = []
        for signal in signals:
            # 最小回帰確率チェック
            if signal.reversion_probability < self.reversion_config["REVERSION_THRESHOLD"]:
                continue
            
            # 最小信頼度チェック
            if signal.confidence < 0.5:
                continue
            
            # 乖離範囲チェック
            min_dev = self.reversion_config["MIN_DEVIATION_PCT"]
            max_dev = self.reversion_config["MAX_DEVIATION_PCT"]
            if not (min_dev <= signal.deviation_pct <= max_dev):
                continue
            
            # リスクリワード比チェック
            if signal.direction == "BUY":
                risk = signal.current_price - signal.stop_loss_price
                reward = signal.target_price - signal.current_price
            else:
                risk = signal.stop_loss_price - signal.current_price
                reward = signal.current_price - signal.target_price
            
            if risk <= 0 or reward <= 0 or (reward / risk) < 0.8:  # 平均回帰は控えめなリスクリワード
                continue
            
            filtered.append(signal)
        
        # ランキング（回帰確率 × 信頼度でソート）
        filtered.sort(key=lambda x: x.reversion_probability * x.confidence, reverse=True)
        
        return filtered
    
    async def execute_reversion_trade(self, signal: ReversionSignal) -> bool:
        """平均回帰取引実行"""
        
        try:
            current_price = await self.data_source.get_current_price(signal.symbol)
            current_time = self.data_source.get_current_time()
            
            # ポジションサイズ計算（小さめ）
            risk_per_trade = 1000 * (self.reversion_config["POSITION_SIZE_PCT"] / 100)  # 8%
            if signal.direction == "BUY":
                risk_per_share = signal.current_price - signal.stop_loss_price
            else:
                risk_per_share = signal.stop_loss_price - signal.current_price
            
            if risk_per_share <= 0:
                return False
            
            quantity = risk_per_trade / risk_per_share
            
            # 最小取引量チェック
            if quantity * current_price < 30:  # 最小$30
                quantity = 30 / current_price
            
            # エントリー実行
            order_result = await self.data_source.place_order(
                symbol=signal.symbol,
                side=signal.direction,
                order_type="MARKET",
                quantity=quantity
            )
            
            if order_result.get("status") == "FILLED":
                # ポジション記録
                position = ReversionPosition(
                    signal=signal,
                    entry_time=current_time,
                    entry_price=current_price,
                    quantity=quantity
                )
                
                self.active_positions[signal.symbol] = position
                
                self.logger.info(f"📈 平均回帰取引実行: {signal.reversion_type.value}")
                self.logger.info(f"   {signal.symbol} {signal.direction} ${current_price:.4f}")
                self.logger.info(f"   乖離{signal.deviation_pct:.1f}% 回帰確率{signal.reversion_probability:.1f}")
                
                return True
            else:
                self.logger.error(f"❌ 平均回帰取引失敗: {signal.symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 平均回帰取引実行エラー: {str(e)}")
            return False
    
    async def manage_reversion_positions(self) -> List[TradeResult]:
        """平均回帰ポジション管理"""
        
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
                    move_toward_target = current_price - position.entry_price
                    move_against = position.entry_price - current_price if current_price < position.entry_price else 0
                else:
                    unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    move_toward_target = position.entry_price - current_price
                    move_against = current_price - position.entry_price if current_price > position.entry_price else 0
                
                # 最大有利・不利移動更新
                if move_toward_target > 0:
                    position.max_favorable_move = max(position.max_favorable_move, move_toward_target)
                if move_against > 0:
                    position.max_adverse_move = max(position.max_adverse_move, move_against)
                
                # 目標達成チェック（段階的利確）
                target_distance = abs(signal.target_price - signal.current_price)
                current_distance = abs(current_price - signal.target_price)
                reversion_progress = 1 - (current_distance / target_distance) if target_distance > 0 else 0
                
                # 80%回帰で利確
                if reversion_progress >= self.reversion_config["REVERSION_TARGET_PCT"]:
                    trade = await self._close_reversion_position(position, current_price, current_time, "目標回帰達成")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                
                # 部分利確（1%利益）
                elif not position.target_reached:
                    profit_pct = (unrealized_pnl / (position.entry_price * position.quantity)) * 100
                    if profit_pct >= self.reversion_config["PARTIAL_PROFIT_PCT"]:
                        position.target_reached = True
                        # 部分決済は実装しないが、利確準備完了をマーク
                
                # ストップロス条件チェック
                elif signal.direction == "BUY" and current_price <= signal.stop_loss_price:
                    trade = await self._close_reversion_position(position, current_price, current_time, "ストップロス")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                elif signal.direction == "SELL" and current_price >= signal.stop_loss_price:
                    trade = await self._close_reversion_position(position, current_price, current_time, "ストップロス")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                
                # 時間切れチェック
                hold_hours = (current_time - position.entry_time).total_seconds() / 3600
                if hold_hours >= self.reversion_config["MAX_HOLD_HOURS"]:
                    trade = await self._close_reversion_position(position, current_price, current_time, "時間切れ")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(symbol)
                
                # 逆行拡大チェック（回帰失敗）
                elif position.max_adverse_move > 0:
                    adverse_pct = (position.max_adverse_move / position.entry_price) * 100
                    if adverse_pct > self.reversion_config["STOP_LOSS_PCT"] * 1.5:  # 1.5倍で強制決済
                        trade = await self._close_reversion_position(position, current_price, current_time, "回帰失敗")
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
    
    async def _close_reversion_position(self, position: ReversionPosition, current_price: float,
                                      current_time: datetime, exit_reason: str) -> Optional[TradeResult]:
        """平均回帰ポジション決済"""
        
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
                    exit_reason=f"平均回帰_{exit_reason}"
                )
                
                self.trade_history.append(trade)
                
                self.logger.info(f"📈 平均回帰決済: {signal.symbol} {exit_reason}")
                self.logger.info(f"   エントリー: ${position.entry_price:.4f} → 決済: ${current_price:.4f}")
                self.logger.info(f"   利益: ${profit_loss:.2f} ({profit_pct:+.2f}%) 保有{hold_hours:.1f}時間")
                
                return trade
            else:
                self.logger.error(f"❌ 平均回帰決済失敗: {signal.symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"平均回帰決済エラー: {str(e)}")
            return None

class MeanReversionBacktestSystem(AnnualBacktestSystem):
    """平均回帰バックテストシステム"""
    
    def __init__(self, config: Dict, historical_data: Dict[str, pd.DataFrame], 
                 symbols: List[str], start_date: datetime, end_date: datetime):
        super().__init__(config, historical_data, symbols, start_date, end_date)
        
        self.reversion_strategy = MeanReversionStrategy(
            config, WindowsDataSource(ExecutionMode.BACKTEST, historical_data), ExecutionMode.BACKTEST
        )
        
        # 平均回帰専用設定
        self.enhanced_config.update({
            "STRATEGY_NAME": "平均回帰戦略",
            "EXPECTED_ANNUAL_RETURN": 17.0,  # 12-22%の中央値
            "MAX_POSITIONS": 4,              # 最大ポジション数（保守的）
            "REBALANCE_INTERVAL": 6,         # 6時間ごとチェック
        })
    
    async def _execute_annual_backtest(self):
        """平均回帰年間バックテスト実行"""
        
        capital = self.enhanced_config["INITIAL_CAPITAL"]
        
        # 時間ステップ（6時間ごと）
        timestamps = list(self.historical_data[self.symbols[0]].index[::6])
        
        for i, timestamp in enumerate(timestamps):
            try:
                # データソース時刻設定
                self.reversion_strategy.data_source.set_current_time(timestamp)
                current_portfolio_value = capital
                
                # 既存平均回帰ポジション管理
                trades = await self.reversion_strategy.manage_reversion_positions()
                for trade in trades:
                    capital += trade.profit_loss + (trade.entry_price * trade.quantity)
                    self.trades.append(trade)
                    current_portfolio_value += trade.profit_loss
                
                # 新規平均回帰シグナル検索
                active_positions = len(self.reversion_strategy.active_positions)
                if active_positions < self.enhanced_config["MAX_POSITIONS"]:
                    
                    signals = await self.reversion_strategy.analyze_reversion_signals(self.symbols)
                    
                    for signal in signals[:2]:  # TOP2実行（保守的）
                        if active_positions >= self.enhanced_config["MAX_POSITIONS"]:
                            break
                        
                        # 重複チェック
                        if signal.symbol not in self.reversion_strategy.active_positions:
                            required_capital = signal.current_price * 50  # 最小$50
                            
                            if capital > required_capital:
                                success = await self.reversion_strategy.execute_reversion_trade(signal)
                                if success:
                                    capital -= required_capital
                                    active_positions += 1
                                    self.logger.info(f"📈 {signal.symbol} 平均回帰開始: {signal.reversion_type.value}")
                
                # ポートフォリオ価値計算
                reversion_investment = sum([
                    pos.entry_price * pos.quantity for pos in self.reversion_strategy.active_positions.values()
                ])
                portfolio_value = capital + reversion_investment
                
                # 日次記録
                if i % 4 == 0:  # 24時間ごと
                    self.daily_portfolio.append({
                        'timestamp': timestamp,
                        'portfolio_value': portfolio_value,
                        'cash': capital,
                        'positions': len(self.reversion_strategy.active_positions),
                        'return_pct': ((portfolio_value - self.enhanced_config["INITIAL_CAPITAL"]) / 
                                     self.enhanced_config["INITIAL_CAPITAL"]) * 100
                    })
                
                # 進捗表示
                if i % 168 == 0:  # 週次
                    progress = (i / len(timestamps)) * 100
                    weeks = i // 28
                    active_positions = len(self.reversion_strategy.active_positions)
                    self.logger.info(f"  進捗: {progress:.1f}% ({weeks}週経過) アクティブポジション:{active_positions}")
                
            except Exception as e:
                self.logger.warning(f"タイムスタンプ {timestamp} でエラー: {str(e)}")
                continue

# メイン実行関数
async def run_mean_reversion_backtest():
    """平均回帰戦略バックテスト実行"""
    
    logger = logging.getLogger(__name__)
    logger.info("📈 平均回帰戦略 1年間バックテスト開始")
    
    # 平均回帰用設定
    config = {
        "STRATEGY_TYPE": "MEAN_REVERSION",
        "BB_PERIOD": 20,
        "BB_STD_DEV": 2.0,
        "MIN_DEVIATION_PCT": 1.5,
        "MAX_DEVIATION_PCT": 8.0,
        "REVERSION_THRESHOLD": 0.7,
        "RSI_PERIOD": 14,
        "RSI_OVERSOLD": 25,
        "RSI_OVERBOUGHT": 75,
        "TARGET_PROFIT_PCT": 2.0,
        "STOP_LOSS_PCT": 3.0,
        "MAX_POSITIONS": 4,
        "POSITION_SIZE_PCT": 8.0
    }
    
    # 1年間バックテストシステム作成
    logger.info("📊 平均回帰バックテストシステム作成中...")
    annual_system = await WindowsUnifiedSystemFactory.create_annual_backtest_system(
        config, use_real_data=True
    )
    
    # 平均回帰システムに変換
    reversion_system = MeanReversionBacktestSystem(
        config, annual_system.historical_data, annual_system.symbols,
        annual_system.start_date, annual_system.end_date
    )
    
    # バックテスト実行
    logger.info("📈 平均回帰バックテスト実行中...")
    results = await reversion_system.run_annual_comprehensive_backtest()
    
    # 結果表示
    print("\n" + "="*80)
    print("📈 平均回帰戦略 1年間バックテスト完了")
    print("📊 調査報告書準拠実装（期待年利12-22%）")
    print("="*80)
    
    perf = results['performance_metrics']
    print(f"\n📊 パフォーマンス結果:")
    print(f"   戦略タイプ: 平均回帰（統計的回帰）")
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
    
    if perf['total_return'] >= 12.0:
        print("✅ 調査報告書期待値（年12-22%）達成")
    else:
        print("❌ 調査報告書期待値未達成")
    
    # 全戦略比較
    print(f"\n📊 戦略比較:")
    print(f"   グリッド取引: +0.2% (勝率100%, 取引53)")
    print(f"   DCA Bot: +0.0% (勝率100%, 取引1)")
    print(f"   アービトラージ: -0.2% (勝率49.3%, 取引505)")
    print(f"   モメンタム: -23.3% (勝率38.2%, 取引131)")
    print(f"   平均回帰: {perf['total_return']:+.1f}% (勝率{perf['win_rate']:.1f}%, 取引{perf['total_trades']})")
    
    return results

async def main():
    """メイン実行"""
    await run_mean_reversion_backtest()

if __name__ == "__main__":
    asyncio.run(main())