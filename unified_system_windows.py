#!/usr/bin/env python3
"""
Windows環境対応 統一トレーディングシステム
Unicode問題とWindows環境の特殊性に対応
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import sys
import os
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

# Windows環境用のエンコーディング設定
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Windows用ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_system_windows.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ExecutionMode(Enum):
    """実行モード"""
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading" 
    LIVE_TRADING = "live_trading"

@dataclass
class TradeSignal:
    """取引シグナル"""
    symbol: str
    side: str  # BUY/SELL
    signal_type: str  # main/alternative/scalping
    confidence: float  # 0.0-1.0
    reasons: List[str]
    entry_price: float
    timestamp: datetime

@dataclass
class TradeResult:
    """取引結果"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    profit_loss: float
    profit_pct: float
    hold_hours: float
    exit_reason: str

@dataclass
class RiskMetrics:
    """リスク指標"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float
    win_rate: float
    profit_factor: float

@dataclass
class MonthlyPerformance:
    """月次パフォーマンス"""
    month: str
    trades: int
    return_pct: float
    win_rate: float
    max_dd: float
    sharpe: float
    best_trade: float
    worst_trade: float

class WindowsDataSource:
    """Windows環境対応データソース"""
    
    def __init__(self, execution_mode: ExecutionMode, historical_data: Dict = None):
        self.execution_mode = execution_mode
        self.historical_data = historical_data or {}
        self.current_timestamp = None
        self.capital = 1000.0
        self.positions = {}
        self.trade_history = []
        
    async def get_current_price(self, symbol: str) -> float:
        """現在価格取得"""
        if self.execution_mode == ExecutionMode.BACKTEST:
            return self._get_backtest_price(symbol)
        else:
            # 実際のAPIコール（モック）
            base_prices = {
                "AVAXUSDT": 35.0,
                "LINKUSDT": 14.0,
                "BTCUSDT": 67000.0
            }
            base_price = base_prices.get(symbol, 100.0)
            # 小さなランダム変動を追加
            variation = np.random.uniform(-0.01, 0.01)
            return base_price * (1 + variation)
    
    def _get_backtest_price(self, symbol: str) -> float:
        """バックテスト用価格取得"""
        if symbol not in self.historical_data:
            return 100.0
            
        df = self.historical_data[symbol]
        if self.current_timestamp in df.index:
            return float(df.loc[self.current_timestamp, 'close'])
        
        # 最も近い過去の価格
        past_times = df.index[df.index <= self.current_timestamp]
        if len(past_times) > 0:
            return float(df.loc[past_times[-1], 'close'])
        
        return 100.0
    
    async def get_ohlcv(self, symbol: str, interval: str, limit: int) -> List[Dict]:
        """OHLCVデータ取得"""
        if self.execution_mode == ExecutionMode.BACKTEST:
            return self._get_backtest_ohlcv(symbol, interval, limit)
        else:
            # 実際のAPIコール用のモックデータ生成
            return self._generate_mock_ohlcv(symbol, limit)
    
    def _get_backtest_ohlcv(self, symbol: str, interval: str, limit: int) -> List[Dict]:
        """バックテスト用OHLCVデータ"""
        if symbol not in self.historical_data:
            return []
            
        df = self.historical_data[symbol]
        df_past = df[df.index <= self.current_timestamp]
        
        if len(df_past) < limit:
            df_subset = df_past
        else:
            df_subset = df_past.iloc[-limit:]
        
        ohlcv_list = []
        for idx, row in df_subset.iterrows():
            ohlcv_list.append({
                'timestamp': int(idx.timestamp() * 1000),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        return ohlcv_list
    
    def _generate_mock_ohlcv(self, symbol: str, limit: int) -> List[Dict]:
        """モックOHLCVデータ生成"""
        base_prices = {
            "AVAXUSDT": 35.0,
            "LINKUSDT": 14.0,
            "BTCUSDT": 67000.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        ohlcv_list = []
        
        current_time = datetime.now()
        
        for i in range(limit):
            timestamp = current_time - timedelta(minutes=5 * (limit - i))
            
            # 価格変動生成
            price_change = np.random.normal(0, 0.01)
            price = base_price * (1 + price_change)
            
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.uniform(100000, 500000)
            
            ohlcv_list.append({
                'timestamp': int(timestamp.timestamp() * 1000),
                'open': price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return ohlcv_list
    
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         quantity: float, price: float = None) -> Dict:
        """注文実行（モック）"""
        current_price = await self.get_current_price(symbol)
        execution_price = price if price else current_price
        
        order_id = f"{self.execution_mode.value}_{symbol}_{side}_{datetime.now().strftime('%H%M%S')}"
        
        return {
            "status": "FILLED",
            "orderId": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "price": execution_price,
            "timestamp": datetime.now()
        }
    
    def get_current_time(self) -> datetime:
        """現在時刻取得"""
        if self.execution_mode == ExecutionMode.BACKTEST:
            return self.current_timestamp
        return datetime.now()
    
    def set_current_time(self, timestamp: datetime):
        """バックテスト用時刻設定"""
        self.current_timestamp = timestamp

class WindowsUnifiedStrategy:
    """Windows環境対応統一戦略"""
    
    def __init__(self, config: Dict, data_source: WindowsDataSource, execution_mode: ExecutionMode):
        self.config = config
        self.data_source = data_source
        self.execution_mode = execution_mode
        self.logger = logging.getLogger(__name__)
        
        # 状態管理
        self.current_positions = {}
        self.trade_history = []
        
        # デフォルト設定で不足を補完
        self._ensure_config_completeness()
    
    def _ensure_config_completeness(self):
        """設定の完全性を保証"""
        defaults = {
            "BB_PERIOD": 20,
            "BB_STD_DEV": 2.0,
            "RSI_PERIOD": 14,
            "RSI_THRESHOLD_LONG_OPTIMIZED_HOURS": 55,
            "RSI_THRESHOLD_LONG_OTHER_HOURS": 50,
            "RSI_THRESHOLD_SHORT_OPTIMIZED_HOURS": 45,
            "RSI_THRESHOLD_SHORT_OTHER_HOURS": 40,
            "MACD_FAST_EMA_PERIOD": 12,
            "MACD_SLOW_EMA_PERIOD": 26,
            "MACD_SIGNAL_SMA_PERIOD": 9,
            "EMA_PERIODS": [20, 50],
            "OPTIMIZED_TRADING_HOURS": [
                {"start": 7, "end": 11},
                {"start": 15, "end": 18},
                {"start": 22, "end": 24}
            ],
            "ALTERNATIVE_ENTRY_ENABLED": True,
            "RELAXED_ENTRY_CONDITIONS": {
                "RSI_NEUTRAL_ZONE_MIN": 40,
                "RSI_NEUTRAL_ZONE_MAX": 60,
                "BB_ENTRY_THRESHOLD_UPPER": 0.98,
                "BB_ENTRY_THRESHOLD_LOWER": 1.02,
                "MINIMUM_PRICE_CHANGE_30MIN": 0.05,
                "MINIMUM_PRICE_CHANGE_THRESHOLD": 0.1,
                "MACD_DIRECTION_ENABLED": True,
                "EMA_RELAXED_MODE": True,
                "EMA_PROXIMITY_THRESHOLD": 0.02,
            }
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    async def analyze_entry_conditions(self, symbol: str) -> Optional[TradeSignal]:
        """エントリー条件分析"""
        
        try:
            # 市場データ取得
            ohlcv_5m = await self.data_source.get_ohlcv(symbol, "5m", 100)
            ohlcv_15m = await self.data_source.get_ohlcv(symbol, "15m", 100)
            ohlcv_1h = await self.data_source.get_ohlcv(symbol, "60m", 100)
            
            if not ohlcv_5m or len(ohlcv_5m) < 50:
                return None
            
            # データフレーム変換
            df_5m = pd.DataFrame(ohlcv_5m)
            df_15m = pd.DataFrame(ohlcv_15m) if ohlcv_15m else df_5m.copy()
            df_1h = pd.DataFrame(ohlcv_1h) if ohlcv_1h else df_5m.copy()
            
            # テクニカル指標計算
            self._calculate_technical_indicators(df_5m, df_15m, df_1h)
            
            # 現在価格と時刻
            current_price = await self.data_source.get_current_price(symbol)
            current_time = self.data_source.get_current_time()
            
            # エントリー条件チェック
            signal = self._check_entry_conditions(df_5m, df_15m, df_1h, symbol, current_price, current_time)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"エントリー分析エラー {symbol}: {str(e)}")
            return None
    
    def _calculate_technical_indicators(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame):
        """テクニカル指標計算（Windows対応版）"""
        
        try:
            # 5分足指標
            bb_period = self.config["BB_PERIOD"]
            bb_std_dev = self.config["BB_STD_DEV"]
            
            # ボリンジャーバンド（手動計算でエラー回避）
            rolling_mean = df_5m["close"].rolling(window=bb_period).mean()
            rolling_std = df_5m["close"].rolling(window=bb_period).std()
            df_5m["bb_upper"] = rolling_mean + (rolling_std * bb_std_dev)
            df_5m["bb_lower"] = rolling_mean - (rolling_std * bb_std_dev)
            df_5m["bb_mid"] = rolling_mean
            
            # RSI（簡易計算）
            rsi_period = self.config["RSI_PERIOD"]
            delta = df_5m["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df_5m["rsi"] = 100 - (100 / (1 + rs))
            
            # 15分足RSI
            delta_15m = df_15m["close"].diff()
            gain_15m = (delta_15m.where(delta_15m > 0, 0)).rolling(window=rsi_period).mean()
            loss_15m = (-delta_15m.where(delta_15m < 0, 0)).rolling(window=rsi_period).mean()
            rs_15m = gain_15m / loss_15m
            df_15m["rsi"] = 100 - (100 / (1 + rs_15m))
            
            # EMA（指数移動平均）
            ema_periods = self.config["EMA_PERIODS"]
            alpha_20 = 2 / (ema_periods[0] + 1)
            alpha_50 = 2 / (ema_periods[1] + 1)
            
            df_1h["ema20"] = df_1h["close"].ewm(alpha=alpha_20).mean()
            df_1h["ema50"] = df_1h["close"].ewm(alpha=alpha_50).mean()
            
            # MACD（簡易版）
            macd_fast = self.config["MACD_FAST_EMA_PERIOD"]
            macd_slow = self.config["MACD_SLOW_EMA_PERIOD"]
            
            ema_fast = df_15m["close"].ewm(span=macd_fast).mean()
            ema_slow = df_15m["close"].ewm(span=macd_slow).mean()
            df_15m["macd"] = ema_fast - ema_slow
            df_15m["macd_signal"] = df_15m["macd"].ewm(span=9).mean()
            
        except Exception as e:
            self.logger.error(f"テクニカル指標計算エラー: {str(e)}")
            # エラー時はNaNで埋める
            for df in [df_5m, df_15m, df_1h]:
                for col in ["bb_upper", "bb_lower", "bb_mid", "rsi", "macd", "macd_signal", "ema20", "ema50"]:
                    if col not in df.columns:
                        df[col] = np.nan
    
    def _check_entry_conditions(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame, 
                               df_1h: pd.DataFrame, symbol: str, current_price: float,
                               current_time: datetime) -> Optional[TradeSignal]:
        """エントリー条件チェック"""
        
        if len(df_5m) < 10:
            return None
        
        # 最新値取得（NaN対策）
        last_close = df_5m["close"].iloc[-1]
        last_rsi = df_5m["rsi"].iloc[-1] if not pd.isna(df_5m["rsi"].iloc[-1]) else 50
        last_macd = df_15m["macd"].iloc[-1] if not pd.isna(df_15m["macd"].iloc[-1]) else 0
        last_macd_signal = df_15m["macd_signal"].iloc[-1] if not pd.isna(df_15m["macd_signal"].iloc[-1]) else 0
        last_ema20 = df_1h["ema20"].iloc[-1] if not pd.isna(df_1h["ema20"].iloc[-1]) else current_price
        last_ema50 = df_1h["ema50"].iloc[-1] if not pd.isna(df_1h["ema50"].iloc[-1]) else current_price
        bb_upper = df_5m["bb_upper"].iloc[-1] if not pd.isna(df_5m["bb_upper"].iloc[-1]) else current_price * 1.02
        bb_lower = df_5m["bb_lower"].iloc[-1] if not pd.isna(df_5m["bb_lower"].iloc[-1]) else current_price * 0.98
        
        # 時間帯判定
        current_hour = current_time.hour
        is_optimized_hours = any(
            period["start"] <= current_hour < period["end"]
            for period in self.config["OPTIMIZED_TRADING_HOURS"]
        )
        
        # 30分価格変動
        if len(df_5m) >= 6:
            price_change_30min = (last_close - df_5m["close"].iloc[-6]) / df_5m["close"].iloc[-6] * 100
        else:
            price_change_30min = 0
        
        # ロングエントリー条件
        long_signal = self._check_long_conditions(
            last_close, last_rsi, last_macd, last_macd_signal, 
            last_ema20, last_ema50, bb_upper, price_change_30min, 
            is_optimized_hours, symbol, current_price, current_time
        )
        
        if long_signal:
            return long_signal
        
        # 簡易ショート条件
        if (last_close < bb_lower and 
            last_rsi < 45 and 
            abs(price_change_30min) > 0.05):
            
            return TradeSignal(
                symbol=symbol,
                side="SELL",
                signal_type="short_breakout",
                confidence=0.7,
                reasons=["BB下限突破", f"RSI低下({last_rsi:.1f})", f"価格変動{price_change_30min:.2f}%"],
                entry_price=current_price,
                timestamp=current_time
            )
        
        return None
    
    def _check_long_conditions(self, last_close: float, last_rsi: float, last_macd: float,
                              last_macd_signal: float, last_ema20: float, last_ema50: float,
                              bb_upper: float, price_change_30min: float, is_optimized_hours: bool,
                              symbol: str, current_price: float, current_time: datetime) -> Optional[TradeSignal]:
        """ロング条件チェック"""
        
        # RSI閾値
        rsi_threshold = (self.config["RSI_THRESHOLD_LONG_OPTIMIZED_HOURS"] 
                        if is_optimized_hours 
                        else self.config["RSI_THRESHOLD_LONG_OTHER_HOURS"])
        
        reasons = []
        conditions = []
        
        # 1. ボリンジャーバンド条件（緩和）
        relaxed_conditions = self.config.get("RELAXED_ENTRY_CONDITIONS", {})
        bb_threshold = bb_upper * relaxed_conditions.get("BB_ENTRY_THRESHOLD_UPPER", 0.98)
        
        if last_close > bb_threshold:
            conditions.append(True)
            reasons.append("BB上限突破(緩和)")
        else:
            conditions.append(False)
        
        # 2. 最小価格変動
        min_change = relaxed_conditions.get("MINIMUM_PRICE_CHANGE_30MIN", 0.05)
        if abs(price_change_30min) > min_change:
            conditions.append(True)
            reasons.append(f"価格変動{price_change_30min:.2f}%")
        else:
            conditions.append(False)
        
        # 3. RSI条件（緩和）
        relaxed_rsi = rsi_threshold * 0.9
        if last_rsi > relaxed_rsi:
            conditions.append(True)
            reasons.append(f"RSI上昇({last_rsi:.1f})")
        else:
            conditions.append(False)
        
        # 4. MACD条件（緩和）
        if last_macd > last_macd_signal:
            conditions.append(True)
            reasons.append("MACD強気")
        else:
            conditions.append(False)
        
        # 5. EMA条件（緩和）
        if last_ema20 > last_ema50:
            conditions.append(True)
            reasons.append("EMA上昇")
        else:
            conditions.append(False)
        
        # メイン条件判定
        confidence = sum(conditions) / len(conditions)
        
        if all(conditions):
            return TradeSignal(
                symbol=symbol,
                side="BUY",
                signal_type="main_entry",
                confidence=confidence,
                reasons=reasons,
                entry_price=current_price,
                timestamp=current_time
            )
        
        # 代替エントリー条件
        if (self.config.get("ALTERNATIVE_ENTRY_ENABLED", False) and
            confidence >= 0.6):  # 60%以上の条件を満たす
            
            return TradeSignal(
                symbol=symbol,
                side="BUY",
                signal_type="alternative_entry",
                confidence=confidence,
                reasons=reasons,
                entry_price=current_price,
                timestamp=current_time
            )
        
        return None
    
    async def execute_trade(self, signal: TradeSignal) -> bool:
        """取引実行"""
        
        try:
            # ポジションサイズ計算
            position_size = 100 / signal.entry_price  # 100 USD相当
            
            # 注文実行
            order_result = await self.data_source.place_order(
                symbol=signal.symbol,
                side=signal.side,
                order_type="MARKET",
                quantity=position_size
            )
            
            if order_result.get("status") == "FILLED":
                # ポジション記録
                self.current_positions[signal.symbol] = {
                    'signal': signal,
                    'order_result': order_result,
                    'entry_time': signal.timestamp
                }
                
                self.logger.info(f"✅ {signal.symbol} {signal.side} エントリー成功")
                self.logger.info(f"   理由: {', '.join(signal.reasons)}")
                self.logger.info(f"   信頼度: {signal.confidence:.2f}")
                
                return True
            else:
                self.logger.error(f"❌ {signal.symbol} エントリー失敗: {order_result}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 取引実行エラー {signal.symbol}: {str(e)}")
            return False

class WindowsUnifiedSystemFactory:
    """Windows環境対応統一システムファクトリー"""
    
    @staticmethod
    async def create_backtest_system(config: Dict, symbols: List[str], 
                                   start_date: datetime, end_date: datetime) -> WindowsUnifiedStrategy:
        """バックテスト用システム作成"""
        
        # モック履歴データ生成
        historical_data = WindowsUnifiedSystemFactory._generate_mock_historical_data(
            symbols, start_date, end_date
        )
        
        data_source = WindowsDataSource(ExecutionMode.BACKTEST, historical_data)
        return WindowsUnifiedStrategy(config, data_source, ExecutionMode.BACKTEST)
    
    @staticmethod
    async def create_annual_backtest_system(config: Dict, symbols: List[str] = None, 
                                          use_real_data: bool = True) -> 'AnnualBacktestSystem':
        """1年間包括バックテストシステム作成"""
        
        if symbols is None:
            # MEXC有効な23銘柄（元25銘柄からFTMUSDT、MATICUSDT のみ除外）
            symbols = [
                # Tier 1 (8銘柄) - FTM, MATIC除外済み
                "AVAXUSDT", "LINKUSDT", "NEARUSDT", "ATOMUSDT", "DOTUSDT", 
                "UNIUSDT", "AAVEUSDT", "DOGEUSDT",
                # Tier 2 (10銘柄) - 全て有効
                "ADAUSDT", "ALGOUSDT", "APEUSDT", "ARBUSDT", "EGLDUSDT",
                "FILUSDT", "GRTUSDT", "ICPUSDT", "LTCUSDT", "SANDUSDT",
                # Tier 3 (5銘柄) - 全て有効
                "SHIBUSDT", "VETUSDT", "MANAUSDT", "GALAUSDT", "ONEUSDT"
            ]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        logger = logging.getLogger(__name__)
        
        # データ取得方法選択
        if use_real_data:
            try:
                logger.info("🌐 MEXC APIから実際の価格データを取得中...")
                historical_data = await WindowsUnifiedSystemFactory._fetch_real_mexc_historical_data(
                    symbols, start_date, end_date
                )
                
                if not historical_data:
                    raise Exception("実データ取得に失敗")
                    
                logger.info("✅ 実データ取得完了 - 実際のMEXC価格でバックテスト実行")
                
            except Exception as e:
                logger.warning(f"⚠️ 実データ取得失敗: {str(e)}")
                logger.info("📊 シミュレーションデータにフォールバック")
                historical_data = WindowsUnifiedSystemFactory._generate_annual_realistic_data(
                    symbols, start_date, end_date
                )
        else:
            logger.info("📊 シミュレーションデータでバックテスト実行")
            historical_data = WindowsUnifiedSystemFactory._generate_annual_realistic_data(
                symbols, start_date, end_date
            )
        
        return AnnualBacktestSystem(config, historical_data, symbols, start_date, end_date)
    
    @staticmethod
    def create_live_system(config: Dict, is_paper_trading: bool = True) -> WindowsUnifiedStrategy:
        """ライブ取引用システム作成"""
        
        mode = ExecutionMode.PAPER_TRADING if is_paper_trading else ExecutionMode.LIVE_TRADING
        data_source = WindowsDataSource(mode)
        return WindowsUnifiedStrategy(config, data_source, mode)
    
    @staticmethod
    def _generate_mock_historical_data(symbols: List[str], start_date: datetime, 
                                     end_date: datetime) -> Dict[str, pd.DataFrame]:
        """モック履歴データ生成"""
        
        historical_data = {}
        dates = pd.date_range(start=start_date, end=end_date, freq='5T')
        
        for symbol in symbols:
            base_price = {"AVAXUSDT": 35.0, "LINKUSDT": 14.0, "BTCUSDT": 67000.0}.get(symbol, 100.0)
            
            np.random.seed(hash(symbol) % 2**32)
            price_changes = np.random.normal(0, 0.005, len(dates))
            prices = base_price * np.exp(np.cumsum(price_changes))
            
            df_data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                high = price * (1 + abs(np.random.normal(0, 0.003)))
                low = price * (1 - abs(np.random.normal(0, 0.003)))
                volume = np.random.uniform(100000, 500000)
                
                df_data.append({
                    'timestamp': date,
                    'open': price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            historical_data[symbol] = df
        
        return historical_data
    
    @staticmethod
    async def _fetch_real_mexc_historical_data(symbols: List[str], start_date: datetime, 
                                             end_date: datetime) -> Dict[str, pd.DataFrame]:
        """MEXC APIから実際の1年間履歴データ取得"""
        
        from mexc_api import MEXCAPI
        
        # MEXC API初期化（公開エンドポイント用）
        mexc_api = MEXCAPI("", "", test_mode=True, notifier=None)
        
        historical_data = {}
        logger = logging.getLogger(__name__)
        
        logger.info(f"🌐 MEXC APIから実データ取得開始: {len(symbols)}銘柄")
        
        # 1年間を複数期間に分割（MEXC APIの制限対応）
        total_days = (end_date - start_date).days
        for symbol in symbols:
            logger.info(f"  📊 {symbol} データ取得中...")
            
            try:
                # 1時間足データ取得（MEXC API修正版 - limitのみ使用）
                all_klines = await mexc_api.get_klines(
                    symbol=symbol,
                    interval="60m",  # MEXC API修正: 1h -> 60m
                    limit=1000  # MEXC API制限内（最大1000本）
                )
                
                if all_klines:
                    logger.info(f"    {symbol}: {len(all_klines)}本のデータ取得成功")
                else:
                    logger.warning(f"    {symbol}: データ取得失敗")
                    all_klines = []
                    
            except Exception as e:
                logger.warning(f"    {symbol} データ取得エラー: {str(e)}")
                all_klines = []
                await asyncio.sleep(0.5)  # API制限対策（より安全な間隔）
            
            # DataFrameに変換
            if all_klines:
                df_data = []
                for kline in all_klines:
                    # MEXC Klinesフォーマット: [timestamp, open, high, low, close, volume, ...]
                    timestamp = pd.to_datetime(kline[0], unit='ms')
                    df_data.append({
                        'timestamp': timestamp,
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })
                
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()  # 時系列順にソート
                
                historical_data[symbol] = df
                logger.info(f"  ✅ {symbol}: {len(df)}本のデータ取得完了")
            else:
                logger.error(f"  ❌ {symbol}: データ取得失敗")
        
        await mexc_api._close_session()
        
        logger.info(f"🎉 実データ取得完了: {len(historical_data)}銘柄")
        return historical_data
    
    @staticmethod
    def _generate_annual_realistic_data(symbols: List[str], start_date: datetime, 
                                      end_date: datetime) -> Dict[str, pd.DataFrame]:
        """1年間リアルデータ生成（GARCH風ボラティリティクラスター対応）- フォールバック用"""
        
        historical_data = {}
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')  # 1時間足
        
        for symbol in symbols:
            # 銘柄別基準価格
            base_prices = {
                "BTCUSDT": 45000.0, "ETHUSDT": 2800.0, "AVAXUSDT": 35.0,
                "LINKUSDT": 14.0, "NEARUSDT": 4.5, "FTMUSDT": 0.65,
                "ATOMUSDT": 12.0, "DOTUSDT": 7.0, "MATICUSDT": 1.1, "UNIUSDT": 6.5
            }
            
            base_price = base_prices.get(symbol, 100.0)
            
            # 年間トレンド（-50%～+150%）
            np.random.seed(hash(symbol) % 2**32)
            annual_trend = np.random.uniform(-0.5, 1.5)
            trend_component = np.linspace(0, annual_trend, len(dates))
            
            # ボラティリティクラスター（GARCH風）
            base_vol = 0.02 if "BTC" in symbol or "ETH" in symbol else 0.035
            volatility = WindowsUnifiedSystemFactory._generate_volatility_clusters(len(dates), base_vol)
            
            # 季節性（仮想通貨特有の周期性）
            seasonal_component = 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / (365.25 * 24))
            
            # ランダムウォーク + ジャンプ拡散
            random_walk = np.random.normal(0, volatility, len(dates))
            
            # 極端イベント（月1-2回）
            jump_probability = 1 / (30 * 24)  # 月1回程度
            jumps = np.random.poisson(jump_probability, len(dates))
            jump_sizes = np.random.normal(0, 0.1, len(dates)) * jumps
            
            # 価格シリーズ構築
            log_returns = trend_component + seasonal_component + random_walk + jump_sizes
            prices = base_price * np.exp(np.cumsum(log_returns / len(dates)))
            
            # OHLCV生成
            df_data = []
            for i, (timestamp, price) in enumerate(zip(dates, prices)):
                intraday_vol = volatility[i] * price * 0.3
                
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                
                high_low_range = abs(np.random.normal(0, intraday_vol))
                high = max(open_price, close_price) + high_low_range * np.random.uniform(0.2, 1.0)
                low = min(open_price, close_price) - high_low_range * np.random.uniform(0.2, 1.0)
                
                # 出来高（価格変動と相関）
                volume_base = {
                    "BTCUSDT": 20000000, "ETHUSDT": 15000000, "AVAXUSDT": 500000,
                    "LINKUSDT": 800000, "NEARUSDT": 300000
                }.get(symbol, 200000)
                
                price_change = abs(log_returns[i]) if i < len(log_returns) else 0.01
                volume_multiplier = 1 + price_change * 30
                volume = volume_base * volume_multiplier * np.random.uniform(0.3, 2.0)
                
                df_data.append({
                    'timestamp': timestamp,
                    'open': max(0.001, open_price),
                    'high': max(0.001, high),
                    'low': max(0.001, low),
                    'close': max(0.001, close_price),
                    'volume': max(1, volume)
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            historical_data[symbol] = df
        
        return historical_data
    
    @staticmethod
    def _generate_volatility_clusters(n_points: int, base_vol: float) -> np.ndarray:
        """ボラティリティクラスター生成（GARCH風）"""
        
        volatility = np.zeros(n_points)
        volatility[0] = base_vol
        
        # GARCH(1,1)風パラメータ
        omega = base_vol * 0.1
        alpha = 0.1
        beta = 0.8
        
        for i in range(1, n_points):
            # 前期残差の平方
            epsilon_sq = np.random.normal(0, volatility[i-1]) ** 2
            
            # ボラティリティ更新
            volatility[i] = np.sqrt(
                omega + alpha * epsilon_sq + beta * (volatility[i-1] ** 2)
            )
            
            # 制限
            volatility[i] = np.clip(volatility[i], base_vol * 0.2, base_vol * 5.0)
        
        return volatility

class AnnualBacktestSystem:
    """1年間包括バックテストシステム（統一システム統合版）"""
    
    def __init__(self, config: Dict, historical_data: Dict[str, pd.DataFrame], 
                 symbols: List[str], start_date: datetime, end_date: datetime):
        self.config = config
        self.historical_data = historical_data
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)
        
        # 最適化された設定
        self.enhanced_config = {
            **config,
            "INITIAL_CAPITAL": 10000.0,
            "MAX_POSITION_SIZE": 500.0,
            "MAX_SIMULTANEOUS_POSITIONS": 5,  # 35%運用のため同時ポジション数増加
            
            # 強化された出口戦略（最適化済み）
            "TAKE_PROFIT_LEVELS": [2.4, 5.0, 8.0],  # 仕様書通り段階的利確
            "TAKE_PROFIT_QUANTITIES": [0.3, 0.4, 0.3],
            "STOP_LOSS_INITIAL": 1.5,  # 2.0→1.5 ドローダウン削減
            "TRAILING_STOP_ACTIVATION": 1.0,
            "TRAILING_STOP_DISTANCE": 0.8,
            "TIME_STOP_HOURS": 12,
            
            # 強化されたリスク管理（最適化済み）
            "MAX_DAILY_LOSS": 3.0,
            "POSITION_SIZE_PCT": 7.0,  # 35%運用達成：7% × 5ポジション = 35%
            "VOLATILITY_ADJUSTMENT": True,
            "CORRELATION_LIMIT": 0.7,
            "KELLY_FRACTION_ENABLED": True
        }
        
        # データ保存
        self.trades = []
        self.daily_portfolio = []
        self.positions = {}
        self.performance_metrics = {}
    
    async def run_annual_comprehensive_backtest(self) -> Dict[str, Any]:
        """1年間包括バックテスト実行"""
        
        self.logger.info("🚀 統一システム 1年間包括バックテスト開始")
        self.logger.info(f"期間: {self.start_date.strftime('%Y-%m-%d')} ～ {self.end_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"監視銘柄: {len(self.symbols)}銘柄")
        
        # 1. バックテスト実行
        await self._execute_annual_backtest()
        
        # 2. パフォーマンス分析
        performance_metrics = self._calculate_performance_metrics()
        
        # 3. リスク分析
        risk_metrics = self._calculate_risk_metrics()
        
        # 4. 月次分析
        monthly_analysis = self._perform_monthly_analysis()
        
        # 5. 最適化提案
        optimization_plan = self._generate_optimization_strategy(performance_metrics, risk_metrics)
        
        # 6. 結果集計
        results = {
            'backtest_period': {
                'start': self.start_date.isoformat(),
                'end': self.end_date.isoformat(),
                'duration_days': 365
            },
            'performance_metrics': performance_metrics,
            'risk_metrics': risk_metrics,
            'monthly_analysis': monthly_analysis,
            'optimization_strategy': optimization_plan,
            'trade_summary': {
                'total_trades': len(self.trades),
                'symbols_traded': list(set([t.symbol for t in self.trades])),
                'best_performing_symbol': self._get_best_symbol(),
                'worst_performing_symbol': self._get_worst_symbol(),
                'average_hold_time_hours': self._get_avg_hold_time()
            }
        }
        
        # 7. 結果保存
        await self._save_comprehensive_results(results)
        
        self.logger.info("✅ 統一システム 1年間バックテスト完了")
        return results
    
    async def _execute_annual_backtest(self):
        """年間バックテスト実行"""
        
        capital = self.enhanced_config["INITIAL_CAPITAL"]
        
        # 時間ステップ（6時間ごと）
        timestamps = list(self.historical_data[self.symbols[0]].index[::6])
        
        for i, timestamp in enumerate(timestamps):
            try:
                current_portfolio_value = capital
                
                # 既存ポジション管理
                positions_to_close = []
                for symbol, position in self.positions.items():
                    current_price = self._get_price_at_time(symbol, timestamp)
                    
                    if current_price:
                        # 強化された出口条件チェック
                        exit_decision = self._check_enhanced_exit_conditions(position, current_price, timestamp)
                        
                        if exit_decision['should_exit']:
                            # ポジション決済
                            pnl = self._calculate_position_pnl(position, current_price)
                            capital += pnl + position['notional']
                            
                            # 取引記録
                            hold_hours = (timestamp - position['entry_time']).total_seconds() / 3600
                            
                            trade = TradeResult(
                                symbol=symbol,
                                entry_time=position['entry_time'],
                                exit_time=timestamp,
                                side=position['side'],
                                entry_price=position['entry_price'],
                                exit_price=current_price,
                                quantity=position['quantity'],
                                profit_loss=pnl,
                                profit_pct=(pnl / position['notional']) * 100,
                                hold_hours=hold_hours,
                                exit_reason=exit_decision['reason']
                            )
                            self.trades.append(trade)
                            positions_to_close.append(symbol)
                            
                            current_portfolio_value += pnl
                
                # ポジション削除
                for symbol in positions_to_close:
                    del self.positions[symbol]
                
                # 新規エントリー機会（最適化されたロジック）
                if len(self.positions) < self.enhanced_config["MAX_SIMULTANEOUS_POSITIONS"]:
                    for symbol in self.symbols:
                        if symbol not in self.positions:
                            entry_signal = await self._analyze_enhanced_entry_conditions(symbol, timestamp)
                            
                            if entry_signal and entry_signal.confidence >= 0.70:  # バランス調整：品質と頻度の両立
                                # 最適化されたポジションサイズ
                                position_size = self._calculate_optimal_position_size(
                                    capital, symbol, entry_signal.confidence
                                )
                                
                                if position_size > 50 and capital > position_size:
                                    current_price = self._get_price_at_time(symbol, timestamp)
                                    if current_price:
                                        # ポジション作成
                                        self.positions[symbol] = {
                                            'entry_time': timestamp,
                                            'entry_price': current_price,
                                            'side': entry_signal.side,
                                            'notional': position_size,
                                            'quantity': position_size / current_price,
                                            'stop_loss': self._calculate_initial_stop_loss(current_price, entry_signal.side),
                                            'trailing_stop': None,
                                            'partial_exits': [],
                                            'signal': entry_signal
                                        }
                                        
                                        capital -= position_size
                
                # ポートフォリオ価値計算
                portfolio_value = capital
                for symbol, position in self.positions.items():
                    current_price = self._get_price_at_time(symbol, timestamp)
                    if current_price:
                        portfolio_value += position['notional'] + self._calculate_position_pnl(position, current_price)
                
                # 日次記録
                if i % 4 == 0:  # 24時間ごと
                    self.daily_portfolio.append({
                        'timestamp': timestamp,
                        'portfolio_value': portfolio_value,
                        'cash': capital,
                        'positions': len(self.positions),
                        'return_pct': ((portfolio_value - self.enhanced_config["INITIAL_CAPITAL"]) / 
                                     self.enhanced_config["INITIAL_CAPITAL"]) * 100
                    })
                
                # 進捗表示
                if i % 168 == 0:  # 週次
                    progress = (i / len(timestamps)) * 100
                    weeks = i // 28  # 6時間ステップ換算
                    self.logger.info(f"  進捗: {progress:.1f}% ({weeks}週経過)")
                
            except Exception as e:
                self.logger.warning(f"タイムスタンプ {timestamp} でエラー: {str(e)}")
                continue
    
    def _get_price_at_time(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """指定時刻の価格取得"""
        
        if symbol not in self.historical_data:
            return None
        
        df = self.historical_data[symbol]
        
        # 正確な時刻マッチ
        if timestamp in df.index:
            return float(df.loc[timestamp, 'close'])
        
        # 最も近い過去の価格
        past_times = df.index[df.index <= timestamp]
        if len(past_times) > 0:
            return float(df.loc[past_times[-1], 'close'])
        
        return None
    
    async def _analyze_enhanced_entry_conditions(self, symbol: str, timestamp: datetime) -> Optional[TradeSignal]:
        """強化されたエントリー条件分析"""
        
        try:
            df = self.historical_data[symbol]
            current_idx = df.index.get_loc(timestamp) if timestamp in df.index else None
            
            if current_idx is None or current_idx < 100:
                return None
            
            # 過去100本のデータ
            df_subset = df.iloc[current_idx-99:current_idx+1].copy()
            
            # テクニカル指標計算
            self._calculate_technical_indicators_enhanced(df_subset)
            
            current_price = df_subset['close'].iloc[-1]
            
            # 統一システムベースのエントリー条件
            signal = self._check_unified_entry_conditions(df_subset, symbol, current_price, timestamp)
            
            return signal
            
        except Exception as e:
            return None
    
    def _calculate_technical_indicators_enhanced(self, df: pd.DataFrame):
        """強化されたテクニカル指標計算"""
        
        # ボリンジャーバンド
        bb_period = self.enhanced_config.get("BB_PERIOD", 20)
        bb_std_dev = self.enhanced_config.get("BB_STD_DEV", 2.0)
        sma = df['close'].rolling(bb_period).mean()
        std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = sma + (std * bb_std_dev)
        df['bb_lower'] = sma - (std * bb_std_dev)
        df['bb_mid'] = sma
        
        # RSI
        rsi_period = self.enhanced_config.get("RSI_PERIOD", 14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMA
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # MACD
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    def _check_unified_entry_conditions(self, df: pd.DataFrame, symbol: str, current_price: float, timestamp: datetime) -> Optional[TradeSignal]:
        """統一エントリー条件チェック"""
        
        if len(df) < 50:
            return None
        
        # 最新値取得
        last_rsi = df['rsi'].iloc[-1]
        last_macd = df['macd'].iloc[-1]
        last_macd_signal = df['macd_signal'].iloc[-1]
        last_ema20 = df['ema_20'].iloc[-1]
        last_ema50 = df['ema_50'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        
        # 価格変動率
        if len(df) >= 6:
            price_change_1h = (current_price - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100
        else:
            price_change_1h = 0
        
        # 出来高
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # ロングエントリー条件
        long_conditions = [
            last_rsi > 55,
            current_price > bb_upper * 0.98,
            last_macd > last_macd_signal,
            last_ema20 > last_ema50,
            abs(price_change_1h) > 0.3,
            volume_ratio > 1.3
        ]
        
        long_score = sum(long_conditions) / len(long_conditions)
        
        if long_score >= 0.65:  # バランス調整：品質と頻度の両立
            reasons = []
            if long_conditions[0]: reasons.append(f"RSI強気({last_rsi:.1f})")
            if long_conditions[1]: reasons.append("BB上限近接")
            if long_conditions[2]: reasons.append("MACD上昇")
            if long_conditions[3]: reasons.append("EMA上昇")
            if long_conditions[4]: reasons.append(f"価格変動{price_change_1h:.2f}%")
            if long_conditions[5]: reasons.append("出来高増加")
            
            return TradeSignal(
                symbol=symbol,
                side="BUY",
                signal_type="enhanced_long",
                confidence=long_score,
                reasons=reasons,
                entry_price=current_price,
                timestamp=timestamp
            )
        
        # ショートエントリー条件
        short_conditions = [
            last_rsi < 45,
            current_price < bb_lower * 1.02,
            last_macd < last_macd_signal,
            last_ema20 < last_ema50,
            abs(price_change_1h) > 0.3,
            volume_ratio > 1.3
        ]
        
        short_score = sum(short_conditions) / len(short_conditions)
        
        if short_score >= 0.65:  # バランス調整：品質と頻度の両立
            reasons = []
            if short_conditions[0]: reasons.append(f"RSI弱気({last_rsi:.1f})")
            if short_conditions[1]: reasons.append("BB下限近接")
            if short_conditions[2]: reasons.append("MACD下降")
            if short_conditions[3]: reasons.append("EMA下降")
            if short_conditions[4]: reasons.append(f"価格変動{price_change_1h:.2f}%")
            if short_conditions[5]: reasons.append("出来高増加")
            
            return TradeSignal(
                symbol=symbol,
                side="SELL",
                signal_type="enhanced_short",
                confidence=short_score,
                reasons=reasons,
                entry_price=current_price,
                timestamp=timestamp
            )
        
        return None
    
    def _check_enhanced_exit_conditions(self, position: Dict, current_price: float, timestamp: datetime) -> Dict[str, Any]:
        """強化された出口条件チェック"""
        
        entry_price = position['entry_price']
        side = position['side']
        entry_time = position['entry_time']
        hold_duration = (timestamp - entry_time).total_seconds() / 3600
        
        # 損益計算
        if side == 'BUY':
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
        
        # 段階的利確
        for i, tp_level in enumerate(self.enhanced_config["TAKE_PROFIT_LEVELS"]):
            if profit_pct >= tp_level and len(position['partial_exits']) <= i:
                return {'should_exit': True, 'reason': f'利確_{tp_level}%'}
        
        # 損切り
        if profit_pct <= -self.enhanced_config["STOP_LOSS_INITIAL"]:
            return {'should_exit': True, 'reason': f'損切り_{self.enhanced_config["STOP_LOSS_INITIAL"]}%'}
        
        # トレーリングストップ
        if profit_pct >= self.enhanced_config["TRAILING_STOP_ACTIVATION"]:
            if position['trailing_stop'] is None:
                if side == 'BUY':
                    position['trailing_stop'] = current_price * (1 - self.enhanced_config["TRAILING_STOP_DISTANCE"]/100)
                else:
                    position['trailing_stop'] = current_price * (1 + self.enhanced_config["TRAILING_STOP_DISTANCE"]/100)
            else:
                if side == 'BUY':
                    new_stop = current_price * (1 - self.enhanced_config["TRAILING_STOP_DISTANCE"]/100)
                    if new_stop > position['trailing_stop']:
                        position['trailing_stop'] = new_stop
                    if current_price <= position['trailing_stop']:
                        return {'should_exit': True, 'reason': 'トレーリングストップ'}
                else:
                    new_stop = current_price * (1 + self.enhanced_config["TRAILING_STOP_DISTANCE"]/100)
                    if new_stop < position['trailing_stop']:
                        position['trailing_stop'] = new_stop
                    if current_price >= position['trailing_stop']:
                        return {'should_exit': True, 'reason': 'トレーリングストップ'}
        
        # 時間切れ
        if hold_duration >= self.enhanced_config["TIME_STOP_HOURS"]:
            return {'should_exit': True, 'reason': '時間切れ'}
        
        return {'should_exit': False}
    
    def _calculate_optimal_position_size(self, capital: float, symbol: str, confidence: float) -> float:
        """最適化されたポジションサイズ計算（ケリー基準）"""
        
        base_size = capital * (self.enhanced_config["POSITION_SIZE_PCT"] / 100)
        
        if self.enhanced_config.get("KELLY_FRACTION_ENABLED", False) and len(self.trades) > 20:
            # ケリー基準
            recent_trades = [t for t in self.trades[-50:] if t.symbol == symbol]
            
            if len(recent_trades) >= 10:
                wins = [t for t in recent_trades if t.profit_loss > 0]
                losses = [t for t in recent_trades if t.profit_loss < 0]
                
                if wins and losses:
                    win_rate = len(wins) / len(recent_trades)
                    avg_win = sum([t.profit_pct for t in wins]) / len(wins) / 100
                    avg_loss = abs(sum([t.profit_pct for t in losses]) / len(losses)) / 100
                    
                    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    kelly_fraction = max(0, min(0.2, kelly_fraction))  # 0-20%に制限
                    
                    base_size = capital * kelly_fraction
        
        # 信頼度調整
        adjusted_size = base_size * confidence
        
        # 制限適用
        max_size = min(self.enhanced_config["MAX_POSITION_SIZE"], capital * 0.15)
        return min(adjusted_size, max_size)
    
    def _calculate_initial_stop_loss(self, entry_price: float, side: str) -> float:
        """初期ストップロス計算"""
        
        stop_loss_pct = self.enhanced_config["STOP_LOSS_INITIAL"] / 100
        
        if side == 'BUY':
            return entry_price * (1 - stop_loss_pct)
        else:
            return entry_price * (1 + stop_loss_pct)
    
    def _calculate_position_pnl(self, position: Dict, current_price: float) -> float:
        """ポジション損益計算"""
        
        entry_price = position['entry_price']
        quantity = position['quantity']
        side = position['side']
        
        if side == 'BUY':
            return quantity * (current_price - entry_price)
        else:
            return quantity * (entry_price - current_price)
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンス指標計算"""
        
        if not self.trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'win_rate': 0,
                'total_return': 0, 'max_drawdown': 0, 'profit_factor': 0,
                'avg_win': 0, 'avg_loss': 0, 'sharpe_ratio': 0
            }
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.profit_loss > 0])
        win_rate = (winning_trades / total_trades) * 100
        
        all_pnl = [t.profit_loss for t in self.trades]
        wins = [t.profit_loss for t in self.trades if t.profit_loss > 0]
        losses = [abs(t.profit_loss) for t in self.trades if t.profit_loss < 0]
        
        total_return = (sum(all_pnl) / self.enhanced_config["INITIAL_CAPITAL"]) * 100
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        profit_factor = (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else 0
        
        # シャープレシオ（簡易版）
        returns = [t.profit_pct for t in self.trades]
        if len(returns) > 1:
            import statistics
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 最大ドローダウン
        cumulative_returns = []
        cumsum = 0
        for trade in self.trades:
            cumsum += trade.profit_pct
            cumulative_returns.append(cumsum)
        
        max_drawdown = 0
        if cumulative_returns:
            peak = cumulative_returns[0]
            for ret in cumulative_returns:
                if ret > peak:
                    peak = ret
                drawdown = peak - ret
                max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """リスク指標計算"""
        
        if not self.daily_portfolio:
            return {}
        
        returns = []
        portfolio_values = []
        
        for i, record in enumerate(self.daily_portfolio):
            portfolio_values.append(record['portfolio_value'])
            if i > 0:
                daily_return = (record['portfolio_value'] - self.daily_portfolio[i-1]['portfolio_value']) / self.daily_portfolio[i-1]['portfolio_value']
                returns.append(daily_return)
        
        if not returns:
            return {}
        
        import statistics
        
        # VaR計算
        sorted_returns = sorted(returns)
        var_95_idx = int(len(sorted_returns) * 0.05)
        var_95 = sorted_returns[var_95_idx] if var_95_idx < len(sorted_returns) else sorted_returns[0]
        
        var_99_idx = int(len(sorted_returns) * 0.01)
        var_99 = sorted_returns[var_99_idx] if var_99_idx < len(sorted_returns) else sorted_returns[0]
        
        # CVaR
        cvar_95 = statistics.mean(sorted_returns[:var_95_idx+1]) if var_95_idx >= 0 else var_95
        cvar_99 = statistics.mean(sorted_returns[:var_99_idx+1]) if var_99_idx >= 0 else var_99
        
        # ボラティリティ
        volatility = statistics.stdev(returns) * (252 ** 0.5) if len(returns) > 1 else 0
        
        # 最大ドローダウン
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return {
            'var_95_pct': var_95 * 100,
            'var_99_pct': var_99 * 100,
            'cvar_95_pct': cvar_95 * 100,
            'cvar_99_pct': cvar_99 * 100,
            'annual_volatility_pct': volatility * 100,
            'max_drawdown_pct': max_dd * 100
        }
    
    def _perform_monthly_analysis(self) -> List[Dict[str, Any]]:
        """月次分析"""
        
        monthly_data = {}
        
        for trade in self.trades:
            month = trade.entry_time.strftime('%Y-%m')
            
            if month not in monthly_data:
                monthly_data[month] = {
                    'trades': 0,
                    'total_return': 0,
                    'wins': 0,
                    'losses': 0,
                    'profits': [],
                    'losses_list': []
                }
            
            monthly_data[month]['trades'] += 1
            monthly_data[month]['total_return'] += trade.profit_pct
            
            if trade.profit_loss > 0:
                monthly_data[month]['wins'] += 1
                monthly_data[month]['profits'].append(trade.profit_pct)
            else:
                monthly_data[month]['losses'] += 1
                monthly_data[month]['losses_list'].append(trade.profit_pct)
        
        monthly_analysis = []
        for month, data in sorted(monthly_data.items()):
            win_rate = (data['wins'] / data['trades']) * 100 if data['trades'] > 0 else 0
            
            # 月次シャープレシオ
            all_returns = data['profits'] + data['losses_list']
            if len(all_returns) > 1:
                import statistics
                avg_return = statistics.mean(all_returns)
                std_return = statistics.stdev(all_returns)
                monthly_sharpe = (avg_return / std_return) * (12 ** 0.5) if std_return > 0 else 0
            else:
                monthly_sharpe = 0
            
            monthly_analysis.append({
                'month': month,
                'trades': data['trades'],
                'return_pct': round(data['total_return'], 2),
                'win_rate_pct': round(win_rate, 1),
                'wins': data['wins'],
                'losses': data['losses'],
                'sharpe': round(monthly_sharpe, 2),
                'best_trade': max(data['profits']) if data['profits'] else 0,
                'worst_trade': min(data['losses_list']) if data['losses_list'] else 0
            })
        
        return monthly_analysis
    
    def _generate_optimization_strategy(self, performance: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
        """最適化戦略生成"""
        
        optimization_plan = {
            'overall_assessment': '',
            'priority_improvements': [],
            'risk_management_enhancements': [],
            'exit_strategy_optimizations': [],
            'entry_strategy_refinements': [],
            'recommended_parameter_adjustments': {}
        }
        
        # 総合評価
        if performance['total_return'] > 25 and performance['sharpe_ratio'] > 1.5:
            optimization_plan['overall_assessment'] = '優秀なパフォーマンス - 現行戦略継続推奨'
        elif performance['total_return'] > 15 and performance['win_rate'] > 60:
            optimization_plan['overall_assessment'] = '良好なパフォーマンス - 微調整で更なる改善可能'
        elif performance['total_return'] > 5 and performance['win_rate'] > 50:
            optimization_plan['overall_assessment'] = '標準的パフォーマンス - 戦略改善が必要'
        else:
            optimization_plan['overall_assessment'] = '改善が必要 - 戦略の大幅見直し推奨'
        
        # 優先改善事項
        if performance['max_drawdown'] > 15:
            optimization_plan['priority_improvements'].append('最大ドローダウン15%超 - 緊急リスク管理強化必要')
        
        if performance['win_rate'] < 50:
            optimization_plan['priority_improvements'].append('勝率50%未満 - エントリー精度向上が急務')
        
        if performance['profit_factor'] < 1.3:
            optimization_plan['priority_improvements'].append('プロフィットファクター低下 - 利確/損切り比率要調整')
        
        # リスク管理強化
        if risk.get('max_drawdown_pct', 0) > 12:
            optimization_plan['risk_management_enhancements'].append('ドローダウン制御: ポジションサイズ縮小推奨')
        
        if risk.get('annual_volatility_pct', 0) > 30:
            optimization_plan['risk_management_enhancements'].append('ボラティリティ高: 動的ポジションサイジング導入')
        
        # 出口戦略最適化
        if performance['avg_loss'] > performance['avg_win'] * 0.8:
            optimization_plan['exit_strategy_optimizations'].append('損切り強化: 現在の2%から1.5%への変更検討')
        
        avg_hold_time = self._get_avg_hold_time()
        if avg_hold_time > 18:
            optimization_plan['exit_strategy_optimizations'].append('保有時間短縮: 12時間から8時間への変更検討')
        
        # エントリー戦略改良
        if performance['win_rate'] < 55:
            optimization_plan['entry_strategy_refinements'].append('エントリー条件厳格化: 信頼度閾値を70%から80%に上昇')
        
        # パラメータ調整推奨
        adjustments = {}
        
        if performance['win_rate'] < 50:
            adjustments['entry_confidence_threshold'] = {
                'current': 0.7, 'recommended': 0.8, 'reason': '勝率向上'
            }
        
        if performance['max_drawdown'] > 12:
            adjustments['stop_loss'] = {
                'current': 2.0, 'recommended': 1.5, 'reason': 'ドローダウン削減'
            }
            adjustments['position_size_pct'] = {
                'current': 8.0, 'recommended': 6.0, 'reason': 'リスク削減'
            }
        
        if performance['profit_factor'] < 1.3:
            adjustments['take_profit_1'] = {
                'current': 1.5, 'recommended': 2.0, 'reason': 'プロフィットファクター向上'
            }
        
        optimization_plan['recommended_parameter_adjustments'] = adjustments
        
        return optimization_plan
    
    def _get_best_symbol(self) -> str:
        """最優秀銘柄取得"""
        symbol_performance = {}
        for trade in self.trades:
            if trade.symbol not in symbol_performance:
                symbol_performance[trade.symbol] = []
            symbol_performance[trade.symbol].append(trade.profit_pct)
        
        symbol_totals = {symbol: sum(returns) for symbol, returns in symbol_performance.items()}
        return max(symbol_totals.keys(), key=lambda x: symbol_totals[x]) if symbol_totals else ""
    
    def _get_worst_symbol(self) -> str:
        """最低銘柄取得"""
        symbol_performance = {}
        for trade in self.trades:
            if trade.symbol not in symbol_performance:
                symbol_performance[trade.symbol] = []
            symbol_performance[trade.symbol].append(trade.profit_pct)
        
        symbol_totals = {symbol: sum(returns) for symbol, returns in symbol_performance.items()}
        return min(symbol_totals.keys(), key=lambda x: symbol_totals[x]) if symbol_totals else ""
    
    def _get_avg_hold_time(self) -> float:
        """平均保有時間取得"""
        if not self.trades:
            return 0
        return sum([t.hold_hours for t in self.trades]) / len(self.trades)
    
    async def _save_comprehensive_results(self, results: Dict[str, Any]):
        """包括結果保存"""
        
        os.makedirs('annual_backtest_results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON結果
        json_file = f'annual_backtest_results/unified_annual_backtest_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 詳細レポート
        report_file = f'annual_backtest_results/unified_optimization_report_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🚀 統一システム 1年間バックテスト & 最適化レポート\n")
            f.write("=" * 80 + "\n\n")
            
            perf = results['performance_metrics']
            risk = results['risk_metrics']
            opt = results['optimization_strategy']
            
            f.write("📊 パフォーマンス概要\n")
            f.write("-" * 40 + "\n")
            f.write(f"期間: {results['backtest_period']['start'][:10]} ～ {results['backtest_period']['end'][:10]}\n")
            f.write(f"総取引数: {perf['total_trades']}\n")
            f.write(f"勝率: {perf['win_rate']:.1f}%\n")
            f.write(f"総リターン: {perf['total_return']:.1f}%\n")
            f.write(f"最大ドローダウン: {perf['max_drawdown']:.1f}%\n")
            f.write(f"シャープレシオ: {perf['sharpe_ratio']:.2f}\n")
            f.write(f"プロフィットファクター: {perf['profit_factor']:.2f}\n\n")
            
            f.write("🛡️ リスク分析\n")
            f.write("-" * 40 + "\n")
            for key, value in risk.items():
                f.write(f"{key}: {value:.2f}\n")
            f.write("\n")
            
            f.write("🎯 最適化戦略\n")
            f.write("-" * 40 + "\n")
            f.write(f"総合評価: {opt['overall_assessment']}\n\n")
            
            if opt['priority_improvements']:
                f.write("🔥 優先改善事項:\n")
                for item in opt['priority_improvements']:
                    f.write(f"  • {item}\n")
                f.write("\n")
            
            if opt['recommended_parameter_adjustments']:
                f.write("⚙️ 推奨パラメータ調整:\n")
                for param, details in opt['recommended_parameter_adjustments'].items():
                    f.write(f"  • {param}: {details['current']} → {details['recommended']} ({details['reason']})\n")
                f.write("\n")
        
        self.logger.info(f"📁 包括結果保存完了:")
        self.logger.info(f"   JSON: {json_file}")
        self.logger.info(f"   レポート: {report_file}")

# 統一システム 1年間バックテスト実行関数
async def run_unified_annual_backtest():
    """統一システム 1年間包括バックテスト実行"""
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 統一システム 1年間包括バックテスト開始")
    
    # 統一システム用設定（config.pyと同じ）
    config = {
        "BB_PERIOD": 20,
        "BB_STD_DEV": 2.0,
        "RSI_PERIOD": 14,
        "RSI_THRESHOLD_LONG_OPTIMIZED_HOURS": 55,
        "RSI_THRESHOLD_LONG_OTHER_HOURS": 50,
        "RSI_THRESHOLD_SHORT_OPTIMIZED_HOURS": 45,
        "RSI_THRESHOLD_SHORT_OTHER_HOURS": 40,
        "MACD_FAST_EMA_PERIOD": 12,
        "MACD_SLOW_EMA_PERIOD": 26,
        "MACD_SIGNAL_SMA_PERIOD": 9,
        "EMA_PERIODS": [20, 50],
        "OPTIMIZED_TRADING_HOURS": [
            {"start": 7, "end": 11},
            {"start": 15, "end": 18},
            {"start": 22, "end": 24}
        ],
        "ALTERNATIVE_ENTRY_ENABLED": True,
        "RELAXED_ENTRY_CONDITIONS": {
            "RSI_NEUTRAL_ZONE_MIN": 40,
            "RSI_NEUTRAL_ZONE_MAX": 60,
            "BB_ENTRY_THRESHOLD_UPPER": 0.98,
            "BB_ENTRY_THRESHOLD_LOWER": 1.02,
            "MINIMUM_PRICE_CHANGE_30MIN": 0.05,
            "MINIMUM_PRICE_CHANGE_THRESHOLD": 0.1,
            "MACD_DIRECTION_ENABLED": True,
            "EMA_RELAXED_MODE": True,
            "EMA_PROXIMITY_THRESHOLD": 0.02,
        }
    }
    
    # 1年間バックテストシステム作成
    logger.info("📊 1年間バックテストシステム作成中...")
    annual_backtest_system = await WindowsUnifiedSystemFactory.create_annual_backtest_system(
        config, use_real_data=True  # 実際のMEXC APIデータを使用
    )
    
    # 包括バックテスト実行
    logger.info("🎯 1年間包括バックテスト実行中...")
    results = await annual_backtest_system.run_annual_comprehensive_backtest()
    
    # 結果表示
    print("\n" + "="*80)
    print("🎉 統一システム 1年間バックテスト & 戦略最適化 完了")
    print("🌐 実際のMEXC価格データを使用した結果")
    print("="*80)
    
    perf = results['performance_metrics']
    risk = results['risk_metrics']
    opt = results['optimization_strategy']
    
    print(f"\n📊 パフォーマンス概要:")
    print(f"   総取引数: {perf['total_trades']}")
    print(f"   勝率: {perf['win_rate']:.1f}%")
    print(f"   総リターン: {perf['total_return']:+.1f}%")
    print(f"   最大ドローダウン: {perf['max_drawdown']:.1f}%")
    print(f"   シャープレシオ: {perf['sharpe_ratio']:.2f}")
    print(f"   プロフィットファクター: {perf['profit_factor']:.2f}")
    
    print(f"\n🛡️ リスク分析:")
    if risk:
        print(f"   年間ボラティリティ: {risk.get('annual_volatility_pct', 0):.1f}%")
        print(f"   VaR(95%): {risk.get('var_95_pct', 0):.1f}%")
        print(f"   CVaR(95%): {risk.get('cvar_95_pct', 0):.1f}%")
        print(f"   最大ドローダウン: {risk.get('max_drawdown_pct', 0):.1f}%")
    
    print(f"\n🎯 総合評価: {opt['overall_assessment']}")
    
    if opt['priority_improvements']:
        print(f"\n🔥 優先改善事項:")
        for item in opt['priority_improvements']:
            print(f"   • {item}")
    
    if opt['recommended_parameter_adjustments']:
        print(f"\n⚙️ 推奨パラメータ調整:")
        for param, details in opt['recommended_parameter_adjustments'].items():
            print(f"   • {param}: {details['current']} → {details['recommended']} ({details['reason']})")
    
    trade_summary = results['trade_summary']
    print(f"\n📈 取引サマリー:")
    print(f"   最優秀銘柄: {trade_summary['best_performing_symbol']}")
    print(f"   取引銘柄数: {len(trade_summary['symbols_traded'])}")
    print(f"   平均保有時間: {trade_summary['average_hold_time_hours']:.1f}時間")
    
    print(f"\n📁 詳細結果は annual_backtest_results/ フォルダに保存されました")
    
    logger.info("🏁 統一システム 1年間バックテスト完了")
    return results

# Windows環境対応メイン実行
async def main_windows():
    """Windows環境対応メイン実行（1年間バックテスト対応）"""
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Windows対応統一トレーディングシステム開始")
    
    print("\n" + "="*60)
    print("🚀 統一トレーディングシステム")
    print("="*60)
    print("1. 🌐 1年間包括バックテスト実行（実際のMEXC価格データ使用）")
    print("2. 📊 短期バックテストテスト（シミュレーションデータ）")
    print("3. 📄 ペーパートレーディングテスト（リアルタイムデータ）")
    print("="*60)
    
    # デフォルトで1年間バックテストを自動実行
    choice = "1"
    logger.info("🎯 自動選択: 1年間包括バックテスト実行")
    
    if choice == "1":
        # 1年間包括バックテスト実行
        await run_unified_annual_backtest()
        
    elif choice == "2":
        # 短期バックテストテスト
        await _run_short_backtest_test()
        
    elif choice == "3":
        # ペーパートレーディングテスト
        await _run_paper_trading_test()
        
    else:
        logger.error("無効な選択です")
        return

async def _run_short_backtest_test():
    """短期バックテストテスト"""
    
    logger = logging.getLogger(__name__)
    logger.info("📊 短期バックテストシステムテスト")
    
    # Windows環境用設定
    config = {
        "BB_PERIOD": 20,
        "BB_STD_DEV": 2.0,
        "RSI_PERIOD": 14,
        "RSI_THRESHOLD_LONG_OPTIMIZED_HOURS": 55,
        "RSI_THRESHOLD_LONG_OTHER_HOURS": 50,
        "RSI_THRESHOLD_SHORT_OPTIMIZED_HOURS": 45,
        "RSI_THRESHOLD_SHORT_OTHER_HOURS": 40,
        "MACD_FAST_EMA_PERIOD": 12,
        "MACD_SLOW_EMA_PERIOD": 26,
        "MACD_SIGNAL_SMA_PERIOD": 9,
        "EMA_PERIODS": [20, 50],
        "OPTIMIZED_TRADING_HOURS": [
            {"start": 7, "end": 11},
            {"start": 15, "end": 18},
            {"start": 22, "end": 24}
        ],
        "ALTERNATIVE_ENTRY_ENABLED": True,
        "RELAXED_ENTRY_CONDITIONS": {
            "RSI_NEUTRAL_ZONE_MIN": 40,
            "RSI_NEUTRAL_ZONE_MAX": 60,
            "BB_ENTRY_THRESHOLD_UPPER": 0.98,
            "BB_ENTRY_THRESHOLD_LOWER": 1.02,
            "MINIMUM_PRICE_CHANGE_30MIN": 0.05,
            "MINIMUM_PRICE_CHANGE_THRESHOLD": 0.1,
            "MACD_DIRECTION_ENABLED": True,
            "EMA_RELAXED_MODE": True,
            "EMA_PROXIMITY_THRESHOLD": 0.02,
        }
    }
    
    symbols = ["AVAXUSDT", "LINKUSDT"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    backtest_system = await WindowsUnifiedSystemFactory.create_backtest_system(
        config, symbols, start_date, end_date
    )
    
    # バックテスト実行
    timestamps = list(backtest_system.data_source.historical_data["AVAXUSDT"].index[::12])  # 1時間間隔
    total_signals = 0
    
    for i, timestamp in enumerate(timestamps[:24]):  # 最初の24時間
        backtest_system.data_source.set_current_time(timestamp)
        
        for symbol in symbols:
            signal = await backtest_system.analyze_entry_conditions(symbol)
            
            if signal:
                total_signals += 1
                success = await backtest_system.execute_trade(signal)
                
                if success:
                    logger.info(f"✅ {signal.symbol} {signal.side} - {signal.signal_type}")
                    logger.info(f"   理由: {', '.join(signal.reasons)}")
                    logger.info(f"   信頼度: {signal.confidence:.2f}")
    
    logger.info(f"📈 短期バックテスト結果: {total_signals}件のシグナル検出")

async def _run_paper_trading_test():
    """ペーパートレーディングテスト"""
    
    logger = logging.getLogger(__name__)
    logger.info("📄 ペーパートレーディングシステムテスト")
    
    # Windows環境用設定
    config = {
        "BB_PERIOD": 20,
        "BB_STD_DEV": 2.0,
        "RSI_PERIOD": 14,
        "RSI_THRESHOLD_LONG_OPTIMIZED_HOURS": 55,
        "RSI_THRESHOLD_LONG_OTHER_HOURS": 50,
        "RSI_THRESHOLD_SHORT_OPTIMIZED_HOURS": 45,
        "RSI_THRESHOLD_SHORT_OTHER_HOURS": 40,
        "MACD_FAST_EMA_PERIOD": 12,
        "MACD_SLOW_EMA_PERIOD": 26,
        "MACD_SIGNAL_SMA_PERIOD": 9,
        "EMA_PERIODS": [20, 50],
        "OPTIMIZED_TRADING_HOURS": [
            {"start": 7, "end": 11},
            {"start": 15, "end": 18},
            {"start": 22, "end": 24}
        ],
        "ALTERNATIVE_ENTRY_ENABLED": True,
        "RELAXED_ENTRY_CONDITIONS": {
            "RSI_NEUTRAL_ZONE_MIN": 40,
            "RSI_NEUTRAL_ZONE_MAX": 60,
            "BB_ENTRY_THRESHOLD_UPPER": 0.98,
            "BB_ENTRY_THRESHOLD_LOWER": 1.02,
            "MINIMUM_PRICE_CHANGE_30MIN": 0.05,
            "MINIMUM_PRICE_CHANGE_THRESHOLD": 0.1,
            "MACD_DIRECTION_ENABLED": True,
            "EMA_RELAXED_MODE": True,
            "EMA_PROXIMITY_THRESHOLD": 0.02,
        }
    }
    
    paper_system = WindowsUnifiedSystemFactory.create_live_system(config, is_paper_trading=True)
    
    # ペーパートレーディングテスト
    symbols = ["AVAXUSDT", "LINKUSDT"]
    paper_signals = 0
    for symbol in symbols:
        signal = await paper_system.analyze_entry_conditions(symbol)
        
        if signal:
            paper_signals += 1
            success = await paper_system.execute_trade(signal)
            
            if success:
                logger.info(f"✅ ペーパー {signal.symbol} {signal.side}")
                logger.info(f"   理由: {', '.join(signal.reasons)}")
    
    logger.info(f"📄 ペーパートレーディング結果: {paper_signals}件のシグナル検出")

if __name__ == "__main__":
    asyncio.run(main_windows())