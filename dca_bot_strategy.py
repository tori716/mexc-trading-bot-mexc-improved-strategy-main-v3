#!/usr/bin/env python3
"""
DCA Bot戦略 - 調査報告書準拠実装
期待利益率: 年間18-35%（実績例193% ROI）
ドルコスト平均法による段階的買い増し戦略
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
        logging.FileHandler('dca_bot_strategy.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

@dataclass
class DCAOrder:
    """DCAオーダー定義"""
    order_id: int
    order_type: str  # 'base' or 'safety'
    price: float
    quantity: float
    deviation_pct: float  # 価格乖離率
    is_filled: bool = False
    fill_time: Optional[datetime] = None
    multiplier: float = 1.0  # ポジション倍数

@dataclass
class DCAPosition:
    """DCAポジション管理"""
    symbol: str
    entry_time: datetime
    base_order: DCAOrder
    safety_orders: List[DCAOrder]
    average_price: float = 0.0
    total_quantity: float = 0.0
    total_investment: float = 0.0
    target_profit_pct: float = 0.0
    max_safety_orders: int = 0
    is_active: bool = True

@dataclass
class DCAConfiguration:
    """DCA設定"""
    symbol: str
    base_order_size: float  # 基本オーダーサイズ
    safety_order_size: float  # セーフティオーダーサイズ
    max_safety_orders: int  # 最大セーフティオーダー数
    price_deviation_pct: float  # 価格乖離率（各レベル）
    take_profit_pct: float  # 利確率
    safety_order_multiplier: float  # セーフティオーダー倍数
    max_investment: float  # 最大投資額

class DCABotStrategy:
    """DCA Bot戦略（調査報告書準拠）"""
    
    def __init__(self, config: Dict, data_source: WindowsDataSource, execution_mode: ExecutionMode):
        self.config = config
        self.data_source = data_source
        self.execution_mode = execution_mode
        self.logger = logging.getLogger(__name__)
        
        # DCA管理
        self.active_positions: Dict[str, DCAPosition] = {}
        self.trade_history = []
        
        # 調査報告書準拠の設定
        self.dca_config = {
            "BASE_ORDER_PCT": 2.0,           # 総資金の2%（1-3%の中央値）
            "SAFETY_ORDER_PCT": 1.5,        # セーフティオーダー率
            "MAX_SAFETY_ORDERS": 6,          # 最大セーフティオーダー数（3-10の中央値）
            "PRICE_DEVIATION_PCT": 3.0,      # 価格乖離率（5-10%の下限より攻撃的）
            "TAKE_PROFIT_PCT": 5.0,          # 利確率（3-8%の中央値）
            "SAFETY_ORDER_MULTIPLIER": 2.0,  # セーフティオーダー倍数（2-3倍の下限）
            "MAX_SIMULTANEOUS_DCAS": 4,      # 同時DCA数制限
            "TREND_CONFIRMATION_PERIOD": 20, # トレンド確認期間
            "VOLUME_CONFIRMATION": 1.3,      # 出来高確認倍数
            "MA_DEVIATION_THRESHOLD": 7.0,   # 移動平均乖離閾値（5-10%）
            "MAX_INVESTMENT_PER_SYMBOL": 3000.0  # 銘柄あたり最大投資額
        }
    
    async def analyze_dca_opportunity(self, symbol: str) -> Optional[DCAConfiguration]:
        """DCA機会分析"""
        
        try:
            # 市場データ取得
            ohlcv_data = await self.data_source.get_ohlcv(symbol, "60m", 100)
            
            if not ohlcv_data or len(ohlcv_data) < 50:
                return None
            
            df = pd.DataFrame(ohlcv_data)
            current_price = await self.data_source.get_current_price(symbol)
            
            # 移動平均と価格乖離分析
            ma_period = self.dca_config["TREND_CONFIRMATION_PERIOD"]
            df['ma'] = df['close'].rolling(window=ma_period).mean()
            current_ma = df['ma'].iloc[-1]
            
            if pd.isna(current_ma):
                return None
            
            # 価格が移動平均から乖離しているかチェック
            price_deviation = ((current_price - current_ma) / current_ma) * 100
            
            # DCA適用条件：価格が移動平均より大幅に下落
            if price_deviation > -self.dca_config["MA_DEVIATION_THRESHOLD"]:
                self.logger.info(f"{symbol}: 価格乖離不足({price_deviation:.2f}%) - DCA開始条件未達")
                return None
            
            # トレンド確認
            if not self._is_suitable_for_dca(df, current_price):
                return None
            
            # DCA設定計算
            dca_config = self._calculate_dca_configuration(symbol, current_price)
            
            self.logger.info(f"💰 {symbol} DCA設定: 乖離{price_deviation:.2f}%, セーフティ{dca_config.max_safety_orders}レベル")
            return dca_config
            
        except Exception as e:
            self.logger.error(f"DCA機会分析エラー {symbol}: {str(e)}")
            return None
    
    def _is_suitable_for_dca(self, df: pd.DataFrame, current_price: float) -> bool:
        """DCA適用条件判定"""
        
        if len(df) < 20:
            return False
        
        # 下降トレンド確認（DCAに適した条件）
        short_ma = df['close'].rolling(window=10).mean().iloc[-1]
        long_ma = df['close'].rolling(window=20).mean().iloc[-1]
        
        # 出来高確認
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # ボラティリティ確認（DCAに適度なボラティリティが必要）
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(window=10).std().iloc[-1]
        
        # DCA適用条件
        conditions = [
            current_price < short_ma,  # 短期移動平均より下
            short_ma < long_ma,        # 下降トレンド
            volume_ratio > self.dca_config["VOLUME_CONFIRMATION"],  # 出来高確認
            volatility > 0.01,         # 適度なボラティリティ
            volatility < 0.1           # 過度なボラティリティは回避
        ]
        
        suitability_score = sum(conditions) / len(conditions)
        is_suitable = suitability_score >= 0.6  # 60%以上で適用
        
        self.logger.info(f"DCA適用判定: スコア{suitability_score:.2f}, 判定={is_suitable}")
        return is_suitable
    
    def _calculate_dca_configuration(self, symbol: str, current_price: float) -> DCAConfiguration:
        """DCA設定計算"""
        
        # 基本オーダーサイズ（調査報告書：総資金の1-3%）
        base_order_size = 1000 * (self.dca_config["BASE_ORDER_PCT"] / 100)
        
        # セーフティオーダーサイズ
        safety_order_size = 1000 * (self.dca_config["SAFETY_ORDER_PCT"] / 100)
        
        # セーフティオーダー数
        max_safety_orders = self.dca_config["MAX_SAFETY_ORDERS"]
        
        # 価格乖離率（段階的に増加）
        price_deviation = self.dca_config["PRICE_DEVIATION_PCT"]
        
        # 利確率
        take_profit = self.dca_config["TAKE_PROFIT_PCT"]
        
        # セーフティオーダー倍数
        multiplier = self.dca_config["SAFETY_ORDER_MULTIPLIER"]
        
        # 最大投資額計算
        total_investment = base_order_size
        for i in range(max_safety_orders):
            total_investment += safety_order_size * (multiplier ** i)
        
        max_investment = min(total_investment, self.dca_config["MAX_INVESTMENT_PER_SYMBOL"])
        
        return DCAConfiguration(
            symbol=symbol,
            base_order_size=base_order_size,
            safety_order_size=safety_order_size,
            max_safety_orders=max_safety_orders,
            price_deviation_pct=price_deviation,
            take_profit_pct=take_profit,
            safety_order_multiplier=multiplier,
            max_investment=max_investment
        )
    
    async def setup_dca_position(self, config: DCAConfiguration) -> bool:
        """DCAポジション設定"""
        
        try:
            current_price = await self.data_source.get_current_price(config.symbol)
            current_time = self.data_source.get_current_time()
            
            # ベースオーダー作成
            base_order = DCAOrder(
                order_id=1,
                order_type='base',
                price=current_price,
                quantity=config.base_order_size / current_price,
                deviation_pct=0.0,
                multiplier=1.0
            )
            
            # ベースオーダー実行
            order_result = await self.data_source.place_order(
                symbol=config.symbol,
                side="BUY",
                order_type="MARKET",
                quantity=base_order.quantity
            )
            
            if order_result.get("status") == "FILLED":
                base_order.is_filled = True
                base_order.fill_time = current_time
                
                # セーフティオーダー準備
                safety_orders = []
                for i in range(config.max_safety_orders):
                    deviation = config.price_deviation_pct * (i + 1)
                    target_price = current_price * (1 - deviation / 100)
                    multiplier = config.safety_order_multiplier ** i
                    quantity = (config.safety_order_size * multiplier) / target_price
                    
                    safety_order = DCAOrder(
                        order_id=i + 2,
                        order_type='safety',
                        price=target_price,
                        quantity=quantity,
                        deviation_pct=deviation,
                        multiplier=multiplier
                    )
                    safety_orders.append(safety_order)
                
                # DCAポジション作成
                dca_position = DCAPosition(
                    symbol=config.symbol,
                    entry_time=current_time,
                    base_order=base_order,
                    safety_orders=safety_orders,
                    average_price=current_price,
                    total_quantity=base_order.quantity,
                    total_investment=config.base_order_size,
                    target_profit_pct=config.take_profit_pct,
                    max_safety_orders=config.max_safety_orders
                )
                
                # ポジション管理に追加
                self.active_positions[config.symbol] = dca_position
                
                self.logger.info(f"✅ {config.symbol} DCAポジション開始: ベース${config.base_order_size:.0f}")
                return True
            else:
                self.logger.error(f"❌ {config.symbol} ベースオーダー失敗")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ DCAポジション設定エラー {config.symbol}: {str(e)}")
            return False
    
    async def manage_dca_position(self, symbol: str) -> List[TradeResult]:
        """DCAポジション管理"""
        
        if symbol not in self.active_positions:
            return []
        
        trades = []
        position = self.active_positions[symbol]
        current_price = await self.data_source.get_current_price(symbol)
        current_time = self.data_source.get_current_time()
        
        # セーフティオーダーの約定チェック
        for safety_order in position.safety_orders:
            if not safety_order.is_filled and current_price <= safety_order.price:
                # セーフティオーダー約定
                order_result = await self.data_source.place_order(
                    symbol=symbol,
                    side="BUY",
                    order_type="MARKET",
                    quantity=safety_order.quantity
                )
                
                if order_result.get("status") == "FILLED":
                    safety_order.is_filled = True
                    safety_order.fill_time = current_time
                    
                    # 平均取得単価更新
                    total_cost = position.total_investment + (safety_order.quantity * current_price)
                    total_quantity = position.total_quantity + safety_order.quantity
                    position.average_price = total_cost / total_quantity
                    position.total_quantity = total_quantity
                    position.total_investment = total_cost
                    
                    self.logger.info(f"📉 {symbol} セーフティオーダー{safety_order.order_id}約定: ${current_price:.4f}")
                    self.logger.info(f"   平均単価更新: ${position.average_price:.4f}")
        
        # 利確条件チェック
        if current_price >= position.average_price * (1 + position.target_profit_pct / 100):
            # 利確実行
            profit_trade = await self._execute_dca_exit(position, current_price, current_time, "利確")
            if profit_trade:
                trades.append(profit_trade)
                del self.active_positions[symbol]
        
        # ストップロス条件チェック（全セーフティオーダー消化後）
        active_safety_orders = [so for so in position.safety_orders if not so.is_filled]
        if not active_safety_orders:
            # 最大損失からの復帰チェック（時間ベース）
            hold_hours = (current_time - position.entry_time).total_seconds() / 3600
            if hold_hours > 48:  # 48時間経過で強制決済
                exit_trade = await self._execute_dca_exit(position, current_price, current_time, "時間切れ")
                if exit_trade:
                    trades.append(exit_trade)
                    del self.active_positions[symbol]
        
        return trades
    
    async def _execute_dca_exit(self, position: DCAPosition, current_price: float, 
                              current_time: datetime, exit_reason: str) -> Optional[TradeResult]:
        """DCA決済実行"""
        
        try:
            # 全ポジション決済
            order_result = await self.data_source.place_order(
                symbol=position.symbol,
                side="SELL",
                order_type="MARKET",
                quantity=position.total_quantity
            )
            
            if order_result.get("status") == "FILLED":
                # 損益計算
                total_revenue = position.total_quantity * current_price
                profit_loss = total_revenue - position.total_investment
                profit_pct = (profit_loss / position.total_investment) * 100
                
                # 保有時間計算
                hold_hours = (current_time - position.entry_time).total_seconds() / 3600
                
                trade = TradeResult(
                    symbol=position.symbol,
                    entry_time=position.entry_time,
                    exit_time=current_time,
                    side="BUY",  # DCAは基本的にロング戦略
                    entry_price=position.average_price,
                    exit_price=current_price,
                    quantity=position.total_quantity,
                    profit_loss=profit_loss,
                    profit_pct=profit_pct,
                    hold_hours=hold_hours,
                    exit_reason=f"DCA_{exit_reason}"
                )
                
                self.trade_history.append(trade)
                
                # セーフティオーダー使用数
                filled_safety_orders = len([so for so in position.safety_orders if so.is_filled])
                
                self.logger.info(f"💰 DCA決済: {position.symbol} {exit_reason}")
                self.logger.info(f"   平均単価: ${position.average_price:.4f} → 決済: ${current_price:.4f}")
                self.logger.info(f"   利益: ${profit_loss:.2f} ({profit_pct:+.1f}%)")
                self.logger.info(f"   セーフティオーダー使用: {filled_safety_orders}/{len(position.safety_orders)}")
                
                return trade
            else:
                self.logger.error(f"❌ DCA決済失敗: {position.symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"DCA決済エラー: {str(e)}")
            return None

class DCABotBacktestSystem(AnnualBacktestSystem):
    """DCA Botバックテストシステム"""
    
    def __init__(self, config: Dict, historical_data: Dict[str, pd.DataFrame], 
                 symbols: List[str], start_date: datetime, end_date: datetime):
        super().__init__(config, historical_data, symbols, start_date, end_date)
        
        self.dca_strategy = DCABotStrategy(
            config, WindowsDataSource(ExecutionMode.BACKTEST, historical_data), ExecutionMode.BACKTEST
        )
        
        # DCA Bot専用設定
        self.enhanced_config.update({
            "STRATEGY_NAME": "DCA Bot戦略",
            "EXPECTED_ANNUAL_RETURN": 25.0,  # 18-35%の中央値
            "MAX_SIMULTANEOUS_DCAS": 4,      # 同時DCA数制限
            "DCA_REBALANCE_INTERVAL": 6,     # 6時間ごとチェック
        })
    
    async def _execute_annual_backtest(self):
        """DCA Bot年間バックテスト実行"""
        
        capital = self.enhanced_config["INITIAL_CAPITAL"]
        
        # 時間ステップ（6時間ごと）
        timestamps = list(self.historical_data[self.symbols[0]].index[::6])
        
        for i, timestamp in enumerate(timestamps):
            try:
                # データソース時刻設定
                self.dca_strategy.data_source.set_current_time(timestamp)
                current_portfolio_value = capital
                
                # 既存DCAポジション管理
                for symbol in list(self.dca_strategy.active_positions.keys()):
                    trades = await self.dca_strategy.manage_dca_position(symbol)
                    
                    for trade in trades:
                        capital += trade.profit_loss + self.dca_strategy.active_positions.get(symbol, 
                                     type('obj', (object,), {'total_investment': 0})).total_investment
                        self.trades.append(trade)
                        current_portfolio_value += trade.profit_loss
                
                # 新規DCA機会検索
                active_dcas = len(self.dca_strategy.active_positions)
                if active_dcas < self.enhanced_config["MAX_SIMULTANEOUS_DCAS"]:
                    
                    for symbol in self.symbols:
                        if symbol not in self.dca_strategy.active_positions:
                            dca_config = await self.dca_strategy.analyze_dca_opportunity(symbol)
                            
                            if dca_config and capital > dca_config.max_investment:
                                success = await self.dca_strategy.setup_dca_position(dca_config)
                                
                                if success:
                                    capital -= dca_config.base_order_size
                                    self.logger.info(f"💰 {symbol} DCA開始: 投資額${dca_config.base_order_size}")
                                    break  # 1回につき1DCAまで
                
                # ポートフォリオ価値計算
                dca_investment = sum([
                    pos.total_investment for pos in self.dca_strategy.active_positions.values()
                ])
                portfolio_value = capital + dca_investment
                
                # 日次記録
                if i % 4 == 0:  # 24時間ごと
                    self.daily_portfolio.append({
                        'timestamp': timestamp,
                        'portfolio_value': portfolio_value,
                        'cash': capital,
                        'positions': len(self.dca_strategy.active_positions),
                        'return_pct': ((portfolio_value - self.enhanced_config["INITIAL_CAPITAL"]) / 
                                     self.enhanced_config["INITIAL_CAPITAL"]) * 100
                    })
                
                # 進捗表示
                if i % 168 == 0:  # 週次
                    progress = (i / len(timestamps)) * 100
                    weeks = i // 28
                    active_dcas = len(self.dca_strategy.active_positions)
                    self.logger.info(f"  進捗: {progress:.1f}% ({weeks}週経過) アクティブDCA:{active_dcas}")
                
            except Exception as e:
                self.logger.warning(f"タイムスタンプ {timestamp} でエラー: {str(e)}")
                continue

# メイン実行関数
async def run_dca_bot_backtest():
    """DCA Bot戦略バックテスト実行"""
    
    logger = logging.getLogger(__name__)
    logger.info("💰 DCA Bot戦略 1年間バックテスト開始")
    
    # DCA Bot用設定
    config = {
        "STRATEGY_TYPE": "DCA_BOT",
        "BASE_ORDER_PCT": 2.0,
        "SAFETY_ORDER_PCT": 1.5,
        "MAX_SAFETY_ORDERS": 6,
        "PRICE_DEVIATION_PCT": 3.0,
        "TAKE_PROFIT_PCT": 5.0,
        "SAFETY_ORDER_MULTIPLIER": 2.0,
        "MAX_INVESTMENT_PER_SYMBOL": 3000.0
    }
    
    # 1年間バックテストシステム作成
    logger.info("📊 DCA Botバックテストシステム作成中...")
    annual_system = await WindowsUnifiedSystemFactory.create_annual_backtest_system(
        config, use_real_data=True
    )
    
    # DCA Botシステムに変換
    dca_system = DCABotBacktestSystem(
        config, annual_system.historical_data, annual_system.symbols,
        annual_system.start_date, annual_system.end_date
    )
    
    # バックテスト実行
    logger.info("💰 DCA Botバックテスト実行中...")
    results = await dca_system.run_annual_comprehensive_backtest()
    
    # 結果表示
    print("\n" + "="*80)
    print("💰 DCA Bot戦略 1年間バックテスト完了")
    print("📊 調査報告書準拠実装（期待年利18-35%）")
    print("="*80)
    
    perf = results['performance_metrics']
    print(f"\n📈 パフォーマンス結果:")
    print(f"   戦略タイプ: DCA Bot（ドルコスト平均法）")
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
    
    if perf['total_return'] >= 18.0:
        print("✅ 調査報告書期待値（年18-35%）達成")
    else:
        print("❌ 調査報告書期待値未達成")
    
    # グリッド戦略との比較
    print(f"\n📊 戦略比較（vs グリッド取引）:")
    print(f"   グリッド: +0.2% (勝率100%, ドローダウン0%)")
    print(f"   DCA Bot: {perf['total_return']:+.1f}% (勝率{perf['win_rate']:.1f}%, ドローダウン{perf['max_drawdown']:.1f}%)")
    
    return results

async def main():
    """メイン実行"""
    await run_dca_bot_backtest()

if __name__ == "__main__":
    asyncio.run(main())