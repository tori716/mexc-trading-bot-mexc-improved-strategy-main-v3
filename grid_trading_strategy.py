#!/usr/bin/env python3
"""
グリッド取引戦略 - 調査報告書準拠実装
期待利益率: 年間15-25%（レンジ相場で安定）
MEXCバッチ注文機能対応版
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
        logging.FileHandler('grid_trading_strategy.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

@dataclass
class GridLevel:
    """グリッドレベル定義"""
    price: float
    level_id: int
    order_type: str  # 'buy' or 'sell'
    is_filled: bool = False
    order_id: Optional[str] = None
    quantity: float = 0.0

@dataclass
class GridConfiguration:
    """グリッド設定"""
    symbol: str
    center_price: float
    grid_spacing_pct: float  # グリッド間隔（%）
    total_levels: int  # 総グリッド数
    upper_range_pct: float  # 上限範囲（%）
    lower_range_pct: float  # 下限範囲（%）
    total_investment: float  # 総投資額
    stop_loss_pct: float  # ストップロス（%）

class GridTradingStrategy:
    """グリッド取引戦略（調査報告書準拠）"""
    
    def __init__(self, config: Dict, data_source: WindowsDataSource, execution_mode: ExecutionMode):
        self.config = config
        self.data_source = data_source
        self.execution_mode = execution_mode
        self.logger = logging.getLogger(__name__)
        
        # グリッド管理
        self.active_grids: Dict[str, List[GridLevel]] = {}
        self.grid_configs: Dict[str, GridConfiguration] = {}
        self.trade_history = []
        
        # 調査報告書準拠の設定
        self.grid_config = {
            "GRID_SPACING_MIN": 0.5,  # 最小グリッド間隔（%）
            "GRID_SPACING_MAX": 2.0,  # 最大グリッド間隔（%）
            "UPPER_RANGE": 15.0,      # 上限範囲（%）
            "LOWER_RANGE": 15.0,      # 下限範囲（%）
            "MIN_GRID_LEVELS": 10,    # 最小グリッド数
            "MAX_GRID_LEVELS": 30,    # 最大グリッド数（MEXC API制限考慮）
            "STOP_LOSS_RANGE": 20.0,  # ストップロス範囲（%）
            "VOLATILITY_LOOKBACK": 20, # ボラティリティ計算期間
            "REBALANCE_THRESHOLD": 5.0, # リバランス閾値（%）
            "MAX_INVESTMENT_PER_SYMBOL": 2000.0  # 銘柄あたり最大投資額
        }
    
    async def analyze_grid_opportunity(self, symbol: str) -> Optional[GridConfiguration]:
        """グリッド設定機会分析"""
        
        try:
            # 市場データ取得
            ohlcv_data = await self.data_source.get_ohlcv(symbol, "60m", 100)
            
            if not ohlcv_data or len(ohlcv_data) < 50:
                return None
            
            df = pd.DataFrame(ohlcv_data)
            current_price = await self.data_source.get_current_price(symbol)
            
            # ボラティリティベースのグリッド間隔計算
            volatility = self._calculate_volatility(df)
            grid_spacing = self._calculate_optimal_grid_spacing(volatility)
            
            # レンジ相場検出
            if not self._is_ranging_market(df):
                self.logger.info(f"{symbol}: トレンド相場のため、グリッド取引に不適")
                return None
            
            # グリッド設定計算
            grid_config = self._calculate_grid_configuration(
                symbol, current_price, grid_spacing, volatility
            )
            
            self.logger.info(f"📊 {symbol} グリッド設定: 間隔{grid_spacing:.2f}%, レベル数{grid_config.total_levels}")
            return grid_config
            
        except Exception as e:
            self.logger.error(f"グリッド機会分析エラー {symbol}: {str(e)}")
            return None
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """ボラティリティ計算（調査報告書準拠）"""
        
        # 価格変動率計算
        returns = df['close'].pct_change().dropna()
        
        # 20期間の標準偏差（年率換算）
        volatility = returns.rolling(
            window=self.grid_config["VOLATILITY_LOOKBACK"]
        ).std().iloc[-1]
        
        # パーセンテージ変換
        volatility_pct = volatility * 100
        
        return volatility_pct if not pd.isna(volatility_pct) else 1.0
    
    def _calculate_optimal_grid_spacing(self, volatility: float) -> float:
        """最適グリッド間隔計算（ボラティリティベース）"""
        
        # ボラティリティに基づく動的グリッド間隔
        if volatility < 1.0:
            # 低ボラティリティ: 狭い間隔
            spacing = self.grid_config["GRID_SPACING_MIN"]
        elif volatility > 3.0:
            # 高ボラティリティ: 広い間隔
            spacing = self.grid_config["GRID_SPACING_MAX"]
        else:
            # 中間ボラティリティ: 比例調整
            spacing = self.grid_config["GRID_SPACING_MIN"] + (
                (volatility - 1.0) / 2.0 * 
                (self.grid_config["GRID_SPACING_MAX"] - self.grid_config["GRID_SPACING_MIN"])
            )
        
        return spacing
    
    def _is_ranging_market(self, df: pd.DataFrame) -> bool:
        """レンジ相場検出（グリッド取引適性判定）"""
        
        if len(df) < 20:
            return False
        
        # 価格レンジ分析
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_price = df['close'].iloc[-1]
        
        # レンジ幅
        range_pct = (recent_high - recent_low) / current_price * 100
        
        # トレンド強度分析（ADX相当）
        price_changes = df['close'].diff().abs()
        trend_strength = price_changes.tail(10).mean() / current_price * 100
        
        # レンジ相場判定条件
        is_ranging = (
            range_pct > 8.0 and  # 適度なレンジ幅
            range_pct < 25.0 and  # 過度に広くない
            trend_strength < 2.0   # 弱いトレンド
        )
        
        self.logger.info(f"レンジ分析: 幅{range_pct:.1f}%, トレンド強度{trend_strength:.2f}%, レンジ判定={is_ranging}")
        return is_ranging
    
    def _calculate_grid_configuration(self, symbol: str, current_price: float, 
                                    grid_spacing: float, volatility: float) -> GridConfiguration:
        """グリッド設定計算"""
        
        # レンジ設定（ボラティリティベース）
        if volatility < 1.5:
            upper_range = self.grid_config["UPPER_RANGE"] * 0.8
            lower_range = self.grid_config["LOWER_RANGE"] * 0.8
        elif volatility > 2.5:
            upper_range = self.grid_config["UPPER_RANGE"] * 1.2
            lower_range = self.grid_config["LOWER_RANGE"] * 1.2
        else:
            upper_range = self.grid_config["UPPER_RANGE"]
            lower_range = self.grid_config["LOWER_RANGE"]
        
        # グリッドレベル数計算
        total_range = upper_range + lower_range
        estimated_levels = int(total_range / grid_spacing)
        
        # レベル数制限
        total_levels = max(
            self.grid_config["MIN_GRID_LEVELS"],
            min(estimated_levels, self.grid_config["MAX_GRID_LEVELS"])
        )
        
        # 実際のグリッド間隔調整
        actual_spacing = total_range / total_levels
        
        # 投資額計算（資金管理）
        total_investment = min(
            self.grid_config["MAX_INVESTMENT_PER_SYMBOL"],
            1000.0  # デフォルト投資額
        )
        
        return GridConfiguration(
            symbol=symbol,
            center_price=current_price,
            grid_spacing_pct=actual_spacing,
            total_levels=total_levels,
            upper_range_pct=upper_range,
            lower_range_pct=lower_range,
            total_investment=total_investment,
            stop_loss_pct=self.grid_config["STOP_LOSS_RANGE"]
        )
    
    def _create_grid_levels(self, config: GridConfiguration) -> List[GridLevel]:
        """グリッドレベル作成"""
        
        grid_levels = []
        
        # 中心価格から上下にグリッドを配置
        upper_levels = int(config.total_levels * config.upper_range_pct / 
                          (config.upper_range_pct + config.lower_range_pct))
        lower_levels = config.total_levels - upper_levels
        
        # 上方向グリッド（売り注文）
        for i in range(1, upper_levels + 1):
            price = config.center_price * (1 + (config.grid_spacing_pct * i) / 100)
            grid_levels.append(GridLevel(
                price=price,
                level_id=i,
                order_type='sell',
                quantity=config.total_investment / config.total_levels / price
            ))
        
        # 下方向グリッド（買い注文）
        for i in range(1, lower_levels + 1):
            price = config.center_price * (1 - (config.grid_spacing_pct * i) / 100)
            grid_levels.append(GridLevel(
                price=price,
                level_id=-i,
                order_type='buy',
                quantity=config.total_investment / config.total_levels / price
            ))
        
        return sorted(grid_levels, key=lambda x: x.price, reverse=True)
    
    async def setup_grid(self, config: GridConfiguration) -> bool:
        """グリッド設定実行"""
        
        try:
            # グリッドレベル作成
            grid_levels = self._create_grid_levels(config)
            
            # バッチ注文実行（MEXC API制限対応）
            success_orders = 0
            batch_size = 20  # MEXCバッチ制限
            
            for i in range(0, len(grid_levels), batch_size):
                batch = grid_levels[i:i + batch_size]
                
                for level in batch:
                    # 注文実行（模擬）
                    order_result = await self.data_source.place_order(
                        symbol=config.symbol,
                        side="BUY" if level.order_type == 'buy' else "SELL",
                        order_type="LIMIT",
                        quantity=level.quantity,
                        price=level.price
                    )
                    
                    if order_result.get("status") == "FILLED":
                        level.is_filled = False  # 指値注文なので待機状態
                        level.order_id = order_result.get("orderId")
                        success_orders += 1
                
                # API制限対策（0.5秒待機）
                await asyncio.sleep(0.5)
            
            # グリッド管理に追加
            self.active_grids[config.symbol] = grid_levels
            self.grid_configs[config.symbol] = config
            
            self.logger.info(f"✅ {config.symbol} グリッド設定完了: {success_orders}/{len(grid_levels)}注文")
            return success_orders > len(grid_levels) * 0.8  # 80%以上成功で有効
            
        except Exception as e:
            self.logger.error(f"❌ グリッド設定エラー {config.symbol}: {str(e)}")
            return False
    
    async def manage_grid(self, symbol: str) -> List[TradeResult]:
        """グリッド管理・取引実行"""
        
        if symbol not in self.active_grids:
            return []
        
        trades = []
        current_price = await self.data_source.get_current_price(symbol)
        grid_levels = self.active_grids[symbol]
        config = self.grid_configs[symbol]
        
        # 価格がグリッドレベルに到達した場合の処理
        for level in grid_levels:
            if not level.is_filled:
                # 指値注文の約定チェック（簡易模擬）
                if level.order_type == 'buy' and current_price <= level.price:
                    # 買い注文約定
                    trade = await self._execute_grid_trade(symbol, level, 'filled')
                    if trade:
                        trades.append(trade)
                        level.is_filled = True
                
                elif level.order_type == 'sell' and current_price >= level.price:
                    # 売り注文約定
                    trade = await self._execute_grid_trade(symbol, level, 'filled')
                    if trade:
                        trades.append(trade)
                        level.is_filled = True
        
        # ストップロス チェック
        if self._check_stop_loss(current_price, config):
            self.logger.warning(f"⚠️ {symbol} ストップロス発動")
            await self._close_grid(symbol)
        
        # リバランスチェック
        if self._should_rebalance(current_price, config):
            self.logger.info(f"🔄 {symbol} グリッドリバランス実行")
            await self._rebalance_grid(symbol)
        
        return trades
    
    async def _execute_grid_trade(self, symbol: str, level: GridLevel, action: str) -> Optional[TradeResult]:
        """グリッド取引実行"""
        
        try:
            current_time = self.data_source.get_current_time()
            current_price = await self.data_source.get_current_price(symbol)
            
            # 利益計算（グリッド間隔分の利益）
            if level.order_type == 'buy':
                # 買い注文約定後、次の売りレベルでの利益想定
                expected_sell_price = level.price * (1 + self.grid_configs[symbol].grid_spacing_pct / 100)
                profit_loss = (expected_sell_price - level.price) * level.quantity
            else:
                # 売り注文約定後、次の買いレベルでの利益想定
                expected_buy_price = level.price * (1 - self.grid_configs[symbol].grid_spacing_pct / 100)
                profit_loss = (level.price - expected_buy_price) * level.quantity
            
            profit_pct = (profit_loss / (level.price * level.quantity)) * 100
            
            trade = TradeResult(
                symbol=symbol,
                entry_time=current_time,
                exit_time=current_time,
                side="BUY" if level.order_type == 'buy' else "SELL",
                entry_price=level.price,
                exit_price=current_price,
                quantity=level.quantity,
                profit_loss=profit_loss,
                profit_pct=profit_pct,
                hold_hours=0.0,  # 即座約定
                exit_reason=f"グリッド約定_レベル{level.level_id}"
            )
            
            self.trade_history.append(trade)
            self.logger.info(f"💰 グリッド取引: {symbol} {level.order_type} ${level.price:.4f} 利益${profit_loss:.2f}")
            
            return trade
            
        except Exception as e:
            self.logger.error(f"グリッド取引実行エラー: {str(e)}")
            return None
    
    def _check_stop_loss(self, current_price: float, config: GridConfiguration) -> bool:
        """ストップロス判定"""
        
        upper_limit = config.center_price * (1 + config.stop_loss_pct / 100)
        lower_limit = config.center_price * (1 - config.stop_loss_pct / 100)
        
        return current_price > upper_limit or current_price < lower_limit
    
    def _should_rebalance(self, current_price: float, config: GridConfiguration) -> bool:
        """リバランス判定"""
        
        price_deviation = abs(current_price - config.center_price) / config.center_price * 100
        return price_deviation > self.grid_config["REBALANCE_THRESHOLD"]
    
    async def _rebalance_grid(self, symbol: str):
        """グリッドリバランス"""
        
        # 現在のグリッドを一旦クリア
        await self._close_grid(symbol)
        
        # 新しい中心価格でグリッド再設定
        current_price = await self.data_source.get_current_price(symbol)
        config = self.grid_configs[symbol]
        config.center_price = current_price
        
        await self.setup_grid(config)
    
    async def _close_grid(self, symbol: str):
        """グリッドクローズ"""
        
        if symbol in self.active_grids:
            self.logger.info(f"🔚 {symbol} グリッドクローズ")
            del self.active_grids[symbol]
            del self.grid_configs[symbol]

class GridTradingBacktestSystem(AnnualBacktestSystem):
    """グリッド取引バックテストシステム"""
    
    def __init__(self, config: Dict, historical_data: Dict[str, pd.DataFrame], 
                 symbols: List[str], start_date: datetime, end_date: datetime):
        super().__init__(config, historical_data, symbols, start_date, end_date)
        
        self.grid_strategy = GridTradingStrategy(
            config, WindowsDataSource(ExecutionMode.BACKTEST, historical_data), ExecutionMode.BACKTEST
        )
        
        # グリッド取引専用設定
        self.enhanced_config.update({
            "STRATEGY_NAME": "グリッド取引戦略",
            "EXPECTED_ANNUAL_RETURN": 20.0,  # 15-25%の中央値
            "MAX_SIMULTANEOUS_GRIDS": 3,     # 同時グリッド数制限
            "GRID_REBALANCE_INTERVAL": 24,   # 24時間ごとリバランス
        })
    
    async def _execute_annual_backtest(self):
        """グリッド取引年間バックテスト実行"""
        
        capital = self.enhanced_config["INITIAL_CAPITAL"]
        
        # 時間ステップ（6時間ごと）
        timestamps = list(self.historical_data[self.symbols[0]].index[::6])
        
        for i, timestamp in enumerate(timestamps):
            try:
                # データソース時刻設定
                self.grid_strategy.data_source.set_current_time(timestamp)
                current_portfolio_value = capital
                
                # 既存グリッド管理
                for symbol in list(self.grid_strategy.active_grids.keys()):
                    trades = await self.grid_strategy.manage_grid(symbol)
                    
                    for trade in trades:
                        capital += trade.profit_loss
                        self.trades.append(trade)
                        current_portfolio_value += trade.profit_loss
                
                # 新規グリッド機会検索
                active_grids = len(self.grid_strategy.active_grids)
                if active_grids < self.enhanced_config["MAX_SIMULTANEOUS_GRIDS"]:
                    
                    for symbol in self.symbols:
                        if symbol not in self.grid_strategy.active_grids:
                            grid_config = await self.grid_strategy.analyze_grid_opportunity(symbol)
                            
                            if grid_config and capital > grid_config.total_investment:
                                success = await self.grid_strategy.setup_grid(grid_config)
                                
                                if success:
                                    capital -= grid_config.total_investment
                                    self.logger.info(f"📊 {symbol} グリッド開始: 投資額${grid_config.total_investment}")
                                    break  # 1回につき1グリッドまで
                
                # ポートフォリオ価値計算
                grid_value = sum([
                    config.total_investment for config in self.grid_strategy.grid_configs.values()
                ])
                portfolio_value = capital + grid_value
                
                # 日次記録
                if i % 4 == 0:  # 24時間ごと
                    self.daily_portfolio.append({
                        'timestamp': timestamp,
                        'portfolio_value': portfolio_value,
                        'cash': capital,
                        'positions': len(self.grid_strategy.active_grids),
                        'return_pct': ((portfolio_value - self.enhanced_config["INITIAL_CAPITAL"]) / 
                                     self.enhanced_config["INITIAL_CAPITAL"]) * 100
                    })
                
                # 進捗表示
                if i % 168 == 0:  # 週次
                    progress = (i / len(timestamps)) * 100
                    weeks = i // 28
                    active_grids = len(self.grid_strategy.active_grids)
                    self.logger.info(f"  進捗: {progress:.1f}% ({weeks}週経過) アクティブグリッド:{active_grids}")
                
            except Exception as e:
                self.logger.warning(f"タイムスタンプ {timestamp} でエラー: {str(e)}")
                continue

# メイン実行関数
async def run_grid_trading_backtest():
    """グリッド取引戦略バックテスト実行"""
    
    logger = logging.getLogger(__name__)
    logger.info("🔷 グリッド取引戦略 1年間バックテスト開始")
    
    # グリッド取引用設定
    config = {
        "STRATEGY_TYPE": "GRID_TRADING",
        "GRID_SPACING_MIN": 0.5,
        "GRID_SPACING_MAX": 2.0,
        "UPPER_RANGE": 15.0,
        "LOWER_RANGE": 15.0,
        "MIN_GRID_LEVELS": 10,
        "MAX_GRID_LEVELS": 30,
        "STOP_LOSS_RANGE": 20.0,
        "MAX_INVESTMENT_PER_SYMBOL": 2000.0
    }
    
    # 1年間バックテストシステム作成
    logger.info("📊 グリッド取引バックテストシステム作成中...")
    annual_system = await WindowsUnifiedSystemFactory.create_annual_backtest_system(
        config, use_real_data=True
    )
    
    # グリッド取引システムに変換
    grid_system = GridTradingBacktestSystem(
        config, annual_system.historical_data, annual_system.symbols,
        annual_system.start_date, annual_system.end_date
    )
    
    # バックテスト実行
    logger.info("🔷 グリッド取引バックテスト実行中...")
    results = await grid_system.run_annual_comprehensive_backtest()
    
    # 結果表示
    print("\n" + "="*80)
    print("🔷 グリッド取引戦略 1年間バックテスト完了")
    print("📊 調査報告書準拠実装（期待年利15-25%）")
    print("="*80)
    
    perf = results['performance_metrics']
    print(f"\n📈 パフォーマンス結果:")
    print(f"   戦略タイプ: グリッド取引（レンジ相場特化）")
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
    
    if perf['total_return'] >= 15.0:
        print("✅ 調査報告書期待値（年15-25%）達成")
    else:
        print("❌ 調査報告書期待値未達成")
    
    return results

async def main():
    """メイン実行"""
    await run_grid_trading_backtest()

if __name__ == "__main__":
    asyncio.run(main())