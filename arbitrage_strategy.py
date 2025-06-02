#!/usr/bin/env python3
"""
アービトラージ戦略 - 調査報告書準拠実装
期待利益率: 年間50-150%（超高勝率85-95%）
価格差を利用した低リスク高頻度取引
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
        logging.FileHandler('arbitrage_strategy.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

@dataclass
class ArbitrageOpportunity:
    """アービトラージ機会定義"""
    symbol_pair: Tuple[str, str]  # (symbol1, symbol2) or (symbol, symbol_timeframe)
    price_1: float
    price_2: float
    price_diff_pct: float
    arbitrage_type: str  # 'temporal', 'cross_symbol', 'mean_reversion'
    confidence: float  # 0.0-1.0
    expected_profit_pct: float
    timestamp: datetime
    duration_estimate: float  # 期待保有時間（分）

@dataclass
class ArbitragePosition:
    """アービトラージポジション"""
    opportunity: ArbitrageOpportunity
    entry_time: datetime
    entry_price_1: float
    entry_price_2: Optional[float]
    quantity: float
    expected_exit_price: float
    max_hold_minutes: float
    is_active: bool = True

class ArbitrageStrategy:
    """アービトラージ戦略（調査報告書準拠）"""
    
    def __init__(self, config: Dict, data_source: WindowsDataSource, execution_mode: ExecutionMode):
        self.config = config
        self.data_source = data_source
        self.execution_mode = execution_mode
        self.logger = logging.getLogger(__name__)
        
        # アービトラージ管理
        self.active_positions: Dict[str, ArbitragePosition] = {}
        self.trade_history = []
        self.price_history = {}  # 価格履歴保存
        
        # 調査報告書準拠の設定
        self.arbitrage_config = {
            "MIN_PROFIT_PCT": 0.3,           # 最小利益率（0.1-2.5%の下限より保守的）
            "MAX_PROFIT_PCT": 3.0,           # 最大利益率（2.5%より攻撃的）
            "MIN_CONFIDENCE": 0.7,           # 最小信頼度
            "MAX_HOLD_MINUTES": 60,          # 最大保有時間（1時間）
            "TEMPORAL_WINDOW_MINUTES": 30,   # 時間的価格差検出ウィンドウ
            "CROSS_SYMBOL_CORRELATION": 0.8, # シンボル間相関閾値
            "MEAN_REVERSION_THRESHOLD": 2.0, # 平均回帰閾値（標準偏差倍数）
            "MAX_SIMULTANEOUS_ARBITRAGES": 8, # 同時アービトラージ数
            "PRICE_IMPACT_BUFFER": 0.05,     # 価格インパクトバッファ（0.05%）
            "API_RATE_LIMIT_BUFFER": 0.2,    # API制限バッファ（0.2秒）
            "VOLUME_REQUIREMENT": 50000,     # 最小出来高要件（$50,000）
        }
    
    async def scan_arbitrage_opportunities(self, symbols: List[str]) -> List[ArbitrageOpportunity]:
        """アービトラージ機会スキャン"""
        
        opportunities = []
        current_time = self.data_source.get_current_time()
        
        try:
            # 現在価格とOHLCVデータ取得
            current_prices = {}
            ohlcv_data = {}
            
            for symbol in symbols:
                current_prices[symbol] = await self.data_source.get_current_price(symbol)
                ohlcv_data[symbol] = await self.data_source.get_ohlcv(symbol, "5m", 50)
                
                # 価格履歴更新
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                self.price_history[symbol].append({
                    'timestamp': current_time,
                    'price': current_prices[symbol]
                })
                
                # 履歴制限（最新100件）
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol] = self.price_history[symbol][-100:]
            
            # 1. 時間的アービトラージ機会検出
            temporal_opportunities = self._detect_temporal_arbitrage(current_prices, ohlcv_data)
            opportunities.extend(temporal_opportunities)
            
            # 2. シンボル間アービトラージ機会検出
            cross_symbol_opportunities = self._detect_cross_symbol_arbitrage(current_prices, ohlcv_data)
            opportunities.extend(cross_symbol_opportunities)
            
            # 3. 平均回帰アービトラージ機会検出
            mean_reversion_opportunities = self._detect_mean_reversion_arbitrage(current_prices, ohlcv_data)
            opportunities.extend(mean_reversion_opportunities)
            
            # 機会をフィルタリング・ソート
            filtered_opportunities = self._filter_and_rank_opportunities(opportunities)
            
            if filtered_opportunities:
                self.logger.info(f"🔍 アービトラージ機会検出: {len(filtered_opportunities)}件")
                for i, opp in enumerate(filtered_opportunities[:3]):  # TOP3を表示
                    self.logger.info(f"   {i+1}. {opp.arbitrage_type}: {opp.symbol_pair} 利益{opp.expected_profit_pct:.2f}%")
            
            return filtered_opportunities
            
        except Exception as e:
            self.logger.error(f"アービトラージ機会スキャンエラー: {str(e)}")
            return []
    
    def _detect_temporal_arbitrage(self, current_prices: Dict[str, float], 
                                 ohlcv_data: Dict[str, List]) -> List[ArbitrageOpportunity]:
        """時間的アービトラージ検出（短期価格差）"""
        
        opportunities = []
        current_time = self.data_source.get_current_time()
        
        for symbol, current_price in current_prices.items():
            if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
                continue
            
            try:
                # 過去価格との比較
                recent_prices = [p['price'] for p in self.price_history[symbol][-10:]]
                price_changes = [(current_price - p) / p * 100 for p in recent_prices]
                
                # 急激な価格変動検出（アービトラージ機会）
                max_change = max(abs(change) for change in price_changes)
                
                if max_change > self.arbitrage_config["MIN_PROFIT_PCT"]:
                    # 反発期待の時間的アービトラージ
                    if current_price < min(recent_prices) * 0.99:  # 1%以上下落
                        expected_profit = min(max_change, self.arbitrage_config["MAX_PROFIT_PCT"])
                        confidence = min(0.95, 0.7 + (max_change / 10))  # 変動が大きいほど高信頼度
                        
                        opportunity = ArbitrageOpportunity(
                            symbol_pair=(symbol, f"{symbol}_temporal"),
                            price_1=current_price,
                            price_2=sum(recent_prices) / len(recent_prices),  # 平均価格
                            price_diff_pct=max_change,
                            arbitrage_type="temporal",
                            confidence=confidence,
                            expected_profit_pct=expected_profit,
                            timestamp=current_time,
                            duration_estimate=self.arbitrage_config["TEMPORAL_WINDOW_MINUTES"]
                        )
                        opportunities.append(opportunity)
                        
            except Exception as e:
                self.logger.warning(f"時間的アービトラージ検出エラー {symbol}: {e}")
                continue
        
        return opportunities
    
    def _detect_cross_symbol_arbitrage(self, current_prices: Dict[str, float], 
                                     ohlcv_data: Dict[str, List]) -> List[ArbitrageOpportunity]:
        """シンボル間アービトラージ検出（相関ペア）"""
        
        opportunities = []
        current_time = self.data_source.get_current_time()
        symbols = list(current_prices.keys())
        
        # 相関の高いペアを特定（例：BTC系ペア、ETH系ペア）
        correlated_pairs = [
            # Layer 1 tokens (相関が高い)
            ("AVAXUSDT", "NEARUSDT"),
            ("ATOMUSDT", "DOTUSDT"),
            ("ADAUSDT", "ALGOUSDT"),
            # DeFi tokens (相関が高い)
            ("UNIUSDT", "AAVEUSDT"),
            ("LINKUSDT", "GRTUSDT"),
            # Small cap pairs
            ("MANAUSDT", "SANDUSDT"),
            ("GALAUSDT", "VETUSDT"),
        ]
        
        for symbol1, symbol2 in correlated_pairs:
            if symbol1 not in current_prices or symbol2 not in current_prices:
                continue
            
            try:
                price1 = current_prices[symbol1]
                price2 = current_prices[symbol2]
                
                # 過去価格から正規化比率計算
                if (symbol1 in self.price_history and symbol2 in self.price_history and
                    len(self.price_history[symbol1]) >= 20 and len(self.price_history[symbol2]) >= 20):
                    
                    # 過去20期間の価格比率
                    ratios = []
                    for i in range(-20, 0):
                        try:
                            p1 = self.price_history[symbol1][i]['price']
                            p2 = self.price_history[symbol2][i]['price']
                            ratios.append(p1 / p2)
                        except (IndexError, ZeroDivisionError):
                            continue
                    
                    if len(ratios) >= 10:
                        avg_ratio = sum(ratios) / len(ratios)
                        current_ratio = price1 / price2
                        ratio_deviation = abs(current_ratio - avg_ratio) / avg_ratio * 100
                        
                        if ratio_deviation > self.arbitrage_config["MIN_PROFIT_PCT"]:
                            # ペアトレーディング機会
                            expected_profit = min(ratio_deviation * 0.5, self.arbitrage_config["MAX_PROFIT_PCT"])
                            confidence = min(0.9, 0.6 + (ratio_deviation / 20))
                            
                            opportunity = ArbitrageOpportunity(
                                symbol_pair=(symbol1, symbol2),
                                price_1=price1,
                                price_2=price2,
                                price_diff_pct=ratio_deviation,
                                arbitrage_type="cross_symbol",
                                confidence=confidence,
                                expected_profit_pct=expected_profit,
                                timestamp=current_time,
                                duration_estimate=30  # 30分程度で収束期待
                            )
                            opportunities.append(opportunity)
                            
            except Exception as e:
                self.logger.warning(f"シンボル間アービトラージ検出エラー {symbol1}-{symbol2}: {e}")
                continue
        
        return opportunities
    
    def _detect_mean_reversion_arbitrage(self, current_prices: Dict[str, float], 
                                       ohlcv_data: Dict[str, List]) -> List[ArbitrageOpportunity]:
        """平均回帰アービトラージ検出（統計的裁定）"""
        
        opportunities = []
        current_time = self.data_source.get_current_time()
        
        for symbol, current_price in current_prices.items():
            if symbol not in ohlcv_data or not ohlcv_data[symbol]:
                continue
            
            try:
                df = pd.DataFrame(ohlcv_data[symbol])
                if len(df) < 20:
                    continue
                
                # ボリンジャーバンドによる平均回帰検出
                sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
                std_20 = df['close'].rolling(window=20).std().iloc[-1]
                
                if pd.isna(sma_20) or pd.isna(std_20) or std_20 == 0:
                    continue
                
                # Z-スコア計算
                z_score = (current_price - sma_20) / std_20
                
                if abs(z_score) > self.arbitrage_config["MEAN_REVERSION_THRESHOLD"]:
                    # 平均回帰機会
                    expected_return_to_mean = abs(current_price - sma_20) / current_price * 100
                    expected_profit = min(expected_return_to_mean * 0.6, self.arbitrage_config["MAX_PROFIT_PCT"])
                    confidence = min(0.95, 0.7 + (abs(z_score) / 10))
                    
                    # 出来高確認
                    recent_volume = df['volume'].tail(5).mean()
                    if recent_volume < self.arbitrage_config["VOLUME_REQUIREMENT"]:
                        continue
                    
                    opportunity = ArbitrageOpportunity(
                        symbol_pair=(symbol, f"{symbol}_mean_reversion"),
                        price_1=current_price,
                        price_2=sma_20,
                        price_diff_pct=expected_return_to_mean,
                        arbitrage_type="mean_reversion",
                        confidence=confidence,
                        expected_profit_pct=expected_profit,
                        timestamp=current_time,
                        duration_estimate=45  # 45分程度で平均回帰期待
                    )
                    opportunities.append(opportunity)
                    
            except Exception as e:
                self.logger.warning(f"平均回帰アービトラージ検出エラー {symbol}: {e}")
                continue
        
        return opportunities
    
    def _filter_and_rank_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """アービトラージ機会のフィルタリング・ランキング"""
        
        # フィルタリング
        filtered = []
        for opp in opportunities:
            # 最小利益率チェック
            if opp.expected_profit_pct < self.arbitrage_config["MIN_PROFIT_PCT"]:
                continue
            
            # 最小信頼度チェック
            if opp.confidence < self.arbitrage_config["MIN_CONFIDENCE"]:
                continue
            
            # 最大利益率制限
            if opp.expected_profit_pct > self.arbitrage_config["MAX_PROFIT_PCT"]:
                opp.expected_profit_pct = self.arbitrage_config["MAX_PROFIT_PCT"]
            
            filtered.append(opp)
        
        # ランキング（期待利益 × 信頼度でソート）
        filtered.sort(key=lambda x: x.expected_profit_pct * x.confidence, reverse=True)
        
        return filtered
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """アービトラージ実行"""
        
        try:
            symbol = opportunity.symbol_pair[0]
            current_price = await self.data_source.get_current_price(symbol)
            current_time = self.data_source.get_current_time()
            
            # ポジションサイズ計算（リスクを最小化）
            position_size = 100  # 固定$100（低リスク）
            quantity = position_size / current_price
            
            # エントリー実行
            side = "BUY" if opportunity.price_diff_pct > 0 else "SELL"
            order_result = await self.data_source.place_order(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=quantity
            )
            
            if order_result.get("status") == "FILLED":
                # ポジション記録
                position = ArbitragePosition(
                    opportunity=opportunity,
                    entry_time=current_time,
                    entry_price_1=current_price,
                    entry_price_2=opportunity.price_2,
                    quantity=quantity,
                    expected_exit_price=current_price * (1 + opportunity.expected_profit_pct / 100),
                    max_hold_minutes=opportunity.duration_estimate
                )
                
                self.active_positions[f"{symbol}_{current_time.timestamp()}"] = position
                
                self.logger.info(f"⚡ アービトラージ実行: {opportunity.arbitrage_type}")
                self.logger.info(f"   {symbol} {side} ${current_price:.4f} 期待利益{opportunity.expected_profit_pct:.2f}%")
                
                return True
            else:
                self.logger.error(f"❌ アービトラージ実行失敗: {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ アービトラージ実行エラー: {str(e)}")
            return False
    
    async def manage_arbitrage_positions(self) -> List[TradeResult]:
        """アービトラージポジション管理"""
        
        trades = []
        current_time = self.data_source.get_current_time()
        positions_to_close = []
        
        for pos_id, position in self.active_positions.items():
            if not position.is_active:
                continue
            
            try:
                symbol = position.opportunity.symbol_pair[0]
                current_price = await self.data_source.get_current_price(symbol)
                
                # 利益確認
                if position.opportunity.arbitrage_type == "temporal":
                    # 時間的アービトラージ：目標価格到達で決済
                    profit_pct = (current_price - position.entry_price_1) / position.entry_price_1 * 100
                    
                    if profit_pct >= position.opportunity.expected_profit_pct * 0.8:  # 80%達成で決済
                        trade = await self._close_arbitrage_position(position, current_price, current_time, "利確")
                        if trade:
                            trades.append(trade)
                            positions_to_close.append(pos_id)
                
                elif position.opportunity.arbitrage_type == "mean_reversion":
                    # 平均回帰：平均価格への接近で決済
                    distance_to_mean = abs(current_price - position.entry_price_2) / position.entry_price_2 * 100
                    original_distance = abs(position.entry_price_1 - position.entry_price_2) / position.entry_price_2 * 100
                    
                    if distance_to_mean < original_distance * 0.3:  # 平均に70%接近で決済
                        trade = await self._close_arbitrage_position(position, current_price, current_time, "平均回帰")
                        if trade:
                            trades.append(trade)
                            positions_to_close.append(pos_id)
                
                # 時間切れチェック
                hold_minutes = (current_time - position.entry_time).total_seconds() / 60
                if hold_minutes >= position.max_hold_minutes:
                    trade = await self._close_arbitrage_position(position, current_price, current_time, "時間切れ")
                    if trade:
                        trades.append(trade)
                        positions_to_close.append(pos_id)
                
            except Exception as e:
                self.logger.warning(f"ポジション管理エラー {pos_id}: {e}")
                continue
        
        # ポジションクローズ
        for pos_id in positions_to_close:
            del self.active_positions[pos_id]
        
        return trades
    
    async def _close_arbitrage_position(self, position: ArbitragePosition, current_price: float,
                                      current_time: datetime, exit_reason: str) -> Optional[TradeResult]:
        """アービトラージポジション決済"""
        
        try:
            symbol = position.opportunity.symbol_pair[0]
            
            # 反対売買実行
            side = "SELL" if position.opportunity.price_diff_pct > 0 else "BUY"
            order_result = await self.data_source.place_order(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=position.quantity
            )
            
            if order_result.get("status") == "FILLED":
                # 損益計算
                if position.opportunity.price_diff_pct > 0:  # ロングポジション
                    profit_loss = (current_price - position.entry_price_1) * position.quantity
                else:  # ショートポジション
                    profit_loss = (position.entry_price_1 - current_price) * position.quantity
                
                profit_pct = (profit_loss / (position.entry_price_1 * position.quantity)) * 100
                hold_minutes = (current_time - position.entry_time).total_seconds() / 60
                
                trade = TradeResult(
                    symbol=symbol,
                    entry_time=position.entry_time,
                    exit_time=current_time,
                    side="BUY" if position.opportunity.price_diff_pct > 0 else "SELL",
                    entry_price=position.entry_price_1,
                    exit_price=current_price,
                    quantity=position.quantity,
                    profit_loss=profit_loss,
                    profit_pct=profit_pct,
                    hold_hours=hold_minutes / 60,
                    exit_reason=f"アービトラージ_{exit_reason}"
                )
                
                self.trade_history.append(trade)
                
                self.logger.info(f"⚡ アービトラージ決済: {symbol} {exit_reason}")
                self.logger.info(f"   エントリー: ${position.entry_price_1:.4f} → 決済: ${current_price:.4f}")
                self.logger.info(f"   利益: ${profit_loss:.2f} ({profit_pct:+.2f}%) 保有{hold_minutes:.1f}分")
                
                return trade
            else:
                self.logger.error(f"❌ アービトラージ決済失敗: {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"アービトラージ決済エラー: {str(e)}")
            return None

class ArbitrageBacktestSystem(AnnualBacktestSystem):
    """アービトラージバックテストシステム"""
    
    def __init__(self, config: Dict, historical_data: Dict[str, pd.DataFrame], 
                 symbols: List[str], start_date: datetime, end_date: datetime):
        super().__init__(config, historical_data, symbols, start_date, end_date)
        
        self.arbitrage_strategy = ArbitrageStrategy(
            config, WindowsDataSource(ExecutionMode.BACKTEST, historical_data), ExecutionMode.BACKTEST
        )
        
        # アービトラージ専用設定
        self.enhanced_config.update({
            "STRATEGY_NAME": "アービトラージ戦略",
            "EXPECTED_ANNUAL_RETURN": 100.0,  # 50-150%の中央値
            "MAX_SIMULTANEOUS_ARBITRAGES": 8,  # 同時アービトラージ数
            "SCAN_INTERVAL_MINUTES": 5,        # 5分ごとスキャン
        })
    
    async def _execute_annual_backtest(self):
        """アービトラージ年間バックテスト実行"""
        
        capital = self.enhanced_config["INITIAL_CAPITAL"]
        
        # 時間ステップ（5分ごと、高頻度）
        timestamps = list(self.historical_data[self.symbols[0]].index[::1])  # 全データポイント使用
        
        for i, timestamp in enumerate(timestamps):
            try:
                # データソース時刻設定
                self.arbitrage_strategy.data_source.set_current_time(timestamp)
                current_portfolio_value = capital
                
                # 既存アービトラージポジション管理
                trades = await self.arbitrage_strategy.manage_arbitrage_positions()
                for trade in trades:
                    capital += trade.profit_loss + 100  # $100固定ポジション回収
                    self.trades.append(trade)
                    current_portfolio_value += trade.profit_loss
                
                # 新規アービトラージ機会スキャン（5分ごと）
                if i % 1 == 0:  # 毎回スキャン（高頻度）
                    active_arbitrages = len(self.arbitrage_strategy.active_positions)
                    if active_arbitrages < self.enhanced_config["MAX_SIMULTANEOUS_ARBITRAGES"]:
                        
                        opportunities = await self.arbitrage_strategy.scan_arbitrage_opportunities(self.symbols)
                        
                        for opportunity in opportunities[:3]:  # TOP3実行
                            if active_arbitrages >= self.enhanced_config["MAX_SIMULTANEOUS_ARBITRAGES"]:
                                break
                            
                            if capital > 100:  # $100以上で実行
                                success = await self.arbitrage_strategy.execute_arbitrage(opportunity)
                                if success:
                                    capital -= 100  # $100投資
                                    active_arbitrages += 1
                                    self.logger.info(f"⚡ {opportunity.symbol_pair[0]} アービトラージ開始: {opportunity.arbitrage_type}")
                
                # ポートフォリオ価値計算
                arbitrage_investment = len(self.arbitrage_strategy.active_positions) * 100
                portfolio_value = capital + arbitrage_investment
                
                # 日次記録（6時間ごと）
                if i % 72 == 0:  # 6時間ごと（5分足×72）
                    self.daily_portfolio.append({
                        'timestamp': timestamp,
                        'portfolio_value': portfolio_value,
                        'cash': capital,
                        'positions': len(self.arbitrage_strategy.active_positions),
                        'return_pct': ((portfolio_value - self.enhanced_config["INITIAL_CAPITAL"]) / 
                                     self.enhanced_config["INITIAL_CAPITAL"]) * 100
                    })
                
                # 進捗表示
                if i % 2016 == 0:  # 週次（5分足×2016 = 1週間）
                    progress = (i / len(timestamps)) * 100
                    days = i // 288  # 1日=288×5分
                    active_arbitrages = len(self.arbitrage_strategy.active_positions)
                    total_trades = len(self.trades)
                    self.logger.info(f"  進捗: {progress:.1f}% ({days}日経過) アクティブ:{active_arbitrages} 取引数:{total_trades}")
                
            except Exception as e:
                self.logger.warning(f"タイムスタンプ {timestamp} でエラー: {str(e)}")
                continue

# メイン実行関数
async def run_arbitrage_backtest():
    """アービトラージ戦略バックテスト実行"""
    
    logger = logging.getLogger(__name__)
    logger.info("⚡ アービトラージ戦略 1年間バックテスト開始")
    
    # アービトラージ用設定
    config = {
        "STRATEGY_TYPE": "ARBITRAGE",
        "MIN_PROFIT_PCT": 0.3,
        "MAX_PROFIT_PCT": 3.0,
        "MIN_CONFIDENCE": 0.7,
        "MAX_HOLD_MINUTES": 60,
        "MAX_SIMULTANEOUS_ARBITRAGES": 8
    }
    
    # 1年間バックテストシステム作成
    logger.info("📊 アービトラージバックテストシステム作成中...")
    annual_system = await WindowsUnifiedSystemFactory.create_annual_backtest_system(
        config, use_real_data=True
    )
    
    # アービトラージシステムに変換
    arbitrage_system = ArbitrageBacktestSystem(
        config, annual_system.historical_data, annual_system.symbols,
        annual_system.start_date, annual_system.end_date
    )
    
    # バックテスト実行
    logger.info("⚡ アービトラージバックテスト実行中...")
    results = await arbitrage_system.run_annual_comprehensive_backtest()
    
    # 結果表示
    print("\n" + "="*80)
    print("⚡ アービトラージ戦略 1年間バックテスト完了")
    print("📊 調査報告書準拠実装（期待年利50-150%）")
    print("="*80)
    
    perf = results['performance_metrics']
    print(f"\n📈 パフォーマンス結果:")
    print(f"   戦略タイプ: アービトラージ（価格差利用）")
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
    
    if perf['total_return'] >= 50.0:
        print("✅ 調査報告書期待値（年50-150%）達成")
    else:
        print("❌ 調査報告書期待値未達成")
    
    # 戦略比較
    print(f"\n📊 戦略比較:")
    print(f"   グリッド取引: +0.2% (勝率100%, 取引53)")
    print(f"   DCA Bot: +0.0% (勝率100%, 取引1)")
    print(f"   アービトラージ: {perf['total_return']:+.1f}% (勝率{perf['win_rate']:.1f}%, 取引{perf['total_trades']})")
    
    # 取引頻度分析
    if perf['total_trades'] > 0:
        daily_trades = perf['total_trades'] / 365
        print(f"\n📈 取引頻度分析:")
        print(f"   1日平均取引数: {daily_trades:.1f}")
        print(f"   調査報告書期待値: 10-100取引/日")
        
        if daily_trades >= 10:
            print("✅ 高頻度取引目標達成")
        else:
            print("❌ 高頻度取引目標未達成")
    
    return results

async def main():
    """メイン実行"""
    await run_arbitrage_backtest()

if __name__ == "__main__":
    asyncio.run(main())