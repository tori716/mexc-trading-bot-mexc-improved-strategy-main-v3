#!/usr/bin/env python3
"""
バックテストエンジン
MEXCトレーディングボットの戦略を過去データでテストするためのモジュール
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import defaultdict

from mexc_api import MEXCApi
from config import Config
from utils import DiscordNotifier
from strategy import TradingStrategy
from constants import MEXC_FEE_RATE_TAKER


class BacktestAPI:
    """
    バックテスト用のMEXC APIモック
    履歴データを使って実際のAPIコールをシミュレーション
    """
    
    def __init__(self, historical_data: Dict[str, pd.DataFrame], logger: logging.Logger):
        self.historical_data = historical_data
        self.logger = logger
        self.current_index = 0
        self.max_index = 0
        self.test_mode = True
        
        # 初期設定
        if historical_data:
            first_symbol = list(historical_data.keys())[0]
            self.max_index = len(historical_data[first_symbol]) - 1
        
        # キャッシュ
        self.market_data = {}
        self.ohlcv_data = {}
        
    def set_current_time_index(self, index: int):
        """現在の時間インデックスを設定"""
        self.current_index = min(index, self.max_index)
        
    async def get_ticker_price(self, symbol: str) -> dict:
        """現在の価格を返す"""
        if symbol not in self.historical_data:
            return None
            
        df = self.historical_data[symbol]
        if self.current_index >= len(df):
            return None
            
        current_data = df.iloc[self.current_index]
        return {
            "symbol": symbol,
            "price": str(current_data['close'])
        }
        
    async def get_current_price(self, symbol: str) -> float:
        """現在価格を取得"""
        ticker = await self.get_ticker_price(symbol)
        if ticker and ticker.get("price"):
            return float(ticker["price"])
        return 0.0
        
    async def get_24hr_volume(self, symbol: str) -> float:
        """24時間出来高を取得"""
        if symbol not in self.historical_data:
            return 0.0
            
        df = self.historical_data[symbol]
        # 過去24時間（288本の5分足）の出来高を合計
        start_idx = max(0, self.current_index - 288)
        end_idx = self.current_index + 1
        
        if start_idx < len(df):
            return df.iloc[start_idx:end_idx]['volume'].sum()
        return 0.0
        
    async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> list:
        """K線データを取得"""
        if symbol not in self.historical_data:
            return []
            
        df = self.historical_data[symbol]
        
        # インターバルに応じてデータを集約
        # バックテストでは5分足データを基準にする
        interval_minutes = {
            "Min5": 1, "5m": 1,  # 5分足はそのまま
            "Min15": 3, "15m": 3,  # 15分足は3本分
            "Min60": 12, "60m": 12, "1h": 12,  # 1時間足は12本分
            "Min240": 48, "4h": 48,  # 4時間足は48本分
            "1d": 288  # 日足は288本分
        }
        
        multiplier = interval_minutes.get(interval, 1)
        
        # 現在のインデックスから過去のデータを取得
        end_idx = self.current_index + 1
        start_idx = max(0, end_idx - (limit * multiplier))
        
        if start_idx >= len(df):
            return []
            
        subset = df.iloc[start_idx:end_idx]
        
        # インターバルに応じて集約
        klines = []
        for i in range(0, len(subset), multiplier):
            group = subset.iloc[i:i+multiplier]
            if len(group) > 0:
                kline = [
                    int(group.iloc[0]['timestamp'] * 1000),  # タイムスタンプ
                    str(group.iloc[0]['open']),  # 始値
                    str(group['high'].max()),  # 高値
                    str(group['low'].min()),  # 安値
                    str(group.iloc[-1]['close']),  # 終値
                    str(group['volume'].sum()),  # 出来高
                    0,  # Quote asset volume
                    0,  # Number of trades
                    0,  # Taker buy base asset volume
                    0,  # Taker buy quote asset volume
                    0   # Ignore
                ]
                klines.append(kline)
                
        return klines[-limit:]  # 最新のlimit本を返す
        
    async def get_ohlcv(self, symbol: str, interval: str, limit: int = 100) -> list:
        """OHLCVデータを取得"""
        klines = await self.get_klines(symbol, interval, limit)
        if klines:
            parsed_klines = []
            for kline in klines:
                parsed_klines.append({
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                    "timestamp": int(kline[0])
                })
            return parsed_klines
        return []
        
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None):
        """注文をシミュレーション"""
        order_id = f"BACKTEST_{int(self.current_index)}_{symbol}_{side}"
        return {
            "orderId": order_id,
            "status": "FILLED",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "price": price or await self.get_current_price(symbol)
        }


class BacktestDiscordNotifier:
    """バックテスト用のDiscord通知モック"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.messages = []
        
    def send_discord_message(self, message: str, is_entry: bool = False, is_exit: bool = False):
        """メッセージを記録"""
        self.messages.append({
            "message": message,
            "is_entry": is_entry,
            "is_exit": is_exit,
            "timestamp": datetime.now()
        })
        self.logger.debug(f"Discord通知: {message}")


class BacktestEngine:
    """
    バックテストエンジン本体
    """
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.results = {
            "trades": [],
            "metrics": {},
            "equity_curve": []
        }
        
    async def fetch_historical_data(self, symbol: str, period_days: int) -> pd.DataFrame:
        """
        指定期間の履歴データを取得
        """
        self.logger.info(f"{symbol}の過去{period_days}日間のデータを取得中...")
        
        # 実際のMEXC APIを使用してデータ取得
        real_api = MEXCApi(self.config, DiscordNotifier(self.config, self.logger), self.logger)
        
        # 5分足データを取得（1日288本）
        total_candles = period_days * 288
        batch_size = 1000  # MEXC APIの制限
        
        all_data = []
        end_time = None
        
        # バッチで過去データを取得
        while len(all_data) < total_candles:
            try:
                params = {
                    "symbol": symbol,
                    "interval": "5m",
                    "limit": min(batch_size, total_candles - len(all_data))
                }
                
                if end_time:
                    params["endTime"] = end_time
                    
                klines = await real_api._send_request("GET", "/api/v3/klines", params=params)
                
                if not klines:
                    break
                    
                # データを追加
                for kline in klines:
                    all_data.append({
                        'timestamp': int(kline[0]) / 1000,
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })
                
                # 次のバッチの終了時刻を設定
                if klines:
                    end_time = int(klines[0][0]) - 1
                    
                # API制限を考慮して待機
                await asyncio.sleep(1)
                
                self.logger.info(f"{symbol}: {len(all_data)}/{total_candles}本のデータを取得済み")
                
            except Exception as e:
                self.logger.error(f"{symbol}のデータ取得エラー: {e}")
                break
                
        # DataFrameに変換してソート
        if all_data:
            df = pd.DataFrame(all_data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.drop_duplicates(subset=['timestamp'])
            self.logger.info(f"{symbol}: {len(df)}本のデータ取得完了")
            return df
        else:
            self.logger.error(f"{symbol}のデータが取得できませんでした")
            return pd.DataFrame()
            
    async def run_backtest(self, symbols: List[str], period_days: int) -> dict:
        """
        バックテストを実行
        """
        self.logger.info(f"バックテスト開始: {period_days}日間, {len(symbols)}銘柄")
        
        # 履歴データを取得
        historical_data = {}
        for symbol in symbols:
            df = await self.fetch_historical_data(symbol, period_days + 1)  # 余分に1日取得
            if not df.empty:
                historical_data[symbol] = df
                
        if not historical_data:
            self.logger.error("履歴データが取得できませんでした")
            return self.results
            
        # バックテスト用のAPIとNotifierを作成
        backtest_api = BacktestAPI(historical_data, self.logger)
        backtest_notifier = BacktestDiscordNotifier(self.logger)
        
        # 戦略インスタンスを作成
        strategy = TradingStrategy(backtest_api, self.config, backtest_notifier, self.logger)
        
        # 初期資金を設定
        initial_capital = self.config.INITIAL_CAPITAL_USD
        current_capital = initial_capital
        strategy.current_capital = current_capital
        
        # 時系列でバックテストを実行
        first_symbol = list(historical_data.keys())[0]
        num_candles = len(historical_data[first_symbol])
        
        # 最初の100本はウォームアップ期間（指標計算のため）
        start_index = 100
        
        self.logger.info(f"バックテスト期間: {num_candles - start_index}本の5分足データ")
        
        # 各時点でのエクイティを記録
        equity_curve = []
        
        for i in range(start_index, num_candles):
            # 現在の時間を設定
            backtest_api.set_current_time_index(i)
            current_time = historical_data[first_symbol].iloc[i]['timestamp']
            
            # 市場データを更新
            for symbol in symbols:
                if symbol in historical_data:
                    # 価格データを更新
                    ticker = await backtest_api.get_ticker_price(symbol)
                    if ticker:
                        backtest_api.market_data[symbol] = {
                            "price": float(ticker["price"]),
                            "volume_24h": await backtest_api.get_24hr_volume(symbol),
                            "timestamp": current_time
                        }
                        
                    # OHLCVデータを更新
                    backtest_api.ohlcv_data[symbol] = {}
                    for interval in ["Min5", "Min15", "Min60"]:
                        ohlcv = await backtest_api.get_ohlcv(symbol, interval, 50)
                        if ohlcv:
                            backtest_api.ohlcv_data[symbol][interval] = ohlcv
                            
            # 戦略を実行
            # まずエグジットをチェック
            await strategy._check_and_execute_exit()
            
            # 次にエントリーをチェック（5分ごと）
            if i % 1 == 0:  # 5分足データなので毎回チェック
                await strategy._check_and_execute_entry(symbols)
                
            # 現在の総資産を計算
            total_equity = strategy.current_capital
            for pos in strategy.current_positions:
                current_price = await backtest_api.get_current_price(pos['symbol'])
                if current_price > 0:
                    if pos['direction'] == 'long':
                        pos_value = pos['quantity'] * current_price
                    else:
                        pos_value = pos['quantity'] * (2 * pos['entry_price'] - current_price)
                    total_equity += pos_value - (pos['quantity'] * pos['entry_price'])
                    
            equity_curve.append({
                'timestamp': current_time,
                'equity': total_equity,
                'capital': strategy.current_capital,
                'positions': len(strategy.current_positions)
            })
            
            # 進捗を表示（10%ごと）
            progress = (i - start_index) / (num_candles - start_index) * 100
            if progress % 10 < 0.1:
                self.logger.info(f"バックテスト進捗: {progress:.0f}%")
                
        # 結果を集計
        self.results = {
            "trades": strategy.trade_history,
            "equity_curve": equity_curve,
            "metrics": self._calculate_metrics(
                strategy.trade_history,
                initial_capital,
                strategy.current_capital,
                equity_curve
            ),
            "period_days": period_days,
            "symbols": symbols,
            "start_date": datetime.fromtimestamp(historical_data[first_symbol].iloc[start_index]['timestamp']),
            "end_date": datetime.fromtimestamp(historical_data[first_symbol].iloc[-1]['timestamp'])
        }
        
        return self.results
        
    def _calculate_metrics(self, trades: List[dict], initial_capital: float, 
                          final_capital: float, equity_curve: List[dict]) -> dict:
        """
        パフォーマンスメトリクスを計算
        """
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_return": 0,
                "roi": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "profit_factor": 0,
                "average_win": 0,
                "average_loss": 0,
                "largest_win": 0,
                "largest_loss": 0
            }
            
        # 基本統計
        winning_trades = [t for t in trades if t['pnl_amount'] > 0]
        losing_trades = [t for t in trades if t['pnl_amount'] < 0]
        
        total_trades = len(trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        win_rate = (num_winning / total_trades * 100) if total_trades > 0 else 0
        
        # 収益性
        total_profit = sum(t['pnl_amount'] for t in winning_trades)
        total_loss = abs(sum(t['pnl_amount'] for t in losing_trades))
        net_profit = total_profit - total_loss
        roi = ((final_capital - initial_capital) / initial_capital * 100)
        
        # プロフィットファクター
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
        
        # 平均勝敗
        average_win = (total_profit / num_winning) if num_winning > 0 else 0
        average_loss = (total_loss / num_losing) if num_losing > 0 else 0
        
        # 最大勝敗
        largest_win = max((t['pnl_amount'] for t in trades), default=0)
        largest_loss = min((t['pnl_amount'] for t in trades), default=0)
        
        # ドローダウン計算
        equity_values = [e['equity'] for e in equity_curve]
        max_drawdown = self._calculate_max_drawdown(equity_values)
        
        # シャープレシオ計算
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        
        # 決済理由の集計
        exit_reasons = defaultdict(int)
        for trade in trades:
            exit_reasons[trade.get('reason', 'unknown')] += 1
            
        return {
            "total_trades": total_trades,
            "winning_trades": num_winning,
            "losing_trades": num_losing,
            "win_rate": round(win_rate, 2),
            "total_return": round(net_profit, 2),
            "roi": round(roi, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown, 2),
            "profit_factor": round(profit_factor, 2),
            "average_win": round(average_win, 2),
            "average_loss": round(average_loss, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "exit_reasons": dict(exit_reasons)
        }
        
    def _calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """最大ドローダウンを計算"""
        if not equity_values:
            return 0
            
        peak = equity_values[0]
        max_dd = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak * 100
                if dd > max_dd:
                    max_dd = dd
                    
        return max_dd
        
    def _calculate_sharpe_ratio(self, equity_curve: List[dict], risk_free_rate: float = 0.02) -> float:
        """シャープレシオを計算（年率換算）"""
        if len(equity_curve) < 2:
            return 0
            
        # 日次リターンを計算
        returns = []
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i-1]['equity']
            curr_equity = equity_curve[i]['equity']
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)
                
        if not returns:
            return 0
            
        # 年率換算（5分足データなので1日288本）
        periods_per_year = 288 * 365
        periods_in_data = len(equity_curve)
        annualization_factor = periods_per_year / periods_in_data
        
        # 平均リターンと標準偏差
        avg_return = np.mean(returns) * annualization_factor
        std_return = np.std(returns) * np.sqrt(annualization_factor)
        
        # シャープレシオ
        if std_return > 0:
            sharpe = (avg_return - risk_free_rate) / std_return
            return sharpe
        else:
            return 0
            
    def save_results(self, filename: str):
        """結果をJSONファイルに保存"""
        # datetime オブジェクトを文字列に変換
        results_to_save = self.results.copy()
        
        # 日付を文字列に変換
        if 'start_date' in results_to_save:
            results_to_save['start_date'] = results_to_save['start_date'].isoformat()
        if 'end_date' in results_to_save:
            results_to_save['end_date'] = results_to_save['end_date'].isoformat()
            
        # トレード履歴の日付も変換
        for trade in results_to_save.get('trades', []):
            if 'entry_time' in trade and isinstance(trade['entry_time'], float):
                trade['entry_time'] = datetime.fromtimestamp(trade['entry_time']).isoformat()
            if 'exit_time' in trade and isinstance(trade['exit_time'], float):
                trade['exit_time'] = datetime.fromtimestamp(trade['exit_time']).isoformat()
                
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"バックテスト結果を保存しました: {filename}")