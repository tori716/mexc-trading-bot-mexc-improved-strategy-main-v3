# backtest.py
# 過去データを使用してトレーディング戦略をバックテストするモジュール

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json
from collections import defaultdict
import ta
from strategy import TradingStrategy
from config import Config
import constants

class BacktestAPI:
    """
    MEXCAPIの代替となるバックテスト用のモックAPI
    過去データから価格情報を提供します
    """
    def __init__(self, historical_data: Dict[str, pd.DataFrame]):
        self.historical_data = historical_data
        self.current_timestamp = None
        self.logger = logging.getLogger(__name__)
        
    async def get_current_price(self, symbol: str) -> float:
        """現在のタイムスタンプにおける価格を返す"""
        if symbol not in self.historical_data:
            return 0.0
            
        df = self.historical_data[symbol]
        if self.current_timestamp in df.index:
            return float(df.loc[self.current_timestamp, 'close'])
        
        # 最も近い過去のタイムスタンプを探す
        past_times = df.index[df.index <= self.current_timestamp]
        if len(past_times) > 0:
            return float(df.loc[past_times[-1], 'close'])
        
        return 0.0
    
    async def get_ohlcv(self, symbol: str, interval: str, limit: int) -> List[Dict]:
        """指定された期間のOHLCVデータを返す"""
        if symbol not in self.historical_data:
            return []
            
        df = self.historical_data[symbol]
        
        # 現在時刻以前のデータのみを使用
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
    
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         quantity: float, price: float = None) -> Dict:
        """バックテストでは常に注文が成功したと仮定"""
        return {
            "status": "FILLED",
            "orderId": f"BACKTEST_{symbol}_{side}_{self.current_timestamp}",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "price": price or await self.get_current_price(symbol)
        }
    
    async def get_all_symbols(self) -> List[str]:
        """利用可能な全てのシンボルを返す"""
        return list(self.historical_data.keys())
    
    async def verify_api_permissions(self) -> bool:
        """バックテストでは常にTrue"""
        return True
    
    async def start_rest_api_only_mode(self, symbols: List[str]):
        """バックテストでは何もしない"""
        pass

class BacktestNotifier:
    """バックテスト用のモック通知クラス"""
    def __init__(self):
        self.messages = []
    
    def send_discord_message(self, message: str):
        """メッセージを記録するだけ"""
        self.messages.append({
            'timestamp': datetime.now(),
            'message': message
        })

class BacktestEngine:
    """
    バックテストエンジンのメインクラス
    戦略の過去パフォーマンスを評価します
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.historical_data = {}
        self.trades = []
        self.portfolio_history = []
        
    def fetch_historical_data(self, symbol: str, start_date: datetime, 
                            end_date: datetime) -> pd.DataFrame:
        """
        MEXCから過去データを取得します
        実際の実装では、MEXC APIから履歴データを取得する必要があります
        """
        # ここは実際のMEXC APIを使用して実装する必要があります
        # 今回はプレースホルダーとして空のデータフレームを返します
        self.logger.warning(f"実際のMEXC APIから{symbol}の履歴データを取得する必要があります")
        
        # ダミーデータの生成（実際のAPIが利用できない場合のテスト用）
        date_range = pd.date_range(start=start_date, end=end_date, freq='5min')
        df = pd.DataFrame(index=date_range)
        
        # ランダムな価格データを生成（実際のAPIを使用すべき）
        base_price = 100.0
        df['open'] = base_price
        df['high'] = base_price * 1.01
        df['low'] = base_price * 0.99
        df['close'] = base_price
        df['volume'] = 1000.0
        
        return df
    
    async def run_backtest(self, symbols: List[str], start_date: datetime, 
                          end_date: datetime) -> Dict[str, Any]:
        """
        バックテストを実行します
        
        Args:
            symbols: テストする銘柄リスト
            start_date: バックテスト開始日
            end_date: バックテスト終了日
            
        Returns:
            バックテスト結果の辞書
        """
        self.logger.info(f"バックテスト開始: {start_date} から {end_date}")
        
        # 過去データの取得
        for symbol in symbols:
            self.historical_data[symbol] = self.fetch_historical_data(
                symbol, start_date, end_date
            )
        
        # モックAPIとNotifierの作成
        mock_api = BacktestAPI(self.historical_data)
        mock_notifier = BacktestNotifier()
        
        # 戦略インスタンスの作成
        strategy = TradingStrategy(mock_api, self.config, mock_notifier)
        
        # 初期資金の設定
        initial_capital = self.config["INITIAL_CAPITAL_USD"]
        current_capital = initial_capital
        
        # タイムスタンプごとにシミュレーション
        all_timestamps = set()
        for df in self.historical_data.values():
            all_timestamps.update(df.index.tolist())
        
        all_timestamps = sorted(list(all_timestamps))
        
        for timestamp in all_timestamps:
            mock_api.current_timestamp = timestamp
            
            # 戦略の実行（簡略化版）
            # 実際にはstrategy.pyの全てのロジックを正確に再現する必要があります
            
            # ポートフォリオ履歴の記録
            self.portfolio_history.append({
                'timestamp': timestamp,
                'capital': current_capital,
                'positions': len(strategy.current_positions)
            })
        
        # パフォーマンス指標の計算
        results = self._calculate_performance_metrics(
            initial_capital, current_capital, self.trades
        )
        
        return results
    
    def _calculate_performance_metrics(self, initial_capital: float, 
                                     final_capital: float, 
                                     trades: List[Dict]) -> Dict[str, Any]:
        """
        パフォーマンス指標を計算します
        """
        if not trades:
            return {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'trades': []
            }
        
        # 基本的な指標の計算
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]
        
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
        
        avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        
        # プロフィットファクター
        total_wins = sum([t['profit'] for t in winning_trades]) if winning_trades else 0
        total_losses = abs(sum([t['profit'] for t in losing_trades])) if losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # シャープレシオ（簡易版）
        returns = [t['profit'] / initial_capital for t in trades]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
        
        # 最大ドローダウン
        capital_curve = [initial_capital]
        for trade in trades:
            capital_curve.append(capital_curve[-1] + trade['profit'])
        
        peak = capital_curve[0]
        max_dd = 0
        for value in capital_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': round(total_return, 2),
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'average_win': round(avg_win, 2),
            'average_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_dd * 100, 2),
            'trades': trades[:10]  # 最初の10件のみ
        }

# 実際のMEXC APIを使用してデータを取得する関数（実装が必要）
async def fetch_mexc_historical_data(api_key: str, secret_key: str, 
                                   symbol: str, interval: str,
                                   start_time: int, end_time: int) -> pd.DataFrame:
    """
    MEXC APIから過去のOHLCVデータを取得します
    
    注意: この関数は実際のMEXC APIエンドポイントを使用して実装する必要があります
    """
    # TODO: 実際のMEXC API実装
    pass