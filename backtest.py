#!/usr/bin/env python3
"""
MEXC Trading Bot - Backtesting Module
Simulates trading strategy on historical data
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import json

from mexc_api import MEXCApi
from strategy import TradingStrategy
from config import Config
from utils import ColoredFormatter


class BacktestTrade:
    """Represents a single trade in backtesting"""
    def __init__(self, symbol: str, entry_time: datetime, entry_price: float, 
                 quantity: float, side: str = 'long'):
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.quantity = quantity
        self.side = side
        self.exit_time = None
        self.exit_price = None
        self.profit_loss = 0.0
        self.profit_loss_pct = 0.0
        self.fees = 0.0
        self.exit_reason = None
        self.max_profit = 0.0
        self.max_loss = 0.0
        
    def close(self, exit_time: datetime, exit_price: float, exit_reason: str, fees: float):
        """Close the trade"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.fees = fees
        
        if self.side == 'long':
            self.profit_loss = (exit_price - self.entry_price) * self.quantity - fees
            self.profit_loss_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            self.profit_loss = (self.entry_price - exit_price) * self.quantity - fees
            self.profit_loss_pct = ((self.entry_price - exit_price) / self.entry_price) * 100


class BacktestEngine:
    """Backtesting engine for MEXC trading strategy"""
    
    def __init__(self, config: Config, initial_balance: float = 1000.0):
        self.config = config
        self.api = MEXCApi(config)
        self.strategy = TradingStrategy(config)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.logger = self._setup_logger()
        
        # Trading state
        self.open_trades: Dict[str, BacktestTrade] = {}
        self.closed_trades: List[BacktestTrade] = []
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
        # Performance tracking
        self.equity_curve = []
        self.daily_returns = []
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        
    def _setup_logger(self) -> logging.Logger:
        """Setup colored logger for backtesting"""
        logger = logging.getLogger('backtest')
        logger.setLevel(logging.INFO)
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('backtest_results.log')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
        
    async def fetch_historical_data(self, symbol: str, start_date: datetime, 
                                  end_date: datetime) -> pd.DataFrame:
        """Fetch historical kline data for a symbol"""
        self.logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
        
        all_data = []
        current_end = end_date
        
        while current_end > start_date:
            try:
                # MEXC限制: 最大1000本のKライン
                klines = await self.api.get_klines(symbol, "5m", limit=1000)
                
                if not klines:
                    break
                    
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                # Filter by date range
                df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                
                if df.empty:
                    break
                    
                all_data.append(df)
                
                # Update current_end for next iteration
                current_end = df['timestamp'].min() - timedelta(minutes=5)
                
                # Rate limit
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                break
                
        if all_data:
            # Combine all data
            result = pd.concat(all_data, ignore_index=True)
            result = result.sort_values('timestamp').drop_duplicates('timestamp')
            
            # Calculate additional data needed for strategy
            result = self._calculate_indicators(result)
            
            return result
        else:
            return pd.DataFrame()
            
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataframe"""
        if len(df) < 50:  # Need enough data for indicators
            return df
            
        # Price changes
        df['price_change_30m'] = df['close'].pct_change(6) * 100  # 6 * 5min = 30min
        df['price_change_1h'] = df['close'].pct_change(12) * 100  # 12 * 5min = 1h
        
        # Volume
        df['volume_avg_24h'] = df['volume'].rolling(288).mean()  # 288 * 5min = 24h
        df['volume_ratio'] = df['volume'] / df['volume_avg_24h']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # EMA
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        return df
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    async def run_backtest(self, symbols: List[str], start_date: datetime, 
                          end_date: datetime) -> Dict:
        """Run backtest for given symbols and date range"""
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        self.logger.info(f"Symbols: {symbols}")
        
        # Reset state
        self.current_balance = self.initial_balance
        self.open_trades.clear()
        self.closed_trades.clear()
        self.equity_curve.clear()
        
        # Fetch historical data for all symbols
        for symbol in symbols:
            data = await self.fetch_historical_data(symbol, start_date, end_date)
            if not data.empty:
                self.historical_data[symbol] = data
                self.logger.info(f"Loaded {len(data)} data points for {symbol}")
            else:
                self.logger.warning(f"No data available for {symbol}")
                
        if not self.historical_data:
            self.logger.error("No historical data available for backtesting")
            return {}
            
        # Get all unique timestamps
        all_timestamps = set()
        for df in self.historical_data.values():
            all_timestamps.update(df['timestamp'].tolist())
        all_timestamps = sorted(list(all_timestamps))
        
        # Simulate trading for each timestamp
        for timestamp in all_timestamps:
            await self._process_timestamp(timestamp)
            
        # Close any remaining open trades
        for symbol, trade in list(self.open_trades.items()):
            await self._close_trade(trade, timestamp, "Backtest End")
            
        # Calculate final metrics
        results = self._calculate_performance_metrics()
        
        # Save detailed results
        self._save_results(results, start_date, end_date)
        
        return results
        
    async def _process_timestamp(self, timestamp: datetime):
        """Process trading logic for a specific timestamp"""
        # Track equity
        equity = self.current_balance
        for trade in self.open_trades.values():
            current_price = self._get_price_at_timestamp(trade.symbol, timestamp)
            if current_price:
                if trade.side == 'long':
                    equity += (current_price - trade.entry_price) * trade.quantity
                else:  # short
                    equity += (trade.entry_price - current_price) * trade.quantity
        self.equity_curve.append({'timestamp': timestamp, 'equity': equity})
        
        # Check exit conditions for open trades
        for symbol, trade in list(self.open_trades.items()):
            exit_signal = await self._check_exit_conditions(trade, timestamp)
            if exit_signal:
                await self._close_trade(trade, timestamp, exit_signal)
                
        # Check entry conditions for new trades
        if len(self.open_trades) < self.config.max_concurrent_trades:
            for symbol in self.historical_data.keys():
                if symbol not in self.open_trades:
                    entry_signal = await self._check_entry_conditions(symbol, timestamp)
                    if entry_signal:
                        await self._open_trade(symbol, timestamp, entry_signal)
                        
    def _get_price_at_timestamp(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get price for symbol at specific timestamp"""
        if symbol not in self.historical_data:
            return None
            
        df = self.historical_data[symbol]
        row = df[df['timestamp'] == timestamp]
        
        if row.empty:
            # Find closest timestamp
            idx = df['timestamp'].searchsorted(timestamp)
            if idx > 0:
                row = df.iloc[idx - 1]
                return float(row['close'])
        else:
            return float(row['close'].iloc[0])
            
        return None
        
    def _get_data_at_timestamp(self, symbol: str, timestamp: datetime) -> Optional[pd.Series]:
        """Get full data row for symbol at specific timestamp"""
        if symbol not in self.historical_data:
            return None
            
        df = self.historical_data[symbol]
        row = df[df['timestamp'] == timestamp]
        
        if row.empty:
            # Find closest timestamp
            idx = df['timestamp'].searchsorted(timestamp)
            if idx > 0:
                return df.iloc[idx - 1]
        else:
            return row.iloc[0]
            
        return None
        
    async def _check_entry_conditions(self, symbol: str, timestamp: datetime) -> Optional[str]:
        """Check if entry conditions are met"""
        data = self._get_data_at_timestamp(symbol, timestamp)
        if data is None or pd.isna(data['rsi']) or pd.isna(data['bb_upper']):
            return None
            
        # Get current hour for time-based adjustments
        current_hour = timestamp.hour
        
        # Long entry conditions
        if (data['close'] > data['bb_upper'] and
            data['rsi'] > self.config.get_time_based_threshold('rsi_entry_long', current_hour) and
            data['macd'] > data['macd_signal'] and
            data['ema_20'] > data['ema_50'] and
            data['volume_ratio'] > 2.0 and
            data['price_change_30m'] > 1.5):
            return 'long'
            
        # Short entry conditions (if enabled)
        if self.config.enable_short_trading:
            if (data['close'] < data['bb_lower'] and
                data['rsi'] < 35 and
                data['macd'] < data['macd_signal'] and
                data['ema_20'] < data['ema_50'] and
                data['volume_ratio'] > 2.0 and
                data['price_change_30m'] < -1.5):
                return 'short'
                
        # Special strategy: Buy dips after significant drop
        if (data['price_change_1h'] < -4.0 and
            data['rsi'] < 30 and
            data['close'] > data['bb_lower']):
            return 'long'
            
        return None
        
    async def _check_exit_conditions(self, trade: BacktestTrade, timestamp: datetime) -> Optional[str]:
        """Check if exit conditions are met"""
        data = self._get_data_at_timestamp(trade.symbol, timestamp)
        if data is None:
            return None
            
        current_price = data['close']
        entry_price = trade.entry_price
        
        # Calculate profit/loss percentage
        if trade.side == 'long':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # short
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
        # Update max profit/loss
        trade.max_profit = max(trade.max_profit, pnl_pct)
        trade.max_loss = min(trade.max_loss, pnl_pct)
        
        # Check time-based exit
        time_held = (timestamp - trade.entry_time).total_seconds() / 60  # minutes
        if time_held > self.config.time_based_exit_minutes and pnl_pct < 0.5:
            return "Time-based Exit"
            
        # Progressive stop loss
        if time_held < 15 and pnl_pct < -0.8:
            return "Progressive Stop Loss (15min)"
        elif time_held < 30 and pnl_pct < -1.2:
            return "Progressive Stop Loss (30min)"
        elif pnl_pct < -self.config.stop_loss_percentage:
            return "Stop Loss"
            
        # Take profit
        if pnl_pct >= self.config.take_profit_phase2_percentage:
            return "Take Profit Phase 2"
        elif pnl_pct >= self.config.take_profit_phase1_percentage:
            # Phase 1: Exit 50% (simplified for backtest - exit all)
            if data['rsi'] > 70 or data['close'] < data['bb_middle']:
                return "Take Profit Phase 1"
                
        # Trailing stop
        if trade.max_profit >= self.config.trailing_stop_activation_percentage:
            trailing_stop = trade.max_profit - self.config.trailing_stop_percentage
            if pnl_pct <= trailing_stop:
                return "Trailing Stop"
                
        # Technical exit signals
        if trade.side == 'long':
            if data['macd'] < data['macd_signal'] and pnl_pct > 0:
                return "MACD Bearish Cross"
            if data['close'] < data['bb_middle'] and pnl_pct > 1:
                return "BB Middle Break"
        else:  # short
            if data['macd'] > data['macd_signal'] and pnl_pct > 0:
                return "MACD Bullish Cross"
            if data['close'] > data['bb_middle'] and pnl_pct > 1:
                return "BB Middle Break"
                
        return None
        
    async def _open_trade(self, symbol: str, timestamp: datetime, side: str):
        """Open a new trade"""
        price = self._get_price_at_timestamp(symbol, timestamp)
        if not price:
            return
            
        # Calculate position size
        position_size_pct = self.config.position_size_percentage / 100
        position_value = self.current_balance * position_size_pct
        quantity = position_value / price
        
        # Check if we have enough balance
        if position_value > self.current_balance * 0.95:  # Keep 5% reserve
            return
            
        # Create trade
        trade = BacktestTrade(symbol, timestamp, price, quantity, side)
        self.open_trades[symbol] = trade
        
        # Deduct from balance (including fees)
        entry_fee = position_value * (self.config.taker_fee_percentage / 100)
        self.current_balance -= (position_value + entry_fee)
        
        self.logger.info(f"OPEN {side.upper()} - {symbol} @ {price:.4f}, "
                        f"Quantity: {quantity:.4f}, Value: ${position_value:.2f}")
        
    async def _close_trade(self, trade: BacktestTrade, timestamp: datetime, reason: str):
        """Close an existing trade"""
        exit_price = self._get_price_at_timestamp(trade.symbol, timestamp)
        if not exit_price:
            return
            
        # Calculate fees
        exit_value = exit_price * trade.quantity
        exit_fee = exit_value * (self.config.taker_fee_percentage / 100)
        total_fees = (trade.entry_price * trade.quantity * self.config.taker_fee_percentage / 100) + exit_fee
        
        # Close trade
        trade.close(timestamp, exit_price, reason, total_fees)
        
        # Update balance
        if trade.side == 'long':
            self.current_balance += exit_value - exit_fee
        else:  # short
            # For short: Return borrowed amount and keep profit/loss
            self.current_balance += (trade.entry_price * trade.quantity) + trade.profit_loss
            
        # Move to closed trades
        del self.open_trades[trade.symbol]
        self.closed_trades.append(trade)
        
        self.logger.info(f"CLOSE {trade.side.upper()} - {trade.symbol} @ {exit_price:.4f}, "
                        f"P&L: ${trade.profit_loss:.2f} ({trade.profit_loss_pct:.2f}%), "
                        f"Reason: {reason}")
        
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_pnl_pct': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'final_balance': self.current_balance,
                'roi': 0.0
            }
            
        # Basic metrics
        winning_trades = [t for t in self.closed_trades if t.profit_loss > 0]
        losing_trades = [t for t in self.closed_trades if t.profit_loss <= 0]
        
        total_profit = sum(t.profit_loss for t in winning_trades)
        total_loss = abs(sum(t.profit_loss for t in losing_trades))
        
        # Calculate metrics
        metrics = {
            'total_trades': len(self.closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.closed_trades) * 100,
            'total_pnl': sum(t.profit_loss for t in self.closed_trades),
            'total_pnl_pct': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            'average_win': total_profit / len(winning_trades) if winning_trades else 0,
            'average_loss': total_loss / len(losing_trades) if losing_trades else 0,
            'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'final_balance': self.current_balance,
            'roi': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        }
        
        # Add trade distribution
        metrics['trades_by_exit_reason'] = self._get_exit_reason_distribution()
        metrics['average_hold_time'] = self._calculate_average_hold_time()
        
        return metrics
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        if not self.equity_curve:
            return 0.0
            
        equity_values = [e['equity'] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0.0
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (simplified)"""
        if len(self.equity_curve) < 2:
            return 0.0
            
        # Calculate daily returns
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Annualized Sharpe (assuming 252 trading days)
        if equity_df['returns'].std() > 0:
            sharpe = (equity_df['returns'].mean() * 252) / (equity_df['returns'].std() * np.sqrt(252))
            return sharpe
        return 0.0
        
    def _get_exit_reason_distribution(self) -> Dict[str, int]:
        """Get distribution of exit reasons"""
        distribution = defaultdict(int)
        for trade in self.closed_trades:
            distribution[trade.exit_reason] += 1
        return dict(distribution)
        
    def _calculate_average_hold_time(self) -> float:
        """Calculate average trade holding time in minutes"""
        if not self.closed_trades:
            return 0.0
            
        total_time = sum(
            (trade.exit_time - trade.entry_time).total_seconds() / 60
            for trade in self.closed_trades
        )
        return total_time / len(self.closed_trades)
        
    def _save_results(self, results: Dict, start_date: datetime, end_date: datetime):
        """Save detailed backtest results"""
        # Create results directory
        import os
        os.makedirs('backtest_results', exist_ok=True)
        
        # Save summary
        summary_file = f"backtest_results/summary_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Save trade details
        trades_data = []
        for trade in self.closed_trades:
            trades_data.append({
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_time': trade.entry_time.isoformat(),
                'entry_price': trade.entry_price,
                'exit_time': trade.exit_time.isoformat(),
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'profit_loss': trade.profit_loss,
                'profit_loss_pct': trade.profit_loss_pct,
                'exit_reason': trade.exit_reason,
                'max_profit': trade.max_profit,
                'max_loss': trade.max_loss
            })
            
        trades_file = f"backtest_results/trades_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        with open(trades_file, 'w') as f:
            json.dump(trades_data, f, indent=2)
            
        # Save equity curve
        equity_file = f"backtest_results/equity_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        with open(equity_file, 'w') as f:
            json.dump(self.equity_curve, f, indent=2, default=str)
            
        self.logger.info(f"Results saved to backtest_results/")


async def main():
    """Run backtests for multiple time periods"""
    # Initialize configuration
    config = Config()
    config.load_from_env()
    
    # Get monitored symbols
    symbols = (config.tier1_symbols.split(',') + 
               config.tier2_symbols.split(',') + 
               config.tier3_symbols.split(','))
    symbols = [s.strip() + 'USDT' for s in symbols if s.strip()]
    
    # Initialize backtest engine
    engine = BacktestEngine(config)
    
    # Define time periods
    end_date = datetime.now()
    time_periods = [
        ("1_month", end_date - timedelta(days=30)),
        ("3_months", end_date - timedelta(days=90)),
        ("6_months", end_date - timedelta(days=180)),
        ("1_year", end_date - timedelta(days=365))
    ]
    
    # Run backtests for each period
    all_results = {}
    
    for period_name, start_date in time_periods:
        logging.info(f"\n{'='*60}")
        logging.info(f"Running backtest for {period_name}")
        logging.info(f"{'='*60}")
        
        results = await engine.run_backtest(symbols[:10], start_date, end_date)  # Limit to 10 symbols for testing
        all_results[period_name] = results
        
        # Print summary
        logging.info(f"\n{period_name.upper()} RESULTS:")
        logging.info(f"Total Trades: {results.get('total_trades', 0)}")
        logging.info(f"Win Rate: {results.get('win_rate', 0):.2f}%")
        logging.info(f"Total P&L: ${results.get('total_pnl', 0):.2f}")
        logging.info(f"ROI: {results.get('roi', 0):.2f}%")
        logging.info(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        logging.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        logging.info(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        
    # Save consolidated results
    with open('backtest_results/all_periods_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
        
    logging.info(f"\n{'='*60}")
    logging.info("All backtests completed!")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())