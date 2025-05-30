# reports.py
# 日次損益レポート生成モジュール

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import asyncio
import schedule
import time

class DailyReportGenerator:
    """
    日次損益レポートを生成し、ファイル出力とDiscord通知を行うクラス。
    """
    
    def __init__(self, strategy, notifier, config):
        """
        DailyReportGeneratorクラスのコンストラクタ。
        
        Args:
            strategy: TradingStrategyのインスタンス
            notifier: DiscordNotifierのインスタンス
            config: Botの設定
        """
        self.strategy = strategy
        self.notifier = notifier
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # レポート保存ディレクトリの作成
        self.report_dir = "reports"
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
            self.logger.info(f"レポートディレクトリを作成しました: {self.report_dir}")

    def generate_daily_report(self, target_date: datetime = None) -> str:
        """
        指定された日付の日次損益レポートを生成します。
        
        Args:
            target_date: レポート対象日（指定しない場合は前日）
            
        Returns:
            str: 生成されたレポートの内容
        """
        if target_date is None:
            target_date = datetime.now() - timedelta(days=1)
        
        date_str = target_date.strftime("%Y-%m-%d")
        self.logger.info(f"日次レポートを生成中: {date_str}")
        
        # 対象日の取引履歴を抽出
        daily_trades = self._extract_daily_trades(target_date)
        
        # レポートデータを計算
        report_data = self._calculate_report_data(daily_trades, target_date)
        
        # レポート文字列を生成
        report_content = self._format_report(report_data, date_str)
        
        # ファイルに保存
        self._save_report_to_file(report_content, date_str)
        
        # Discord通知（設定で有効な場合）
        if self.config.get("DAILY_REPORT_DISCORD_ENABLED", True):
            asyncio.create_task(self._send_discord_summary(report_data, date_str))
        
        self.logger.info(f"日次レポート生成完了: {date_str}")
        return report_content

    def _extract_daily_trades(self, target_date: datetime) -> List[Dict]:
        """
        指定された日付の取引履歴を抽出します。
        
        Args:
            target_date: 対象日
            
        Returns:
            List[Dict]: 対象日の取引履歴
        """
        daily_trades = []
        target_date_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        target_date_end = target_date_start + timedelta(days=1)
        
        for trade in self.strategy.trade_history:
            # exit_timeが対象日内にある取引を抽出
            if hasattr(trade, 'exit_time') and trade['exit_time']:
                if target_date_start <= trade['exit_time'] < target_date_end:
                    daily_trades.append(trade)
        
        return daily_trades

    def _calculate_report_data(self, daily_trades: List[Dict], target_date: datetime) -> Dict:
        """
        レポートに必要なデータを計算します。
        
        Args:
            daily_trades: 対象日の取引履歴
            target_date: 対象日
            
        Returns:
            Dict: レポートデータ
        """
        if not daily_trades:
            return {
                "start_capital": self.strategy.initial_capital,
                "end_capital": self.strategy.current_capital,
                "net_profit_loss": 0.0,
                "net_profit_loss_percentage": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "average_profit": 0.0,
                "average_loss": 0.0,
                "max_profit": 0.0,
                "max_loss": 0.0,
                "max_profit_symbol": "",
                "max_loss_symbol": "",
                "symbol_performance": {},
                "total_fees": 0.0,
                "net_profit_after_fees": 0.0
            }
        
        # 基本統計の計算
        total_trades = len(daily_trades)
        total_profit_loss = sum(trade.get('profit_loss_usd', 0) for trade in daily_trades if trade.get('profit_loss_usd') != "N/A")
        
        winning_trades = [trade for trade in daily_trades if trade.get('profit_loss_usd', 0) > 0]
        losing_trades = [trade for trade in daily_trades if trade.get('profit_loss_usd', 0) <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
        
        # 平均利益・損失の計算
        average_profit = sum(trade.get('profit_loss_usd', 0) for trade in winning_trades) / win_count if win_count > 0 else 0.0
        average_loss = sum(trade.get('profit_loss_usd', 0) for trade in losing_trades) / loss_count if loss_count > 0 else 0.0
        
        # 最大利益・損失の計算
        max_profit_trade = max(winning_trades, key=lambda x: x.get('profit_loss_usd', 0), default=None)
        max_loss_trade = min(losing_trades, key=lambda x: x.get('profit_loss_usd', 0), default=None)
        
        max_profit = max_profit_trade.get('profit_loss_usd', 0) if max_profit_trade else 0.0
        max_loss = max_loss_trade.get('profit_loss_usd', 0) if max_loss_trade else 0.0
        max_profit_symbol = max_profit_trade.get('symbol', '') if max_profit_trade else ''
        max_loss_symbol = max_loss_trade.get('symbol', '') if max_loss_trade else ''
        
        # 銘柄別パフォーマンスの計算
        symbol_performance = self._calculate_symbol_performance(daily_trades)
        
        # 手数料の計算
        total_fees = self._calculate_total_fees(daily_trades)
        
        # 開始・終了資金の推定（テストモードの場合）
        if self.config.get("TEST_MODE", True):
            end_capital = self.strategy.current_capital
            start_capital = end_capital - total_profit_loss
        else:
            # 本番モードでは実際のAPI呼び出しが必要
            start_capital = self.strategy.initial_capital
            end_capital = self.strategy.current_capital
        
        net_profit_loss = total_profit_loss
        net_profit_loss_percentage = (net_profit_loss / start_capital * 100) if start_capital > 0 else 0.0
        net_profit_after_fees = net_profit_loss - total_fees
        
        return {
            "start_capital": start_capital,
            "end_capital": end_capital,
            "net_profit_loss": net_profit_loss,
            "net_profit_loss_percentage": net_profit_loss_percentage,
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": win_rate,
            "average_profit": average_profit,
            "average_loss": average_loss,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "max_profit_symbol": max_profit_symbol,
            "max_loss_symbol": max_loss_symbol,
            "symbol_performance": symbol_performance,
            "total_fees": total_fees,
            "net_profit_after_fees": net_profit_after_fees
        }

    def _calculate_symbol_performance(self, daily_trades: List[Dict]) -> Dict:
        """
        銘柄別のパフォーマンスを計算します。
        
        Args:
            daily_trades: 日次取引履歴
            
        Returns:
            Dict: 銘柄別パフォーマンス
        """
        symbol_stats = defaultdict(lambda: {"profit_loss": 0.0, "trade_count": 0})
        
        for trade in daily_trades:
            symbol = trade.get('symbol', '')
            profit_loss = trade.get('profit_loss_usd', 0)
            if profit_loss != "N/A":
                symbol_stats[symbol]["profit_loss"] += profit_loss
                symbol_stats[symbol]["trade_count"] += 1
        
        return dict(symbol_stats)

    def _calculate_total_fees(self, daily_trades: List[Dict]) -> float:
        """
        取引手数料の総額を計算します。
        
        Args:
            daily_trades: 日次取引履歴
            
        Returns:
            float: 手数料総額
        """
        total_fees = 0.0
        fee_rate = 0.001  # 0.1% (MEXC標準手数料)
        
        for trade in daily_trades:
            quantity = trade.get('quantity', 0)
            exit_price = trade.get('exit_price', 0)
            # 往復手数料（エントリー時とイグジット時）
            total_fees += quantity * exit_price * fee_rate * 2
        
        return total_fees

    def _format_report(self, data: Dict, date_str: str) -> str:
        """
        レポートデータを文字列形式にフォーマットします。
        
        Args:
            data: レポートデータ
            date_str: 日付文字列
            
        Returns:
            str: フォーマットされたレポート
        """
        profit_loss_sign = "+" if data["net_profit_loss"] >= 0 else ""
        percentage_sign = "+" if data["net_profit_loss_percentage"] >= 0 else ""
        
        report = f"""=== 日次損益レポート ({date_str}) ===
開始資金: {data['start_capital']:,.2f} USDT
終了資金: {data['end_capital']:,.2f} USDT
純損益: {profit_loss_sign}{data['net_profit_loss']:,.2f} USDT ({percentage_sign}{data['net_profit_loss_percentage']:.2f}%)

取引統計:
- 総取引回数: {data['total_trades']}回
- 勝率: {data['win_rate']:.1f}% ({data['winning_trades']}勝{data['losing_trades']}敗)
- 平均利益: +{data['average_profit']:.2f} USDT
- 平均損失: {data['average_loss']:.2f} USDT
- 最大利益: +{data['max_profit']:.2f} USDT ({data['max_profit_symbol']})
- 最大損失: {data['max_loss']:.2f} USDT ({data['max_loss_symbol']})

銘柄別損益:"""

        # 銘柄別パフォーマンスを利益順にソート
        sorted_symbols = sorted(
            data['symbol_performance'].items(),
            key=lambda x: x[1]['profit_loss'],
            reverse=True
        )
        
        for symbol, performance in sorted_symbols:
            profit_loss = performance['profit_loss']
            trade_count = performance['trade_count']
            sign = "+" if profit_loss >= 0 else ""
            report += f"\n- {symbol}: {sign}{profit_loss:.2f} USDT ({trade_count}回取引)"

        report += f"""

手数料合計: -{data['total_fees']:.2f} USDT
実質純損益: {profit_loss_sign}{data['net_profit_after_fees']:.2f} USDT ({percentage_sign}{(data['net_profit_after_fees']/data['start_capital']*100):.2f}%)
==================================="""

        return report

    def _save_report_to_file(self, report_content: str, date_str: str):
        """
        レポートをファイルに保存します。
        
        Args:
            report_content: レポート内容
            date_str: 日付文字列
        """
        filename = f"daily_{date_str}.txt"
        filepath = os.path.join(self.report_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"日次レポートをファイルに保存しました: {filepath}")
        except Exception as e:
            self.logger.error(f"レポートファイル保存エラー: {e}")

    async def _send_discord_summary(self, data: Dict, date_str: str):
        """
        Discord用の要約版レポートを送信します。
        
        Args:
            data: レポートデータ
            date_str: 日付文字列
        """
        try:
            profit_loss_sign = "+" if data["net_profit_loss"] >= 0 else ""
            percentage_sign = "+" if data["net_profit_loss_percentage"] >= 0 else ""
            
            # Discord用要約版
            summary = f"""📊 **日次レポート {date_str}**

💰 **損益概要**
• 純損益: {profit_loss_sign}{data['net_profit_loss']:.2f} USDT ({percentage_sign}{data['net_profit_loss_percentage']:.2f}%)
• 総資金: {data['start_capital']:.2f} → {data['end_capital']:.2f} USDT

📈 **取引統計**
• 取引回数: {data['total_trades']}回
• 勝率: {data['win_rate']:.1f}% ({data['winning_trades']}勝{data['losing_trades']}敗)
• 最大利益: +{data['max_profit']:.2f} USDT ({data['max_profit_symbol']})
• 最大損失: {data['max_loss']:.2f} USDT ({data['max_loss_symbol']})

💸 **実質損益**: {profit_loss_sign}{data['net_profit_after_fees']:.2f} USDT (手数料差引後)"""

            # 色の設定（利益/損失に応じて）
            color = 0x00FF00 if data["net_profit_loss"] >= 0 else 0xFF0000  # 緑or赤
            
            self.notifier.send_discord_message(
                message=summary,
                embed_title=f"日次損益レポート - {date_str}",
                color=color
            )
            
        except Exception as e:
            self.logger.error(f"Discord要約レポート送信エラー: {e}")

    def schedule_daily_reports(self):
        """
        日次レポートの定期実行をスケジュールします。
        """
        if not self.config.get("DAILY_REPORT_ENABLED", True):
            self.logger.info("日次レポート機能は無効です")
            return
        
        report_time = self.config.get("DAILY_REPORT_TIME_JST", "00:00")
        
        try:
            # スケジュール設定
            schedule.every().day.at(report_time).do(self._scheduled_report_job)
            self.logger.info(f"日次レポートを {report_time} JST にスケジュールしました")
            
        except Exception as e:
            self.logger.error(f"日次レポートスケジュール設定エラー: {e}")

    def _scheduled_report_job(self):
        """
        スケジュールされたレポート生成ジョブ。
        """
        try:
            self.logger.info("定期日次レポート生成を開始します")
            self.generate_daily_report()
        except Exception as e:
            self.logger.error(f"定期日次レポート生成エラー: {e}")
            # エラー時はDiscordに通知
            self.notifier.send_discord_message(
                f"❌ 日次レポート生成でエラーが発生しました: {str(e)}",
                embed_title="レポート生成エラー",
                color=0xFF0000
            )

    async def run_scheduler(self):
        """
        スケジューラーを実行します（非同期）。
        """
        while True:
            try:
                schedule.run_pending()
                await asyncio.sleep(60)  # 1分間隔でスケジュールをチェック
            except Exception as e:
                self.logger.error(f"スケジューラー実行エラー: {e}")
                await asyncio.sleep(300)  # エラー時は5分待機

    def generate_performance_summary(self, days: int = 7) -> Dict:
        """
        指定期間のパフォーマンス要約を生成します。
        
        Args:
            days: 対象期間（日数）
            
        Returns:
            Dict: パフォーマンス要約データ
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        period_trades = []
        for trade in self.strategy.trade_history:
            if hasattr(trade, 'exit_time') and trade['exit_time']:
                if start_date <= trade['exit_time'] <= end_date:
                    period_trades.append(trade)
        
        if not period_trades:
            return {
                "period": f"{days}日間",
                "total_trades": 0,
                "total_profit_loss": 0.0,
                "win_rate": 0.0,
                "best_day": None,
                "worst_day": None
            }
        
        # 日別損益の計算
        daily_profits = defaultdict(float)
        for trade in period_trades:
            day_key = trade['exit_time'].strftime("%Y-%m-%d")
            profit_loss = trade.get('profit_loss_usd', 0)
            if profit_loss != "N/A":
                daily_profits[day_key] += profit_loss
        
        total_profit_loss = sum(daily_profits.values())
        winning_days = len([p for p in daily_profits.values() if p > 0])
        total_days = len(daily_profits)
        win_rate = (winning_days / total_days * 100) if total_days > 0 else 0.0
        
        best_day = max(daily_profits.items(), key=lambda x: x[1], default=(None, 0))
        worst_day = min(daily_profits.items(), key=lambda x: x[1], default=(None, 0))
        
        return {
            "period": f"{days}日間",
            "total_trades": len(period_trades),
            "total_profit_loss": total_profit_loss,
            "win_rate": win_rate,
            "daily_profits": dict(daily_profits),
            "best_day": {"date": best_day[0], "profit": best_day[1]} if best_day[0] else None,
            "worst_day": {"date": worst_day[0], "loss": worst_day[1]} if worst_day[0] else None
        }

    def export_trade_history_csv(self, filename: Optional[str] = None) -> str:
        """
        取引履歴をCSVファイルにエクスポートします。
        
        Args:
            filename: 出力ファイル名（指定しない場合は自動生成）
            
        Returns:
            str: 出力ファイルパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_history_{timestamp}.csv"
        
        filepath = os.path.join(self.report_dir, filename)
        
        try:
            import csv
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'symbol', 'side', 'entry_price', 'exit_price', 'quantity',
                    'entry_time', 'exit_time', 'reason', 'profit_loss_usd'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for trade in self.strategy.trade_history:
                    writer.writerow({
                        'symbol': trade.get('symbol', ''),
                        'side': trade.get('side', ''),
                        'entry_price': trade.get('entry_price', 0),
                        'exit_price': trade.get('exit_price', 0),
                        'quantity': trade.get('quantity', 0),
                        'entry_time': trade.get('entry_time', ''),
                        'exit_time': trade.get('exit_time', ''),
                        'reason': trade.get('reason', ''),
                        'profit_loss_usd': trade.get('profit_loss_usd', 0)
                    })
            
            self.logger.info(f"取引履歴をCSVファイルにエクスポートしました: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"CSV エクスポートエラー: {e}")
            return ""