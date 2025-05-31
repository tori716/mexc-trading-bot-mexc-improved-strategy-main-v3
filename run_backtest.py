#!/usr/bin/env python3
"""
バックテスト実行スクリプト
4つの期間（1ヶ月、3ヶ月、6ヶ月、1年）でバックテストを実行し、
結果をレポートとして出力する
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties

from backtest import BacktestEngine
from config import Config
from utils import setup_logging


async def run_backtest_for_period(config: Config, logger: logging.Logger, 
                                 symbols: List[str], period_days: int, 
                                 period_name: str) -> Dict:
    """
    指定期間でバックテストを実行
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{period_name}のバックテストを開始します")
    logger.info(f"{'='*60}\n")
    
    # バックテストエンジンを作成
    engine = BacktestEngine(config, logger)
    
    # バックテストを実行
    results = await engine.run_backtest(symbols, period_days)
    
    # 結果を保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_results/{period_name}_{timestamp}.json"
    os.makedirs("backtest_results", exist_ok=True)
    engine.save_results(filename)
    
    return results


def generate_performance_report(all_results: Dict[str, Dict], config: Config) -> str:
    """
    全期間の結果をまとめたパフォーマンスレポートを生成
    """
    report = []
    report.append("# バックテスト総合レポート")
    report.append(f"\n実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"初期資金: ${config.INITIAL_CAPITAL_USD:,.2f}")
    report.append("\n## 📊 パフォーマンスサマリー\n")
    
    # 期間ごとの結果をテーブル形式で表示
    report.append("| 期間 | 取引数 | 勝率 | ROI | 最大DD | PF | シャープ比 |")
    report.append("|------|--------|------|-----|--------|-----|-----------|")
    
    for period_name, results in all_results.items():
        metrics = results.get('metrics', {})
        report.append(f"| {period_name} | {metrics.get('total_trades', 0)} | "
                     f"{metrics.get('win_rate', 0):.1f}% | "
                     f"{metrics.get('roi', 0):.1f}% | "
                     f"{metrics.get('max_drawdown', 0):.1f}% | "
                     f"{metrics.get('profit_factor', 0):.2f} | "
                     f"{metrics.get('sharpe_ratio', 0):.2f} |")
    
    # 詳細分析
    report.append("\n## 📈 期間別詳細分析\n")
    
    for period_name, results in all_results.items():
        metrics = results.get('metrics', {})
        trades = results.get('trades', [])
        
        report.append(f"### {period_name}")
        report.append(f"- **期間**: {results.get('start_date', 'N/A')} ～ {results.get('end_date', 'N/A')}")
        report.append(f"- **監視銘柄数**: {len(results.get('symbols', []))}")
        report.append(f"- **総取引数**: {metrics.get('total_trades', 0)}")
        report.append(f"- **勝ちトレード**: {metrics.get('winning_trades', 0)}")
        report.append(f"- **負けトレード**: {metrics.get('losing_trades', 0)}")
        report.append(f"- **勝率**: {metrics.get('win_rate', 0):.2f}%")
        report.append(f"- **総収益**: ${metrics.get('total_return', 0):,.2f}")
        report.append(f"- **ROI**: {metrics.get('roi', 0):.2f}%")
        report.append(f"- **最大ドローダウン**: {metrics.get('max_drawdown', 0):.2f}%")
        report.append(f"- **プロフィットファクター**: {metrics.get('profit_factor', 0):.2f}")
        report.append(f"- **シャープレシオ**: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"- **平均勝ち**: ${metrics.get('average_win', 0):.2f}")
        report.append(f"- **平均負け**: ${metrics.get('average_loss', 0):.2f}")
        report.append(f"- **最大勝ち**: ${metrics.get('largest_win', 0):.2f}")
        report.append(f"- **最大負け**: ${metrics.get('largest_loss', 0):.2f}")
        
        # 決済理由の分析
        exit_reasons = metrics.get('exit_reasons', {})
        if exit_reasons:
            report.append("\n**決済理由の内訳:**")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / metrics.get('total_trades', 1)) * 100
                report.append(f"- {reason}: {count}回 ({percentage:.1f}%)")
        
        # 銘柄別パフォーマンス
        if trades:
            symbol_performance = {}
            for trade in trades:
                symbol = trade.get('symbol', 'unknown')
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = {
                        'trades': 0,
                        'wins': 0,
                        'total_pnl': 0
                    }
                symbol_performance[symbol]['trades'] += 1
                if trade.get('pnl_amount', 0) > 0:
                    symbol_performance[symbol]['wins'] += 1
                symbol_performance[symbol]['total_pnl'] += trade.get('pnl_amount', 0)
            
            report.append("\n**銘柄別パフォーマンス (上位5):**")
            sorted_symbols = sorted(symbol_performance.items(), 
                                  key=lambda x: x[1]['total_pnl'], 
                                  reverse=True)[:5]
            for symbol, perf in sorted_symbols:
                win_rate = (perf['wins'] / perf['trades'] * 100) if perf['trades'] > 0 else 0
                report.append(f"- {symbol}: {perf['trades']}取引, "
                            f"勝率{win_rate:.1f}%, "
                            f"損益${perf['total_pnl']:.2f}")
        
        report.append("")
    
    # 戦略の評価
    report.append("\n## 🎯 戦略評価\n")
    
    # 全期間の平均パフォーマンスを計算
    avg_win_rate = sum(r['metrics'].get('win_rate', 0) for r in all_results.values()) / len(all_results)
    avg_roi = sum(r['metrics'].get('roi', 0) for r in all_results.values()) / len(all_results)
    avg_sharpe = sum(r['metrics'].get('sharpe_ratio', 0) for r in all_results.values()) / len(all_results)
    
    report.append(f"### 平均パフォーマンス")
    report.append(f"- **平均勝率**: {avg_win_rate:.2f}%")
    report.append(f"- **平均ROI**: {avg_roi:.2f}%")
    report.append(f"- **平均シャープレシオ**: {avg_sharpe:.2f}")
    
    # 目標との比較
    report.append("\n### 目標達成度")
    target_win_rate = 65
    target_roi_monthly = 65
    target_sharpe = 2.5
    
    # 月次ROIに換算（簡易計算）
    monthly_roi_1m = all_results.get('1ヶ月', {}).get('metrics', {}).get('roi', 0)
    monthly_roi_3m = all_results.get('3ヶ月', {}).get('metrics', {}).get('roi', 0) / 3
    monthly_roi_6m = all_results.get('6ヶ月', {}).get('metrics', {}).get('roi', 0) / 6
    monthly_roi_1y = all_results.get('1年', {}).get('metrics', {}).get('roi', 0) / 12
    avg_monthly_roi = (monthly_roi_1m + monthly_roi_3m + monthly_roi_6m + monthly_roi_1y) / 4
    
    report.append(f"- **勝率目標 (≥{target_win_rate}%)**: "
                 f"{'✅ 達成' if avg_win_rate >= target_win_rate else '❌ 未達成'} "
                 f"({avg_win_rate:.1f}%)")
    report.append(f"- **月次ROI目標 (≥{target_roi_monthly}%)**: "
                 f"{'✅ 達成' if avg_monthly_roi >= target_roi_monthly else '❌ 未達成'} "
                 f"({avg_monthly_roi:.1f}%)")
    report.append(f"- **シャープレシオ目標 (≥{target_sharpe})**: "
                 f"{'✅ 達成' if avg_sharpe >= target_sharpe else '❌ 未達成'} "
                 f"({avg_sharpe:.2f})")
    
    # 推奨事項
    report.append("\n## 💡 推奨事項\n")
    
    if avg_win_rate < target_win_rate:
        report.append("- **勝率改善**: エントリー条件をより厳格にすることを検討してください")
    
    if avg_monthly_roi < 0:
        report.append("- **損失対策**: ストップロス戦略の見直しが必要です")
    elif avg_monthly_roi < target_roi_monthly:
        report.append("- **収益性向上**: 利確目標の調整やポジションサイズの最適化を検討してください")
    
    if avg_sharpe < 1:
        report.append("- **リスク調整**: ボラティリティが高すぎる可能性があります")
    elif avg_sharpe < target_sharpe:
        report.append("- **安定性向上**: より一貫した収益を目指すため、戦略の安定化を図ってください")
    
    # 最も収益性の高い銘柄
    all_symbol_performance = {}
    for results in all_results.values():
        for trade in results.get('trades', []):
            symbol = trade.get('symbol', 'unknown')
            if symbol not in all_symbol_performance:
                all_symbol_performance[symbol] = 0
            all_symbol_performance[symbol] += trade.get('pnl_amount', 0)
    
    if all_symbol_performance:
        top_symbols = sorted(all_symbol_performance.items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:5]
        report.append("\n### 🏆 最も収益性の高い銘柄（全期間）")
        for symbol, total_pnl in top_symbols:
            report.append(f"- {symbol}: ${total_pnl:.2f}")
    
    return "\n".join(report)


def create_performance_charts(all_results: Dict[str, Dict]):
    """
    パフォーマンスチャートを作成
    """
    # 日本語フォントの設定を試みる
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    except:
        pass
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Backtest Performance Analysis', fontsize=16)
    
    periods = list(all_results.keys())
    
    # 1. 勝率比較
    win_rates = [all_results[p]['metrics'].get('win_rate', 0) for p in periods]
    axes[0, 0].bar(periods, win_rates)
    axes[0, 0].set_title('Win Rate by Period')
    axes[0, 0].set_ylabel('Win Rate (%)')
    axes[0, 0].axhline(y=65, color='r', linestyle='--', label='Target (65%)')
    axes[0, 0].legend()
    
    # 2. ROI比較
    rois = [all_results[p]['metrics'].get('roi', 0) for p in periods]
    axes[0, 1].bar(periods, rois)
    axes[0, 1].set_title('ROI by Period')
    axes[0, 1].set_ylabel('ROI (%)')
    
    # 3. ドローダウン比較
    drawdowns = [all_results[p]['metrics'].get('max_drawdown', 0) for p in periods]
    axes[1, 0].bar(periods, drawdowns, color='red')
    axes[1, 0].set_title('Max Drawdown by Period')
    axes[1, 0].set_ylabel('Max Drawdown (%)')
    axes[1, 0].axhline(y=15, color='r', linestyle='--', label='Limit (15%)')
    axes[1, 0].legend()
    
    # 4. シャープレシオ比較
    sharpe_ratios = [all_results[p]['metrics'].get('sharpe_ratio', 0) for p in periods]
    axes[1, 1].bar(periods, sharpe_ratios, color='green')
    axes[1, 1].set_title('Sharpe Ratio by Period')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].axhline(y=2.5, color='r', linestyle='--', label='Target (2.5)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # 保存
    os.makedirs("backtest_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"backtest_results/performance_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # エクイティカーブも作成
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    
    for period_name, results in all_results.items():
        equity_curve = results.get('equity_curve', [])
        if equity_curve:
            timestamps = [e['timestamp'] for e in equity_curve]
            equities = [e['equity'] for e in equity_curve]
            
            # タイムスタンプを日付に変換
            dates = [datetime.fromtimestamp(ts) for ts in timestamps]
            
            # データポイントを間引く（表示を見やすくするため）
            step = max(1, len(dates) // 1000)
            dates = dates[::step]
            equities = equities[::step]
            
            ax2.plot(dates, equities, label=period_name, linewidth=2)
    
    ax2.set_title('Equity Curves by Period')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Equity ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # X軸の日付フォーマット
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"backtest_results/equity_curves_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()


async def main():
    """
    メイン実行関数
    """
    # 設定とロガーの初期化
    config = Config()
    logger = setup_logging()
    
    logger.info("バックテストシステムを起動します...")
    
    # 監視銘柄を取得
    symbols = config.get_all_target_symbols()
    logger.info(f"監視銘柄数: {len(symbols)}")
    logger.info(f"監視銘柄: {symbols}")
    
    # バックテスト期間の定義
    test_periods = [
        ("1ヶ月", 30),
        ("3ヶ月", 90),
        ("6ヶ月", 180),
        ("1年", 365)
    ]
    
    # 各期間でバックテストを実行
    all_results = {}
    
    for period_name, period_days in test_periods:
        try:
            results = await run_backtest_for_period(
                config, logger, symbols, period_days, period_name
            )
            all_results[period_name] = results
            
            # 結果のサマリーを表示
            metrics = results.get('metrics', {})
            logger.info(f"\n{period_name}の結果:")
            logger.info(f"  - 取引数: {metrics.get('total_trades', 0)}")
            logger.info(f"  - 勝率: {metrics.get('win_rate', 0):.2f}%")
            logger.info(f"  - ROI: {metrics.get('roi', 0):.2f}%")
            logger.info(f"  - 最大ドローダウン: {metrics.get('max_drawdown', 0):.2f}%")
            logger.info(f"  - シャープレシオ: {metrics.get('sharpe_ratio', 0):.2f}")
            
        except Exception as e:
            logger.error(f"{period_name}のバックテスト中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    # 総合レポートを生成
    if all_results:
        logger.info("\n総合レポートを生成中...")
        
        # テキストレポート
        report = generate_performance_report(all_results, config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"backtest_results/comprehensive_report_{timestamp}.md"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"総合レポートを保存しました: {report_filename}")
        
        # パフォーマンスチャート
        try:
            create_performance_charts(all_results)
            logger.info("パフォーマンスチャートを作成しました")
        except Exception as e:
            logger.error(f"チャート作成中にエラーが発生しました: {e}")
        
        # 全結果をJSONでも保存
        all_results_filename = f"backtest_results/all_results_{timestamp}.json"
        with open(all_results_filename, 'w', encoding='utf-8') as f:
            # datetime を文字列に変換
            all_results_serializable = {}
            for period, results in all_results.items():
                results_copy = results.copy()
                if 'start_date' in results_copy:
                    results_copy['start_date'] = results_copy['start_date'].isoformat() if hasattr(results_copy['start_date'], 'isoformat') else str(results_copy['start_date'])
                if 'end_date' in results_copy:
                    results_copy['end_date'] = results_copy['end_date'].isoformat() if hasattr(results_copy['end_date'], 'isoformat') else str(results_copy['end_date'])
                all_results_serializable[period] = results_copy
            
            json.dump(all_results_serializable, f, ensure_ascii=False, indent=2)
        
        logger.info(f"全結果を保存しました: {all_results_filename}")
    
    logger.info("\nバックテストが完了しました！")
    logger.info("結果は backtest_results/ ディレクトリに保存されています。")


if __name__ == "__main__":
    asyncio.run(main())