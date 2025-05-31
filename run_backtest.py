#!/usr/bin/env python3
"""
バックテスト実行スクリプト
4つの期間（1ヶ月、3ヶ月、6ヶ月、1年）でバックテストを実行し、
結果をレポートとして出力します。
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from backtest import BacktestEngine

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# バックテスト期間の定義
BACKTEST_PERIODS = {
    '1_month': 30,
    '3_months': 90,
    '6_months': 180,
    '1_year': 365
}

def create_results_directory():
    """結果保存用のディレクトリを作成"""
    os.makedirs('backtest_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'backtest_results/run_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def generate_markdown_report(results: Dict[str, Dict[str, Any]], 
                           results_dir: str) -> str:
    """
    バックテスト結果のMarkdownレポートを生成
    """
    timestamp = datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')
    
    report = f"""# バックテスト結果レポート

生成日時: {timestamp}

## 📊 概要

このレポートは、MEXC取引ボットの戦略を過去データでバックテストした結果をまとめたものです。

## 🎯 テスト条件

- **初期資金**: $1,000
- **取引手数料**: テイカー 0.05% / メイカー 0%
- **監視銘柄**: Tier1銘柄を中心に最大25銘柄
- **戦略**: 改良版マルチインジケーター戦略（ボリンジャーバンド、RSI、MACD、EMA）

## 📈 期間別パフォーマンス

"""
    
    # 各期間の結果を表形式で表示
    report += "| 期間 | 総取引数 | 勝率 | 総収益率 | シャープレシオ | 最大DD | プロフィットファクター |\n"
    report += "|------|----------|------|----------|----------------|--------|----------------------|\n"
    
    for period, result in results.items():
        report += f"| {period.replace('_', ' ')} | "
        report += f"{result.get('total_trades', 0)} | "
        report += f"{result.get('win_rate', 0):.1f}% | "
        report += f"{result.get('total_return', 0):.2f}% | "
        report += f"{result.get('sharpe_ratio', 0):.2f} | "
        report += f"{result.get('max_drawdown', 0):.1f}% | "
        report += f"{result.get('profit_factor', 0):.2f} |\n"
    
    # 詳細な分析
    report += "\n## 🔍 詳細分析\n\n"
    
    for period, result in results.items():
        report += f"### {period.replace('_', ' ').title()}\n\n"
        report += f"- **最終資金**: ${result.get('final_capital', 0):,.2f}\n"
        report += f"- **勝ちトレード**: {result.get('winning_trades', 0)}回\n"
        report += f"- **負けトレード**: {result.get('losing_trades', 0)}回\n"
        report += f"- **平均利益**: ${result.get('average_win', 0):.2f}\n"
        report += f"- **平均損失**: ${result.get('average_loss', 0):.2f}\n\n"
    
    # 推奨事項
    report += "## 💡 推奨事項\n\n"
    
    # 全期間の平均勝率を計算
    avg_win_rate = sum(r.get('win_rate', 0) for r in results.values()) / len(results)
    
    if avg_win_rate > 55:
        report += "✅ **良好なパフォーマンス**: 戦略は全体的に良好な結果を示しています。\n"
    elif avg_win_rate > 45:
        report += "⚠️ **改善の余地あり**: 戦略のパラメータ調整を検討してください。\n"
    else:
        report += "❌ **要改善**: 戦略の大幅な見直しが必要です。\n"
    
    # リスク管理の推奨
    max_dd_all = max(r.get('max_drawdown', 0) for r in results.values())
    if max_dd_all > 20:
        report += f"\n⚠️ **リスク警告**: 最大ドローダウンが{max_dd_all:.1f}%と高いため、ポジションサイズの見直しを推奨します。\n"
    
    report += "\n## 📝 注意事項\n\n"
    report += "- このバックテストは過去データに基づくものであり、将来の収益を保証するものではありません。\n"
    report += "- 実際の取引では、スリッページや約定の遅延などが発生する可能性があります。\n"
    report += "- 市場環境の変化により、戦略のパフォーマンスは変動する可能性があります。\n"
    
    return report

def create_performance_chart(results: Dict[str, Dict[str, Any]], 
                           results_dir: str):
    """
    パフォーマンスの可視化チャートを作成
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('バックテストパフォーマンス分析', fontsize=16)
    
    periods = list(results.keys())
    
    # 1. 収益率の比較
    returns = [results[p].get('total_return', 0) for p in periods]
    ax1 = axes[0, 0]
    ax1.bar(periods, returns, color=['green' if r > 0 else 'red' for r in returns])
    ax1.set_title('期間別収益率')
    ax1.set_ylabel('収益率 (%)')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 2. 勝率の比較
    win_rates = [results[p].get('win_rate', 0) for p in periods]
    ax2 = axes[0, 1]
    ax2.bar(periods, win_rates, color='blue')
    ax2.set_title('期間別勝率')
    ax2.set_ylabel('勝率 (%)')
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1)
    
    # 3. 最大ドローダウンの比較
    max_dds = [results[p].get('max_drawdown', 0) for p in periods]
    ax3 = axes[1, 0]
    ax3.bar(periods, max_dds, color='orange')
    ax3.set_title('期間別最大ドローダウン')
    ax3.set_ylabel('最大DD (%)')
    
    # 4. シャープレシオの比較
    sharpe_ratios = [results[p].get('sharpe_ratio', 0) for p in periods]
    ax4 = axes[1, 1]
    ax4.bar(periods, sharpe_ratios, color='purple')
    ax4.set_title('期間別シャープレシオ')
    ax4.set_ylabel('シャープレシオ')
    ax4.axhline(y=1, color='green', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    chart_path = os.path.join(results_dir, 'performance_analysis.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"パフォーマンスチャートを保存: {chart_path}")

async def run_all_backtests():
    """
    全ての期間でバックテストを実行
    """
    # 設定の読み込み
    config = Config()
    
    # 結果保存ディレクトリの作成
    results_dir = create_results_directory()
    logger.info(f"結果保存ディレクトリ: {results_dir}")
    
    # バックテストエンジンの初期化
    engine = BacktestEngine(config)
    
    # 監視銘柄の設定
    symbols = [s + "USDT" for s in config["TARGET_SYMBOLS_TIER1"]]
    
    # 各期間でバックテストを実行
    all_results = {}
    
    for period_name, days in BACKTEST_PERIODS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"{period_name.replace('_', ' ').upper()}のバックテストを開始...")
        logger.info(f"{'='*50}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            # バックテストの実行
            result = await engine.run_backtest(symbols, start_date, end_date)
            all_results[period_name] = result
            
            # 結果をJSON形式で保存
            result_file = os.path.join(results_dir, f'{period_name}_results.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ {period_name}のバックテスト完了")
            logger.info(f"  - 総取引数: {result.get('total_trades', 0)}")
            logger.info(f"  - 勝率: {result.get('win_rate', 0):.1f}%")
            logger.info(f"  - 総収益率: {result.get('total_return', 0):.2f}%")
            
        except Exception as e:
            logger.error(f"❌ {period_name}のバックテストでエラー: {str(e)}")
            all_results[period_name] = {
                'error': str(e),
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0
            }
    
    # 総合レポートの生成
    logger.info("\n📊 総合レポートを生成中...")
    
    # Markdownレポートの生成
    report = generate_markdown_report(all_results, results_dir)
    report_file = os.path.join(results_dir, 'comprehensive_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"✅ レポートを保存: {report_file}")
    
    # パフォーマンスチャートの生成
    try:
        create_performance_chart(all_results, results_dir)
    except Exception as e:
        logger.error(f"チャート生成エラー: {str(e)}")
    
    # サマリーの表示
    logger.info("\n" + "="*60)
    logger.info("バックテスト完了サマリー")
    logger.info("="*60)
    
    for period, result in all_results.items():
        if 'error' not in result:
            logger.info(f"\n{period.replace('_', ' ').upper()}:")
            logger.info(f"  初期資金: ${result.get('initial_capital', 0):,.2f}")
            logger.info(f"  最終資金: ${result.get('final_capital', 0):,.2f}")
            logger.info(f"  収益: ${result.get('final_capital', 0) - result.get('initial_capital', 0):,.2f}")
            logger.info(f"  収益率: {result.get('total_return', 0):.2f}%")
    
    logger.info(f"\n📁 全ての結果は {results_dir} に保存されました。")

def main():
    """メイン関数"""
    logger.info("🚀 MEXCトレーディングボット バックテストシステム")
    logger.info(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # イベントループの実行
        asyncio.run(run_all_backtests())
    except KeyboardInterrupt:
        logger.info("\n⚠️ ユーザーによってバックテストが中断されました。")
    except Exception as e:
        logger.error(f"\n❌ エラーが発生しました: {str(e)}")
        raise
    finally:
        logger.info(f"\n終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()