#!/usr/bin/env python3
"""
Run comprehensive backtests for MEXC trading bot
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
import json
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from backtest import BacktestEngine
from config import Config
from utils import setup_logging


def generate_backtest_report(results: Dict[str, Dict], symbols: List[str]) -> str:
    """Generate a comprehensive backtest report"""
    report = []
    report.append("# MEXC Trading Bot - Backtest Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Symbols tested: {len(symbols)}")
    report.append("\n## Summary Results\n")
    
    # Create summary table
    report.append("| Period | Total Trades | Win Rate | ROI | Max Drawdown | Sharpe Ratio | Profit Factor |")
    report.append("|--------|--------------|----------|-----|--------------|--------------|---------------|")
    
    for period, metrics in results.items():
        if metrics:
            report.append(
                f"| {period.replace('_', ' ').title()} | "
                f"{metrics.get('total_trades', 0)} | "
                f"{metrics.get('win_rate', 0):.1f}% | "
                f"{metrics.get('roi', 0):.1f}% | "
                f"{metrics.get('max_drawdown', 0):.1f}% | "
                f"{metrics.get('sharpe_ratio', 0):.2f} | "
                f"{metrics.get('profit_factor', 0):.2f} |"
            )
    
    # Detailed results for each period
    for period, metrics in results.items():
        if not metrics:
            continue
            
        report.append(f"\n## {period.replace('_', ' ').title()} Detailed Results\n")
        report.append(f"- **Total Trades**: {metrics.get('total_trades', 0)}")
        report.append(f"- **Winning Trades**: {metrics.get('winning_trades', 0)}")
        report.append(f"- **Losing Trades**: {metrics.get('losing_trades', 0)}")
        report.append(f"- **Win Rate**: {metrics.get('win_rate', 0):.2f}%")
        report.append(f"- **Average Win**: ${metrics.get('average_win', 0):.2f}")
        report.append(f"- **Average Loss**: ${metrics.get('average_loss', 0):.2f}")
        report.append(f"- **Total P&L**: ${metrics.get('total_pnl', 0):.2f}")
        report.append(f"- **ROI**: {metrics.get('roi', 0):.2f}%")
        report.append(f"- **Final Balance**: ${metrics.get('final_balance', 0):.2f}")
        report.append(f"- **Average Hold Time**: {metrics.get('average_hold_time', 0):.1f} minutes")
        
        # Exit reason distribution
        if 'trades_by_exit_reason' in metrics:
            report.append("\n### Exit Reason Distribution")
            for reason, count in metrics['trades_by_exit_reason'].items():
                percentage = (count / metrics.get('total_trades', 1)) * 100
                report.append(f"- {reason}: {count} ({percentage:.1f}%)")
    
    # Trading recommendations
    report.append("\n## Trading Recommendations\n")
    
    # Analyze results and provide recommendations
    avg_win_rate = sum(r.get('win_rate', 0) for r in results.values() if r) / len([r for r in results.values() if r])
    avg_roi = sum(r.get('roi', 0) for r in results.values() if r) / len([r for r in results.values() if r])
    
    if avg_win_rate >= 65:
        report.append("✅ **Win rate is excellent** (>65%). The strategy shows consistent profitability.")
    elif avg_win_rate >= 55:
        report.append("⚠️ **Win rate is good** (55-65%). Consider fine-tuning entry conditions.")
    else:
        report.append("❌ **Win rate needs improvement** (<55%). Review entry/exit criteria.")
    
    if avg_roi >= 50:
        report.append("✅ **ROI is excellent** (>50%). The strategy generates strong returns.")
    elif avg_roi >= 20:
        report.append("⚠️ **ROI is acceptable** (20-50%). Look for optimization opportunities.")
    else:
        report.append("❌ **ROI is low** (<20%). Consider adjusting position sizing or targets.")
    
    # Risk assessment
    max_dd = max(r.get('max_drawdown', 0) for r in results.values() if r)
    if max_dd > 20:
        report.append("⚠️ **High drawdown risk** (>20%). Consider tighter risk management.")
    
    # Market condition analysis
    report.append("\n### Market Condition Considerations")
    report.append("- The backtest covers various market conditions over different timeframes")
    report.append("- Shorter periods may not capture full market cycles")
    report.append("- Consider the impact of major market events during test periods")
    
    return "\n".join(report)


def create_performance_charts(results: Dict[str, Dict]):
    """Create performance visualization charts"""
    # Setup the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MEXC Trading Bot - Backtest Performance Analysis', fontsize=16)
    
    # Extract data for plotting
    periods = list(results.keys())
    win_rates = [results[p].get('win_rate', 0) for p in periods]
    rois = [results[p].get('roi', 0) for p in periods]
    max_dds = [results[p].get('max_drawdown', 0) for p in periods]
    profit_factors = [results[p].get('profit_factor', 0) for p in periods]
    
    # 1. Win Rate by Period
    ax1 = axes[0, 0]
    bars1 = ax1.bar(periods, win_rates, color='green', alpha=0.7)
    ax1.axhline(y=65, color='r', linestyle='--', label='Target: 65%')
    ax1.set_title('Win Rate by Period')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_ylim(0, 100)
    ax1.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 2. ROI by Period
    ax2 = axes[0, 1]
    bars2 = ax2.bar(periods, rois, color='blue', alpha=0.7)
    ax2.axhline(y=65, color='r', linestyle='--', label='Target: 65%')
    ax2.set_title('Return on Investment by Period')
    ax2.set_ylabel('ROI (%)')
    ax2.legend()
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 3. Max Drawdown by Period
    ax3 = axes[1, 0]
    bars3 = ax3.bar(periods, max_dds, color='red', alpha=0.7)
    ax3.axhline(y=15, color='g', linestyle='--', label='Target: <15%')
    ax3.set_title('Maximum Drawdown by Period')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.legend()
    ax3.invert_yaxis()  # Invert y-axis since lower is better
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{abs(height):.1f}%', ha='center', va='top')
    
    # 4. Profit Factor by Period
    ax4 = axes[1, 1]
    bars4 = ax4.bar(periods, profit_factors, color='purple', alpha=0.7)
    ax4.axhline(y=2.0, color='g', linestyle='--', label='Good: >2.0')
    ax4.set_title('Profit Factor by Period')
    ax4.set_ylabel('Profit Factor')
    ax4.legend()
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs('backtest_results', exist_ok=True)
    plt.savefig('backtest_results/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create equity curve comparison if available
    try:
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        for period in periods:
            equity_file = f"backtest_results/equity_*_{period}.json"
            # Note: This is simplified - in production you'd load actual equity curves
            
        ax.set_title('Equity Curves Comparison')
        ax.set_xlabel('Time')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        plt.savefig('backtest_results/equity_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    except:
        pass


async def run_comprehensive_backtest():
    """Run comprehensive backtests with all monitored symbols"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize configuration
    config = Config()
    
    # Set test mode and load config
    os.environ['TEST_MODE'] = 'true'  # Ensure we're in test mode
    config.load_from_env()
    
    # Get all monitored symbols
    tier1 = config.tier1_symbols.split(',') if config.tier1_symbols else []
    tier2 = config.tier2_symbols.split(',') if config.tier2_symbols else []
    tier3 = config.tier3_symbols.split(',') if config.tier3_symbols else []
    
    all_symbols = []
    for symbol in tier1 + tier2 + tier3:
        symbol = symbol.strip()
        if symbol:
            all_symbols.append(symbol + 'USDT')
    
    logger.info(f"Total symbols to test: {len(all_symbols)}")
    logger.info(f"Symbols: {all_symbols}")
    
    # Initialize backtest engine
    engine = BacktestEngine(config, initial_balance=1000.0)
    
    # Define time periods
    end_date = datetime.now()
    time_periods = [
        ("1_month", end_date - timedelta(days=30)),
        ("3_months", end_date - timedelta(days=90)),
        ("6_months", end_date - timedelta(days=180)),
        ("1_year", end_date - timedelta(days=365))
    ]
    
    # Run backtests
    all_results = {}
    
    for period_name, start_date in time_periods:
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting backtest for {period_name.replace('_', ' ').upper()}")
        logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"{'='*80}")
        
        try:
            # Run backtest (limit symbols for API rate limits)
            # In production, you might want to batch this or add delays
            symbols_to_test = all_symbols[:15]  # Test top 15 symbols
            
            results = await engine.run_backtest(symbols_to_test, start_date, end_date)
            all_results[period_name] = results
            
            # Print summary
            if results:
                logger.info(f"\n📊 {period_name.upper()} SUMMARY:")
                logger.info(f"├─ Total Trades: {results.get('total_trades', 0)}")
                logger.info(f"├─ Win Rate: {results.get('win_rate', 0):.2f}%")
                logger.info(f"├─ Total P&L: ${results.get('total_pnl', 0):.2f}")
                logger.info(f"├─ ROI: {results.get('roi', 0):.2f}%")
                logger.info(f"├─ Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
                logger.info(f"├─ Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
                logger.info(f"└─ Profit Factor: {results.get('profit_factor', 0):.2f}")
            
        except Exception as e:
            logger.error(f"Error running backtest for {period_name}: {e}")
            all_results[period_name] = {}
        
        # Add delay between periods to respect API limits
        await asyncio.sleep(2)
    
    # Generate comprehensive report
    logger.info("\n📝 Generating comprehensive report...")
    report = generate_backtest_report(all_results, all_symbols)
    
    # Save report
    os.makedirs('backtest_results', exist_ok=True)
    report_file = f"backtest_results/comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save JSON results
    json_file = f"backtest_results/all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate performance charts
    try:
        logger.info("📈 Creating performance charts...")
        create_performance_charts(all_results)
    except Exception as e:
        logger.error(f"Error creating charts: {e}")
    
    logger.info(f"\n✅ Backtest completed!")
    logger.info(f"📄 Report saved to: {report_file}")
    logger.info(f"📊 Results saved to: {json_file}")
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY - AVERAGE ACROSS ALL PERIODS")
    logger.info("="*80)
    
    valid_results = [r for r in all_results.values() if r and r.get('total_trades', 0) > 0]
    if valid_results:
        avg_metrics = {
            'win_rate': sum(r.get('win_rate', 0) for r in valid_results) / len(valid_results),
            'roi': sum(r.get('roi', 0) for r in valid_results) / len(valid_results),
            'max_drawdown': max(r.get('max_drawdown', 0) for r in valid_results),
            'sharpe_ratio': sum(r.get('sharpe_ratio', 0) for r in valid_results) / len(valid_results),
            'profit_factor': sum(r.get('profit_factor', 0) for r in valid_results) / len(valid_results)
        }
        
        logger.info(f"Average Win Rate: {avg_metrics['win_rate']:.2f}%")
        logger.info(f"Average ROI: {avg_metrics['roi']:.2f}%")
        logger.info(f"Worst Drawdown: {avg_metrics['max_drawdown']:.2f}%")
        logger.info(f"Average Sharpe Ratio: {avg_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Average Profit Factor: {avg_metrics['profit_factor']:.2f}")
        
        # Performance assessment
        if avg_metrics['win_rate'] >= 65 and avg_metrics['roi'] >= 65:
            logger.info("\n✅ Strategy meets performance targets!")
        else:
            logger.info("\n⚠️ Strategy needs optimization to meet targets.")
    
    return all_results


if __name__ == "__main__":
    # Check for matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        logging.warning("matplotlib not installed. Charts will not be generated.")
        logging.warning("Install with: pip install matplotlib seaborn")
    
    # Run the backtest
    asyncio.run(run_comprehensive_backtest())