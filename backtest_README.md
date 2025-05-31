# MEXC Trading Bot - Backtesting Module

## 概要

このバックテストモジュールは、MEXC取引ボットの戦略を過去の市場データでシミュレーションし、パフォーマンスを評価するためのツールです。

## 特徴

- 📊 **複数期間の分析**: 1ヶ月、3ヶ月、6ヶ月、1年の4つの期間でバックテスト
- 📈 **包括的なメトリクス**: 勝率、ROI、最大ドローダウン、シャープレシオなど
- 🎯 **実際の取引ロジック**: 本番と同じエントリー/エグジット条件を使用
- 📝 **詳細なレポート**: Markdown形式の分析レポートとJSON形式の生データ
- 📉 **ビジュアル分析**: パフォーマンスチャートの自動生成

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境設定

```bash
cp env_template.sh .env
# .envファイルを編集してMEXC APIキーを設定
source .env
```

## 使用方法

### 基本的な実行

```bash
python run_backtest.py
```

### カスタムバックテスト

```python
from backtest import BacktestEngine
from config import Config
from datetime import datetime, timedelta

# 設定の初期化
config = Config()
config.load_from_env()

# バックテストエンジンの作成
engine = BacktestEngine(config, initial_balance=1000.0)

# 特定期間でのバックテスト実行
end_date = datetime.now()
start_date = end_date - timedelta(days=90)  # 3ヶ月前
symbols = ['BTCUSDT', 'ETHUSDT', 'AVAXUSDT']

results = await engine.run_backtest(symbols, start_date, end_date)
```

## 出力ファイル

バックテスト実行後、以下のファイルが `backtest_results/` ディレクトリに生成されます：

### 1. 包括的レポート
- `comprehensive_report_YYYYMMDD_HHMMSS.md`: Markdown形式の分析レポート
- 各期間の詳細な統計情報と推奨事項を含む

### 2. JSONデータ
- `all_results_YYYYMMDD_HHMMSS.json`: 全期間の統計データ
- `trades_*.json`: 個別取引の詳細
- `equity_*.json`: ポートフォリオ価値の推移

### 3. パフォーマンスチャート
- `performance_analysis.png`: 主要メトリクスの比較チャート
  - 勝率の推移
  - ROIの比較
  - 最大ドローダウン
  - プロフィットファクター

## メトリクスの説明

### 基本メトリクス
- **勝率 (Win Rate)**: 利益を出した取引の割合
- **ROI**: 投資収益率
- **総損益 (Total P&L)**: 全取引の合計損益

### リスクメトリクス
- **最大ドローダウン**: ピークからの最大下落率
- **シャープレシオ**: リスク調整後リターン
- **プロフィットファクター**: 総利益÷総損失

### 取引統計
- **平均勝ち**: 勝ち取引の平均利益
- **平均負け**: 負け取引の平均損失
- **平均保有時間**: ポジションの平均保有時間

## パフォーマンス目標

現在の戦略は以下の目標を設定しています：

| メトリクス | 目標値 | 説明 |
|-----------|--------|------|
| 勝率 | ≥ 65% | 高い一貫性 |
| 月間ROI | ≥ 65% | 積極的な成長目標 |
| 最大DD | ≤ 15% | リスク管理 |
| シャープレシオ | ≥ 2.5 | 優れたリスク調整後リターン |

## 注意事項

### API制限
- MEXCのAPIレート制限に注意してください
- 大量のシンボルでバックテストする場合は、適切な遅延を設定してください

### データの精度
- 5分足データを使用しています
- スリッページや約定遅延は考慮されていません
- 実際の取引では追加のコストが発生する可能性があります

### メモリ使用量
- 長期間のバックテストは大量のメモリを使用します
- 必要に応じてシンボル数や期間を調整してください

## トラブルシューティング

### "No historical data available"エラー
- APIキーが正しく設定されているか確認
- シンボル名が正しいか確認（例: BTCUSDT）
- ネットワーク接続を確認

### メモリ不足エラー
- テストするシンボル数を減らす
- より短い期間でテストする
- データをバッチで処理する

### チャートが生成されない
- matplotlibとseabornがインストールされているか確認
- `pip install matplotlib seaborn`

## 改善案

将来的な改善として以下を検討：

1. **より詳細なシミュレーション**
   - スリッページのモデリング
   - 流動性の考慮
   - 注文タイプの多様化

2. **最適化機能**
   - パラメータの自動最適化
   - ウォークフォワード分析
   - モンテカルロシミュレーション

3. **追加分析**
   - 銘柄別パフォーマンス
   - 時間帯別分析
   - 市場環境別分析