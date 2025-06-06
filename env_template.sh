# .env.template
# 環境変数設定のテンプレートファイル
# このファイルを .env にコピーして、適切な値を設定してください
# 重要: .envファイルはGitリポジトリにコミットしないでください（機密情報が含まれるため）

# ==============================================
# MEXC API認証情報（本番モードでのみ必須）
# ==============================================
MEXC_API_KEY=""
MEXC_SECRET_KEY=""

# ==============================================
# Discord通知設定（必須）
# ==============================================
DISCORD_WEBHOOK_URL=""

# ==============================================
# Botの動作モード
# ==============================================
# True: テストモード（架空取引）, False: 本番モード（実取引）
TEST_MODE="True"

# ==============================================
# 基本取引設定
# ==============================================
# テストモードでの初期資金（USD）
INITIAL_CAPITAL_USD="1000"

# 1取引あたりのリスク許容額（総資金のパーセンテージ）
POSITION_SIZE_PERCENTAGE="5.0"

# 同時保有ポジション上限
MAX_SIMULTANEOUS_POSITIONS="2"

# 同一銘柄への連続エントリー制限（24時間以内）
MAX_CONSECUTIVE_ENTRIES_SAME_COIN="2"

# 最大監視銘柄数
MAX_MONITORING_SYMBOLS="25"

# ==============================================
# テクニカル指標パラメータ
# ==============================================
# ボリンジャーバンド設定
BB_PERIOD="20"
BB_STD_DEV="2.0"

# RSI設定
RSI_PERIOD="14"

# MACD設定
MACD_FAST_EMA_PERIOD="12"
MACD_SLOW_EMA_PERIOD="26"
MACD_SIGNAL_SMA_PERIOD="9"

# EMA期間設定（カンマ区切りで複数指定）
EMA_PERIODS="20,50"

# ==============================================
# 利益確定・損切りパラメータ
# ==============================================
# 1段階目利益確定（手数料考慮後の実質利益率 %）
TAKE_PROFIT_PERCENTAGE_PHASE1="2.4"

# 2段階目利益確定オプション1（価格変動による決済 %）
TAKE_PROFIT_PERCENTAGE_PHASE2_OPTION1="5.0"

# 2段階目利益確定オプション3（利益到達による決済 %）
TAKE_PROFIT_PERCENTAGE_PHASE2_OPTION3="3.0"

# トレーリングストップ発動条件（利益率 %）
TRAILING_STOP_ACTIVATION_PERCENTAGE="2.4"

# トレーリングストップ逆行率（%）
TRAILING_STOP_PERCENTAGE="1.5"

# 固定損切り率（%）
FIXED_STOP_LOSS_PERCENTAGE="1.5"

# 時間経過損切り（分）
TIME_BASED_STOP_LOSS_MINUTES="30"

# 時間経過損切り発動の利益閾値（%）
TIME_BASED_STOP_LOSS_PROFIT_THRESHOLD="1.0"

# ==============================================
# 改善戦略パラメータ
# ==============================================
# 下落相場でのショートRSI閾値
SHORT_RSI_THRESHOLD_BEAR_MARKET="35"

# 最適時間帯でのロング価格変動率閾値
PRICE_CHANGE_RATE_OPTIMIZED_HOURS_LONG="1.5"

# 最適時間帯でのショート価格変動率閾値
PRICE_CHANGE_RATE_OPTIMIZED_HOURS_SHORT="1.3"

# その他時間帯での価格変動率閾値
PRICE_CHANGE_RATE_OTHER_HOURS="1.8"

# 最適時間帯でのロングRSI閾値
RSI_THRESHOLD_LONG_OPTIMIZED_HOURS="60"

# その他時間帯でのロングRSI閾値
RSI_THRESHOLD_LONG_OTHER_HOURS="65"

# 最適時間帯でのショートRSI閾値
RSI_THRESHOLD_SHORT_OPTIMIZED_HOURS="35"

# その他時間帯でのショートRSI閾値
RSI_THRESHOLD_SHORT_OTHER_HOURS="35"

# 最適時間帯での出来高倍率
VOLUME_MULTIPLIER_OPTIMIZED_HOURS="2.0"

# その他時間帯での出来高倍率
VOLUME_MULTIPLIER_OTHER_HOURS="3.0"

# スキャルピング戦略のRSI閾値
SCALPING_LONG_RSI_THRESHOLD="30"

# スキャルピング戦略の価格下落率（%）
SCALPING_LONG_PRICE_DROP_PERCENTAGE="4.0"

# BTC急落時の緊急損切り発動率（%）
BTC_CRASH_STOP_LOSS_PERCENTAGE="3.0"

# 許容スリッページ率（%）
SLIPPAGE_PERCENTAGE="0.2"

# ==============================================
# 価格表示機能設定（新機能）
# ==============================================
# 価格表示機能の有効/無効
PRICE_DISPLAY_ENABLED="True"

# 価格表示間隔（秒）
PRICE_DISPLAY_INTERVAL_SECONDS="10"

# ==============================================
# 日次レポート機能設定（新機能）
# ==============================================
# 日次レポート機能の有効/無効
DAILY_REPORT_ENABLED="True"

# 日次レポート生成時刻（JST）
DAILY_REPORT_TIME_JST="00:00"

# Discord日次レポート通知の有効/無効
DAILY_REPORT_DISCORD_ENABLED="True"

# ==============================================
# パフォーマンス設定（新機能）
# ==============================================
# APIレスポンスキャッシュ持続時間（秒）
API_CACHE_DURATION_SECONDS="5"

# 最大同時API呼び出し数
MAX_CONCURRENT_API_CALLS="3"

# ==============================================
# 新しいリスク管理設定（改善版）
# ==============================================
# 最小保有時間（分）- この時間未満はポジションを決済しない
MINIMUM_HOLDING_TIME_MINUTES="5"

# 最小取引サイズ（USD）- この金額未満の取引は実行しない
MINIMUM_TRADE_SIZE_USD="10.0"

# 最小損失額（USD）- この金額未満の損失は連続損失としてカウントしない
MINIMUM_LOSS_AMOUNT_USD="0.50"

# 連続損失による取引停止の閾値
CONSECUTIVE_LOSS_THRESHOLD="3"

# ==============================================
# 改善された資金管理設定（Ultra-Think実装）
# ==============================================
# 動的ポジションサイジングの有効化
DYNAMIC_POSITION_SIZING_ENABLED="True"

# 時間帯別ポジションサイジングの有効化
TIME_BASED_POSITION_SIZING="True"

# 最適時間帯のポジションサイズ（%）
OPTIMIZED_HOURS_POSITION_SIZE="6.0"

# 非最適時間帯のポジションサイズ（%）
NON_OPTIMIZED_HOURS_POSITION_SIZE="4.0"

# ピラミッディング機能の有効化
PYRAMIDING_ENABLED="True"

# ピラミッディング発動の利益閾値（%）
PYRAMIDING_THRESHOLD_PERCENTAGE="2.0"

# ピラミッディング追加ポジションサイズ（%）
PYRAMIDING_SIZE_PERCENTAGE="3.0"

# 最大ポートフォリオリスク（%）
MAX_PORTFOLIO_RISK_PERCENTAGE="25.0"