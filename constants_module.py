# constants.py
# Botで使用する定数を定義するモジュール

# ============================================
# 技術指標関連の定数
# ============================================

# ルックバック期間
LOOKBACK_PERIODS_30MIN = 6  # 30分間のデータ（5分足6本分）
VOLUME_ANALYSIS_PERIODS = 3  # 出来高分析期間（直近3本）
BTC_MONITORING_PERIODS = 2  # BTC監視期間（15分足2本分）

# ============================================
# API関連の定数
# ============================================

# APIレート制限対応
API_RATE_LIMIT_DELAY = 5  # API呼び出し間隔（秒）
API_RETRY_ATTEMPTS = 3  # APIリトライ回数
API_TIMEOUT_SECONDS = 30  # APIタイムアウト（秒）

# ============================================
# 時間関連の定数
# ============================================

# データ更新間隔
MARKET_DATA_UPDATE_INTERVAL = 10  # 市場データ更新間隔（秒）
POSITION_CHECK_INTERVAL = 60  # ポジションチェック間隔（秒）
SYMBOL_ROTATION_CHECK_INTERVAL = 300  # 銘柄ローテーションチェック間隔（秒）

# 時間足マッピング
TIMEFRAME_MAPPING = {
    "Min5": "5m",
    "Min15": "15m", 
    "Min60": "60m",
    "1h": "60m"
}

# ============================================
# リスク管理関連の定数
# ============================================

# 連続損失制限
MAX_CONSECUTIVE_LOSSES = 3  # 連続損失の上限

# ポジション管理
DEFAULT_POSITION_SIZE_PERCENTAGE = 5.0  # デフォルトポジションサイズ（%）
POSITION_SIZE_REDUCTION_FACTOR = 0.5  # 連続損失時のポジションサイズ減少係数

# 手数料率
MEXC_TAKER_FEE_RATE = 0.001  # MEXCテイカー手数料（0.1%）
MEXC_MAKER_FEE_RATE = 0.001  # MEXCメイカー手数料（0.1%）
ROUND_TRIP_FEE_RATE = 0.002  # 往復手数料（0.2%）

# ============================================
# テクニカル分析関連の定数
# ============================================

# RSI閾値
RSI_OVERBOUGHT_THRESHOLD = 70  # RSI買われすぎ閾値
RSI_OVERSOLD_THRESHOLD = 30   # RSI売られすぎ閾値

# ボリンジャーバンド
BB_DEFAULT_PERIOD = 20  # ボリンジャーバンドデフォルト期間
BB_DEFAULT_STD_DEV = 2.0  # ボリンジャーバンドデフォルト標準偏差

# MACD
MACD_DEFAULT_FAST_PERIOD = 12  # MACDデフォルト短期EMA
MACD_DEFAULT_SLOW_PERIOD = 26  # MACDデフォルト長期EMA
MACD_DEFAULT_SIGNAL_PERIOD = 9  # MACDデフォルトシグナル期間

# ============================================
# データ保存関連の定数
# ============================================

# ログ・レポート
DAILY_REPORT_DIR = "reports"  # 日次レポート保存ディレクトリ
LOG_DIR = "logs"  # ログファイル保存ディレクトリ
REPORT_FILE_PREFIX = "daily_"  # レポートファイル名プレフィックス

# データ保持期間
OHLCV_DATA_LIMIT = 500  # OHLCVデータ保持数
TRADE_HISTORY_LIMIT = 1000  # 取引履歴保持数

# ============================================
# 市場環境判定関連の定数
# ============================================

# BTC価格変動閾値
BTC_UPTREND_THRESHOLD = 2.0    # BTC上昇トレンド判定閾値（%）
BTC_DOWNTREND_THRESHOLD = -2.0  # BTC下降トレンド判定閾値（%）

# 市場環境
MARKET_CONDITION_UPTREND = "uptrend"
MARKET_CONDITION_SIDEWAYS = "sideways"  
MARKET_CONDITION_DOWNTREND = "downtrend"

# ============================================
# エラーハンドリング関連の定数
# ============================================

# エラーメッセージ
ERROR_API_CONNECTION = "API接続エラー"
ERROR_INSUFFICIENT_FUNDS = "資金不足エラー"
ERROR_INVALID_SYMBOL = "無効なシンボルエラー"
ERROR_ORDER_FAILED = "注文失敗エラー"

# リトライ設定
EXPONENTIAL_BACKOFF_BASE = 2  # 指数バックオフの基数
MAX_BACKOFF_SECONDS = 300  # 最大バックオフ時間（秒）

# ============================================
# 通知関連の定数
# ============================================

# Discord通知カラー
DISCORD_COLOR_SUCCESS = 0x00FF00  # 成功（緑）
DISCORD_COLOR_WARNING = 0xFFFF00  # 警告（黄）
DISCORD_COLOR_ERROR = 0xFF0000    # エラー（赤）
DISCORD_COLOR_INFO = 0x0099FF     # 情報（青）

# 通知頻度制限
DISCORD_RATE_LIMIT_SECONDS = 5  # Discord通知レート制限（秒）

# ============================================
# ファイルパス関連の定数
# ============================================

# 設定ファイル
ENV_TEMPLATE_FILE = ".env.template"
ENV_FILE = ".env"
CONFIG_FILE = "config.py"

# 出力ファイル
BOT_LOG_FILE = "bot.log"
ERROR_LOG_FILE = "error.log"