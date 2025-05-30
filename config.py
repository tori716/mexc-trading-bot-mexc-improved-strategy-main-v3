# config.py
# Botの設定を管理するモジュール

import os
import logging
from dotenv import load_dotenv

def load_config():
    """
    .envファイルから環境変数を読み込み、Botの設定を辞書として返します。
    APIキー、シークレット、Discord Webhook URL、モード選択などを管理します。
    """
    load_dotenv() # .env ファイルから環境変数を読み込む

    config = {
        # MEXC API認証情報
        "MEXC_API_KEY": os.getenv("MEXC_API_KEY"),
        "MEXC_SECRET_KEY": os.getenv("MEXC_SECRET_KEY"),
        "DISCORD_WEBHOOK_URL": os.getenv("DISCORD_WEBHOOK_URL"),

        # Botの動作モード (True: テストモード, False: 本番モード)
        "TEST_MODE": os.getenv("TEST_MODE", "True").lower() == "true",

        # 取引パラメータ
        "INITIAL_CAPITAL_USD": float(os.getenv("INITIAL_CAPITAL_USD", "1000")),
        
        # 重複設定項目を削除
        # TARGET_CRYPTOCURRENCIES_EXCLUDED と TARGET_CRYPTOCURRENCIES_PRIORITIZED は削除
        
        # 銘柄制限設定
        "TARGET_SYMBOLS_TIER1": ["AVAX", "LINK", "NEAR", "FTM", "ATOM", "DOT", "MATIC", "UNI", "AAVE", "DOGE"],
        "TARGET_SYMBOLS_TIER2": ["ADA", "ALGO", "APE", "ARB", "EGLD", "FIL", "GRT", "ICP", "LTC", "SAND"],
        "TARGET_SYMBOLS_TIER3": ["SHIB", "VET", "MANA", "GALA", "ONE"],
        "MAX_MONITORING_SYMBOLS": int(os.getenv("MAX_MONITORING_SYMBOLS", "25")),  # 環境変数から取得
        "MARKET_CONDITION_ROTATION": True,  # 市場環境に応じたローテーション有効化
        "POSITION_SIZE_PERCENTAGE": float(os.getenv("POSITION_SIZE_PERCENTAGE", "5.0")), # 1取引あたりのリスク許容額: 総資金の最大5%
        "MAX_SIMULTANEOUS_POSITIONS": int(os.getenv("MAX_SIMULTANEOUS_POSITIONS", "2")), # 同時保有ポジション上限: 最大2つまで
        "MAX_CONSECUTIVE_ENTRIES_SAME_COIN": int(os.getenv("MAX_CONSECUTIVE_ENTRIES_SAME_COIN", "2")), # 同一銘柄への連続エントリー制限: 24時間以内に最大2回まで

        # 戦略パラメータ
        "BB_PERIOD": int(os.getenv("BB_PERIOD", "20")),
        "BB_STD_DEV": float(os.getenv("BB_STD_DEV", "2.0")),
        "RSI_PERIOD": int(os.getenv("RSI_PERIOD", "14")),
        "MACD_FAST_EMA_PERIOD": int(os.getenv("MACD_FAST_EMA_PERIOD", "12")),
        "MACD_SLOW_EMA_PERIOD": int(os.getenv("MACD_SLOW_EMA_PERIOD", "26")),
        "MACD_SIGNAL_SMA_PERIOD": int(os.getenv("MACD_SIGNAL_SMA_PERIOD", "9")),
        "EMA_PERIODS": [int(p) for p in os.getenv("EMA_PERIODS", "20,50").split(',')],

        # 利益確定・損切りパラメータ
        "TAKE_PROFIT_PERCENTAGE_PHASE1": float(os.getenv("TAKE_PROFIT_PERCENTAGE_PHASE1", "2.4")),
        "TAKE_PROFIT_PERCENTAGE_PHASE2_OPTION1": float(os.getenv("TAKE_PROFIT_PERCENTAGE_PHASE2_OPTION1", "5.0")),
        "TAKE_PROFIT_PERCENTAGE_PHASE2_OPTION3": float(os.getenv("TAKE_PROFIT_PERCENTAGE_PHASE2_OPTION3", "3.0")),
        "TRAILING_STOP_ACTIVATION_PERCENTAGE": float(os.getenv("TRAILING_STOP_ACTIVATION_PERCENTAGE", "2.4")),
        "TRAILING_STOP_PERCENTAGE": float(os.getenv("TRAILING_STOP_PERCENTAGE", "1.5")),
        "FIXED_STOP_LOSS_PERCENTAGE": float(os.getenv("FIXED_STOP_LOSS_PERCENTAGE", "1.5")),
        "TIME_BASED_STOP_LOSS_MINUTES": int(os.getenv("TIME_BASED_STOP_LOSS_MINUTES", "30")),
        "TIME_BASED_STOP_LOSS_PROFIT_THRESHOLD": float(os.getenv("TIME_BASED_STOP_LOSS_PROFIT_THRESHOLD", "1.0")),

        # 改善戦略パラメータ
        "SHORT_RSI_THRESHOLD_BEAR_MARKET": float(os.getenv("SHORT_RSI_THRESHOLD_BEAR_MARKET", "35")),
        "PRICE_CHANGE_RATE_OPTIMIZED_HOURS_LONG": float(os.getenv("PRICE_CHANGE_RATE_OPTIMIZED_HOURS_LONG", "1.5")),
        "PRICE_CHANGE_RATE_OPTIMIZED_HOURS_SHORT": float(os.getenv("PRICE_CHANGE_RATE_OPTIMIZED_HOURS_SHORT", "1.3")),
        "PRICE_CHANGE_RATE_OTHER_HOURS": float(os.getenv("PRICE_CHANGE_RATE_OTHER_HOURS", "1.8")),
        "RSI_THRESHOLD_LONG_OPTIMIZED_HOURS": float(os.getenv("RSI_THRESHOLD_LONG_OPTIMIZED_HOURS", "60")),
        "RSI_THRESHOLD_LONG_OTHER_HOURS": float(os.getenv("RSI_THRESHOLD_LONG_OTHER_HOURS", "65")),
        "RSI_THRESHOLD_SHORT_OPTIMIZED_HOURS": float(os.getenv("RSI_THRESHOLD_SHORT_OPTIMIZED_HOURS", "35")),
        "RSI_THRESHOLD_SHORT_OTHER_HOURS": float(os.getenv("RSI_THRESHOLD_SHORT_OTHER_HOURS", "35")),
        "VOLUME_MULTIPLIER_OPTIMIZED_HOURS": float(os.getenv("VOLUME_MULTIPLIER_OPTIMIZED_HOURS", "2.0")),
        "VOLUME_MULTIPLIER_OTHER_HOURS": float(os.getenv("VOLUME_MULTIPLIER_OTHER_HOURS", "3.0")),
        "SCALPING_LONG_RSI_THRESHOLD": float(os.getenv("SCALPING_LONG_RSI_THRESHOLD", "30")),
        "SCALPING_LONG_PRICE_DROP_PERCENTAGE": float(os.getenv("SCALPING_LONG_PRICE_DROP_PERCENTAGE", "4.0")),
        "BTC_CRASH_STOP_LOSS_PERCENTAGE": float(os.getenv("BTC_CRASH_STOP_LOSS_PERCENTAGE", "3.0")),
        "SLIPPAGE_PERCENTAGE": float(os.getenv("SLIPPAGE_PERCENTAGE", "0.2")),

        # 価格表示設定（新機能）
        "PRICE_DISPLAY_ENABLED": os.getenv("PRICE_DISPLAY_ENABLED", "True").lower() == "true",
        "PRICE_DISPLAY_INTERVAL_SECONDS": int(os.getenv("PRICE_DISPLAY_INTERVAL_SECONDS", "10")),

        # レポート設定（新機能）
        "DAILY_REPORT_ENABLED": os.getenv("DAILY_REPORT_ENABLED", "True").lower() == "true",
        "DAILY_REPORT_TIME_JST": os.getenv("DAILY_REPORT_TIME_JST", "00:00"),
        "DAILY_REPORT_DISCORD_ENABLED": os.getenv("DAILY_REPORT_DISCORD_ENABLED", "True").lower() == "true",

        # パフォーマンス設定（新機能）
        "API_CACHE_DURATION_SECONDS": int(os.getenv("API_CACHE_DURATION_SECONDS", "5")),
        "MAX_CONCURRENT_API_CALLS": int(os.getenv("MAX_CONCURRENT_API_CALLS", "3")),

        # 銘柄別調整 (例)
        "COIN_SPECIFIC_ADJUSTMENTS": {
            "AVAX": {"BB_STD_DEV": 2.2},
            "LINK": {"RSI_THRESHOLD_LONG_OPTIMIZED_HOURS": 58},
            "NEAR": {"MACD_FAST_EMA_PERIOD": 10},
            "FTM": {"PRICE_CHANGE_RATE_OPTIMIZED_HOURS_LONG": 1.4},
            "ATOM": {"VOLUME_MULTIPLIER_OPTIMIZED_HOURS": 1.8}
        },

        # 最適取引時間帯 (JST)
        "OPTIMIZED_TRADING_HOURS": [
            {"start": 15, "end": 18}, # 15:00-18:00 JST
            {"start": 22, "end": 24}  # 22:00-24:00 JST
        ]
    }

    # 必須環境変数のチェック
    # テストモードの場合はAPIキーとシークレットは必須ではない
    required_vars = ["DISCORD_WEBHOOK_URL"]
    if not config["TEST_MODE"]:
        required_vars.extend(["MEXC_API_KEY", "MEXC_SECRET_KEY"])

    for var in required_vars:
        if not config.get(var):
            logging.error(f"環境変数 {var} が設定されていません。Botを終了します。")
            return None
    
    # 数値型に変換されるべき環境変数のチェック
    numeric_vars = {
        "INITIAL_CAPITAL_USD": float,
        "POSITION_SIZE_PERCENTAGE": float,
        "MAX_SIMULTANEOUS_POSITIONS": int,
        "MAX_CONSECUTIVE_ENTRIES_SAME_COIN": int,
        "BB_PERIOD": int,
        "BB_STD_DEV": float,
        "RSI_PERIOD": int,
        "MACD_FAST_EMA_PERIOD": int,
        "MACD_SLOW_EMA_PERIOD": int,
        "MACD_SIGNAL_SMA_PERIOD": int,
        "TAKE_PROFIT_PERCENTAGE_PHASE1": float,
        "TAKE_PROFIT_PERCENTAGE_PHASE2_OPTION1": float,
        "TAKE_PROFIT_PERCENTAGE_PHASE2_OPTION3": float,
        "TRAILING_STOP_ACTIVATION_PERCENTAGE": float,
        "TRAILING_STOP_PERCENTAGE": float,
        "FIXED_STOP_LOSS_PERCENTAGE": float,
        "TIME_BASED_STOP_LOSS_MINUTES": int,
        "TIME_BASED_STOP_LOSS_PROFIT_THRESHOLD": float,
        "SHORT_RSI_THRESHOLD_BEAR_MARKET": float,
        "PRICE_CHANGE_RATE_OPTIMIZED_HOURS_LONG": float,
        "PRICE_CHANGE_RATE_OPTIMIZED_HOURS_SHORT": float,
        "PRICE_CHANGE_RATE_OTHER_HOURS": float,
        "RSI_THRESHOLD_LONG_OPTIMIZED_HOURS": float,
        "RSI_THRESHOLD_LONG_OTHER_HOURS": float,
        "RSI_THRESHOLD_SHORT_OPTIMIZED_HOURS": float,
        "RSI_THRESHOLD_SHORT_OTHER_HOURS": float,
        "VOLUME_MULTIPLIER_OPTIMIZED_HOURS": float,
        "VOLUME_MULTIPLIER_OTHER_HOURS": float,
        "SCALPING_LONG_RSI_THRESHOLD": float,
        "SCALPING_LONG_PRICE_DROP_PERCENTAGE": float,
        "BTC_CRASH_STOP_LOSS_PERCENTAGE": float,
        "SLIPPAGE_PERCENTAGE": float,
        "PRICE_DISPLAY_INTERVAL_SECONDS": int,
        "API_CACHE_DURATION_SECONDS": int,
        "MAX_CONCURRENT_API_CALLS": int,
        "MAX_MONITORING_SYMBOLS": int,
    }

    for var, type_func in numeric_vars.items():
        env_val = os.getenv(var)
        if env_val is not None:
            try:
                config[var] = type_func(env_val)
            except ValueError:
                logging.error(f"環境変数 {var} の値 '{env_val}' が不正な形式です。数値に変換できません。")
                return None

    return config

def setup_logging():
    """
    Botのロギングを設定します。
    コンソール出力とファイル出力の両方を設定します。
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "bot.log")),
            logging.StreamHandler()
        ]
    )
    # 外部ライブラリのログレベルを調整（必要に応じて）
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)