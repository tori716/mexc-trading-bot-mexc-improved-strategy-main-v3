# main.py
# Botのメイン実行ファイル

import asyncio
import logging
from config import load_config, setup_logging
from mexc_api import MEXCAPI
from strategy import TradingStrategy
from utils import DiscordNotifier

async def main():
    """
    Botのメイン実行関数。
    設定の読み込み、API接続、戦略の初期化、取引ループの開始を行います。
    """
    # ロギングのセットアップ
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Botを起動しています...")

    # 設定の読み込み
    config = load_config()
    if not config:
        logger.error("設定ファイルの読み込みに失敗しました。Botを終了します。")
        return

    # Discord通知の初期化
    notifier = DiscordNotifier(config.get("DISCORD_WEBHOOK_URL"))

    # MEXC APIの初期化
    mexc_api = MEXCAPI(
        api_key=config["MEXC_API_KEY"],
        secret_key=config["MEXC_SECRET_KEY"],
        test_mode=config["TEST_MODE"],
        notifier=notifier
    )

    # 取引戦略の初期化
    strategy = TradingStrategy(
        mexc_api=mexc_api,
        config=config,
        notifier=notifier
    )

    # Botのモードに応じて処理を開始
    if config["TEST_MODE"]:
        logger.info("テストモードでBotを起動します。")
        # テストモードのロジック（例: ヒストリカルデータでのシミュレーション、またはリアルタイムデータでのペーパートレード）
        # 現時点ではリアルタイムデータでのペーパートレードを想定
        await strategy.start_paper_trading()
    else:
        logger.info("本番モードでBotを起動します。")
        # 本番モードのロジック
        await strategy.start_live_trading()

    logger.info("Botの実行が完了しました。")

if __name__ == "__main__":
    asyncio.run(main())
