# utils.py
# ユーティリティ関数と通知機能を管理するモジュール

import logging
import requests
import json
import time
from typing import Optional
import asyncio

class DiscordNotifier:
    """
    Discord Webhookを使用して通知を送信するクラス。
    """
    def __init__(self, webhook_url: str):
        """
        DiscordNotifierクラスのコンストラクタ。

        Args:
            webhook_url (str): Discord WebhookのURL
        """
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)
        self.last_notification_time = {}  # 通知頻度制限用
        self.rate_limit_seconds = 5  # 同一メッセージの制限時間

    def send_discord_message(self, message: str, embed_title: str = None, color: int = None):
        """
        Discordにメッセージを送信します。

        Args:
            message (str): 送信するメッセージ本文
            embed_title (str, optional): Embedのタイトル. Defaults to None.
            color (int, optional): Embedのサイドバーの色 (HEXコードの整数表現). Defaults to None.
        """
        if not self.webhook_url:
            self.logger.warning("Discord Webhook URLが設定されていません。通知をスキップします。")
            return

        # 通知頻度制限チェック
        if self._is_rate_limited(message):
            self.logger.debug(f"通知頻度制限により送信をスキップ: {message[:50]}...")
            return

        headers = {"Content-Type": "application/json"}
        
        if embed_title:
            embed = {
                "title": embed_title,
                "description": message,
                "color": color if color is not None else 3447003,  # デフォルトは青
                "timestamp": self._get_iso_timestamp()
            }
            payload = {
                "embeds": [embed]
            }
        else:
            payload = {
                "content": message
            }

        try:
            response = requests.post(self.webhook_url, data=json.dumps(payload), headers=headers, timeout=10)
            response.raise_for_status() # HTTPエラーがあれば例外を発生させる
            self.logger.info(f"Discordにメッセージを送信しました: {message[:100]}...")
            
            # 成功時に最後の送信時刻を記録
            self._update_last_notification_time(message)
            
        except requests.exceptions.Timeout:
            self.logger.error("Discord通知のタイムアウト")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Discord通知の送信に失敗しました: {e}")
        except Exception as e:
            self.logger.error(f"Discord通知で予期せぬエラーが発生しました: {e}")

    def send_price_alert(self, symbol: str, price: float, change_percent: float):
        """
        価格急変アラートを送信します。

        Args:
            symbol (str): 銘柄シンボル
            price (float): 現在価格
            change_percent (float): 変動率（%）
        """
        if abs(change_percent) >= 5.0:  # 5%以上の変動で緊急アラート
            emoji = "🚨" if change_percent > 0 else "⚠️"
            direction = "急騰" if change_percent > 0 else "急落"
            color = 0xFF4500 if change_percent > 0 else 0xFF0000
            
            message = f"{emoji} **{symbol} {direction}アラート**\n現在価格: {price:,.2f} USDT\n変動率: {change_percent:+.2f}%"
            
            self.send_discord_message(
                message=message,
                embed_title=f"{symbol} 価格急変アラート",
                color=color
            )

    def send_trade_notification(self, trade_type: str, symbol: str, side: str, quantity: float, price: float, reason: str = ""):
        """
        取引通知を送信します。

        Args:
            trade_type (str): 取引タイプ（"ENTRY", "EXIT", "STOP_LOSS", "TAKE_PROFIT"）
            symbol (str): 銘柄シンボル
            side (str): 売買方向（"BUY", "SELL"）
            quantity (float): 数量
            price (float): 価格
            reason (str): 取引理由
        """
        emoji_map = {
            "ENTRY": "📈" if side == "BUY" else "📉",
            "EXIT": "🔄",
            "STOP_LOSS": "🛑",
            "TAKE_PROFIT": "💰"
        }
        
        color_map = {
            "ENTRY": 0x0099FF,
            "EXIT": 0x808080,
            "STOP_LOSS": 0xFF0000,
            "TAKE_PROFIT": 0x00FF00
        }
        
        emoji = emoji_map.get(trade_type, "📊")
        color = color_map.get(trade_type, 0x0099FF)
        
        side_text = "ロング" if side == "BUY" else "ショート"
        
        message = f"{emoji} **{trade_type} - {symbol}**\n"
        message += f"方向: {side_text}\n"
        message += f"数量: {quantity:.4f}\n"
        message += f"価格: {price:,.2f} USDT"
        
        if reason:
            message += f"\n理由: {reason}"
        
        self.send_discord_message(
            message=message,
            embed_title=f"{trade_type} 通知",
            color=color
        )

    def send_error_notification(self, error_type: str, error_message: str, severity: str = "ERROR"):
        """
        エラー通知を送信します。

        Args:
            error_type (str): エラータイプ
            error_message (str): エラーメッセージ
            severity (str): 重要度（"ERROR", "WARNING", "CRITICAL"）
        """
        emoji_map = {
            "CRITICAL": "🔥",
            "ERROR": "❌",
            "WARNING": "⚠️"
        }
        
        color_map = {
            "CRITICAL": 0x800000,
            "ERROR": 0xFF0000,
            "WARNING": 0xFFFF00
        }
        
        emoji = emoji_map.get(severity, "❌")
        color = color_map.get(severity, 0xFF0000)
        
        message = f"{emoji} **{severity}: {error_type}**\n{error_message}"
        
        self.send_discord_message(
            message=message,
            embed_title=f"システム{severity}",
            color=color
        )

    def send_daily_summary(self, summary_data: dict):
        """
        日次サマリーを送信します。

        Args:
            summary_data (dict): サマリーデータ
        """
        profit_loss = summary_data.get('profit_loss', 0)
        trades_count = summary_data.get('trades_count', 0)
        win_rate = summary_data.get('win_rate', 0)
        
        emoji = "📈" if profit_loss >= 0 else "📉"
        color = 0x00FF00 if profit_loss >= 0 else 0xFF0000
        
        message = f"{emoji} **日次取引サマリー**\n"
        message += f"純損益: {profit_loss:+.2f} USDT\n"
        message += f"取引回数: {trades_count}回\n"
        message += f"勝率: {win_rate:.1f}%"
        
        self.send_discord_message(
            message=message,
            embed_title="日次サマリー",
            color=color
        )

    def send_system_status(self, status: str, details: str = ""):
        """
        システム状態通知を送信します。

        Args:
            status (str): ステータス（"STARTED", "STOPPED", "RESTARTED", "MAINTENANCE"）
            details (str): 詳細情報
        """
        emoji_map = {
            "STARTED": "🚀",
            "STOPPED": "⏹️",
            "RESTARTED": "🔄",
            "MAINTENANCE": "🔧"
        }
        
        color_map = {
            "STARTED": 0x00FF00,
            "STOPPED": 0xFF0000,
            "RESTARTED": 0xFFFF00,
            "MAINTENANCE": 0x0099FF
        }
        
        emoji = emoji_map.get(status, "📊")
        color = color_map.get(status, 0x0099FF)
        
        message = f"{emoji} **Bot {status}**"
        if details:
            message += f"\n{details}"
        
        self.send_discord_message(
            message=message,
            embed_title="システム状態",
            color=color
        )

    def _is_rate_limited(self, message: str) -> bool:
        """
        通知頻度制限をチェックします。

        Args:
            message (str): メッセージ

        Returns:
            bool: 制限されている場合True
        """
        # メッセージの最初の50文字をキーとして使用
        message_key = message[:50]
        current_time = time.time()
        
        if message_key in self.last_notification_time:
            elapsed = current_time - self.last_notification_time[message_key]
            return elapsed < self.rate_limit_seconds
        
        return False

    def _update_last_notification_time(self, message: str):
        """
        最後の通知時刻を更新します。

        Args:
            message (str): メッセージ
        """
        message_key = message[:50]
        self.last_notification_time[message_key] = time.time()
        
        # 古いエントリを削除（メモリリーク防止）
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.last_notification_time.items()
            if current_time - timestamp > 3600  # 1時間経過したエントリを削除
        ]
        for key in expired_keys:
            del self.last_notification_time[key]

    def _get_iso_timestamp(self) -> str:
        """
        ISO形式のタイムスタンプを取得します。

        Returns:
            str: ISO形式のタイムスタンプ
        """
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"

    async def send_async_notification(self, message: str, embed_title: str = None, color: int = None):
        """
        非同期でDiscord通知を送信します。

        Args:
            message (str): 送信するメッセージ本文
            embed_title (str, optional): Embedのタイトル
            color (int, optional): Embedのサイドバーの色
        """
        # 非同期で実行するために別スレッドで実行
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            self.send_discord_message,
            message, embed_title, color
        )

# ヘルパー関数

def format_number(value: float, decimals: int = 2) -> str:
    """
    数値を3桁区切りでフォーマットします。

    Args:
        value (float): フォーマットする数値
        decimals (int): 小数点以下の桁数

    Returns:
        str: フォーマットされた数値文字列
    """
    return f"{value:,.{decimals}f}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    変動率を計算します。

    Args:
        old_value (float): 旧値
        new_value (float): 新値

    Returns:
        float: 変動率（%）
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def validate_config_parameters(config: dict) -> list:
    """
    設定パラメータの妥当性をチェックします。

    Args:
        config (dict): 設定辞書

    Returns:
        list: エラーメッセージのリスト（空の場合は正常）
    """
    errors = []
    
    # 必須パラメータのチェック
    required_params = [
        "POSITION_SIZE_PERCENTAGE",
        "MAX_SIMULTANEOUS_POSITIONS",
        "FIXED_STOP_LOSS_PERCENTAGE",
        "TAKE_PROFIT_PERCENTAGE_PHASE1"
    ]
    
    for param in required_params:
        if param not in config:
            errors.append(f"必須パラメータが不足: {param}")
    
    # 数値範囲のチェック
    if config.get("POSITION_SIZE_PERCENTAGE", 0) <= 0 or config.get("POSITION_SIZE_PERCENTAGE", 0) > 100:
        errors.append("POSITION_SIZE_PERCENTAGE は0-100の範囲で設定してください")
    
    if config.get("MAX_SIMULTANEOUS_POSITIONS", 0) <= 0:
        errors.append("MAX_SIMULTANEOUS_POSITIONS は正の整数で設定してください")
    
    if config.get("FIXED_STOP_LOSS_PERCENTAGE", 0) <= 0:
        errors.append("FIXED_STOP_LOSS_PERCENTAGE は正の値で設定してください")
    
    return errors

def safe_float_conversion(value, default: float = 0.0) -> float:
    """
    安全なfloat型変換を行います。

    Args:
        value: 変換する値
        default (float): 変換失敗時のデフォルト値

    Returns:
        float: 変換された値
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def truncate_string(text: str, max_length: int = 100) -> str:
    """
    文字列を指定長で切り詰めます。

    Args:
        text (str): 対象文字列
        max_length (int): 最大長

    Returns:
        str: 切り詰められた文字列
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

# ロギングはconfig.pyで設定されるため、ここでは特別なロギング関数は不要です。
# テクニカル指標の計算はstrategy.pyでpandasとtaライブラリを使用して直接行われます。