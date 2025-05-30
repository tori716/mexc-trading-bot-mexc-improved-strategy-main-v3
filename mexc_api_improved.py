# mexc_api.py
# MEXC取引所とのAPI連携を管理するモジュール

import hmac
import hashlib
import time
import json
import asyncio
import logging
import aiohttp
from urllib.parse import urlencode
from datetime import datetime

class MEXCAPI:
    """
    MEXC取引所のREST APIとの連携を処理するクラス。
    注文の発注、市場データの取得、口座情報の管理などを行います。
    """
    BASE_REST_URL = "https://api.mexc.com"

    def __init__(self, api_key: str, secret_key: str, test_mode: bool, notifier):
        """
        MEXCAPIクラスのコンストラクタ。

        Args:
            api_key (str): MEXC APIキー
            secret_key (str): MEXCシークレットキー
            test_mode (bool): テストモードかどうかのフラグ
            notifier: 通知を行うためのNotifierインスタンス
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.test_mode = test_mode
        self.notifier = notifier
        self.logger = logging.getLogger(__name__)
        self.session = None # aiohttp.ClientSession
        self.market_data = {} # リアルタイム市場データを保持 (例: {"BTCUSDT": {"price": 0, "volume": 0})
        self.ohlcv_data = {} # OHLCVデータを保持 (例: {"BTCUSDT": {"5m": [], "15m": [], "1h": []}})
        # account_info変数を削除（未使用のため）

    async def _create_session(self):
        """
        aiohttp.ClientSessionを作成または再利用します。
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def _close_session(self):
        """
        aiohttp.ClientSessionを閉じます。
        """
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def _generate_signature(self, params: dict) -> str:
        """
        MEXC APIリクエストの署名を生成します。

        Args:
            params (dict): リクエストパラメータ

        Returns:
            str: 生成された署名
        """
        # パラメータをアルファベット順にソートし、URLエンコード
        sorted_params = sorted(params.items())
        query_string = urlencode(sorted_params)
        
        # シークレットキーでHMAC SHA256署名を生成
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _send_request(self, method: str, path: str, params: dict = None, signed: bool = False):
        """
        MEXC REST APIにリクエストを送信します。

        Args:
            method (str): HTTPメソッド (GET, POST, PUT, DELETE)
            path (str): APIエンドポイントのパス
            params (dict, optional): リクエストパラメータ. Defaults to None.
            signed (bool, optional): 署名が必要かどうかのフラグ. Defaults to False.

        Returns:
            dict: APIレスポンスのJSONデータ
        """
        await self._create_session()
        url = f"{self.BASE_REST_URL}{path}"
        headers = {
            "ApiKey": self.api_key,
            "Request-Time": str(int(time.time() * 1000))
        }
        
        if params is None:
            params = {}

        if signed:
            # 署名が必要な場合、パラメータに署名を追加
            params["signature"] = self._generate_signature(params)
            headers["Signature"] = params["signature"] # MEXCはヘッダーにSignatureを要求する場合がある

        try:
            if method == "GET":
                async with self.session.get(url, params=params, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            elif method == "POST":
                async with self.session.post(url, json=params, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            # 他のメソッド（PUT, DELETE）も必要に応じて追加
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"APIリクエストエラー ({method} {path}): {e.status} - {e.message}")
            self.notifier.send_discord_message(f"APIリクエストエラー: {e.status} - {e.message} for {method} {path}")
            return None
        except aiohttp.ClientError as e:
            self.logger.error(f"ネットワークエラー ({method} {path}): {e}")
            self.notifier.send_discord_message(f"ネットワークエラー: {e} for {method} {path}")
            return None
        except Exception as e:
            self.logger.error(f"予期せぬエラー ({method} {path}): {e}")
            self.notifier.send_discord_message(f"予期せぬエラー: {e} for {method} {path}")
            return None

    async def get_server_time(self):
        """
        サーバー時刻を取得します。
        """
        path = "/api/v3/time"
        return await self._send_request("GET", path)

    async def get_exchange_info(self):
        """
        取引所の情報を取得します（シンボル、取引ルールなど）。
        """
        path = "/api/v3/exchangeInfo"
        return await self._send_request("GET", path)

    async def get_ticker_price(self, symbol: str):
        """
        指定されたシンボルの最新価格を取得します。
        """
        path = "/api/v3/ticker/price"
        params = {"symbol": symbol}
        return await self._send_request("GET", path, params=params)

    async def get_klines(self, symbol: str, interval: str, limit: int = 300):
        """
        指定されたシンボルと時間足のKライン（OHLCV）データを取得します。

        Args:
            symbol (str): 取引ペア (例: "BTCUSDT")
            interval (str): 時間足 (例: "1m", "5m", "15m", "30m", "60m", "4h", "1d", "1M")
            limit (int): 取得するKラインの数 (最大1000)

        Returns:
            list: Kラインデータのリスト
        """
        interval_map = {
            "Min5": "5m",
            "Min15": "15m",
            "Min60": "60m",
            "1h": "60m" # 念のため
        }
        mapped_interval = interval_map.get(interval, interval) # マッピングがあれば変換、なければそのまま

        path = "/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": mapped_interval,
            "limit": limit
        }
        return await self._send_request("GET", path, params=params)

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None):
        """
        注文を発注します。

        Args:
            symbol (str): 取引ペア (例: "BTCUSDT")
            side (str): 注文方向 ("BUY" or "SELL")
            order_type (str): 注文タイプ ("LIMIT", "MARKET", "TRAILING_STOP_MARKET")
            quantity (float): 注文数量
            price (float, optional): 指値注文の場合の価格. Defaults to None.

        Returns:
            dict: 注文レスポンス
        """
        path = "/api/v3/order"
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "newClientOrderId": f"bot_{int(time.time() * 1000)}" # ユニークなクライアントオーダーID
        }
        if price:
            params["price"] = price

        if self.test_mode:
            self.logger.info(f"テストモード: 注文発注シミュレーション - {symbol} {side} {order_type} Qty:{quantity} Price:{price}")
            self.notifier.send_discord_message(f"テストモード: 注文発注シミュレーション - {symbol} {side} {order_type} Qty:{quantity} Price:{price}")
            # テストモードではダミーの注文IDを返す
            return {"orderId": f"TEST_{int(time.time() * 1000)}", "status": "FILLED" if order_type == "MARKET" else "NEW"}
        else:
            return await self._send_request("POST", path, params=params, signed=True)

    async def cancel_order(self, symbol: str, order_id: str):
        """
        注文をキャンセルします。

        Args:
            symbol (str): 取引ペア
            order_id (str): キャンセルする注文のID

        Returns:
            dict: キャンセルレスポンス
        """
        path = "/api/v3/order"
        params = {
            "symbol": symbol,
            "orderId": order_id
        }
        if self.test_mode:
            self.logger.info(f"テストモード: 注文キャンセルシミュレーション - {symbol} OrderID:{order_id}")
            self.notifier.send_discord_message(f"テストモード: 注文キャンセルシミュレーション - {symbol} OrderID:{order_id}")
            return {"orderId": order_id, "status": "CANCELED"}
        else:
            return await self._send_request("DELETE", path, params=params, signed=True)

    async def get_open_orders(self, symbol: str = None):
        """
        オープンな注文リストを取得します。

        Args:
            symbol (str, optional): 特定のシンボルの注文のみ取得する場合. Defaults to None.

        Returns:
            list: オープンな注文のリスト
        """
        path = "/api/v3/openOrders"
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._send_request("GET", path, params=params, signed=True)

    async def get_account_info(self):
        """
        口座情報を取得します（残高など）。
        """
        path = "/api/v3/account"
        return await self._send_request("GET", path, signed=True)

    async def get_all_tickers(self):
        """
        全ての取引ペアのティッカー情報を取得します。
        """
        path = "/api/v3/ticker/24hr"
        return await self._send_request("GET", path)

    async def get_all_symbols(self) -> list:
        """
        MEXCで取引可能な全てのシンボル（取引ペア）を取得します。
        """
        exchange_info = await self.get_exchange_info()
        if exchange_info and "symbols" in exchange_info:
            return [s["symbol"] for s in exchange_info["symbols"]]
        return []

    async def get_current_price(self, symbol: str) -> float:
        """
        REST APIから最新の価格を返します。
        価格取得時にコンソールとログファイルに価格情報を表示します。
        """
        ticker = await self.get_ticker_price(symbol)
        if ticker and ticker.get("price"):
            price = float(ticker["price"])
            
            # 価格表示機能を追加
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            price_message = f"[{timestamp}] 価格取得: {symbol} = {price:,.2f} USDT"
            
            # INFOレベルでログ出力とコンソール表示
            self.logger.info(price_message)
            
            return price
        return 0.0

    async def get_ohlcv(self, symbol: str, interval: str, limit: int = 100) -> list:
        """
        REST APIからOHLCVデータを返します。
        """
        klines = await self.get_klines(symbol, interval, limit)
        if klines:
            parsed_klines = []
            for kline in klines:
                parsed_klines.append({
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                    "timestamp": int(kline[0])
                })
            return parsed_klines
        return []

    async def start_rest_api_only_mode(self, symbols: list):
        """
        REST APIのみで市場データをポーリングします。
        """
        self.logger.warning("REST APIのみで運用します") # WebSocketが使用できないという文言は削除
        self.notifier.send_discord_message("情報: REST APIのみで運用します")
        
        # 監視銘柄を大幅に削減（API制限対応）
        limited_symbols = symbols[:4]  # 4銘柄まで制限
        self.logger.info(f"REST APIモード: 監視銘柄を{len(limited_symbols)}個に制限: {limited_symbols}")
        
        while True:
            try:
                for symbol in limited_symbols:
                    try:
                        # 価格データ取得
                        ticker = await self.get_ticker_price(symbol)
                        if ticker and ticker.get("price"):
                            self.market_data[symbol] = {
                                "price": float(ticker["price"]),
                                "volume_24h": 0,
                                "timestamp": time.time()
                            }
                            self.logger.debug(f"REST API価格更新: {symbol} = {ticker['price']}")
                        
                        # OHLCVデータ取得
                        for interval in ["Min5", "Min15", "Min60"]:
                            klines = await self.get_klines(symbol, interval, 50)
                            if klines:
                                if symbol not in self.ohlcv_data:
                                    self.ohlcv_data[symbol] = {}
                                
                                self.ohlcv_data[symbol][interval] = []
                                for kline in klines:
                                    self.ohlcv_data[symbol][interval].append({
                                        "open": float(kline[1]),
                                        "high": float(kline[2]),
                                        "low": float(kline[3]),
                                        "close": float(kline[4]),
                                        "volume": float(kline[5]),
                                        "timestamp": int(kline[0])
                                    })
                        
                        # API制限対応の待機
                        await asyncio.sleep(5)  # 銘柄間で5秒待機 (100回/秒制限を考慮)
                        
                    except Exception as e:
                        self.logger.error(f"REST API データ取得エラー ({symbol}): {e}")
                
                # 次回更新まで待機
                self.logger.info(f"取得データ: {len(self.market_data)}銘柄")
                await asyncio.sleep(10)  # 10秒間隔で更新
                
            except Exception as e:
                self.logger.error(f"REST APIモード中のエラー: {e}")
                await asyncio.sleep(120)

    async def verify_api_permissions(self):
        """
        APIキーの権限とアクセス可能性を確認
        """
        try:
            # サーバー時刻の確認
            server_time = await self.get_server_time()
            if not server_time:
                self.logger.error("サーバー時刻の取得に失敗。API接続に問題があります。")
                return False
            
            # 取引所情報の取得（権限チェック）
            exchange_info = await self.get_exchange_info()
            if not exchange_info:
                self.logger.error("取引所情報の取得に失敗。APIキーの権限に問題があります。")
                return False
            
            # アカウント情報の取得（署名付きAPI権限チェック）
            if not self.test_mode:
                account_info = await self.get_account_info()
                if not account_info:
                    self.logger.error("アカウント情報の取得に失敗。APIキーの署名権限に問題があります。")
                    return False
            
            self.logger.info("APIキーの権限確認完了")
            return True
            
        except Exception as e:
            self.logger.error(f"API権限確認中にエラー: {e}")
            return False

    async def check_websocket_data_status(self):
        """
        WebSocketデータの受信状況を確認するデバッグ機能
        (REST APIモードではこの関数は実質的に市場データの確認のみ)
        """
        self.logger.info("=== 市場データ状況確認 ===")
        self.logger.info(f"市場データ数: {len(self.market_data)}銘柄")
        self.logger.info(f"OHLCVデータ数: {len(self.ohlcv_data)}銘柄")
        
        for symbol in self.market_data:
            price_data = self.market_data[symbol]
            age = time.time() - price_data.get("timestamp", 0)
            self.logger.info(f"{symbol}: 価格={price_data['price']}, 更新から{age:.1f}秒経過")
        
        for symbol in self.ohlcv_data:
            for interval, data_list in self.ohlcv_data[symbol].items():
                self.logger.info(f"{symbol} {interval}: {len(data_list)}本のデータ")
        
        self.logger.info("=== 確認終了 ===")

    async def close_connections(self):
        """
        全ての接続を閉じます。
        """
        self.logger.info("MEXC API接続を閉じています...")
        await self._close_session()
        self.logger.info("MEXC API接続を閉じました。")