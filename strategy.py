# strategy.py
# 取引戦略のロジックを実装するモジュール

import asyncio
import logging
import pandas as pd
import ta # Technical Analysis Library
from datetime import datetime, time as dt_time
import constants

class TradingStrategy:
    """
    MEXC取引所での自動取引戦略を実装するクラス。
    エントリー、イグジット、リスク管理の各ロジックを含みます。
    """
    def __init__(self, mexc_api, config, notifier):
        """
        TradingStrategyクラスのコンストラクタ。

        Args:
            mexc_api: MEXCAPIのインスタンス
            config (dict): Botの設定
            notifier: 通知を行うためのNotifierインスタンス
        """
        self.mexc_api = mexc_api
        self.config = config
        self.notifier = notifier
        self.logger = logging.getLogger(__name__)
        self.current_positions = {} # 現在保有しているポジションを管理
        self.trade_history = [] # 取引履歴を記録
        self.initial_capital = config["INITIAL_CAPITAL_USD"]
        self.current_capital = self.initial_capital
        self.consecutive_losses = 0 # 連続損失回数

        # 銘柄別調整を適用するための辞書を初期化
        self.coin_specific_params = self._initialize_coin_specific_params()
        self.current_symbols = [] # 現在監視中のシンボルリスト
        
        # 銘柄別損失履歴の追跡
        self.symbol_loss_history = {} # {symbol: [{"time": datetime, "type": "loss"}, ...]}
        
        # 改善された資金管理のための変数
        self.winning_streak = 0  # 連続勝利回数
        self.daily_profit_percentage = 0.0  # 日次利益率
        self.portfolio_risk_usage = 0.0  # 現在のポートフォリオリスク使用率

    def _calculate_trading_fee(self, order_type: str, trade_amount: float, is_maker: bool = None) -> float:
        """
        Ultra-Think手数料計算: MEXCの実際の手数料体系に基づく正確な計算
        
        Args:
            order_type (str): 注文タイプ ("LIMIT" or "MARKET")
            trade_amount (float): 取引金額
            is_maker (bool): 指値注文時のメイカー/テイカー判定（Noneの場合は保守的にテイカーと仮定）
            
        Returns:
            float: 手数料金額
        """
        if order_type == "MARKET":
            # 成行注文は常にテイカー
            return trade_amount * constants.MEXC_TAKER_FEE_RATE
        elif order_type == "LIMIT":
            if is_maker is True:
                # メイカーとして約定
                return trade_amount * constants.MEXC_MAKER_FEE_RATE  # 0%
            else:
                # テイカーとして約定（保守的な見積もり）
                return trade_amount * constants.MEXC_TAKER_FEE_RATE
        else:
            # 不明な注文タイプは保守的にテイカー手数料を適用
            return trade_amount * constants.MEXC_TAKER_FEE_RATE
    
    def _estimate_round_trip_fee(self, trade_amount: float) -> float:
        """
        往復手数料の見積もり（エントリー＋決済）
        
        Args:
            trade_amount (float): 取引金額
            
        Returns:
            float: 往復手数料の見積もり
        """
        # エントリー: 指値注文（保守的にテイカーと仮定）
        entry_fee = self._calculate_trading_fee("LIMIT", trade_amount, is_maker=False)
        
        # 決済: 成行注文（常にテイカー）
        exit_fee = self._calculate_trading_fee("MARKET", trade_amount)
        
        return entry_fee + exit_fee

    def _initialize_coin_specific_params(self):
        """
        設定ファイルから銘柄別調整パラメータを初期化します。
        """
        params = {}
        for coin, adjustments in self.config["COIN_SPECIFIC_ADJUSTMENTS"].items():
            params[coin] = {}
            for key, value in adjustments.items():
                params[coin][key] = value
        return params

    async def _select_active_symbols(self):
        """
        市場環境に基づいて監視対象銘柄を動的選択（最大25個）
        """
        # Tier 1は常時監視
        active_symbols = [s + "USDT" for s in self.config["TARGET_SYMBOLS_TIER1"]]
        
        # 市場環境の判定
        market_condition = await self._analyze_market_condition()
        
        if market_condition == "uptrend":
            # 上昇相場: Tier 1 + Tier 2前半 + Tier 3前半 = 18銘柄
            active_symbols.extend([s + "USDT" for s in self.config["TARGET_SYMBOLS_TIER2"][:5]])
            active_symbols.extend([s + "USDT" for s in self.config["TARGET_SYMBOLS_TIER3"][:3]])
        elif market_condition == "sideways":
            # 横ばい相場: Tier 1 + Tier 2全部 = 20銘柄
            active_symbols.extend([s + "USDT" for s in self.config["TARGET_SYMBOLS_TIER2"]])
        else:  # downtrend
            # 下落相場: Tier 1 + Tier 2後半 + Tier 3後半 = 17銘柄
            active_symbols.extend([s + "USDT" for s in self.config["TARGET_SYMBOLS_TIER2"][5:]])
            active_symbols.extend([s + "USDT" for s in self.config["TARGET_SYMBOLS_TIER3"][3:]])
        
        # 実際に取引可能なシンボルのみフィルタリング
        all_symbols = await self.mexc_api.get_all_symbols()
        filtered_symbols = [s for s in active_symbols if s in all_symbols]
        
        # 最大25個に制限
        return filtered_symbols[:self.config["MAX_MONITORING_SYMBOLS"]]

    async def _analyze_market_condition(self):
        """
        複数時間軸とボラティリティを考慮した市場環境判定
        """
        try:
            if not self.config["DYNAMIC_THRESHOLD_ENABLED"]:
                # 従来のロジック
                btc_ohlcv = await self.mexc_api.get_ohlcv("BTCUSDT", "Min15", 4)
                if len(btc_ohlcv) >= 4:
                    recent_change = (btc_ohlcv[-1]["close"] - btc_ohlcv[-4]["close"]) / btc_ohlcv[-4]["close"] * 100
                    
                    if recent_change > 2.0:
                        return "uptrend"
                    elif recent_change < -2.0:
                        return "downtrend"
                
                return "sideways"
            
            # 改善されたロジック
            # 複数時間軸のBTCデータ取得
            btc_15m = await self.mexc_api.get_ohlcv("BTCUSDT", "Min15", 8)  # 2時間
            btc_1h = await self.mexc_api.get_ohlcv("BTCUSDT", "Min60", 12)  # 12時間
            btc_4h = await self.mexc_api.get_ohlcv("BTCUSDT", "Min240", 6)  # 24時間
            
            if not all([btc_15m, btc_1h, btc_4h]):
                return "sideways"
            
            # 動的閾値計算（過去24時間のボラティリティベース）
            recent_volatility = self._calculate_volatility(btc_1h)
            dynamic_threshold = max(
                self.config["MIN_MARKET_THRESHOLD"], 
                min(self.config["MAX_MARKET_THRESHOLD"], recent_volatility * self.config["VOLATILITY_MULTIPLIER"])
            )
            
            # 複数時間軸での判定
            change_15m = self._calculate_price_change(btc_15m, 4)  # 1時間
            change_1h = self._calculate_price_change(btc_1h, 6)    # 6時間  
            change_4h = self._calculate_price_change(btc_4h, 6)    # 24時間
            
            # 重み付け判定
            weighted_score = (
                change_15m * 0.4 +    # 短期重視
                change_1h * 0.4 +     # 中期
                change_4h * 0.2       # 長期
            )
            
            if weighted_score > dynamic_threshold:
                return "uptrend"
            elif weighted_score < -dynamic_threshold:
                return "downtrend"
            else:
                return "sideways"
                
        except Exception as e:
            self.logger.warning(f"市場環境判定でエラー: {e}")
            return "sideways"  # デフォルト

    def _calculate_volatility(self, ohlcv_data):
        """
        OHLCVデータからボラティリティを計算
        """
        if len(ohlcv_data) < 2:
            return 2.0  # デフォルト値
        
        prices = [candle["close"] for candle in ohlcv_data]
        price_changes = []
        for i in range(1, len(prices)):
            change = abs((prices[i] - prices[i-1]) / prices[i-1] * 100)
            price_changes.append(change)
        
        return sum(price_changes) / len(price_changes) if price_changes else 2.0
    
    def _calculate_price_change(self, ohlcv_data, periods):
        """
        指定期間の価格変動率を計算
        """
        if len(ohlcv_data) < periods:
            return 0.0
        
        start_price = ohlcv_data[-periods]["close"]
        end_price = ohlcv_data[-1]["close"]
        return (end_price - start_price) / start_price * 100

    def get_adjusted_param(self, coin: str, param_name: str, default_value):
        """
        銘柄別の調整パラメータを取得します。存在しない場合はデフォルト値を返します。
        """
        return self.coin_specific_params.get(coin, {}).get(param_name, default_value)
    
    def _is_optimized_trading_hours(self) -> bool:
        """
        現在時刻が最適化された取引時間帯内にあるかをチェックします。
        
        Returns:
            bool: 最適化された取引時間帯内の場合True
        """
        from datetime import datetime
        current_hour_jst = datetime.now().hour
        
        # 最適化された取引時間帯のチェック（JST）
        # 15:00-18:00 JST (昼間の時間帯)
        # 22:00-24:00 JST (夜間の時間帯)
        for period in self.config["OPTIMIZED_TRADING_HOURS"]:
            if period["start"] <= current_hour_jst < period["end"]:
                return True
        return False
    
    def _calculate_dynamic_position_size(self, symbol: str, side: str) -> float:
        """
        改善された動的ポジションサイジングを計算します。
        
        Args:
            symbol (str): 取引銘柄
            side (str): 取引方向 (BUY/SELL)
            
        Returns:
            float: 調整されたポジションサイズ（%）
        """
        base_size = self.config["POSITION_SIZE_PERCENTAGE"]
        
        # 1. 時間帯ベース調整
        if self.config["TIME_BASED_POSITION_SIZING"]:
            if self._is_optimized_trading_hours():
                base_size = self.config["OPTIMIZED_HOURS_POSITION_SIZE"]
            else:
                base_size = self.config["NON_OPTIMIZED_HOURS_POSITION_SIZE"]
        
        # 2. 勝率ベース調整
        if self.winning_streak >= 3:
            base_size *= 1.2  # 連続勝利時は20%増加
        elif self.winning_streak >= 2:
            base_size *= 1.1  # 2連勝時は10%増加
        
        # 3. 日次パフォーマンスベース調整
        if self.daily_profit_percentage > 5.0:
            base_size *= 1.15  # 好調な日は15%増加
        elif self.daily_profit_percentage < -3.0:
            base_size *= 0.8   # 不調な日は20%減少
        
        # 4. 連続損失による調整（既存ロジック）
        if self.consecutive_losses == 2:
            base_size /= 2  # 2連敗時は半減
        elif self.consecutive_losses >= 3:
            base_size = 0   # 3連敗時は停止
        
        # 5. 上限・下限の設定
        base_size = max(2.0, min(10.0, base_size))  # 2-10%の範囲
        
        return base_size
    
    def _calculate_current_portfolio_risk(self) -> float:
        """
        現在のポートフォリオリスク使用率を計算します。
        
        Returns:
            float: 現在のリスク使用率（%）
        """
        total_risk = 0.0
        
        for symbol, position in self.current_positions.items():
            # 各ポジションの現在価値を計算
            position_value = position["quantity"] * position["entry_price"]
            risk_percentage = (position_value / self.current_capital) * 100
            total_risk += risk_percentage
        
        return total_risk
    
    async def _check_pyramiding_opportunity(self, symbol: str, current_price: float):
        """
        ピラミッディング（利益追加）の機会をチェックします。
        
        Args:
            symbol (str): 銘柄シンボル
            current_price (float): 現在価格
        """
        if not self.config["PYRAMIDING_ENABLED"]:
            return
            
        if symbol not in self.current_positions:
            return
            
        position = self.current_positions[symbol]
        side = position["side"]
        entry_price = position["entry_price"]
        
        # 利益率を計算
        if side == "BUY":
            profit_percentage = ((current_price - entry_price) / entry_price) * 100
        else: # SELL
            profit_percentage = ((entry_price - current_price) / entry_price) * 100
        
        # ピラミッディング条件チェック
        if (profit_percentage >= self.config["PYRAMIDING_THRESHOLD_PERCENTAGE"] and 
            not position.get("pyramided", False)):  # まだピラミッディングしていない
            
            # ポートフォリオリスク制限チェック
            pyramid_size = self.config["PYRAMIDING_SIZE_PERCENTAGE"]
            trade_amount_usd = self.current_capital * (pyramid_size / 100)
            
            current_portfolio_risk = self._calculate_current_portfolio_risk()
            new_position_risk = trade_amount_usd / self.current_capital * 100
            
            if current_portfolio_risk + new_position_risk <= self.config["MAX_PORTFOLIO_RISK_PERCENTAGE"]:
                await self._execute_pyramiding(symbol, side, current_price, trade_amount_usd)
    
    async def _execute_pyramiding(self, symbol: str, side: str, current_price: float, trade_amount_usd: float):
        """
        ピラミッディング取引を実行します。
        
        Args:
            symbol (str): 銘柄シンボル
            side (str): 取引方向
            current_price (float): 現在価格
            trade_amount_usd (float): 追加取引金額
        """
        quantity = trade_amount_usd / current_price
        
        # 追加注文を実行
        order_response = await self.mexc_api.place_order(
            symbol=symbol,
            side=side,
            order_type="MARKET",
            quantity=quantity
        )
        
        if order_response and order_response.get("status") == "FILLED":
            # 既存ポジションを更新（平均価格計算）
            existing_position = self.current_positions[symbol]
            existing_quantity = existing_position["quantity"]
            existing_value = existing_quantity * existing_position["entry_price"]
            new_value = quantity * current_price
            
            total_quantity = existing_quantity + quantity
            average_price = (existing_value + new_value) / total_quantity
            
            # ポジション更新
            self.current_positions[symbol]["quantity"] = total_quantity
            self.current_positions[symbol]["entry_price"] = average_price
            self.current_positions[symbol]["pyramided"] = True
            
            self.logger.info(f"ピラミッディング実行: {symbol} {side} +{quantity:.4f} @ {current_price:.4f} (新平均価格: {average_price:.4f})")
            self.notifier.send_discord_message(f"ピラミッディング実行: {symbol} {side} +{quantity:.4f} @ {current_price:.4f}")
            
            # 資金更新（テストモードの場合）
            if self.config["TEST_MODE"]:
                # Ultra-Think手数料計算: ピラミッディングは指値注文（保守的にテイカーと仮定）
                trade_amount = quantity * current_price
                transaction_fee = self._calculate_trading_fee("LIMIT", trade_amount, is_maker=False)
                self.current_capital -= transaction_fee
        else:
            self.logger.error(f"ピラミッディング注文失敗: {symbol} {side}")
            self.notifier.send_discord_message(f"ピラミッディング注文失敗: {symbol} {side}")

    async def start_paper_trading(self):
        """
        テストモード（ペーパートレード）を開始します。
        リアルタイムデータに基づいて取引をシミュレートします。
        """
        self.logger.info("ペーパートレードを開始します。")
        await self._start_trading_loop()

    async def start_live_trading(self):
        """
        本番モード（ライブトレード）を開始します。
        実際の資金で取引を実行します。
        """
        self.logger.info("ライブトレードを開始します。")
        await self._start_trading_loop()

    async def _start_trading_loop(self):
        """
        MEXCのAPI制限に対応した取引ループ
        """
        self.logger.info("取引ループを開始します")
        
        # APIキー権限の確認
        if not await self.mexc_api.verify_api_permissions():
            self.logger.error("API権限確認に失敗。Botを終了します")
            return
        
        # 監視銘柄を選択（最大25個）
        self.current_symbols = await self._select_active_symbols()
        self.logger.info(f"選択された監視銘柄: {len(self.current_symbols)}個")
        self.logger.info(f"監視銘柄リスト: {self.current_symbols}")
        
        # REST APIのみで市場データをポーリング
        asyncio.create_task(self.mexc_api.start_rest_api_only_mode(self.current_symbols))
        
        # 既存の取引ロジック継続
        loop_count = 0
        while True:
            try:
                loop_count += 1
                
                # 5分ごとにWebSocketデータ状況を確認 (REST APIモードでは不要だが、ログ出力として残す)
                if loop_count % 5 == 0:
                    self.logger.info("REST APIモード: WebSocketデータ状況確認はスキップされます。")

                # 1時間ごとに監視銘柄を再選択
                if self.config["MARKET_CONDITION_ROTATION"] and self._should_rotate_symbols():
                    new_symbols = await self._select_active_symbols()
                    if new_symbols != self.current_symbols:
                        self.logger.info(f"監視銘柄をローテーション: {len(new_symbols)}個")
                        # REST APIモードではWebSocketの再起動は不要
                        self.current_symbols = new_symbols
                
                await self._manage_positions_and_risk()
                
                if len(self.current_positions) < self.config["MAX_SIMULTANEOUS_POSITIONS"]:
                    for symbol in self.current_symbols:
                        if symbol not in self.current_positions:
                            await self._check_and_execute_entry(symbol)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"取引ループエラー: {e}")
                await asyncio.sleep(300)

    def _should_rotate_symbols(self):
        """
        1時間ごとまたは市場環境変化時にローテーション実行
        """
        current_time = datetime.now()
        # 毎時00分にローテーションを検討
        # 市場環境変化の検知は_analyze_market_condition内で完結するため、ここでは時間ベースのトリガーのみ
        return current_time.minute == 0

    async def _manage_positions_and_risk(self):
        """
        現在保有しているポジションの管理とリスク管理ルールを適用します。
        利益確定、損切り、緊急損切りなどをチェックします。
        """
        # ビットコイン急落時の緊急損切りチェック
        await self._check_emergency_stop_loss()

        # 各ポジションのチェック
        positions_to_remove = []
        for symbol, position in list(self.current_positions.items()):
            current_price = await self.mexc_api.get_current_price(symbol)
            if current_price == 0:
                self.logger.warning(f"{symbol}の現在価格が取得できません。ポジション管理をスキップします。")
                continue

            # 利益確定とトレーリングストップのチェック
            if await self._check_take_profit(symbol, position, current_price):
                positions_to_remove.append(symbol)
                continue
            
            # ピラミッディング機会のチェック
            await self._check_pyramiding_opportunity(symbol, current_price)
            
            # 固定損切り、時間経過損切り、テクニカル指標ベース損切りのチェック
            if await self._check_stop_loss(symbol, position, current_price):
                positions_to_remove.append(symbol)
                continue
        
        for symbol in positions_to_remove:
            del self.current_positions[symbol]
            self.logger.info(f"ポジション {symbol} をクローズしました。")
            self.notifier.send_discord_message(f"ポジションクローズ: {symbol}")

        # 連続損失時のポジションサイズ自動縮小/取引停止のロジックは、
        # _execute_trade_and_update_capital で処理されるため、ここでは不要。
        # ただし、その日の取引停止状態のチェックはここでも行うべき。
        if self.consecutive_losses >= self.config["CONSECUTIVE_LOSS_THRESHOLD"]:
            self.logger.warning(f"{self.config['CONSECUTIVE_LOSS_THRESHOLD']}回連続で実質的な損失が発生したため、本日の取引を停止します。")
            self.notifier.send_discord_message(f"警告: {self.config['CONSECUTIVE_LOSS_THRESHOLD']}回連続で実質的な損失が発生したため、本日の取引を停止します。")
            # 翌日まで待機するロジック（例: 日付が変わるまでsleep）
            await self._wait_until_next_day()
            self.consecutive_losses = 0 # 日付が変わったらリセット
            self.logger.info("日付が変わりました。連続損失カウントをリセットし、取引を再開します。")
            self.notifier.send_discord_message("日付が変わりました。連続損失カウントをリセットし、取引を再開します。")

    async def _check_and_execute_entry(self, symbol: str):
        """
        指定されたシンボルのエントリー条件をチェックし、満たしていれば取引を実行します。
        """
        self.logger.info(f"{symbol}のエントリーシグナルをチェック中...")
        
        # OHLCVデータの取得
        ohlcv_5m = await self.mexc_api.get_ohlcv(symbol, "Min5", 100)
        ohlcv_15m = await self.mexc_api.get_ohlcv(symbol, "Min15", 100)
        ohlcv_1h = await self.mexc_api.get_ohlcv(symbol, "Min60", 100) # 1hはMin60

        if not ohlcv_5m or not ohlcv_15m or not ohlcv_1h:
            self.logger.warning(f"{symbol}のOHLCVデータが不足しているため、エントリーチェックをスキップします。")
            return

        df_5m = pd.DataFrame(ohlcv_5m)
        df_15m = pd.DataFrame(ohlcv_15m)
        df_1h = pd.DataFrame(ohlcv_1h)

        # テクニカル指標の計算
        # Bollinger Bands
        bb_period = self.get_adjusted_param(symbol, "BB_PERIOD", self.config["BB_PERIOD"])
        bb_std_dev = self.get_adjusted_param(symbol, "BB_STD_DEV", self.config["BB_STD_DEV"])
        df_5m["bb_upper"] = ta.volatility.bollinger_hband(df_5m["close"], window=bb_period, window_dev=bb_std_dev)
        df_5m["bb_lower"] = ta.volatility.bollinger_lband(df_5m["close"], window=bb_period, window_dev=bb_std_dev)
        df_5m["bb_mid"] = ta.volatility.bollinger_mavg(df_5m["close"], window=bb_period)

        # RSI
        rsi_period = self.get_adjusted_param(symbol, "RSI_PERIOD", self.config["RSI_PERIOD"])
        df_5m["rsi"] = ta.momentum.rsi(df_5m["close"], window=rsi_period)
        df_15m["rsi"] = ta.momentum.rsi(df_15m["close"], window=rsi_period) # 急落後反発スキャルピング用

        # MACD
        macd_fast_ema_period = self.get_adjusted_param(symbol, "MACD_FAST_EMA_PERIOD", self.config["MACD_FAST_EMA_PERIOD"])
        macd_slow_ema_period = self.get_adjusted_param(symbol, "MACD_SLOW_EMA_PERIOD", self.config["MACD_SLOW_EMA_PERIOD"])
        macd_signal_sma_period = self.get_adjusted_param(symbol, "MACD_SIGNAL_SMA_PERIOD", self.config["MACD_SIGNAL_SMA_PERIOD"])
        df_15m["macd"] = ta.trend.macd(df_15m["close"], window_fast=macd_fast_ema_period, window_slow=macd_slow_ema_period)
        df_15m["macd_signal"] = ta.trend.macd_signal(df_15m["close"], window_fast=macd_fast_ema_period, window_slow=macd_slow_ema_period, window_sign=macd_signal_sma_period)

        # EMA (1時間足)
        ema_periods = self.config["EMA_PERIODS"]
        df_1h["ema20"] = ta.trend.ema_indicator(df_1h["close"], window=ema_periods[0])
        df_1h["ema50"] = ta.trend.ema_indicator(df_1h["close"], window=ema_periods[1])

        # 出来高平均 (過去24時間)
        # 5分足データから過去24時間（24 * 12 = 288本）の出来高平均を計算
        if len(df_5m) >= 288:
            avg_volume_24h = df_5m["volume"].iloc[-288:].mean()
        else:
            avg_volume_24h = df_5m["volume"].mean() # データが少ない場合は全期間平均

        current_price = await self.mexc_api.get_current_price(symbol)
        if current_price == 0:
            self.logger.warning(f"{symbol}の現在価格が取得できません。エントリーチェックをスキップします。")
            return

        # 最新のデータポイントを取得
        last_5m_close = df_5m["close"].iloc[-1]
        last_5m_volume = df_5m["volume"].iloc[-1]
        last_5m_rsi = df_5m["rsi"].iloc[-1]
        last_15m_close = df_15m["close"].iloc[-1]
        last_15m_rsi = df_15m["rsi"].iloc[-1]
        last_15m_macd = df_15m["macd"].iloc[-1]
        last_15m_macd_signal = df_15m["macd_signal"].iloc[-1]
        last_1h_ema20 = df_1h["ema20"].iloc[-1]
        last_1h_ema50 = df_1h["ema50"].iloc[-1]

        # 直近30分価格変動率 (5分足6本分)
        if len(df_5m) >= 6:
            price_change_30min = (last_5m_close - df_5m["close"].iloc[-6]) / df_5m["close"].iloc[-6] * 100
        else:
            price_change_30min = 0 # データ不足

        # 出来高連続増加 (直近3本)
        volume_increasing_3_bars = False
        if len(df_5m) >= 3:
            if df_5m["volume"].iloc[-1] > df_5m["volume"].iloc[-2] and \
               df_5m["volume"].iloc[-2] > df_5m["volume"].iloc[-3]:
                volume_increasing_3_bars = True

        # 最適取引時間帯の判定
        current_hour_jst = datetime.now().hour
        is_optimized_hours = False
        for period in self.config["OPTIMIZED_TRADING_HOURS"]:
            if period["start"] <= current_hour_jst < period["end"]:
                is_optimized_hours = True
                break

        # ビットコイン価格のチェック (急落/急騰中でないか)
        btc_price = await self.mexc_api.get_current_price("BTCUSDT")
        btc_ohlcv_15m = await self.mexc_api.get_ohlcv("BTCUSDT", "Min15", 2) # 直近2本の15分足
        btc_is_crashing = False
        if btc_ohlcv_15m and len(btc_ohlcv_15m) >= 2:
            btc_price_change_15m = (btc_ohlcv_15m[-1]["close"] - btc_ohlcv_15m[0]["close"]) / btc_ohlcv_15m[0]["close"] * 100
            if btc_price_change_15m <= -self.config["BTC_CRASH_STOP_LOSS_PERCENTAGE"]:
                btc_is_crashing = True
                self.logger.warning(f"BTCが15分で{btc_price_change_15m:.2f}%下落。緊急損切り条件に近づいています。")

        # 市場全体の恐怖・強欲指数は外部APIが必要なため、今回は実装をスキップします。
        # 必要であれば、後で追加することを検討してください。

        # ロングエントリー条件チェック
        long_signal = self._check_long_entry_conditions(
            df_5m, df_15m, df_1h, symbol, current_price, is_optimized_hours, 
            avg_volume_24h, volume_increasing_3_bars, btc_is_crashing
        )

        # 急落後反発スキャルピング戦略 (ロング限定)
        scalping_long_signal = self._check_scalping_long_conditions(df_15m)

        if long_signal[0] or scalping_long_signal[0]:
            reason = long_signal[1] if long_signal[0] else scalping_long_signal[1]
            
            # エントリーフィルターチェック
            if self.config["ENTRY_FILTER_ENABLED"]:
                filter_passed = await self._additional_entry_filters(symbol, df_5m, df_15m)
                if not filter_passed:
                    self.logger.info(f"エントリーフィルターにより{symbol}のロングエントリーを回避")
                    return
            
            self.logger.info(f"ロングシグナル発生: {symbol}. 理由: {', '.join(reason)}")
            await self._execute_trade_and_update_capital(symbol, "BUY", current_price)
            return

        # ショートエントリー条件チェック
        short_signal = self._check_short_entry_conditions(
            df_5m, df_15m, df_1h, symbol, current_price, is_optimized_hours,
            avg_volume_24h, volume_increasing_3_bars, btc_ohlcv_15m
        )

        if short_signal[0]:
            # エントリーフィルターチェック
            if self.config["ENTRY_FILTER_ENABLED"]:
                filter_passed = await self._additional_entry_filters(symbol, df_5m, df_15m)
                if not filter_passed:
                    self.logger.info(f"エントリーフィルターにより{symbol}のショートエントリーを回避")
                    return
            
            self.logger.info(f"ショートシグナル発生: {symbol}. 理由: {', '.join(short_signal[1])}")
            await self._execute_trade_and_update_capital(symbol, "SELL", current_price)
            return

    def _check_long_entry_conditions(self, df_5m, df_15m, df_1h, symbol, current_price, 
                                     is_optimized_hours, avg_volume_24h, volume_increasing_3_bars, btc_is_crashing):
        """
        ロングエントリー条件をチェックします。
        
        Returns:
            tuple: (bool: signal, list: reasons)
        """
        long_signal = False
        long_reason = []

        # 最新のデータポイントを取得
        last_5m_close = df_5m["close"].iloc[-1]
        last_5m_volume = df_5m["volume"].iloc[-1]
        last_5m_rsi = df_5m["rsi"].iloc[-1]
        last_15m_macd = df_15m["macd"].iloc[-1]
        last_15m_macd_signal = df_15m["macd_signal"].iloc[-1]
        last_1h_ema20 = df_1h["ema20"].iloc[-1]
        last_1h_ema50 = df_1h["ema50"].iloc[-1]

        # 直近30分価格変動率 (5分足6本分)
        if len(df_5m) >= 6:
            price_change_30min = (last_5m_close - df_5m["close"].iloc[-6]) / df_5m["close"].iloc[-6] * 100
        else:
            price_change_30min = 0

        # 1. ボラティリティブレイクアウト
        bb_upper = df_5m["bb_upper"].iloc[-1]
        price_change_rate_long = self.get_adjusted_param(symbol, "PRICE_CHANGE_RATE_OPTIMIZED_HOURS_LONG", self.config["PRICE_CHANGE_RATE_OPTIMIZED_HOURS_LONG"])
        if not is_optimized_hours:
            price_change_rate_long = self.config["PRICE_CHANGE_RATE_OTHER_HOURS"]

        if last_5m_close > bb_upper and price_change_30min > (price_change_rate_long * df_5m["close"].pct_change().abs().mean() * 100):
            long_signal = True
            long_reason.append("ボラティリティブレイクアウト (ロング)")

        # 2. モメンタム確認
        rsi_threshold_long = self.get_adjusted_param(symbol, "RSI_THRESHOLD_LONG_OPTIMIZED_HOURS", self.config["RSI_THRESHOLD_LONG_OPTIMIZED_HOURS"])
        if not is_optimized_hours:
            rsi_threshold_long = self.config["RSI_THRESHOLD_LONG_OTHER_HOURS"]

        if not (last_5m_rsi > rsi_threshold_long and df_5m["rsi"].iloc[-1] > df_5m["rsi"].iloc[-2]):
            return False, []

        if last_5m_rsi > rsi_threshold_long and df_5m["rsi"].iloc[-1] > df_5m["rsi"].iloc[-2]:
            long_reason.append("RSI上昇モメンタム (ロング)")
        
        if not (last_15m_macd > last_15m_macd_signal):
            return False, []
        
        long_reason.append("MACDゴールデンクロス (ロング)")

        if not (last_1h_ema20 > last_1h_ema50):
            return False, []
        
        long_reason.append("1時間足EMAトレンド上昇 (ロング)")

        # 3. 出来高確認
        volume_multiplier_long = self.get_adjusted_param(symbol, "VOLUME_MULTIPLIER_OPTIMIZED_HOURS", self.config["VOLUME_MULTIPLIER_OPTIMIZED_HOURS"])
        if not is_optimized_hours:
            volume_multiplier_long = self.config["VOLUME_MULTIPLIER_OTHER_HOURS"]

        if not ((last_5m_volume > (avg_volume_24h * volume_multiplier_long)) or volume_increasing_3_bars):
            return False, []
        
        long_reason.append("出来高確認 (ロング)")

        # 4. 市場環境チェック
        if btc_is_crashing:
            return False, ["BTC急落中 (ロングエントリー回避)"]

        return long_signal, long_reason

    def _check_scalping_long_conditions(self, df_15m):
        """
        急落後反発スキャルピング戦略をチェックします。
        
        Returns:
            tuple: (bool: signal, list: reasons)
        """
        if len(df_15m) >= 2:
            price_drop_15m = (df_15m["close"].iloc[-1] - df_15m["close"].iloc[-2]) / df_15m["close"].iloc[-2] * 100
            if price_drop_15m <= -self.config["SCALPING_LONG_PRICE_DROP_PERCENTAGE"] and \
               df_15m["rsi"].iloc[-1] > self.config["SCALPING_LONG_RSI_THRESHOLD"] and \
               df_15m["rsi"].iloc[-2] <= self.config["SCALPING_LONG_RSI_THRESHOLD"]:
                return True, ["急落後反発スキャルピング (ロング)"]
        return False, []

    def _check_short_entry_conditions(self, df_5m, df_15m, df_1h, symbol, current_price,
                                      is_optimized_hours, avg_volume_24h, volume_increasing_3_bars, btc_ohlcv_15m):
        """
        ショートエントリー条件をチェックします。
        
        Returns:
            tuple: (bool: signal, list: reasons)
        """
        short_signal = False
        short_reason = []

        # 最新のデータポイントを取得
        last_5m_close = df_5m["close"].iloc[-1]
        last_5m_volume = df_5m["volume"].iloc[-1]
        last_5m_rsi = df_5m["rsi"].iloc[-1]
        last_15m_macd = df_15m["macd"].iloc[-1]
        last_15m_macd_signal = df_15m["macd_signal"].iloc[-1]
        last_1h_ema20 = df_1h["ema20"].iloc[-1]
        last_1h_ema50 = df_1h["ema50"].iloc[-1]

        # 直近30分価格変動率
        if len(df_5m) >= 6:
            price_change_30min = (last_5m_close - df_5m["close"].iloc[-6]) / df_5m["close"].iloc[-6] * 100
        else:
            price_change_30min = 0

        # 1. ボラティリティブレイクアウト
        bb_lower = df_5m["bb_lower"].iloc[-1]
        price_change_rate_short = self.get_adjusted_param(symbol, "PRICE_CHANGE_RATE_OPTIMIZED_HOURS_SHORT", self.config["PRICE_CHANGE_RATE_OPTIMIZED_HOURS_SHORT"])
        if not is_optimized_hours:
            price_change_rate_short = self.config["PRICE_CHANGE_RATE_OTHER_HOURS"]

        if last_5m_close < bb_lower and price_change_30min < -(price_change_rate_short * df_5m["close"].pct_change().abs().mean() * 100):
            short_signal = True
            short_reason.append("ボラティリティブレイクアウト (ショート)")

        # 2. モメンタム確認
        rsi_threshold_short = self.get_adjusted_param(symbol, "RSI_THRESHOLD_SHORT_OPTIMIZED_HOURS", self.config["RSI_THRESHOLD_SHORT_OPTIMIZED_HOURS"])
        if not is_optimized_hours:
            rsi_threshold_short = self.config["RSI_THRESHOLD_SHORT_OTHER_HOURS"]

        if not (last_5m_rsi < rsi_threshold_short and df_5m["rsi"].iloc[-1] < df_5m["rsi"].iloc[-2]):
            return False, []
        
        short_reason.append("RSI下降モメンタム (ショート)")

        if not (last_15m_macd < last_15m_macd_signal):
            return False, []
        
        short_reason.append("MACDデッドクロス (ショート)")

        if not (last_1h_ema20 < last_1h_ema50):
            return False, []
        
        short_reason.append("1時間足EMAトレンド下降 (ショート)")

        # 3. 出来高確認
        volume_multiplier_short = self.get_adjusted_param(symbol, "VOLUME_MULTIPLIER_OPTIMIZED_HOURS", self.config["VOLUME_MULTIPLIER_OPTIMIZED_HOURS"])
        if not is_optimized_hours:
            volume_multiplier_short = self.config["VOLUME_MULTIPLIER_OTHER_HOURS"]

        if not ((last_5m_volume > (avg_volume_24h * volume_multiplier_short)) or volume_increasing_3_bars):
            return False, []
        
        short_reason.append("出来高確認 (ショート)")

        # 4. 市場環境チェック
        # ビットコイン急騰中はショートエントリー回避
        btc_is_surging = False
        if btc_ohlcv_15m and len(btc_ohlcv_15m) >= 2:
            btc_price_change_15m = (btc_ohlcv_15m[-1]["close"] - btc_ohlcv_15m[0]["close"]) / btc_ohlcv_15m[0]["close"] * 100
            if btc_price_change_15m >= self.config["BTC_CRASH_STOP_LOSS_PERCENTAGE"]:
                btc_is_surging = True
                self.logger.warning(f"BTCが15分で{btc_price_change_15m:.2f}%上昇。ショートエントリー回避。")

        if btc_is_surging:
            return False, ["BTC急騰中 (ショートエントリー回避)"]

        return short_signal, short_reason

    async def _additional_entry_filters(self, symbol, df_5m, df_15m):
        """
        エントリー前の追加検証フィルター
        """
        try:
            # 1. 出来高確認の厳格化
            volume_trend = df_5m["volume"].rolling(3).mean().iloc[-1] > df_5m["volume"].rolling(10).mean().iloc[-1]
            
            # 2. 価格の安定性確認
            price_stability = df_5m["close"].rolling(3).std().iloc[-1] < df_5m["close"].rolling(10).std().iloc[-1] * self.config["PRICE_STABILITY_FACTOR"]
            
            # 3. 直近の損失取引履歴確認
            recent_losses = self._count_recent_losses_for_symbol(symbol, hours=6)
            
            # 4. 市場全体の相関確認 (簡略版)
            market_correlation = await self._check_market_correlation()
            
            return (volume_trend and 
                    price_stability and 
                    recent_losses < self.config["MAX_RECENT_LOSSES_PER_SYMBOL"] and 
                    market_correlation > self.config["REQUIRED_MARKET_CORRELATION"])
                    
        except Exception as e:
            self.logger.warning(f"エントリーフィルターでエラー: {e}")
            return True  # エラー時はフィルターをパス
    
    def _count_recent_losses_for_symbol(self, symbol, hours=6):
        """
        指定時間内の特定銘柄の損失回数をカウント
        """
        if symbol not in self.symbol_loss_history:
            return 0
        
        cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
        recent_losses = [
            entry for entry in self.symbol_loss_history[symbol]
            if entry["time"] > cutoff_time and entry["type"] == "loss"
        ]
        return len(recent_losses)
    
    async def _check_market_correlation(self):
        """
        市場全体の相関確認（簡略版）
        """
        try:
            # BTC、ETH、主要アルトコインの相関を簡単にチェック
            btc_price = await self.mexc_api.get_current_price("BTCUSDT")
            eth_price = await self.mexc_api.get_current_price("ETHUSDT")
            
            if btc_price > 0 and eth_price > 0:
                return 0.7  # 簡略化された相関値
            return 0.3
        except:
            return 0.5  # デフォルト値

    async def _execute_trade_and_update_capital(self, symbol: str, side: str, entry_price: float):
        """
        取引を実行し、現在の資金とポジションを更新します。
        ポジションサイズ調整、連続損失時の対応を含みます。
        """
        # ポジションサイズの計算
        position_size_percentage = self.config["POSITION_SIZE_PERCENTAGE"]
        if self.consecutive_losses == 1:
            # 1回目の損失は通常サイズ (総資金の5%) - これはデフォルトなので変更なし
            pass
        elif self.consecutive_losses == 2:
            # 2回目の連続損失はポジションサイズ半減 (総資金の2.5%)
            position_size_percentage /= 2
            self.logger.warning(f"2回連続損失のため、ポジションサイズを {position_size_percentage:.2f}% に縮小します。")
            self.notifier.send_discord_message(f"警告: 2回連続損失のため、ポジションサイズを {position_size_percentage:.2f}% に縮小します。")
        elif self.consecutive_losses >= 3:
            # 3回目の連続損失でその日の取引を自動停止 (この関数が呼ばれる前にチェックされるはずだが念のため)
            self.logger.warning("3回連続損失のため、本日の取引は停止されています。")
            self.notifier.send_discord_message("警告: 3回連続損失のため、本日の取引は停止されています。")
            return

        # 改善された動的ポジションサイジング
        if self.config["DYNAMIC_POSITION_SIZING_ENABLED"]:
            position_size_percentage = self._calculate_dynamic_position_size(symbol, side)
        
        trade_amount_usd = self.current_capital * (position_size_percentage / 100)
        
        # ポートフォリオリスク制限チェック
        current_portfolio_risk = self._calculate_current_portfolio_risk()
        new_position_risk = trade_amount_usd / self.current_capital * 100
        
        if current_portfolio_risk + new_position_risk > self.config["MAX_PORTFOLIO_RISK_PERCENTAGE"]:
            self.logger.warning(f"{symbol} ポートフォリオリスク超過 (現在: {current_portfolio_risk:.1f}% + 新規: {new_position_risk:.1f}% > 上限: {self.config['MAX_PORTFOLIO_RISK_PERCENTAGE']:.1f}%)")
            self.notifier.send_discord_message(f"{symbol} ポートフォリオリスク上限によりエントリーを回避しました。")
            return
        
        # 最小取引サイズチェック
        if trade_amount_usd < self.config["MINIMUM_TRADE_SIZE_USD"]:
            self.logger.warning(f"{symbol} 取引金額${trade_amount_usd:.2f}が最小取引サイズ${self.config['MINIMUM_TRADE_SIZE_USD']}を下回るため、エントリーを回避します。")
            self.notifier.send_discord_message(f"{symbol} 取引金額が最小サイズを下回るため、エントリーを回避しました。")
            return
        
        quantity = trade_amount_usd / entry_price # 数量を計算

        # 注文執行の最適化: 指値注文の価格設定を市場流動性に応じて動的に調整
        # ここでは簡略化のため、現在の価格にスリッページを考慮した指値価格を設定
        # 実際のMEXCの板情報や流動性を考慮した動的な調整はより複雑になります。
        order_price = entry_price
        if side == "BUY":
            order_price *= (1 + self.config["SLIPPAGE_PERCENTAGE"] / 100)
        else: # SELL
            order_price *= (1 - self.config["SLIPPAGE_PERCENTAGE"] / 100)

        # 注文発注
        order_response = await self.mexc_api.place_order(
            symbol=symbol,
            side=side,
            order_type="LIMIT", # 基本は指値注文
            quantity=quantity,
            price=order_price
        )

        if order_response and order_response.get("status") in ["FILLED", "NEW", "PARTIALLY_FILLED"]:
            self.logger.info(f"注文成功: {symbol} {side} {quantity:.4f} @ {entry_price:.4f} (注文価格: {order_price:.4f})")
            self.notifier.send_discord_message(f"注文成功: {symbol} {side} {quantity:.4f} @ {entry_price:.4f} (注文価格: {order_price:.4f})")
            
            # ポジション情報を更新
            self.current_positions[symbol] = {
                "entry_price": entry_price,
                "quantity": quantity,
                "side": side,
                "entry_time": datetime.now(),
                "order_id": order_response.get("orderId"),
                "highest_price": entry_price if side == "BUY" else None, # トレーリングストップ用
                "lowest_price": entry_price if side == "SELL" else None # トレーリングストップ用
            }
            self.consecutive_losses = 0 # 注文が成功したら連続損失をリセット
        else:
            self.logger.error(f"注文失敗: {symbol} {side}. レスポンス: {order_response}")
            self.notifier.send_discord_message(f"注文失敗: {symbol} {side}. レスポンス: {order_response}")
            # 注文失敗時は連続損失カウントを増やすべきか検討。今回は成功時のみリセットとする。

    async def _check_take_profit(self, symbol: str, position: dict, current_price: float) -> bool:
        """
        利益確定条件をチェックし、満たしていればポジションを決済します。

        Returns:
            bool: ポジションが決済された場合はTrue、それ以外はFalse
        """
        entry_price = position["entry_price"]
        side = position["side"]
        quantity = position["quantity"]
        
        # Ultra-Think手数料計算: MEXCの実際の手数料体系を使用
        # エントリー: LIMIT注文（メイカー0%またはテイカー0.05%）
        # 決済: MARKET注文（テイカー0.05%）

        # 現在の利益率を計算 (手数料考慮前)
        if side == "BUY":
            profit_percentage = ((current_price - entry_price) / entry_price) * 100
        else: # SELL (ショート)
            profit_percentage = ((entry_price - current_price) / entry_price) * 100

        # 手数料を考慮した実質利益率
        trade_amount = quantity * entry_price
        estimated_round_trip_fee = self._estimate_round_trip_fee(trade_amount)
        fee_percentage = (estimated_round_trip_fee / trade_amount) * 100
        adjusted_profit_percentage = profit_percentage - fee_percentage

        # 1段階目利益確定
        if adjusted_profit_percentage >= self.config["TAKE_PROFIT_PERCENTAGE_PHASE1"]:
            self.logger.info(f"{symbol} 1段階目利益確定シグナル発生 (利益率: {adjusted_profit_percentage:.2f}%)")
            self.notifier.send_discord_message(f"{symbol} 1段階目利益確定シグナル発生 (利益率: {adjusted_profit_percentage:.2f}%)")
            
            # ポジションの50%を決済
            half_quantity = quantity / 2
            order_response = await self.mexc_api.place_order(
                symbol=symbol,
                side="SELL" if side == "BUY" else "BUY", # 逆方向の注文
                order_type="MARKET", # 成行で即時決済
                quantity=half_quantity
            )
            if order_response and order_response.get("status") == "FILLED":
                self.logger.info(f"{symbol} 1段階目利益確定成功: {half_quantity:.4f} @ {current_price:.4f}")
                self.notifier.send_discord_message(f"{symbol} 1段階目利益確定成功: {half_quantity:.4f} @ {current_price:.4f}")
                position["quantity"] -= half_quantity # 残りの数量を更新
                # 資金を更新 (シミュレーションの場合)
                if self.config["TEST_MODE"]:
                    profit_amount = half_quantity * (current_price - entry_price if side == "BUY" else entry_price - current_price)
                    # Ultra-Think手数料計算: 決済は成行注文（テイカー手数料）
                    exit_trade_amount = half_quantity * current_price
                    exit_fee = self._calculate_trading_fee("MARKET", exit_trade_amount)
                    self.current_capital += profit_amount - exit_fee
                return False # 残りポジションがあるのでFalseを返す
            else:
                self.logger.error(f"{symbol} 1段階目利益確定失敗: {order_response}")
                self.notifier.send_discord_message(f"{symbol} 1段階目利益確定失敗: {order_response}")
                return False

        # トレーリングストップ
        trailing_stop_activation_percentage = self.config["TRAILING_STOP_ACTIVATION_PERCENTAGE"]
        trailing_stop_percentage = self.config["TRAILING_STOP_PERCENTAGE"]

        if side == "BUY":
            # 最高値の更新
            if position["highest_price"] is None or current_price > position["highest_price"]:
                position["highest_price"] = current_price
            
            # トレーリングストップ発動条件
            if (position["highest_price"] - entry_price) / entry_price * 100 >= trailing_stop_activation_percentage:
                # 価格が最高値から1.5%逆行したら決済
                if current_price <= position["highest_price"] * (1 - trailing_stop_percentage / 100):
                    self.logger.info(f"{symbol} トレーリングストップ発動 (ロング)")
                    self.notifier.send_discord_message(f"{symbol} トレーリングストップ発動 (ロング)")
                    await self._close_position(symbol, position, current_price, "TRAILING_STOP")
                    return True
        else: # SELL (ショート)
            # 最安値の更新
            if position["lowest_price"] is None or current_price < position["lowest_price"]:
                position["lowest_price"] = current_price
            
            # トレーリングストップ発動条件
            if (entry_price - position["lowest_price"]) / entry_price * 100 >= trailing_stop_activation_percentage:
                # 価格が最安値から1.5%逆行したら決済
                if current_price >= position["lowest_price"] * (1 + trailing_stop_percentage / 100):
                    self.logger.info(f"{symbol} トレーリングストップ発動 (ショート)")
                    self.notifier.send_discord_message(f"{symbol} トレーリングストップ発動 (ショート)")
                    await self._close_position(symbol, position, current_price, "TRAILING_STOP")
                    return True

        # 2段階目利益確定 (残り50%の決済条件)
        if position["quantity"] < quantity: # 1段階目決済後、残りがある場合
            # 価格がエントリー価格から5%以上有利に変動した場合
            if (side == "BUY" and adjusted_profit_percentage >= self.config["TAKE_PROFIT_PERCENTAGE_PHASE2_OPTION1"]) or \
               (side == "SELL" and adjusted_profit_percentage >= self.config["TAKE_PROFIT_PERCENTAGE_PHASE2_OPTION1"]):
                self.logger.info(f"{symbol} 2段階目利益確定シグナル発生 (価格変動)")
                self.notifier.send_discord_message(f"{symbol} 2段階目利益確定シグナル発生 (価格変動)")
                await self._close_position(symbol, position, current_price, "TAKE_PROFIT_PHASE2_PRICE")
                return True
            
            # 5分足RSIが逆転シグナルを示した場合
            ohlcv_5m = await self.mexc_api.get_ohlcv(symbol, "Min5", 100)
            if ohlcv_5m:
                df_5m = pd.DataFrame(ohlcv_5m)
                rsi_period = self.get_adjusted_param(symbol, "RSI_PERIOD", self.config["RSI_PERIOD"])
                df_5m["rsi"] = ta.momentum.rsi(df_5m["close"], window=rsi_period)
                last_5m_rsi = df_5m["rsi"].iloc[-1]

                if (side == "BUY" and last_5m_rsi >= 70) or \
                   (side == "SELL" and last_5m_rsi <= 30):
                    self.logger.info(f"{symbol} 2段階目利益確定シグナル発生 (RSI逆転)")
                    self.notifier.send_discord_message(f"{symbol} 2段階目利益確定シグナル発生 (RSI逆転)")
                    await self._close_position(symbol, position, current_price, "TAKE_PROFIT_PHASE2_RSI")
                    return True

            # 利益が3%に到達した場合 (手数料考慮後の実質利益か要確認、そうでなければ調整)
            if adjusted_profit_percentage >= self.config["TAKE_PROFIT_PERCENTAGE_PHASE2_OPTION3"]:
                self.logger.info(f"{symbol} 2段階目利益確定シグナル発生 (利益到達)")
                self.notifier.send_discord_message(f"{symbol} 2段階目利益確定シグナル発生 (利益到達)")
                await self._close_position(symbol, position, current_price, "TAKE_PROFIT_PHASE2_PROFIT")
                return True

        return False

    async def _check_stop_loss(self, symbol: str, position: dict, current_price: float) -> bool:
        """
        損切り条件をチェックし、満たしていればポジションを決済します。

        Returns:
            bool: ポジションが決済された場合はTrue、それ以外はFalse
        """
        entry_price = position["entry_price"]
        side = position["side"]
        quantity = position["quantity"]
        
        # Ultra-Think手数料計算: MEXCの実際の手数料体系を使用

        # 現在の損失率を計算 (手数料考慮前)
        if side == "BUY":
            loss_percentage = ((entry_price - current_price) / entry_price) * 100
        else: # SELL (ショート)
            loss_percentage = ((current_price - entry_price) / entry_price) * 100
        
        # 手数料を考慮した実質損失率
        trade_amount = quantity * entry_price
        estimated_round_trip_fee = self._estimate_round_trip_fee(trade_amount)
        fee_percentage = (estimated_round_trip_fee / trade_amount) * 100
        adjusted_loss_percentage = loss_percentage + fee_percentage # 損失なので手数料分は加算

        # 段階的損切りチェック（改善版）
        if self.config["PROGRESSIVE_STOP_LOSS_ENABLED"]:
            time_in_position = (datetime.now() - position["entry_time"]).total_seconds() / 60
            
            # 最小保有時間チェック
            if time_in_position < self.config["MINIMUM_HOLDING_TIME_MINUTES"]:
                return False
            
            # 実際の損失額計算
            if side == "BUY":
                actual_loss_amount = position["quantity"] * (entry_price - current_price)
            else:
                actual_loss_amount = position["quantity"] * (current_price - entry_price)
            
            # Ultra-Think手数料計算: 決済は成行注文（テイカー手数料）
            exit_trade_amount = position["quantity"] * current_price
            transaction_fee = self._calculate_trading_fee("MARKET", exit_trade_amount)
            actual_loss_amount += transaction_fee
            
            # 最小損失額以上かつ段階的条件を満たす場合のみ損切り
            if actual_loss_amount >= self.config["MINIMUM_LOSS_AMOUNT_USD"]:
                for stage in self.config["PROGRESSIVE_STOP_LOSS_STAGES"]:
                    if (time_in_position >= stage["time_minutes"] and 
                        adjusted_loss_percentage >= stage["loss_percentage"]):
                        self.logger.info(f"{symbol} 段階的損切りシグナル発生 (時間: {time_in_position:.0f}分, 損失率: {adjusted_loss_percentage:.2f}%, 損失額: ${actual_loss_amount:.2f})")
                        self.notifier.send_discord_message(f"{symbol} 段階的損切りシグナル発生 (時間: {time_in_position:.0f}分, 損失率: {adjusted_loss_percentage:.2f}%, 損失額: ${actual_loss_amount:.2f})")
                        await self._close_position(symbol, position, current_price, "PROGRESSIVE_STOP_LOSS")
                        # 実際の損失額ベースで連続損失をカウント
                        if actual_loss_amount >= self.config["MINIMUM_LOSS_AMOUNT_USD"]:
                            self.consecutive_losses += 1
                        return True
        else:
            # 固定損切り（改善版）
            time_in_position = (datetime.now() - position["entry_time"]).total_seconds() / 60
            if time_in_position >= self.config["MINIMUM_HOLDING_TIME_MINUTES"]:
                # 実際の損失額計算
                if side == "BUY":
                    actual_loss_amount = position["quantity"] * (entry_price - current_price)
                else:
                    actual_loss_amount = position["quantity"] * (current_price - entry_price)
                
                # Ultra-Think手数料計算: 決済は成行注文（テイカー手数料）
                exit_trade_amount = position["quantity"] * current_price
                transaction_fee = self._calculate_trading_fee("MARKET", exit_trade_amount)
                actual_loss_amount += transaction_fee
                
                if (adjusted_loss_percentage >= self.config["FIXED_STOP_LOSS_PERCENTAGE"] and 
                    actual_loss_amount >= self.config["MINIMUM_LOSS_AMOUNT_USD"]):
                    self.logger.info(f"{symbol} 固定損切りシグナル発生 (損失率: {adjusted_loss_percentage:.2f}%, 損失額: ${actual_loss_amount:.2f})")
                    self.notifier.send_discord_message(f"{symbol} 固定損切りシグナル発生 (損失率: {adjusted_loss_percentage:.2f}%, 損失額: ${actual_loss_amount:.2f})")
                    await self._close_position(symbol, position, current_price, "FIXED_STOP_LOSS")
                    self.consecutive_losses += 1
                    return True

        # 時間経過損切り（改善版）
        time_in_position = (datetime.now() - position["entry_time"]).total_seconds() / 60 # 分
        if (time_in_position >= self.config["TIME_BASED_STOP_LOSS_MINUTES"] and 
            time_in_position >= self.config["MINIMUM_HOLDING_TIME_MINUTES"] and
            adjusted_loss_percentage >= -self.config["TIME_BASED_STOP_LOSS_PROFIT_THRESHOLD"]):
            
            # 実際の損失額計算
            if side == "BUY":
                actual_loss_amount = position["quantity"] * (entry_price - current_price)
            else:
                actual_loss_amount = position["quantity"] * (current_price - entry_price)
            
            # Ultra-Think手数料計算: 決済は成行注文（テイカー手数料）
            exit_trade_amount = position["quantity"] * current_price
            transaction_fee = self._calculate_trading_fee("MARKET", exit_trade_amount)
            actual_loss_amount += transaction_fee
            
            if actual_loss_amount >= self.config["MINIMUM_LOSS_AMOUNT_USD"]:
                self.logger.info(f"{symbol} 時間経過損切りシグナル発生 (保有時間: {time_in_position:.0f}分, 利益率: {adjusted_loss_percentage:.2f}%, 損失額: ${actual_loss_amount:.2f})")
                self.notifier.send_discord_message(f"{symbol} 時間経過損切りシグナル発生 (保有時間: {time_in_position:.0f}分, 利益率: {adjusted_loss_percentage:.2f}%, 損失額: ${actual_loss_amount:.2f})")
                await self._close_position(symbol, position, current_price, "TIME_BASED_STOP_LOSS")
                self.consecutive_losses += 1
            else:
                # 微少損失の場合は連続損失としてカウントしない
                self.logger.info(f"{symbol} 時間経過決済 (微少損失のため連続損失カウントなし: ${actual_loss_amount:.2f})")
                await self._close_position(symbol, position, current_price, "TIME_BASED_STOP_LOSS")
            return True

        # テクニカル指標ベース損切り
        ohlcv_15m = await self.mexc_api.get_ohlcv(symbol, "Min15", 100)
        ohlcv_5m = await self.mexc_api.get_ohlcv(symbol, "Min5", 100)
        if ohlcv_15m and ohlcv_5m:
            df_15m = pd.DataFrame(ohlcv_15m)
            df_5m = pd.DataFrame(ohlcv_5m)

            # 15分足MACDがエントリー方向と逆のクロスを示した場合
            macd_fast_ema_period = self.get_adjusted_param(symbol, "MACD_FAST_EMA_PERIOD", self.config["MACD_FAST_EMA_PERIOD"])
            macd_slow_ema_period = self.get_adjusted_param(symbol, "MACD_SLOW_EMA_PERIOD", self.config["MACD_SLOW_EMA_PERIOD"])
            macd_signal_sma_period = self.get_adjusted_param(symbol, "MACD_SIGNAL_SMA_PERIOD", self.config["MACD_SIGNAL_SMA_PERIOD"])
            df_15m["macd"] = ta.trend.macd(df_15m["close"], window_fast=macd_fast_ema_period, window_slow=macd_slow_ema_period)
            df_15m["macd_signal"] = ta.trend.macd_signal(df_15m["close"], window_fast=macd_fast_ema_period, window_slow=macd_slow_ema_period, window_sign=macd_signal_sma_period)
            
            if len(df_15m) >= 2:
                macd_cross_signal = False
                if side == "BUY" and df_15m["macd"].iloc[-1] < df_15m["macd_signal"].iloc[-1] and df_15m["macd"].iloc[-2] >= df_15m["macd_signal"].iloc[-2]: # デッドクロス
                    macd_cross_signal = True
                elif side == "SELL" and df_15m["macd"].iloc[-1] > df_15m["macd_signal"].iloc[-1] and df_15m["macd"].iloc[-2] <= df_15m["macd_signal"].iloc[-2]: # ゴールデンクロス
                    macd_cross_signal = True
                
                if macd_cross_signal:
                    # 最小保有時間チェック
                    time_in_position = (datetime.now() - position["entry_time"]).total_seconds() / 60
                    if time_in_position >= self.config["MINIMUM_HOLDING_TIME_MINUTES"]:
                        # 実際の損失額計算
                        if side == "BUY":
                            actual_loss_amount = position["quantity"] * (entry_price - current_price)
                        else:
                            actual_loss_amount = position["quantity"] * (current_price - entry_price)
                        
                        # Ultra-Think手数料計算: 決済は成行注文（テイカー手数料）
                        exit_trade_amount = position["quantity"] * current_price
                        transaction_fee = self._calculate_trading_fee("MARKET", exit_trade_amount)
                        actual_loss_amount += transaction_fee
                        
                        self.logger.info(f"{symbol} テクニカル損切りシグナル発生 (MACD逆転, 損失額: ${actual_loss_amount:.2f})")
                        self.notifier.send_discord_message(f"{symbol} テクニカル損切りシグナル発生 (MACD逆転, 損失額: ${actual_loss_amount:.2f})")
                        await self._close_position(symbol, position, current_price, "MACD_STOP_LOSS")
                        
                        # 実際の損失額ベースで連続損失をカウント
                        if actual_loss_amount >= self.config["MINIMUM_LOSS_AMOUNT_USD"]:
                            self.consecutive_losses += 1
                        return True

            # 改善された5分足ボリンジャーバンドのミドルライン損切り
            bb_period = self.get_adjusted_param(symbol, "BB_PERIOD", self.config["BB_PERIOD"])
            df_5m["bb_mid"] = ta.volatility.bollinger_mavg(df_5m["close"], window=bb_period)
            
            confirmation_candles = self.config["BB_MID_CONFIRMATION_CANDLES"]
            threshold_percentage = self.config["BB_MID_BREAK_THRESHOLD_PERCENTAGE"]
            
            if len(df_5m) >= confirmation_candles + 1:
                bb_mid_break_signal = False
                current_price_data = df_5m["close"].iloc[-1]
                current_bb_mid = df_5m["bb_mid"].iloc[-1]
                
                # 価格差による閾値チェック
                price_deviation = abs(current_price_data - current_bb_mid) / current_bb_mid * 100
                
                if price_deviation >= threshold_percentage:
                    # 複数キャンドル確認
                    consecutive_breaks = 0
                    for i in range(1, confirmation_candles + 1):
                        if side == "BUY":
                            if df_5m["close"].iloc[-i] < df_5m["bb_mid"].iloc[-i]:
                                consecutive_breaks += 1
                        else:  # SELL
                            if df_5m["close"].iloc[-i] > df_5m["bb_mid"].iloc[-i]:
                                consecutive_breaks += 1
                    
                    if consecutive_breaks >= confirmation_candles:
                        bb_mid_break_signal = True
                
                if bb_mid_break_signal:
                    # 最小保有時間チェック
                    time_in_position = (datetime.now() - position["entry_time"]).total_seconds() / 60
                    if time_in_position >= self.config["MINIMUM_HOLDING_TIME_MINUTES"]:
                        # 実際の損失額計算
                        if side == "BUY":
                            actual_loss_amount = position["quantity"] * (entry_price - current_price)
                        else:
                            actual_loss_amount = position["quantity"] * (current_price - entry_price)
                        
                        # Ultra-Think手数料計算: 決済は成行注文（テイカー手数料）
                        exit_trade_amount = position["quantity"] * current_price
                        transaction_fee = self._calculate_trading_fee("MARKET", exit_trade_amount)
                        actual_loss_amount += transaction_fee
                        
                        self.logger.info(f"{symbol} 改善されたテクニカル損切りシグナル発生 (BBミドルラインブレイク: {price_deviation:.2f}%, 損失額: ${actual_loss_amount:.2f})")
                        self.notifier.send_discord_message(f"{symbol} 改善されたテクニカル損切りシグナル発生 (BBミドルラインブレイク: {price_deviation:.2f}%, 損失額: ${actual_loss_amount:.2f})")
                        await self._close_position(symbol, position, current_price, "BB_MID_STOP_LOSS")
                        
                        # 実際の損失額ベースで連続損失をカウント
                        if actual_loss_amount >= self.config["MINIMUM_LOSS_AMOUNT_USD"]:
                            self.consecutive_losses += 1
                        return True
        
        return False

    async def _check_emergency_stop_loss(self):
        """
        ビットコイン価格が急落した場合の緊急損切りルールをチェックし、実行します。
        """
        btc_ohlcv_15m = await self.mexc_api.get_ohlcv("BTCUSDT", "Min15", 2) # 直近2本の15分足
        if btc_ohlcv_15m and len(btc_ohlcv_15m) >= 2:
            btc_price_change_15m = (btc_ohlcv_15m[-1]["close"] - btc_ohlcv_15m[0]["close"]) / btc_ohlcv_15m[0]["close"] * 100
            
            if btc_price_change_15m <= -self.config["BTC_CRASH_STOP_LOSS_PERCENTAGE"]:
                self.logger.warning(f"ビットコインが15分で{btc_price_change_15m:.2f}%急落しました。緊急損切りを実行します。")
                self.notifier.send_discord_message(f"緊急損切り: ビットコインが15分で{btc_price_change_15m:.2f}%急落。全ロングポジションを決済します。")
                
                positions_to_close = list(self.current_positions.keys()) # イテレーション中に辞書を変更しないようにコピー
                for symbol in positions_to_close:
                    position = self.current_positions[symbol]
                    if position["side"] == "BUY": # ロングポジションのみ対象
                        current_price = await self.mexc_api.get_current_price(symbol)
                        if current_price != 0:
                            # 実際の損失額計算
                            entry_price = position["entry_price"]
                            actual_loss_amount = position["quantity"] * (entry_price - current_price)
                            
                            # Ultra-Think手数料計算: 緊急決済は成行注文（テイカー手数料）
                            exit_trade_amount = position["quantity"] * current_price
                            transaction_fee = self._calculate_trading_fee("MARKET", exit_trade_amount)
                            actual_loss_amount += transaction_fee
                            
                            await self._close_position(symbol, position, current_price, "EMERGENCY_STOP_LOSS_BTC_CRASH")
                            del self.current_positions[symbol] # 決済後削除
                            
                            # 実際の損失額ベースで連続損失をカウント
                            if actual_loss_amount >= self.config["MINIMUM_LOSS_AMOUNT_USD"]:
                                self.consecutive_losses += 1
                        else:
                            self.logger.error(f"緊急損切り中に{symbol}の価格が取得できませんでした。")
                            self.notifier.send_discord_message(f"エラー: 緊急損切り中に{symbol}の価格が取得できませんでした。")
        
    async def _close_position(self, symbol: str, position: dict, exit_price: float, reason: str):
        """
        ポジションを決済し、取引履歴を記録します。
        """
        side = position["side"]
        quantity = position["quantity"]
        entry_price = position["entry_price"]
        
        # 決済注文
        order_response = await self.mexc_api.place_order(
            symbol=symbol,
            side="SELL" if side == "BUY" else "BUY", # 逆方向の注文
            order_type="MARKET", # 成行で即時決済
            quantity=quantity
        )

        if order_response and order_response.get("status") == "FILLED":
            self.logger.info(f"ポジション決済成功: {symbol} {side} {quantity:.4f} @ {exit_price:.4f} (理由: {reason})")
            self.notifier.send_discord_message(f"ポジション決済成功: {symbol} {side} {quantity:.4f} @ {exit_price:.4f} (理由: {reason})")
            
            # 資金を更新 (シミュレーションの場合)
            if self.config["TEST_MODE"]:
                profit_loss_amount = quantity * (exit_price - entry_price if side == "BUY" else entry_price - exit_price)
                # Ultra-Think手数料計算: 決済は成行注文（テイカー手数料）
                exit_trade_amount = quantity * exit_price
                transaction_fee = self._calculate_trading_fee("MARKET", exit_trade_amount)
                profit_loss_amount -= transaction_fee
                
                self.current_capital += profit_loss_amount
                self.logger.info(f"現在の総資金 (テストモード): {self.current_capital:.2f} USD")
                self.notifier.send_discord_message(f"現在の総資金 (テストモード): {self.current_capital:.2f} USD")

            # 取引履歴に追加
            self.trade_history.append({
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "entry_time": position["entry_time"],
                "exit_time": datetime.now(),
                "reason": reason,
                "profit_loss_usd": profit_loss_amount if self.config["TEST_MODE"] else "N/A" # 本番モードでは実際の損益をAPIから取得する必要がある
            })
            
            # 勝敗履歴の記録（改善版）
            if self.config["TEST_MODE"]:
                if profit_loss_amount >= 0:
                    # 勝利の記録
                    self.winning_streak += 1
                    self.consecutive_losses = 0  # 連続損失をリセット
                    self.logger.info(f"勝利記録: {symbol} 利益 ${profit_loss_amount:.2f} (連続勝利: {self.winning_streak})")
                else:
                    # 損失の記録
                    self.winning_streak = 0  # 連続勝利をリセット
                    if symbol not in self.symbol_loss_history:
                        self.symbol_loss_history[symbol] = []
                    self.symbol_loss_history[symbol].append({
                        "time": datetime.now(),
                        "type": "loss",
                        "amount": profit_loss_amount,
                        "reason": reason
                    })
                
                # 日次利益率の更新
                self.daily_profit_percentage = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
            
            # 連続損失カウントは_check_stop_lossや_check_emergency_stop_lossで更新されるため、ここでは行わない
        else:
            self.logger.error(f"ポジション決済失敗: {symbol} {side}. レスポンス: {order_response}")
            self.notifier.send_discord_message(f"ポジション決済失敗: {symbol} {side}. レスポンス: {order_response}")

    async def _wait_until_next_day(self):
        """
        翌日の0時まで待機します。
        """
        now = datetime.now()
        tomorrow = now.date() + pd.Timedelta(days=1)
        midnight_tomorrow = datetime.combine(tomorrow, dt_time(0, 0, 0))
        wait_seconds = (midnight_tomorrow - now).total_seconds()
        self.logger.info(f"翌日 {midnight_tomorrow} まで待機します ({wait_seconds:.0f}秒)。")
        await asyncio.sleep(wait_seconds)

    def get_performance_metrics(self):
        """
        取引履歴からパフォーマンス指標を計算し、返します。
        """
        total_trades = len(self.trade_history)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_profit_loss_usd": 0.0,
                "average_profit_percentage": 0.0,
                "average_loss_percentage": 0.0,
                "max_drawdown": 0.0,
                "final_capital": self.current_capital
            }

        winning_trades = [t for t in self.trade_history if t["profit_loss_usd"] > 0]
        losing_trades = [t for t in self.trade_history if t["profit_loss_usd"] <= 0]

        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0.0
        total_profit_loss_usd = sum(t["profit_loss_usd"] for t in self.trade_history)

        avg_profit_percentage = 0.0
        if winning_trades:
            total_profit_percentage = sum(
                ((t["exit_price"] - t["entry_price"]) / t["entry_price"] * 100) if t["side"] == "BUY" else \
                ((t["entry_price"] - t["exit_price"]) / t["entry_price"] * 100)
                for t in winning_trades
            )
            # Ultra-Think手数料計算: 実際の往復手数料を考慮
            # 保守的見積もりとして、往復でテイカー手数料を適用
            round_trip_fee_percentage = constants.MEXC_TAKER_FEE_RATE * 2 * 100  # 0.1%
            avg_profit_percentage = (total_profit_percentage / len(winning_trades)) - round_trip_fee_percentage

        avg_loss_percentage = 0.0
        if losing_trades:
            total_loss_percentage = sum(
                ((t["entry_price"] - t["exit_price"]) / t["entry_price"] * 100) if t["side"] == "BUY" else \
                ((t["exit_price"] - t["entry_price"]) / t["entry_price"] * 100)
                for t in losing_trades
            )
            # Ultra-Think手数料計算: 実際の往復手数料を考慮  
            # 保守的見積もりとして、往復でテイカー手数料を適用
            round_trip_fee_percentage = constants.MEXC_TAKER_FEE_RATE * 2 * 100  # 0.1%
            avg_loss_percentage = (total_loss_percentage / len(losing_trades)) + round_trip_fee_percentage # 損失なので加算

        # 最大ドローダウンの計算 (簡略版)
        peak_capital = self.initial_capital
        max_drawdown = 0.0
        current_max_capital = self.initial_capital
        for trade in self.trade_history:
            # ここでは簡略化のため、各トレード後の資本を追跡する
            # より正確なDD計算には、時系列での資本推移を追う必要があります
            # テストモードでのみ有効
            if self.config["TEST_MODE"]:
                current_max_capital += trade["profit_loss_usd"]
                if current_max_capital > peak_capital:
                    peak_capital = current_max_capital
                drawdown = (peak_capital - current_max_capital) / peak_capital
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "total_profit_loss_usd": round(total_profit_loss_usd, 2),
            "average_profit_percentage": round(avg_profit_percentage, 2),
            "average_loss_percentage": round(avg_loss_percentage, 2),
            "max_drawdown": round(max_drawdown * 100, 2), # パーセンテージで返す
            "final_capital": round(self.current_capital, 2)
        }

    async def test_websocket_connection(self):
        """
        WebSocket接続のテスト（1銘柄から開始）
        (REST APIモードではこの関数は使用されません)
        """
        self.logger.info("WebSocket接続テストはREST APIモードではスキップされます。")
        return False