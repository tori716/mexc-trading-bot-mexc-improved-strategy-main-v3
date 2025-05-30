# decorators.py
# 共通デコレータを定義するモジュール

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, Optional
import aiohttp

def api_error_handler(func: Callable) -> Callable:
    """
    API呼び出し用統一エラーハンドリングデコレータ。
    
    Args:
        func: デコレートする関数
        
    Returns:
        Callable: デコレートされた関数
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        try:
            return await func(*args, **kwargs)
            
        except aiohttp.ClientResponseError as e:
            logger.error(f"APIレスポンスエラー in {func.__name__}: {e.status} - {e.message}")
            return None
            
        except aiohttp.ClientError as e:
            logger.error(f"APIクライアントエラー in {func.__name__}: {e}")
            return None
            
        except asyncio.TimeoutError:
            logger.error(f"APIタイムアウトエラー in {func.__name__}")
            return None
            
        except Exception as e:
            logger.error(f"予期せぬエラー in {func.__name__}: {e}")
            return None
    
    return wrapper

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True):
    """
    失敗時の自動リトライデコレータ。
    
    Args:
        max_retries: 最大リトライ回数
        delay: 初期遅延時間（秒）
        exponential_backoff: 指数バックオフを使用するか
        
    Returns:
        Callable: デコレータ関数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"最大リトライ回数に達しました in {func.__name__}: {e}")
                        raise
                    
                    current_delay = delay * (2 ** attempt) if exponential_backoff else delay
                    logger.warning(f"リトライ {attempt + 1}/{max_retries} in {func.__name__}: {e}, {current_delay}秒後に再試行")
                    await asyncio.sleep(current_delay)
            
        return wrapper
    return decorator

def rate_limit(calls_per_second: float = 1.0):
    """
    レート制限デコレータ。
    
    Args:
        calls_per_second: 1秒あたりの最大呼び出し回数
        
    Returns:
        Callable: デコレータ関数
    """
    min_interval = 1.0 / calls_per_second
    last_call_time = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_key = f"{func.__module__}.{func.__name__}"
            current_time = time.time()
            
            if func_key in last_call_time:
                elapsed = current_time - last_call_time[func_key]
                if elapsed < min_interval:
                    wait_time = min_interval - elapsed
                    await asyncio.sleep(wait_time)
            
            last_call_time[func_key] = time.time()
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator

def cache_result(duration: int = 60):
    """
    結果キャッシュデコレータ。
    
    Args:
        duration: キャッシュ持続時間（秒）
        
    Returns:
        Callable: デコレータ関数
    """
    cache = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # キャッシュキーの生成
            cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            current_time = time.time()
            
            # キャッシュから結果を取得
            if cache_key in cache:
                result, cached_time = cache[cache_key]
                if current_time - cached_time < duration:
                    return result
            
            # 新しい結果を取得してキャッシュ
            result = await func(*args, **kwargs)
            cache[cache_key] = (result, current_time)
            
            # 古いキャッシュエントリを削除
            expired_keys = [
                key for key, (_, cached_time) in cache.items()
                if current_time - cached_time >= duration
            ]
            for key in expired_keys:
                del cache[key]
            
            return result
            
        return wrapper
    return decorator

def log_execution_time(func: Callable) -> Callable:
    """
    関数の実行時間をログに記録するデコレータ。
    
    Args:
        func: デコレートする関数
        
    Returns:
        Callable: デコレートされた関数
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} 実行時間: {execution_time:.3f}秒")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} 実行時間（エラー）: {execution_time:.3f}秒")
            raise
    
    return wrapper

def validate_parameters(**validators):
    """
    パラメータ検証デコレータ。
    
    Args:
        **validators: パラメータ名と検証関数のマッピング
        
    Returns:
        Callable: デコレータ関数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            # 引数名の取得
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # パラメータ検証
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        logger.error(f"パラメータ検証失敗 in {func.__name__}: {param_name}={value}")
                        raise ValueError(f"Invalid parameter {param_name}: {value}")
            
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator

def circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60):
    """
    サーキットブレーカーパターンのデコレータ。
    
    Args:
        failure_threshold: 失敗回数の閾値
        recovery_timeout: 回復タイムアウト（秒）
        
    Returns:
        Callable: デコレータ関数
    """
    state = {"failures": 0, "last_failure_time": None, "is_open": False}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            current_time = time.time()
            
            # サーキットブレーカーが開いている場合の処理
            if state["is_open"]:
                if current_time - state["last_failure_time"] > recovery_timeout:
                    # タイムアウト後にハーフオープン状態へ
                    logger.info(f"サーキットブレーカー ハーフオープン: {func.__name__}")
                    state["is_open"] = False
                    state["failures"] = 0
                else:
                    logger.warning(f"サーキットブレーカー オープン中: {func.__name__}")
                    raise Exception("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                # 成功時は失敗カウントをリセット
                state["failures"] = 0
                return result
                
            except Exception as e:
                state["failures"] += 1
                state["last_failure_time"] = current_time
                
                if state["failures"] >= failure_threshold:
                    state["is_open"] = True
                    logger.error(f"サーキットブレーカー オープン: {func.__name__} (失敗回数: {state['failures']})")
                
                raise
            
        return wrapper
    return decorator

def singleton_execution(func: Callable) -> Callable:
    """
    同時実行を防ぐシングルトン実行デコレータ。
    
    Args:
        func: デコレートする関数
        
    Returns:
        Callable: デコレートされた関数
    """
    executing = set()
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        func_key = f"{func.__name__}:{id(args[0]) if args else 'global'}"
        
        if func_key in executing:
            logger.warning(f"関数 {func.__name__} は既に実行中です。スキップします。")
            return None
        
        executing.add(func_key)
        try:
            return await func(*args, **kwargs)
        finally:
            executing.discard(func_key)
    
    return wrapper

def monitor_performance(threshold_seconds: float = 1.0):
    """
    パフォーマンス監視デコレータ。
    
    Args:
        threshold_seconds: 警告を出す実行時間の閾値（秒）
        
    Returns:
        Callable: デコレータ関数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if execution_time > threshold_seconds:
                    logger.warning(f"パフォーマンス警告: {func.__name__} の実行時間が {execution_time:.3f}秒 (閾値: {threshold_seconds}秒)")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"エラー発生: {func.__name__} (実行時間: {execution_time:.3f}秒) - {e}")
                raise
            
        return wrapper
    return decorator

# 使用例とヘルパー関数
def is_valid_symbol(symbol: str) -> bool:
    """
    シンボル名の検証関数。
    
    Args:
        symbol: 検証するシンボル名
        
    Returns:
        bool: 有効なシンボルかどうか
    """
    return isinstance(symbol, str) and len(symbol) >= 3 and symbol.isupper()

def is_positive_number(value: Any) -> bool:
    """
    正の数値の検証関数。
    
    Args:
        value: 検証する値
        
    Returns:
        bool: 正の数値かどうか
    """
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return False

def is_valid_percentage(value: Any) -> bool:
    """
    有効なパーセンテージの検証関数。
    
    Args:
        value: 検証する値
        
    Returns:
        bool: 有効なパーセンテージ（0-100）かどうか
    """
    try:
        num_value = float(value)
        return 0 <= num_value <= 100
    except (ValueError, TypeError):
        return False