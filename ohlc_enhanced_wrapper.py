# ============================================================================
# ENHANCED OHLC FETCHER WITH RETRY LOGIC AND ERROR HANDLING
# ============================================================================
"""
Wrapper around fetchOHLC1 with:
- Automatic retry logic (3 attempts)
- Connection timeout management
- Better error logging
- Cache support for frequently accessed symbols
- Fallback validation
"""

import logging
import time
from typing import Optional, Dict, Tuple
import pandas as pd

# Setup logger
logger = logging.getLogger('OHLC_FETCHER')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | OHLC_FETCHER | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)

# Cache for OHLC fetches (symbol -> (df, timestamp))
_ohlc_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
_CACHE_TTL_SECONDS = 60  # Refresh cache every 60 seconds


class OHLCFetchConfig:
    """Configuration for OHLC fetching behavior"""
    
    def __init__(self):
        self.max_retries = 3
        self.initial_timeout = 5
        self.retry_backoff = 1.5
        self.cache_enabled = True
        self.cache_ttl = 60
        self.min_rows_required = 20


config = OHLCFetchConfig()


def fetch_ohlc_with_retry(
    symbol: str,
    interval: str = "5",
    duration: int = 10,
    timeout: Optional[float] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLC data with automatic retry logic.
    
    Args:
        symbol: Trading symbol (e.g., "BSE:SENSEX")
        interval: Timeframe (default "5" = 5 minutes)
        duration: Number of candles (default 10)
        timeout: Request timeout in seconds
    
    Returns:
        DataFrame with OHLC data or None if all retries fail
    """
    
    # Check cache first
    if config.cache_enabled:
        cached = _check_cache(symbol)
        if cached is not None:
            logger.debug(f"[CACHE HIT] {symbol} - Using cached data ({len(cached)} rows)")
            return cached
    
    # Retry loop
    attempt = 0
    current_timeout = timeout or config.initial_timeout
    
    while attempt < config.max_retries:
        attempt += 1
        try:
            logger.info(f"[ATTEMPT {attempt}/{config.max_retries}] Fetching {symbol} | interval={interval} | duration={duration}")
            
            # Import here to avoid circular imports
            from modules.Fyers.adx_efi_mom.service import fetchOHLC1
            
            # FIXED: fetchOHLC1() does NOT accept timeout parameter
            # Removed timeout=current_timeout which caused TypeError
            ohlc_df = fetchOHLC1(
                symbol,
                interval=str(interval),
                duration=duration
            )
            
            # Validate result
            if ohlc_df is not None and not ohlc_df.empty:
                rows = len(ohlc_df)
                
                if rows >= config.min_rows_required:
                    logger.info(f"[SUCCESS] {symbol} fetched successfully ({rows} rows)")
                    
                    # Store in cache
                    if config.cache_enabled:
                        _ohlc_cache[symbol] = (ohlc_df, time.time())
                    
                    return ohlc_df
                else:
                    logger.warning(f"[INSUFFICIENT DATA] {symbol} has only {rows} rows (need {config.min_rows_required})")
                    if attempt < config.max_retries:
                        wait_time = config.initial_timeout * (config.retry_backoff ** (attempt - 1))
                        logger.info(f"[RETRY] Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                        current_timeout *= config.retry_backoff
                        continue
            else:
                logger.warning(f"[EMPTY RESPONSE] {symbol} returned None or empty DataFrame")
                
        except TimeoutError:
            logger.warning(f"[TIMEOUT] {symbol} - Request timed out after {current_timeout}s")
            
        except ConnectionError as e:
            logger.warning(f"[CONNECTION ERROR] {symbol} - {str(e)}")
            
        except Exception as e:
            logger.error(f"[FETCH ERROR] {symbol} - {type(e).__name__}: {str(e)}")
        
        # Calculate backoff for next retry
        if attempt < config.max_retries:
            wait_time = config.initial_timeout * (config.retry_backoff ** (attempt - 1))
            logger.info(f"[RETRY {attempt+1}] Retrying {symbol} in {wait_time:.1f}s...")
            time.sleep(wait_time)
            current_timeout *= config.retry_backoff
    
    # All retries exhausted
    logger.error(f"[FAILED] {symbol} - All {config.max_retries} retry attempts exhausted")
    return None


def _check_cache(symbol: str) -> Optional[pd.DataFrame]:
    """Check if symbol data is cached and still valid"""
    if symbol not in _ohlc_cache:
        return None
    
    df, timestamp = _ohlc_cache[symbol]
    age_seconds = time.time() - timestamp
    
    if age_seconds > config.cache_ttl:
        logger.debug(f"[CACHE EXPIRED] {symbol} - Cache age: {age_seconds:.1f}s")
        del _ohlc_cache[symbol]
        return None
    
    return df


def clear_cache(symbol: Optional[str] = None):
    """Clear OHLC cache"""
    if symbol:
        if symbol in _ohlc_cache:
            del _ohlc_cache[symbol]
            logger.debug(f"[CACHE CLEARED] {symbol}")
    else:
        _ohlc_cache.clear()
        logger.debug("[CACHE CLEARED] All entries cleared")


def get_cache_stats() -> Dict:
    """Get cache statistics"""
    return {
        "cached_symbols": list(_ohlc_cache.keys()),
        "cache_size": len(_ohlc_cache),
        "cache_enabled": config.cache_enabled,
        "cache_ttl": config.cache_ttl
    }


# Fallback: if fetchOHLC1 not available
def generate_fallback_ohlc(symbol: str, rows: int = 100) -> pd.DataFrame:
    """Generate fallback mock OHLC data for testing"""
    import numpy as np
    from datetime import datetime, timedelta
    
    logger.warning(f"[FALLBACK] Generating mock OHLC data for {symbol} ({rows} rows)")
    
    data = []
    base_price = 84200.0  # SENSEX base
    current_time = datetime.now() - timedelta(minutes=5 * rows)
    
    for i in range(rows):
        # Random walk
        change = np.random.normal(0, 50)
        open_p = base_price + (i * 10) + change
        close_p = open_p + np.random.normal(0, 30)
        high_p = max(open_p, close_p) + abs(np.random.normal(0, 20))
        low_p = min(open_p, close_p) - abs(np.random.normal(0, 20))
        volume = int(np.random.uniform(10000, 100000))
        
        data.append({
            'Date': current_time,
            'Open': open_p,
            'High': high_p,
            'Low': low_p,
            'Close': close_p,
            'Volume': volume
        })
        
        current_time += timedelta(minutes=5)
    
    df = pd.DataFrame(data)
    logger.warning(f"[FALLBACK] Mock data created: {symbol} | {len(df)} rows | Close range: {df['Close'].min():.2f}-{df['Close'].max():.2f}")
    
    return df


# ============================================================================
# Integration function - use this in place of fetchOHLC1
# ============================================================================

def get_ohlc(
    symbol: str,
    interval: str = "5",
    duration: int = 10,
    use_fallback: bool = False,
    strict: bool = False
) -> Optional[pd.DataFrame]:
    """
    Get OHLC data with automatic retry, caching, and optional fallback.
    
    Args:
        symbol: Trading symbol
        interval: Timeframe (default "5")
        duration: Number of candles
        use_fallback: If True, generate mock data if real fetch fails
        strict: If True, return None rather than fallback data
    
    Returns:
        DataFrame or None
    """
    
    # Try real fetch first
    df = fetch_ohlc_with_retry(symbol, interval, duration)
    
    if df is not None:
        return df
    
    # Fallback handling
    if strict:
        logger.warning(f"[STRICT MODE] No data available for {symbol} - returning None")
        return None
    
    if use_fallback:
        return generate_fallback_ohlc(symbol, rows=max(100, duration * 10))
    
    return None


# ============================================================================
# Configuration helpers
# ============================================================================

def set_retry_config(max_retries: int = 3, backoff: float = 1.5, 
                     initial_timeout: float = 5, cache_ttl: int = 60):
    """Configure retry behavior"""
    config.max_retries = max_retries
    config.retry_backoff = backoff
    config.initial_timeout = initial_timeout
    config.cache_ttl = cache_ttl
    logger.info(f"[CONFIG] Retries={max_retries}, Backoff={backoff}, TTL={cache_ttl}s")


def enable_cache(enabled: bool = True):
    """Enable/disable OHLC caching"""
    config.cache_enabled = enabled
    logger.info(f"[CONFIG] Cache {'ENABLED' if enabled else 'DISABLED'}")


