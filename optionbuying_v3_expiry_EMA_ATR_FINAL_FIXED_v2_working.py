#!/usr/bin/env python3
# trading_bot.py - EXPIRY DAY VERSION (EMA 200/20 + ATR Ratio)
# ============================================================================
# MODIFICATIONS FOR EXPIRY DAY TRADING:
# - Replaced EFI + Momentum with EMA 200/20 alignment + ATR ratio
# - EMA 200/20 provides faster trend detection vs lagging EFI
# - ATR ratio detects market EXPANSION (>1.2) vs COMPRESSION (<0.8)
# - Synchronized with MERGED.py institutional logic
# - Better suited for expiry day "zero to hero" fast moves
# ============================================================================
from __future__ import annotations
import logging
import os
import csv
import json
import time
import sys
import threading
import warnings
import requests

# ---------------------------------------------------------------------------
# Forward declaration / compatibility shim
# ---------------------------------------------------------------------------
# NOTE:
# The legacy TradingBot (V1) initializes a "GenericGreeksOptionBuyerV2" inside
# its __init__. In this file, the refactored V2/V3 architecture is appended near
# the bottom. That means the class must exist at import time to avoid NameError.
#
# This shim defines the class early and lazily wires it to the later-defined
# OptionEngineV3 + data models.

class GenericGreeksOptionBuyerV2:
    """Lazy wrapper over OptionEngineV3.

    - Exists early so TradingBot.__init__ can reference it safely.
    - Builds the actual OptionEngineV3 on first use.
    """

    def __init__(self, mkt_adapter, configs=None):
        self.mkt = mkt_adapter
        self.configs = configs
        self._engine = None

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        eng_cls = globals().get("OptionEngineV3")
        if eng_cls is None:
            raise NameError("OptionEngineV3 is not defined yet")
        cfgs = self.configs or globals().get("DEFAULT_CONFIGS_V2") or {}
        self._engine = eng_cls(self.mkt, cfgs)
        return self._engine

    def analyze(self, resolved, *, voting_signal=None, votes_meta=None):
        """Return OptionBuySignalV2 or None."""
        engine = self._get_engine()

        desired_type = None
        force_atm = False
        if voting_signal:
            vs = str(voting_signal).upper()
            if "BULL" in vs:
                desired_type = "CE"
            elif "BEAR" in vs:
                desired_type = "PE"
            force_atm = ("STRONG" in vs)

        return engine.analyze(
            resolved,
            desired_type=desired_type,
            force_atm=force_atm,
            extra_meta=(votes_meta or {}),
        )
import traceback
from csv import DictWriter
import pickle
import datetime as dt
from datetime import time as dt_time
import pandas as pd
import numpy as np
import pytz
try:
    import joblib  # preferred for sklearn pipelines
except ImportError:
    joblib = None
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
from modules.Fyers.service import save_to_json, load_from_json, bollinger_bands, ema, atr
from modules.Fyers.adx_efi_mom.service import fetchOHLC1, register_ohlc_provider
from math import isfinite as _math_isfinite, log, sqrt, exp, erf, pi
from cpr_ai_predictor_v3 import CPR_AIPredictor
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum
from dataclasses import dataclass, field
try:
    # Python 3.9+ includes zoneinfo (your env shows Python 3.12)
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# ---------------------------------------------------------------------------
# Robust numeric guard
#
# FYERS payloads (and some config loaders) sometimes return numeric fields as
# strings. Several parts of this bot use `isfinite()` as a gate; `math.isfinite`
# raises TypeError when given a string ("must be real number, not str").
#
# We redefine `isfinite` to be string-safe across the whole script.
# ---------------------------------------------------------------------------

def isfinite(x) -> bool:  # type: ignore[override]
    try:
        return _math_isfinite(float(x))
    except Exception:
        return False

# ============================================================================
# PANDAS COMPATIBILITY: Handle SettingWithCopyWarning across versions
# ============================================================================
try:
    # Try to access SettingWithCopyWarning (pandas 1.x - 1.5.x)
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
except (AttributeError, TypeError):
    # If not available (pandas 2.0+), use generic pandas warning filter
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


DEFAULT_TIMEZONE = "Asia/Kolkata"
IST = pytz.timezone(DEFAULT_TIMEZONE)
tf_selected = sys.argv[1] if len(sys.argv) > 1 else "5"
print(f"[INFO] Selected timeframe: {tf_selected}m")

def get_now_iso():
    """Global function to get current timestamp in ISO format"""
    return dt.datetime.now(IST).isoformat()

AI_GATE_TRADES = True # Set to True to make AI signal a requirement for entry, False to just log it.
logger = logging.getLogger(__name__)

# ==========================
# TELEGRAM CONFIG
# ==========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7595713211:AAExZd-t8dzUtQK4OVj3kk5T6RDHnSe65d4")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1010802960")

# telegram_bot_token = "7595713211:AAExZd-t8dzUtQK4OVj3kk5T6RDHnSe65d4"
# telegram_chat_id = 1010802960

def send_telegram(msg: str):
    """Sends a notification to Telegram."""
    if TELEGRAM_TOKEN == "PUT_YOUR_TOKEN":
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARNING] [TELEGRAM] Failed to send msg: {e}")

def last(x, default=None):
    try:
        return x.iloc[-1]
    except Exception:
        return default

def elder_force_index(df, period=13):
    """
    Elder Force Index = EMA(Volume * (Close - Previous Close), period)
    """
    # Standardize column names to lowercase for the helper
    c = df["Close"] if "Close" in df.columns else df["close"]
    v = df["Volume"] if "Volume" in df.columns else df["volume"]
    force = v * (c - c.shift(1))
    return force.ewm(span=period, adjust=False).mean()

def momentum_indicator(df, period=10):
    """
    Momentum = Close - Close(n periods ago)
    """
    c = df["Close"] if "Close" in df.columns else df["close"]
    return c - c.shift(period)

def atr_based_strike_shift(atr, strike_step=100):
    """Simple ATR-based strike shift logic"""
    if atr > strike_step * 3: return 2
    if atr > strike_step * 1.5: return 1
    return 0


# =============================================================================
# SENSEX OPTION BUYING MODULE
# =============================================================================
# Implements: Black-76 Greeks, PCR Analysis, VIX Integration, OI Analysis
# Dynamic lot size, Dynamic profit targets based on ATR/volatility
# =============================================================================

@dataclass
class ChainStrikeRow:
    """Data structure for option chain strike information"""
    strike: float
    ce_oi: Optional[float] = None
    pe_oi: Optional[float] = None
    ce_oi_chg: Optional[float] = None
    pe_oi_chg: Optional[float] = None
    ce_symbol: Optional[str] = None
    pe_symbol: Optional[str] = None
    ce_ltp: Optional[float] = None
    pe_ltp: Optional[float] = None
    ce_iv: Optional[float] = None
    pe_iv: Optional[float] = None


@dataclass
class OptionBuySignal:
    """Signal result from SENSEX option buy analysis"""
    should_buy: bool = False
    symbol: str = ""
    strike: float = 0.0
    option_type: str = "CE"  # CE (Call) or PE (Put)
    reasons: List[str] = field(default_factory=list)
    greeks: Dict[str, float] = field(default_factory=dict)
    pcr: Optional[float] = None
    iv: Optional[float] = None
    vix: Optional[float] = None
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    sl_points: Optional[float] = None  # SL in points for OrderManager
    tp_points: Optional[float] = None  # TP in points for OrderManager
    lot_size: int = 20
    atr: Optional[float] = None  # ATR for dynamic trailing
    debug_info: Dict[str, Any] = field(default_factory=dict)


class Black76Greeks:
    """
    Black-76 Option Pricing Model for Index Futures Options
    Calculates IV, Delta, Gamma, Theta, Vega
    """
    
    @staticmethod
    def _N(x: float) -> float:
        """Standard normal CDF using error function"""
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    
    @staticmethod
    def _n(x: float) -> float:
        """Standard normal PDF"""
        return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x * x)
    
    @staticmethod
    def d1(F: float, K: float, sigma: float, T: float) -> float:
        """Calculate d1 for Black-76"""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt(T))
    
    @staticmethod
    def d2(F: float, K: float, sigma: float, T: float) -> float:
        """Calculate d2 for Black-76"""
        return Black76Greeks.d1(F, K, sigma, T) - sigma * sqrt(T)
    
    @staticmethod
    def price(F: float, K: float, r: float, sigma: float, T: float, opt_type: str = 'C') -> float:
        """
        Black-76 option price
        F: Futures price
        K: Strike price
        r: Risk-free rate (annualized decimal, e.g., 0.07 for 7%)
        sigma: Implied volatility (decimal, e.g., 0.18 for 18%)
        T: Time to expiry in years
        opt_type: 'C' for call, 'P' for put
        """
        if T <= 0:
            if opt_type == 'C':
                return max(F - K, 0.0)
            else:
                return max(K - F, 0.0)
        
        _d1 = Black76Greeks.d1(F, K, sigma, T)
        _d2 = _d1 - sigma * sqrt(T)
        df = exp(-r * T)
        
        if opt_type == 'C':
            return df * (F * Black76Greeks._N(_d1) - K * Black76Greeks._N(_d2))
        else:
            return df * (K * Black76Greeks._N(-_d2) - F * Black76Greeks._N(-_d1))
    
    @staticmethod
    def implied_vol(F: float, K: float, r: float, T: float, opt_type: str, 
                   market_price: float, tol: float = 1e-7, max_iter: int = 150) -> float:
        """
        Calculate implied volatility using bisection method
        Returns IV as a decimal (e.g., 0.18 for 18%)
        """
        if market_price <= 0 or T <= 0:
            return 0.0
            
        lo, hi = 1e-6, 5.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            model_price = Black76Greeks.price(F, K, r, mid, T, opt_type)
            if abs(model_price - market_price) < tol:
                return mid
            if model_price > market_price:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0
    
    @staticmethod
    def greeks(F: float, K: float, r: float, sigma: float, T: float, 
               opt_type: str = 'C') -> Dict[str, float]:
        """
        Calculate all Greeks for Black-76 model
        Returns: dict with delta, gamma, theta_day, vega, itm_prob
        """
        if T <= 0 or sigma <= 0:
            return {
                "delta": 0.0, "gamma": 0.0, "theta_day": 0.0, 
                "vega": 0.0, "itm_prob": 0.0, "d1": 0.0, "d2": 0.0
            }
        
        _d1 = Black76Greeks.d1(F, K, sigma, T)
        _d2 = _d1 - sigma * sqrt(T)
        df = exp(-r * T)
        pdf = Black76Greeks._n(_d1)
        
        if opt_type == 'C':
            delta = df * Black76Greeks._N(_d1)
            theta = (
                -df * (F * pdf * sigma / (2 * sqrt(T))) 
                - r * K * df * Black76Greeks._N(_d2)
                + r * F * df * Black76Greeks._N(_d1)
            )
            itm_prob = Black76Greeks._N(_d2)
        else:
            delta = -df * Black76Greeks._N(-_d1)
            theta = (
                -df * (F * pdf * sigma / (2 * sqrt(T)))
                + r * K * df * Black76Greeks._N(-_d2)
                - r * F * df * Black76Greeks._N(-_d1)
            )
            itm_prob = Black76Greeks._N(-_d2)
        
        gamma = df * pdf / (F * sigma * sqrt(T))
        vega = df * F * pdf * sqrt(T) / 100.0  # Per 1% vol change
        
        return {
            "delta": delta,
            "gamma": gamma,
            "theta_day": theta / 365.0,  # Daily theta
            "vega": vega,
            "itm_prob": itm_prob,
            "d1": _d1,
            "d2": _d2
        }


    @staticmethod
    def greeks_extended(F: float, K: float, r: float, sigma: float, T: float, opt_type: str = 'C',
                        premium: Optional[float] = None) -> Dict[str, float]:
        """Greeks plus rho and vanna.
        - rho: dPrice/dr (per 1% rate change)
        - vanna: dDelta/dSigma (per 1% vol change), computed numerically
        premium: optional market premium to compute rho based on model price if not provided
        """
        g = Black76Greeks.greeks(F, K, r, sigma, T, opt_type)
        # Rho: Black-76 price = exp(-rT) * (...) so dV/dr = -T * V
        model_price = Black76Greeks.price(F, K, r, sigma, T, opt_type)
        V = float(premium) if premium is not None and premium > 0 else float(model_price)
        rho = -T * V / 100.0  # per 1% rate move

        # Vanna: numerical dDelta/dSigma
        eps = 1e-4
        s_up = max(1e-6, sigma + eps)
        s_dn = max(1e-6, sigma - eps)
        d_up = Black76Greeks.greeks(F, K, r, s_up, T, opt_type).get('delta', 0.0)
        d_dn = Black76Greeks.greeks(F, K, r, s_dn, T, opt_type).get('delta', 0.0)
        vanna = ((d_up - d_dn) / (2.0 * eps)) / 100.0  # per 1% vol move

        g['rho'] = float(rho)
        g['vanna'] = float(vanna)
        return g


# =============================================================================
# OPTION ANALYTICS HELPERS (Premium, Moneyness, Volatility Estimates, Edge)
# =============================================================================

class OptionBasics:
    """Utility functions for option definitions used by the strategy."""

    @staticmethod
    def moneyness(underlying: float, strike: float, opt_type: str) -> str:
        """Return ITM/ATM/OTM based on underlying vs strike.
        opt_type: 'C' (call) or 'P' (put)
        """
        if underlying is None or strike is None or strike <= 0:
            return "UNKNOWN"
        if abs(underlying - strike) < 1e-9:
            return "ATM"
        if opt_type.upper() == 'C':
            return "ITM" if underlying > strike else "OTM"
        else:
            return "ITM" if underlying < strike else "OTM"

    @staticmethod
    def intrinsic_value(underlying: float, strike: float, opt_type: str) -> float:
        """Intrinsic value (floored at zero)."""
        if underlying is None or strike is None:
            return 0.0
        if opt_type.upper() == 'C':
            return max(underlying - strike, 0.0)
        else:
            return max(strike - underlying, 0.0)

    @staticmethod
    def premium_components(premium: float, underlying: float, strike: float, opt_type: str) -> Dict[str, float]:
        """Return intrinsic and time value components of option premium."""
        premium = float(premium or 0.0)
        ivalue = OptionBasics.intrinsic_value(underlying, strike, opt_type)
        tvalue = max(premium - ivalue, 0.0)
        return {
            "premium": premium,
            "intrinsic": ivalue,
            "time_value": tvalue,
            "time_value_pct": (tvalue / premium) if premium > 0 else 0.0,
        }

    @staticmethod
    def put_call_parity_fut(C: float, P: float, F: float, K: float, r: float, T: float) -> Dict[str, float]:
        """Put-call parity for futures options under Black-76:
        C - P = exp(-rT) * (F - K)
        Returns parity_lhs, parity_rhs, and mismatch.
        """
        df = exp(-r * T) if T and T > 0 else 1.0
        lhs = (C or 0.0) - (P or 0.0)
        rhs = df * ((F or 0.0) - (K or 0.0))
        return {
            "parity_lhs": lhs,
            "parity_rhs": rhs,
            "parity_mismatch": lhs - rhs,
        }


class VolatilityEstimators:
    """Realized volatility estimators for edge checks (IV vs forecast/realized)."""

    @staticmethod
    def _log_returns(close: pd.Series) -> pd.Series:
        close = close.astype(float)
        return np.log(close / close.shift(1)).dropna()

    @staticmethod
    def realized_vol(close: pd.Series, annualization: float) -> float:
        """Std dev of log returns, annualized."""
        r = VolatilityEstimators._log_returns(close)
        if len(r) < 2:
            return 0.0
        return float(r.std(ddof=1) * np.sqrt(annualization))

    @staticmethod
    def parkinson_vol(high: pd.Series, low: pd.Series, annualization: float) -> float:
        """Parkinson range-based volatility estimator, annualized."""
        h = high.astype(float)
        l = low.astype(float)
        x = np.log(h / l).dropna()
        if len(x) < 2:
            return 0.0
        var = (x.pow(2).mean()) / (4.0 * np.log(2.0))
        return float(np.sqrt(var * annualization))

    @staticmethod
    def ewma_vol(close: pd.Series, annualization: float, lam: float = 0.94) -> float:
        """EWMA volatility on log returns, annualized."""
        r = VolatilityEstimators._log_returns(close)
        if len(r) < 2:
            return 0.0
        # EWMA variance recursion
        var = float(r.var(ddof=1))
        for x in r.values[::-1]:
            var = lam * var + (1.0 - lam) * float(x * x)
        return float(np.sqrt(var * annualization))

    @staticmethod
    def trader_garch_vol(close: pd.Series, annualization: float, V: float, alpha: float = 0.05, beta: float = 0.90) -> float:
        """Simple GARCH(1,1)-style ("Trader GARCH") variance forecast.
        V is the long-run variance level (NOT volatility). Provide V as (long_run_vol^2 / annualization).
        """
        r = VolatilityEstimators._log_returns(close)
        if len(r) < 5:
            return 0.0
        gamma = max(0.0, 1.0 - alpha - beta)
        var = float(r.var(ddof=1))
        for x in r.values[::-1]:
            var = gamma * V + alpha * float(x * x) + beta * var
        return float(np.sqrt(var * annualization))


def annualization_factor_from_tf_minutes(tf_minutes: int) -> float:
    """Approx bars-per-year for Indian market intraday.
    Uses 252 trading days and ~375 minutes per day (9:15-15:30).
    """
    mins_per_day = 375
    bars_per_day = max(1, int(round(mins_per_day / max(1, tf_minutes))))
    return float(252 * bars_per_day)


class EdgeCalculator:
    """Option buying edge checks."""

    @staticmethod
    def vol_edge_points(vega_per_1pct: float, iv: float, forecast_vol: float) -> float:
        """Expected P/L from volatility edge only: Vega * (sigma_forecast - sigma_implied).
        vega_per_1pct is per 1% vol change. iv/forecast_vol are decimals.
        """
        if not (isfinite(vega_per_1pct) and isfinite(iv) and isfinite(forecast_vol)):
            return 0.0
        vol_diff_pct = (forecast_vol - iv) * 100.0
        return float(vega_per_1pct * vol_diff_pct)

    @staticmethod
    def total_edge_points(vega_per_1pct: float, iv: float, forecast_vol: float, theta_day: float, hold_days: float) -> float:
        """Edge including theta cost over expected holding window."""
        return EdgeCalculator.vol_edge_points(vega_per_1pct, iv, forecast_vol) + float(theta_day * hold_days)



class SensexOptionBuyer:
    """
    SENSEX Option Buying Strategy Implementation
    
    Features:
    - Black-76 Greeks calculation (Delta, Gamma, Theta, Vega)
    - Strike-wise PCR analysis
    - India VIX integration
    - OI and OI change analysis
    - Momentum/Trend confirmation
    - Dynamic lot size fetching
    - Dynamic profit targets based on ATR
    """
    
    # Default configuration - can be overridden in constructor
    DEFAULT_CONFIG = {
        # API Symbols (to be updated for current expiry)
        "underlying_index": "BSE:SENSEX-INDEX",
        "symbol_spot": "BSE:SENSEX-INDEX",
        "symbol_fut": "BSE:SENSEX26FEBFUT",  # Update for current expiry
        "symbol_vix": "NSE:INDIAVIX-INDEX",
        
        # Risk-free rate
        "risk_free_rate": 0.07,
        
        # Strike selection
        "strike_step": 100,  # SENSEX strike step
        "delta_min": 0.45,
        "delta_max": 0.65,
        
        # Greeks filters
        "max_theta_per_day": -3.0,  # Theta must be >= this
        "iv_min": 0.08,  # 8%
        "iv_max": 0.45,  # 45%
        "vix_max": 22.0,  
        # Premium decomposition filters
        "min_time_value_pct": 0.15,  # Require at least 15% time value in premium
        "min_edge_points": 0.0,      # Expected edge points (IV vs forecast + theta)
        "edge_hold_days": 1.0,       # Holding window for theta cost (intraday=1)
        "use_vol_edge_filter": True, # Enable IV vs forecast vol edge filter

        
        # Liquidity
        "min_bid_ask_spread_pct": 1.2,
        
        # PCR/OI filters
        "strike_pcr_min": 1.05,  # For Call: PE_OI / CE_OI at chosen strike
        "strike_pcr_max": 0.95,  # For Put: PE_OI / CE_OI at chosen strike
        "require_put_oi_support": True,
        "require_call_oi_shortcover": True,
        
        # Put-specific Delta
        "put_delta_min": -0.65,
        "put_delta_max": -0.45,
        
        # EMA periods for momentum
        "ema_fast": 9,
        "ema_slow": 21,
        
        # Historical data for momentum
        "hist_minutes": 120,
        "tf_minutes": 5,
        
        # Polling interval
        "poll_seconds": 15,
    }
    
    def __init__(self, fyers_client, log_fn=print, config: Dict = None):
        """
        Initialize SENSEX Option Buyer
        
        Args:
            fyers_client: Initialized Fyers API client
            log_fn: Logging function
            config: Configuration overrides
        """
        self.fyers = fyers_client
        self.log = log_fn
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.IST = pytz.timezone("Asia/Kolkata")
        
        # Cache for market data
        self._spot_ltp = None
        self._fut_ltp = None
        self._vix_ltp = None
        self._option_chain = None
        self._momentum_df = None
        
        # Dynamic lot size cache
        self._lot_size_cache = {}
        
        self.log("[SENSEX-OPT] Option Buyer initialized", True)
    
    def _safe_float(self, x, default=None) -> Optional[float]:
        """Safely convert to float"""
        try:
            if x is None or x == "":
                return default
            val = float(x)
            return val if isfinite(val) else default
        except (ValueError, TypeError):
            return default
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch quote via Fyers API
        Returns dict with lp (last price), bid, ask, volume, etc.
        """
        try:
            resp = self.fyers.quotes(data={"symbols": symbol})
            if resp.get("s") == "ok" and "d" in resp and resp["d"]:
                return resp["d"][0].get("v", {})
            self.log(f"[SENSEX-OPT] Quote fetch failed for {symbol}: {resp}", False)
            return {}
        except Exception as e:
            self.log(f"[SENSEX-OPT] Quote error {symbol}: {e}", False)
            return {}
    
    def get_dynamic_lot_size(self, symbol: str = None) -> int:
        """
        Fetch lot size dynamically from Fyers API
        Falls back to default (20 for SENSEX options) if API fails
        """
        symbol = symbol or self.config["underlying_index"]
        
        # Check cache first
        if symbol in self._lot_size_cache:
            return self._lot_size_cache[symbol]
        
        try:
            # Try to get lot size from symbol master/quotes
            resp = self.fyers.quotes(data={"symbols": symbol})
            if resp.get("s") == "ok" and "d" in resp:
                for item in resp.get("d", []):
                    v = item.get("v", {})
                    lot = v.get("lot_size") or v.get("min_qty") or v.get("lotSize")
                    if lot:
                        lot_size = int(lot)
                        self._lot_size_cache[symbol] = lot_size
                        self.log(f"[SENSEX-OPT] Dynamic lot size for {symbol}: {lot_size}", True)
                        return lot_size
            
            # Fallback: Try option chain for lot size
            chain_resp = self.get_option_chain()
            if chain_resp:
                data = chain_resp.get("data", {})
                lot = data.get("lotSize") or data.get("lot_size")
                if lot:
                    lot_size = int(lot)
                    self._lot_size_cache[symbol] = lot_size
                    return lot_size
                    
        except Exception as e:
            self.log(f"[SENSEX-OPT] Error fetching lot size: {e}", False)
        
        # Default lot sizes by index
        default_lots = {
            "SENSEX": 20,
            "NIFTY": 75,  # Nifty 50 lot size
            "BANKNIFTY": 30,  # Bank Nifty lot size
        }
        
        for idx, lot in default_lots.items():
            if idx in symbol.upper():
                self._lot_size_cache[symbol] = lot
                self.log(f"[SENSEX-OPT] Using default lot size for {symbol}: {lot}", True)
                return lot
        
        # Ultimate fallback
        self._lot_size_cache[symbol] = 20
        return 20
    
    def get_option_chain(self, strikecount: str = "") -> Dict:
        """Fetch option chain from Fyers"""
        try:
            data = {
                "symbol": self.config["underlying_index"],
                "strikecount": strikecount,
                "timestamp": ""
            }
            resp = self.fyers.optionchain(data=data)
            return resp if resp.get("s") == "ok" else {}
        except Exception as e:
            self.log(f"[SENSEX-OPT] Option chain error: {e}", False)
            return {}
    
    def fetch_ohlc(self, symbol: str, tf_min: int, lookback_min: int) -> pd.DataFrame:
        """Fetch OHLC historical data for momentum calculation"""
        try:
            now = dt.datetime.now(self.IST)
            start = now - dt.timedelta(minutes=lookback_min)
            
            data = {
                "symbol": symbol,
                "resolution": str(tf_min),
                "date_format": "0",
                "range_from": str(int(start.timestamp())),
                "range_to": str(int(now.timestamp())),
                "cont_flag": "1"
            }
            resp = self.fyers.history(data=data)
            
            if resp.get("s") != "ok" or "candles" not in resp:
                return pd.DataFrame()
            
            candles = resp["candles"]
            df = pd.DataFrame(candles, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
            df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(self.IST)
            df.set_index("dt", inplace=True)
            return df
            
        except Exception as e:
            self.log(f"[SENSEX-OPT] OHLC fetch error: {e}", False)
            return pd.DataFrame()
    
    def calculate_momentum_signal(self, df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        """
        Calculate momentum confirmation using EMAs
        Returns: (direction, info_dict) where direction is BULLISH, BEARISH, or NEUTRAL
        """
        if df is None or df.empty or len(df) < 30:
            return "NEUTRAL", {"error": "Insufficient data"}
        
        close = df["Close"].astype(float)
        ema_fast = close.ewm(span=self.config["ema_fast"], adjust=False).mean()
        ema_slow = close.ewm(span=self.config["ema_slow"], adjust=False).mean()
        
        is_bullish = bool(
            ema_fast.iloc[-1] > ema_slow.iloc[-1] and 
            close.iloc[-1] > ema_fast.iloc[-1]
        )
        is_bearish = bool(
            ema_fast.iloc[-1] < ema_slow.iloc[-1] and 
            close.iloc[-1] < ema_fast.iloc[-1]
        )
        
        direction = "BULLISH" if is_bullish else ("BEARISH" if is_bearish else "NEUTRAL")
        
        return direction, {
            f"ema_{self.config['ema_fast']}": float(ema_fast.iloc[-1]),
            f"ema_{self.config['ema_slow']}": float(ema_slow.iloc[-1]),
            "close": float(close.iloc[-1]),
            "direction": direction
        }
    
    def parse_option_chain_to_strikes(self, chain_resp: Dict) -> Dict[float, ChainStrikeRow]:
        """Parse option chain response to strike-wise data"""
        strikes: Dict[float, ChainStrikeRow] = {}
        
        data = chain_resp.get("data", {})
        chain = data.get("optionsChain") or data.get("options_chain") or []
        
        for row in chain:
            try:
                sp = self._safe_float(
                    row.get("strike_price") or row.get("strike") or row.get("sp"),
                    None
                )
                if sp is None:
                    continue
                
                opt_type = row.get("option_type") or row.get("type")
                oi = self._safe_float(row.get("oi") or row.get("open_interest"))
                oi_chg = self._safe_float(row.get("oich") or row.get("oi_chg") or row.get("oi_change"))
                sym = row.get("symbol") or row.get("tsym") or row.get("tradingsymbol")
                ltp = self._safe_float(row.get("ltp") or row.get("lp") or row.get("last_price"))
                iv = self._safe_float(row.get("iv") or row.get("implied_volatility"))
                
                if sp not in strikes:
                    strikes[sp] = ChainStrikeRow(strike=sp)
                
                if opt_type == "CE":
                    strikes[sp].ce_oi = oi
                    strikes[sp].ce_oi_chg = oi_chg
                    strikes[sp].ce_symbol = sym
                    strikes[sp].ce_ltp = ltp
                    strikes[sp].ce_iv = iv
                elif opt_type == "PE":
                    strikes[sp].pe_oi = oi
                    strikes[sp].pe_oi_chg = oi_chg
                    strikes[sp].pe_symbol = sym
                    strikes[sp].pe_ltp = ltp
                    strikes[sp].pe_iv = iv
                    
            except Exception as e:
                self.log(f"[SENSEX-OPT] Parse strike error: {e}", True)
                continue
        
        return strikes
    
    def calculate_strike_pcr(self, row: ChainStrikeRow) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate PCR at a specific strike"""
        if row.ce_oi and row.pe_oi and row.ce_oi > 0:
            return row.pe_oi / row.ce_oi, row.ce_oi, row.pe_oi
        return None, row.ce_oi, row.pe_oi
    
    def get_bid_ask(self, quote: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Extract bid and ask prices from quote"""
        bid = None
        ask = None
        
        for key in ("bid", "best_bid_price", "bidPrice", "bp"):
            if key in quote and quote[key] not in (None, "", 0):
                bid = self._safe_float(quote[key])
                break
        
        for key in ("ask", "best_ask_price", "askPrice", "ap"):
            if key in quote and quote[key] not in (None, "", 0):
                ask = self._safe_float(quote[key])
                break
        
        return bid, ask
    
    def calculate_spread_pct(self, bid: Optional[float], ask: Optional[float]) -> Optional[float]:
        """Calculate bid-ask spread percentage"""
        if bid is None or ask is None or bid <= 0:
            return None
        return (ask - bid) / bid * 100.0
    
    def parse_expiry_from_chain(self, chain_resp: Dict) -> Optional[dt.datetime]:
        """Parse expiry date from option chain response"""
        data = chain_resp.get("data", {})
        
        for key in ("expiry", "expiryDate", "expirydt", "expiry_dt"):
            val = data.get(key)
            if not val:
                continue
            try:
                if isinstance(val, (int, float)):
                    return dt.datetime.fromtimestamp(val, tz=self.IST)
                s = str(val)
                return dt.datetime.fromisoformat(s).astimezone(self.IST)
            except Exception:
                pass
        return None
    
    def time_to_expiry_years(self, expiry_dt: dt.datetime) -> float:
        """Calculate time to expiry in years"""
        now = dt.datetime.now(self.IST)
        seconds = max((expiry_dt - now).total_seconds(), 1)
        return seconds / (365.0 * 24 * 3600)
    
    def pick_atm_strike(self, spot_price: float) -> float:
        """Select ATM strike based on current spot price"""
        step = self.config["strike_step"]
        return round(spot_price / step) * step
    
    def build_option_symbol(self, base_example: str, strike: float, opt_type: str) -> str:
        """
        Build option symbol from example format
        Example: BSE:SENSEX26FEB83500CE -> BSE:SENSEX26FEB84000CE
        """
        import re
        m = re.search(r'^(.*?)(\d+)(CE|PE)$', base_example)
        if not m:
            raise ValueError(f"Cannot parse symbol format: {base_example}")
        prefix = m.group(1)
        return f"{prefix}{int(strike)}{opt_type}"
    
    def calculate_dynamic_profit_targets(self, entry_price: float, atr: float, 
                                         iv: float, vix: float = None) -> Dict[str, float]:
        """
        Calculate dynamic stop loss and take profit based on ATR and volatility
        
        Args:
            entry_price: Option entry price
            atr: ATR of underlying
            iv: Implied volatility of the option
            vix: India VIX (optional)
        
        Returns:
            Dict with sl_points, tp_points, sl_price, tp_price
        """
        # Base multipliers
        sl_atr_mult = 1.2  # SL at 1.2x ATR
        tp_atr_mult = 2.0  # TP at 2x ATR (2:1 risk-reward)
        
        # Adjust for volatility
        if iv is not None:
            if iv > 0.30:  # High IV (>30%)
                sl_atr_mult *= 1.3  # Wider SL in high IV
                tp_atr_mult *= 1.5  # Higher target possible
            elif iv < 0.15:  # Low IV (<15%)
                sl_atr_mult *= 0.8  # Tighter SL
                tp_atr_mult *= 0.8  # Lower target expectation
        
        # Adjust for VIX
        if vix is not None:
            if vix > 20:  # High fear
                sl_atr_mult *= 1.2
                tp_atr_mult *= 1.3
            elif vix < 12:  # Low fear/complacency
                sl_atr_mult *= 0.9
                tp_atr_mult *= 0.9
        
        # Calculate points with ATR scaling
        # For options, use percentage of entry price scaled by ATR
        atr_pct = atr / max(entry_price, 1) if entry_price > 0 else 0.1
        
        sl_points = max(entry_price * atr_pct * sl_atr_mult, entry_price * 0.10)  # Min 10% SL
        tp_points = max(entry_price * atr_pct * tp_atr_mult, entry_price * 0.20)  # Min 20% TP
        
        # Apply max caps
        sl_points = min(sl_points, entry_price * 0.50)  # Max 50% SL
        tp_points = min(tp_points, entry_price * 2.0)   # Max 200% TP
        
        return {
            "sl_points": round(sl_points, 2),
            "tp_points": round(tp_points, 2),
            "sl_price": round(max(entry_price - sl_points, 0.05), 2),
            "tp_price": round(entry_price + tp_points, 2),
            "risk_reward_ratio": round(tp_points / sl_points, 2) if sl_points > 0 else 0,
            "sl_pct": round(sl_points / entry_price * 100, 2) if entry_price > 0 else 0,
            "tp_pct": round(tp_points / entry_price * 100, 2) if entry_price > 0 else 0,
        }
    
    def evaluate_call_buy_conditions(
        self,
        spot_ltp: float,
        fut_ltp: float,
        opt_symbol: str,
        opt_quote: Dict,
        chain_row: ChainStrikeRow,
        iv: float,
        greeks: Dict,
        vix: Optional[float],
        momentum_direction: str,
        momentum_info: Dict,
        expiry_dt: Optional[dt.datetime],
        atr: float = None,
        forecast_vol: Optional[float] = None
    ) -> OptionBuySignal:
        """
        Evaluate all conditions for SENSEX call option buy
        
        Returns: OptionBuySignal with all decision data
        """
        signal = OptionBuySignal()
        signal.symbol = opt_symbol
        signal.strike = chain_row.strike
        signal.greeks = greeks
        signal.iv = iv
        signal.vix = vix
        
        opt_ltp = self._safe_float(opt_quote.get("lp"), 0.0)
        bid, ask = self.get_bid_ask(opt_quote)
        spread = self.calculate_spread_pct(bid, ask)
        
        signal.entry_price = opt_ltp
        signal.option_type = "CE"
        signal.lot_size = self.get_dynamic_lot_size()
        signal.atr = atr
        
        debug = {
            "spot_ltp": spot_ltp,
            "fut_ltp": fut_ltp,
            "opt_ltp": opt_ltp,
            "bid": bid,
            "ask": ask,
            "spread_pct": spread,
            "iv": iv,
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta_day": greeks.get("theta_day"),
            "vega": greeks.get("vega"),
            "itm_prob": greeks.get("itm_prob"),
            "vix": vix,
            "momentum_direction": momentum_direction,
            "momentum_info": momentum_info,
            "ce_oi": chain_row.ce_oi,
            "pe_oi": chain_row.pe_oi,
            "ce_oi_chg": chain_row.ce_oi_chg,
            "pe_oi_chg": chain_row.pe_oi_chg,
        }
        signal.debug_info = debug
        
        reasons = []
        
        # 1) Liquidity check
        if spread is None:
            signal.reasons = ["No bid/ask available (illiquid)"]
            return signal
        if spread > self.config["min_bid_ask_spread_pct"]:
            signal.reasons = [f"Spread too wide: {spread:.2f}%"]
            return signal
        reasons.append(f"[OK] Liquidity OK (spread {spread:.2f}%)")
        
        # 2) Momentum confirmation
        if momentum_direction != "BULLISH":
            signal.reasons = [f"Momentum not bullish: {momentum_direction}"]
            return signal
        reasons.append("[OK] Momentum OK (Bullish Trend)")
        
        # 3) VIX filter
        if vix is not None and vix > self.config["vix_max"]:
            signal.reasons = [f"VIX too high: {vix:.2f} > {self.config['vix_max']}"]
            return signal
        if vix is not None:
            reasons.append(f"[OK] VIX OK ({vix:.2f})")
        
        # 4) IV filter
        if iv < self.config["iv_min"] or iv > self.config["iv_max"]:
            signal.reasons = [
                f"IV out of range: {iv*100:.2f}% "
                f"(allowed {self.config['iv_min']*100:.0f}%–{self.config['iv_max']*100:.0f}%)"
            ]
            return signal
        reasons.append(f"[OK] IV OK ({iv*100:.2f}%)")

        # Premium decomposition / moneyness
        prem = float(opt_ltp or 0.0)
        comps = OptionBasics.premium_components(prem, fut_ltp, chain_row.strike, 'C' if signal.option_type=='CE' else 'P')
        mny = OptionBasics.moneyness(fut_ltp, chain_row.strike, 'C' if signal.option_type=='CE' else 'P')
        signal.debug_info.update({
            'premium': comps['premium'],
            'intrinsic': comps['intrinsic'],
            'time_value': comps['time_value'],
            'time_value_pct': comps['time_value_pct'],
            'moneyness': mny,
        })
        if comps['time_value_pct'] < self.config.get('min_time_value_pct', 0.0):
            signal.reasons = [f"Time value too low: {comps['time_value_pct']*100:.1f}%"]
            return signal
        reasons.append(f"[OK] Premium OK (TV {comps['time_value_pct']*100:.1f}%, {mny})")

        # Edge check: IV vs forecast vol (plus theta cost)
        if self.config.get('use_vol_edge_filter', True) and forecast_vol is not None:
            edge = EdgeCalculator.total_edge_points(
                vega_per_1pct=greeks.get('vega', 0.0),
                iv=iv,
                forecast_vol=float(forecast_vol),
                theta_day=greeks.get('theta_day', 0.0),
                hold_days=float(self.config.get('edge_hold_days', 1.0)),
            )
            signal.debug_info['forecast_vol'] = float(forecast_vol)
            signal.debug_info['edge_points'] = float(edge)
            if edge < float(self.config.get('min_edge_points', 0.0)):
                signal.reasons = [f"No edge (IV vs forecast): {edge:.2f} pts"]
                return signal
            reasons.append(f"[OK] Edge OK ({edge:.2f} pts)")

        
        # 5) Delta filter
        delta = greeks.get("delta", 0.0)
        if not (self.config["delta_min"] <= delta <= self.config["delta_max"]):
            signal.reasons = [f"Delta not in preferred range: {delta:.3f}"]
            return signal
        reasons.append(f"[OK] Delta OK ({delta:.3f})")
        
        # 6) Theta filter
        theta_day = greeks.get("theta_day", -999.0)
        if theta_day < self.config["max_theta_per_day"]:
            signal.reasons = [f"Theta/day too negative: {theta_day:.2f}"]
            return signal
        reasons.append(f"[OK] Theta OK ({theta_day:.2f}/day)")
        
        # 7) Strike PCR filter
        pcr, ce_oi, pe_oi = self.calculate_strike_pcr(chain_row)
        signal.pcr = pcr
        
        if pcr is None:
            signal.reasons = ["Strike PCR not available (missing OI)"]
            return signal
            
        if pcr < self.config["strike_pcr_min"]:
            signal.reasons = [f"Strike PCR too low: {pcr:.2f} < {self.config['strike_pcr_min']}"]
            return signal
        reasons.append(f"[OK] Strike PCR OK ({pcr:.2f})")
        
        # 8) OI change filters (if available)
        if self.config["require_put_oi_support"] and chain_row.pe_oi_chg is not None:
            if chain_row.pe_oi_chg <= 0:
                signal.reasons = [f"Put OI change not supportive: {chain_row.pe_oi_chg}"]
                return signal
            reasons.append(f"[OK] Put OI support OK (chg {chain_row.pe_oi_chg})")
        
        if self.config["require_call_oi_shortcover"] and chain_row.ce_oi_chg is not None:
            if chain_row.ce_oi_chg >= 0:
                signal.reasons = [f"Call OI not short-covering: {chain_row.ce_oi_chg}"]
                return signal
            reasons.append(f"[OK] Call OI short-covering OK (chg {chain_row.ce_oi_chg})")
        
        # 9) Time to expiry info
        if expiry_dt:
            dte_days = (expiry_dt - dt.datetime.now(self.IST)).total_seconds() / 86400.0
            signal.debug_info["dte_days"] = dte_days
            if dte_days <= 1:
                reasons.append(f"[WARNING] DTE very close ({dte_days:.2f} days) - high theta risk")
            else:
                reasons.append(f"[OK] DTE ~ {dte_days:.2f} days")
        
        # Calculate dynamic SL/TP
        if atr and opt_ltp > 0:
            targets = self.calculate_dynamic_profit_targets(opt_ltp, atr, iv, vix)
            signal.sl_price = targets["sl_price"]
            signal.tp_price = targets["tp_price"]
            signal.debug_info["dynamic_targets"] = targets
            reasons.append(
                f"[OK] Dynamic SL={targets['sl_price']:.2f} "
                f"TP={targets['tp_price']:.2f} "
                f"(RR={targets['risk_reward_ratio']:.2f})"
            )
        
        # All conditions passed!
        signal.should_buy = True
        signal.reasons = reasons
        
        if atr and opt_ltp > 0:
            signal.sl_points = targets["sl_points"]
            signal.tp_points = targets["tp_points"]
            
        return signal

    def evaluate_put_buy_conditions(
        self,
        spot_ltp: float,
        fut_ltp: float,
        opt_symbol: str,
        opt_quote: Dict,
        chain_row: ChainStrikeRow,
        iv: float,
        greeks: Dict,
        vix: Optional[float],
        momentum_direction: str,
        momentum_info: Dict,
        expiry_dt: Optional[dt.datetime],
        atr: float = None,
        forecast_vol: Optional[float] = None
    ) -> OptionBuySignal:
        """
        Evaluate all conditions for SENSEX put option buy
        
        Returns: OptionBuySignal with all decision data
        """
        signal = OptionBuySignal()
        signal.symbol = opt_symbol
        signal.strike = chain_row.strike
        signal.option_type = "PE"
        signal.greeks = greeks
        signal.iv = iv
        signal.vix = vix
        signal.atr = atr
        
        opt_ltp = self._safe_float(opt_quote.get("lp"), 0.0)
        bid, ask = self.get_bid_ask(opt_quote)
        spread = self.calculate_spread_pct(bid, ask)
        
        signal.entry_price = opt_ltp
        signal.lot_size = self.get_dynamic_lot_size()
        
        signal.debug_info = {
            "spot_ltp": spot_ltp,
            "fut_ltp": fut_ltp,
            "opt_ltp": opt_ltp,
            "iv": iv,
            "delta": greeks.get("delta"),
            "momentum_direction": momentum_direction,
            "ce_oi": chain_row.ce_oi,
            "pe_oi": chain_row.pe_oi,
        }
        
        reasons = []
        
        # 1) Liquidity check
        if spread is None or spread > self.config["min_bid_ask_spread_pct"]:
            signal.reasons = [f"Liquidity/Spread Issue: {spread if spread else 'N/A'}%"]
            return signal
        reasons.append(f"[OK] Liquidity OK (spread {spread:.2f}%)")
        
        # 2) Momentum confirmation
        if momentum_direction != "BEARISH":
            signal.reasons = [f"Momentum not bearish: {momentum_direction}"]
            return signal
        reasons.append("[OK] Momentum OK (Bearish Trend)")
        
        # 3) VIX filter
        if vix is not None and vix > self.config["vix_max"]:
            signal.reasons = [f"VIX too high: {vix:.2f}"]
            return signal
        reasons.append(f"[OK] VIX OK ({vix:.2f})" if vix else "[OK] VIX N/A")
        
        # 4) IV filter
        if not (self.config["iv_min"] <= iv <= self.config["iv_max"]):
            signal.reasons = [f"IV out of range: {iv*100:.1f}%"]
            return signal
        reasons.append(f"[OK] IV OK ({iv*100:.2f}%)")

        # Premium decomposition / moneyness
        prem = float(opt_ltp or 0.0)
        comps = OptionBasics.premium_components(prem, fut_ltp, chain_row.strike, 'C' if signal.option_type=='CE' else 'P')
        mny = OptionBasics.moneyness(fut_ltp, chain_row.strike, 'C' if signal.option_type=='CE' else 'P')
        signal.debug_info.update({
            'premium': comps['premium'],
            'intrinsic': comps['intrinsic'],
            'time_value': comps['time_value'],
            'time_value_pct': comps['time_value_pct'],
            'moneyness': mny,
        })
        if comps['time_value_pct'] < self.config.get('min_time_value_pct', 0.0):
            signal.reasons = [f"Time value too low: {comps['time_value_pct']*100:.1f}%"]
            return signal
        reasons.append(f"[OK] Premium OK (TV {comps['time_value_pct']*100:.1f}%, {mny})")

        # Edge check: IV vs forecast vol (plus theta cost)
        if self.config.get('use_vol_edge_filter', True) and forecast_vol is not None:
            edge = EdgeCalculator.total_edge_points(
                vega_per_1pct=greeks.get('vega', 0.0),
                iv=iv,
                forecast_vol=float(forecast_vol),
                theta_day=greeks.get('theta_day', 0.0),
                hold_days=float(self.config.get('edge_hold_days', 1.0)),
            )
            signal.debug_info['forecast_vol'] = float(forecast_vol)
            signal.debug_info['edge_points'] = float(edge)
            if edge < float(self.config.get('min_edge_points', 0.0)):
                signal.reasons = [f"No edge (IV vs forecast): {edge:.2f} pts"]
                return signal
            reasons.append(f"[OK] Edge OK ({edge:.2f} pts)")

        
        # 5) Delta filter (Puts have negative delta)
        delta = greeks.get("delta", 0.0)
        if not (self.config["put_delta_min"] <= delta <= self.config["put_delta_max"]):
            signal.reasons = [f"Delta not in preferred range: {delta:.3f} (need {self.config['put_delta_min']} to {self.config['put_delta_max']})"]
            return signal
        reasons.append(f"[OK] Delta OK ({delta:.3f})")
        
        # 6) Theta filter
        theta_day = greeks.get("theta_day", -999.0)
        if theta_day < self.config["max_theta_per_day"]:
            signal.reasons = [f"Theta too negative: {theta_day:.2f}"]
            return signal
        reasons.append(f"[OK] Theta OK ({theta_day:.2f}/day)")
        
        # 7) Strike PCR filter (For Puts, we want low PCR = CE dominance at strike)
        pcr, _, _ = self.calculate_strike_pcr(chain_row)
        signal.pcr = pcr
        if pcr is None or pcr > self.config["strike_pcr_max"]:
            signal.reasons = [f"Strike PCR not supportive for PE: {pcr:.2f} (max {self.config['strike_pcr_max']})"]
            return signal
        reasons.append(f"[OK] Strike PCR OK ({pcr:.2f})")
        
        # Calculate dynamic SL/TP
        if atr and opt_ltp > 0:
            targets = self.calculate_dynamic_profit_targets(opt_ltp, atr, iv, vix)
            signal.sl_price = targets["sl_price"]
            signal.tp_price = targets["tp_price"]
            signal.sl_points = targets["sl_points"]
            signal.tp_points = targets["tp_points"]
            reasons.append(f"[OK] Dynamic Targets: SL={targets['sl_price']:.2f} TP={targets['tp_price']:.2f}")
        
        # All conditions passed!
        signal.should_buy = True
        signal.reasons = reasons
        return signal
    
    def analyze_and_get_signal(self) -> OptionBuySignal:
        """
        Main analysis function - analyzes market and returns buy signal
        
        Returns: OptionBuySignal with decision and reasoning
        """
        ts = dt.datetime.now(self.IST).strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"\n[{ts}] ═══════════════════════════════════════════", False)
        self.log("[SENSEX-OPT] Running BUY analysis (Call/Put)...", False)
        
        try:
            # Fetch futures price (for Black-76)
            fut_quote = self.get_quote(self.config["symbol_fut"])
            fut_ltp = self._safe_float(fut_quote.get("lp"))
            
            if not fut_ltp or fut_ltp <= 0:
                return OptionBuySignal(reasons=["Future LTP not available"])
            
            # Fetch spot price (optional)
            spot_quote = self.get_quote(self.config["symbol_spot"])
            spot_ltp = self._safe_float(spot_quote.get("lp"), fut_ltp)
            
            # Fetch VIX
            vix_quote = self.get_quote(self.config["symbol_vix"])
            vix_ltp = self._safe_float(vix_quote.get("lp"))
            
            # Fetch momentum data
            df = self.fetch_ohlc(
                self.config["symbol_spot"],
                self.config["tf_minutes"],
                self.config["hist_minutes"]
            )
            momentum_direction, momentum_info = self.calculate_momentum_signal(df)
            
            # Calculate ATR for dynamic targets
            atr_val = None
            if not df.empty and len(df) >= 14:
                tr = pd.concat([
                    df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift()).abs(),
                    (df["Low"] - df["Close"].shift()).abs()
                ], axis=1).max(axis=1)
                atr_val = tr.rolling(14).mean().iloc[-1]
            
            # Volatility estimates (for IV vs realized/forecast edge)
            forecast_vol = None
            try:
                ann = annualization_factor_from_tf_minutes(int(self.config.get('tf_minutes', 5)))
                if df is not None and not df.empty and 'Close' in df.columns:
                    rv = VolatilityEstimators.realized_vol(df['Close'].tail(120), ann)
                    ev = VolatilityEstimators.ewma_vol(df['Close'].tail(240), ann, lam=0.94)
                    pv = VolatilityEstimators.parkinson_vol(df['High'].tail(240), df['Low'].tail(240), ann) if 'High' in df.columns and 'Low' in df.columns else 0.0
                    # Simple long-run variance guess from realized vol
                    long_run_vol = max(0.05, min(1.0, rv))
                    V = (long_run_vol * long_run_vol) / ann
                    gv = VolatilityEstimators.trader_garch_vol(df['Close'].tail(500), ann, V=V, alpha=0.05, beta=0.90)
                    # Choose a conservative forecast for buyers: max of EWMA/GARCH (captures trending volatility)
                    forecast_vol = max(ev, gv, rv, pv)
                    self.log(f"[SENSEX-OPT] Vol forecast: RV={rv:.3f} EWMA={ev:.3f} GARCH={gv:.3f} PARK={pv:.3f} -> FCAST={forecast_vol:.3f}", False)
            except Exception as _e:
                forecast_vol = None

            # Fetch option chain
            chain_resp = self.get_option_chain()
            if not chain_resp:
                return OptionBuySignal(reasons=["Option chain not available"])
            
            expiry_dt = self.parse_expiry_from_chain(chain_resp)
            T = self.time_to_expiry_years(expiry_dt) if expiry_dt else (2.0 / 365.0)
            
            strikes = self.parse_option_chain_to_strikes(chain_resp)
            if not strikes:
                return OptionBuySignal(reasons=["No strikes parsed from option chain"])
            
            # Select candidate strikes
            atm = self.pick_atm_strike(fut_ltp)
            step = self.config["strike_step"]
            candidates = [atm, atm + step, atm - step]
            candidates = [s for s in candidates if s in strikes and strikes[s].ce_oi is not None]
            
            if not candidates:
                available = sorted(strikes.keys(), key=lambda x: abs(x - atm))
                candidates = available[:3]
            
            self.log(
                f"[SENSEX-OPT] Spot={spot_ltp:.2f} Fut={fut_ltp:.2f} "
                f"VIX={vix_ltp or 'NA'} Trend={momentum_direction}",
                False
            )
            
            # Evaluate each candidate for both CE and PE
            best_signal = None
            best_score = float('inf')
            
            for strike in candidates:
                row = strikes[strike]
                
                # Check CE and PE symbols
                opt_symbols = [
                    (row.ce_symbol, "CE", self.evaluate_call_buy_conditions),
                    (row.pe_symbol, "PE", self.evaluate_put_buy_conditions)
                ]
                
                for opt_symbol_raw, opt_type, evaluator in opt_symbols:
                    opt_symbol = opt_symbol_raw
                    if not opt_symbol:
                        try:
                            opt_symbol = self.build_option_symbol(self.config["symbol_fut"], strike, opt_type)
                        except:
                            continue
                            
                    # Get option quote
                    opt_quote = self.get_quote(opt_symbol)
                    opt_ltp = self._safe_float(opt_quote.get("lp"), 0.0)
                    if opt_ltp <= 0:
                        continue
                    
                    # Calculate IV and Greeks
                    iv = Black76Greeks.implied_vol(
                        fut_ltp, strike, self.config["risk_free_rate"], T, opt_type[0], opt_ltp
                    )
                    greeks = Black76Greeks.greeks_extended(
                        fut_ltp, strike, self.config["risk_free_rate"], iv, T, opt_type[0], premium=opt_ltp
                    )
                    
                    # Evaluate conditions
                    signal = evaluator(
                        spot_ltp=spot_ltp,
                        fut_ltp=fut_ltp,
                        opt_symbol=opt_symbol,
                        opt_quote=opt_quote,
                        chain_row=row,
                        iv=iv,
                        greeks=greeks,
                        vix=vix_ltp,
                        momentum_direction=momentum_direction,
                        momentum_info=momentum_info,
                        expiry_dt=expiry_dt,
                        atr=atr_val,
                        forecast_vol=forecast_vol
                    )
                    
                    if not signal.should_buy:
                        continue
                    
                    # Score: prioritize delta near 0.55/ -0.55
                    target_delta = 0.55 if opt_type == "CE" else -0.55
                    bid, ask = self.get_bid_ask(opt_quote)
                    spread = self.calculate_spread_pct(bid, ask) or 999
                    
                    score = (
                        abs(greeks["delta"] - target_delta) * 10.0 + 
                        spread * 0.5 - 
                        abs(greeks["gamma"]) * 1000.0
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_signal = signal
            
            if best_signal is None:
                self.log("[SENSEX-OPT] [ERROR] No valid BUY candidate found (Call or Put)", False)
                return OptionBuySignal(reasons=["No valid candidate found"])
            
            # Log the winning signal
            self.log("", False)
            self.log("=" * 60, False)
            self.log(f"[OK] BUY SIGNAL: SENSEX {best_signal.option_type} OPTION!", False)
            self.log("=" * 60, False)
            self.log(f"   Symbol      : {best_signal.symbol}", False)
            self.log(f"   Strike      : {best_signal.strike}", False)
            self.log(f"   Entry Price : {best_signal.entry_price}", False)
            self.log(f"   IV          : {best_signal.iv*100:.2f}%", False)
            self.log(f"   Lot Size    : {best_signal.lot_size} (dynamic)", False)
            self.log(f"   Greeks      : Δ={best_signal.greeks.get('delta', 0):.3f} "
                     f"Γ={best_signal.greeks.get('gamma', 0):.6f} "
                     f"Θ/day={best_signal.greeks.get('theta_day', 0):.2f} "
                     f"Vega={best_signal.greeks.get('vega', 0):.4f}", False)
            self.log(f"   PCR         : {best_signal.pcr:.2f}" if best_signal.pcr else "   PCR         : N/A", False)
            self.log(f"   VIX         : {best_signal.vix:.2f}" if best_signal.vix else "   VIX         : N/A", False)
            if best_signal.sl_price and best_signal.tp_price:
                self.log(f"   SL (Dynamic): {best_signal.sl_price:.2f}", False)
                self.log(f"   TP (Dynamic): {best_signal.tp_price:.2f}", False)
            self.log("   Reasons     :", False)
            for r in best_signal.reasons:
                self.log(f"     - {r}", False)
            self.log("=" * 60, False)
            
            return best_signal
            
        except Exception as e:
            self.log(f"[SENSEX-OPT] Analysis error: {e}", False)
            import traceback
            traceback.print_exc()
            return OptionBuySignal(reasons=[f"Analysis error: {str(e)}"])


# =============================================================================
# END OF SENSEX OPTION BUYING MODULE
# =============================================================================


# =============================================================================
# OPTION ORDER MANAGER - Bridge between OptionBuySignal and OrderManager
# =============================================================================
class OptionOrderManager:
    """
    Bridge class that connects SensexOptionBuyer signals to OrderManager.
    
    Flow:
    1. Receive OptionBuySignal from SensexOptionBuyer
    2. Create/update OrderManager for the option symbol
    3. Execute entry via OrderManager._process_entry()
    4. Manage trailing stops via OrderManager._check_trailing_profit()
    5. Execute exits via OrderManager._process_exit()
    """
    
    # Point value for SENSEX options (in Rs per point)
    SENSEX_OPTION_POINT_VALUE = 10  # Each 1 point move = Rs 10 per lot
    
    def __init__(self, trading_bot, log_fn):
        """
        Initialize OptionOrderManager
        
        Args:
            trading_bot: Reference to TradingBot instance
            log_fn: Logging function
        """
        self.bot = trading_bot
        self.log = log_fn
        self.IST = pytz.timezone("Asia/Kolkata")
        
        # Active option trade state
        self.active_option_order_manager = None
        self.active_signal: Optional[OptionBuySignal] = None
        self.in_option_trade = False
        
        # Trade statistics
        self.option_trades_today = 0
        self.option_daily_pnl = 0.0
        self.max_option_trades_per_day = 3
        self.option_daily_loss_limit = 2000  # Rs 2000 max loss
        
        # Cooldown management
        self.last_exit_time = None
        self.cooldown_minutes = 5

    # ---------------------------------------------------------------------
    # V3 (Generic) Option Signal Support
    # ---------------------------------------------------------------------
    def _get_signal_for_current_symbol(self) -> Optional["OptionBuySignal"]:
        """
        Return an OptionBuySignal for the bot's current symbol.
        - If the symbol is a supported FUT (e.g., MCX:NATGASMINI26MARFUT), use the
          V3 GenericGreeksOptionBuyerV2 to select CE/PE from the FUT option chain.
        - Otherwise, fall back to the legacy SensexOptionBuyer.
        """
        # Prefer V3 buyer when available and symbol is routable
        try:
            v3_buyer = getattr(self.bot, "_v3_option_buyer", None)
            if v3_buyer is not None:
                resolved = SymbolRouterV3.resolve(getattr(self.bot, "symbol", ""))
                if resolved is not None and resolved.underlying_key in DEFAULT_CONFIGS_V2:
                    sig_v2 = v3_buyer.analyze(resolved)
                    if sig_v2 is None:
                        return None

                    cfg = DEFAULT_CONFIGS_V2.get(sig_v2.underlying)
                    # Apply per-underlying limits
                    if cfg:
                        self.max_option_trades_per_day = int(getattr(cfg, "max_trades_per_day", self.max_option_trades_per_day))
                        self.option_daily_loss_limit = float(getattr(cfg, "daily_loss_limit", self.option_daily_loss_limit))

                    return self._convert_v2_to_legacy(sig_v2)
        except Exception as e:
            self.log(f"[OPT-V3] Signal error: {e}", False)

        # Legacy fallback (SENSEX-only)
        if getattr(self.bot, "sensex_option_buyer", None) is None:
            return None
        return self.bot.sensex_option_buyer.analyze_and_get_signal()

    def _convert_v2_to_legacy(self, sig_v2: "OptionBuySignalV2") -> "OptionBuySignal":
        """
        Convert OptionBuySignalV2 -> OptionBuySignal so the existing OrderManager
        and OptionOrderManager flow can continue unchanged.
        """
        cfg = DEFAULT_CONFIGS_V2.get(sig_v2.underlying)
        sl_pct = float(getattr(cfg, "default_sl_pct", 0.25)) if cfg else 0.25
        tp_pct = float(getattr(cfg, "default_tp_pct", 0.40)) if cfg else 0.40
        point_value = float(getattr(cfg, "point_value", 1.0)) if cfg else 1.0

        entry = float(sig_v2.entry_price or 0.0)
        sl_price = entry * (1.0 - sl_pct) if entry > 0 else None
        tp_price = entry * (1.0 + tp_pct) if entry > 0 else None

        legacy = OptionBuySignal(
            should_buy=True,
            symbol=sig_v2.option_symbol,
            strike=float(sig_v2.strike or 0.0),
            option_type=sig_v2.option_type,
            reasons=[
                f"V3 momentum={sig_v2.momentum}",
                f"delta={sig_v2.greeks.delta:.2f}",
                f"pcr={sig_v2.pcr:.2f}",
                f"vix={sig_v2.vix:.2f}" if sig_v2.vix is not None else "vix=N/A",
                f"score={sig_v2.score:.2f}",
            ],
            greeks={
                "delta": float(sig_v2.greeks.delta),
                "gamma": float(sig_v2.greeks.gamma),
                "theta_day": float(sig_v2.greeks.theta),
                "vega": float(sig_v2.greeks.vega),
            },
            pcr=float(sig_v2.pcr),
            iv=float(sig_v2.greeks.iv),
            vix=float(sig_v2.vix) if sig_v2.vix is not None else None,
            entry_price=entry,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_points=(entry - sl_price) if (entry and sl_price) else None,
            tp_points=(tp_price - entry) if (entry and tp_price) else None,
            lot_size=int(self.bot.get_dynamic_lot_size(sig_v2.option_symbol)) if hasattr(self.bot, "get_dynamic_lot_size") else 1,
            atr=None,
            debug_info={
                "underlying": sig_v2.underlying,
                "point_value": point_value,
                "max_trades_per_day": int(getattr(cfg, "max_trades_per_day", self.max_option_trades_per_day)) if cfg else self.max_option_trades_per_day,
                "daily_loss_limit": float(getattr(cfg, "daily_loss_limit", self.option_daily_loss_limit)) if cfg else self.option_daily_loss_limit,
                **(sig_v2.meta or {}),
            },
        )
        return legacy

        
    def analyze_and_trade(self) -> Optional[OptionBuySignal]:
        """
        Main flow: Analyze → Select PE/CE → Execute via OrderManager
        
        Returns:
            OptionBuySignal if trade executed, None otherwise
        """
        # Safety checks
        if self._is_trading_halted():
            return None
            
        if self._is_cooldown_active():
            return None
            
        # Check if already in option trade
        if self.in_option_trade and self.active_option_order_manager:
            return self._manage_active_trade()
        
        # Get signal from SensexOptionBuyer
        if not self.bot.sensex_option_buyer:
            self.log("[OPT-MGR] SensexOptionBuyer not initialized", False)
            return None
            
        signal = self._get_signal_for_current_symbol()
        
        if not signal.should_buy:
            return None
            
        # Execute entry
        return self._execute_entry(signal)
    
    def _is_trading_halted(self) -> bool:
        """Check if option trading is halted for the day"""
        today = dt.datetime.now(self.IST).date()
        
        # Reset daily counters if new day
        if not hasattr(self, '_last_reset_date') or self._last_reset_date != today:
            self.option_trades_today = 0
            self.option_daily_pnl = 0.0
            self._last_reset_date = today
            
        # Check daily loss limit
        if self.option_daily_pnl <= -self.option_daily_loss_limit:
            self.log(
                f"[ALERT] [OPT-MGR] Trading HALTED - Daily loss limit reached\n"
                f"  Loss: ₹{self.option_daily_pnl:.0f} / ₹{self.option_daily_loss_limit}",
                False
            )
            return True
            
        # Check max trades
        if self.option_trades_today >= self.max_option_trades_per_day:
            self.log(
                f"[WARNING] [OPT-MGR] Max option trades reached\n"
                f"  Trades: {self.option_trades_today} / {self.max_option_trades_per_day}",
                False
            )
            return True
            
        return False
    
    def _is_cooldown_active(self) -> bool:
        """Check if cooldown period is active"""
        if self.last_exit_time is None:
            return False
            
        elapsed = (dt.datetime.now(self.IST) - self.last_exit_time).total_seconds() / 60
        if elapsed < self.cooldown_minutes:
            self.log(
                f"⏳ [OPT-MGR] Cooldown active: {self.cooldown_minutes - elapsed:.1f}m remaining",
                True
            )
            return True
            
        return False
    
    def _execute_entry(self, signal: OptionBuySignal) -> Optional[OptionBuySignal]:
        """
        Execute option buy entry via OrderManager
        
        Args:
            signal: OptionBuySignal with all trade parameters
            
        Returns:
            OptionBuySignal if entry successful, None otherwise
        """
        self.log("", False)
        self.log("╔" + "═" * 58 + "╗", False)
        self.log("║  [SIGNAL] EXECUTING OPTION BUY ENTRY                            ║", False)
        self.log("╠" + "═" * 58 + "╣", False)
        
        try:
            # Create FyersService for option symbol
            fyers_service = FyersService(
                self.bot.fyers_sdk_instance,
                self.bot.data_paths.get(self.bot.symbol, {}).get('raw_api_log', 'logs/raw_api.log'),
                self.log,
                self.bot.get_websocket_ltp
            )
            
            # Prepare state path for this option symbol
            symbol_clean = signal.symbol.replace(":", "_")
            state_path = f"state/{symbol_clean}_om_state.json"
            event_log = f"logs/{symbol_clean}_events.json"
            report_dir = f"reports_bot/{symbol_clean}"
            
            # Create directories if needed
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            os.makedirs(os.path.dirname(event_log), exist_ok=True)
            os.makedirs(report_dir, exist_ok=True)
            
            # Create OrderManager for this option trade
            self.active_option_order_manager = OrderManager(
                fyers_service=fyers_service,
                symbol=signal.symbol,
                lot_size=signal.lot_size,
                log_fn=self.log,
                state_path=state_path,
                event_log=event_log,
                report_dir=report_dir,
                ai_predictor=self.bot.ai_predictor if hasattr(self.bot, 'ai_predictor') else None,
                bot=self.bot,
                is_option=True,
                point_value=(signal.debug_info.get('point_value', self.SENSEX_OPTION_POINT_VALUE) if getattr(signal, 'debug_info', None) else self.SENSEX_OPTION_POINT_VALUE)
            )
            
            # Configure for options trading (always BUY side)
            om = self.active_option_order_manager
            om.PAPER_TRADING_MODE = False  # Live trading
            
            # Set dynamic targets from signal
            if signal.sl_points and signal.tp_points:
                om.position["trail_start_profit"] = signal.tp_points * 0.3  # Start trailing at 30% of target
                om.position["trail_gap"] = signal.sl_points * 0.5  # Trail gap = half of SL
            
            # Calculate ATR for entry (use signal's ATR or fallback)
            atr = signal.atr or 50.0  # Fallback ATR for options
            
            # Execute entry
            entry_success = om._process_entry(
                side="BUY",  # Options are always BUY
                reason=f"Greek Signal ({signal.option_type}): {', '.join(signal.reasons[:2])}",
                ltp=signal.entry_price,
                atr=atr,
                bar_key=dt.datetime.now(self.IST).strftime("%Y-%m-%d %H:%M"),
                indsP={
                    "delta": signal.greeks.get("delta", 0),
                    "gamma": signal.greeks.get("gamma", 0),
                    "theta_day": signal.greeks.get("theta_day", 0),
                    "vega": signal.greeks.get("vega", 0),
                    "iv": signal.iv,
                    "pcr": signal.pcr,
                    "vix": signal.vix
                }
            )
            
            if entry_success:
                self.in_option_trade = True
                self.active_signal = signal
                self.option_trades_today += 1
                
                # Set stop loss and take profit in position
                om.position["initial_sl"] = signal.sl_price
                om.position["take_profit"] = signal.tp_price
                om.position["option_type"] = signal.option_type
                om.position["greeks"] = signal.greeks
                om._save_state()
                
                self.log(f"║  [OK] Entry SUCCESS: {signal.symbol}                          ", False)
                self.log(f"║  Entry: ₹{signal.entry_price:.2f} | SL: ₹{signal.sl_price:.2f} | TP: ₹{signal.tp_price:.2f}", False)
                self.log(f"║  Lot Size: {signal.lot_size} | Greeks: Δ={signal.greeks.get('delta', 0):.3f}", False)
                self.log("╚" + "═" * 58 + "╝", False)
                
                return signal
            else:
                self.log("║  [ERROR] Entry FAILED - See logs above                          ║", False)
                self.log("╚" + "═" * 58 + "╝", False)
                self.active_option_order_manager = None
                return None
                
        except Exception as e:
            self.log(f"║  [ERROR] Entry ERROR: {str(e)[:40]}        ║", False)
            self.log("╚" + "═" * 58 + "╝", False)
            import traceback
            traceback.print_exc()
            self.active_option_order_manager = None
            return None
    
    def _manage_active_trade(self) -> Optional[OptionBuySignal]:
        """
        Manage active option trade - check trailing stops and exits
        
        Returns:
            OptionBuySignal if still active, None if exited
        """
        if not self.active_option_order_manager:
            return None
            
        om = self.active_option_order_manager
        
        try:
            # Get current LTP for option
            ltp = self.bot.get_websocket_ltp(self.active_signal.symbol)
            if not ltp:
                # Fallback to API quote
                try:
                    resp = self.bot.fyers_sdk_instance.quotes({"symbols": self.active_signal.symbol})
                    if resp.get("s") == "ok" and resp.get("d"):
                        ltp = resp["d"][0].get("v", {}).get("lp")
                except:
                    pass
            
            if not ltp:
                self.log(f"[OPT-MGR] Cannot get LTP for {self.active_signal.symbol}", False)
                return self.active_signal
            
            # Calculate profit
            entry_price = om.position.get("entry_price", self.active_signal.entry_price)
            profit_points = ltp - entry_price
            profit_rupees = profit_points * self.active_signal.lot_size * self.SENSEX_OPTION_POINT_VALUE
            
            # Check stop loss
            sl_price = om.position.get("stop_loss") or self.active_signal.sl_price
            if sl_price and ltp <= sl_price:
                return self._execute_exit(ltp, f"Stop Loss Hit @ ₹{sl_price:.2f}")
            
            # Check take profit
            tp_price = om.position.get("take_profit") or self.active_signal.tp_price
            if tp_price and ltp >= tp_price:
                return self._execute_exit(ltp, f"Take Profit Hit @ ₹{tp_price:.2f}")
            
            # Check trailing profit
            should_exit, exit_reason = om._check_trailing_profit(ltp)
            if should_exit:
                return self._execute_exit(ltp, f"Trailing: {exit_reason}")
            
            # Log status periodically
            self.log(
                f"[OPT-MGR] Active: {self.active_signal.symbol} | "
                f"LTP=₹{ltp:.2f} | Entry=₹{entry_price:.2f} | "
                f"P&L=₹{profit_rupees:.0f} ({profit_points:+.2f}pts)",
                True
            )
            
            return self.active_signal
            
        except Exception as e:
            self.log(f"[OPT-MGR] Error managing trade: {e}", False)
            import traceback
            traceback.print_exc()
            return self.active_signal
    
    def _execute_exit(self, ltp: float, reason: str) -> None:
        """
        Execute option trade exit
        
        Args:
            ltp: Current LTP at exit
            reason: Exit reason for logging
        """
        self.log("", False)
        self.log("╔" + "═" * 58 + "╗", False)
        self.log("║  🔔 EXECUTING OPTION EXIT                                 ║", False)
        self.log("╠" + "═" * 58 + "╣", False)
        
        try:
            om = self.active_option_order_manager
            entry_price = om.position.get("entry_price", self.active_signal.entry_price)
            
            # Calculate P&L
            profit_points = ltp - entry_price
            profit_rupees = profit_points * self.active_signal.lot_size * self.SENSEX_OPTION_POINT_VALUE
            
            # Execute exit
            om._process_exit(reason, ltp)
            
            # Update daily P&L
            self.option_daily_pnl += profit_rupees
            
            # Log result
            pnl_emoji = "💰" if profit_rupees > 0 else "📉"
            self.log(f"║  {pnl_emoji} Exit: {self.active_signal.symbol}                                ║", False)
            self.log(f"║  Entry: ₹{entry_price:.2f} → Exit: ₹{ltp:.2f}                    ║", False)
            self.log(f"║  P&L: ₹{profit_rupees:+.0f} ({profit_points:+.2f} pts)                       ║", False)
            self.log(f"║  Reason: {reason[:45]}                                 ║", False)
            self.log(f"║  Daily P&L: ₹{self.option_daily_pnl:+.0f}                                   ║", False)
            self.log("╚" + "═" * 58 + "╝", False)
            
            # Reset state
            self.in_option_trade = False
            self.active_signal = None
            self.active_option_order_manager = None
            self.last_exit_time = dt.datetime.now(self.IST)
            
        except Exception as e:
            self.log(f"║  [ERROR] Exit ERROR: {str(e)[:40]}         ║", False)
            self.log("╚" + "═" * 58 + "╝", False)
            import traceback
            traceback.print_exc()
            
        return None
    
    def force_exit(self, reason: str = "Manual Exit") -> None:
        """
        Force exit active option position
        
        Args:
            reason: Exit reason
        """
        if not self.in_option_trade:
            self.log("[OPT-MGR] No active option trade to exit", False)
            return
            
        try:
            ltp = self.bot.get_websocket_ltp(self.active_signal.symbol)
            if not ltp:
                # Fallback
                ltp = self.active_signal.entry_price
                self.log("[OPT-MGR] Using entry price as exit LTP (no market data)", False)
                
            self._execute_exit(ltp, reason)
        except Exception as e:
            self.log(f"[OPT-MGR] Force exit error: {e}", False)
    
    def get_status(self) -> Dict:
        """Get current option trading status"""
        return {
            "in_option_trade": self.in_option_trade,
            "active_symbol": self.active_signal.symbol if self.active_signal else None,
            "option_type": self.active_signal.option_type if self.active_signal else None,
            "entry_price": self.active_signal.entry_price if self.active_signal else None,
            "lot_size": self.active_signal.lot_size if self.active_signal else None,
            "trades_today": self.option_trades_today,
            "daily_pnl": self.option_daily_pnl,
            "last_exit_time": self.last_exit_time.isoformat() if self.last_exit_time else None
        }


def _build_ai_cpr_features(ltp: float, indicators: dict, pivot_data: dict, ohlc_df=None) -> np.ndarray:
    """
    FIXED: Matches training data exactly
    Total: 30 features (18 technical + 12 candles)

    [WARNING] CRITICAL: Feature order MUST match train_ai_model.py FEATURE_COLUMNS
    """
    logger = logging.getLogger(__name__)
    features = []

    # [OK] VALIDATE PIVOT DATA FIRST
    if not pivot_data or not isinstance(pivot_data, dict):
        logger.error("[AI-CPR] [ERROR] pivot_data is invalid or empty")
        return np.zeros((1, 30))  # Return zeros instead of crashing

    # Get TC and BC with validation
    tc = pivot_data.get("TC")
    bc = pivot_data.get("BC")

    if tc is None or bc is None:
        logger.error(f"[AI-CPR] [ERROR] Missing CPR levels: TC={tc}, BC={bc}")
        # Try to get from nested structure
        if "cpr_levels" in pivot_data:
            tc = pivot_data["cpr_levels"].get("TC")
            bc = pivot_data["cpr_levels"].get("BC")
            logger.info(f"[AI-CPR] Found in nested structure: TC={tc}, BC={bc}")

    if tc is None or bc is None:
        logger.error("[AI-CPR] [ERROR] Still missing TC/BC - returning zero features")
        return np.zeros((1, 30))

    def _get_ind_value(key, default_value=0.0):
        """Safely extract indicator value"""
        val = indicators.get(key, default_value)
        try:
            val = float(val)
            return val if np.isfinite(val) else default_value
        except (ValueError, TypeError):
            return default_value

    if ltp is None or ltp <= 0:
        ltp = _get_ind_value("close", 1.0)
        logger.warning(f"[AI-CPR] Invalid LTP, using {ltp}")

    # ==========================================
    # SECTION 1: Trend Indicators (5 features)
    # ==========================================
    features.append(_get_ind_value("ema_5", ltp))
    features.append(_get_ind_value("ema_9", ltp))
    features.append(_get_ind_value("ema_21", ltp))
    features.append(_get_ind_value("ema_50", ltp))
    features.append(_get_ind_value("ema_200", ltp))

    # ==========================================
    # SECTION 2: Momentum Indicators (3 features)
    # ==========================================
    features.append(_get_ind_value("rsi", 50.0))
    features.append(_get_ind_value("momentum", 0.0))
    features.append(_get_ind_value("roc_10", 0.0))

    # ==========================================
    # SECTION 3: Volatility Indicators (2 features)
    # ==========================================
    features.append(_get_ind_value("ATR", 1.0))
    features.append(_get_ind_value("bb_bandwidth", 0.01))

    # ==========================================
    # SECTION 4: Volume Indicators (2 features)
    # ==========================================
    features.append(_get_ind_value("volume_ratio", 1.0))
    features.append(_get_ind_value("efi", 0.0))

    # ==========================================
    # SECTION 5: Directional Indicators (3 features)
    # ==========================================
    features.append(_get_ind_value("adx", 25.0))
    features.append(_get_ind_value("plus_di", 0.0))
    features.append(_get_ind_value("minus_di", 0.0))

    # ==========================================
    # SECTION 6: CPR-Specific Features (3 features)
    # 🔥 FIX: Use EXACT same names and formulas as training
    # ==========================================
    tc = pivot_data.get("TC", None)
    bc = pivot_data.get("BC", None)

    # Feature 16: price_to_tc_ratio
    if tc and tc > 0:
        price_to_tc_ratio = (ltp - tc) / tc
        features.append(price_to_tc_ratio)
    else:
        features.append(0.0)
        logger.warning("[AI-CPR] TC not available")

    # Feature 17: price_to_bc_ratio
    if bc and bc > 0:
        price_to_bc_ratio = (ltp - bc) / bc
        features.append(price_to_bc_ratio)
    else:
        features.append(0.0)
        logger.warning("[AI-CPR] BC not available")

    # Feature 18: cpr_width_ratio
    # 🔥 FIX: Match training formula exactly: (TC - BC) / Close
    if tc and bc and ltp > 0:
        cpr_width_ratio = (tc - bc) / ltp
        features.append(cpr_width_ratio)
    else:
        features.append(0.0)
        logger.warning("[AI-CPR] CPR width calculation failed")

    # ==========================================
    # SECTION 7: Candle Pattern Features (12 features)
    # ==========================================
    if ohlc_df is not None and not ohlc_df.empty and len(ohlc_df) >= 3:
        try:
            from kiran_aicpr_best import extract_candle_features
            candle_features = extract_candle_features(ohlc_df)

            # Validate candle features
            if len(candle_features) != 12:
                logger.error(f"[AI-CPR] Expected 12 candle features, got {len(candle_features)}")
                candle_features = [0] * 12

            # Log for debugging
            logger.debug(
                f"[AI-CANDLE] Body={candle_features[0]:.1f}%, "
                f"Dir={candle_features[1]}, UWick={candle_features[2]:.1f}%, "
                f"LWick={candle_features[3]:.1f}%, Vol={candle_features[11]:.2f}x"
            )

        except Exception as e:
            logger.error(f"[AI-CPR] Candle feature extraction failed: {e}")
            candle_features = [0] * 12
    else:
        candle_features = [0] * 12
        if ohlc_df is None or ohlc_df.empty:
            logger.warning("[AI-CPR] No OHLC data for candle features")
        else:
            logger.warning(f"[AI-CPR] Insufficient candles: {len(ohlc_df)}/3 required")

    features.extend(candle_features)

    # ==========================================
    # VALIDATION
    # ==========================================
    expected_count = 30
    actual_count = len(features)

    if actual_count != expected_count:
        logger.error(
            f"[AI-CPR] [ERROR] FEATURE COUNT MISMATCH! "
            f"Expected={expected_count}, Got={actual_count}"
        )

        # Pad or truncate to fix
        if actual_count < expected_count:
            padding = [0.0] * (expected_count - actual_count)
            features.extend(padding)
            logger.warning(f"[AI-CPR] Padded {len(padding)} zeros")
        else:
            features = features[:expected_count]
            logger.warning(f"[AI-CPR] Truncated to {expected_count}")

    # Check for NaN/Inf
    features_array = np.array(features, dtype=float)
    if np.isnan(features_array).any():
        nan_count = np.isnan(features_array).sum()
        logger.error(f"[AI-CPR] [ERROR] Found {nan_count} NaN values - replacing with 0")
        features_array = np.nan_to_num(features_array, nan=0.0)

    if np.isinf(features_array).any():
        inf_count = np.isinf(features_array).sum()
        logger.error(f"[AI-CPR] [ERROR] Found {inf_count} Inf values - replacing with 0")
        features_array = np.nan_to_num(features_array, posinf=0.0, neginf=0.0)

    logger.debug(f"[AI-CPR] [OK] Built {len(features_array)} valid features")

    return features_array.reshape(1, -1)


# ==========================================
# FEATURE ORDER REFERENCE (for debugging)
# ==========================================
FEATURE_ORDER = [
    # Trend (5)
    'ema_5', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
    # Momentum (3)
    'rsi', 'momentum', 'roc_10',
    # Volatility (2)
    'atr', 'bb_bandwidth',
    # Volume (2)
    'volume_ratio', 'efi',
    # Directional (3)
    'adx', 'plus_di', 'minus_di',
    # CPR (3) - MUST match training names
    'price_to_tc_ratio', 'price_to_bc_ratio', 'cpr_width_ratio',
    # Candles (12)
    'body_pct', 'body_direction', 'upper_wick_pct', 'lower_wick_pct',
    'engulfing_score', 'reversal_pattern', 'marubozu_score',
    'momentum_3', 'range_expansion', 'gap_score',
    'close_position', 'vol_ratio'
]


def validate_features(features_array: np.ndarray) -> dict:
    """Validate features match training expectations"""
    issues = []

    if features_array.shape[1] != 30:
        issues.append(f"Wrong feature count: {features_array.shape[1]}/30")

    if np.isnan(features_array).any():
        issues.append(f"Contains {np.isnan(features_array).sum()} NaN values")

    if np.isinf(features_array).any():
        issues.append(f"Contains {np.isinf(features_array).sum()} Inf values")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "feature_names": FEATURE_ORDER
    }


def extract_candle_features(ohlc_df):
    """
    Extract 12 powerful candle-based features
    These are SEMI-LEADING (catch reversals early)
    """

    if ohlc_df is None or len(ohlc_df) < 3:
        return [0] * 12  # Return zeros if not enough data

    latest = ohlc_df.iloc[-1]
    prev = ohlc_df.iloc[-2]
    prev2 = ohlc_df.iloc[-3] if len(ohlc_df) >= 3 else prev

    features = []

    # ==========================================
    # Feature 1-2: Body Size (momentum strength)
    # ==========================================
    body = abs(latest['Close'] - latest['Open'])
    total_range = latest['High'] - latest['Low']
    body_pct = (body / total_range * 100) if total_range > 0 else 0

    features.append(body_pct)  # Feature 1: Body strength

    # Is it bullish or bearish?
    body_direction = 1 if latest['Close'] > latest['Open'] else -1
    features.append(body_direction)  # Feature 2: Direction

    # ==========================================
    # Feature 3-4: Wick Analysis (rejection)
    # ==========================================
    upper_wick = latest['High'] - max(latest['Open'], latest['Close'])
    lower_wick = min(latest['Open'], latest['Close']) - latest['Low']

    upper_wick_pct = (upper_wick / total_range * 100) if total_range > 0 else 0
    lower_wick_pct = (lower_wick / total_range * 100) if total_range > 0 else 0

    features.append(upper_wick_pct)  # Feature 3: Upper rejection
    features.append(lower_wick_pct)  # Feature 4: Lower rejection

    # ==========================================
    # Feature 5: Engulfing Pattern
    # ==========================================
    bullish_engulfing = (
            (prev['Close'] < prev['Open']) and  # Prev was bearish
            (latest['Close'] > latest['Open']) and  # Current is bullish
            (latest['Open'] <= prev['Close']) and
            (latest['Close'] >= prev['Open'])
    )

    bearish_engulfing = (
            (prev['Close'] > prev['Open']) and
            (latest['Close'] < latest['Open']) and
            (latest['Open'] >= prev['Close']) and
            (latest['Close'] <= prev['Open'])
    )

    engulfing_score = 1 if bullish_engulfing else -1 if bearish_engulfing else 0
    features.append(engulfing_score)  # Feature 5

    # ==========================================
    # Feature 6: Hammer/Shooting Star
    # ==========================================
    is_hammer = (
            body > 0 and
            lower_wick >= 2 * body and
            upper_wick <= 0.1 * total_range
    )

    is_shooting_star = (
            body > 0 and
            upper_wick >= 2 * body and
            lower_wick <= 0.1 * total_range
    )

    reversal_pattern = 1 if is_hammer else -1 if is_shooting_star else 0
    features.append(reversal_pattern)  # Feature 6

    # ==========================================
    # Feature 7: Marubozu (strong momentum)
    # ==========================================
    buffer = 0.15 * total_range  # 15% tolerance

    bullish_marubozu = (
            latest['Close'] > latest['Open'] and
            (latest['High'] - latest['Close']) <= buffer and
            (latest['Open'] - latest['Low']) <= buffer
    )

    bearish_marubozu = (
            latest['Close'] < latest['Open'] and
            (latest['High'] - latest['Open']) <= buffer and
            (latest['Close'] - latest['Low']) <= buffer
    )

    marubozu_score = 1 if bullish_marubozu else -1 if bearish_marubozu else 0
    features.append(marubozu_score)  # Feature 7

    # ==========================================
    # Feature 8: 3-Candle Momentum
    # ==========================================
    three_bull = (
            prev2['Close'] > prev2['Open'] and
            prev['Close'] > prev['Open'] and
            latest['Close'] > latest['Open']
    )

    three_bear = (
            prev2['Close'] < prev2['Open'] and
            prev['Close'] < prev['Open'] and
            latest['Close'] < latest['Open']
    )

    momentum_3 = 1 if three_bull else -1 if three_bear else 0
    features.append(momentum_3)  # Feature 8

    # ==========================================
    # Feature 9: Range Expansion
    # ==========================================
    avg_range = ohlc_df['High'].rolling(5).max() - ohlc_df['Low'].rolling(5).min()
    avg_range_val = avg_range.iloc[-1] if len(avg_range) > 0 else total_range

    range_expansion = (total_range / avg_range_val) if avg_range_val > 0 else 1.0
    features.append(range_expansion)  # Feature 9

    # ==========================================
    # Feature 10: Gap Detection
    # ==========================================
    gap_up = (latest['Open'] > prev['High'])
    gap_down = (latest['Open'] < prev['Low'])

    gap_score = 1 if gap_up else -1 if gap_down else 0
    features.append(gap_score)  # Feature 10

    # ==========================================
    # Feature 11: Close Position in Range
    # ==========================================
    close_position = (
            (latest['Close'] - latest['Low']) / total_range * 100
    ) if total_range > 0 else 50

    features.append(close_position)  # Feature 11
    # 100 = closed at high (bullish)
    # 0 = closed at low (bearish)
    # 50 = closed mid-range (neutral)

    # ==========================================
    # Feature 12: Volume Confirmation
    # ==========================================
    if 'Volume' in ohlc_df.columns:
        vol_avg = ohlc_df['Volume'].rolling(5).mean().iloc[-1]
        vol_ratio = (latest['Volume'] / vol_avg) if vol_avg > 0 else 1.0
    else:
        vol_ratio = 1.0

    features.append(vol_ratio)  # Feature 12

    return features


def log_candle_features_detail(candle_features, ohlc_df, logger):
    """
    Log detailed candle pattern analysis for debugging
    """
    if not candle_features or len(candle_features) != 12:
        logger("[CANDLE-AI] No candle features to log", True)
        return

    latest = ohlc_df.iloc[-1] if ohlc_df is not None and not ohlc_df.empty else None
    if latest is None:
        return

    # Unpack the 12 features
    (body_pct, body_direction, upper_wick_pct, lower_wick_pct,
     engulfing_score, reversal_pattern, marubozu_score, momentum_3,
     range_expansion, gap_score, close_position, vol_ratio) = candle_features

    # Build human-readable interpretation
    patterns_detected = []

    # Feature 1-2: Body Analysis
    if body_pct > 60:
        strength = "STRONG" if body_pct > 80 else "MODERATE"
        direction = "BULLISH" if body_direction > 0 else "BEARISH"
        patterns_detected.append(f"{strength} {direction} body ({body_pct:.1f}%)")

    # Feature 3-4: Wick Analysis (Rejection)
    if upper_wick_pct > 30:
        patterns_detected.append(f"Upper rejection ({upper_wick_pct:.1f}% wick)")
    if lower_wick_pct > 30:
        patterns_detected.append(f"Lower support ({lower_wick_pct:.1f}% wick)")

    # Feature 5: Engulfing
    if engulfing_score == 1:
        patterns_detected.append("🟢 BULLISH ENGULFING")
    elif engulfing_score == -1:
        patterns_detected.append("🔴 BEARISH ENGULFING")

    # Feature 6: Reversal Patterns
    if reversal_pattern == 1:
        patterns_detected.append("🔨 HAMMER (bullish reversal)")
    elif reversal_pattern == -1:
        patterns_detected.append("⭐ SHOOTING STAR (bearish reversal)")

    # Feature 7: Marubozu (Strong Momentum)
    if marubozu_score == 1:
        patterns_detected.append("📈 BULLISH MARUBOZU (strong momentum)")
    elif marubozu_score == -1:
        patterns_detected.append("📉 BEARISH MARUBOZU (strong momentum)")

    # Feature 8: 3-Candle Momentum
    if momentum_3 == 1:
        patterns_detected.append("[START] 3-CANDLE BULLISH MOMENTUM")
    elif momentum_3 == -1:
        patterns_detected.append("🔻 3-CANDLE BEARISH MOMENTUM")

    # Feature 9: Range Expansion
    if range_expansion > 1.5:
        patterns_detected.append(f"[DATA] RANGE EXPANSION ({range_expansion:.2f}x)")

    # Feature 10: Gap Detection
    if gap_score == 1:
        patterns_detected.append("⬆️ GAP UP")
    elif gap_score == -1:
        patterns_detected.append("⬇️ GAP DOWN")

    # Feature 11: Close Position
    if close_position > 80:
        patterns_detected.append(f"Strong close near high ({close_position:.1f}%)")
    elif close_position < 20:
        patterns_detected.append(f"Weak close near low ({close_position:.1f}%)")

    # Feature 12: Volume Confirmation
    if vol_ratio > 1.5:
        patterns_detected.append(f"💪 HIGH VOLUME ({vol_ratio:.2f}x)")
    elif vol_ratio < 0.7:
        patterns_detected.append(f"[WARNING] LOW VOLUME ({vol_ratio:.2f}x)")

    # Overall Signal
    bullish_count = sum([
        body_direction > 0,
        engulfing_score > 0,
        reversal_pattern > 0,
        marubozu_score > 0,
        momentum_3 > 0,
        gap_score > 0,
        close_position > 70
    ])

    bearish_count = sum([
        body_direction < 0,
        engulfing_score < 0,
        reversal_pattern < 0,
        marubozu_score < 0,
        momentum_3 < 0,
        gap_score < 0,
        close_position < 30
    ])

    overall_signal = "BULLISH" if bullish_count > bearish_count else "BEARISH" if bearish_count > bullish_count else "NEUTRAL"
    signal_strength = max(bullish_count, bearish_count)

    # Format the log message
    logger(
        f"\n{'=' * 60}\n"
        f"🕯️ CANDLE PATTERN ANALYSIS\n"
        f"{'=' * 60}\n"
        f"Time: {latest.name if hasattr(latest, 'name') else 'N/A'}\n"
        f"OHLC: O:{latest['Open']:.2f} H:{latest['High']:.2f} "
        f"L:{latest['Low']:.2f} C:{latest['Close']:.2f}\n"
        f"\n"
        f"PATTERNS DETECTED:\n" +
        ("\n".join(f"  • {p}" for p in patterns_detected) if patterns_detected else "  • No significant patterns\n") +
        f"\n\n"
        f"OVERALL SIGNAL: {overall_signal} (strength: {signal_strength}/7)\n"
        f"  Bullish signals: {bullish_count}\n"
        f"  Bearish signals: {bearish_count}\n"
        f"{'=' * 60}",
        False
    )

    return {
        "patterns": patterns_detected,
        "overall_signal": overall_signal,
        "signal_strength": signal_strength,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count
    }

# ───────────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────────

def last(x):
    """
    Extract the most recent scalar value from an indicator (Series or scalar).
    Crucial to prevent 'truth value of a Series is ambiguous' errors.
    """
    if x is None:
        return 0.0
    if isinstance(x, pd.Series):
        return float(x.iloc[-1]) if not x.empty else 0.0
    try:
        return float(x)
    except:
        return 0.0

def call_with_rate_limit_retry(api_func, *args, max_retries=5, **kwargs):
    delay = 5
    for attempt in range(max_retries):
        resp = api_func(*args, **kwargs)
        if isinstance(resp, dict) and (resp.get("code") == 429 or "limit" in str(resp.get("message", "")).lower()):
            print(f"[RATE LIMIT] Rate limit hit. Sleeping for {delay} seconds...")
            time.sleep(delay)
            delay *= 2
        else:
            return resp
    print("[RATE LIMIT] Max retries reached. Skipping this request.")
    return None

def robust_load_json(path, logger_func=print, default=None, debug_only=True):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger_func(f"[DEBUG] Empty JSON: {path}", debug_only)
                    return default
                return json.loads(content)
        logger_func(f"[DEBUG] JSON not found: {path}", debug_only)
        return default
    except json.JSONDecodeError as e:
        logger_func(f"[ERROR] Load JSON decode {path}: {e}", False)
        return default
    except Exception as e:
        logger_func(f"[ERROR] Load JSON {path}: {e}", False)
        return default

def robust_save_json(data, path, logger_func=print, debug_only=True):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable = convert_to_serializable(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        if not debug_only:
            logger_func(f"[SAVE JSON] Wrote {path}")
        else:
            logger_func(f"[SAVE JSON] Wrote {path} (debug)")
        return True

    except Exception as e:
        # Fallback: try using json dump with default=str to avoid crashes
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str, ensure_ascii=False, indent=2)
            logger_func(f"[SAVE JSON - FALLBACK] Wrote {path} using default=str")
            return True
        except Exception as ex:
            logger_func(f"[ERROR] Save JSON {path}: {ex}")
            return False

def convert_to_serializable(obj):
    if isinstance(obj, (bool, np.bool_)): return bool(obj)
    if isinstance(obj, (int, np.integer)): return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (dt.datetime, dt.date, pd.Timestamp)): return obj.isoformat()
    if isinstance(obj, (pd.Series, np.ndarray)):
        return [convert_to_serializable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    if pd.isna(obj): return None
    try:
        return str(obj)
    except Exception as e:
        return None

def convert_dict_to_serializable(d):
    return {str(k): convert_to_serializable(v) for k, v in d.items()}

# ───────────────────────────────────────────────────────────────────────────────
# Market data utils
# ───────────────────────────────────────────────────────────────────────────────

def get_tick_size(fyers_client, symbol):
    """Fetch tick size via fyers.quotes; default to 0.05 on error."""
    try:
        resp = fyers_client.quotes({"symbols": symbol})
        if resp.get("s") == "ok":
            for itm in resp.get("d", []):
                if itm.get("n") == symbol:
                    ts = itm.get("v", {}).get("tick_size")
                    if ts is not None:
                        return float(ts)
        print(f"[WARN] Tick size fetch failed for {symbol}: {resp}")
    except Exception as e:
        print(f"[ERROR] Tick size {symbol}: {e}")
    return 0.05

def vwap(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("VWAP requires DatetimeIndex")
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['CumVol'] = df.groupby(df.index.date)['Volume'].cumsum()
    df['CumPV']  = (tp * df['Volume']).groupby(df.index.date).cumsum()
    df['VWAP']   = df['CumPV'] / df['CumVol']
    df.drop(['CumVol', 'CumPV'], axis=1, inplace=True)
    return df

def adx_efi_mom_trade_signal(df: pd.DataFrame, symbol: str):
    """Calculate ADX, EFI, Momentum, RSI and generate relaxed trade signal."""
    try:
        if df is None or df.empty or len(df) < 14:
            return ("NO TRADE",) + (0.0,)*6  # Use 0.0 instead of None to avoid downstream issues
        high, low, close = df['High'], df['Low'], df['Close']

        plus_dm  = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)
        tr_cols  = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1)
        tr = tr_cols.max(axis=1)
        atr14 = tr.rolling(14).mean()

        plus_di  = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        dx       = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx      = dx.rolling(14).mean().iloc[-1]
        di_p, di_m = plus_di.iloc[-1], minus_di.iloc[-1]
        mom    = close.pct_change(10).iloc[-1] * 100 if len(close) >= 10 else None
        efi    = ((close.diff() * df['Volume']).rolling(13).mean()).iloc[-1]
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = -delta.clip(upper=0).rolling(14).mean()
        rs_v   = gain / loss.replace(0, np.nan)
        rsi_v  = (100 - (100/(1+rs_v))).iloc[-1]

        sig = "NO TRADE"
        if adx > 20:
            if di_p > di_m and efi > 0 and mom > 0:
                sig = "BUY"
            elif di_m > di_p and efi < 0 and mom < 0:
                sig = "SELL"

        return sig, float(adx), float(di_p), float(di_m), float(mom), float(efi), float(rsi_v)
    except Exception as e:
        print(f"[ERROR] ADX/EFI/MOM {symbol}: {e}")
        return ("NO TRADE",) + (None,)*6

def fibonacci_retracement(df: pd.DataFrame, period=20, levels=None, logger=print,
                          trend_period=5, proximity_threshold=0.01):
    """Compute Fib levels, trend, behavior & confidence predictions."""
    levels = levels or [0, .236, .382, .5, .618, .786, 1.0]
    if not isinstance(df, pd.DataFrame):
        logger("Fib: df must be DataFrame"); return {}
    for c in ['High','Low','Close','Open']:
        if c not in df.columns:
            logger(f"Fib: missing {c}"); return {}
    if len(df) < period:
        logger(f"Fib: need {period} rows, got {len(df)}"); return {}

    window = df.rolling(window=period)
    high = window['High'].max().iloc[-1]
    low  = window['Low'].min().iloc[-1]
    if pd.isna(high) or pd.isna(low) or high <= low:
        logger("Fib: invalid high/low"); return {}

    diff = high - low
    fibs = {f"Fib_{lvl*100:.1f}%": high - lvl*diff for lvl in levels}

    df2 = df.copy()
    df2['EMA5']  = df2['Close'].ewm(span=trend_period).mean()
    df2['EMA21'] = df2['Close'].ewm(span=21).mean()
    latest = df2.iloc[-1]
    trend = ('Bullish' if latest['EMA5'] > latest['EMA21']
             else 'Bearish' if latest['EMA5'] < latest['EMA21']
             else 'Neutral')

    threshold = diff * proximity_threshold
    respect = {}
    for k, v in fibs.items():
        near = df2[(df2['Low'].between(v-threshold, v+threshold)) |
                   (df2['High'].between(v-threshold, v+threshold))]
        respect[k] = min(len(near)/10, 1.0)

    beh, preds = {}, {}
    price = latest['Close']
    for k, v in fibs.items():
        dist = abs(price - v)/diff
        prox_score = max(0, 1-dist*5)
        resp_score = respect[k]
        if trend == 'Bullish':
            beh[k] = f"Likely support; bounce if reaches {v:.2f}"
            sc = (prox_score*0.5 + resp_score*0.3 + 0.2)*0.9
            rc = (1-prox_score)*0.3
        elif trend == 'Bearish':
            beh[k] = f"Likely resistance; reverse at {v:.2f}"
            sc = (1-prox_score)*0.3
            rc = (prox_score*0.5 + resp_score*0.3 + 0.2)*0.9
        else:
            beh[k] = f"Neutral; watch {v:.2f}"
            sc = (prox_score*0.4 + resp_score*0.2)*0.6
            rc = sc

        preds[k] = {
            'support_confidence': min(max(sc, 0), 1),
            'resistance_confidence': min(max(rc, 0), 1)
        }

    return {'levels': fibs, 'trend': trend, 'behavior': beh, 'predictions': preds}

def rsi_divergence(df: pd.DataFrame, rsi_period=14, lookback=5):
    df2 = df.copy()
    if len(df2) < rsi_period + lookback:
        return False, False

    delta = df2['Close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.rolling(rsi_period).mean()
    avg_l = loss.rolling(rsi_period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df2['RSI'] = 100 - (100/(1+rs))

    df2['Ph'] = df2['High'].rolling(lookback).max()
    df2['Pl'] = df2['Low'].rolling(lookback).min()
    df2['Rh'] = df2['RSI'].rolling(lookback).max()
    df2['Rl'] = df2['RSI'].rolling(lookback).min()

    bullish = ((df2['Low'].iloc[-1] <= df2['Pl'].shift(1).iloc[-1]) and
               (df2['RSI'].iloc[-1] > df2['Rl'].shift(1).iloc[-1]))
    bearish = ((df2['High'].iloc[-1] >= df2['Ph'].shift(1).iloc[-1]) and
               (df2['RSI'].iloc[-1] < df2['Rh'].shift(1).iloc[-1]))
    return bool(bullish), bool(bearish)

def supertrend(df: pd.DataFrame, period=7, multiplier=3, ema5=None, ema21=None, super_guppy=None):
    df2 = df.copy()
    df2['ATR'] = atr(df2, period)
    hl2 = (df2['High'] + df2['Low']) / 2
    df2['BasicUp']   = hl2 + multiplier * df2['ATR']
    df2['BasicDown'] = hl2 - multiplier * df2['ATR']
    df2['FinalUp']   = df2['BasicUp'].copy()
    df2['FinalDown'] = df2['BasicDown'].copy()
    df2['Strend']    = np.nan
    df2['Trend']     = 0

    ind = df2.index
    for i in range(period, len(df2)):
        prev_idx, cur_idx = i-1, i
        # Use .iloc for scalar extraction to avoid Series ambiguity
        if df2['Close'].iloc[prev_idx] <= df2['FinalUp'].iloc[prev_idx]:
            df2.iloc[cur_idx, df2.columns.get_loc('FinalUp')] = min(df2['BasicUp'].iloc[cur_idx], df2['FinalUp'].iloc[prev_idx])
        else:
            df2.iloc[cur_idx, df2.columns.get_loc('FinalUp')] = df2['BasicUp'].iloc[cur_idx]
            
        if df2['Close'].iloc[prev_idx] >= df2['FinalDown'].iloc[prev_idx]:
            df2.iloc[cur_idx, df2.columns.get_loc('FinalDown')] = max(df2['BasicDown'].iloc[cur_idx], df2['FinalDown'].iloc[prev_idx])
        else:
            df2.iloc[cur_idx, df2.columns.get_loc('FinalDown')] = df2['BasicDown'].iloc[cur_idx]

    flip = None
    for i in range(period, len(df2)):
        prev_idx, cur_idx = i-1, i
        if df2['Close'].iloc[prev_idx] <= df2['FinalUp'].iloc[prev_idx] and df2['Close'].iloc[cur_idx] > df2['FinalUp'].iloc[cur_idx]:
            df2.iloc[cur_idx, df2.columns.get_loc('Strend')], df2.iloc[cur_idx, df2.columns.get_loc('Trend')] = df2['FinalDown'].iloc[cur_idx], 1
            flip = i; break
        if df2['Close'].iloc[prev_idx] >= df2['FinalDown'].iloc[prev_idx] and df2['Close'].iloc[cur_idx] < df2['FinalDown'].iloc[cur_idx]:
            df2.iloc[cur_idx, df2.columns.get_loc('Strend')], df2.iloc[cur_idx, df2.columns.get_loc('Trend')] = df2['FinalUp'].iloc[cur_idx], -1
            flip = i; break
    if flip is None:
        flip = period

    for i in range(flip+1, len(df2)):
        prev_idx, cur_idx = i-1, i
        # CRITICAL FIX: Use .iloc for scalar comparisons to prevent ambiguity error
        if df2['Strend'].iloc[prev_idx] == df2['FinalUp'].iloc[prev_idx]:
            if df2['Close'].iloc[cur_idx] <= df2['FinalUp'].iloc[cur_idx]:
                df2.iloc[cur_idx, df2.columns.get_loc('Strend')], df2.iloc[cur_idx, df2.columns.get_loc('Trend')] = df2['FinalUp'].iloc[cur_idx], -1
            else:
                df2.iloc[cur_idx, df2.columns.get_loc('Strend')], df2.iloc[cur_idx, df2.columns.get_loc('Trend')] = df2['FinalDown'].iloc[cur_idx], 1
        else:
            if df2['Close'].iloc[cur_idx] >= df2['FinalDown'].iloc[cur_idx]:
                df2.iloc[cur_idx, df2.columns.get_loc('Strend')], df2.iloc[cur_idx, df2.columns.get_loc('Trend')] = df2['FinalDown'].iloc[cur_idx], 1
            else:
                df2.iloc[cur_idx, df2.columns.get_loc('Strend')], df2.iloc[cur_idx, df2.columns.get_loc('Trend')] = df2['FinalUp'].iloc[cur_idx], -1

    return df2['Strend'], df2['FinalUp'], df2['FinalDown'], df2['Trend']

# ───────────────────────────────────────────────────────────────────────────────
# EMA50/200 crossover helper
# ───────────────────────────────────────────────────────────────────────────────

class EMA50_200:
    def __init__(self, fyers_client, ticker, interval="30", duration=60):
        self.fyers    = fyers_client
        self.ticker   = ticker
        self.interval = interval
        self.duration = min(duration, 60)
        self.df       = self._fetch_ohlc()

        if not self.df.empty:
            self._validate()
            self._indicators()
            self._signals()
            self._crossovers()

    def _fetch_ohlc(self):
        try:
            today = dt.date.today()
            frm   = (today - dt.timedelta(days=self.duration)).strftime("%Y-%m-%d")
            to    = today.strftime("%Y-%m-%d")
            payload = {
                "symbol": self.ticker,
                "resolution": self.interval,
                "date_format": "1",
                "range_from": frm,
                "range_to": to,
                "cont_flag":"1"
            }
            resp = self.fyers.history(data=payload)
            candles = resp.get("candles", [])
            if not candles:
                print(f"[WARN] No data for {self.ticker}")
                return pd.DataFrame()

            df = pd.DataFrame(candles, columns=["Ts","Open","High","Low","Close","Volume"])
            df["Timestamp"] = pd.to_datetime(df["Ts"], unit="s", utc=True).dt.tz_convert(IST)
            df.set_index("Timestamp", inplace=True)
            return df.sort_index()
        except Exception as e:
            print(f"[ERROR] Fetch OHLC {self.ticker}: {e}")
            return pd.DataFrame()

    def _validate(self):
        missing = [c for c in ['Open','High','Low','Close','Volume'] if c not in self.df]
        if missing:
            raise ValueError(f"Missing cols {missing}")
        if len(self.df) < 200:
            print(f"[WARN] Only {len(self.df)} rows; MA200 may be unreliable")

    def _indicators(self):
        self.df['MA_50']  = self.df['Close'].rolling(50,1).mean()
        self.df['MA_200'] = self.df['Close'].rolling(200,1).mean()

    def _signals(self):
        self.df['Signal']     = np.where(self.df['Close'] > self.df['MA_200'], 'BUY', 'SELL')
        self.df['Distance_%'] = (self.df['Close'] - self.df['MA_200']) / self.df['MA_200'] * 100

    def _crossovers(self):
        self.df['Above'] = self.df['MA_50'] > self.df['MA_200']
        self.df['Cross'] = self.df['Above'].ne(self.df['Above'].shift(1))
        self.df['Type']  = np.where(self.df['MA_50'] > self.df['MA_200'], 'Golden Cross', 'Death Cross')
        self.df['Type']  = self.df['Type'].where(self.df['Cross'], np.nan)
        self.df['Trend'] = np.where(self.df['MA_50'] > self.df['MA_200'], 'Bullish', 'Bearish')

    def get_current_signal(self):
        latest = self.df.iloc[-1]
        return {
            'timestamp': latest.name.strftime('%Y-%m-%d %H:%M:%S'),
            'price': round(latest['Close'], 2),
            'ma_50': round(latest['MA_50'], 2),
            'ma_200': round(latest['MA_200'], 2),
            'signal': latest['Signal'],
            'trend_strength': latest['Distance_%'],
            'trend': latest['Trend'],
            'last_crossover': self._last_crossover(),
            'timeframe': self.interval
        }

    def _last_crossover(self):
        cs = self.df[self.df['Cross']]
        if not cs.empty:
            last = cs.iloc[-1]
            return {
                'date': last.name.strftime('%Y-%m-%d'),
                'timestamp': last.name.strftime('%Y-%m-%d %H:%M:%S'),
                'type': last['Type'],
                'price': round(last['Close'], 2),
                'bars_since': len(self.df) - self.df.index.get_loc(last.name)
            }
        return None

# ─────────────────────────────────
# Ultimate MA (Super Guppy wrapper)
# ─────────────────────────────────

class UltimateMAIndicator:
    def __init__(self, df, params):
        self.df     = df.copy()
        self.params = params

    def calculate(self):
        df = self.df.copy()
        df["ema5"] = df["Close"].ewm(span=5, adjust=False).mean()

        len1 = self.params.get("len", 13)
        len2 = self.params.get("len2", 34)
        df["ma1"] = df["Close"].ewm(span=len1, adjust=False).mean()
        df["ma2"] = df["Close"].ewm(span=len2, adjust=False).mean()

        df["ma_up"]   = df["ma1"] > df["ma2"]
        df["ma_down"] = df["ma1"] < df["ma2"]

        prev_up   = df["ma_up"].shift(1).eq(True)
        prev_down = df["ma_down"].shift(1).eq(True)

        df["cross_up"]   = df["ma_up"] & ~prev_up
        df["cross_down"] = df["ma_down"] & ~prev_down

        df["cr_up"]    = (df["Close"] > df["ma1"]) & ~(df["Close"].shift(1) > df["ma1"].shift(1))
        df["cr_down"]  = (df["Close"] < df["ma1"]) & ~(df["Close"].shift(1) < df["ma1"].shift(1))
        df["cr_up2"]   = (df["Close"] > df["ma2"]) & ~(df["Close"].shift(1) > df["ma2"].shift(1))
        df["cr_down2"] = (df["Close"] < df["ma2"]) & ~(df["Close"].shift(1) < df["ma2"].shift(1))

        df["ema5_above_ma1"] = df["ema5"] > df["ma1"]
        df["ema5_below_ma1"] = df["ema5"] < df["ma1"]

        prev_ema5_up   = df["ema5_above_ma1"].shift(1).eq(True)
        prev_ema5_down = df["ema5_below_ma1"].shift(1).eq(True)

        df["ema5_cross_up"]   = df["ema5_above_ma1"] & ~prev_ema5_up
        df["ema5_cross_down"] = df["ema5_below_ma1"] & ~prev_ema5_down

        return df

    def summarized(self):
        last = self.calculate().iloc[-1]
        return {
            "ma1_value": float(last["ma1"]),
            "ma2_value": float(last["ma2"]),
            "green": bool(last["ma_up"]),
            "red":   bool(last["ma_down"]),
            "cross_up": bool(last["cross_up"]),
            "cross_down": bool(last["cross_down"]),
            "price_cross_ma1_up":   bool(last["cr_up"]),
            "price_cross_ma1_down": bool(last["cr_down"]),
            "price_cross_ma2_up":   bool(last["cr_up2"]),
            "price_cross_ma2_down": bool(last["cr_down2"]),
            "ema5_above_ma1": bool(last["ema5_above_ma1"]),
            "ema5_below_ma1": bool(last["ema5_below_ma1"]),
            "ema5_cross_up":  bool(last["ema5_cross_up"]),
            "ema5_cross_down":bool(last["ema5_cross_down"]),
        }

# ────────────────────────
# Candlestick patterns
# ────────────────────────

class CandlestickAnalyzer:
    bullish_patterns    = ['Hammer','BullishMarubozu','BullishEngulfing','MorningStar',
                           'PiercingLine','BullishThreeLineStrike','BullishKicker']
    bearish_patterns    = ['ShootingStar','BearishMarubozu','BearishEngulfing','EveningStar',
                           'DarkPoolCover','BearishThreeLineStrike','BearishKicker']
    indecision_patterns = ['Doji','SpinningTop','HaramiCross']

    def __init__(self, bot):
        self.bot = bot

    def detect_all(self, df: pd.DataFrame, buffer=0.25):
        df2 = df.copy()
        rng  = df2['High'] - df2['Low']
        body = (df2['Open'] - df2['Close']).abs()
        lower= df2[['Open','Close']].min(axis=1) - df2['Low']
        upper= df2['High'] - df2[['Open','Close']].max(axis=1)

        df2['Doji']            = body <= 0.1*rng
        df2['Hammer']          = (body>0) & (lower>=2*body) & (upper<=0.1*rng)
        df2['ShootingStar']    = (body>0) & (upper>=2*body) & (lower<=0.1*rng)
        df2['BullishMarubozu'] = (df2['Close']>df2['Open']) & \
                                  (df2['High'] - df2['Close'] <= buffer*rng) & \
                                  (df2['Open'] - df2['Low']   <= buffer*rng)
        df2['BearishMarubozu'] = (df2['Open']>df2['Close']) & \
                                  (df2['High'] - df2['Open'] <= buffer*rng) & \
                                  (df2['Close'] - df2['Low']  <= buffer*rng)

        be, se = [ [False] for _ in range(2) ]
        for i in range(1, len(df2)):
            p, c = df2.iloc[i-1], df2.iloc[i]
            be.append((p['Open']>p['Close']) and (c['Open']<c['Close']) and
                      (c['Open']<=p['Close']) and (c['Close']>=p['Open']))
            se.append((p['Open']<p['Close']) and (c['Open']>c['Close']) and
                      (c['Open']>=p['Close']) and (c['Close']<=p['Open']))
        df2['BullishEngulfing'], df2['BearishEngulfing'] = be, se

        ms, es = [False,False], [False,False]
        for i in range(2, len(df2)):
            f, s, t = df2.iloc[i-2], df2.iloc[i-1], df2.iloc[i]
            ms.append((f['Open']>f['Close']) and
                      abs(s['Open']-s['Close']) <= 0.3*(s['High']-s['Low']) and
                      (t['Open']<t['Close']) and
                      (t['Close']>(f['Open']+f['Close'])/2))
            es.append((f['Open']<f['Close']) and
                      abs(s['Open']-s['Close']) <= 0.3*(s['High']-s['Low']) and
                      (t['Open']>t['Close']) and
                      (t['Close']<(f['Open']+f['Close'])/2))
        df2['MorningStar'], df2['EveningStar'] = ms, es

        pl, dpc = [False], [False]
        for i in range(1, len(df2)):
            p, c = df2.iloc[i-1], df2.iloc[i]
            pl.append((p['Open']>p['Close']) and (c['Open']<c['Close']) and
                      (c['Open']<p['Low']) and (c['Close']>(p['Open']+p['Close'])/2) and
                      (c['Close']<p['Open']))
            dpc.append((p['Open']<p['Close']) and (c['Open']>c['Close']) and
                       (c['Open']>p['High']) and (c['Close']<(p['Open']+p['Close'])/2) and
                       (c['Close']>p['Open']))
        df2['PiercingLine'], df2['DarkPoolCover'] = pl, dpc

        df2['SpinningTop'] = (body>0) & (upper>=body) & (lower>=body) & (body<=0.3*rng)

        for pat in self.bullish_patterns + self.bearish_patterns + self.indecision_patterns:
            if pat not in df2.columns:
                df2[pat] = False

        cols = ['Open','High','Low','Close','Volume'] + \
               self.bullish_patterns + self.bearish_patterns + self.indecision_patterns
        return df2[[c for c in cols if c in df2.columns]]

    def detect_patterns(self, df: pd.DataFrame):
        if len(df) < 4:
            return {}
        df2 = df.copy()
        if isinstance(df2.index, pd.DatetimeIndex):
            df2 = df2.reset_index().rename(columns={'index':'Timestamp'})
        pats = self.detect_all(df2)
        for idx in range(len(pats)-1, -1, -1):
            for pat in (self.bullish_patterns + self.bearish_patterns + self.indecision_patterns):
                if pats.at[idx, pat]:
                    ts = pats.at[idx, 'Timestamp'] if 'Timestamp' in pats else pats.index[idx]
                    if isinstance(ts, (pd.Timestamp, dt.datetime)):
                        ts = ts.isoformat()
                    return {
                        pat: {
                            'index': idx,
                            'timestamp': ts,
                            'candlestick': {
                                'Open':  float(pats.at[idx,'Open']),
                                'High':  float(pats.at[idx,'High']),
                                'Low':   float(pats.at[idx,'Low']),
                                'Close': float(pats.at[idx,'Close'])
                            }
                        }
                    }
        return {}

# =========================
# AI CPR MODEL INTEGRATION
# =========================

# --- Optional: safe fallback if your project already defines this elsewhere ---
if 'convert_to_serializable' not in globals():
    def convert_to_serializable(x):
        try:
            if x is None:
                return None
            if hasattr(x, 'item'):  # numpy scalar
                return x.item()
            return float(x)
        except Exception:
            try:
                return str(x)
            except Exception:
                return None

AI_MIN_CONF = 0.25   # confidence gate (tune as you like)
AI_GATE_TRADES = True  # set False to only annotate without gating


def _ai_dir_from_label(lbl: Optional[str]):
    """
    Map class labels to directional intent
    """
    if not lbl:
        return None

    # [OK] HANDLE NUMERIC LABELS FROM YOUR MODEL:
    if isinstance(lbl, (int, float)):
        lbl_num = int(lbl)
        if lbl_num == 2:
            return +1  # STRONG_BUY
        elif lbl_num == 1:
            return +1  # BUY
        elif lbl_num == -1:
            return -1  # SELL
        elif lbl_num == -2:
            return -1  # STRONG_SELL
        elif lbl_num == 0:
            return 0  # HOLD
        return None

    # Handle string labels
    s = str(lbl).lower()
    if any(k in s for k in ["strong_buy", "2", "bull"]):
        return +1
    if any(k in s for k in ["strong_sell", "-2", "bear"]):
        return -1
    if any(k in s for k in ["buy", "1", "long"]):
        return +1
    if any(k in s for k in ["sell", "-1", "short"]):
        return -1
    if any(k in s for k in ["hold", "0", "neutral"]):
        return 0
    return None


# Add these BEFORE analyze_cpr_strategy function (around line 640)

def detect_candle_patterns(ohlc_df):
    """
    Detect key candle patterns at pivot levels
    Returns dict with pattern flags
    """
    if ohlc_df is None or len(ohlc_df) < 3:
        return {}

    patterns = {}
    try:
        # Safe access with bounds checking
        df_len = len(ohlc_df)
        if df_len < 3:
            return {}
        latest = ohlc_df.iloc[-1] if df_len > 0 else None
        prev = ohlc_df.iloc[-2] if df_len > 1 else None
        prev2 = ohlc_df.iloc[-3] if df_len > 2 else None

        if latest is None or prev is None or prev2 is None:
            return {}

        # Big Bull Take Out: Large bullish candle breaking resistance
        bull_body = latest['Close'] - latest['Open']
        patterns['big_bull_takeout'] = (
                bull_body > 0 and
                bull_body > (latest['High'] - latest['Low']) * 0.7 and
                latest['Close'] > prev['High']
        )

        # Big Bear Take Out: Large bearish candle breaking support
        bear_body = latest['Open'] - latest['Close']
        patterns['big_bear_takeout'] = (
                bear_body > 0 and
                bear_body > (latest['High'] - latest['Low']) * 0.7 and
                latest['Close'] < prev['Low']
        )

        # Fake Bull: Bullish candle rejected (next candle closes below)
        patterns['fake_bull'] = (
                prev['Close'] > prev['Open'] and
                latest['Close'] < prev['Close']
        )

        # Fake Bear: Bearish candle rejected (next candle closes above)
        patterns['fake_bear'] = (
                prev['Close'] < prev['Open'] and
                latest['Close'] > prev['Close']
        )

        # Bull Retracement: Pullback in uptrend (higher lows)
        patterns['bull_retracement'] = (
                latest['Close'] > latest['Open'] and
                latest['Low'] > prev2['Low'] and
                prev['Low'] > prev2['Low']
        )

        # Bear Retracement: Pullback in downtrend (lower highs)
        patterns['bear_retracement'] = (
                latest['Close'] < latest['Open'] and
                latest['High'] < prev2['High'] and
                prev['High'] < prev2['High']
        )

    except Exception as e:
        logger.error(f"[CANDLE-PATTERNS] Error: {e}")

    return patterns


def check_pivot_level_interactions(ohlc_df, pivot_data):
    """
    Check if price is interacting with key pivot levels
    Returns dict with interaction flags
    """
    if ohlc_df is None or len(ohlc_df) < 2:
        return {}

    interactions = {}
    try:
        latest = ohlc_df.iloc[-1]
        tolerance = 0.002  # 0.2% tolerance

        # Get pivot levels
        tc = pivot_data.get("TC", 0)
        bc = pivot_data.get("BC", 0)
        r1 = pivot_data.get("R1", 0)
        r2 = pivot_data.get("R2", 0)
        s1 = pivot_data.get("S1", 0)
        s2 = pivot_data.get("S2", 0)
        pdh = pivot_data.get("High", 0)
        pdl = pivot_data.get("Low", 0)

        # Check interactions (within tolerance)
        interactions['at_tc'] = tc and abs(latest['Close'] - tc) / tc < tolerance
        interactions['at_bc'] = bc and abs(latest['Close'] - bc) / bc < tolerance
        interactions['at_r1'] = r1 and abs(latest['Close'] - r1) / r1 < tolerance
        interactions['at_r2'] = r2 and abs(latest['Close'] - r2) / r2 < tolerance
        interactions['at_s1'] = s1 and abs(latest['Close'] - s1) / s1 < tolerance
        interactions['at_s2'] = s2 and abs(latest['Close'] - s2) / s2 < tolerance
        interactions['at_pdh'] = pdh and abs(latest['Close'] - pdh) / pdh < tolerance
        interactions['at_pdl'] = pdl and abs(latest['Close'] - pdl) / pdl < tolerance

        # Check if testing levels (wicks)
        interactions['tested_tc'] = tc and latest['High'] >= tc * 0.998 and latest['Low'] <= tc * 1.002
        interactions['tested_bc'] = bc and latest['High'] >= bc * 0.998 and latest['Low'] <= bc * 1.002
        interactions['tested_pdh'] = pdh and latest['High'] >= pdh * 0.998
        interactions['tested_pdl'] = pdl and latest['Low'] <= pdl * 1.002

    except Exception as e:
        logger.error(f"[PIVOT-INTERACTIONS] Error: {e}")

    return interactions

# ==============================
# UPDATED CPR ANALYSIS (WITH AI)
# ==============================
def analyze_cpr_strategy(indicators, pivot_data, ai_predictor, ohlc_df=None):
    """
    Analyze market using Enhanced CPR with Key Price Action Levels and Candle Patterns
    """
    # [OK] NEW: Comprehensive CPR validation
    required_cpr_keys = ["TC", "BC", "R1", "R2", "R3", "S1", "S2", "S3"]
    missing_keys = [key for key in required_cpr_keys
                    if key not in pivot_data or pivot_data[key] is None]

    if missing_keys:
        error_msg = f"Missing CPR levels: {', '.join(missing_keys)}"
        symbol = indicators.get('symbol', 'unknown')
        # Suppress warning for options which are expected to have missing CPR
        is_opt = any(x in str(symbol).upper() for x in ["CE", "PE"])
        if not is_opt:
            print(f"[CPR-WARNING] {error_msg} for {symbol}")
        
        # Return partial data instead of hard error to allow bot to continue
        return {
            "warning": error_msg,
            "trade_strategy": "None",
            "reason": "Incomplete CPR data",
            "cpr_levels": pivot_data
        }

    # Validate TC > BC
    tc = pivot_data.get("TC")
    bc = pivot_data.get("BC")

    if tc is None or bc is None:
        return {"error": "TC or BC is None", "trade_strategy": "None"}

    if tc < bc:
        logger.warning(f"[CPR-WARNING] TC ({tc}) < BC ({bc}) - Inverted CPR!")

    # [OK] Check for minimal required indicators before proceeding
    has_minimal_data = all([
        indicators.get("ema_5"),
        indicators.get("ema_21"),
        indicators.get("close")
    ])

    if not has_minimal_data:
        return {"error": f"Insufficient indicators for CPR analysis: ema_5={indicators.get('ema_5')}, ema_21={indicators.get('ema_21')}, close={indicators.get('close')}",
                "trade_strategy": "None"}

    def safe_float(value, default=0.0):
        """Safely convert to float with default, handles Series via last()"""
        if value is None:
            return default
        try:
            # FIX: Use last() to handle Series ambiguity
            return last(value)
        except (ValueError, TypeError):
            return default

    ema_21 = safe_float(indicators.get("ema_21"))
    ema_5 = safe_float(indicators.get("ema_5"))
    ema_20 = safe_float(indicators.get("ema_20"))
    ema_50 = safe_float(indicators.get("ema_50"))
    ema_200 = safe_float(indicators.get("ema_200"))
    st21Trend = safe_float(indicators.get("st21Trend"))
    adx = safe_float(indicators.get("adx"))
    close = safe_float(indicators.get("close"))

    # CPR/Pivot levels
    tc = safe_float(pivot_data.get("TC"))
    bc = safe_float(pivot_data.get("BC"))
    r1 = safe_float(pivot_data.get("R1"))
    r2 = safe_float(pivot_data.get("R2"))
    r3 = safe_float(pivot_data.get("R3"))
    s1 = safe_float(pivot_data.get("S1"))
    s2 = safe_float(pivot_data.get("S2"))
    s3 = safe_float(pivot_data.get("S3"))

    pdh = safe_float(pivot_data.get("High"))
    pdl = safe_float(pivot_data.get("Low"))
    pwh = safe_float(pivot_data.get("PWH"))
    pwl = safe_float(pivot_data.get("PWL"))
    pmh = safe_float(pivot_data.get("PMH"))
    pml = safe_float(pivot_data.get("PML"))
    wh_52 = safe_float(pivot_data.get("52WH"))
    wl_52 = safe_float(pivot_data.get("52WL"))

    # Check if essential data is available and valid
    if not (tc and bc and close and tc > 0 and bc > 0 and close > 0):
        return {"error": f"Invalid CPR data: TC={tc or 0}, BC={bc or 0}, Close={close or 0}",
                "trade_strategy": "None"}


    # Previous day CPR for comparison (if available)
    prev_tc = safe_float(pivot_data.get("prev_TC"))
    prev_bc = safe_float(pivot_data.get("prev_BC"))

    if not all([tc, bc, close]):
        return {"error": "Invalid CPR or price data"}

    # Key Price Action View Formulation (Market Bias)
    key_price_action_view = "NEUTRAL"
    position_sizing = "CONSERVATIVE"

    # AGGRESSIVE BULLISH MOMENTUM: Break above key levels with follow-through
    if (close > pdh * 1.002 and adx > 35 and ema_5 > ema_21):
        key_price_action_view = "AGGRESSIVE_BULLISH"
        position_sizing = "AGGRESSIVE"
    elif (close > pmh * 1.002 and adx > 30):  # Break above monthly high
        key_price_action_view = "AGGRESSIVE_BULLISH"
        position_sizing = "AGGRESSIVE"
    elif (close > wh_52 * 1.002 and adx > 25):  # Break above 52-week high
        key_price_action_view = "AGGRESSIVE_BULLISH"
        position_sizing = "AGGRESSIVE"

    # AGGRESSIVE BEARISH REVERSAL: Rejection at resistance or break below support
    elif (close < pdl * 0.998 and adx > 35 and ema_5 < ema_21):
        key_price_action_view = "AGGRESSIVE_BEARISH"
        position_sizing = "AGGRESSIVE"
    elif (close < pml * 0.998 and adx > 30):  # Break below monthly low
        key_price_action_view = "AGGRESSIVE_BEARISH"
        position_sizing = "AGGRESSIVE"
    elif (close < wl_52 * 0.998 and adx > 25):  # Break below 52-week low
        key_price_action_view = "AGGRESSIVE_BEARISH"
        position_sizing = "AGGRESSIVE"

    # DEFENSIVE RETRACEMENT: Break then reverse (caution mode)
    elif (close > pdh * 1.001 and close < pdh * 1.005 and adx < 30):
        key_price_action_view = "DEFENSIVE_RETRACEMENT"
        position_sizing = "DEFENSIVE"
    elif (close < pdl * 0.999 and close > pdl * 0.995 and adx < 30):
        key_price_action_view = "DEFENSIVE_RETRACEMENT"
        position_sizing = "DEFENSIVE"

    # SWING REVERSAL: Invalidating previous view
    elif (pwh and close < pwh * 0.998):  # Break below previous week high (long invalidation)
        key_price_action_view = "SWING_REVERSAL_SHORT"
        position_sizing = "AGGRESSIVE"
    elif (pwl and close > pwl * 1.002):  # Break above previous week low (short invalidation)
        key_price_action_view = "SWING_REVERSAL_LONG"
        position_sizing = "AGGRESSIVE"

    # CPR Width Analysis (Trend Bias)
    cpr_width = tc - bc if tc and bc else 0
    cpr_trend_bias = "NEUTRAL"
    if cpr_width > 0:
        # Compare with typical width (simplified - could be enhanced with historical avg)
        if cpr_width > (close * 0.02):  # > 2% of price
            cpr_trend_bias = "WIDE"  # Sideways/Range bound
        else:
            cpr_trend_bias = "NARROW"  # Trending day expected

    # CPR Position Analysis
    cpr_position = "NEUTRAL"
    if close > tc:
        cpr_position = "ABOVE_TC"  # Bullish bias
    elif close < bc:
        cpr_position = "BELOW_BC"  # Bearish bias
    elif bc <= close <= tc:
        cpr_position = "IN_CPR"  # Neutral/Range bound

    # CPR vs Previous Day Analysis
    cpr_vs_prev = "NEUTRAL"
    if prev_tc and prev_bc:
        if tc > prev_tc and bc > prev_bc:  # Shifted up
            cpr_vs_prev = "SHIFTED_UP"
        elif tc < prev_tc and bc < prev_bc:  # Shifted down
            cpr_vs_prev = "SHIFTED_DOWN"
        elif (prev_bc <= tc <= prev_tc and prev_bc <= bc <= prev_tc):  # Inside previous
            cpr_vs_prev = "INSIDE_PREV"
        elif (tc > prev_tc or bc < prev_bc):  # Outside previous
            cpr_vs_prev = "OUTSIDE_PREV"

    # MA Trend Analysis (20>50/50>20 as per PDF)
    ma_trend = "NEUTRAL"
    if ema_20 and ema_50:
        if ema_20 > ema_50:
            ma_trend = "BULLISH"
        elif ema_50 > ema_20:
            ma_trend = "BEARISH"

    # 200MA as support/resistance (as per PDF)
    ma_200_signal = "NEUTRAL"
    if ema_200:
        if close > ema_200 and ema_200 > 0:
            ma_200_signal = "ABOVE_200MA"
        elif close < ema_200 and ema_200 > 0:
            ma_200_signal = "BELOW_200MA"

    # AI CPR inference
    #ai_label, ai_conf, ai_dist, _ = ai_predictor.predict(indicators, pivot_data, _build_ai_cpr_features)
    if ohlc_df is None or ohlc_df.empty:
        logger.warning("[AI-CPR] No OHLC data provided - fetching fresh data")
    ai_label, ai_conf, ai_dist, feature_array = ai_predictor.predict(
        indicators, pivot_data, _build_ai_cpr_features, ohlc_df=ohlc_df)
    ai_dir = _ai_dir_from_label(ai_label)

    if feature_array is not None and len(feature_array) > 0:
        features_flat = feature_array.flatten()

        # Extract candle features (last 12 elements)
        candle_features = features_flat[-12:] if len(features_flat) >= 12 else None

        if candle_features is not None:
            logger.info(
                f"\n{'=' * 60}\n"
                f"🤖 AI PREDICTION BREAKDOWN\n"
                f"{'=' * 60}\n"
                f"Prediction: {ai_label} (confidence: {ai_conf:.3f})\n"
                f"\n"
                f"CANDLE FEATURES AI SAW:\n"
                f"  • Body Strength: {candle_features[0]:.1f}%\n"
                f"  • Direction: {'BULLISH' if candle_features[1] > 0 else 'BEARISH'}\n"
                f"  • Upper Wick: {candle_features[2]:.1f}% {'(rejection)' if candle_features[2] > 30 else ''}\n"
                f"  • Lower Wick: {candle_features[3]:.1f}% {'(support)' if candle_features[3] > 30 else ''}\n"
                f"  • Engulfing: {candle_features[4]} {'🟢 BULLISH' if candle_features[4] > 0 else '🔴 BEARISH' if candle_features[4] < 0 else ''}\n"
                f"  • Reversal Pattern: {candle_features[5]} {'🔨' if candle_features[5] > 0 else '⭐' if candle_features[5] < 0 else ''}\n"
                f"  • Marubozu: {candle_features[6]} {'📈' if candle_features[6] > 0 else '📉' if candle_features[6] < 0 else ''}\n"
                f"  • 3-Candle Momentum: {candle_features[7]} {'[START]' if candle_features[7] > 0 else '🔻' if candle_features[7] < 0 else ''}\n"
                f"  • Range Expansion: {candle_features[8]:.2f}x\n"
                f"  • Gap: {candle_features[9]} {'⬆️' if candle_features[9] > 0 else '⬇️' if candle_features[9] < 0 else ''}\n"
                f"  • Close Position: {candle_features[10]:.1f}% {'(near high)' if candle_features[10] > 70 else '(near low)' if candle_features[10] < 30 else '(mid)'}\n"
                f"  • Volume: {candle_features[11]:.2f}x {'💪' if candle_features[11] > 1.5 else '[WARNING]' if candle_features[11] < 0.7 else ''}\n"
                f"\n"
                f"KEY TECHNICAL FEATURES:\n"
                f"  • EMA5: {features_flat[0]:.2f}\n"
                f"  • EMA21: {features_flat[2]:.2f}\n"
                f"  • RSI: {features_flat[5]:.1f}\n"
                f"  • ADX: {features_flat[12]:.1f}\n"
                f"  • CPR Distance: {features_flat[15] * 100:.2f}%\n"
                f"{'=' * 60}"
            )

    # Candle pattern detection at pivot levels
    candle_patterns = {}
    pivot_interactions = {}
    if ohlc_df is not None and not ohlc_df.empty:
        candle_patterns = detect_candle_patterns(ohlc_df)
        pivot_interactions = check_pivot_level_interactions(ohlc_df, pivot_data)

    # Reversal detection at R2/R3 for exit management
    reversal_at_r2_r3 = False
    if ohlc_df is not None and len(ohlc_df) >= 3:
        latest = ohlc_df.iloc[-1]
        prev = ohlc_df.iloc[-2]
        # Bearish reversal at resistance (R2/R3)
        if (latest['Close'] < latest['Open'] and  # Bear candle
                (r2 or r3) and
                latest['High'] >= (r2 or r3) * 0.995 and  # Tested resistance
                latest['Close'] < (r2 or r3) * 0.998):  # Rejected
            reversal_at_r2_r3 = True

    trade_strategy = "None"
    reason = ""

    # AGGRESSIVE LONG ENTRY (Confirmed Trend) - Must be above all key levels
    if (key_price_action_view == "AGGRESSIVE_BULLISH" and
            close > tc and close > pdh and close > r1 and close > r2 and
            adx > 30 and ema_5 > ema_21):
        trade_strategy = "Buy"
        reason = f"AGGRESSIVE LONG: Above CPR,PDH,R1,R2 | View:{key_price_action_view} | MA:{ma_trend}"

    # AGGRESSIVE LONG ENTRY (Confirmed Support/Pullback Reversal)
    elif (key_price_action_view in ["AGGRESSIVE_BULLISH", "DEFENSIVE_RETRACEMENT"] and
          close > pmh * 0.998 and  # At monthly support
          close > tc and close > r1 and  # Breaking CPR and R1
          adx > 25 and ema_5 > ema_21 and
          candle_patterns.get('bull_retracement')):
        trade_strategy = "Buy"
        reason = f"AGGRESSIVE LONG: PMH support, broke CPR/R1 | View:{key_price_action_view} | Pattern: Bull Retracement"

    # AGGRESSIVE SHORT ENTRY (Symmetrical to long - using S1/S2)
    elif (key_price_action_view == "AGGRESSIVE_BEARISH" and
          close < bc and close < pdl and close < s1 and close < s2 and
          adx > 30 and ema_5 < ema_21):
        trade_strategy = "Sell"
        reason = f"AGGRESSIVE SHORT: Below CPR,PDL,S1,S2 | View:{key_price_action_view} | MA:{ma_trend}"

    # AGGRESSIVE SHORT ENTRY (Support break)
    elif (key_price_action_view in ["AGGRESSIVE_BEARISH", "DEFENSIVE_RETRACEMENT"] and
          close < pml * 1.002 and  # At monthly resistance
          close < bc and close < s1 and  # Breaking CPR and S1
          adx > 25 and ema_5 < ema_21 and
          candle_patterns.get('bear_retracement')):
        trade_strategy = "Sell"
        reason = f"AGGRESSIVE SHORT: PML resistance, broke CPR/S1 | View:{key_price_action_view} | Pattern: Bear Retracement"

    # Enhanced strategy selection with Key Price Action + Candle Patterns + MA Analysis + CPR Rules
    elif close > pdh * 1.002:  # Break above PDH
        if adx and adx > 35 and close > tc and close > r1 and ema_5 > ema_21:
            # Check for bullish candle patterns at pivot levels
            bullish_signals = []
            if candle_patterns.get('big_bull_takeout'):
                bullish_signals.append("Big Bull Take Out")
            if candle_patterns.get('fake_bear'):
                bullish_signals.append("Fake Bear")
            if candle_patterns.get('bull_retracement'):
                bullish_signals.append("Bull Retracement")

            # Add MA trend confirmation
            ma_confirmed = ma_trend == "BULLISH" or ma_200_signal == "ABOVE_200MA"

            if bullish_signals:
                trade_strategy = "Buy"
                reason = f"AGGRESSIVE: Break above PDH, above TC/R1, EMA5>21, ADX>{adx:.0f} | MA:{ma_trend}/{ma_200_signal} | Candles: {', '.join(bullish_signals)}"
            elif ma_confirmed:
                trade_strategy = "Buy"
                reason = f"AGGRESSIVE: Break above PDH, above TC/R1, EMA5>21, ADX>{adx:.0f} | MA:{ma_trend}/{ma_200_signal}"
            else:
                trade_strategy = "Buy"
                reason = f"AGGRESSIVE: Break above PDH, above TC/R1, EMA5>21, ADX>{adx:.0f}"
    elif close < pdl * 0.998:  # Break below PDL
        if adx and adx > 35 and close < bc and close < s1 and ema_5 < ema_21:
            # Check for bearish candle patterns at pivot levels
            bearish_signals = []
            if candle_patterns.get('big_bear_takeout'):
                bearish_signals.append("Big Bear Take Out")
            if candle_patterns.get('fake_bull'):
                bearish_signals.append("Fake Bull")
            if candle_patterns.get('bear_retracement'):
                bearish_signals.append("Bear Retracement")

            # Add MA trend confirmation
            ma_confirmed = ma_trend == "BEARISH" or ma_200_signal == "BELOW_200MA"

            if bearish_signals:
                trade_strategy = "Sell"
                reason = f"AGGRESSIVE: Break below PDL, below BC/S1, EMA5<21, ADX>{adx:.0f} | MA:{ma_trend}/{ma_200_signal} | Candles: {', '.join(bearish_signals)}"
            elif ma_confirmed:
                trade_strategy = "Sell"
                reason = f"AGGRESSIVE: Break below PDL, below BC/S1, EMA5<21, ADX>{adx:.0f} | MA:{ma_trend}/{ma_200_signal}"
            else:
                trade_strategy = "Sell"
                reason = f"AGGRESSIVE: Break below PDL, below BC/S1, EMA5<21, ADX>{adx:.0f}"
    elif close > tc and close > pdh:
        key_price_action_view = "BULLISH_MOMENTUM"
        if adx and ema_5 > ema_21 and ma_trend == "BULLISH":
            trade_strategy = "Buy"
            reason = f"MOMENTUM: Above TC & PDH, EMA5>21, ADX>{adx:.0f} | MA:{ma_trend}"
    elif close < bc and close < pdl:
        key_price_action_view = "BEARISH_MOMENTUM"
        if adx and adx > 20 and ema_5 < ema_21 and ma_trend == "BEARISH":
            trade_strategy = "Sell"
            reason = f"MOMENTUM: Below BC & PDL, EMA5<21, ADX>{adx:.0f} | MA:{ma_trend}"

    # NEW: CPR Breakout Rules (Page 37)
    elif cpr_position == "ABOVE_TC" and cpr_trend_bias == "NARROW" and ma_trend == "BULLISH":
        # Bullish breakout setup
        if adx > 25 and ema_5 > ema_21:
            trade_strategy = "Buy"
            reason = f"CPR BREAKOUT: Above TC, narrow width, EMA5>21, ADX>{adx:.0f} | MA:{ma_trend}"
    elif cpr_position == "BELOW_BC" and cpr_trend_bias == "NARROW" and ma_trend == "BEARISH":
        # Bearish breakout setup
        if adx > 25 and ema_5 < ema_21:
            trade_strategy = "Sell"
            reason = f"CPR BREAKOUT: Below BC, narrow width, EMA5<21, ADX>{adx:.0f} | MA:{ma_trend}"

    # NEW: CPR Retest Rules
    elif cpr_position == "ABOVE_TC" and close < tc * 1.001:  # Retesting TC from above (support)
        if adx > 20 and ma_trend == "BULLISH" and candle_patterns.get('bull_retracement'):
            trade_strategy = "Buy"
            reason = f"CPR SUPPORT: Retest TC, bullish MA, ADX>{adx:.0f} | Pattern: Bull Retracement"
    elif cpr_position == "BELOW_BC" and close > bc * 0.999:  # Retesting BC from below (resistance)
        if adx > 20 and ma_trend == "BEARISH" and candle_patterns.get('bear_retracement'):
            trade_strategy = "Sell"
            reason = f"CPR RESISTANCE: Retest BC, bearish MA, ADX>{adx:.0f} | Pattern: Bear Retracement"

    # NEW: Confluence Setups (High Probability)
    elif (cpr_position in ["ABOVE_TC", "BELOW_BC"] and
          cpr_vs_prev in ["SHIFTED_UP", "SHIFTED_DOWN", "OUTSIDE_PREV"] and
          cpr_trend_bias == "NARROW" and
          ma_trend != "NEUTRAL"):
        if cpr_position == "ABOVE_TC" and ma_trend == "BULLISH":
            trade_strategy = "Buy"
            reason = f"CONFLUENCE: CPR breakout, {cpr_vs_prev}, narrow width, bullish MA"
        elif cpr_position == "BELOW_BC" and ma_trend == "BEARISH":
            trade_strategy = "Sell"
            reason = f"CONFLUENCE: CPR breakout, {cpr_vs_prev}, narrow width, bearish MA"

    # EXIT MANAGEMENT: Reversal at R2/R3 for aggressive long positions
    elif reversal_at_r2_r3 and key_price_action_view == "AGGRESSIVE_BULLISH":
        trade_strategy = "Exit"
        reason = f"EXIT: Reversal at R2/R3 resistance | View:{key_price_action_view}"

    # AI veto/assist
    ai_filter_pass = True
    if ai_dir is not None and ai_conf is not None:
        if trade_strategy.startswith("Buy") and ai_dir < 0 and ai_conf >= AI_MIN_CONF:
            ai_filter_pass = False
            reason += f" | AI disagrees (label={ai_label}, conf={round(ai_conf, 2)})"
        elif trade_strategy.startswith("Sell") and ai_dir > 0 and ai_conf >= AI_MIN_CONF:
            ai_filter_pass = False
            reason += f" | AI disagrees (label={ai_label}, conf={round(ai_conf, 2)})"
        elif trade_strategy == "Exit" and ai_dir < 0 and ai_conf >= AI_MIN_CONF:
            # AI confirms exit
            reason += f" | AI confirms exit (label={ai_label}, conf={round(ai_conf, 2)})"
        else:
            reason += f" | AI:{ai_label}({round(ai_conf, 2) if ai_conf is not None else 'NA'})"

    return {
        "trade_strategy": trade_strategy,
        "reason": reason,
        "ai_cpr_label": ai_label,
        "ai_confidence": ai_conf,
        "ai_distribution": ai_dist,
        "ai_filter_pass": ai_filter_pass,
        "key_price_action_view": key_price_action_view,
        "position_sizing": position_sizing,
        "candle_patterns": candle_patterns,
        "pivot_interactions": pivot_interactions,
        "reversal_at_r2_r3": reversal_at_r2_r3,
        "ma_trend": ma_trend,
        "ma_200_signal": ma_200_signal,
        "cpr_trend_bias": cpr_trend_bias,
        "cpr_position": cpr_position,
        "cpr_vs_prev": cpr_vs_prev,
        "cpr_width": cpr_width,
        # 🔥 ADD THIS LINE:
        "cpr_levels": pivot_data  # ← Pass through the original CPR levels!
    }


class PriceActionAnalyzer:
    """
    Pure Price Action Trading Strategy
    Uses: Support/Resistance, CPR Levels, Fibonacci, and Candle Patterns
    """

    def __init__(self, bot):
        self.bot = bot
        self.logger = bot.log_message

    @staticmethod
    def _f(v, alt=None):
        """Safe float conversion"""
        try:
            if v is None: return alt
            x = float(v)
            return x if np.isfinite(x) else alt
        except:
            return alt

    def analyze(self, symbol, ltp, indicators, pivot_data, ohlc_df):
        """
        Main Price Action Analysis
        Returns: (signal, confidence, reason, levels)
        """

        # [OK] VALIDATION 1: Check inputs
        if not ltp or ltp <= 0:
            return {
                "signal": None,
                "confidence": 0.0,
                "reason": "Invalid LTP",
                "key_level": None,
                "level_price": None,
                "all_levels": {}
            }

        if not pivot_data or not isinstance(pivot_data, dict):
            return {
                "signal": None,
                "confidence": 0.0,
                "reason": "No pivot data",
                "key_level": None,
                "level_price": None,
                "all_levels": {}
            }

        # [OK] VALIDATION 2: Check CPR levels exist
        required_cpr = ["TC", "BC", "R1", "S1"]
        missing = [k for k in required_cpr if k not in pivot_data or pivot_data[k] is None]

        if missing:
            return {
                "signal": None,
                "confidence": 0.0,
                "reason": f"Missing CPR levels: {', '.join(missing)}",
                "key_level": None,
                "level_price": None,
                "all_levels": {}
            }

        # [OK] VALIDATION 3: Check OHLC data
        if ohlc_df is None or ohlc_df.empty or len(ohlc_df) < 3:
            return {
                "signal": None,
                "confidence": 0.0,
                "reason": f"Insufficient OHLC data ({len(ohlc_df) if ohlc_df is not None else 0} candles)",
                "key_level": None,
                "level_price": None,
                "all_levels": {}
            }

        # Get all key levels
        levels = self._get_all_levels(indicators, pivot_data, ohlc_df)

        # [OK] Log what levels we found
        self.logger(
            f"[PRICE-ACTION] Levels found: "
            f"{len(levels.get('resistance', []))} resistance, "
            f"{len(levels.get('support', []))} support",
            True
        )

        # Check which level we're at
        level_interaction = self._check_level_interaction(ltp, levels)

        # [OK] Early exit if not at level
        if not level_interaction.get("at_level"):
            return {
                "signal": None,
                "confidence": 0.0,
                "reason": level_interaction.get("reason", "Not at key level"),
                "key_level": None,
                "level_price": None,
                "all_levels": levels
            }

        # Check candle pattern at level
        candle_signal = self._check_candle_at_level(ohlc_df, level_interaction)

        # Check Fibonacci confluence
        fib_signal = self._check_fibonacci_confluence(ltp, indicators, level_interaction)

        # Generate final signal
        signal, confidence, reason = self._generate_signal(
            level_interaction,
            candle_signal,
            fib_signal,
            indicators
        )

        return {
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "key_level": level_interaction.get("level_type"),
            "level_price": level_interaction.get("price"),
            "all_levels": levels
        }

    def _get_all_levels(self, indicators, pivot_data, ohlc_df):
        """
        Collect ALL support/resistance levels
        Returns unified level structure
        """
        levels = {
            "resistance": [],
            "support": []
        }

        # 1. CPR Levels
        tc = pivot_data.get("TC")
        bc = pivot_data.get("BC")
        r1, r2, r3 = pivot_data.get("R1"), pivot_data.get("R2"), pivot_data.get("R3")
        s1, s2, s3 = pivot_data.get("S1"), pivot_data.get("S2"), pivot_data.get("S3")

        if tc: levels["resistance"].append({"price": tc, "type": "CPR_TC", "strength": "strong"})
        if bc: levels["support"].append({"price": bc, "type": "CPR_BC", "strength": "strong"})

        for r, name in [(r1, "R1"), (r2, "R2"), (r3, "R3")]:
            if r: levels["resistance"].append({"price": r, "type": name, "strength": "medium"})

        for s, name in [(s1, "S1"), (s2, "S2"), (s3, "S3")]:
            if s: levels["support"].append({"price": s, "type": name, "strength": "medium"})

        # 2. Daily High/Low
        pdh = pivot_data.get("High")
        pdl = pivot_data.get("Low")
        if pdh: levels["resistance"].append({"price": pdh, "type": "PDH", "strength": "strong"})
        if pdl: levels["support"].append({"price": pdl, "type": "PDL", "strength": "strong"})

        # 3. Fibonacci Levels
        fib_data = indicators.get("fib", {})
        fib_levels = fib_data.get("levels", {})

        for name, price in fib_levels.items():
            if "61.8" in name or "78.6" in name:  # Key Fib levels
                # Determine if S or R based on trend
                trend = fib_data.get("trend", "Neutral")
                if trend == "Bullish":
                    levels["support"].append({"price": price, "type": f"Fib_{name}", "strength": "medium"})
                elif trend == "Bearish":
                    levels["resistance"].append({"price": price, "type": f"Fib_{name}", "strength": "medium"})

        # 4. Rolling Support/Resistance (20-period)
        support_20 = indicators.get("support")
        resistance_20 = indicators.get("resistance")

        if support_20:
            levels["support"].append({"price": support_20, "type": "SR_20", "strength": "weak"})
        if resistance_20:
            levels["resistance"].append({"price": resistance_20, "type": "SR_20", "strength": "weak"})

        # Sort by price
        levels["resistance"].sort(key=lambda x: x["price"], reverse=True)
        levels["support"].sort(key=lambda x: x["price"], reverse=True)

        return levels

    def _check_level_interaction(self, ltp, levels, tolerance=0.008):  # ← Changed from 0.003 to 0.008
        """
        Check if price is near any key level (within 0.5%)
        Returns: {"at_level": bool, "level_type": str, "price": float, "direction": str}
        """

        if not ltp or ltp <= 0:
            return {"at_level": False, "level_type": None, "price": None, "direction": None}

        # [OK] Check resistance levels
        for r_level in levels.get("resistance", []):
            price = r_level.get("price")
            if not price or price <= 0:
                continue

            distance_pct = abs(ltp - price) / price

            if distance_pct <= tolerance:
                return {
                    "at_level": True,
                    "level_type": r_level.get("type"),
                    "price": price,
                    "direction": "resistance",
                    "strength": r_level.get("strength"),
                    "distance_pct": ((ltp - price) / price * 100)
                }

        # [OK] Check support levels
        for s_level in levels.get("support", []):
            price = s_level.get("price")
            if not price or price <= 0:
                continue

            distance_pct = abs(ltp - price) / price

            if distance_pct <= tolerance:
                return {
                    "at_level": True,
                    "level_type": s_level.get("type"),
                    "price": price,
                    "direction": "support",
                    "strength": s_level.get("strength"),
                    "distance_pct": ((ltp - price) / price * 100)
                }

        # [OK] Log why not at level (for debugging)
        closest_resistance = None
        closest_support = None
        min_r_dist = float('inf')
        min_s_dist = float('inf')

        for r_level in levels.get("resistance", []):
            price = r_level.get("price")
            if price and price > 0:
                dist = abs(ltp - price) / price
                if dist < min_r_dist:
                    min_r_dist = dist
                    closest_resistance = (r_level.get("type"), price, dist * 100)

        for s_level in levels.get("support", []):
            price = s_level.get("price")
            if price and price > 0:
                dist = abs(ltp - price) / price
                if dist < min_s_dist:
                    min_s_dist = dist
                    closest_support = (s_level.get("type"), price, dist * 100)

        # Log nearest levels
        if closest_resistance:
            self.bot.log_message(
                f"[PRICE-ACTION] Not at resistance - Nearest: {closest_resistance[0]} "
                f"at ₹{closest_resistance[1]:.2f} ({closest_resistance[2]:.2f}% away, need ≤{tolerance * 100:.1f}%)",
                True
            )

        if closest_support:
            self.bot.log_message(
                f"[PRICE-ACTION] Not at support - Nearest: {closest_support[0]} "
                f"at ₹{closest_support[1]:.2f} ({closest_support[2]:.2f}% away, need ≤{tolerance * 100:.1f}%)",
                True
            )

        return {
            "at_level": False,
            "level_type": None,
            "price": None,
            "direction": None,
            "strength": None
        }

    def _check_candle_at_level(self, ohlc_df, level_interaction):
        """
        Check if there's a rejection/reversal candle at the level
        RELAXED criteria for Natural Gas
        """

        if not level_interaction.get("at_level"):
            return {"pattern": None, "confidence": 0.0}

        if ohlc_df is None or len(ohlc_df) < 2:  # ← Changed from 3 to 2
            return {"pattern": None, "confidence": 0.0}

        latest = ohlc_df.iloc[-1]
        prev = ohlc_df.iloc[-2] if len(ohlc_df) >= 2 else latest

        # Calculate candle components
        body = abs(latest['Close'] - latest['Open'])
        total_range = latest['High'] - latest['Low']

        # Avoid division by zero
        if total_range <= 0:
            return {"pattern": None, "confidence": 0.0}

        upper_wick = latest['High'] - max(latest['Open'], latest['Close'])
        lower_wick = min(latest['Open'], latest['Close']) - latest['Low']

        direction = level_interaction.get("direction")

        # ==========================================
        # AT RESISTANCE - Look for bearish signals
        # ==========================================
        if direction == "resistance":

            # Pattern 1: Upper wick rejection (most common)
            # [OK] RELAXED: Upper wick > 1.5x body (was 2x)
            if upper_wick > body * 1.5 and upper_wick > total_range * 0.3:
                confidence = 0.75

                # Bonus: Strong red candle
                if latest['Close'] < latest['Open']:
                    confidence += 0.10

                # Bonus: High volume
                if 'Volume' in latest.index:
                    try:
                        vol_ratio = latest['Volume'] / ohlc_df['Volume'].tail(10).mean()
                        if vol_ratio > 1.3:
                            confidence += 0.10
                    except:
                        pass

                return {
                    "pattern": "Upper Wick Rejection",
                    "confidence": min(confidence, 0.95),
                    "signal": "SELL"
                }

            # Pattern 2: Shooting star (classic reversal)
            # [OK] RELAXED: Upper wick > 1.8x body (was 2.5x)
            if (upper_wick > body * 1.8 and
                    lower_wick < body * 0.5 and
                    body > 0):
                return {
                    "pattern": "Shooting Star",
                    "confidence": 0.80,
                    "signal": "SELL"
                }

            # Pattern 3: Doji at resistance (indecision)
            # [OK] NEW: Small body at resistance = reversal likely
            if body < total_range * 0.2:  # Small body (< 20% of range)
                return {
                    "pattern": "Doji at Resistance",
                    "confidence": 0.65,
                    "signal": "SELL"
                }

            # Pattern 4: Failed breakout
            # [OK] NEW: Price tried to break resistance but closed below
            if (latest['High'] > level_interaction.get("price", 0) and
                    latest['Close'] < level_interaction.get("price", 0)):
                return {
                    "pattern": "Failed Breakout",
                    "confidence": 0.70,
                    "signal": "SELL"
                }

        # ==========================================
        # AT SUPPORT - Look for bullish signals
        # ==========================================
        elif direction == "support":

            # Pattern 1: Lower wick rejection (most common)
            # [OK] RELAXED: Lower wick > 1.5x body (was 2x)
            if lower_wick > body * 1.5 and lower_wick > total_range * 0.3:
                confidence = 0.75

                # Bonus: Strong green candle
                if latest['Close'] > latest['Open']:
                    confidence += 0.10

                # Bonus: High volume
                if 'Volume' in latest.index:
                    try:
                        vol_ratio = latest['Volume'] / ohlc_df['Volume'].tail(10).mean()
                        if vol_ratio > 1.3:
                            confidence += 0.10
                    except:
                        pass

                return {
                    "pattern": "Lower Wick Rejection",
                    "confidence": min(confidence, 0.95),
                    "signal": "BUY"
                }

            # Pattern 2: Hammer (classic reversal)
            # [OK] RELAXED: Lower wick > 1.8x body (was 2.5x)
            if (lower_wick > body * 1.8 and
                    upper_wick < body * 0.5 and
                    body > 0):
                return {
                    "pattern": "Hammer",
                    "confidence": 0.80,
                    "signal": "BUY"
                }

            # Pattern 3: Doji at support
            # [OK] NEW: Small body at support = bounce likely
            if body < total_range * 0.2:
                return {
                    "pattern": "Doji at Support",
                    "confidence": 0.65,
                    "signal": "BUY"
                }

            # Pattern 4: Failed breakdown
            # [OK] NEW: Price tried to break support but closed above
            if (latest['Low'] < level_interaction.get("price", 0) and
                    latest['Close'] > level_interaction.get("price", 0)):
                return {
                    "pattern": "Failed Breakdown",
                    "confidence": 0.70,
                    "signal": "BUY"
                }

            # Pattern 5: Bullish candle at support
            # [OK] NEW: ANY green candle at support counts
            if latest['Close'] > latest['Open'] and body > total_range * 0.4:
                return {
                    "pattern": "Bullish Bounce",
                    "confidence": 0.60,
                    "signal": "BUY"
                }

        # ==========================================
        # FALLBACK: Weak signal if at level
        # ==========================================
        # [OK] NEW: Don't return None - give at least SOME signal
        if direction == "support":
            # At support, bias toward BUY even without perfect pattern
            return {
                "pattern": "At Support Level",
                "confidence": 0.50,  # Low confidence but not zero
                "signal": "BUY"
            }
        elif direction == "resistance":
            # At resistance, bias toward SELL
            return {
                "pattern": "At Resistance Level",
                "confidence": 0.50,
                "signal": "SELL"
            }

        return {"pattern": None, "confidence": 0.0}

    def _check_fibonacci_confluence(self, ltp, indicators, level_interaction):
        """
        Check if current level aligns with Fibonacci levels
        Returns: {"confluence": bool, "confidence_boost": float}
        """

        fib_data = indicators.get("fib", {})
        fib_levels = fib_data.get("levels", {})

        if not fib_levels:
            return {"confluence": False, "confidence_boost": 0.0}

        # Check if near any key Fib level (61.8% or 78.6%)
        for name, fib_price in fib_levels.items():
            if "61.8" in name or "78.6" in name:
                if abs(ltp - fib_price) / ltp <= 0.005:  # Within 0.5%
                    return {
                        "confluence": True,
                        "confidence_boost": 0.15,  # +15% confidence
                        "fib_level": name
                    }

        return {"confluence": False, "confidence_boost": 0.0}

    def _generate_signal(self, level_interaction, candle_signal, fib_signal, indicators):
        """
        Generate final trading signal based on price action WITH CONTEXT
        """

        if not level_interaction.get("at_level"):
            return None, 0.0, "Not at key level"

        level_type = level_interaction.get("level_type")
        level_price = level_interaction.get("price")
        direction = level_interaction.get("direction")
        strength = level_interaction.get("strength")

        # ==========================================
        # 🔥 NEW: CHECK TREND CONTEXT
        # ==========================================
        trend = indicators.get("trend", "").lower() if indicators.get("trend") else ""

        # Get EMA context for trend
        ema5 = self._f(indicators.get("ema_5"))
        ema21 = self._f(indicators.get("ema_21"))
        ema_trend = None
        if ema5 and ema21:
            if ema5 > ema21 * 1.003:  # 0.3% gap
                ema_trend = "bullish"
            elif ema5 < ema21 * 0.997:
                ema_trend = "bearish"
            else:
                ema_trend = "neutral"

        # Get momentum
        momentum_pct = self._f(indicators.get("momentum_pct"), 0.0)

        # Get ADX for trend strength
        adx = self._f(indicators.get("adx"), 0)

        self.bot.log_message(
            f"[PRICE-ACTION-CONTEXT] Trend: {trend}, EMA: {ema_trend}, "
            f"Momentum: {momentum_pct:.2f}%, ADX: {adx:.1f}",
            True
        )

        # ==========================================
        # Base confidence from level strength
        # ==========================================
        confidence = {
            "strong": 0.70,
            "medium": 0.60,
            "weak": 0.50
        }.get(strength, 0.50)

        # ==========================================
        # Check candle pattern
        # ==========================================
        pattern = candle_signal.get("pattern")
        pattern_conf = candle_signal.get("confidence", 0.0)
        signal = candle_signal.get("signal")

        # ==========================================
        # 🔥 CRITICAL: TREND FILTERING
        # ==========================================

        if direction == "resistance":
            # At resistance - want to SELL

            # [OK] RULE 1: Don't fight strong uptrends
            if ema_trend == "bullish" and adx > 25:
                # Strong uptrend - resistance likely to break
                if not pattern or pattern_conf < 0.80:
                    self.bot.log_message(
                        f"⛔ [PRICE-ACTION] SELL blocked at {level_type} - "
                        f"Strong uptrend (EMA bullish, ADX {adx:.1f})",
                        False
                    )
                    return None, 0.0, f"At {level_type} but uptrend too strong"

            # [OK] RULE 2: Need CLEAR rejection in uptrends
            if ema_trend == "bullish":
                if not pattern or pattern_conf < 0.70:
                    self.bot.log_message(
                        f"⛔ [PRICE-ACTION] SELL blocked at {level_type} - "
                        f"In uptrend, need strong rejection pattern (got: {pattern or 'None'})",
                        False
                    )
                    return None, 0.0, f"At {level_type} in uptrend - need confirmation"

            # [OK] RULE 3: Check momentum direction
            if momentum_pct > 0.5:
                # Strong upward momentum - don't sell yet
                if not pattern or pattern_conf < 0.75:
                    self.bot.log_message(
                        f"⛔ [PRICE-ACTION] SELL blocked at {level_type} - "
                        f"Momentum still bullish ({momentum_pct:.2f}%)",
                        False
                    )
                    return None, 0.0, f"At {level_type} but momentum bullish"

        elif direction == "support":
            # At support - want to BUY

            # [OK] RULE 1: Don't fight strong downtrends
            if ema_trend == "bearish" and adx > 25:
                if not pattern or pattern_conf < 0.80:
                    self.bot.log_message(
                        f"⛔ [PRICE-ACTION] BUY blocked at {level_type} - "
                        f"Strong downtrend (EMA bearish, ADX {adx:.1f})",
                        False
                    )
                    return None, 0.0, f"At {level_type} but downtrend too strong"

            # [OK] RULE 2: Need CLEAR bounce in downtrends
            if ema_trend == "bearish":
                if not pattern or pattern_conf < 0.70:
                    self.bot.log_message(
                        f"⛔ [PRICE-ACTION] BUY blocked at {level_type} - "
                        f"In downtrend, need strong bounce pattern (got: {pattern or 'None'})",
                        False
                    )
                    return None, 0.0, f"At {level_type} in downtrend - need confirmation"

            # [OK] RULE 3: Check momentum direction
            if momentum_pct < -0.5:
                if not pattern or pattern_conf < 0.75:
                    self.bot.log_message(
                        f"⛔ [PRICE-ACTION] BUY blocked at {level_type} - "
                        f"Momentum still bearish ({momentum_pct:.2f}%)",
                        False
                    )
                    return None, 0.0, f"At {level_type} but momentum bearish"

        # ==========================================
        # If no pattern, generate weak signal
        # ==========================================
        if not pattern:
            if direction == "support":
                signal = "BUY"
                pattern = "At Support (No Pattern)"
                pattern_conf = 0.50
            elif direction == "resistance":
                signal = "SELL"
                pattern = "At Resistance (No Pattern)"
                pattern_conf = 0.50
            else:
                return None, 0.0, f"At {level_type} but no clear direction"

        # Add candle pattern confidence
        confidence = min(0.95, confidence + (pattern_conf * 0.3))

        # ==========================================
        # Add Fibonacci confluence boost
        # ==========================================
        if fib_signal.get("confluence"):
            confidence += fib_signal.get("confidence_boost", 0.0)
            confluence_msg = f" + Fib {fib_signal.get('fib_level')}"
        else:
            confluence_msg = ""

        # ==========================================
        # 🔥 CRITICAL: VOLUME VALIDATION
        # ==========================================
        volume_ratio = self._f(indicators.get("volume_ratio"), 1.0)

        if volume_ratio < 0.3:  # Extremely low volume (like your 0.04x)
            self.bot.log_message(
                f"[WARNING] [PRICE-ACTION] Very low volume ({volume_ratio:.2f}x) - "
                f"reducing confidence",
                True
            )
            confidence *= 0.7  # Reduce confidence by 30%
            volume_msg = f" (VERY LOW volume: {volume_ratio:.2f}x)"

        elif volume_ratio < 0.8:
            volume_msg = f" (low volume: {volume_ratio:.2f}x)"
            confidence *= 0.85  # Reduce confidence by 15%

        elif volume_ratio >= 1.5:
            confidence = min(0.95, confidence + 0.10)
            volume_msg = f" + Volume {volume_ratio:.1f}x"

        else:
            volume_msg = ""

        # Generate reason
        reason = (
            f"Price Action: {pattern} at {level_type} "
            f"({level_price:.2f}){confluence_msg}{volume_msg}"
        )

        # ==========================================
        # 🔥 FINAL CONFIDENCE CHECK
        # ==========================================
        if confidence < 0.55:
            return None, confidence, f"{reason} - Confidence too low ({confidence:.2f})"

        return signal, confidence, reason

    def debug_levels(self, ltp, pivot_data):
        """Quick diagnostic to see what levels exist"""
        self.logger(f"\n{'=' * 60}", False)
        self.logger(f"[PRICE-ACTION-DEBUG] Current LTP: ₹{ltp:.2f}", False)
        self.logger(f"{'=' * 60}", False)

        # Check CPR levels
        tc = pivot_data.get("TC")
        bc = pivot_data.get("BC")
        r1 = pivot_data.get("R1")
        s1 = pivot_data.get("S1")

        if tc:
            dist = abs(ltp - tc) / tc * 100
            self.logger(f"  TC: ₹{tc:.2f} (distance: {dist:.2f}%)", False)

        if bc:
            dist = abs(ltp - bc) / bc * 100
            self.logger(f"  BC: ₹{bc:.2f} (distance: {dist:.2f}%)", False)

        if r1:
            dist = abs(ltp - r1) / r1 * 100
            self.logger(f"  R1: ₹{r1:.2f} (distance: {dist:.2f}%)", False)

        if s1:
            dist = abs(ltp - s1) / s1 * 100
            self.logger(f"  S1: ₹{s1:.2f} (distance: {dist:.2f}%)", False)

        self.logger(f"{'=' * 60}\n", False)

# ───────────────────────────────────────────────────────────────────────────────
# IndicatorCalculator.calculate_indicators (full replacement)
# ───────────────────────────────────────────────────────────────────────────────
class IndicatorCalculator:
    def __init__(self, bot):
        self.bot = bot

    def calculate_pivot_points(self, df_day):
        """Enhanced CPR calculation with comprehensive validation"""

        # [OK] VALIDATION 1: Check DataFrame
        if df_day is None or not isinstance(df_day, pd.DataFrame):
            self.bot.log_message("[ERROR] Pivot: Invalid DataFrame type", False)
            return {}

        if df_day.empty:
            self.bot.log_message("[ERROR] Pivot: Empty DataFrame", False)
            return {}

        # [OK] VALIDATION 2: Check required columns
        required_cols = ["High", "Low", "Close"]
        for c in required_cols:
            if c not in df_day.columns:
                self.bot.log_message(f"[ERROR] Pivot: Missing column '{c}'", False)
                return {}

        # [OK] VALIDATION 3: Check row count
        if len(df_day) < 2:
            self.bot.log_message(f"[ERROR] Pivot: Need at least 2 days, got {len(df_day)}", False)
            return {}

        # Use previous day's data (most recent complete day)
        prev_day = df_day.iloc[-2]

        high = float(prev_day["High"])
        low = float(prev_day["Low"])
        close = float(prev_day["Close"])

        # [OK] VALIDATION 4: Check values are positive
        if high <= 0 or low <= 0 or close <= 0:
            self.bot.log_message(f"[ERROR] Pivot: Invalid OHLC - H:{high}, L:{low}, C:{close}", False)
            return {}

        # [OK] VALIDATION 5: Check high >= low
        if high < low:
            self.bot.log_message(f"[ERROR] Pivot: High ({high}) < Low ({low}) - Invalid data!", False)
            return {}

        # Calculate pivot levels
        PP = round((high + low + close) / 3, 2)
        BC = round((high + low) / 2, 2)
        TC = round((PP - BC) + PP, 2)

        # [OK] FIX: Ensure TC > BC
        if TC < BC:
            TC, BC = BC, TC
            self.bot.log_message(f"[WARNING] Swapped TC/BC: TC={TC}, BC={BC}", True)

        # Calculate support/resistance levels
        R1 = round(2 * PP - low, 2)
        S1 = round(2 * PP - high, 2)
        R2 = round(PP + (high - low), 2)
        S2 = round(PP - (high - low), 2)
        R3 = round(high + 2 * (PP - low), 2)
        S3 = round(low - 2 * (high - PP), 2)

        # [OK] Check Virgin CPR
        virgin_cpr = False
        if prev_day["High"] < BC or prev_day["Low"] > TC:
            virgin_cpr = True

        # [OK] Calculate additional levels
        try:
            PWH = round(df_day['High'].tail(5).max(), 2) if len(df_day) >= 5 else high
            PWL = round(df_day['Low'].tail(5).min(), 2) if len(df_day) >= 5 else low
            PMH = round(df_day['High'].tail(20).max(), 2) if len(df_day) >= 20 else high
            PML = round(df_day['Low'].tail(20).min(), 2) if len(df_day) >= 20 else low
            WH_52 = round(df_day['High'].tail(252).max(), 2) if len(df_day) >= 252 else high
            WL_52 = round(df_day['Low'].tail(252).min(), 2) if len(df_day) >= 252 else low
        except Exception as e:
            self.bot.log_message(f"[WARNING] Weekly/Monthly levels calc error: {e}", True)
            PWH = PWL = PMH = PML = WH_52 = WL_52 = None

        # Build result dictionary
        result = {
            "PP": PP, "TC": TC, "BC": BC,
            "High": high, "Low": low, "Close": close,
            "virgin_cpr": virgin_cpr,
            "R1": R1, "S1": S1,
            "R2": R2, "S2": S2,
            "R3": R3, "S3": S3,
            "PWH": PWH, "PWL": PWL,
            "PMH": PMH, "PML": PML,
            "52WH": WH_52, "52WL": WL_52,
        }

        # [OK] FINAL VALIDATION: Ensure all core levels exist
        required_result_keys = ["TC", "BC", "PP", "R1", "R2", "R3", "S1", "S2", "S3"]
        missing_result_keys = [k for k in required_result_keys if k not in result or result[k] is None]

        if missing_result_keys:
            self.bot.log_message(f"[ERROR] Pivot result incomplete! Missing: {', '.join(missing_result_keys)}", False)
            return {}

        # Log successful calculation
        self.bot.log_message(
            f"\n[DATA] CPR LEVELS CALCULATED:\n"
            f"  Core: TC={TC}, BC={BC}, PP={PP} (Virgin: {virgin_cpr})\n"
            f"  Resistance: R1={R1}, R2={R2}, R3={R3}\n"
            f"  Support: S1={S1}, S2={S2}, S3={S3}",
            False
        )

        return result

    def calculate_support_resistance(self, df, period=20):
        df["Support"]    = df["Low"].rolling(period).min()
        df["Resistance"] = df["High"].rolling(period).max()
        return df

    def calculate_indicators(self, symbol, timeframe, pivot_data=None, ohlc_df=None):
        # [OK] CRITICAL: Validate pivot_data received
        if isinstance(pivot_data, list):
            self.bot.log_message(
                f"[ERROR] [INDICATORS] pivot_data is a LIST for {symbol} - converting to dict",
                False
            )
            pivot_data = {}

            # Check if pivot_data is None
        if not pivot_data:
            # If it's an option symbol, don't attempt emergency load (options don't have CPR)
            is_option = any(x in str(symbol).upper() for x in ["CE", "PE"])
            if is_option:
                # Use empty dict for options, CPR analysis will skip gracefully
                pivot_data = {}
            else:
                self.bot.log_message(
                    f"[WARNING] [INDICATORS] No pivot data for {symbol} - attempting emergency load",
                    False
                )
                pivot_data = {}

            # Check if pivot_data is not a dictionary
        if not isinstance(pivot_data, dict):
            self.bot.log_message(
                f"[ERROR] [INDICATORS] pivot_data is {type(pivot_data)} for {symbol} - using empty dict",
                False
            )
            pivot_data = {}

            # If empty, try to load from file (BUT SKIP FOR OPTIONS)
        # Re-check if it is an option to avoid file load attempt
        is_option_check = any(x in str(symbol).upper() for x in ["CE", "PE"])
        if not pivot_data and not is_option_check:
            try:
                pivot_json_path = self.bot.data_paths[symbol]['pivot_json']
                loaded_data = robust_load_json(pivot_json_path, self.bot.log_message, default={})

                # Extract symbol-specific data
                if isinstance(loaded_data, dict) and symbol in loaded_data:
                    pivot_data = loaded_data[symbol]
                    self.bot.log_message(f"[OK] [INDICATORS] Loaded pivots from file for {symbol}", True)
                else:
                    self.bot.log_message(
                        f"[ERROR] [INDICATORS] Cannot find {symbol} in pivot file\n"
                        f"   File keys: {list(loaded_data.keys()) if isinstance(loaded_data, dict) else 'NOT A DICT'}",
                        False
                    )
            except Exception as e:
                self.bot.log_message(f"[ERROR] [INDICATORS] Emergency pivot load failed: {e}", False)

            # Final validation
        required_keys = ["TC", "BC", "R1", "R2", "R3", "S1", "S2", "S3"]
        missing_keys = [k for k in required_keys if k not in pivot_data or pivot_data[k] is None]

        #if missing_keys:
        #    self.bot.log_message(
        #        f"[WARNING] [INDICATORS] Pivot data incomplete for {symbol}: missing {', '.join(missing_keys)}",
        #        False
        #    )
            # Continue anyway - CPR analysis will handle gracefully
        #else:
        #   self.bot.log_message(
        #        f"[OK] [INDICATORS] Valid pivots for {symbol}: TC={pivot_data['TC']}, BC={pivot_data['BC']}",
        #        True
        #    )
        if ohlc_df is not None and not ohlc_df.empty:
            ohlc = ohlc_df.copy()
            # FIX: Aggressive index cleaning to prevent .at[] ambiguity
            ohlc = ohlc.reset_index(drop=True)
            self.bot.log_message("IndicatorCalc: Using pre-fetched OHLC for backtesting (indexed cleaned).", True)
        else:
            ohlc = self.bot.fetch_ohlc(symbol, timeframe, 60)
            # FIX: Clean indices after fetching fresh data
            if not ohlc.empty:
                ohlc = ohlc.copy()
                ohlc = ohlc.reset_index(drop=True)

        # --- ADDED: Robustness check for backtesting ---
        # Ensure there's enough data for lookbacks (e.g., iloc[-2], rolling windows)
        # A minimum of 30 is a safe starting point for most indicators used.
        if len(ohlc) < 30:
            self.bot.log_message(f"IndicatorCalc: Not enough data ({len(ohlc)} bars) for full calculation. Need at least 30.", True)
            return {"error": f"Not enough historical data ({len(ohlc)} bars)."}
        # --- END ADD ---

        if ohlc.empty:
            self.bot.log_message(
                f"IndicatorCalc: No valid OHLC data for {symbol} after filtering.",
                False
            )
            return {"error": "No data available after filtering."}

        # 2) Baseline transforms
        ohlc.index = pd.to_datetime(ohlc.index, utc=True)
        ohlc = vwap(ohlc)

        # ATR with safe fallback
        #ohlc["ATR"] = atr(ohlc, 14)
        #atr_series = pd.to_numeric(ohlc["ATR"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        #if atr_series.isna().all():
        #    atr_series = (ohlc["High"] - ohlc["Low"]).rolling(14, min_periods=1).mean()
        #ohlc["ATR"] = atr_series.ffill().bfill()
        ohlc["ATR"] = atr(ohlc, 14)
        atr_series = pd.to_numeric(ohlc["ATR"], errors="coerce").replace([np.inf, -np.inf], np.nan)

        # Better fallback with validation
        if atr_series.isna().all() or len(ohlc) < 14:
            self.log_message("[WARN] ATR calculation failed, using High-Low range", True)
            atr_series = (ohlc["High"] - ohlc["Low"]).rolling(14, min_periods=5).mean()

        # Ensure no NaN values
        ohlc["ATR"] = atr_series.ffill().bfill()

        # Final safety check
        if ohlc["ATR"].iloc[-1] is None or pd.isna(ohlc["ATR"].iloc[-1]):
            fallback_atr = (ohlc["High"].iloc[-1] - ohlc["Low"].iloc[-1]) * 1.5
            ohlc["ATR"].iloc[-1] = fallback_atr
            self.log_message(f"[ATR-FALLBACK] Using H-L * 1.5 = {fallback_atr:.2f}", True)
        ohlc["atr"] = ohlc["ATR"]
        # ATR baseline (MA50) + regime ratio (EXPANSION/COMPRESSION)
        ohlc["ATR_MA50"] = ohlc["ATR"].rolling(50, min_periods=10).mean()
        ohlc["ATR_RATIO"] = (ohlc["ATR"] / ohlc["ATR_MA50"]).replace([np.inf, -np.inf], np.nan)
        ohlc["ATR_RATIO"] = ohlc["ATR_RATIO"].fillna(1.0)


        # Bands & EMAs
        ohlc = bollinger_bands(ohlc)
        for span in (5, 9, 21, 50, 200):
            ohlc = ema(ohlc, span)

        ohlc["EMA20"] = ohlc["Close"].ewm(span=20, adjust=False).mean()
        # --- MACD (12/26/9) + histogram (+ prev) + stds (robust color) ---
        ema_fast    = ohlc["Close"].ewm(span=12, adjust=False).mean()
        ema_slow    = ohlc["Close"].ewm(span=26, adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist   = macd_line - macd_signal
        ohlc["MACD_hist"] = macd_hist

        # statistics we later use
        macd_spread = macd_line - macd_signal
        hist_std    = macd_hist.rolling(50, min_periods=10).std()
        spread_std  = macd_spread.rolling(50, min_periods=10).std()

        hist_now  = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else None
        hist_prev = float(macd_hist.iloc[-2]) if pd.notna(macd_hist.iloc[-2]) else None
        line_now  = float(macd_line.iloc[-1]) if pd.notna(macd_line.iloc[-1]) else None
        sig_now   = float(macd_signal.iloc[-1]) if pd.notna(macd_signal.iloc[-1]) else None

        # ===== robust color classification =====
        hist_std_last = float(hist_std.iloc[-1]) if pd.notna(hist_std.iloc[-1]) else None
        eps = 1e-7
        if hist_std_last is not None and hist_std_last > 0:
            eps = max(eps, 0.05 * hist_std_last)  # 5% of σ50

        macd_color = "Neutral"
        if hist_now is not None and hist_prev is not None:
            d = hist_now - hist_prev
            if   hist_now >=  eps and d >  +eps: macd_color = "Dark Green"
            elif hist_now >=  eps and d <= +eps: macd_color = "Light Green"
            elif hist_now <= -eps and d <  -eps: macd_color = "Dark Red"
            elif hist_now <= -eps and d >= -eps: macd_color = "Light Red"
            # else Neutral (|hist_now| < eps)

        # ===== 'flat' detector (line≈signal AND tiny histogram) =====
        spread_std_last = float(spread_std.iloc[-1]) if pd.notna(spread_std.iloc[-1]) else None
        macd_is_flat = False
        if (line_now is not None and sig_now is not None and
            spread_std_last is not None and spread_std_last > 0 and
            hist_std_last is not None and hist_std_last > 0 and
            hist_now is not None):
            near_line = abs(line_now - sig_now) <= 0.10 * spread_std_last
            tiny_hist = abs(hist_now)           <= 0.20 * hist_std_last
            macd_is_flat = near_line and tiny_hist

        # ADX(+DI/−DI)
        high, low, close = ohlc["High"], ohlc["Low"], ohlc["Close"]
        plus_dm  = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr_cols  = pd.concat([high - low,
                            (high - close.shift()).abs(),
                            (low - close.shift()).abs()], axis=1)
        tr    = tr_cols.max(axis=1)
        atr14 = tr.rolling(14, min_periods=14).mean()

        plus_di  = 100 * (plus_dm.rolling(14, min_periods=14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14, min_periods=14).mean() / atr14)
        dx   = 100 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di))
        adx  = dx.rolling(14, min_periods=14).mean()
        ohlc["ADX_series"] = adx
        ohlc["DI_plus"]    = plus_di
        ohlc["DI_minus"]   = minus_di

        # S/R + Volume
        ohlc["Support"]    = ohlc["Low"].rolling(20).min()
        ohlc["Resistance"] = ohlc["High"].rolling(20).max()
        ohlc["VolSMA20"]   = ohlc["Volume"].rolling(20, min_periods=1).mean()
        ohlc["volume_sma_20"] = ohlc["VolSMA20"]

        # [OK] NEW: Volume Ratio (Volume / Average)
        ohlc["volume_ratio"] = ohlc["Volume"] / ohlc["VolSMA20"]

        # [OK] NEW: Volume Surge Detection
        ohlc["volume_surge"] = ohlc["volume_ratio"] > 1.5  # 50% above average

        # [OK] NEW: Extreme Volume (for very strong moves)
        ohlc["volume_extreme"] = ohlc["volume_ratio"] > 2.0  # 100% above average

        # ==========================================
        # [START] MOMENTUM INDICATORS
        # ==========================================

        # [OK] NEW: 10-Period Momentum (Price - Price[10])
        ohlc["momentum_10"] = ohlc["Close"].diff(10)

        # [OK] NEW: Momentum Percentage
        ohlc["momentum_pct"] = ((ohlc["Close"] - ohlc["Close"].shift(10)) /
                                ohlc["Close"].shift(10) * 100)

        # [OK] NEW: Rate of Change (ROC) - Alternative momentum
        ohlc["roc_10"] = ((ohlc["Close"] - ohlc["Close"].shift(10)) /
                          ohlc["Close"].shift(10) * 100)

        # [OK] NEW: Acceleration (momentum of momentum)
        ohlc["acceleration"] = ohlc["momentum_10"].diff(3)

        # RSI calculation
        delta_rsi = ohlc['Close'].diff()
        gain = delta_rsi.clip(lower=0)
        loss = -delta_rsi.clip(upper=0)
        avg_g = gain.rolling(14).mean()
        avg_l = loss.rolling(14).mean()
        rs = avg_g / avg_l.replace(0, np.nan)
        ohlc['RSI'] = 100 - (100 / (1 + rs))

        # ==========================================
        # [SIGNAL] COMBINED SIGNALS
        # ==========================================

        # [OK] NEW: Strong Bearish Signal
        ohlc["strong_bearish"] = (
                (ohlc["momentum_pct"] < -0.5) &  # Dropping fast
                (ohlc["volume_ratio"] > 1.3)  # With volume
        )

        # [OK] NEW: Strong Bullish Signal
        ohlc["strong_bullish"] = (
                (ohlc["momentum_pct"] > 0.5) &  # Rising fast
                (ohlc["volume_ratio"] > 1.3)  # With volume
        )

        if len(ohlc) < 2:
            self.bot.log_message(
                f"IndicatorCalc: Not enough data points ({len(ohlc)}) for {symbol} to calculate indicators.",
                False
            )
            return {"error": "Not enough data for indicators."}

        latest, prev = ohlc.iloc[-1], ohlc.iloc[-2]

        # EMA50/200 summary (safe)
        try:
            ema50 = EMA50_200(self.bot.fyers_sdk_instance, symbol, timeframe, 40)
            ema_sig = ema50.get_current_signal()
            if len(ema50.df) < 50:
                ema_sig = {"signal": "NO TRADE", "trend": "Neutral", "trend_strength": 0}
        except Exception:
            ema_sig = {"signal": "NO TRADE", "trend": "Neutral", "trend_strength": 0}

        # Supertrend (21/14/7)
        #st21, _, _, tr21 = supertrend(ohlc, 21, 1)
        #st14, _, _, tr14 = supertrend(ohlc, 14, 2)
        #st_fast, _, _, tr_fast = supertrend(ohlc, period=7, multiplier=2.5)  # Quick exits
        #st_main, _, _, tr_main = supertrend(ohlc, period=10, multiplier=3.0)  # Main trend
        st_main, _, _, tr_main = supertrend(ohlc, period=21, multiplier=1.0)  # Less noise
        st_fast, _, _, tr_fast = supertrend(ohlc, period=5, multiplier=2.0)  # Very responsive
        #st_main, _, _, tr_main = supertrend(ohlc, period=7, multiplier=2.5)
        #st7,  _, _, tr7  = supertrend(ohlc, 7,  3)

        # Extras (unchanged)
        bull_div, bear_div = rsi_divergence(ohlc)
        patterns   = self.bot.candle_analyzer.detect_patterns(ohlc)
        ohlc_for_adx = fetchOHLC1(symbol, interval=str(timeframe), duration=60)
        adx_bundle = adx_efi_mom_trade_signal(ohlc_for_adx, symbol)
        fib        = fibonacci_retracement(ohlc, logger=self.bot.log_message)

        # =============================
        # SUPER TREND (14,2) & (21,1)
        # =============================
        _, _, _, tr14_2 = supertrend(ohlc, period=14, multiplier=2)
        _, _, _, tr21_1 = supertrend(ohlc, period=21, multiplier=1)

        # =============================
        # ELDER FORCE
        # =============================
        elder = elder_force_index(ohlc)

        # =============================
        # MOMENTUM
        # =============================
        mom = momentum_indicator(ohlc, period=10)

        # --- NEW: CPR & AI Analysis ---
        #cpr_analysis = analyze_cpr_strategy(latest.to_dict(), pivot_data or {}, self.bot.ai_predictor)
        # NEW CALL (add ohlc_df):
        cpr_analysis = analyze_cpr_strategy(
            indicators={**latest.to_dict(), "symbol": symbol},
            pivot_data=pivot_data or {},
            ai_predictor=self.bot.ai_predictor,
            ohlc_df=ohlc  # [OK] Pass full OHLC dataframe for candle pattern detection
        )

        return {
            "symbol": symbol,
            "pivot_data": pivot_data,
            "timestamp": latest.name.isoformat(),
            "close": float(latest["Close"]),
            "close_prev": float(prev["Close"]),
            "high_prev": float(prev["High"]),
            "low_prev": float(prev["Low"]),

            # EMAs (+prev)
            "ema_5": float(latest["EMA5"]), "ema_9": float(latest["EMA9"]), "ema_21": float(latest["EMA21"]),
            "ema_50": float(latest["EMA50"]), "ema_200": float(latest["EMA200"]),
            "ema_5_prev": float(prev["EMA5"]), "ema_9_prev": float(prev["EMA9"]),
            "ema_20": float(latest["EMA20"]),
            "ema_21_prev": float(prev["EMA21"]), "ema_50_prev": float(prev["EMA50"]),
            "ema_200_prev": float(prev["EMA200"]),

            # Volume (+prev, +SMA20)
            #"volume": float(latest["Volume"]),
            #"volume_prev": float(prev["Volume"]),
            #"volume_sma_20": float(ohlc["VolSMA20"].iloc[-1]),
            # [OK] NEW: Volume indicators
            "volume": float(latest["Volume"]),
            "volume_prev": float(prev["Volume"]),
            "volume_sma_20": float(ohlc["VolSMA20"].iloc[-1]),
            "volume_ratio": float(latest["volume_ratio"]) if pd.notna(latest["volume_ratio"]) else 1.0,
            "volume_surge": bool(latest["volume_surge"]) if pd.notna(latest["volume_surge"]) else False,
            "volume_extreme": bool(latest["volume_extreme"]) if pd.notna(latest["volume_extreme"]) else False,

            # [OK] NEW: Momentum indicators
            "momentum_10": float(latest["momentum_10"]) if pd.notna(latest["momentum_10"]) else 0.0,
            "momentum_pct": float(latest["momentum_pct"]) if pd.notna(latest["momentum_pct"]) else 0.0,
            "roc_10": float(latest["roc_10"]) if pd.notna(latest["roc_10"]) else 0.0,
            "acceleration": float(latest["acceleration"]) if pd.notna(latest["acceleration"]) else 0.0,
            
            # Section 2 additions
            "st_14_2_signal": int(tr14_2.iloc[-1]) if not tr14_2.empty else 0,
            "st_21_1_signal": int(tr21_1.iloc[-1]) if not tr21_1.empty else 0,
            "elder_force_now": float(elder.iloc[-1]) if pd.notna(elder.iloc[-1]) else 0.0,
            "elder_force_prev": float(elder.iloc[-2]) if pd.notna(elder.iloc[-2]) else 0.0,
            "momentum_now": float(mom.iloc[-1]) if pd.notna(mom.iloc[-1]) else 0.0,
            "momentum_prev": float(mom.iloc[-2]) if pd.notna(mom.iloc[-2]) else 0.0,

            # [OK] NEW: Combined signals
            "strong_bearish": bool(latest["strong_bearish"]) if pd.notna(latest["strong_bearish"]) else False,
            "strong_bullish": bool(latest["strong_bullish"]) if pd.notna(latest["strong_bullish"]) else False,

            # MACD (+line/signal & stds for strategy)
            "macd_line": line_now,
            "macd_signal": sig_now,
            "macd_hist": hist_now,
            "macd_hist_prev": hist_prev,
            "macd_hist_std50": (float(hist_std.iloc[-1]) if pd.notna(hist_std.iloc[-1]) else None),
            "macd_spread_std50": (float(spread_std.iloc[-1]) if pd.notna(spread_std.iloc[-1]) else None),
            "macd_color": macd_color,
            "macd_is_flat": macd_is_flat,

            # ADX (+prev)
            "adx": float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else None,
            "adx_prev": float(adx.iloc[-2]) if pd.notna(adx.iloc[-2]) else None,

            # Other indicators
            "supertrend": float(st_main.iloc[-1]) if pd.notna(st_main.iloc[-1]) else None,
            "st_main": int(tr_main.iloc[-1]) if not tr_main.empty else 0,
            #"supertrend": float(st21.iloc[-1]) if pd.notna(st21.iloc[-1]) else None,
            "BB_upper": float(latest.get("BB_upper")) if pd.notna(latest.get("BB_upper")) else None,
            "BB_lower": float(latest.get("BB_lower")) if pd.notna(latest.get("BB_lower")) else None,
            "BB_mid": float(latest.get("BB_mid")) if pd.notna(latest.get("BB_mid")) else None,
            "VWAP": float(latest["VWAP"]),
            "vwap": float(latest["VWAP"]), # Lowercase for compatibility
            "support": float(latest["Support"]), "resistance": float(latest["Resistance"]),
            # Add RSI
            "rsi": float(ohlc["RSI"].iloc[-1]) if "RSI" in ohlc.columns else (float(latest["RSI"]) if "RSI" in latest else 50.0),
            # --- CORRECTED: Bollinger Bandwidth for volatility filter ---
            "bb_bandwidth": ((latest.get("BB_upper", 0) - latest.get("BB_lower", 0)) / latest.get("BB_mid", 1))
                            if (latest.get("BB_mid") and latest.get("BB_mid") > 0 and
                                latest.get("BB_upper") and latest.get("BB_lower"))
                            else 0.0,

            "ATR": float(latest["ATR"]),

            
            "atr": float(latest["ATR"]),
            "atr_ma50": float(latest["ATR_MA50"]) if "ATR_MA50" in latest and pd.notna(latest.get("ATR_MA50")) else 0.0,
            "atr_ratio": float(latest["ATR_RATIO"]) if "ATR_RATIO" in latest and pd.notna(latest.get("ATR_RATIO")) else 1.0,
# EMA50/200 summary
            "ema50_200_signal": ema_sig["signal"],
            "ema50_200_trend":  ema_sig["trend"] ,

            # Ultimate MA / Super-Guppy summary & ST trend flags
            "super_guppy": UltimateMAIndicator(ohlc, {"len": 13, "len2": 34}).summarized(),
            "supertrend_main": float(st_main.iloc[-1]) if pd.notna(st_main.iloc[-1]) else None,
            "supertrend_fast": float(st_fast.iloc[-1]) if pd.notna(st_fast.iloc[-1]) else None,
            "st_main_trend": int(tr_main.iloc[-1]) if not tr_main.empty else 0,
            "st_fast_trend": int(tr_fast.iloc[-1]) if not tr_fast.empty else 0,
            #"st21Trend": int(tr21.iloc[-1]) if not tr21.empty else 0,
            #"st14Trend": int(tr14.iloc[-1]) if not tr14.empty else 0,
            #"st7Trend":  int(tr7.iloc[-1])  if not tr7.empty  else 0,

            # Patterns, ADX/EFI/MOM bundle, Fib, RSI divergences
            "patterns": patterns,
            "rsi": adx_bundle[6] if len(adx_bundle) > 6 else (float(ohlc["rsi"].iloc[-1]) if "rsi" in ohlc else None),
            "adx_efi": {
                "signal": adx_bundle[0], "ADX": adx_bundle[1], "DI+": adx_bundle[2],
                "DI-": adx_bundle[3], "Momentum": adx_bundle[4], "EFI": adx_bundle[5]
            },
            "fib": fib,
            "rsi_div": {"bull": bull_div, "bear": bear_div},
            "cpr_analysis": cpr_analysis,
        }



class FyersService:
    def __init__(self, fyers_sdk, raw_log_path, log_fn, websocket_ltp_fn=None):
        self.sdk     = fyers_sdk
        self.raw_log = raw_log_path
        self.log     = log_fn
        self.get_websocket_ltp = websocket_ltp_fn

    def place_market_order(self, symbol, side, qty):
        self.log(f"Market {side} {symbol} qty={qty}", debug_only=True)
        ltp = self.get_websocket_ltp(symbol, timeout=5) if self.get_websocket_ltp else None

        payload = {
            "symbol": symbol,
            "qty": int(qty),
            "type": 2,
            "side": (1 if side == "BUY" else -1),
            "productType": "MARGIN",
            "limitPrice": 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": 0,
            "takeProfit": 0
        }
        resp = call_with_rate_limit_retry(self.sdk.place_order, data=payload)

        now = dt.datetime.now(IST).isoformat()
        event = {
            "timestamp": now,
            "action":    "place_market",
            "symbol":    symbol,
            "side":      side,
            "qty":       qty,
            "ltp":       ltp,
            "payload":   payload,
            "response":  resp
        }
        save_order_data_event(event, self.raw_log)
        return resp, None

    def exit_all_positions_for_symbol(self, symbol: str):
        """
        Close ONLY the position matching `symbol` using the positions DELETE API.
        - Finds the matching position id from GET /positions (netPositions).
        - Sends {"id": <position_id>} to DELETE /positions.
        - Does NOT send `symbol` in the delete payload (the API ignores/doesn't expect it).
        """
        self.log(f"Exit (single symbol) request for {symbol}", debug_only=False)

        # 1) Find the position id for this symbol
        pos_id = None
        positions = call_with_rate_limit_retry(self.sdk.positions) or {}
        net_positions = positions.get("netPositions") or []
        try:
            for p in net_positions:
                candidates = [p.get("symbol"), p.get("symbolName"), p.get("tradingSymbol"), p.get("segmentSymbol")]
                p_symbol = next((s for s in candidates if isinstance(s, str) and s.strip()), None)
                if (p_symbol or "").strip() != symbol:
                    continue
                raw_qty = (p.get("netQty") or p.get("qty") or p.get("net_quantity") or p.get("quantity"))
                try:
                    net_qty = float(str(raw_qty).strip())
                except Exception:
                    net_qty = 0.0
                if abs(net_qty) < 1e-6:
                    continue
                pos_id = p.get("id") or p.get("positionId") or p.get("posId")
                if pos_id:
                    break
        except Exception as e:
            self.log(f"[EXIT] Failed to parse positions: {e}", debug_only=False)

        if not pos_id:
            self.log(f"[EXIT] No open position id found for {symbol}.", debug_only=False)
            resp = {"s": "error", "code": -66, "message": "No open position for symbol"}
            event = {
                "timestamp": dt.datetime.now(IST).isoformat(),
                "action":    "exit_symbol",
                "symbol":    symbol,
                "ltp":       self.get_websocket_ltp(symbol, timeout=5) if self.get_websocket_ltp else None,
                "payload":   {"id": None, "note": "no matching open position id"},
                "response":  resp
            }
            save_order_data_event(event, self.raw_log)
            return resp, None

        payload = {"id": pos_id}
        ltp = self.get_websocket_ltp(symbol, timeout=5) if self.get_websocket_ltp else None
        resp = call_with_rate_limit_retry(self.sdk.exit_positions, data=payload)

        event = {
            "timestamp": dt.datetime.now(IST).isoformat(),
            "action":    "exit_symbol",
            "symbol":    symbol,
            "ltp":       ltp,
            "payload":   payload,
            "response":  resp
        }
        save_order_data_event(event, self.raw_log)
        return resp, None



def save_order_data_event(event, path):
    lst = robust_load_json(path, print, default=[])
    if not isinstance(lst, list):
        lst = []
    lst.append(convert_to_serializable(event))
    robust_save_json(lst, path, print, debug_only=True)


# ==============================
# ENHANCED OrderManager CLASS WITH AI CPR
# ==============================
class OrderManager:
    # -----------------------------
    # CONFIGURATION CONSTANTS
    # -----------------------------
    # Paper trading mode
    PAPER_TRADING_MODE = False

    # AI CPR Configuration
    AI_CPR_ENABLED = True
    AI_MIN_CONF = 0.25  # Minimum confidence for AI signal acceptance
    AI_SOLO_MIN_CONF = 0.35  # Higher threshold for AI-only entries
    AI_SOLO_MAX_OPPOSITION = 2  # Max opposing signals for AI solo entry
    AI_GATE_TRADES = True  # Require AI confirmation for entries

    # Entry Thresholds
    MIN_VOTES_STRONG = 2  # Strong confluence (50% agreement)
    MIN_VOTES_MEDIUM = 1  # Medium confluence
    MEDIUM_CONF_MIN_SCORE = 1.5  # Minimum score for 2-vote entry without AI
    MEDIUM_CONF_MIN_AI = 0.30  # Minimum AI confidence for 2-vote entry with AI
    MEDIUM_CONF_MIN_SCORE_WITH_AI = 1.3  # Lower score if AI involved

    # Risk Management
    STRONG_TREND_ATR_GAP = 1.5
    CONSOL_TOL_NARROW_ATR = 0.30
    PCT_FALLBACK = 0.0015
    PCT_FALLBACK_NARROW = 0.0010
    #INITIAL_SL_ATR = 1.5
    #TRAIL_ATR = 1.5
    #BREAKEVEN_TRIGGER_R = 1.0
    CANDLE_SL_TRIGGER_R = 1.5
    EMERGENCY_SL_MULTIPLIER = 1.5  # Exit if loss exceeds 1.5x initial risk
    MAX_LOSS_PERCENT = 0.015  # Max 1.5% loss on position

    # Volatility Filter
    BB_BANDWIDTH_THRESHOLD = 0.005  # Below this = choppy market

    # Cooldown & Deduplication
    #FLIP_COOLDOWN_BARS = 1
    FLIP_COOLDOWN_BARS = 0
    MAX_REPEATED_EXITS_PER_BAR = 1
    DEDUPE_ONE_ENTRY_PER_BAR = True
    DEDUPE_BY_REGIME_AFTER_FAIL = True
    INITIAL_SL_ATR = 1.5  # 1.2 ATR initial stop (20% tighter)
    TRAIL_ATR = 1.2  # 1.0 ATR trailing (faster lock-in)
    BREAKEVEN_TRIGGER_R = 0.8  # Move to breakeven after 0.8R (was 1.0R)

    # Signal mode
    SIGNAL_MODE = "both"  # "both", "macd", "ema"

    # Dynamic CPR Stop Loss
    CPR_SL_BUFFER_ATR_MULTIPLIER = 0.2  # 20% of ATR as buffer
    CPR_SL_PRICE_TOLERANCE_PCT = 0.10  # Max 10% distance from CPR levels
    VOLUME_RATIO_THRESHOLD_HIGH = 1.8  # For high-volume stocks
    VOLUME_RATIO_THRESHOLD_LOW = 0.5  # For low-volume futures (Natural Gas)
    VOLUME_RATIO_THRESHOLD_MIN = 0.3  # Absolute minimum


    def _get_volume_threshold(self, volume_ratio, symbol):
        """
        Adaptive volume threshold based on recent volume patterns
        """
        # For futures (MCX), use lower thresholds
        if "MCX:" in symbol or "NATGAS" in symbol:
            # Natural Gas typically has lower volume
            return self.VOLUME_RATIO_THRESHOLD_LOW

        # For stocks, use higher thresholds
        return self.VOLUME_RATIO_THRESHOLD_HIGH

    def __init__(self, fyers_service, symbol, lot_size, log_fn, state_path, ai_predictor: CPR_AIPredictor, event_log=None, bot=None, is_option=False, point_value=None, **_):
        self.IST = pytz.timezone("Asia/Kolkata")
        self.last_trend_signal = None
        self.bot = bot
        self.df = None
        self.svc        = fyers_service
        self.symbol     = symbol
        self.lot        = int(lot_size)
        self.log        = log_fn
        self.state_path = state_path
        self.event_log  = event_log
        self.bot        = bot  # Add bot reference
        self.is_option  = is_option
        
        # Point value mapping
        if point_value is not None:
            self.point_value = point_value
        elif "NATGAS" in symbol:
            self.point_value = 250
        elif "SENSEX" in symbol and ("CE" in symbol or "PE" in symbol):
            self.point_value = 10
        else:
            self.point_value = 1.0
            
        self.lock       = threading.Lock()
        self.report_dir = _.get("report_dir")
        self.trades_csv = None
        self.ai_predictor = ai_predictor
        if not hasattr(self, "position"):
            self.position = {}
        self.position.setdefault("trail_active", False)
        self.position.setdefault("trail_start_profit", 5)  # Profit (in points) after which trail starts
        self.position.setdefault("trail_gap", 2)  # Lock profit (e.g. 2 points below peak)
        self.position.setdefault("max_profit", 0)
        self.position.setdefault("recent_trail_exit", False)
        self.position.setdefault("max_favorable_excursion", 0.0)
        self.position.setdefault("emergency_sl", None)

        # [OK] NEW: Production Safety Features
        self.daily_loss = 0
        self.daily_loss_limit = 250  # ₹500 max loss per day (5% of ₹10k)
        self.trades_today = 0
        self.max_trades_per_day = 5
        self.TRADING_HALTED = False
        self.last_reset_date = dt.datetime.now(IST).date()
        # Early signal settings
        self.REJECTION_MIN_CONFIDENCE = 0.70  # Minimum for rejection signals
        self.VOLUME_BREAKOUT_MIN_RATIO = 2.0  # Volume must be 2x+ average
        self.MOMENTUM_SHIFT_CANDLES = 3  # Consecutive candles for shift
        self.REJECTION_WICK_MULTIPLIER = 2.0  # Wick must be 2x body
        self.REJECTION_VOLUME_MIN = 1.5
        self.MOMENTUM_SHIFT_CANDLES = 3
        self.price_action_analyzer = PriceActionAnalyzer(bot)

        if "NATGAS" in symbol:
            self.EMERGENCY_SL_MULTIPLIER = 1.2  # Tighter (was 1.5)
            self.MAX_LOSS_PERCENT = 0.012  # 1.2% (was 1.5%)
            self.BID_ASK_BUFFER = 0.10  # 0.10 points = ₹25
            self.FLASH_CRASH_THRESHOLD = 300  # ₹300 in 2 seconds
            self.INITIAL_SL_ATR = 1.2  # Tighter initial SL (was 1.5)
        elif self.is_option:
            self.MAX_LOSS_PERCENT = 0.30  # 30% for options
            self.INITIAL_SL_ATR = 2.0  # Wider for options
        else:
            self.MAX_LOSS_PERCENT = 0.015  # Default 1.5%
            self.INITIAL_SL_ATR = 1.5

            self.log("[NATGAS-SL] Using optimized stop loss settings", False)

        # Entry priority (higher = checked first)
        self.SIGNAL_PRIORITY = {
            "rejection_candle": 10,  # Highest priority
            "volume_breakout": 9,
            "ai_cpr": 8,
            "momentum_shift": 7,
            "cpr_strategy": 6,
            "trend": 5,
            "macd": 4,
            "ema_cross": 3,
            "price_action_trend": 2,
            "volume_momentum": 1
        }
        # 🆕 ADAPTIVE SUPERTREND SETTINGS
        self.SUPERTREND_MODE = "ADAPTIVE"  # Options: "STRICT", "ADAPTIVE", "OFF"

        # Confidence thresholds for bypassing SuperTrend
        self.ST_BYPASS_HIGH_CONFIDENCE = 0.85  # Very strong signal
        self.ST_BYPASS_MEDIUM_CONFIDENCE = 0.75  # Strong signal + confirmation

        # Market regime detection
        self.last_regime = "UNKNOWN"  # TRENDING, CHOPPY, VOLATILE
        self.regime_confidence = 0.0


        # Link AI predictor to this order manager
        if self.ai_predictor:
            self.ai_predictor.order_manager = self

        if self.report_dir:
            os.makedirs(self.report_dir, exist_ok=True)
            self.trades_csv = os.path.join(
                self.report_dir,
                f"{self.symbol.replace(':','_')}_trades.csv"
            )
            self._ensure_trade_csv()

        if self.PAPER_TRADING_MODE:
            self.log("<<<<< PAPER TRADING MODE IS ACTIVE >>>>>", False)

        self._load_state()

        # Market data cache
        self.last_known_ltp = None
        self.last_known_inds = None
        self.last_known_primary_tf = None

        # AI CPR state
        self.last_ai_action = None
        self.last_ai_confidence = 0.0
        self.ai_entry_attempt_bar = None

    # ---------- CORE HELPER METHODS ----------
    @staticmethod
    def _f(v, alt=None):
        try:
            if v is None: return alt
            x = float(v)
            return x if isfinite(x) else alt
        except Exception:
            return alt

    @staticmethod
    def _cross_up(a_prev, b_prev, a, b):
        return (a_prev is not None and b_prev is not None and a is not None and b is not None
                and a_prev <= b_prev and a > b)

    @staticmethod
    def _cross_dn(a_prev, b_prev, a, b):
        return (a_prev is not None and b_prev is not None and a is not None and b is not None
                and a_prev >= b_prev and a < b)

    @staticmethod
    def _near(a, b, tol):
        return (a is not None and b is not None and abs(a - b) <= tol)

    def _atr_gap(self, a, b, atr):
        a, b, atr = self._f(a), self._f(b), self._f(atr)
        if a is None or b is None or (atr is None or atr == 0):
            return None
        return abs(a - b) / atr

    def _now_iso(self):
        try:
            return dt.datetime.now(IST).isoformat()
        except Exception:
            return dt.datetime.utcnow().isoformat()

    def _get_atr_with_fallback(self, inds_tf: dict, price: float):
        atr_val = self._f(inds_tf.get("ATR")) if isinstance(inds_tf, dict) else None
        if atr_val is None:
            atr_val = self._f(inds_tf.get("atr")) if isinstance(inds_tf, dict) else None
        if (atr_val is None or atr_val <= 0) and price is not None:
            atr_val = float(price) * self.PCT_FALLBACK
            self.log(f"[ENGINE] ATR fallback engaged: {atr_val:.6f}", True)
        return atr_val
    
    def calculate_dynamic_profit_targets(self, entry_price: float, atr: float = None, 
                                          volatility: float = None) -> dict:
        """
        Calculate dynamic stop loss, take profit, and trailing parameters based on ATR.
        Replaces static values like SL=60, TP=120 with volatility-adjusted values.
        
        Args:
            entry_price: Entry price of the position
            atr: Current ATR value (if None, uses fallback calculation)
            volatility: Optional volatility measure (IV or VIX)
        
        Returns:
            Dict with:
            - sl_points: Stop loss in points
            - tp_points: Take profit in points
            - sl_price: Stop loss price
            - tp_price: Take profit price
            - trail_start: Profit points after which trailing starts
            - trail_gap: Gap to maintain below peak profit
            - risk_reward: Risk-reward ratio
        """
        # Use ATR if provided, else estimate from price
        if atr is None or atr <= 0:
            atr = entry_price * self.PCT_FALLBACK
        
        # Base ATR multipliers (adjusted for symbol type)
        if "NATGAS" in self.symbol.upper():
            # Natural Gas is volatile, use wider stops
            sl_mult = 1.5
            tp_mult = 2.5
            trail_start_mult = 1.0
            trail_gap_mult = 0.5
        elif "SENSEX" in self.symbol.upper() or "NIFTY" in self.symbol.upper():
            # Index options - tighter parameters due to leverage
            sl_mult = 1.2
            tp_mult = 2.0
            trail_start_mult = 0.8
            trail_gap_mult = 0.4
        else:
            # Default for equities/other
            sl_mult = self.INITIAL_SL_ATR
            tp_mult = self.INITIAL_SL_ATR * 2  # 2:1 RR
            trail_start_mult = 1.0
            trail_gap_mult = 0.5
        
        # Adjust for volatility if provided
        if volatility is not None:
            if volatility > 0.30:  # High volatility
                sl_mult *= 1.3
                tp_mult *= 1.4
                trail_start_mult *= 1.2
            elif volatility < 0.15:  # Low volatility
                sl_mult *= 0.8
                tp_mult *= 0.8
                trail_start_mult *= 0.8
        
        # Calculate points
        sl_points = max(atr * sl_mult, entry_price * 0.005)  # Min 0.5%
        tp_points = max(atr * tp_mult, entry_price * 0.01)   # Min 1.0%
        trail_start = max(atr * trail_start_mult, entry_price * 0.003)
        trail_gap = max(atr * trail_gap_mult, entry_price * 0.002)
        
        # Apply caps
        sl_points = min(sl_points, entry_price * 0.05)  # Max 5% SL
        tp_points = min(tp_points, entry_price * 0.15)  # Max 15% TP
        
        result = {
            "sl_points": round(sl_points, 4),
            "tp_points": round(tp_points, 4),
            "sl_price": round(entry_price - sl_points, 4),
            "tp_price": round(entry_price + tp_points, 4),
            "trail_start": round(trail_start, 4),
            "trail_gap": round(trail_gap, 4),
            "risk_reward": round(tp_points / sl_points, 2) if sl_points > 0 else 0,
            "atr_used": round(atr, 4),
        }
        
        self.log(
            f"[DYNAMIC-PROFIT] Entry={entry_price:.2f} ATR={atr:.4f} | "
            f"SL={result['sl_points']:.2f} TP={result['tp_points']:.2f} "
            f"RR={result['risk_reward']:.2f}",
            True
        )
        
        return result
    
    def update_dynamic_trail_settings(self, atr: float, entry_price: float = None):
        """
        Update trailing stop parameters dynamically based on current ATR.
        Called after entry to adjust trail_start_profit and trail_gap.
        
        Args:
            atr: Current ATR value
            entry_price: Entry price (optional, for logging)
        """
        if atr is None or atr <= 0:
            return
        
        # Calculate dynamic trail settings
        trail_start = max(atr * 1.0, 2.0)  # At least 2 points
        trail_gap = max(atr * 0.5, 1.0)    # At least 1 point
        
        # Update position settings
        self.position["trail_start_profit"] = round(trail_start, 2)
        self.position["trail_gap"] = round(trail_gap, 2)
        
        self.log(
            f"[DYNAMIC-TRAIL] Updated trail settings: "
            f"start_after={trail_start:.2f} pts, gap={trail_gap:.2f} pts "
            f"(ATR={atr:.4f})",
            True
        )



    def _norm_tf(self, all_inds, tf: str):
        tf_str = str(tf)
        root = all_inds or {}
        if isinstance(root, list):
            pick = next((x for x in root if isinstance(x, dict) and isinstance(x.get("Dashboard"), dict)), None)
            if pick is not None:
                root = pick["Dashboard"]
            else:
                pick = next((x for x in root if isinstance(x, dict) and isinstance(x.get(tf_str), dict)), None)
                root = pick if pick is not None else {}
        if isinstance(root, dict) and isinstance(root.get("Dashboard"), dict):
            root = root["Dashboard"]
        if not isinstance(root, dict): return {}
        d = root.get(tf_str)
        if d is None: d = root.get(int(tf_str), {})
        if not isinstance(d, dict): return {}
        inds = d.get("inds")
        return inds if isinstance(inds, dict) else d

    def _load_state(self):
        self.position = robust_load_json(self.state_path, self.log, default={})

        # [OK] ADD THESE VALIDATION CHECKS:
        # Ensure position has valid type
        if "type" not in self.position or self.position["type"] not in ["FLAT", "BUY", "SELL"]:
            self.log(f"[STATE-FIX] Invalid position type: {self.position.get('type')}, resetting to FLAT", False)
            self.position["type"] = "FLAT"

        # Ensure order_id exists
        if "order_id" not in self.position:
            self.position["order_id"] = None

        # Ensure other required fields exist
        self.position.setdefault("_last_bar_key", None)
        self.position.setdefault("_last_action_bar", None)
        self.position.setdefault("_exits_this_bar", 0)
        self.position.setdefault("initialized", False)
        self.position.setdefault("_skip_entry_until_bar", None)

        self.log(f"State loaded: Position is {self.position.get('type', 'FLAT')}", True)

    def _save_state(self):
        robust_save_json(self.position, self.state_path, self.log)
        self.log(f"State saved: Position is now {self.position.get('type', 'FLAT')}", True)

    # ---------- TRADE EXECUTION METHODS ----------
    def _process_entry(self, side, reason, ltp, atr, bar_key=None, indsP=None):

        # [OK] NEW: Reset daily counters if new day
        today = dt.datetime.now(self.IST).date()
        if today != self.last_reset_date:
            self.daily_loss = 0
            self.trades_today = 0
            self.TRADING_HALTED = False
            self.last_reset_date = today
            self.log("[SAFETY] Daily counters reset for new trading day", False)

        # [OK] NEW: Check if trading halted
        if self.TRADING_HALTED:
            self.log(
                f"[ALERT] [SAFETY] Trading HALTED - Circuit breaker active\n"
                f"  Daily Loss: ₹{self.daily_loss}\n"
                f"  Trades Today: {self.trades_today}",
                False
            )
            return False

        # [OK] NEW: Check daily loss limit
        if self.daily_loss <= -self.daily_loss_limit:
            self.log(
                f"[ALERT] [SAFETY] Daily loss limit reached\n"
                f"  Loss: ₹{self.daily_loss} / ₹{self.daily_loss_limit}\n"
                f"  No more entries today",
                False
            )
            return False

        # [OK] NEW: Check max trades per day
        if self.trades_today >= self.max_trades_per_day:
            self.log(
                f"[WARNING] [SAFETY] Max trades per day reached\n"
                f"  Trades: {self.trades_today} / {self.max_trades_per_day}\n"
                f"  No more entries today",
                False
            )
            return False

        # [OK] FIX: Normalize position type first
        current_pos = self.position.get("type", "FLAT")
        if current_pos not in ["FLAT", "BUY", "SELL"]:
            self.log(
                f"[WARNING] [STATE-CORRUPTION] Invalid position type: {current_pos}, resetting to FLAT",
                False
            )
            self.position["type"] = "FLAT"
            current_pos = "FLAT"

        # Check 1: Already in position?
        if current_pos != "FLAT":
            self.log(
                f"[WARNING] [ENTRY-BLOCKED] Already in {current_pos} position | "
                f"Order ID: {self.position.get('order_id')}",
                False
            )
            return False

        # Check 2: Just exited on this bar?
        if self.position.get("_last_action_bar") == bar_key:
            self.log(
                f"[WARNING] [ENTRY-BLOCKED] Just exited on bar {bar_key} | "
                f"Wait for next candle (cooldown active)",
                False
            )
            return False

        # Check 3: Valid ATR?
        if atr is None or atr <= 0:
            self.log(
                f"[WARNING] [ENTRY-BLOCKED] Invalid ATR: {atr} | Cannot calculate stop loss",
                False
            )
            return False

        # Check 4: Valid LTP?
        if ltp is None or ltp <= 0:
            self.log(
                f"[WARNING] [ENTRY-BLOCKED] Invalid LTP: {ltp} | Cannot place order",
                False
            )
            return False

        indsP = indsP or {}
        self.position["_last_entry_attempt_bar"] = bar_key

        if atr is None or atr <= 0:
            self.log("ENTRY BLOCKED: ATR unavailable.", False)
            return False

        resp, _ = self.svc.place_market_order(self.symbol, side, self.lot)

        order_id = None
        if isinstance(resp, dict):
            order_id = resp.get("id") or resp.get("orderId")
            if not order_id and isinstance(resp.get("data"), dict):
                order_id = resp["data"].get("id") or resp["data"].get("orderId")

        # success or paper-override on margin shortfall patterns
        is_live_success = isinstance(resp, dict) and resp.get("s") == "ok" and order_id
        is_paper_override = (
                self.PAPER_TRADING_MODE and isinstance(resp, dict) and
                (resp.get("code") == -99 or "margin" in str(resp.get("message", "")).lower())
        )

        if is_live_success or is_paper_override:
            if is_paper_override:
                self.log("PAPER TRADE: Margin error detected. Simulating successful entry.")
                order_id = order_id or f"PAPER-{int(time.time())}"

            r_mult = self.INITIAL_SL_ATR * atr
            initial_sl = ltp - r_mult if side == "BUY" else ltp + r_mult
            now_iso = self._now_iso()

            # Round initial SL to nearest exchange tick size to avoid rejections
            try:
                ts = get_tick_size(self.svc.sdk, self.symbol)
            except Exception:
                ts = None
            if ts and ts > 0:
                initial_sl = round(round(initial_sl / ts) * ts, 2)

            # 🔥 Extract AI confidence for logging
            ai_confidence = self.position.get("ai_entry_confidence", 0.0)

            # Save position state (use update to preserve preset fields)
            pos_update = {
                "type": side,
                "order_id": order_id,
                "entry_price": ltp,
                "stop_loss": initial_sl,
                "r_mult": r_mult,
                "breakeven_set": False,
                "ts": now_iso,
                "max_profit": 0.0,
                "trail_active": False,
                "ai_entry_confidence": ai_confidence,  # 🔥 Store AI confidence
                "_last_bar_key": self.position.get("_last_bar_key"),
                "_last_action_bar": bar_key,
                "_exits_this_bar": 0
            }
            self.position.update(pos_update)

            # 🔥 Enhanced logging with AI confidence
            self.log(
                f"{'PAPER ' if is_paper_override else ''}ENTRY SUCCESS: {side} {self.symbol}. "
                f"Reason: {reason}. Entry: {ltp:.2f} | Initial SL: {initial_sl:.2f} | 1R={r_mult:.2f}"
                f"{f' | AI Confidence: {ai_confidence:.3f}' if ai_confidence > 0 else ''}",
                False
            )

            self._append_trade_csv({
                "trade_id": order_id, "symbol": self.symbol, "side": side, "event": "ENTRY",
                "entry_time": now_iso, "entry_ltp": ltp, "reason": reason, "order_id": order_id,
                "bar_key": bar_key,
                "adx": self._f(indsP.get("adx")),
                "macd_color": indsP.get("macd_color"),
                "ema5": self._f(indsP.get("ema_5")),
                "ema9": self._f(indsP.get("ema_9")),
                "ema21": self._f(indsP.get("ema_21")),
                "ai_confidence": ai_confidence  # 🔥 Log to CSV
            })
            self.log(f"DEBUG: ENTRY event logged to CSV for {side} {self.symbol}", True)
            self._save_state()
            if hasattr(self, 'sl_manager'):
                self.sl_manager.position = self.position
            return True

        # Genuine failure
        msg = (resp or {}).get("message", "Unknown Error")
        self.log(f"ENTRY FAILED: {msg}", False)
        now_iso = self._now_iso()
        self._append_trade_csv({
            "trade_id": f"{self.symbol}-{int(time.time())}",
            "symbol": self.symbol,
            "side": side,
            "event": "ENTRY_FAIL",
            "entry_time": now_iso,
            "entry_ltp": ltp,
            "reason": msg,
            "order_id": (resp or {}).get("id"),
            "bar_key": bar_key,
            "adx": self._f(indsP.get("adx")),
            "macd_color": indsP.get("macd_color"),
            "ema5": self._f(indsP.get("ema_5")),
            "ema9": self._f(indsP.get("ema_9")),
            "ema21": self._f(indsP.get("ema_21"))
        })
        # [OK] ADD: Store entry details for later PnL calculation
        self.position["_perf_entry_time"] = now_iso
        self.position["_perf_entry_ltp"] = ltp
        self._save_state()
        return False


    def _process_exit(self, reason, ltp):
        """
        Simulates exits in paper mode; otherwise sends exit to broker.
        """
        if not self.position or self.position.get("type") == "FLAT":
            return True

        side      = self.position.get("type")
        entry_ltp = self._f(self.position.get("entry_price"))
        entry_ts  = self.position.get("ts")
        order_id  = self.position.get("order_id")

        resp = None
        if self.PAPER_TRADING_MODE:
            self.log(f"PAPER TRADE: Simulating exit for {self.symbol}. Reason: {reason}")
            resp = {"s": "ok", "code": 200, "message": "Paper Trade Exit"}
        else:
            resp, _ = self.svc.exit_all_positions_for_symbol(self.symbol)

        if isinstance(resp, dict) and (resp.get("s") == "ok" or resp.get("code") in [-66, 204]):
            now_iso = self._now_iso()
            log_msg_prefix = "PAPER EXIT SUCCESS" if self.PAPER_TRADING_MODE else "EXIT SUCCESS"
            self.log(f"{log_msg_prefix}. Reason: {reason}")

            hold_sec = 0
            try:
                if entry_ts:
                    t0 = pd.to_datetime(entry_ts)
                    t1 = pd.to_datetime(now_iso)
                    hold_sec = max(0, int((t1 - t0).total_seconds()))
            except Exception:
                pass

            ltp_diff = None
            if entry_ltp is not None and ltp is not None:
                ltp_diff = (ltp - entry_ltp) if side == "BUY" else (entry_ltp - ltp)

            if ltp_diff is not None:
                # [OK] NEW: Update daily loss tracking
                self.daily_loss += ltp_diff
                self.trades_today += 1

                self.log(
                    f"[SAFETY] Daily Stats Updated:\n"
                    f"  Trade P&L: ₹{ltp_diff:.2f}\n"
                    f"  Daily Total: ₹{self.daily_loss:.2f} / ₹{self.daily_loss_limit}\n"
                    f"  Trades Today: {self.trades_today} / {self.max_trades_per_day}",
                    False
                )

                # [OK] NEW: Circuit breaker check (3 consecutive losses totaling ₹300)
                if self.check_circuit_breaker():
                    self.TRADING_HALTED = True
                    self.log("[ALERT][ALERT][ALERT] CIRCUIT BREAKER ACTIVATED [ALERT][ALERT][ALERT]", False)

            try:
                self._append_trade_csv({
                    "trade_id": order_id, "symbol": self.symbol, "side": side, "event": "EXIT",
                    "entry_time": entry_ts, "exit_time": now_iso,
                    "hold_seconds": hold_sec,
                    "entry_ltp": entry_ltp, "exit_ltp": ltp, "ltp_diff": ltp_diff,
                    "reason": reason, "order_id": order_id,
                    "bar_key": self.position.get("_last_bar_key", "")
                })
                self.log(f"DEBUG: EXIT event logged to CSV for {side} {self.symbol}", True)
            except Exception as e:
                self.log(f"[ERROR] Failed to log EXIT to CSV in _process_exit: {e}", False)


            self.position["last_type"] = side  # Record the last exited position type

            self.position["type"] = "FLAT"
            self.position["exit_price"] = ltp
            self.position["exit_ts"] = now_iso
            # Reset state after exit
            self.position = {
                "_last_bar_key": self.position.get("_last_bar_key"),
                "_last_action_bar": self.position.get("_last_action_bar"),
                "_last_entry_attempt_bar": self.position.get("_last_entry_attempt_bar"),
                "_exits_this_bar": self.position.get("_exits_this_bar", 0),
                "_cooldown_bars": self.FLIP_COOLDOWN_BARS,
                "_last_exit_side": side,
                "type": "FLAT"
            }
            self.position["_skip_entry_until_bar"] = self.position.get("_last_bar_key") #Kiran added
            self._save_state()
            return True

        self.log(f"EXIT FAILED: {resp.get('message', 'Unknown Error') if isinstance(resp, dict) else 'No response'}")
        return False

    def _is_macd_flipped(self, inds):
        if not inds or "MACD" not in inds or "MACD_SIGNAL" not in inds:
            return False
        return (
                (self.position["type"] == "BUY" and inds["MACD"] < inds["MACD_SIGNAL"]) or
                (self.position["type"] == "SELL" and inds["MACD"] > inds["MACD_SIGNAL"])
        )

    def _is_exit_signal(self, inds):
        if not inds:
            return False
        st = inds.get("SUPERTREND")
        ltp = inds.get("LTP")
        if st is None or ltp is None:
            return False

        if self.position["type"] == "BUY" and ltp < st:
            return True
        if self.position["type"] == "SELL" and ltp > st:
            return True
        return False

    def check_circuit_breaker(self):
        """Emergency stop on rapid losses"""
        # Need at least 3 trades
        if hasattr(self, 'perf_tracker') and len(self.bot.perf_tracker.trades) >= 3:
            last_3_trades = self.bot.perf_tracker.trades[-3:]
            total_loss = sum(t['pnl'] for t in last_3_trades if t['pnl'] < 0)

            # 3% of ₹10k = ₹300 in 3 trades triggers circuit breaker
            if total_loss <= -300:
                self.log(
                    f"[ALERT] CIRCUIT BREAKER TRIGGERED [ALERT]\n"
                    f"Last 3 trades total: ₹{total_loss:.2f}\n"
                    f"Threshold: ₹-300\n"
                    f"HALTING ALL TRADING UNTIL MANUAL REVIEW",
                    False
                )
                return True
        return False


    def _get_supertrend_alignment(self, symbol, timeframes=None):
        """
        Check SuperTrend alignment across multiple timeframes.
        Returns (direction, confidence) where direction is "BUY", "SELL", or None
        """
        timeframes = timeframes or ["5", "15", "30"]

        if not self.bot or not hasattr(self.bot, 'ohlc_cache'):
            return None, 0.0

        try:
            aligned_votes = {"BUY": 0, "SELL": 0}
            total_valid = 0

            for tf in timeframes:
                ohlc_df = self.bot.ohlc_cache.get(f"{self.symbol}_{tf}")
                if ohlc_df is None or len(ohlc_df) < 21:
                    continue

                # Calculate SuperTrend for this timeframe
                st_val, _, _, _ = supertrend(ohlc_df, period=21, multiplier=1.0)
                ltp = ohlc_df['Close'].iloc[-1]

                if st_val is not None:
                    if ltp > st_val:
                        aligned_votes["BUY"] += 1
                    else:
                        aligned_votes["SELL"] += 1
                    total_valid += 1

            if total_valid == 0:
                return None, 0.0

            # Determine consensus direction
            buy_count = aligned_votes["BUY"]
            sell_count = aligned_votes["SELL"]

            if buy_count > sell_count:
                direction = "BUY"
                confidence = buy_count / total_valid
            elif sell_count > buy_count:
                direction = "SELL"
                confidence = sell_count / total_valid
            else:
                direction = None
                confidence = 0.0

            return direction, confidence

        except Exception as e:
            self.log(f"[SUPERTREND-ALIGNMENT-ERROR] {e}", True)
            return None, 0.0

    def _check_intelligent_exit(self, ltp, inds):
        """
        Intelligent exit system that predicts reversals BEFORE hitting SL
        Exits early when max profit starts falling
        """
        pos = self.position
        if not pos or pos.get("type") == "FLAT":
            return False

        entry = pos.get("entry_price")
        side = pos["type"]

        if not entry:
            return False
            # ==========================================
            # 🆕 RULE 1: EMA-5 CONFIRMATION EXIT
            # ==========================================
        current_bar_ts = inds.get("timestamp")
        last_checked_bar = pos.get("_last_ema5_check_bar")

        # Get 15-min trend for additional confirmation
        tf_15_inds = None
        trend_15m = "NEUTRAL"
        if hasattr(self, 'bot') and self.bot:
            try:
                # Fetch 15-min indicators for cross-timeframe confirmation
                tf_15_inds = self.bot.indicator_calculator.calculate_indicators(
                    self.bot.symbol, "15", pivot_data={}
                )
                if "error" not in tf_15_inds:
                    st_15m_trend = tf_15_inds.get("st_main_trend", 0)
                    trend_15m = "GREEN" if st_15m_trend > 0 else "RED" if st_15m_trend < 0 else "NEUTRAL"
            except Exception as e:
                self.log(f"[EMA5-EXIT] Failed to get 15m trend: {e}", True)

        # [OK] CHECK 1: On candle close (patient exit)
        if current_bar_ts and current_bar_ts != last_checked_bar:
            pos["_last_ema5_check_bar"] = current_bar_ts

            ema5 = self._f(inds.get("ema_5"))
            close = self._f(inds.get("close"))
            open_price = self._f(inds.get("open"))
            st_main_trend = inds.get("st_main_trend", 0)

            if ema5 and close and open_price:
                trend = "GREEN" if st_main_trend > 0 else "RED" if st_main_trend < 0 else "NEUTRAL"

                is_red_candle = close < open_price
                is_green_candle = close > open_price

                # 🔴 EXIT SELL when trend turns GREEN
                if side == "SELL" and trend == "GREEN":
                    closed_above_ema5 = close > ema5

                    # [OK] Check 15-min trend alignment
                    trend_15m_confirms = (trend_15m == "GREEN")

                    if is_red_candle and closed_above_ema5:
                        current_profit = entry - ltp
                        profit_rupees = current_profit * 250

                        confirmation_msg = ""
                        if trend_15m_confirms:
                            confirmation_msg = " [OK] 15m trend confirms"
                        else:
                            confirmation_msg = " [WARNING] 15m trend mixed"

                        self.log(
                            f"[ALERT] [EMA5-EXIT] Exiting SELL position\n"
                            f"  ⏰ Candle CLOSED: {current_bar_ts}\n"
                            f"  [DATA] 30m Trend: GREEN | 15m Trend: {trend_15m}{confirmation_msg}\n"
                            f"  🕯️ Red candle: O:{open_price:.2f} → C:{close:.2f}\n"
                            f"  [OK] Close {close:.2f} > EMA-5 {ema5:.2f}\n"
                            f"  💰 Profit: ₹{profit_rupees:.0f}\n"
                            f"  [SIGNAL] EXIT NOW",
                            False
                        )
                        return self._process_exit("EMA-5 Reversal (candle closed)", ltp)

                    elif is_red_candle and not closed_above_ema5:
                        self.log(
                            f"⏸️ [EMA5-HOLD] Holding SELL\n"
                            f"  [DATA] 30m: GREEN, 15m: {trend_15m}\n"
                            f"  Close {close:.2f} ≤ EMA-5 {ema5:.2f}\n"
                            f"  [SIGNAL] HOLD - no confirmation",
                            True
                        )

                # 🟢 EXIT BUY when trend turns RED
                elif side == "BUY" and trend == "RED":
                    closed_below_ema5 = close < ema5
                    trend_15m_confirms = (trend_15m == "RED")

                    if is_green_candle and closed_below_ema5:
                        current_profit = ltp - entry
                        profit_rupees = current_profit * 250

                        confirmation_msg = ""
                        if trend_15m_confirms:
                            confirmation_msg = " [OK] 15m trend confirms"
                        else:
                            confirmation_msg = " [WARNING] 15m trend mixed"

                        self.log(
                            f"[ALERT] [EMA5-EXIT] Exiting BUY position\n"
                            f"  ⏰ Candle CLOSED: {current_bar_ts}\n"
                            f"  [DATA] 30m Trend: RED | 15m Trend: {trend_15m}{confirmation_msg}\n"
                            f"  🕯️ Green candle: O:{open_price:.2f} → C:{close:.2f}\n"
                            f"  [OK] Close {close:.2f} < EMA-5 {ema5:.2f}\n"
                            f"  💰 Profit: ₹{profit_rupees:.0f}\n"
                            f"  [SIGNAL] EXIT NOW",
                            False
                        )
                        return self._process_exit("EMA-5 Reversal (candle closed)", ltp)

                    elif is_green_candle and not closed_below_ema5:
                        self.log(
                            f"⏸️ [EMA5-HOLD] Holding BUY\n"
                            f"  [DATA] 30m: RED, 15m: {trend_15m}\n"
                            f"  Close {close:.2f} ≥ EMA-5 {ema5:.2f}\n"
                            f"  [SIGNAL] HOLD - no confirmation",
                            True
                        )

        # ⚡ CHECK 2: Emergency exit (don't wait for candle close)
        # Only triggers on large adverse moves
        ema5 = self._f(inds.get("ema_5"))
        if ema5 and ltp:
            st_main_trend = inds.get("st_main_trend", 0)
            trend = "GREEN" if st_main_trend > 0 else "RED" if st_main_trend < 0 else "NEUTRAL"

            if side == "SELL" and trend == "GREEN":
                current_loss = entry - ltp
                loss_rupees = current_loss * 250
                ema5_distance_pct = ((ltp - ema5) / ema5 * 100) if ema5 else 0

                # [ALERT] Emergency: Losing AND price far above EMA-5 AND 15m confirms
                emergency_condition = (
                        loss_rupees < -250 and
                        ema5_distance_pct > 1.0 and
                        (trend_15m == "GREEN" or trend_15m == "NEUTRAL")  # 15m not opposing
                )

                if emergency_condition:
                    self.log(
                        f"⚡ [EMERGENCY-EMA5-EXIT] Fast exit - large reversal\n"
                        f"  💥 Loss: ₹{loss_rupees:.0f}\n"
                        f"  [DATA] Price {ema5_distance_pct:.2f}% above EMA-5\n"
                        f"  🔴 30m: GREEN, 15m: {trend_15m}\n"
                        f"  [SIGNAL] EXIT NOW (not waiting for candle close)",
                        False
                    )
                    return self._process_exit("Emergency EMA-5 Exit (volatile)", ltp)

            elif side == "BUY" and trend == "RED":
                current_loss = ltp - entry
                loss_rupees = current_loss * 250
                ema5_distance_pct = ((ema5 - ltp) / ema5 * 100) if ema5 else 0

                emergency_condition = (
                        loss_rupees < -250 and
                        ema5_distance_pct > 1.0 and
                        (trend_15m == "RED" or trend_15m == "NEUTRAL")
                )

                if emergency_condition:
                    self.log(
                        f"⚡ [EMERGENCY-EMA5-EXIT] Fast exit - large reversal\n"
                        f"  💥 Loss: ₹{loss_rupees:.0f}\n"
                        f"  [DATA] Price {ema5_distance_pct:.2f}% below EMA-5\n"
                        f"  🔴 30m: RED, 15m: {trend_15m}\n"
                        f"  [SIGNAL] EXIT NOW (not waiting for candle close)",
                        False
                    )
                    return self._process_exit("Emergency EMA-5 Exit (volatile)", ltp)

        # Calculate current profit
        if side == "BUY":
            current_profit = ltp - entry
        else:
            current_profit = entry - ltp

        # Track maximum profit achieved
        max_profit = pos.get("_max_profit", 0)
        if current_profit > max_profit:
            pos["_max_profit"] = current_profit
            max_profit = current_profit

        # ==========================================
        # RULE 2: Exit if profit drops 30%+ from peak
        # ==========================================
        if max_profit > 0:
            profit_drawdown = max_profit - current_profit
            drawdown_pct = (profit_drawdown / max_profit) * 100

            # Tiered exit based on how much profit was made
            if max_profit >= 6:  # Made 2R+
                max_drawdown_allowed = 20  # Allow only 20% drop
            elif max_profit >= 4:  # Made 1R+
                max_drawdown_allowed = 30  # Allow 30% drop
            else:
                max_drawdown_allowed = 40  # Allow 40% drop (still building profit)

            if drawdown_pct >= max_drawdown_allowed:
                # [OK] Check if ultra-tight SL is closer
                current_sl = self._f(self.position.get("stop_loss"))
                if current_sl:
                    if side == "BUY":
                        sl_distance = abs(ltp - current_sl)
                    else:
                        sl_distance = abs(current_sl - ltp)

                    # If SL is within 0.5 points, let it hit naturally
                    if sl_distance <= 0.50:  # 0.5 points = ₹125
                        self.log(
                            f"[SIGNAL] [SL-CLOSE] SL at ₹{current_sl:.2f} is only {sl_distance:.2f} pts away\n"
                            f"  Skipping profit protection - let SL hit naturally for better fill",
                            False
                        )
                        return  # Don't exit yet

                self.log(
                    f"🛡️ [PROFIT-PROTECTION] Exiting early!\n"
                    f"  Max profit: ₹{max_profit:.2f}\n"
                    f"  Current profit: ₹{current_profit:.2f}\n"
                    f"  Dropped: {drawdown_pct:.1f}% (max allowed: {max_drawdown_allowed}%)\n"
                    f"  Better to lock in ₹{current_profit:.2f} than risk hitting SL",
                    False
                )
                return self._process_exit("Profit Protection - Early Exit", ltp)

        # ==========================================
        # RULE 3: AI Detects Reversal
        # ==========================================
        if hasattr(self, 'ai_predictor') and self.ai_predictor:
            try:
                # Get AI prediction
                pivot_data = {}  # Load from your pivot JSON

                # Fetch OHLC data for AI prediction
                ohlc_for_ai_exit = None
                if hasattr(self, 'bot') and self.bot:
                    try:
                        # [OK] FIX: Increased from 2 to 80 candles for sufficient AI context
                        _ptf = getattr(self, 'last_known_primary_tf', None) or getattr(self, 'tf_selected',
                                                                                       None) or '15'
                        try:
                            primary_tf = str(int(float(_ptf)))
                        except Exception:
                            primary_tf = str(_ptf)
                        ohlc_for_ai_exit = self.bot.fetch_ohlc(self.symbol, str(primary_tf), 80)
                        #ohlc_for_ai_exit = self.bot.fetch_ohlc(self.symbol, "5", 2)
                    except Exception as e:
                        self.log(f"[AI-EXIT] Failed to fetch OHLC: {e}", True)
                # [OK] FIX: Use the inds parameter that's passed to the function
                ai_label, ai_conf, ai_dist, feature_array = self.ai_predictor.predict(
                    indicators=inds,
                    pivot_data=pivot_data,
                    feature_builder=_build_ai_cpr_features,
                    ohlc_df=ohlc_for_ai_exit
                )

                # Check for high-confidence reversal
                if ai_label and ai_conf is not None and ai_conf >= 0.70:
                    if side == "BUY" and ai_label in ["SELL", "STRONG_SELL"]:
                        self.log(
                            f"🤖 [AI-REVERSAL] Exiting LONG position\n"
                            f"  AI predicts: {ai_label} (confidence: {ai_conf:.2f})\n"
                            f"  Current profit: ₹{current_profit:.2f}\n"
                            f"  Decision: Exit before reversal completes",
                            False
                        )
                        return self._process_exit(f"AI Reversal Detected: {ai_label}", ltp)

                    elif side == "SELL" and ai_label in ["BUY", "STRONG_BUY"]:
                        self.log(
                            f"🤖 [AI-REVERSAL] Exiting SHORT position\n"
                            f"  AI predicts: {ai_label} (confidence: {ai_conf:.2f})\n"
                            f"  Current profit: ₹{current_profit:.2f}\n"
                            f"  Decision: Exit before reversal completes",
                            False
                        )
                        return self._process_exit(f"AI Reversal Detected: {ai_label}", ltp)

            except Exception as e:
                self.log(f"[AI-REVERSAL] Check failed: {e}", True)


        # ==========================================
        # RULE 4: SuperTrend Multi-TF Alignment Flip
        # ==========================================
        try:
            aligned, conf = self._get_supertrend_alignment(self.symbol, timeframes=["5", "15", "30"])
            if side == "BUY" and aligned == "SELL" and conf >= 0.66 and max_profit > 1:
                self.log(
                    f"[DATA] [SUPERTREND-FLIP] Multi-TF flip to SELL (conf: {conf:.2f})\n"
                    f"  Profit: ₹{current_profit * 250:.0f}\n"
                    f"  Action: Exit before breakdown",
                False
                )
                return self._process_exit("Multi-TF SuperTrend Flip", ltp)

            if side == "SELL" and aligned == "BUY" and conf >= 0.66 and max_profit > 1:
                self.log(
                    f"[DATA] [SUPERTREND-FLIP] Multi-TF flip to BUY (conf: {conf:.2f})\n"
                    f"  Profit: ₹{current_profit * 250:.0f}\n"
                    f"  Action: Exit before breakout",
                    False
                )
                return self._process_exit("Multi-TF SuperTrend Flip", ltp)
        except Exception as e:
            self.log(f"[SUPERTREND-FLIP-ERROR] {e}", True)

        # ==========================================
        # RULE 5: Momentum Reversal
        # ==========================================
        momentum_pct = self._f(inds.get("momentum_pct"), 0.0)
        prev_momentum = pos.get("_prev_momentum", momentum_pct)
        pos["_prev_momentum"] = momentum_pct

        if max_profit > 4:  # Only for 1R+ profits
            if side == "BUY" and prev_momentum > 0.2 and momentum_pct < -0.1:
                self.log(f"⚡ [MOMENTUM-FLIP] BUY momentum turned negative", False)
                return self._process_exit("Momentum Reversal", ltp)
            elif side == "SELL" and prev_momentum < -0.2 and momentum_pct > 0.1:
                self.log(f"⚡ [MOMENTUM-FLIP] SELL momentum turned positive", False)
                return self._process_exit("Momentum Reversal", ltp)

        return False


    def _predict_exit_timing(self, ltp, inds, ohlc_df):
        """
        [SIGNAL] Predict if we should exit in next 1-3 candles
        Returns: (action: str, confidence: float, reason: str)
        Actions: "HOLD", "EXIT_SOON", "EXIT_NOW"
        """

        if not inds or ohlc_df is None or len(ohlc_df) < 10:
            return "HOLD", 0.5, "Insufficient data"

        side = self.position.get("type")
        entry = self._f(self.position.get("entry_price"))

        if not side or side == "FLAT" or not entry:
            return "HOLD", 0.5, "No position"

        # ==========================================
        # FEATURE EXTRACTION
        # ==========================================

        # 1. Trend Strength Decay
        ema5 = self._f(inds.get("ema_5"))
        ema9 = self._f(inds.get("ema_9"))
        ema21 = self._f(inds.get("ema_21"))
        adx = self._f(inds.get("adx"), 0)

        # 2. Momentum Decay
        momentum_pct = self._f(inds.get("momentum_pct"), 0.0)
        roc_10 = self._f(inds.get("roc_10"), 0.0)
        acceleration = self._f(inds.get("acceleration"), 0.0)

        # 3. Volume Exhaustion
        volume_ratio = self._f(inds.get("volume_ratio"), 1.0)

        # 4. Price Action
        latest = ohlc_df.iloc[-1]
        prev = ohlc_df.iloc[-2]
        prev2 = ohlc_df.iloc[-3] if len(ohlc_df) >= 3 else prev

        body_latest = abs(latest['Close'] - latest['Open'])
        body_prev = abs(prev['Close'] - prev['Open'])
        range_latest = latest['High'] - latest['Low']
        range_prev = prev['High'] - prev['Low']

        # ==========================================
        # SIGNAL SCORING (0-100)
        # ==========================================

        exhaustion_score = 0
        reversal_score = 0
        hold_score = 50  # Baseline

        # =========================================
        # EXHAUSTION SIGNALS (Trend dying slowly)
        # =========================================

        if side == "BUY":
            # Signal 1: EMA convergence (EMAs coming together)
            if ema5 and ema9 and ema21:
                ema5_9_gap = (ema5 - ema9) / ema9 * 100
                ema9_21_gap = (ema9 - ema21) / ema21 * 100

                # Check if gaps are shrinking
                if ema5_9_gap < 0.1 and ema9_21_gap < 0.2:  # Very close
                    exhaustion_score += 20
                    self.log(f"[EXIT-PRED] EMA convergence: 5/9 gap {ema5_9_gap:.2f}%", True)

            # Signal 2: Momentum decay
            if momentum_pct < 0.2 and momentum_pct > -0.2:  # Near zero
                exhaustion_score += 15
                self.log(f"[EXIT-PRED] Momentum dying: {momentum_pct:.2f}%", True)

            # Signal 3: ADX declining
            adx_prev = self._f(inds.get("adx_prev"), adx)
            if adx < adx_prev and adx < 25:
                exhaustion_score += 15
                self.log(f"[EXIT-PRED] ADX declining: {adx:.1f} < {adx_prev:.1f}", True)

            # Signal 4: Volume drying up
            if volume_ratio < 0.7:  # Below 70% of average
                exhaustion_score += 15
                self.log(f"[EXIT-PRED] Volume drying: {volume_ratio:.2f}x", True)

            # Signal 5: Shrinking candle bodies
            if body_latest < body_prev * 0.6:  # 40% smaller body
                exhaustion_score += 10
                self.log(f"[EXIT-PRED] Shrinking candles", True)

            # Signal 6: Higher high but lower close (weakening)
            if latest['High'] > prev['High'] and latest['Close'] < prev['Close']:
                exhaustion_score += 10
                self.log(f"[EXIT-PRED] Higher high, lower close", True)

        elif side == "SELL":
            # Mirror logic for short positions
            if ema5 and ema9 and ema21:
                ema5_9_gap = (ema9 - ema5) / ema9 * 100
                ema9_21_gap = (ema21 - ema9) / ema21 * 100

                if ema5_9_gap < 0.1 and ema9_21_gap < 0.2:
                    exhaustion_score += 20

            if momentum_pct > -0.2 and momentum_pct < 0.2:
                exhaustion_score += 15

            adx_prev = self._f(inds.get("adx_prev"), adx)
            if adx < adx_prev and adx < 25:
                exhaustion_score += 15

            if volume_ratio < 0.7:
                exhaustion_score += 15

            if body_latest < body_prev * 0.6:
                exhaustion_score += 10

            if latest['Low'] < prev['Low'] and latest['Close'] > prev['Close']:
                exhaustion_score += 10

        # =========================================
        # REVERSAL SIGNALS (Sharp turn coming)
        # =========================================

        if side == "BUY":
            # Signal 1: Big bearish candle after rally
            if (latest['Close'] < latest['Open'] and
                    body_latest > range_latest * 0.7 and  # Strong bear body
                    volume_ratio > 1.5):  # High volume
                reversal_score += 30
                self.log(f"[EXIT-PRED] Strong bearish candle with volume", True)

            # Signal 2: Upper wick rejection (tested resistance, failed)
            upper_wick = latest['High'] - max(latest['Open'], latest['Close'])
            if upper_wick > body_latest * 2:  # Wick 2x body
                reversal_score += 25
                self.log(f"[EXIT-PRED] Upper wick rejection", True)

            # Signal 3: Bearish divergence (price up, momentum down)
            if latest['Close'] > prev['Close'] and momentum_pct < 0:
                reversal_score += 20
                self.log(f"[EXIT-PRED] Bearish divergence", True)

            # Signal 4: Momentum flip
            if momentum_pct < -0.4:  # Turned strongly negative
                reversal_score += 15
                self.log(f"[EXIT-PRED] Momentum flipped negative", True)

        elif side == "SELL":
            # Mirror for shorts
            if (latest['Close'] > latest['Open'] and
                    body_latest > range_latest * 0.7 and
                    volume_ratio > 1.5):
                reversal_score += 30

            lower_wick = min(latest['Open'], latest['Close']) - latest['Low']
            if lower_wick > body_latest * 2:
                reversal_score += 25

            if latest['Close'] < prev['Close'] and momentum_pct > 0:
                reversal_score += 20

            if momentum_pct > 0.4:
                reversal_score += 15

        # =========================================
        # HOLD SIGNALS (Trend still strong)
        # =========================================

        if side == "BUY":
            if ema5 > ema9 > ema21 and adx > 25:
                hold_score += 20
            if momentum_pct > 0.3:
                hold_score += 15
            if volume_ratio > 1.0:
                hold_score += 10

        elif side == "SELL":
            if ema5 < ema9 < ema21 and adx > 25:
                hold_score += 20
            if momentum_pct < -0.3:
                hold_score += 15
            if volume_ratio > 1.0:
                hold_score += 10

        # ==========================================
        # DECISION LOGIC
        # ==========================================

        max_score = max(exhaustion_score, reversal_score, hold_score)
        confidence = max_score / 100.0

        if reversal_score >= 60:
            action = "EXIT_NOW"
            reason = f"Strong reversal signals ({reversal_score}/100)"

        elif exhaustion_score >= 60:
            action = "EXIT_SOON"
            reason = f"Trend exhaustion detected ({exhaustion_score}/100)"

        elif hold_score >= 50:
            action = "HOLD"
            reason = f"Trend still healthy ({hold_score}/100)"

        else:
            action = "HOLD"
            reason = "Neutral signals, holding position"

        self.log(
            f"[EXIT-PREDICTION]\n"
            f"  Action: {action} (confidence: {confidence:.2f})\n"
            f"  Scores: Exhaustion={exhaustion_score}, Reversal={reversal_score}, Hold={hold_score}\n"
            f"  Reason: {reason}",
            False if action != "HOLD" else True
        )

        return action, confidence, reason

    def _check_trailing_profit(self, ltp, inds=None):
        pos = self.position
        if not pos or pos.get("type") == "FLAT":
            return

        entry = pos.get("entry_price")
        if not entry:
            return

        side = pos["type"]
        current_sl = self._f(self.position.get("stop_loss"))
        if current_sl:
            if side == "BUY":
                ultra_tight_active = current_sl > (entry + 2.0)  # SL is 2+ points above entry
            else:
                ultra_tight_active = current_sl < (entry - 2.0)  # SL is 2+ points below entry

        profit = (ltp - entry) if side == "BUY" else (entry - ltp)
        profit_rupees = profit * self.lot * self.point_value

        # 🔥 NEW: Don't exit if profit < threshold and trend is strong
        profit_threshold = 1000 if not self.is_option else 500
        if profit_rupees < profit_threshold:
            if inds:
                ema5 = self._f(inds.get("ema_5"))
                ema9 = self._f(inds.get("ema_9"))
                ema21 = self._f(inds.get("ema_21"))
                adx = self._f(inds.get("adx"), 0)

                trend_strong = (
                        (side == "BUY" and ema5 > ema9 > ema21 and adx > 25) or
                        (side == "SELL" and ema5 < ema9 < ema21 and adx > 25)
                )

                if trend_strong:
                    self.log(
                        f"💎 [HOLD] Profit ₹{profit_rupees:.0f} < ₹1,100 but trend strong - continuing",
                        True
                    )
                    return
        # ═══════════════════════════════════════════════════════════
        # 🛑 MAX LOSS EXIT (BEFORE TRAILING PROFIT LOGIC)
        # ════════════════════════════════════════════════════════════════════════

        # Calculate loss percentage
        loss_pct = abs(profit / entry * 100) if entry > 0 else 0

        # Exit if loss exceeds threshold
        MAX_LOSS_PCT = (self.MAX_LOSS_PERCENT * 100) if hasattr(self, "MAX_LOSS_PERCENT") else 1.5

        if profit < 0 and loss_pct >= MAX_LOSS_PCT:
            self.log(
                f"\n🛑 MAX LOSS EXIT TRIGGERED 🛑\n"
                f"Position: {side}\n"
                f"Entry: ₹{entry:.2f}\n"
                f"Current: ₹{ltp:.2f}\n"
                f"Loss: ₹{abs(profit):.2f} ({loss_pct:.2f}%)\n"
                f"Max allowed: {MAX_LOSS_PCT}%\n"
                f"Exiting to prevent further loss...",
                False
            )

            try:
                self._process_exit(f"Max Loss Reached: -{loss_pct:.2f}%", ltp)
            except Exception as e:
                self.log(f"[ERROR] Max loss exit failed: {e}", False)

            return

        # Log current loss status if losing
        if profit < 0:
            self.log(
                f"[LOSS-MONITOR] Current loss: ₹{abs(profit):.2f} ({loss_pct:.2f}%) | "
                f"Will exit at {MAX_LOSS_PCT}%",
                True
            )
            return  # Don't run trailing profit logic when losing
        max_profit = pos.get("_max_profit", 0)
        if profit > max_profit:
            pos["_max_profit"] = profit
            max_profit = profit
        drawdown = max_profit - profit if max_profit > 0 else 0
        # [OK] Don't exit too early - wait for bigger profit first
        if max_profit < profit * 0.2:
            # Haven't made 20% of target yet - hold
            self.log(
                f"[TRAILING] Early stage - profit {profit:.2f} < 30% target {profit * 0.2:.2f} - HOLDING",
                True
            )
            return

        atr_val = self._get_atr_with_fallback(inds, ltp) if inds else None
        if atr_val is None or atr_val <= 0:
            atr_val = 1.0

        # === Trend strength detection (adaptive & defensive) ===
        trend_strong = False
        trend_very_strong = False

        self.log(
            f"[DEBUG] _check_trailing_profit called — inds={type(inds)} keys={list(inds.keys())[:10] if inds and isinstance(inds, dict) else None}",
            True
        )

        if inds and isinstance(inds, dict):
            e5 = self._f(inds.get("ema_5"))
            e9 = self._f(inds.get("ema_9"))
            e21 = self._f(inds.get("ema_21"))
            adx = self._f(inds.get("adx"))
            macd_color = str(inds.get("macd_color", "")).strip().lower()
            bb_bandwidth = self._f(inds.get("bb_bandwidth"))
            supertrend = self._f(inds.get("supertrend"))

            # [OK] More defensive price_above_st check
            price_above_st = False
            if supertrend is not None and ltp is not None:
                try:
                    price_above_st = float(ltp) > float(supertrend)
                except (ValueError, TypeError):
                    pass

            self.log(
                f"[DEBUG] Trend check — e5={e5}, e9={e9}, e21={e21}, "
                f"adx={adx}, macd_color='{macd_color}', bb_bw={bb_bandwidth}, "
                f"supertrend={supertrend}, price_above_st={price_above_st}",
                True
            )

            def macd_is_bullish(color):
                if not color:
                    return False
                return any(x in color for x in ["green", "up", "bull"])

            def macd_is_bearish(color):
                if not color:
                    return False
                return any(x in color for x in ["red", "down", "bear"])

            # [OK] More lenient trend detection (was failing too often)
            if side == "BUY":
                # Basic trend check
                basic_trend_ok = (
                        e5 is not None and e21 is not None and e5 > e21 * 0.998  # More lenient (was 0.999)
                )
                # Momentum confirmation (need at least 1)
                momentum_ok = (
                        (adx is not None and adx > 18) or
                        price_above_st or
                        macd_is_bullish(macd_color)
                )

                trend_strong = basic_trend_ok and momentum_ok

                # Very strong requires all aligned
                trend_very_strong = (
                        trend_strong and
                        e9 is not None and e5 > e9 > e21 and
                        adx is not None and adx > 30 and
                        macd_color and "dark green" in macd_color  # Strongest MACD
                )

            elif side == "SELL":
                basic_trend_ok = (
                        e5 is not None and e21 is not None and e5 < e21 * 1.002  # More lenient
                )
                momentum_ok = (
                        (adx is not None and adx > 18) or
                        (supertrend is not None and ltp < supertrend) or
                        macd_is_bearish(macd_color)
                )

                trend_strong = basic_trend_ok and momentum_ok

                trend_very_strong = (
                        trend_strong and
                        e9 is not None and e5 < e9 < e21 and
                        adx is not None and adx > 30 and
                        macd_color and "dark red" in macd_color
                )

            self.log(
                f"[TREND RESULT] Strong={trend_strong}, VeryStrong={trend_very_strong}, "
                f"BasicTrend={basic_trend_ok if 'basic_trend_ok' in locals() else 'N/A'}, "
                f"Momentum={momentum_ok if 'momentum_ok' in locals() else 'N/A'}",
                True
            )
        else:
            self.log(f"[WARN] No valid indicators - defaulting to WEAK trend", True)

        # === Progressive targets (same logic as before) ===
        if atr_val < 1.0:
            base_multiplier = 1.0
        elif atr_val < 2.0:
            base_multiplier = 1.5
        else:
            base_multiplier = 2.0

        profit_tier = pos.get("_profit_tier", 0)
        if trend_very_strong:
            tier_multipliers = [base_multiplier * 2.0, base_multiplier * 3.0, base_multiplier * 4.0]
            tier_label = "VERY_STRONG"
            trailing_pct = 0.10
        elif trend_strong:
            tier_multipliers = [base_multiplier * 1.5, base_multiplier * 2.5]
            tier_label = "STRONG"
            trailing_pct = 0.15
        else:
            tier_multipliers = [base_multiplier]
            tier_label = "WEAK"
            trailing_pct = 0.20

        current_tier = min(profit_tier, len(tier_multipliers) - 1)
        profit_target = tier_multipliers[current_tier] * atr_val

        # Move to next profit tier dynamically
        if profit >= profit_target and current_tier < len(tier_multipliers) - 1:
            pos["_profit_tier"] = current_tier + 1
            next_target = tier_multipliers[current_tier + 1] * atr_val
            self.log(
                f"[SIGNAL] TIER UP! Profit ₹{profit:.2f} hit {profit_target:.2f}. "
                f"Next target ₹{next_target:.2f} (Trend: {tier_label})", False
            )
            profit_target = next_target
            current_tier += 1

        # ==========================================
        # [OK] Exit logic
        # ==========================================
        if profit >= profit_target:
            if trend_strong or trend_very_strong:
                if drawdown >= trailing_pct * max_profit:
                    self.log(
                        f"[WARNING] {tier_label} trend trailing stop: Profit dropped {drawdown:.2f} "
                        f"({trailing_pct * 100:.0f}%) from peak ₹{max_profit:.2f} — exiting.", False
                    )
                    try:
                        self._process_exit(f"Trailing Stop ({tier_label} Trend, Tier {current_tier + 1})", ltp)
                    except Exception as e:
                        self.log(f"[ERROR] Trailing exit failed: {e}")
                    return
                else:
                    self.log(
                        f"💎 HOLDING {tier_label} trend — Profit: ₹{profit:.2f}, "
                        f"Target: ₹{profit_target:.2f}, Peak: ₹{max_profit:.2f}, "
                        f"Tier: {current_tier + 1}/{len(tier_multipliers)}", True
                    )
            else:
                self.log(f"💰 Profit target ₹{profit:.2f} reached in weak trend — exiting!", False)
                try:
                    self._process_exit("Profit Target Reached (Weak Trend)", ltp)
                except Exception as e:
                    self.log(f"[ERROR] Profit target exit failed: {e}")
                return

        # Early profit protection
        if max_profit >= profit_target * 0.4:
            drawdown_pct = (drawdown / max_profit * 100) if max_profit > 0 else 0
            if max_profit >= profit_target * 0.8:  # Made 80%+ of target
                max_drawdown_pct = 10  # Allow only 20% drop
            elif max_profit >= profit_target * 0.6:  # Made 60%+ of target
                max_drawdown_pct = 12  # Allow 25% drop
            else:  # Made 40-60% of target
                max_drawdown_pct = 15

            if drawdown >= (max_drawdown_pct / 100) * max_profit:
                self.log(
                    f"[WARNING] [PROFIT-PROTECTION] Exiting - Dropped {drawdown:.2f} ({drawdown_pct:.0f}%) from peak ₹{max_profit:.2f} | "
                    f"Max allowed: {max_drawdown_pct}% | Current profit: ₹{profit:.2f}",
                    False
                )
                try:
                    self._process_exit("Early Profit Protection", ltp)
                except Exception as e:
                    self.log(f"[ERROR] Profit protection exit failed: {e}")
                return

        # [OK] Improved trend reversal detection
        if inds and isinstance(inds, dict) and max_profit > 0:
            e5 = self._f(inds.get("ema_5"))
            e21 = self._f(inds.get("ema_21"))
            macd_color = str(inds.get("macd_color", "")).strip().lower()
            adx = self._f(inds.get("adx"))

            # [OK] More robust reversal check
            trend_reversed = False
            reversal_confidence = 0

            if side == "BUY":
                # 🔧 FIX #12: Enhanced EMA reversal detection (requires sustained closure below EMA)
                e5_prev = self._f(inds.get("ema_5_prev"))
                e21_prev = self._f(inds.get("ema_21_prev"))
                # Check multiple reversal signals
                ema_reversed = (
                        e5 is not None and e21 is not None and e5 < e21 * 0.998 and  # Current cross
                        e5_prev is not None and e5 < e5_prev  # Sustained downward pressure
                )
                #ema_reversed = (e5 is not None and e21 is not None and e5 < e21 * 0.998)
                macd_reversed = macd_color and any(x in macd_color for x in ["dark red", "light red", "red"])
                weak_momentum = (adx is not None and adx < 20)

                # Count reversal signals
                if ema_reversed:
                    reversal_confidence += 2  # EMA is strongest signal
                if macd_reversed:
                    reversal_confidence += 1
                if weak_momentum:
                    reversal_confidence += 1

                # 🔧 ADAPTIVE THRESHOLD based on ADX strength
                if adx is not None and adx > 35:
                    # Strong trend - require 3/4 signals (be patient)
                    required_confidence = 3
                else:
                    # Normal/weak trend - require 2/4 signals
                    required_confidence = 2

                trend_reversed = reversal_confidence >= required_confidence

                self.log(
                    f"[REVERSAL CHECK BUY] EMA_rev={ema_reversed}, MACD_rev={macd_reversed}, "
                    f"Weak_ADX={weak_momentum}, Confidence={reversal_confidence}/4, "
                    f"Required={required_confidence} (ADX: {adx:.1f})",
                    True
                )

            elif side == "SELL":
                # 🔧 FIX #12: Enhanced EMA reversal for SELL
                e5_prev = self._f(inds.get("ema_5_prev"))
                e21_prev = self._f(inds.get("ema_21_prev"))

                ema_reversed = (
                        e5 is not None and e21 is not None and e5 > e21 * 1.002 and  # Current cross
                        e5_prev is not None and e5 > e5_prev  # Sustained upward pressure
                )

                #ema_reversed = (e5 is not None and e21 is not None and e5 > e21 * 1.002)
                macd_reversed = macd_color and any(x in macd_color for x in ["dark green", "light green", "green"])
                weak_momentum = (adx is not None and adx < 20)

                if ema_reversed:
                    reversal_confidence += 2
                if macd_reversed:
                    reversal_confidence += 1
                if weak_momentum:
                    reversal_confidence += 1

                # 🔧 Same adaptive threshold for SELL
                if adx is not None and adx > 35:
                    required_confidence = 3
                else:
                    required_confidence = 2

                trend_reversed = reversal_confidence >= required_confidence

                self.log(
                    f"[REVERSAL CHECK SELL] EMA_rev={ema_reversed}, MACD_rev={macd_reversed}, "
                    f"Weak_ADX={weak_momentum}, Confidence={reversal_confidence}/4, "
                    f"Required={required_confidence} (ADX: {adx:.1f})",
                    True
                )

            if trend_reversed:
                self.log(
                    f"🔴 Trend REVERSED (confidence: {reversal_confidence}/{required_confidence}) at profit ₹{profit:.2f} — exiting immediately!",
                    False
                )
                try:
                    self._process_exit(f"Trend Reversal Exit (conf: {reversal_confidence}/{required_confidence})", ltp)
                except Exception as e:
                    self.log(f"[ERROR] Trend reversal exit failed: {e}")
                return
            else:
                self.log(
                    f"[REVERSAL] Not reversed yet (confidence: {reversal_confidence}/{required_confidence} needed)",
                    True
                )

        # Trailing status
        self.log(
            f"[TRAILING] {tier_label} trend — LTP={ltp}, Profit=₹{profit:.2f}, "
            f"Peak=₹{max_profit:.2f}, Target=₹{profit_target:.2f}, "
            f"Tier={current_tier + 1}/{len(tier_multipliers)}, Trail={trailing_pct * 100:.0f}%",
            True
        )

        self._save_state()


    # ---------- AI CPR ORDER METHODS ----------
    def ai_buy(self, symbol, qty):
        """Execute AI-driven BUY order"""
        self.log(f"[AI-CPR] Executing BUY order for {symbol}, qty: {qty}", False)
        return self.place_order(symbol, qty, side="BUY", tag="AI-CPR")

    def ai_sell(self, symbol, qty):
        """Execute AI-driven SELL order"""
        self.log(f"[AI-CPR] Executing SELL order for {symbol}, qty: {qty}", False)
        return self.place_order(symbol, qty, side="SELL", tag="AI-CPR")

    def ai_exit_all(self, symbol):
        """Execute AI-driven exit for all positions"""
        self.log(f"[AI-CPR] Exiting all positions for {symbol}", False)
        current_ltp_for_exit = self.last_known_ltp if self.last_known_ltp is not None else 0.0
        if current_ltp_for_exit == 0.0:
            self.log("[AI-CPR] Warning: Current LTP for AI exit is 0.0", False)
        return self._process_exit(reason="AI-CPR requested exit", ltp=current_ltp_for_exit)

    def execute_ai_cpr_strategy(self, ltp, all_inds, primary_tf="5"):
        """
        Execute AI CPR-based trading strategy independently
        """
        if not self.AI_CPR_ENABLED:
            return

        self.last_known_ltp = ltp
        self.last_known_inds = all_inds
        self.last_known_primary_tf = primary_tf

        def _tf(tf):
            return self._norm_tf(all_inds, str(tf))

        inds = _tf(primary_tf)
        if not isinstance(inds, dict) or not inds.get("timestamp"):
            self.log(f"[AI-CPR] No indicators for TF={primary_tf}.", True)
            return

        # Bar key for deduplication
        bar_key = f"{primary_tf}:{inds['timestamp']}"

        # Kiran Added this 3 lines
        if self.DEDUPE_ONE_ENTRY_PER_BAR and self.position.get("_last_entry_attempt_bar") == bar_key:
            self.log("AI entry blocked - already attempted this bar", True)
            return
        # --- ⛔ Re-entry Cooldown Check ---
        if self.position.get("_skip_entry_until_bar") == bar_key:
            self.log("[RE-ENTRY] Skipping new trade this bar (1-bar cooldown active)", True)
            return

        # Prevent multiple AI entries in same bar
        if self.ai_entry_attempt_bar == bar_key:
            return

        # Get CPR analysis
        cpr_analysis = inds.get("cpr_analysis", {})
        ai_label = cpr_analysis.get("ai_cpr_label")
        ai_confidence = cpr_analysis.get("ai_confidence", 0.0)
        ai_filter_pass = cpr_analysis.get("ai_filter_pass", True)

        # Update AI state
        self.last_ai_action = ai_label
        self.last_ai_confidence = ai_confidence

        # Execute AI trading logic
        if ai_label and ai_confidence and ai_confidence >= self.AI_MIN_CONF:
            action = ai_label.upper()

            if action == "BUY" and ai_filter_pass:
                self.log(f"[AI-CPR] STRONG BUY signal (confidence: {ai_confidence:.3f})", False)
                if self.position.get("type") != "BUY":
                    self.ai_exit_all(self.symbol)
                    if self.ai_buy(self.symbol, self.lot):
                        self.ai_entry_attempt_bar = bar_key
                        self._log_ai_trade("BUY", ai_confidence, cpr_analysis.get("reason", "AI Signal"))

            elif action == "SELL" and ai_filter_pass:
                self.log(f"[AI-CPR] STRONG SELL signal (confidence: {ai_confidence:.3f})", False)
                if self.position.get("type") != "SELL":
                    self.ai_exit_all(self.symbol)
                    if self.ai_sell(self.symbol, self.lot):
                        self.ai_entry_attempt_bar = bar_key
                        self._log_ai_trade("SELL", ai_confidence, cpr_analysis.get("reason", "AI Signal"))

            elif action in ["HOLD", "NEUTRAL"]:
                self.log(f"[AI-CPR] HOLD signal (confidence: {ai_confidence:.3f})", True)
                # Optionally exit positions on strong HOLD signal
                if self.position.get("type") != "FLAT" and ai_confidence > 0.7:
                    self.log(f"[AI-CPR] Exiting position due to strong HOLD signal", False)
                    self.ai_exit_all(self.symbol)

    def detect_market_regime(self, ohlc_df, indicators):
        """
        Detects current market regime to adjust SuperTrend strictness
        Returns: ("TRENDING"/"CHOPPY"/"VOLATILE", confidence)
        """
        if ohlc_df is None or len(ohlc_df) < 20:
            return "UNKNOWN", 0.0

        try:
            # Get indicators for regime detection
            adx = self._f(indicators.get("adx"), 0)
            bb_bandwidth = self._f(indicators.get("bb_bandwidth"), 0)
            atr = self._f(indicators.get("ATR"), 0)
            volume_ratio = self._f(indicators.get("volume_ratio"), 1.0)

            # Calculate price range compression
            recent = ohlc_df.iloc[-20:]
            high_low_range = (recent['High'].max() - recent['Low'].min()) / recent['Close'].mean()

            # ═══════════════════════════════════════════════════════════
            # TRENDING MARKET (Strong directional move)
            # ═══════════════════════════════════════════════════════════
            trending_score = 0

            if adx > 30:
                trending_score += 0.4  # Strong trend
            elif adx > 25:
                trending_score += 0.3
            elif adx > 20:
                trending_score += 0.2

            # Check EMA alignment
            ema5 = self._f(indicators.get("ema_5"))
            ema21 = self._f(indicators.get("ema_21"))
            ema50 = self._f(indicators.get("ema_50"))

            if ema5 and ema21 and ema50:
                if ema5 > ema21 > ema50:  # Perfect bull alignment
                    trending_score += 0.3
                elif ema5 < ema21 < ema50:  # Perfect bear alignment
                    trending_score += 0.3
                elif ema5 > ema21 or ema5 < ema21:  # Partial alignment
                    trending_score += 0.2

            # Volume confirmation
            if volume_ratio > 1.2:
                trending_score += 0.2

            if trending_score >= 0.7:
                self.log(
                    f"[REGIME] 📈 TRENDING detected (score: {trending_score:.2f})\n"
                    f"  ADX: {adx:.1f}, BB Width: {bb_bandwidth:.4f}\n"
                    f"  → SuperTrend will be LENIENT",
                    False
                )
                return "TRENDING", trending_score

            # ═══════════════════════════════════════════════════════════
            # CHOPPY MARKET (Sideways, no clear direction)
            # ═══════════════════════════════════════════════════════════
            choppy_score = 0

            if adx < 20:
                choppy_score += 0.4  # Weak trend

            if bb_bandwidth < 0.005:  # Very narrow bands
                choppy_score += 0.3

            if high_low_range < 0.02:  # Price compressed
                choppy_score += 0.2

            if volume_ratio < 0.8:  # Low volume
                choppy_score += 0.1

            if choppy_score >= 0.6:
                self.log(
                    f"[REGIME] 🔒 CHOPPY detected (score: {choppy_score:.2f})\n"
                    f"  ADX: {adx:.1f}, BB Width: {bb_bandwidth:.4f}\n"
                    f"  → SuperTrend will be STRICT",
                    False
                )
                return "CHOPPY", choppy_score

            # ═══════════════════════════════════════════════════════════
            # VOLATILE MARKET (Large swings, high ATR)
            # ═══════════════════════════════════════════════════════════
            volatile_score = 0

            if atr and indicators.get("close"):
                atr_pct = (atr / indicators["close"]) * 100
                if atr_pct > 2.0:
                    volatile_score += 0.4
                elif atr_pct > 1.5:
                    volatile_score += 0.3

            if bb_bandwidth > 0.015:  # Wide bands
                volatile_score += 0.3

            if volume_ratio > 1.5:  # High volume
                volatile_score += 0.2

            if high_low_range > 0.05:  # Large range
                volatile_score += 0.2

            if volatile_score >= 0.6:
                self.log(
                    f"[REGIME] ⚡ VOLATILE detected (score: {volatile_score:.2f})\n"
                    f"  ATR: {atr:.2f}, BB Width: {bb_bandwidth:.4f}\n"
                    f"  → SuperTrend will be MODERATE",
                    False
                )
                return "VOLATILE", volatile_score

            # Default to CHOPPY (safe mode)
            self.log(f"[REGIME] ❓ UNKNOWN (trending:{trending_score:.2f}, choppy:{choppy_score:.2f})", True)
            return "CHOPPY", 0.5

        except Exception as e:
            self.log(f"[REGIME] Error detecting regime: {e}", True)
            return "UNKNOWN", 0.0

    def should_bypass_supertrend(self, final_signal, signals, confidences, market_regime):
        """
        Decides if SuperTrend check should be skipped based on signal strength and market regime
        Returns: (should_bypass: bool, reason: str)
        """
        # ═══════════════════════════════════════════════════════════
        # MODE 1: SUPERTREND OFF (Not recommended)
        # ═══════════════════════════════════════════════════════════
        if self.SUPERTREND_MODE == "OFF":
            return True, "SuperTrend disabled by config"

        # ═══════════════════════════════════════════════════════════
        # MODE 2: STRICT MODE (Always enforce)
        # ═══════════════════════════════════════════════════════════
        if self.SUPERTREND_MODE == "STRICT":
            return False, "Strict mode - SuperTrend always enforced"

        # ═══════════════════════════════════════════════════════════
        # MODE 3: ADAPTIVE MODE (Smart filtering)
        # ═══════════════════════════════════════════════════════════

        regime, regime_conf = market_regime

        # Get signal strengths
        rejection_conf = confidences.get("rejection_candle", 0)
        volume_conf = confidences.get("volume_breakout", 0)
        ai_conf = confidences.get("ai_cpr", 0)

        has_rejection = signals.get("rejection_candle") == final_signal
        has_volume = signals.get("volume_breakout") == final_signal
        has_ai = signals.get("ai_cpr") == final_signal

        # Count total supporting signals
        signal_count = sum(1 for s in signals.values() if s == final_signal)

        # ──────────────────────────────────────────────────────────
        # BYPASS RULE 1: Very High Confidence Signal (0.85+)
        # ──────────────────────────────────────────────────────────
        if has_rejection and rejection_conf >= self.ST_BYPASS_HIGH_CONFIDENCE:
            return True, f"High confidence rejection ({rejection_conf:.2f}) bypasses SuperTrend"

        if has_volume and volume_conf >= self.ST_BYPASS_HIGH_CONFIDENCE:
            return True, f"High confidence volume breakout ({volume_conf:.2f}) bypasses SuperTrend"

        if has_ai and ai_conf >= self.ST_BYPASS_HIGH_CONFIDENCE:
            return True, f"High confidence AI ({ai_conf:.2f}) bypasses SuperTrend"

        # ──────────────────────────────────────────────────────────
        # BYPASS RULE 2: TRENDING Market + Strong Signals
        # ──────────────────────────────────────────────────────────
        if regime == "TRENDING" and regime_conf >= 0.7:
            # In strong trends, allow medium confidence signals
            if (has_rejection and rejection_conf >= 0.75) or \
                    (has_volume and volume_conf >= 0.75) or \
                    (has_ai and ai_conf >= 0.75 and signal_count >= 2):
                return True, f"Trending market + strong signal bypasses SuperTrend"
        # [OK] BYPASS RULE 2.5: EXTREME VOLUME BREAKOUT
        if signals.get("volume_breakout") == final_signal and confidences.get("volume_breakout", 0) >= 0.75:
            return True, f"Extreme volume breakout (conf: {confidences['volume_breakout']:.2f}) bypasses SuperTrend"

        # ──────────────────────────────────────────────────────────
        # BYPASS RULE 3: Multiple Strong Confirmations
        # ──────────────────────────────────────────────────────────
        strong_signal_count = 0
        if has_rejection and rejection_conf >= 0.75:
            strong_signal_count += 1
        if has_volume and volume_conf >= 0.75:
            strong_signal_count += 1
        if has_ai and ai_conf >= 0.75:
            strong_signal_count += 1

        if strong_signal_count >= 2:
            return True, f"Multiple strong signals ({strong_signal_count}) bypass SuperTrend"

        # ──────────────────────────────────────────────────────────
        # ENFORCE RULE: CHOPPY Market - Always check SuperTrend
        # ──────────────────────────────────────────────────────────
        if regime == "CHOPPY":
            self.log(
                f"[ST-ADAPTIVE] Choppy market - SuperTrend enforcement STRICT\n"
                f"  Signal count: {signal_count}, Max conf: {max(rejection_conf, volume_conf, ai_conf):.2f}",
                True
            )
            return False, "Choppy market requires SuperTrend validation"

        # ──────────────────────────────────────────────────────────
        # DEFAULT: Enforce SuperTrend
        # ──────────────────────────────────────────────────────────
        self.log(
            f"[ST-ADAPTIVE] No bypass conditions met\n"
            f"  Regime: {regime} ({regime_conf:.2f})\n"
            f"  Signals: rejection={rejection_conf:.2f}, volume={volume_conf:.2f}, ai={ai_conf:.2f}",
            True
        )
        return False, "Standard SuperTrend validation required"

    def _log_ai_trade(self, side, confidence, reason):
        """Log AI-specific trade details"""
        try:
            self._append_trade_csv({
                "trade_id": f"AI-{int(time.time())}",
                "symbol": self.symbol,
                "side": side,
                "event": f"AI_{side}",
                "entry_time": self._now_iso(),
                "entry_ltp": self.last_known_ltp,
                "reason": f"AI-CPR: {reason} (conf: {confidence:.3f})",
                "order_id": "AI-GENERATED",
                "bar_key": self.position.get("_last_bar_key", ""),
                "ai_confidence": confidence,
                "ai_action": side
            })
        except Exception as e:
            self.log(f"[AI-CPR] Failed to log AI trade: {e}", False)

    # ---------- UNIFIED ORDER PLACEMENT ----------
    def place_order(self, symbol, qty, side, tag):
        """
        Unified order placement method for both manual and AI orders
        """
        current_ltp = self.last_known_ltp
        current_inds = self.last_known_inds
        primary_tf = self.last_known_primary_tf

        if current_ltp is None or current_inds is None or primary_tf is None:
            self.log(f"[{tag}] Cannot place {side} order for {symbol} - missing market data", False)
            return False

        # Get ATR for SL calculation
        atr_val = self._get_atr_with_fallback(self._norm_tf(current_inds, primary_tf), current_ltp)
        if atr_val is None or atr_val <= 0:
            self.log(f"[{tag}] Cannot place {side} order for {symbol} - ATR unavailable", False)
            return False

        reason_str = f"{tag} {side} Signal"
        bar_key = self.position.get("_last_bar_key")
        self.log(f"[AI-Order] Computed ATR: {atr_val}", True)

        # Route to entry processing
        if side == "BUY":
            return self._process_entry("BUY", reason_str, current_ltp, atr_val,
                                     bar_key=bar_key, indsP=self._norm_tf(current_inds, primary_tf))
        elif side == "SELL":
            return self._process_entry("SELL", reason_str, current_ltp, atr_val,
                                     bar_key=bar_key, indsP=self._norm_tf(current_inds, primary_tf))
        else:
            self.log(f"[{tag}] Unknown order side: {side}", False)
            return False


    def _calculate_bid_ask_pressure(self, indicators):
        """
        Calculate bid/ask pressure from order book
        Returns: ("BUY", confidence) or ("SELL", confidence) or (None, 0)
        """
        try:
            # Try to get bid/ask from indicators
            bid = self._f(indicators.get("bid"))
            ask = self._f(indicators.get("ask"))
            bid_qty = self._f(indicators.get("bid_qty"))
            ask_qty = self._f(indicators.get("ask_qty"))

            if not all([bid, ask, bid_qty, ask_qty]):
                return None, 0.0

            # Calculate pressure
            total_qty = bid_qty + ask_qty
            if total_qty == 0:
                return None, 0.0

            pressure = (bid_qty - ask_qty) / total_qty

            # Strong buy pressure
            if pressure > 0.3:  # 30% more bids
                confidence = min(0.8, 0.5 + pressure)
                return "BUY", confidence

            # Strong sell pressure
            elif pressure < -0.3:  # 30% more asks
                confidence = min(0.8, 0.5 + abs(pressure))
                return "SELL", confidence

            return None, 0.0

        except Exception as e:
            self.log(f"[BID-ASK] Error: {e}", True)
            return None, 0.0

    def _detect_trend_strength(self, ohlc_df, indicators):
        """
        Detect if we're in a STRONG trend (don't exit on small pullbacks)
        Returns: ("STRONG_UPTREND", confidence) or ("STRONG_DOWNTREND", conf) or (None, 0)
        """
        if ohlc_df is None or len(ohlc_df) < 20:
            return None, 0.0

        try:
            # Get last 20 candles
            recent = ohlc_df.iloc[-20:]

            # Calculate higher highs / lower lows
            highs = recent['High'].values
            lows = recent['Low'].values
            closes = recent['Close'].values

            # Count higher highs (uptrend)
            higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])

            # Count lower lows (downtrend)
            lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i - 1])

            # Calculate momentum
            momentum_pct = ((closes[-1] - closes[0]) / closes[0]) * 100

            # Get ADX for trend strength
            adx = self._f(indicators.get("adx"), 0)

            # Get volume trend
            volume_ratio = self._f(indicators.get("volume_ratio"), 1.0)

            # [OK] STRONG UPTREND Detection
            if (higher_highs >= 15 and  # 15/19 candles made higher highs
                    momentum_pct > 1.0 and  # +1% move in 20 candles
                    adx > 25 and
                    volume_ratio > 1.0):

                confidence = min(0.9, 0.5 + (higher_highs / 20) + (adx / 100))

                self.log(
                    f"[TREND] 🔥 STRONG UPTREND detected\n"
                    f"  Higher highs: {higher_highs}/19\n"
                    f"  Momentum: +{momentum_pct:.2f}%\n"
                    f"  ADX: {adx:.1f}\n"
                    f"  Confidence: {confidence:.2f}",
                    False
                )
                return "STRONG_UPTREND", confidence

            # [OK] STRONG DOWNTREND Detection
            elif (lower_lows >= 15 and
                  momentum_pct < -1.0 and
                  adx > 25 and
                  volume_ratio > 1.0):

                confidence = min(0.9, 0.5 + (lower_lows / 20) + (adx / 100))

                self.log(
                    f"[TREND] 🔥 STRONG DOWNTREND detected\n"
                    f"  Lower lows: {lower_lows}/19\n"
                    f"  Momentum: {momentum_pct:.2f}%\n"
                    f"  ADX: {adx:.1f}\n"
                    f"  Confidence: {confidence:.2f}",
                    False
                )
                return "STRONG_DOWNTREND", confidence

            return None, 0.0

        except Exception as e:
            self.log(f"[TREND-STRENGTH] Error: {e}", True)
            return None, 0.0

    def _detect_early_breakout(self, ltp, indicators, ohlc_df):
        """
        Detect breakout EARLY (before full voting completes)
        Returns: ("BUY", confidence) or ("SELL", confidence) or (None, 0)
        """
        try:
            if ohlc_df is None or len(ohlc_df) < 5:
                return None, 0.0

            # Get indicators
            volume_ratio = self._f(indicators.get("volume_ratio"), 1.0)
            momentum_pct = self._f(indicators.get("momentum_pct"), 0.0)
            vwap = self._f(indicators.get("VWAP"))

            # Get recent candles
            recent = ohlc_df.iloc[-5:]

            # Calculate range compression (consolidation)
            high_range = recent['High'].max()
            low_range = recent['Low'].min()
            range_pct = (high_range - low_range) / low_range * 100

            # [OK] BREAKOUT CONDITIONS

            # 1. Price breaking out of tight range
            is_tight_range = range_pct < 1.0  # Less than 1% range in 5 candles

            # 2. Strong volume surge
            has_volume = volume_ratio >= 1.5

            # 3. Momentum building
            has_momentum = abs(momentum_pct) >= 0.4

            # 4. VWAP breakout
            vwap_breakout = False
            if vwap:
                vwap_dist = ((ltp - vwap) / vwap) * 100
                vwap_breakout = abs(vwap_dist) >= 0.2

            # [OK] BULLISH BREAKOUT
            if (is_tight_range and has_volume and momentum_pct > 0.4):
                confidence = 0.75 + (0.1 if vwap_breakout else 0)

                self.log(
                    f"[START] [EARLY-BREAKOUT] Bullish breakout detected\n"
                    f"  Range: {range_pct:.2f}% (tight < 1%)\n"
                    f"  Volume: {volume_ratio:.2f}x\n"
                    f"  Momentum: +{momentum_pct:.2f}%\n"
                    f"  VWAP: {vwap_dist:.2f}% from fair value\n"
                    f"  Confidence: {confidence:.2f}",
                    False
                )
                return "BUY", confidence

            # [OK] BEARISH BREAKOUT
            elif (is_tight_range and has_volume and momentum_pct < -0.4):
                confidence = 0.75 + (0.1 if vwap_breakout else 0)

                self.log(
                    f"[START] [EARLY-BREAKOUT] Bearish breakout detected\n"
                    f"  Range: {range_pct:.2f}% (tight < 1%)\n"
                    f"  Volume: {volume_ratio:.2f}x\n"
                    f"  Momentum: {momentum_pct:.2f}%\n"
                    f"  Confidence: {confidence:.2f}",
                    False
                )
                return "SELL", confidence

            return None, 0.0

        except Exception as e:
            self.log(f"[EARLY-BREAKOUT] Error: {e}", True)
            return None, 0.0


    def _should_enter_early(self, ltp, inds, ohlc_df):
        """
        Simple early entry detector with SuperTrend alignment check

        Returns: (signal, confidence, reason) or (None, 0.0, "")

        Logic:
        1. Check if price moved 0.2%+ from candle open
        2. Check if trend is aligned (EMA5 vs EMA9)
        3. Check if SuperTrend supports the direction ⭐ NEW
        4. Check if volume is above average (1.2x+)
        5. Check if momentum is strong (ADX > 20)

        That's it - no complex calculations!
        """

        # Validate inputs
        if ohlc_df is None or len(ohlc_df) < 2:
            return None, 0.0, "No OHLC data"

        try:
            latest = ohlc_df.iloc[-1]
            candle_open = float(latest['Open'])

            if candle_open <= 0:
                return None, 0.0, "Invalid open price"

            # ========================================
            # CHECK 1: PRICE MOVEMENT (0.2% threshold)
            # ========================================
            move_from_open = ((ltp - candle_open) / candle_open) * 100

            if abs(move_from_open) < 0.2:
                return None, 0.0, f"Small move ({move_from_open:.2f}%)"

            # ========================================
            # CHECK 2: TREND ALIGNMENT (EMA5 vs EMA9)
            # ========================================
            ema5 = self._f(inds.get("ema_5"))
            ema9 = self._f(inds.get("ema_9"))

            if not ema5 or not ema9:
                return None, 0.0, "Missing EMAs"

            # Determine direction
            if move_from_open > 0.2:  # Upward move
                signal = "BUY"
                # Check if EMA supports (allow 0.1% tolerance)
                if ema5 < ema9 * 0.999:
                    return None, 0.0, "EMA not aligned for BUY"
            else:  # Downward move
                signal = "SELL"
                if ema5 > ema9 * 1.001:
                    return None, 0.0, "EMA not aligned for SELL"

            # ========================================
            # CHECK 3: SUPERTREND CONFIRMATION ⭐ CRITICAL
            # ========================================
            # Get SuperTrend from current indicators
            st_main = self._f(inds.get("supertrend_main"))  # ST21 value
            st_main_trend = inds.get("st_main_trend", 0)  # 1=green, -1=red

            # If SuperTrend available, it MUST support the signal
            if st_main and st_main_trend != 0:
                if signal == "BUY" and st_main_trend < 0:
                    return None, 0.0, f"SuperTrend RED (bearish) - blocking BUY"
                elif signal == "SELL" and st_main_trend > 0:
                    return None, 0.0, f"SuperTrend GREEN (bullish) - blocking SELL"

                # [OK] SuperTrend confirms direction
                st_status = "GREEN ✓" if st_main_trend > 0 else "RED ✓"
            else:
                # No SuperTrend data - use price vs ST value
                if st_main:
                    if signal == "BUY" and ltp < st_main:
                        return None, 0.0, f"Price below SuperTrend - blocking BUY"
                    elif signal == "SELL" and ltp > st_main:
                        return None, 0.0, f"Price above SuperTrend - blocking SELL"
                    st_status = f"Price vs ST ✓"
                else:
                    # No SuperTrend at all - be more conservative
                    st_status = "No ST (high risk)"
                    self.log(f"[WARNING] [EARLY-ENTRY] No SuperTrend data - using only EMAs", True)

            # ========================================
            # CHECK 4: VOLUME CONFIRMATION (1.2x average)
            # ========================================
            volume_ratio = self._f(inds.get("volume_ratio"), 1.0)

            if volume_ratio < 1.2:
                return None, 0.0, f"Low volume ({volume_ratio:.2f}x)"

            # ========================================
            # CHECK 5: MOMENTUM STRENGTH (ADX > 20)
            # ========================================
            adx = self._f(inds.get("adx"), 0)

            if adx < 20:
                return None, 0.0, f"Weak momentum (ADX {adx:.1f})"

            # ========================================
            # ALL CHECKS PASSED - CALCULATE CONFIDENCE
            # ========================================
            confidence = 0.60  # Base confidence

            # Bonus for strong indicators
            if abs(move_from_open) > 0.3:
                confidence += 0.10
            if volume_ratio > 1.5:
                confidence += 0.10
            if adx > 25:
                confidence += 0.10

            # [OK] Bonus if SuperTrend strongly confirms
            if st_main_trend != 0:
                confidence += 0.05  # +5% for ST confirmation

            confidence = min(confidence, 0.90)  # Cap at 0.90

            reason = (
                f"Move: {move_from_open:.2f}%, "
                f"Vol: {volume_ratio:.2f}x, "
                f"ADX: {adx:.1f}, "
                f"ST: {st_status}"
            )

            self.log(
                f"[OK] [EARLY-ENTRY] {signal} signal detected!\n"
                f"  📈 Move from open: {abs(move_from_open):.2f}%\n"
                f"  [DATA] Volume: {volume_ratio:.2f}x average\n"
                f"  💪 ADX: {adx:.1f}\n"
                f"  📐 EMA: 5={ema5:.2f}, 9={ema9:.2f}\n"
                f"  [SIGNAL] SuperTrend: {st_status}\n"
                f"  ⭐ Confidence: {confidence:.2f}\n"
                f"  ⚡ Entering NOW (not waiting for candle close)",
                False
            )

            return signal, confidence, reason

        except Exception as e:
            self.log(f"[EARLY-ENTRY] Error: {e}", True)
            return None, 0.0, f"Error: {e}"

    # ---------- STRATEGY EXECUTION METHODS ----------
    def execute_unified_strategy(self, ltp, all_inds, primary_tf=None):

        if primary_tf is None:
            primary_tf = tf_selected

        # Get pivot data early (needed for both exit and entry logic)
        pivot_json_path = self.state_path.replace("om_state", "pivot")
        pivot_data = robust_load_json(pivot_json_path, self.log, default={})
        pivot_levels = pivot_data.get(self.symbol, {})

        # ========================================
        # INITIALIZE MANAGERS (once)
        # ========================================
        if not hasattr(self, 'sl_manager'):
            self.sl_manager = UnifiedStopLossManager(self)
            self.log("[INIT] Stop Loss Manager initialized", False)

        if not hasattr(self, 'tp_manager'):
            self.tp_manager = UnifiedTrailingProfitManager(self)
            self.log("[INIT] Trailing Profit Manager initialized", False)

        # ========================================
        # PRIORITY #1: STOP LOSS CHECK
        # ========================================
        should_exit_sl, sl_reason, sl_loss = self.sl_manager.check_stop_loss(ltp)

        if should_exit_sl:
            entry_price = self._f(self.position.get("entry_price"))
            max_profit_achieved = self.position.get("_max_profit", 0)

            self.log(
                f"\n{'=' * 70}\n"
                f"🛑 STOP LOSS TRIGGERED 🛑\n"
                f"{'=' * 70}\n"
                f"Reason: {sl_reason}\n"
                f"Position: {self.position.get('type')}\n"
                f"Entry: ₹{entry_price:.2f}\n"
                f"Current: ₹{ltp:.2f}\n"
                f"Loss: ₹{sl_loss:.2f}\n"
                f"Max Profit: ₹{max_profit_achieved:.0f}\n"
                f"Total DD: ₹{sl_loss + max_profit_achieved:.0f}\n"
                f"{'=' * 70}\n"
                f"[OK] EXITING NOW\n"
                f"{'=' * 70}\n",
                False
            )

            self._process_exit(sl_reason, ltp)
            self._save_state()
            return  # STOP HERE - DO NOT CONTINUE

        # PRIORITY #2: TRAILING PROFIT UPDATE
        current_pos = self.position.get("type", "FLAT")

        if current_pos != "FLAT":
            def _tf(tf):
                return self._norm_tf(all_inds, str(tf))

            inds = _tf(primary_tf)
            atr = self._get_atr_with_fallback(inds, ltp)

            # Get pivot data
            cpr_analysis = inds.get("cpr_analysis", {})
            pivot_data = cpr_analysis.get("cpr_levels", {})

            if atr:
                self.tp_manager.position = self.position
                should_exit_tp, tp_reason = self.tp_manager.update_trailing_stops(
                    ltp,
                    atr,
                    pivot_data if pivot_data and "TC" in pivot_data else None
                )

                if should_exit_tp:
                    self.log(f"[SIGNAL] [TRAILING-EXIT] {tp_reason}", False)
                    self._process_exit(tp_reason, ltp)
                    self._save_state()
                    return

            # [OK] Log position status every cycle
            entry = self._f(self.position.get("entry_price"))
            if entry:
                if current_pos == "BUY":
                    pnl_points = ltp - entry
                else:
                    pnl_points = entry - ltp

                pnl_rupees = pnl_points * 250
                current_sl = self._f(self.position.get("stop_loss"))

                self.log(
                    f"[POSITION] {current_pos} @ ₹{entry:.2f} | "
                    f"LTP: ₹{ltp:.2f} | "
                    f"P&L: ₹{pnl_rupees:.0f} ({pnl_points:+.2f} pts) | "
                    f"SL: ₹{current_sl:.2f}",
                    False
                )

        # ========================================
        # PRIORITY #3: ENTRY LOGIC
        # Only runs if position is FLAT
        # ========================================
        if current_pos != "FLAT":
            self.log(f"[UNIFIED] Already in {current_pos} - skipping entry logic", True)
            return

        # VALIDATE INPUTS
        if ltp is None or ltp <= 0:
            self.log(f"[UNIFIED] Invalid LTP: {ltp} - aborting cycle", False)
            return

        if not all_inds or not isinstance(all_inds, dict):
            self.log("[UNIFIED] Invalid indicators structure - aborting cycle", False)
            return

        self.last_known_ltp = ltp
        self.last_known_inds = all_inds
        self.last_known_primary_tf = primary_tf

        current_pos = self.position.get("type", "FLAT")
        bar_key = None

        def _tf(tf):
            return self._norm_tf(all_inds, str(tf))

        inds = _tf(primary_tf)

        if not isinstance(inds, dict) or not inds.get("timestamp"):
            self.log(f"[UNIFIED] No valid indicators for TF={primary_tf}.", True)
            # [OK] FIXED: Trailing already handled in PRIORITY #2 - just return
            if current_pos != "FLAT":
                self.log("[UNIFIED] Position exists but indicators stale - trailing already updated", True)
            return

        bar_key = f"{primary_tf}:{inds['timestamp']}"

        # ==========================================
        # CRITICAL GUARDS
        # ==========================================
        if current_pos == "FLAT":
            last_exit_bar = self.position.get("_last_action_bar")
            if last_exit_bar == bar_key:
                self.log(f"[RE-ENTRY BLOCK] Just exited on bar {bar_key} - must wait for next bar", False)
                return

        # [OK] FIXED: Same bar - trailing already updated in PRIORITY #2
        if bar_key == self.position.get("_last_bar_key"):
            if current_pos != "FLAT":
                self.log("[UNIFIED] Same bar - trailing already updated in PRIORITY #2", True)
            return

        if bar_key != self.position.get("_last_bar_key"):
            self.position["_exits_this_bar"] = 0
        self.position["_last_bar_key"] = bar_key

        if self.DEDUPE_ONE_ENTRY_PER_BAR and self.position.get("_last_entry_attempt_bar") == bar_key:
            self.log("[UNIFIED] Entry already attempted this bar - skipping", True)
            return

        # [OK] FIXED: Cooldown check - trailing already updated
        if self._cooldown_active():
            self.log("[UNIFIED] Cooldown active - skipping new entries", True)
            if current_pos != "FLAT":
                self.log("[UNIFIED] Position exists - trailing already updated in PRIORITY #2", True)
            return

        if self.position.get("_skip_entry_until_bar") == bar_key:
            self.log("[UNIFIED] Re-entry cooldown active - skipping", True)
            return

        # ==========================================
        # DETECT TREND
        # ==========================================
        detected_trend = None
        try:
            df_5 = None
            if hasattr(self, 'bot'):
                try:
                    df_5 = self.bot.fetch_ohlc(self.symbol, "5", 30)
                except Exception as e:
                    self.log(f"[TREND-DETECT] Failed to fetch OHLC: {e}", True)

            if df_5 is not None and isinstance(df_5, pd.DataFrame) and not df_5.empty:
                df_5 = df_5.copy()
                df_5.columns = [c.lower() for c in df_5.columns]
                if "close" in df_5.columns and len(df_5) > 10:
                    i = len(df_5) - 1
                    try:
                        detected_trend = self.detect_trend(df=df_5, i=i)
                        self.log(f"[TREND-DETECT] Result: {detected_trend}", True)
                    except Exception as e:
                        self.log(f"[TREND-DETECT] Detection error: {e}", True)
                        detected_trend = None
        except Exception as e:
            self.log(f"[TREND-DETECT] Critical error: {e}", True)
            detected_trend = None

        if isinstance(inds, dict):
            inds["trend"] = detected_trend

        # ==========================================
        # AI CPR PREDICTION
        # ==========================================
        ai_label = None
        ai_confidence_raw = 0.0
        ai_distribution = None
        ai_override_allowed = False
        ai_override_signal = None
        AI_OVERRIDE_MIN_CONFIDENCE = 0.75

        try:
            self.log("[AI-CPR] Starting prediction...", True)

            cpr_analysis = inds.get("cpr_analysis", {})
            pivot_data = cpr_analysis.get("cpr_levels", {}) if cpr_analysis else {}

            if not pivot_data or not isinstance(pivot_data, dict) or "TC" not in pivot_data:
                try:
                    pivot_json_path = self.state_path.replace("om_state", "pivot")
                    loaded_pivot = robust_load_json(pivot_json_path, self.log, default={})
                    pivot_data = loaded_pivot.get(self.symbol, {}) if isinstance(loaded_pivot, dict) else {}
                    self.log(f"[AI-CPR] Loaded pivot from file: TC={pivot_data.get('TC')}", True)
                except Exception as e:
                    self.log(f"[AI-CPR] Failed to load pivot data: {e}", True)
                    pivot_data = {}

            if pivot_data and "TC" in pivot_data and "BC" in pivot_data:
                self.log(
                    f"[AI-CPR] Running prediction with TC={pivot_data.get('TC')}, BC={pivot_data.get('BC')}",
                    True
                )

                #ai_label, ai_conf, ai_dist, _ = self.ai_predictor.predict(
                #    indicators=inds,
                #    pivot_data=pivot_data,
                #    feature_builder=_build_ai_cpr_features
                #)
                try:
                    ohlc_for_ai = self.bot.fetch_ohlc(self.symbol, "5", 2)
                except Exception as e:
                    self.log(f"[AI-OHLC] Failed to fetch: {e}", True)
                    ohlc_for_ai = None

                # Then pass it to predictor
                ai_label, ai_conf, ai_dist, feature_array = self.ai_predictor.predict(
                    indicators=inds,
                    pivot_data=pivot_data,
                    feature_builder=_build_ai_cpr_features,
                    ohlc_df=ohlc_for_ai  # 🔥 ADD THIS
                )
                ai_confidence_raw = float(ai_conf) if ai_conf is not None else 0.0
                ai_distribution = ai_dist
                if ai_label and ai_confidence_raw >= 0.25:
                    # Extract patterns for context
                    patterns_str = "No patterns"
                    if feature_array is not None:
                        features_flat = feature_array.flatten()
                        if len(features_flat) >= 12:
                            candle_features = features_flat[-12:]
                            patterns = []
                            if candle_features[4] != 0:
                                patterns.append(f"Engulfing:{candle_features[4]:+.0f}")
                            if candle_features[5] != 0:
                                patterns.append(f"Reversal:{candle_features[5]:+.0f}")
                            if candle_features[6] != 0:
                                patterns.append(f"Marubozu:{candle_features[6]:+.0f}")
                            if patterns:
                                patterns_str = ", ".join(patterns)

                    self.log(
                        f"\n🤖 [AI-TRADING-DECISION]\n"
                        f"  Signal: {ai_label}\n"
                        f"  Confidence: {ai_confidence_raw:.1%}\n"
                        f"  Candle Patterns: {patterns_str}\n"
                        f"  Current Position: {current_pos}\n"
                        f"  Action: {'ENTRY CANDIDATE' if current_pos == 'FLAT' else 'HOLD/EXIT CHECK'}",
                        False
                    )

                self.log(
                    f"\n{'=' * 60}\n"
                    f"🤖 AI CPR PREDICTION\n"
                    f"{'=' * 60}",
                    False
                )
                self.log(
                    f"Label: {ai_label}\n"
                    f"Confidence: {ai_confidence_raw:.4f} ({ai_confidence_raw * 100:.2f}%)\n"
                    f"Threshold (Min): {self.AI_MIN_CONF:.4f}\n"
                    f"Threshold (Override): 0.75\n"
                    f"Status: {'[OK] ACCEPTED' if ai_confidence_raw >= self.AI_MIN_CONF else '[ERROR] REJECTED'}",
                    False
                )
                if ai_distribution:
                    self.log(f"Distribution: {ai_distribution}", True)
                self.log(f"{'=' * 60}", False)

                if ai_label and ai_confidence_raw >= AI_OVERRIDE_MIN_CONFIDENCE:
                    ai_label_upper = str(ai_label).upper()

                    if any(keyword in ai_label_upper for keyword in ["BUY", "BULLISH", "LONG", "UP"]):
                        ai_override_signal = "BUY"
                        ai_override_allowed = True
                        self.log(
                            f"\n[START] [AI-OVERRIDE] HIGH CONFIDENCE BUY DETECTED!\n"
                            f"   Confidence: {ai_confidence_raw:.2f} (≥ {AI_OVERRIDE_MIN_CONFIDENCE})\n"
                            f"   Can bypass consolidation if needed\n",
                            False
                        )

                    elif any(keyword in ai_label_upper for keyword in ["SELL", "BEARISH", "SHORT", "DOWN"]):
                        ai_override_signal = "SELL"
                        ai_override_allowed = True
                        self.log(
                            f"\n[START] [AI-OVERRIDE] HIGH CONFIDENCE SELL DETECTED!\n"
                            f"   Confidence: {ai_confidence_raw:.2f} (≥ {AI_OVERRIDE_MIN_CONFIDENCE})\n"
                            f"   Can bypass consolidation if needed\n",
                            False
                        )

                elif ai_label and ai_confidence_raw >= self.AI_MIN_CONF:
                    self.log(
                        f"[AI-CPR] Medium confidence ({ai_confidence_raw:.2f}) - "
                        f"Will count as vote but won't override consolidation",
                        True
                    )
                else:
                    self.log(
                        f"[AI-CPR] Low confidence ({ai_confidence_raw:.2f}) - Signal rejected",
                        True
                    )
            else:
                self.log(
                    f"[AI-CPR] [WARNING] Pivot data incomplete - skipping prediction\n"
                    f"   TC: {pivot_data.get('TC')}, BC: {pivot_data.get('BC')}",
                    True
                )

        except Exception as e:
            self.log(f"[AI-CPR] [ERROR] Prediction error: {e}", False)
            import traceback
            self.log(f"[AI-CPR] Traceback:\n{traceback.format_exc()}", True)

        # ==========================================
        # CONSOLIDATION BLOCKER
        # ==========================================

        ohlc_for_breakout = None
        try:
            ohlc_for_breakout = self.bot.fetch_ohlc(self.symbol, "5", 1)
        except Exception as e:
            self.log(f"[BREAKOUT] Failed to fetch OHLC: {e}", True)

        early_breakout_signal, early_breakout_conf = self._detect_early_breakout(
            ltp, inds, ohlc_for_breakout
        )

        # ==========================================
        # 🔥 FIX 1: SMART CONSOLIDATION CHECK (with breakout detection)
        # ==========================================
        consolidation_blocks_entry = False

        if detected_trend and "consol" in str(detected_trend).lower():

            # [OK] Check for BREAKOUT signals
            volume_ratio = self._f(inds.get("volume_ratio"), 1.0)
            momentum_pct = self._f(inds.get("momentum_pct"), 0.0)

            # Get VWAP distance
            vwap = self._f(inds.get("VWAP"))
            vwap_distance = abs((ltp - vwap) / vwap * 100) if vwap else 0

            # [OK] BREAKOUT DETECTED = Allow entry
            is_breakout = (
                    early_breakout_signal is not None or  # Early breakout detector
                    (volume_ratio >= 0.6 and abs(momentum_pct) >= 0.2) or  # Volume + Momentum
                    (vwap_distance >= 0.2) or  # Strong VWAP breakout
                    (ai_override_allowed and ai_override_signal)  # AI confirms direction
            )


            # [OK] OPTION 2: Just disable consolidation blocking entirely
            consolidation_blocks_entry = False
            if is_breakout:
                self.log(
                    f"[OK] [BREAKOUT] Consolidation breakout detected\n"
                    f"  Early signal: {early_breakout_signal} ({early_breakout_conf:.2f})\n"
                    f"  Volume: {volume_ratio:.2f}x | Momentum: {momentum_pct:.2f}%\n"
                    f"  VWAP distance: {vwap_distance:.2f}%\n"
                    f"  AI: {ai_override_signal} ({ai_confidence_raw:.2f})\n"
                    f"  Decision: ALLOW ENTRY during consolidation",
                    False
                )
                consolidation_blocks_entry = False
            else:
                self.log(
                    f"[CONSOLIDATION] No breakout - waiting\n"
                    f"  Volume: {volume_ratio:.2f}x (need 1.3x+)\n"
                    f"  Momentum: {momentum_pct:.2f}% (need 0.3%+)\n"
                    f"  VWAP distance: {vwap_distance:.2f}% (need 0.2%+)\n"
                    f"  Early breakout: {early_breakout_signal or 'None'}",
                    True
                )
                consolidation_blocks_entry = True

        # If consolidation blocks entry, manage existing position and return
        if consolidation_blocks_entry:
            if current_pos != "FLAT":
                self.log("[UNIFIED] Same bar - trailing already updated in PRIORITY #2", True)
            return

        # ==========================================
        # EARLY SIGNALS CHECK
        # ==========================================
        early_entry_signal = None
        early_entry_confidence = 0.0

        ohlc_for_early = None
        try:
            ohlc_for_early = self.bot.fetch_ohlc(self.symbol, "5", 2)
        except Exception as e:
            self.log(f"[EARLY] Failed to fetch OHLC: {e}", True)

        if ohlc_for_early is not None and not ohlc_for_early.empty:
            volume_signal, volume_conf = self.detect_volume_breakout(ohlc_for_early, ltp)

            if volume_signal and volume_conf >= 0.75:
                early_entry_signal = "BUY" if volume_signal == "VOLUME_BREAKOUT_BUY" else "SELL"
                early_entry_confidence = volume_conf

                self.log(
                    f"[START] [EARLY-ENTRY] Volume breakout {early_entry_signal}!\n"
                    f"  Confidence: {volume_conf:.2f}\n"
                    f"  Bypassing normal filters for speed",
                    False
                )

            if not early_entry_signal:
                rejection_signal, rejection_conf = self.detect_rejection_candle(
                    ohlc_for_early, ltp, pivot_levels
                )

                if rejection_signal and rejection_conf >= 0.75:
                    early_entry_signal = "BUY" if rejection_signal == "BUY_REJECTION" else "SELL"
                    early_entry_confidence = rejection_conf

                    self.log(
                        f"[SIGNAL] [EARLY-ENTRY] Rejection {early_entry_signal}!\n"
                        f"  Confidence: {rejection_conf:.2f}",
                        False
                    )

        if early_entry_signal and current_pos == "FLAT":
            atr_here = self._get_atr_with_fallback(inds, ltp)
            if atr_here:
                reason = f"EARLY {early_entry_signal} (conf: {early_entry_confidence:.2f}) - Volume Breakout"

                if self._process_entry(
                        early_entry_signal,
                        reason,
                        ltp,
                        atr_here,
                        bar_key=bar_key,
                        indsP=inds
                ):
                    self.position["_last_action_bar"] = bar_key
                    # Zero cooldown if strong momentum, else 1 bar
                    cooldown_bars = 0 if (abs(self._f(inds.get("momentum_pct"), 0)) > 0.5 and self._f(inds.get("adx"),
                                                                                                      0) > 25) else 1
                    self._set_cooldown(cooldown_bars)
                    #self._set_cooldown(1)
                    self._save_state()
                    return

        # ==========================================
        # SIGNAL COLLECTION
        # ==========================================

        # Fetch OHLC data for AI CPR and pattern detection
        ohlc_for_patterns = None
        try:
            ohlc_for_patterns = self.bot.fetch_ohlc(self.symbol, "15", 2)
        except Exception as e:
            self.log(f"[SIGNAL-COLLECTION] Failed to fetch OHLC: {e}", True)

        signals = {
            "vwap": None,
            "bid_ask_pressure": None,
            "volume_momentum": None,
            "ai_cpr": None,
            "cpr_strategy": None,
            "rejection_candle": None,
            "volume_breakout": None,
            "momentum_shift": None,
            "early_breakout": None,
            "price_action": None,
        }

        confidences = {
            "vwap": 0.0,
            "bid_ask_pressure": 0.0,
            "volume_momentum": 0.0,
            "ai_cpr": 0.0,
            "cpr_strategy": 0.0,
            "rejection_candle": 0.0,
            "volume_breakout": 0.0,
            "momentum_shift": 0.0,
            "early_breakout": 0.0,
            "price_action": 0.0,
        }

        # Extract indicators
        e5 = self._f(inds.get("ema_5"))
        e9 = self._f(inds.get("ema_9"))
        e21 = self._f(inds.get("ema_21"))
        macd_color = inds.get("macd_color")
        adx_val = self._f(inds.get("adx"))
        bb_bandwidth = self._f(inds.get("bb_bandwidth"))

        volume_ratio = self._f(inds.get("volume_ratio"), 1.0)
        volume_surge = inds.get("volume_surge", False)
        volume_extreme = inds.get("volume_extreme", False)
        momentum_pct = self._f(inds.get("momentum_pct"), 0.0)
        roc_10 = self._f(inds.get("roc_10"), 0.0)
        strong_bearish = inds.get("strong_bearish", False)
        strong_bullish = inds.get("strong_bullish", False)

        # Volatility filter
        is_choppy = bb_bandwidth is not None and bb_bandwidth < 0.005
        if is_choppy:
            self.log(
                f"[UNIFIED] Choppy market detected (BBW: {bb_bandwidth:.4f}) - blocking entries",
                False
            )
            if current_pos != "FLAT":
                self.log("[UNIFIED] Same bar - trailing already updated in PRIORITY #2", True)
            return

        # 1️⃣ TREND-BASED SIGNAL (EMA Cross)
        #if e9 is not None and e21 is not None:
        #    if e9 > e21:
        #        signals["ema_cross"] = "BUY"
        #        confidences["ema_cross"] = 0.6
        #    elif e9 < e21:
        #        signals["ema_cross"] = "SELL"
        #        confidences["ema_cross"] = 0.6

        vwap = self._f(inds.get("VWAP"))
        if vwap and ltp:
            vwap_distance_pct = ((ltp - vwap) / vwap) * 100

            if vwap_distance_pct > 0.15:  # Above VWAP by 0.15%
                signals["vwap"] = "BUY"
                confidences["vwap"] = 0.75  # High confidence
                self.log(f"[VWAP] BUY signal - Price {vwap_distance_pct:.2f}% above VWAP", True)
            elif vwap_distance_pct < -0.15:  # Below VWAP by 0.15%
                signals["vwap"] = "SELL"
                confidences["vwap"] = 0.75
                self.log(f"[VWAP] SELL signal - Price {vwap_distance_pct:.2f}% below VWAP", True)

        bid_ask_signal, bid_ask_conf = self._calculate_bid_ask_pressure(inds)
        if bid_ask_signal:
            signals["bid_ask_pressure"] = bid_ask_signal
            confidences["bid_ask_pressure"] = bid_ask_conf

        if early_breakout_signal and early_breakout_conf >= 0.75:
            signals["early_breakout"] = early_breakout_signal
            confidences["early_breakout"] = early_breakout_conf
            self.log(
                f"[SIGNAL] [PRIORITY] Early breakout: {early_breakout_signal} ({early_breakout_conf:.2f})",
                False
            )

        # [OK] NEW: Direct momentum signal
        momentum_pct = self._f(inds.get("momentum_pct"), 0.0)
        adx_val = self._f(inds.get("adx"), 0)

        if abs(momentum_pct) > 0.4 and adx_val > 20:
            if momentum_pct > 0:
                signals["momentum_direct"] = "BUY"
                confidences["momentum_direct"] = 0.70
                self.log(f"📈 [MOMENTUM] Direct BUY signal - {momentum_pct:.2f}%", False)
            else:
                signals["momentum_direct"] = "SELL"
                confidences["momentum_direct"] = 0.70
                self.log(f"📉 [MOMENTUM] Direct SELL signal - {momentum_pct:.2f}%", False)

        # 5️⃣ VOLUME + MOMENTUM SIGNAL
        if volume_surge and momentum_pct > 0.5:
            signals["volume_momentum"] = "BUY"
            if volume_extreme and momentum_pct > 0.8:
                confidences["volume_momentum"] = 0.85
                self.log(
                    f"[VOLUME-MOMENTUM] [START] EXTREME BUY - Volume: {volume_ratio:.2f}x, Momentum: {momentum_pct:.2f}%",
                    False
                )
            else:
                confidences["volume_momentum"] = 0.75
                self.log(
                    f"[VOLUME-MOMENTUM] [OK] Strong BUY - Volume: {volume_ratio:.2f}x, Momentum: {momentum_pct:.2f}%",
                    True
                )

        elif volume_surge and momentum_pct < -0.5:
            signals["volume_momentum"] = "SELL"
            if volume_extreme and momentum_pct < -0.8:
                confidences["volume_momentum"] = 0.85
                self.log(
                    f"[VOLUME-MOMENTUM] [START] EXTREME SELL - Volume: {volume_ratio:.2f}x, Momentum: {momentum_pct:.2f}%",
                    False
                )
            else:
                confidences["volume_momentum"] = 0.75
                self.log(
                    f"[VOLUME-MOMENTUM] [OK] Strong SELL - Volume: {volume_ratio:.2f}x, Momentum: {momentum_pct:.2f}%",
                    True
                )

        elif strong_bullish:
            signals["volume_momentum"] = "BUY"
            confidences["volume_momentum"] = 0.70
            self.log(f"[VOLUME-MOMENTUM] [OK] BUY signal (combined)", True)

        elif strong_bearish:
            signals["volume_momentum"] = "SELL"
            confidences["volume_momentum"] = 0.70
            self.log(f"[VOLUME-MOMENTUM] [OK] SELL signal (combined)", True)

        else:
            self.log(
                f"[VOLUME-MOMENTUM] No signal - Volume: {volume_ratio:.2f}x, Momentum: {momentum_pct:.2f}%",
                True
            )

        # 6️⃣ AI CPR SIGNAL
        try:
            pivot_data = inds.get("cpr_analysis", {})

            if not pivot_data or not isinstance(pivot_data, dict) or "TC" not in pivot_data:
                try:
                    pivot_json_path = self.state_path.replace("om_state", "pivot")
                    loaded_pivot = robust_load_json(
                        pivot_json_path,
                        self.log,
                        default={}
                    )
                    pivot_data = loaded_pivot.get(self.symbol, {}) if isinstance(loaded_pivot, dict) else {}
                except Exception as e:
                    self.log(f"[AI-CPR] Failed to load pivot data: {e}", True)
                    pivot_data = {}

            if pivot_data and "TC" in pivot_data and "BC" in pivot_data:
                #ai_label, ai_conf, ai_dist, _ = self.ai_predictor.predict(
                #    indicators=inds,
                #    pivot_data=pivot_data,
                #    feature_builder=_build_ai_cpr_features
                #)
                ai_label, ai_conf, ai_dist, feature_array = self.ai_predictor.predict(
                    indicators=inds,
                    pivot_data=pivot_data,
                    feature_builder=_build_ai_cpr_features,
                    ohlc_df=ohlc_for_patterns  # 🔥 ADD THIS
                )

                ai_confidence_raw = float(ai_conf) if ai_conf is not None else 0.0
                ai_distribution = ai_dist

                self.log(
                    f"[AI-CPR] ========== AI PREDICTION DETAILS ==========",
                    False
                )
                self.log(
                    f"[AI-CPR] Label: {ai_label} | Confidence: {ai_confidence_raw:.4f} ({ai_confidence_raw * 100:.2f}%)",
                    False
                )
                self.log(
                    f"[AI-CPR] Threshold: {self.AI_MIN_CONF:.4f} | Status: {'[OK] ACCEPTED' if ai_confidence_raw >= self.AI_MIN_CONF else '[ERROR] REJECTED'}",
                    False
                )
                if ai_distribution:
                    self.log(
                        f"[AI-CPR] Distribution: {ai_distribution}",
                        False
                    )
                self.log(
                    f"[AI-CPR] ==========================================",
                    False
                )

                if ai_label and ai_confidence_raw >= self.AI_MIN_CONF:
                    ai_label_upper = str(ai_label).upper()

                    # Check if counter-trend trade
                    detected_trend = inds.get("trend")
                    is_counter_trend = False

                    if detected_trend:
                        trend_str = str(detected_trend).lower()

                        if any(keyword in ai_label_upper for keyword in ["BUY", "BULLISH", "LONG", "UP"]):
                            if "down" in trend_str or "bear" in trend_str:
                                is_counter_trend = True
                                self.log(
                                    f"[WARNING] [AI-COUNTER-TREND] AI suggests BUY but trend is {detected_trend}",
                                    False
                                )

                        elif any(keyword in ai_label_upper for keyword in ["SELL", "BEARISH", "SHORT", "DOWN"]):
                            if "up" in trend_str or "bull" in trend_str:
                                is_counter_trend = True
                                self.log(
                                    f"[WARNING] [AI-COUNTER-TREND] AI suggests SELL but trend is {detected_trend}",
                                    False
                                )

                    min_confidence_required = 0.85 if is_counter_trend else self.AI_MIN_CONF

                    if ai_confidence_raw >= min_confidence_required:
                        if any(keyword in ai_label_upper for keyword in ["BUY", "BULLISH", "LONG", "UP"]):
                            signals["ai_cpr"] = "BUY"
                            confidences["ai_cpr"] = ai_confidence_raw

                            if is_counter_trend:
                                self.log(
                                    f"[AI-CPR] [OK] Counter-trend BUY accepted (high confidence: {ai_confidence_raw:.3f})",
                                    False
                                )
                            else:
                                self.log(f"[AI-CPR] [OK] BUY signal accepted (conf: {ai_confidence_raw:.3f})", True)

                        elif any(keyword in ai_label_upper for keyword in ["SELL", "BEARISH", "SHORT", "DOWN"]):
                            signals["ai_cpr"] = "SELL"
                            confidences["ai_cpr"] = ai_confidence_raw
                            if is_counter_trend:
                                self.log(
                                    f"[AI-CPR] [OK] Counter-trend SELL accepted (high confidence: {ai_confidence_raw:.3f})",
                                    False
                                )
                            else:
                                self.log(f"[AI-CPR] [OK] SELL signal accepted (conf: {ai_confidence_raw:.3f})", True)
                        else:
                            self.log(f"[AI-CPR] [WARNING] Neutral/Hold signal: {ai_label}", True)

                    else:
                        conf_msg = f"{ai_confidence_raw:.3f}" if ai_confidence_raw else "N/A"
                        reason = "counter-trend, needs 0.85+" if is_counter_trend else f"< threshold {self.AI_MIN_CONF}"
                        self.log(
                            f"[AI-CPR] [ERROR] Signal rejected: confidence {conf_msg} ({reason})",
                            True
                        )
                else:
                    if ai_label:
                        self.log(
                            f"[AI-CPR] [ERROR] Signal rejected: confidence {ai_confidence_raw:.3f} < {self.AI_MIN_CONF}",
                            True
                        )

        except Exception as e:
            self.log(f"[AI-CPR] [WARNING] Prediction error: {e}", False)
            import traceback
            self.log(f"[AI-CPR] Traceback: {traceback.format_exc()}", True)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # BOOST CONFIDENCE FOR STRONG BUY/SELL PREDICTIONS
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if ai_label and ai_confidence_raw > 0:
            try:
                # Convert label to integer if possible
                label_int = None
                if isinstance(ai_label, (int, float)):
                    label_int = int(ai_label)
                elif str(ai_label).lstrip('-').isdigit():
                    label_int = int(ai_label)

                    # Boost STRONG_BUY (2) or STRONG_SELL (-2) signals
                if label_int in [2, -2] and ai_confidence_raw >= 0.20:
                    original_conf = ai_confidence_raw
                    ai_confidence_raw = min(ai_confidence_raw + 0.15, 0.95)

                    signal_type = 'STRONG_BUY' if label_int == 2 else 'STRONG_SELL'
                    self.log(
                        f"⚡ [AI-BOOST] {signal_type} signal detected\n"
                        f"  Label: {ai_label}\n"
                        f"  Confidence: {original_conf:.2f} → {ai_confidence_raw:.2f} (+15%)",
                        False
                    )
            except Exception as e:
                self.log(f"[AI-BOOST] Error: {e}", True)

        # 7️⃣ CPR STRATEGY SIGNAL
        cpr_analysis = inds.get("cpr_analysis", {})

        if not cpr_analysis or not isinstance(cpr_analysis, dict):
            self.log("[CPR-STRATEGY] CPR analysis missing - signal skipped", True)
        else:
            cpr_levels = cpr_analysis.get("cpr_levels", {})
            cpr_valid = bool(
                cpr_levels and
                isinstance(cpr_levels, dict) and
                cpr_levels.get("TC") and
                cpr_levels.get("BC") and
                cpr_levels.get("R1") and
                cpr_levels.get("S1")
            )

            if not cpr_valid:
                self.log("[CPR-STRATEGY] CPR levels incomplete - signal skipped", True)
            else:
                cpr_trade_signal = cpr_analysis.get("trade_strategy", "None")

                if cpr_trade_signal == "Buy":
                    signals["cpr_strategy"] = "BUY"
                    confidences["cpr_strategy"] = 0.7
                elif cpr_trade_signal == "Sell":
                    signals["cpr_strategy"] = "SELL"
                    confidences["cpr_strategy"] = 0.7

        # 8️⃣ EARLY SIGNALS (Rejection, Volume Breakout, Momentum Shift)
        ohlc_for_patterns = None
        try:
            ohlc_for_patterns = self.bot.fetch_ohlc(self.symbol, "15", 2)
        except Exception as e:
            self.log(f"[EARLY-SIGNALS] Failed to fetch OHLC: {e}", True)

        if ohlc_for_patterns is not None and not ohlc_for_patterns.empty:
            rejection_signal, rejection_conf = self.detect_rejection_candle(
                ohlc_for_patterns,
                ltp,
                pivot_data
            )
            if rejection_signal:
                if rejection_signal == "SELL_REJECTION":
                    signals["rejection_candle"] = "SELL"
                    confidences["rejection_candle"] = rejection_conf
                    self.log(
                        f"🔴 [EARLY] SELL rejection detected (conf: {rejection_conf:.2f})",
                        False
                    )
                elif rejection_signal == "BUY_REJECTION":
                    signals["rejection_candle"] = "BUY"
                    confidences["rejection_candle"] = rejection_conf
                    self.log(
                        f"🟢 [EARLY] BUY rejection detected (conf: {rejection_conf:.2f})",
                        False
                    )

            volume_signal, volume_conf = self.detect_volume_breakout(ohlc_for_patterns, ltp)

            if volume_signal:
                if volume_signal == "VOLUME_BREAKOUT_BUY":
                    signals["volume_breakout"] = "BUY"
                    confidences["volume_breakout"] = volume_conf
                    self.log(
                        f"[START] [EARLY] BUY volume breakout (conf: {volume_conf:.2f})",
                        False
                    )
                elif volume_signal == "VOLUME_BREAKOUT_SELL":
                    signals["volume_breakout"] = "SELL"
                    confidences["volume_breakout"] = volume_conf
                    self.log(
                        f"[START] [EARLY] SELL volume breakout (conf: {volume_conf:.2f})",
                        False
                    )

            momentum_signal, momentum_conf = self.detect_momentum_shift(ohlc_for_patterns)

            if momentum_signal:
                if momentum_signal == "MOMENTUM_SHIFT_BUY":
                    signals["momentum_shift"] = "BUY"
                    confidences["momentum_shift"] = momentum_conf
                    self.log(
                        f"📈 [EARLY] BUY momentum shift (conf: {momentum_conf:.2f})",
                        False
                    )
                elif momentum_signal == "MOMENTUM_SHIFT_SELL":
                    signals["momentum_shift"] = "SELL"
                    confidences["momentum_shift"] = momentum_conf
                    self.log(
                        f"📉 [EARLY] SELL momentum shift (conf: {momentum_conf:.2f})",
                        False
                    )
            # 9️⃣ PRICE ACTION SIGNAL (NEW!)
            # ==========================================
            self.log("[PRICE-ACTION] Starting analysis...", True)

            try:
                # Get pivot data
                pivot_json_path = self.state_path.replace("om_state", "pivot")
                pivot_data = robust_load_json(pivot_json_path, self.log, default={})
                pivot_levels = pivot_data.get(self.symbol, {})
                self.price_action_analyzer.debug_levels(ltp, pivot_levels)

                # Validate pivot data
                required_keys = ["TC", "BC", "R1", "R2", "R3", "S1", "S2", "S3"]
                missing_keys = [k for k in required_keys if k not in pivot_levels or pivot_levels[k] is None]

                if missing_keys:
                    self.log(f"[PRICE-ACTION] Skipping - Missing pivot keys: {', '.join(missing_keys)}", True)
                else:
                    # Run Price Action analysis
                    price_action_result = self.price_action_analyzer.analyze(
                        symbol=self.symbol,
                        ltp=ltp,
                        indicators=inds,
                        pivot_data=pivot_levels,
                        ohlc_df=ohlc_for_patterns
                    )

                    if price_action_result.get("signal"):
                        pa_signal = price_action_result["signal"]
                        pa_conf = price_action_result["confidence"]

                        signals["price_action"] = pa_signal
                        confidences["price_action"] = pa_conf

                        self.log(
                            f"\n{'=' * 60}\n"
                            f"[DATA] PRICE ACTION SIGNAL DETECTED\n"
                            f"{'=' * 60}\n"
                            f"Signal: {pa_signal}\n"
                            f"Confidence: {pa_conf:.2%}\n"
                            f"Key Level: {price_action_result.get('key_level')}\n"
                            f"Level Price: ₹{price_action_result.get('level_price'):.2f}\n"
                            f"Reason: {price_action_result.get('reason')}\n"
                            f"{'=' * 60}",
                            False
                        )
                    else:
                        self.log(
                            f"[PRICE-ACTION] No signal - {price_action_result.get('reason', 'Not at key level')}",
                            True
                        )

            except Exception as e:
                self.log(f"[PRICE-ACTION] Analysis error: {e}", False)
                import traceback
                self.log(f"[PRICE-ACTION] Traceback:\n{traceback.format_exc()}", True)

        # ==========================================
        # VOTE COUNTING & CONFLUENCE
        # ==========================================
        buy_votes = []
        sell_votes = []

        for source, signal in signals.items():
            if signal == "BUY":
                buy_votes.append((source, confidences[source]))
            elif signal == "SELL":
                sell_votes.append((source, confidences[source]))

        buy_count = len(buy_votes)
        sell_count = len(sell_votes)

        buy_score = sum(conf for _, conf in buy_votes)
        sell_score = sum(conf for _, conf in sell_votes)

        pa_in_buy = any(s == "price_action" for s, _ in buy_votes)
        pa_in_sell = any(s == "price_action" for s, _ in sell_votes)

        self.log(
            f"[UNIFIED] Signal Analysis:\n"
            f"  BUY: {buy_count} votes (score: {buy_score:.2f}) - {[f'{s}({c:.2f})' for s, c in buy_votes]}\n"
            f"  SELL: {sell_count} votes (score: {sell_score:.2f}) - {[f'{s}({c:.2f})' for s, c in sell_votes]}\n"
            f"  Price Action: {'BUY' if pa_in_buy else 'SELL' if pa_in_sell else 'None'}",
            True
        )

        # ==========================================
        # DECISION LOGIC
        # ==========================================
        final_signal = None
        reason = ""

        ai_in_buy = any(s == "ai_cpr" for s, _ in buy_votes)
        ai_in_sell = any(s == "ai_cpr" for s, _ in sell_votes)
        rejection_in_buy = any(s == "rejection_candle" for s, _ in buy_votes)
        rejection_in_sell = any(s == "rejection_candle" for s, _ in sell_votes)
        volume_breakout_in_buy = any(s == "volume_breakout" for s, _ in buy_votes)
        volume_breakout_in_sell = any(s == "volume_breakout" for s, _ in sell_votes)
        pa_signal = signals.get("price_action")
        pa_conf = confidences.get("price_action", 0.0)

        # Calculate momentum score
        momentum_score = 0.0
        if abs(momentum_pct) >= 0.8:
            momentum_score = 3.0
        elif abs(momentum_pct) >= 0.5:
            momentum_score = 2.0
        elif abs(momentum_pct) >= 0.3:
            momentum_score = 1.0

        if volume_ratio >= 2.0:
            momentum_score *= 1.5
        elif volume_ratio >= 1.5:
            momentum_score *= 1.2

        if pa_signal and pa_conf >= 0.80:
            # Check if it's contradicting 4+ signals
            opposing_votes = sell_count if pa_signal == "BUY" else buy_count

            if opposing_votes >= 4:
                self.log(
                    f"[WARNING] [PRICE-ACTION] High confidence ({pa_conf:.2f}) but "
                    f"{opposing_votes} strong opposing signals - treating as regular vote",
                    False
                )
            else:
                final_signal = pa_signal
                reason = f"HIGH CONFIDENCE PRICE ACTION at key level (conf: {pa_conf:.2f})"

                self.log(
                    f"\n{'=' * 60}\n"
                    f"[SIGNAL] PRIORITY-0: HIGH CONFIDENCE PRICE ACTION\n"
                    f"{'=' * 60}\n"
                    f"Signal: {final_signal}\n"
                    f"Confidence: {pa_conf:.2%}\n"
                    f"At KEY LEVEL - High conviction trade\n"
                    f"{'=' * 60}",
                    False
                )

        # PRIORITY 1: High Confidence Rejection
        if not final_signal and rejection_in_sell and confidences["rejection_candle"] >= 0.7:
            if sell_count >= 2 or (sell_count == 1 and confidences["rejection_candle"] >= 0.8):
                final_signal = "SELL"
                reason = f"EARLY REJECTION at resistance (conf: {confidences['rejection_candle']:.2f})"

        elif not final_signal and rejection_in_buy and confidences["rejection_candle"] >= 0.7:
            if buy_count >= 2 or (buy_count == 1 and confidences["rejection_candle"] >= 0.8):
                final_signal = "BUY"
                reason = f"EARLY REJECTION at support (conf: {confidences['rejection_candle']:.2f})"

        # PRIORITY 2: Volume Breakout
        if not final_signal and volume_breakout_in_buy and confidences["volume_breakout"] >= 0.7:
            if buy_count >= 2:
                final_signal = "BUY"
                reason = f"VOLUME BREAKOUT BUY (conf: {confidences['volume_breakout']:.2f}) + {buy_count - 1} confirmations"

        elif not final_signal and volume_breakout_in_sell and confidences["volume_breakout"] >= 0.7:
            if sell_count >= 2:
                final_signal = "SELL"
                reason = f"VOLUME BREAKOUT SELL (conf: {confidences['volume_breakout']:.2f}) + {sell_count - 1} confirmations"


        # PRIORITY 3: High Confidence AI
        if not final_signal and ai_in_buy and confidences["ai_cpr"] >= 0.75:
            opposing_signals = sell_count
            supporting_signals = buy_count - 1

            if supporting_signals >= 1:
                final_signal = "BUY"
                reason = f"HIGH CONFIDENCE AI BUY ({confidences['ai_cpr']:.2f}) + {supporting_signals} support"
            elif opposing_signals <= 2:
                final_signal = "BUY"
                reason = f"HIGH CONFIDENCE AI BUY ({confidences['ai_cpr']:.2f}) solo (weak opposition: {opposing_signals})"

        elif not final_signal and ai_in_sell and confidences["ai_cpr"] >= 0.75:
            opposing_signals = buy_count
            supporting_signals = sell_count - 1

            if supporting_signals >= 1:
                final_signal = "SELL"
                reason = f"HIGH CONFIDENCE AI SELL ({confidences['ai_cpr']:.2f}) + {supporting_signals} support"
            elif opposing_signals <= 2:
                final_signal = "SELL"
                reason = f"HIGH CONFIDENCE AI SELL ({confidences['ai_cpr']:.2f}) solo (weak opposition: {opposing_signals})"

        # PRIORITY 4: Strong Confluence (3+ votes)
        if not final_signal and buy_count >= 3:
            has_vwap_confirm = signals.get("vwap") == "BUY"
            has_bid_ask_confirm = signals.get("bid_ask_pressure") == "BUY"

            if has_vwap_confirm or has_bid_ask_confirm:
                final_signal = "BUY"
                ai_part = f" + AI({confidences['ai_cpr']:.2f})" if ai_in_buy else ""
                pa_part = f" + PriceAction({pa_conf:.2f})" if pa_in_buy else ""
                vwap_part = " + VWAP" if has_vwap_confirm else ""
                bid_ask_part = " + BidAsk" if has_bid_ask_confirm else ""
                reason = f"Strong BUY ({buy_count}/10 votes, score: {buy_score:.2f}){ai_part}{pa_part}{vwap_part}{bid_ask_part}"

        elif not final_signal and sell_count >= 3:
            has_vwap_confirm = signals.get("vwap") == "SELL"
            has_bid_ask_confirm = signals.get("bid_ask_pressure") == "SELL"

            if has_vwap_confirm or has_bid_ask_confirm:
                final_signal = "SELL"
                ai_part = f" + AI({confidences['ai_cpr']:.2f})" if ai_in_sell else ""
                pa_part = f" + PriceAction({pa_conf:.2f})" if pa_in_sell else ""
                vwap_part = " + VWAP" if has_vwap_confirm else ""
                bid_ask_part = " + BidAsk" if has_bid_ask_confirm else ""
                reason = f"Strong SELL ({sell_count}/10 votes, score: {sell_score:.2f}){ai_part}{pa_part}{vwap_part}{bid_ask_part}"

        # PRIORITY 5: Medium Confluence (2 votes)
        # python
        elif buy_count == 2:
            ai_conf = confidences.get("ai_cpr", 0.0)
            vwap_confirm = signals.get("vwap") == "BUY"
            bid_ask_confirm = signals.get("bid_ask_pressure") == "BUY"

            # Stronger path when AI supports the 2-vote setup
            if ai_in_buy and ai_conf >= self.MEDIUM_CONF_MIN_AI and buy_score >= self.MEDIUM_CONF_MIN_SCORE_WITH_AI:
                final_signal = "BUY"
                reason = (f"Medium BUY confluence (2/10 votes) with AI support "
                          f"(AI: {ai_conf:.2f}, score: {buy_score:.2f})")
            # Allow non-AI medium confluence if score is strong enough
            elif buy_score >= self.MEDIUM_CONF_MIN_SCORE and (
                    not self.AI_GATE_TRADES or ai_conf >= self.AI_MIN_CONF or not ai_in_buy):
                final_signal = "BUY"
                reason = f"Medium BUY confluence (2/10 votes, score: {buy_score:.2f})"
                if vwap_confirm or bid_ask_confirm:
                    reason += " + confirmation"
            # Lower-threshold acceptance when VWAP or Bid-Ask provides the needed confirmation
            elif (vwap_confirm or bid_ask_confirm) and buy_score >= 1.0:
                final_signal = "BUY"
                reason = f"Medium BUY (2 votes) accepted due to VWAP/Bid-Ask confirmation (score: {buy_score:.2f})"
            else:
                final_signal = None
                reason = f"2-vote BUY insufficient (score: {buy_score:.2f}, AI: {ai_conf:.2f}) - waiting for more confirmation"

        elif sell_count == 2:
            ai_conf = confidences.get("ai_cpr", 0.0)
            vwap_confirm = signals.get("vwap") == "SELL"
            bid_ask_confirm = signals.get("bid_ask_pressure") == "SELL"

            if ai_in_sell and ai_conf >= self.MEDIUM_CONF_MIN_AI and sell_score >= self.MEDIUM_CONF_MIN_SCORE_WITH_AI:
                final_signal = "SELL"
                reason = (f"Medium SELL confluence (2/10 votes) with AI support "
                          f"(AI: {ai_conf:.2f}, score: {sell_score:.2f})")
            elif sell_score >= self.MEDIUM_CONF_MIN_SCORE and (
                    not self.AI_GATE_TRADES or ai_conf >= self.AI_MIN_CONF or not ai_in_sell):
                final_signal = "SELL"
                reason = f"Medium SELL confluence (2/10 votes, score: {sell_score:.2f})"
                if vwap_confirm or bid_ask_confirm:
                    reason += " + confirmation"
            elif (vwap_confirm or bid_ask_confirm) and sell_score >= 1.0:
                final_signal = "SELL"
                reason = f"Medium SELL (2 votes) accepted due to VWAP/Bid-Ask confirmation (score: {sell_score:.2f})"
            else:
                final_signal = None
                reason = f"2-vote SELL insufficient (score: {sell_score:.2f}, AI: {ai_conf:.2f}) - waiting for more confirmation"

        # ==========================================
        # 🔥 CRITICAL FIX: ENHANCED TREND VALIDATION
        # ==========================================
        if final_signal and current_pos == "FLAT":
            detected_trend = inds.get("trend")

            # [OK] FIX: Handle None/missing trend gracefully
            if not detected_trend or detected_trend == "None" or str(detected_trend).lower() == "none":
                self.log(
                    f"[WARNING] [TREND-WARNING] No trend detected, using EMA fallback",
                    True
                )

                # Fallback to EMA9 vs EMA21
                e9 = self._f(inds.get("ema_9"))
                e21 = self._f(inds.get("ema_21"))

                if e9 and e21:
                    trend_validated = False

                    if final_signal == "BUY":
                        if e9 > e21:
                            trend_validated = True
                            self.log(
                                f"[OK] [EMA-FALLBACK] BUY allowed - EMA9({e9:.2f}) > EMA21({e21:.2f})",
                                False
                            )
                        else:
                            self.log(
                                f"⛔ [EMA-FALLBACK] BUY blocked - EMA9({e9:.2f}) < EMA21({e21:.2f})",
                                False
                            )
                            final_signal = None

                    elif final_signal == "SELL":
                        if e9 < e21:
                            trend_validated = True
                            self.log(
                                f"[OK] [EMA-FALLBACK] SELL allowed - EMA9({e9:.2f}) < EMA21({e21:.2f})",
                                False
                            )
                        else:
                            self.log(
                                f"⛔ [EMA-FALLBACK] SELL blocked - EMA9({e9:.2f}) > EMA21({e21:.2f})",
                                False
                            )
                            final_signal = None
                else:
                    # [OK] EMERGENCY FALLBACK: No EMAs available
                    vote_count = buy_count if final_signal == "BUY" else sell_count
                    vote_score = buy_score if final_signal == "BUY" else sell_score
                    ai_conf = confidences.get("ai_cpr", 0)

                    # Allow if very strong signal
                    if vote_count >= 4 and vote_score >= 2.5 and ai_conf >= 0.35:
                        self.log(
                            f"[OK] [EMERGENCY-OVERRIDE] {final_signal} allowed without trend data\n"
                            f"  Reason: Very strong confluence\n"
                            f"  Votes: {vote_count} (≥4)\n"
                            f"  Score: {vote_score:.2f} (≥2.5)\n"
                            f"  AI confidence: {ai_conf:.2f} (≥0.35)",
                            False
                        )
                    else:
                        self.log(
                            f"⛔ [TREND-BLOCK] {final_signal} blocked - No trend/EMA data\n"
                            f"  Votes: {vote_count} (need 4+)\n"
                            f"  Score: {vote_score:.2f} (need 2.5+)\n"
                            f"  AI: {ai_conf:.2f} (need 0.65+)",
                            False
                        )
                        final_signal = None

            else:
                # [OK] NORMAL CASE: Trend detected
                trend_str = str(detected_trend).lower()
                trend_validated = False

                if final_signal == "BUY":
                    # Allow BUY in uptrend
                    if "up" in trend_str or "bull" in trend_str:
                        trend_validated = True
                        self.log(f"[OK] [TREND] BUY allowed - {detected_trend}", True)

                    # Allow breakout from consolidation
                    elif "consol" in trend_str:
                        breakout_valid = (
                                (volume_ratio >= 1.0 and momentum_score >= 1.2) or
                                (buy_count >= 4) or
                                (ai_in_buy and confidences["ai_cpr"] >= 0.75 and buy_count >= 2)
                        )

                        if breakout_valid:
                            trend_validated = True
                            self.log(
                                f"[OK] [BREAKOUT] BUY allowed from consolidation\n"
                                f"  Volume: {volume_ratio:.2f}x | Momentum: {momentum_pct:.2f}%\n"
                                f"  Votes: {buy_count} | AI: {confidences.get('ai_cpr', 0):.2f}",
                                False
                            )
                        else:
                            self.log(
                                f"⛔ [CONSOL-BLOCK] BUY rejected - Weak breakout\n"
                                f"  Volume: {volume_ratio:.2f}x (need 1.8x+)\n"
                                f"  Votes: {buy_count} (need 4+)",
                                False
                            )

                    # Allow counter-trend with VERY strong signal
                    elif "down" in trend_str or "bear" in trend_str:
                        counter_trend_valid = (
                                buy_count >= 5 and
                                buy_score >= 3.0 and
                                ai_in_buy and
                                confidences["ai_cpr"] >= 0.75 and
                                momentum_pct >= 0.8 and
                                volume_ratio >= 1.2
                        )

                        if counter_trend_valid:
                            trend_validated = True
                            self.log(
                                f"[OK] [COUNTER-TREND] BUY allowed against downtrend\n"
                                f"  Votes: {buy_count} | Score: {buy_score:.2f}\n"
                                f"  AI: {confidences['ai_cpr']:.2f} | Momentum: {momentum_pct:.2f}%\n"
                                f"  Volume: {volume_ratio:.2f}x\n"
                                f"  [WARNING] WARNING: Counter-trend trade - higher risk!",
                                False
                            )

                elif final_signal == "SELL":
                    if "down" in trend_str or "bear" in trend_str:
                        trend_validated = True
                        self.log(f"[OK] [TREND] SELL allowed - {detected_trend}", True)

                    elif "consol" in trend_str:
                        breakout_valid = (
                                (volume_ratio >= 1.0 and momentum_score >= 1.2) or
                                (sell_count >= 4) or
                                (ai_in_sell and confidences["ai_cpr"] >= 0.75 and sell_count >= 2)
                        )

                        if breakout_valid:
                            trend_validated = True
                            self.log(
                                f"[OK] [BREAKOUT] SELL allowed from consolidation\n"
                                f"  Volume: {volume_ratio:.2f}x | Momentum: {momentum_pct:.2f}%\n"
                                f"  Votes: {sell_count} | AI: {confidences.get('ai_cpr', 0):.2f}",
                                False
                            )

                    elif "up" in trend_str or "bull" in trend_str:
                        counter_trend_valid = (
                                sell_count >= 5 and
                                sell_score >= 3.0 and
                                ai_in_sell and
                                confidences["ai_cpr"] >= 0.75 and
                                momentum_pct <= -0.8 and
                                volume_ratio >= 1.2
                        )

                        if counter_trend_valid:
                            trend_validated = True
                            self.log(
                                f"[OK] [COUNTER-TREND] SELL allowed against uptrend\n"
                                f"  Votes: {sell_count} | Score: {sell_score:.2f}\n"
                                f"  AI: {confidences['ai_cpr']:.2f} | Momentum: {momentum_pct:.2f}%\n"
                                f"  Volume: {volume_ratio:.2f}x\n"
                                f"  [WARNING] WARNING: Counter-trend trade - higher risk!",
                                False
                            )

                if not trend_validated:
                    self.log(
                        f"⛔ [TREND-BLOCK] {final_signal} rejected\n"
                        f"  Detected trend: {detected_trend}\n"
                        f"  Required: {'Uptrend or strong breakout' if final_signal == 'BUY' else 'Downtrend or strong breakout'}",
                        False
                    )
                    final_signal = None

            # ==========================================
            # LAYER 2: MANDATORY MOMENTUM VALIDATION
            # ==========================================
            if final_signal:
                momentum_validated = False

                # Exception 1: High-confidence rejection signals
                is_rejection = (
                        signals.get("rejection_candle") == final_signal and
                        confidences.get("rejection_candle", 0) >= 0.75
                )

                # Exception 2: Extreme volume breakout
                is_volume_breakout = (
                        signals.get("volume_breakout") == final_signal and
                        confidences.get("volume_breakout", 0) >= 0.80
                )

                if is_rejection or is_volume_breakout:
                    momentum_validated = True
                    skip_reason = "rejection" if is_rejection else "volume breakout"
                    self.log(
                        f"[OK] [MOMENTUM-SKIP] {skip_reason.upper()} bypasses momentum check",
                        True
                    )
                else:
                    # Strict momentum requirements
                    if final_signal == "BUY":
                        if adx_val >= 30:
                            min_momentum = 0.2
                        elif adx_val >= 25:
                            min_momentum = 0.3
                        else:
                            min_momentum = 0.5

                        momentum_validated = (momentum_pct >= min_momentum)

                        if not momentum_validated:
                            self.log(
                                f"⛔ [MOMENTUM-BLOCK] BUY rejected - Insufficient momentum\n"
                                f"  Current: {momentum_pct:.2f}%\n"
                                f"  Required: {min_momentum:.2f}% (ADX: {adx_val:.1f})\n"
                                f"  Momentum score: {momentum_score:.1f}/3.0",
                                False
                            )

                    elif final_signal == "SELL":
                        if adx_val >= 30:
                            min_momentum = -0.2
                        elif adx_val >= 25:
                            min_momentum = -0.3
                        else:
                            min_momentum = -0.5

                        momentum_validated = (momentum_pct <= min_momentum)

                        if not momentum_validated:
                            self.log(
                                f"⛔ [MOMENTUM-BLOCK] SELL rejected - Insufficient momentum\n"
                                f"  Current: {momentum_pct:.2f}%\n"
                                f"  Required: {min_momentum:.2f}% (ADX: {adx_val:.1f})\n"
                                f"  Momentum score: {momentum_score:.1f}/3.0",
                                False
                            )

                if not momentum_validated:
                    final_signal = None

            # ==========================================
            # LAYER 3: SUPERTREND VALIDATION
            # ==========================================
            if final_signal:
                supertrend = self._f(inds.get("supertrend"))
                market_regime = self.detect_market_regime(
                    ohlc_for_patterns,
                    inds
                )

                if supertrend:
                    st_trend = inds.get("st_main_trend", 0)

                    bypass_st, bypass_reason = self.should_bypass_supertrend(
                        final_signal,
                        signals,
                        confidences,
                        market_regime
                    )

                    if bypass_st:
                        self.log(
                            f"[START] [ST-ADAPTIVE] SuperTrend check BYPASSED\n"
                            f"  Reason: {bypass_reason}\n"
                            f"  Signal: {final_signal}\n"
                            f"  Proceeding to entry...",
                            False
                        )
                    else:
                        self.log(
                            f"🛡️ [ST-ADAPTIVE] SuperTrend check ENFORCED\n"
                            f"  Signal: {final_signal}\n"
                            f"  Reason: {bypass_reason}",
                            True
                        )

                        if final_signal == "SELL":
                            gap = ltp - supertrend
                            gap_pct = (gap / ltp) * 100 if ltp > 0 else 0

                            # [OK] CORRECT LOGIC FOR SELL
                            if ltp < supertrend:
                                self.log(
                                    f"[OK] [SUPERTREND] SELL allowed - Price in bearish zone | "
                                    f"LTP:{ltp:.2f} < ST:{supertrend:.2f}",
                                    False
                                )

                            elif gap_pct < 0.3:
                                if sell_count >= 4:
                                    self.log(
                                        f"[OK] [SUPERTREND] SELL allowed - Imminent breakdown + strong signals | "
                                        f"LTP:{ltp:.2f}, ST:{supertrend:.2f} (gap:{gap_pct:.3f}%) | "
                                        f"{sell_count} SELL votes",
                                        False
                                    )
                                else:
                                    self.log(
                                        f"⛔ [SUPERTREND] SELL BLOCKED - Too close without strong confirmation | "
                                        f"LTP:{ltp:.2f}, ST:{supertrend:.2f} (gap:{gap_pct:.3f}%) | "
                                        f"Only {sell_count} votes (need 4+ for override)",
                                        False
                                    )
                                    final_signal = None

                            elif signals.get("rejection_candle") == "SELL" and confidences.get("rejection_candle",
                                                                                               0) >= 0.75:
                                self.log(
                                    f"[OK] [SUPERTREND] SELL allowed - Rejection override | "
                                    f"LTP:{ltp:.2f}, ST:{supertrend:.2f} (gap:{gap_pct:.3f}%) | "
                                    f"Rejection conf: {confidences.get('rejection_candle', 0):.2f}",
                                    False
                                )

                            else:
                                self.log(
                                    f"⛔ [SUPERTREND] SELL BLOCKED - Price above SuperTrend | "
                                    f"LTP:{ltp:.2f} > ST:{supertrend:.2f} (gap:₹{gap:.2f} / {gap_pct:.2f}%) | "
                                    f"Not in bearish zone",
                                    False
                                )
                                final_signal = None

                        elif final_signal == "BUY":
                            gap = supertrend - ltp
                            gap_pct = (gap / ltp) * 100 if ltp > 0 else 0

                            if ltp > supertrend:
                                self.log(
                                    f"[OK] [SUPERTREND] BUY allowed - Price in bullish zone | "
                                    f"LTP:{ltp:.2f} > ST:{supertrend:.2f}",
                                    False
                                )

                            elif gap_pct < 0.3:
                                if buy_count >= 4:
                                    self.log(
                                        f"[OK] [SUPERTREND] BUY allowed - Imminent breakout + strong signals | "
                                        f"LTP:{ltp:.2f}, ST:{supertrend:.2f} (gap:{gap_pct:.3f}%) | "
                                        f"{buy_count} BUY votes",
                                        False
                                    )
                                else:
                                    self.log(
                                        f"⛔ [SUPERTREND] BUY BLOCKED - Too close without strong confirmation | "
                                        f"LTP:{ltp:.2f}, ST:{supertrend:.2f} (gap:{gap_pct:.3f}%) | "
                                        f"Only {buy_count} votes (need 4+ for override)",
                                        False
                                    )
                                    final_signal = None

                            elif signals.get("rejection_candle") == "BUY" and confidences.get("rejection_candle",
                                                                                              0) >= 0.75:
                                self.log(
                                    f"[OK] [SUPERTREND] BUY allowed - Rejection override | "
                                    f"LTP:{ltp:.2f}, ST:{supertrend:.2f} (gap:{gap_pct:.3f}%) | "
                                    f"Rejection conf: {confidences.get('rejection_candle', 0):.2f}",
                                    False
                                )

                            else:
                                self.log(
                                    f"⛔ [SUPERTREND] BUY BLOCKED - Price below SuperTrend | "
                                    f"LTP:{ltp:.2f} < ST:{supertrend:.2f} (gap:₹{gap:.2f} / {gap_pct:.2f}%) | "
                                    f"Not in bullish zone",
                                    False
                                )
                                final_signal = None

                else:
                    self.log("[WARNING] [SUPERTREND] No SuperTrend data - skipping check", True)

            # ==========================================
            # LAYER 4: RSI EXTREME CONDITIONS FILTER
            # ==========================================
            if final_signal:
                rsi = self._f(inds.get("adx_efi", {}).get("RSI"))

                if rsi:
                    if final_signal == "BUY" and rsi < 30:
                        self.log(
                            f"⛔ [RSI-FILTER] BUY BLOCKED - RSI too low | "
                            f"RSI:{rsi:.1f} < 30 (oversold, may continue down)",
                            False
                        )
                        final_signal = None
                    elif final_signal == "SELL" and rsi > 70:
                        self.log(
                            f"⛔ [RSI-FILTER] SELL BLOCKED - RSI too high | "
                            f"RSI:{rsi:.1f} > 70 (overbought, may bounce)",
                            False
                        )
                        final_signal = None
                else:
                    self.log("[WARNING] [RSI-FILTER] No RSI data - skipping check", True)

            # ==========================================
            # LAYER 5: ADX TREND STRENGTH VALIDATION
            # ==========================================
            if final_signal:
                adx_val = self._f(inds.get("adx"))

                if adx_val:
                    if adx_val < 20:
                        vote_count = buy_count if final_signal == "BUY" else sell_count

                        if vote_count < 4:
                            self.log(
                                f"⛔ [ADX-FILTER] {final_signal} BLOCKED - Weak trend | "
                                f"ADX:{adx_val:.1f} < 20, only {vote_count} votes (need 4+)",
                                False
                            )
                            final_signal = None
                        else:
                            self.log(
                                f"[WARNING] [ADX-FILTER] {final_signal} allowed despite weak ADX:{adx_val:.1f} "
                                f"(strong confluence: {vote_count} votes)",
                                False
                            )
                else:
                    self.log("[WARNING] [ADX-FILTER] No ADX data - skipping check", True)

            # ==========================================
            # LAYER 6: VOLATILITY / CHOPPINESS FILTER
            # ==========================================
            if final_signal:
                bb_bandwidth = self._f(inds.get("bb_bandwidth"))

                if bb_bandwidth:
                    now = dt.datetime.now(self.IST)
                    is_morning = (now.hour == 9 and now.minute >= 15) or (now.hour == 10 and now.minute < 30)

                    if bb_bandwidth < 0.003 and not is_morning:
                        self.log(f"⛔ [VOLATILITY] {final_signal} BLOCKED - Market too choppy (BBW: {bb_bandwidth:.4f})",
                                 False)
                        final_signal = None
                    elif bb_bandwidth < 0.003 and is_morning:
                        self.log(f"[WARNING] [VOLATILITY] Low volatility but allowing morning breakout", True)
                else:
                    self.log("[WARNING] [VOLATILITY] No BB data - skipping check", True)

            # ==========================================
            # [OK] ALL FILTERS PASSED - EXECUTE ENTRY
            # ==========================================
            if final_signal:
                self.log(
                    f"[OK] [FILTER-SUMMARY] {final_signal} APPROVED - Passed all 6 filters | "
                    f"Trend✓ Momentum✓ SuperTrend✓ RSI✓ ADX✓ Volatility✓",
                    False
                )

                atr_here = self._get_atr_with_fallback(inds, ltp)
                if atr_here:
                    sources = [f"{s}({c:.2f})" for s, c in (buy_votes if final_signal == "BUY" else sell_votes)]
                    full_reason = f"{reason} | Sources: {', '.join(sources)}"

                    if self._process_entry(final_signal, full_reason, ltp, atr_here, bar_key=bar_key, indsP=inds):
                        self.position["_last_action_bar"] = bar_key
                        self.position["ai_entry_confidence"] = confidences.get("ai_cpr", 0.0)
                        self.position["ai_distribution"] = ai_distribution
                        self._set_cooldown(self.FLIP_COOLDOWN_BARS)
                        self._save_state()
                        self.log(
                            f"[UNIFIED] [OK] Entry executed: {final_signal} | AI Confidence: {confidences.get('ai_cpr', 'N/A')}",
                            False
                        )
                        return

        # ==========================================
        # [START] FAST REVERSAL ENTRY
        # ==========================================
        if current_pos != "FLAT":
            # Initialize reversal history if needed
            if not hasattr(self.position, '_reversal_history'):
                self.position['_reversal_history'] = []

            # Clean old history (keep last 30 minutes)
            current_time = dt.datetime.now(self.IST)
            self.position['_reversal_history'] = [
                r for r in self.position['_reversal_history']
                if (current_time - r['time']).total_seconds() < 1800
            ]

            recent_reversals = len(self.position['_reversal_history'])
            MAX_REVERSALS_30MIN = 3

            # Check for opposing signal
            opposing_signal = None
            opposing_score = 0
            opposing_votes = []

            if current_pos == "BUY" and sell_count >= 3:
                opposing_signal = "SELL"
                opposing_score = sell_score
                opposing_votes = sell_votes
            elif current_pos == "SELL" and buy_count >= 3:
                opposing_signal = "BUY"
                opposing_score = buy_score
                opposing_votes = buy_votes

            if opposing_signal:
                # Check whipsaw protection
                if recent_reversals >= MAX_REVERSALS_30MIN:
                    last_reversal_time = self.position['_reversal_history'][-1]['time']
                    time_since_last = (current_time - last_reversal_time).total_seconds()

                    if time_since_last < 600:  # 10 minutes
                        self.log(
                            f"⛔ [WHIPSAW-PROTECTION] Too many reversals\n"
                            f"  Recent reversals: {recent_reversals}/{MAX_REVERSALS_30MIN}\n"
                            f"  Last reversal: {time_since_last / 60:.1f} min ago\n"
                            f"  Required cooldown: 10 min\n"
                            f"  Total loss from reversals: ₹{sum(r['loss'] for r in self.position['_reversal_history']):.2f}",
                            False
                        )

                        # Switch to defensive mode
                        current_sl = self._f(self.position.get("stop_loss"))
                        entry_price = self._f(self.position.get("entry_price"))
                        atr_here = self._get_atr_with_fallback(inds, ltp)

                        if entry_price and atr_here and current_sl:
                            tighter_sl_distance = 0.8 * atr_here

                            if current_pos == "BUY":
                                new_sl = max(current_sl, ltp - tighter_sl_distance)
                                if new_sl > current_sl:
                                    self.position["stop_loss"] = new_sl
                                    self.log(
                                        f"🛡️ [DEFENSIVE-SL] Tightened stop loss\n"
                                        f"  Old: ₹{current_sl:.2f} → New: ₹{new_sl:.2f}\n"
                                        f"  Distance: {(ltp - new_sl):.2f} ({tighter_sl_distance:.2f} ATR)",
                                        False
                                    )

                            elif current_pos == "SELL":
                                new_sl = min(current_sl, ltp + tighter_sl_distance)
                                if new_sl < current_sl:
                                    self.position["stop_loss"] = new_sl
                                    self.log(
                                        f"🛡️ [DEFENSIVE-SL] Tightened stop loss\n"
                                        f"  Old: ₹{current_sl:.2f} → New: ₹{new_sl:.2f}\n"
                                        f"  Distance: {(new_sl - ltp):.2f} ({tighter_sl_distance:.2f} ATR)",
                                        False
                                    )

                        opposing_signal = None  # Block reversal

                if opposing_signal:
                    # Validation 1: Trend confirmation
                    detected_trend = inds.get("trend")
                    trend_confirms = False

                    if detected_trend:
                        trend_str = str(detected_trend).lower()
                        if opposing_signal == "BUY":
                            trend_confirms = ("up" in trend_str or "bull" in trend_str)
                        elif opposing_signal == "SELL":
                            trend_confirms = ("down" in trend_str or "bear" in trend_str)

                    # Validation 2: Momentum confirmation
                    momentum_pct = self._f(inds.get("momentum_pct"), 0.0)
                    volume_ratio = self._f(inds.get("volume_ratio"), 1.0)

                    momentum_confirms = (
                            (opposing_signal == "BUY" and momentum_pct >= 0.5) or
                            (opposing_signal == "SELL" and momentum_pct <= -0.5)
                    )

                    # Validation 3: Volume confirmation
                    volume_confirms = False
                    if volume_ratio >= 1.5:
                        volume_confirms = True
                    elif opposing_score >= 2.5:
                        volume_confirms = True
                        self.log(
                            f"[WARNING] [REVERSAL-VOLUME] Low volume ({volume_ratio:.2f}x) but strong score ({opposing_score:.2f})",
                            True
                        )

                    if not volume_confirms:
                        self.log(
                            f"⛔ [REVERSAL-BLOCK] Insufficient volume\n"
                            f"  Volume: {volume_ratio:.2f}x (need 1.5x+)\n"
                            f"  Score: {opposing_score:.2f} (need 2.5+ to bypass)",
                            False
                        )
                        opposing_signal = None

                    # Validation 4: Check current loss
                    entry_price = self._f(self.position.get("entry_price"))
                    current_pnl = 0
                    loss_pct = 0

                    if entry_price and ltp:
                        if current_pos == "BUY":
                            current_pnl = ltp - entry_price
                        else:
                            current_pnl = entry_price - ltp

                        loss_pct = (abs(current_pnl) / entry_price * 100) if entry_price else 0

                    # Decision: Execute fast reversal
                    should_reverse = (
                            opposing_signal and
                            trend_confirms and
                            momentum_confirms and
                            volume_confirms and
                            current_pnl < 0 and
                            loss_pct < 1.5
                    )

                    if should_reverse:
                        self.log(
                            f"\n🔄🔄🔄 FAST REVERSAL TRIGGERED 🔄🔄🔄\n"
                            f"{'=' * 60}\n"
                            f"Current Position: {current_pos}\n"
                            f"New Signal: {opposing_signal}\n"
                            f"{'=' * 60}\n"
                            f"VALIDATION:\n"
                            f"  [OK] Trend: {detected_trend}\n"
                            f"  [OK] Momentum: {momentum_pct:.2f}%\n"
                            f"  [OK] Volume: {volume_ratio:.2f}x\n"
                            f"  [OK] Opposing votes: {len(opposing_votes)} (score: {opposing_score:.2f})\n"
                            f"{'=' * 60}\n"
                            f"CURRENT TRADE:\n"
                            f"  Entry: ₹{entry_price:.2f}\n"
                            f"  Current: ₹{ltp:.2f}\n"
                            f"  P&L: ₹{current_pnl:.2f} ({loss_pct:.2f}%)\n"
                            f"{'=' * 60}\n"
                            f"ACTION: Exit {current_pos} → Enter {opposing_signal}\n"
                            f"{'=' * 60}",
                            False
                        )

                        exit_reason = f"Fast Reversal: Strong {opposing_signal} signal ({opposing_score:.2f})"

                        if self._process_exit(exit_reason, ltp):
                            self.log(f"[OK] [REVERSAL] Exit completed", False)

                            atr_here = self._get_atr_with_fallback(inds, ltp)

                            if atr_here:
                                sources = [f"{s}({c:.2f})" for s, c in opposing_votes]
                                entry_reason = (
                                    f"Fast Reversal {opposing_signal} | "
                                    f"Votes: {len(opposing_votes)} | "
                                    f"Score: {opposing_score:.2f} | "
                                    f"Sources: {', '.join(sources[:3])}"
                                )

                                if self._process_entry(
                                        opposing_signal,
                                        entry_reason,
                                        ltp,
                                        atr_here,
                                        bar_key=bar_key,
                                        indsP=inds
                                ):
                                    self.position["_last_action_bar"] = bar_key
                                    self.position["ai_entry_confidence"] = confidences.get("ai_cpr", 0.0)
                                    self.position["_cooldown_bars"] = 0

                                    # Record reversal
                                    self.position['_reversal_history'].append({
                                        'time': current_time,
                                        'from': current_pos,
                                        'to': opposing_signal,
                                        'loss': abs(current_pnl),
                                        'price': ltp
                                    })

                                    self.log(
                                        f"[DATA] [REVERSAL-HISTORY] Total reversals in last 30min: {len(self.position['_reversal_history'])}",
                                        True
                                    )

                                    self._save_state()

                                    self.log(
                                        f"[OK][OK][OK] [REVERSAL-COMPLETE] Now in {opposing_signal} position\n",
                                        False
                                    )
                                    return
                            else:
                                self.log(f"[ERROR] [REVERSAL-FAILED] ATR unavailable for re-entry", False)
                        else:
                            self.log(f"[ERROR] [REVERSAL-FAILED] Exit failed", False)

                    else:
                        rejection_reasons = []
                        if not trend_confirms:
                            rejection_reasons.append(f"Trend: {detected_trend}")
                        if not momentum_confirms:
                            rejection_reasons.append(f"Momentum: {momentum_pct:.2f}%")
                        if not volume_confirms:
                            rejection_reasons.append(f"Volume: {volume_ratio:.2f}x")
                        if current_pnl >= 0:
                            rejection_reasons.append(f"Currently winning: ₹{current_pnl:.2f}")
                        if loss_pct >= 1.5:
                            rejection_reasons.append(f"Loss too large: {loss_pct:.2f}%")

                        self.log(
                            f"[WARNING] [REVERSAL-SKIP] {opposing_signal} signal but reversal not validated\n"
                            f"  Reasons: {', '.join(rejection_reasons)}",
                            True
                        )

        # ==========================================
        # EXIT LOGIC - MULTIPLE LAYERS
        # ==========================================
        reversal_at_resistance = False
        candle_patterns = {}
        pivot_interactions = {}

        if cpr_analysis and isinstance(cpr_analysis, dict):
            reversal_at_resistance = cpr_analysis.get("reversal_at_r2_r3", False)
            candle_patterns = cpr_analysis.get("candle_patterns", {})
            pivot_interactions = cpr_analysis.get("pivot_interactions", {})

            if not isinstance(candle_patterns, dict):
                candle_patterns = {}
            if not isinstance(pivot_interactions, dict):
                pivot_interactions = {}

        # LAYER 1: CPR Reversal at major levels
        if current_pos == "BUY" and reversal_at_resistance:
            bearish_signals = []
            if candle_patterns.get("big_bear_takeout"):
                bearish_signals.append("Big Bear Take Out")
            if candle_patterns.get("fake_bull"):
                bearish_signals.append("Fake Bull")
            if candle_patterns.get("bear_retracement"):
                bearish_signals.append("Bear Retracement")

            if bearish_signals:
                pattern_str = ", ".join(bearish_signals)
                reason = f"EXIT LONG - Bearish reversal at R2/R3 resistance ({pattern_str})"
                self.log(f"[CPR-REVERSAL] {reason}", False)

                if self._process_exit(reason, ltp):
                    # Zero cooldown if strong momentum, else 1 bar
                    cooldown_bars = 0 if (abs(self._f(inds.get("momentum_pct"), 0)) > 0.5 and self._f(inds.get("adx"),
                                                                                                      0) > 25) else 1
                    self._set_cooldown(cooldown_bars)
                    #self._set_cooldown(1)
                    self._save_state()
                    return

        if current_pos == "SELL":
            at_s2 = pivot_interactions.get("at_s2", False)
            at_s3 = pivot_interactions.get("at_s3", False)
            tested_s2 = pivot_interactions.get("tested_s2", False)
            tested_s3 = pivot_interactions.get("tested_s3", False)

            bullish_signals = []
            if candle_patterns.get("big_bull_takeout"):
                bullish_signals.append("Big Bull Take Out")
            if candle_patterns.get("fake_bear"):
                bullish_signals.append("Fake Bear")
            if candle_patterns.get("bull_retracement"):
                bullish_signals.append("Bull Retracement")

            if (at_s2 or at_s3 or tested_s2 or tested_s3) and bullish_signals:
                pattern_str = ", ".join(bullish_signals)
                level = "S2" if (at_s2 or tested_s2) else "S3"
                reason = f"EXIT SHORT - Bullish reversal at {level} support ({pattern_str})"
                self.log(f"[CPR-REVERSAL] {reason}", False)

                if self._process_exit(reason, ltp):
                    # Zero cooldown if strong momentum, else 1 bar
                    cooldown_bars = 0 if (abs(self._f(inds.get("momentum_pct"), 0)) > 0.5 and self._f(inds.get("adx"),
                                                                                                      0) > 25) else 1
                    self._set_cooldown(cooldown_bars)
                    #self._set_cooldown(1)
                    self._save_state()
                    return

        # LAYER 2: High confidence AI reversal
        if current_pos == "BUY":
            if ai_in_sell and confidences["ai_cpr"] >= 0.70:
                # 🔥 CHECK: Verify trend actually reversed
                momentum_pct = self._f(inds.get("momentum_pct"), 0.0)
                ema5 = self._f(inds.get("ema_5"))
                ema21 = self._f(inds.get("ema_21"))

                # AI says SELL, but is trend actually bearish?
                trend_confirms_sell = (
                        momentum_pct < -0.3 and  # Momentum turned negative
                        (ema5 is None or ema21 is None or ema5 < ema21)  # EMAs bearish
                )

                if trend_confirms_sell:
                    reason = f"EXIT LONG - AI reversal + trend confirmed (SELL: {confidences['ai_cpr']:.2f}, Mom: {momentum_pct:.2f}%)"
                    self.log(f"[AI-EXIT] {reason} | Distribution: {ai_distribution}", False)

                    if self._process_exit(reason, ltp):
                        cooldown_bars = 0 if abs(momentum_pct) > 0.5 else 1
                        self._set_cooldown(cooldown_bars)
                        self._save_state()
                        return
                else:
                    self.log(
                        f"[WARNING] [AI-EXIT-BLOCKED] AI says SELL but trend NOT bearish\n"
                        f"  Momentum: {momentum_pct:.2f}% (need < -0.3%)\n"
                        f"  EMA5: {ema5:.2f}, EMA21: {ema21:.2f}\n"
                        f"  Decision: HOLD position (AI may be early)",
                        False
                    )

        elif current_pos == "SELL":
            if ai_in_buy and confidences["ai_cpr"] >= 0.70:
                # 🔥 CHECK: Verify trend actually reversed
                momentum_pct = self._f(inds.get("momentum_pct"), 0.0)
                ema5 = self._f(inds.get("ema_5"))
                ema21 = self._f(inds.get("ema_21"))

                # AI says BUY, but is trend actually bullish?
                trend_confirms_buy = (
                        momentum_pct > 0.3 and  # Momentum turned positive
                        (ema5 is None or ema21 is None or ema5 > ema21)  # EMAs bullish
                )

                if trend_confirms_buy:
                    reason = f"EXIT SHORT - AI reversal + trend confirmed (BUY: {confidences['ai_cpr']:.2f}, Mom: {momentum_pct:.2f}%)"
                    self.log(f"[AI-EXIT] {reason} | Distribution: {ai_distribution}", False)

                    if self._process_exit(reason, ltp):
                        cooldown_bars = 0 if abs(momentum_pct) > 0.5 else 1
                        self._set_cooldown(cooldown_bars)
                        self._save_state()
                        return
                else:
                    self.log(
                        f"[WARNING] [AI-EXIT-BLOCKED] AI says BUY but trend NOT bullish\n"
                        f"  Momentum: {momentum_pct:.2f}% (need > +0.3%)\n"
                        f"  EMA5: {ema5:.2f}, EMA21: {ema21:.2f}\n"
                        f"  Decision: HOLD position (AI may be early)",
                        False
                    )

        # LAYER 3: Multiple signal reversal
        if current_pos == "BUY":
            if sell_count >= 3 or (sell_count >= 2 and sell_score > 1.4):
                ai_part = f" + AI({confidences['ai_cpr']:.2f})" if ai_in_sell else ""
                reason = f"EXIT LONG - Strong reversal ({sell_count} SELL votes, score: {sell_score:.2f}){ai_part}"
                self.log(f"[MULTI-EXIT] {reason}", False)

                if self._process_exit(reason, ltp):
                    # Zero cooldown if strong momentum, else 1 bar
                    cooldown_bars = 0 if (abs(self._f(inds.get("momentum_pct"), 0)) > 0.5 and self._f(inds.get("adx"),
                                                                                                      0) > 25) else 1
                    self._set_cooldown(cooldown_bars)
                    #self._set_cooldown(1)
                    self._save_state()
                    return

        elif current_pos == "SELL":
            if buy_count >= 3 or (buy_count >= 2 and buy_score > 1.4):
                ai_part = f" + AI({confidences['ai_cpr']:.2f})" if ai_in_buy else ""
                reason = f"EXIT SHORT - Strong reversal ({buy_count} BUY votes, score: {buy_score:.2f}){ai_part}"
                self.log(f"[MULTI-EXIT] {reason}", False)

                if self._process_exit(reason, ltp):
                    # Zero cooldown if strong momentum, else 1 bar
                    cooldown_bars = 0 if (abs(self._f(inds.get("momentum_pct"), 0)) > 0.5 and self._f(inds.get("adx"),
                                                                                                      0) > 25) else 1
                    self._set_cooldown(cooldown_bars)
                    #self._set_cooldown(1)
                    self._save_state()
                    return

        # ==========================================
        # MAINTAIN EXISTING POSITION
        # ==========================================
        if current_pos != "FLAT":
            if self._check_intelligent_exit(ltp, inds):
                self._save_state()
                return

        # ==========================================
        # EXIT PREDICTION (for open positions)
        # ==========================================
        if current_pos != "FLAT":
            try:
                # Get OHLC for exit prediction
                ohlc_for_exit = None
                try:
                    ohlc_for_exit = self.bot.fetch_ohlc(self.symbol, "5", 2)
                except Exception as e:
                    self.log(f"[EXIT-PRED] Failed to fetch OHLC: {e}", True)

                if ohlc_for_exit is not None and not ohlc_for_exit.empty:
                    # Run exit prediction model
                    exit_action, exit_conf, exit_reason = self._predict_exit_timing(
                        ltp, inds, ohlc_for_exit
                    )

                    # Store prediction for tracking/debugging
                    self.position["_last_exit_prediction"] = {
                        "action": exit_action,
                        "confidence": exit_conf,
                        "reason": exit_reason,
                        "timestamp": dt.datetime.now(self.IST).isoformat()
                    }

                    # =========================================
                    # ACT ON PREDICTION
                    # =========================================

                    # Immediate exit on strong reversal
                    if exit_action == "EXIT_NOW" and exit_conf >= 0.70:
                        # Calculate current profit
                        entry = self._f(self.position.get("entry_price"))
                        if entry and ltp:
                            if current_pos == "BUY":
                                profit_points = ltp - entry
                            else:
                                profit_points = entry - ltp

                            profit_rupees = profit_points * 250

                            self.log(
                                f"[ALERT] [EXIT-PRED] Immediate exit signal\n"
                                f"  Action: {exit_action}\n"
                                f"  Confidence: {exit_conf:.2f}\n"
                                f"  Reason: {exit_reason}\n"
                                f"  Current profit: ₹{profit_rupees:.0f}\n"
                                f"  Decision: EXIT NOW",
                                False
                            )

                            # Only exit if not a big loss
                            if profit_rupees > -250:  # Not losing more than ₹250
                                if self._process_exit(f"Exit Prediction: {exit_reason}", ltp):
                                    self._save_state()
                                    return
                            else:
                                self.log(
                                    f"[WARNING] [EXIT-PRED] Exit signal ignored - currently losing ₹{abs(profit_rupees):.0f}",
                                    False
                                )

                    # Tighten stops on exhaustion warning
                    elif exit_action == "EXIT_SOON" and exit_conf >= 0.60:
                        self.log(
                            f"[WARNING] [EXIT-PRED] Exhaustion warning\n"
                            f"  Confidence: {exit_conf:.2f}\n"
                            f"  Reason: {exit_reason}\n"
                            f"  Action: Tightening trailing stops",
                            False
                        )

                        # Tighten Layer 2 & 3 trailing distances
                        atr_here = self._get_atr_with_fallback(inds, ltp)
                        if atr_here:
                            current_sl = self._f(self.position.get("stop_loss"))
                            entry = self._f(self.position.get("entry_price"))

                            if current_sl and entry:
                                # Use 0.5 ATR instead of normal trailing
                                tight_distance = 0.5 * atr_here  # Tighter than usual

                                if current_pos == "BUY":
                                    new_sl = max(current_sl, ltp - tight_distance)
                                else:
                                    new_sl = min(current_sl, ltp + tight_distance)

                                # Only update if moving SL in favorable direction
                                should_update = False
                                if current_pos == "BUY" and new_sl > current_sl:
                                    should_update = True
                                elif current_pos == "SELL" and new_sl < current_sl:
                                    should_update = True

                                if should_update:
                                    self.position["stop_loss"] = round(new_sl, 2)

                                    locked_profit = (new_sl - entry) if current_pos == "BUY" else (entry - new_sl)
                                    locked_rupees = locked_profit * 250

                                    self.log(
                                        f"[SIGNAL] [TIGHT-SL] Stop loss tightened (exhaustion warning)\n"
                                        f"  Old SL: ₹{current_sl:.2f}\n"
                                        f"  New SL: ₹{new_sl:.2f}\n"
                                        f"  Trail: {tight_distance:.2f} pts (0.5 ATR)\n"
                                        f"  Locked: ₹{locked_rupees:.0f}",
                                        False
                                    )
                                    self._save_state()

            except Exception as e:
                self.log(f"[EXIT-PRED] Error: {e}", True)
                import traceback
                self.log(f"[EXIT-PRED] Traceback: {traceback.format_exc()}", True)

        self._save_state()

    def detect_rejection_candle(self, ohlc_df, ltp, pivot_data):
        """
        Detects strong rejection candles at key levels
        Returns: ("BUY_REJECTION", confidence) or ("SELL_REJECTION", confidence) or (None, 0)
        """
        if ohlc_df is None or len(ohlc_df) < 3:
            return None, 0.0

        try:
            latest = ohlc_df.iloc[-1]
            prev = ohlc_df.iloc[-2]

            # Get candle properties
            open_price = float(latest['Open'])
            high = float(latest['High'])
            low = float(latest['Low'])
            close = float(latest['Close'])

            body = abs(close - open_price)
            total_range = high - low
            upper_wick = high - max(open_price, close)
            lower_wick = min(open_price, close) - low

            # Volume check
            volume_ratio = 1.0
            if 'Volume' in latest and 'Volume' in ohlc_df.columns:
                avg_volume = ohlc_df['Volume'].rolling(10).mean().iloc[-1]
                if avg_volume > 0:
                    volume_ratio = float(latest['Volume']) / avg_volume

            # Get pivot levels
            tc = pivot_data.get("TC")
            bc = pivot_data.get("BC")
            r1 = pivot_data.get("R1")
            r2 = pivot_data.get("R2")
            s1 = pivot_data.get("S1")
            s2 = pivot_data.get("S2")

            # ==================================================
            # BEARISH REJECTION (Your 17:15 big red candle case)
            # ==================================================

            # Criteria:
            # 1. Large upper wick (wick > 2x body)
            # 2. Close near low (bearish)
            # 3. At resistance level (R1/R2/TC)
            # 4. High volume (1.5x+ average)

            is_bearish_rejection = False
            rejection_confidence = 0.0

            # Check structure
            if upper_wick > body * 2 and close < (low + total_range * 0.3):
                is_bearish_rejection = True
                rejection_confidence += 0.3

                # Check if at resistance
                if r2 and abs(high - r2) / r2 < 0.002:  # Tested R2
                    rejection_confidence += 0.3
                elif r1 and abs(high - r1) / r1 < 0.002:  # Tested R1
                    rejection_confidence += 0.25
                elif tc and abs(high - tc) / tc < 0.002:  # Tested TC
                    rejection_confidence += 0.2

                # Volume confirmation
                if volume_ratio >= 1.5:
                    rejection_confidence += 0.2
                elif volume_ratio >= 1.2:
                    rejection_confidence += 0.1

                # Price action confirmation (failed to hold above prev high)
                if close < prev['High'] * 0.998:
                    rejection_confidence += 0.15

                if rejection_confidence >= 0.6:  # Need 60%+ confidence
                    self.log(
                        f"🔴 [REJECTION] Bearish rejection detected!\n"
                        f"  High: {high:.2f} (tested resistance)\n"
                        f"  Close: {close:.2f} (rejected)\n"
                        f"  Upper wick: {upper_wick:.2f} ({upper_wick / body:.1f}x body)\n"
                        f"  Volume: {volume_ratio:.2f}x avg\n"
                        f"  Confidence: {rejection_confidence:.2f}",
                        False
                    )
                    return "SELL_REJECTION", rejection_confidence

            # ==================================================
            # BULLISH REJECTION (Support bounce)
            # ==================================================

            # Criteria:
            # 1. Large lower wick (wick > 2x body)
            # 2. Close near high (bullish)
            # 3. At support level (S1/S2/BC)
            # 4. High volume (1.5x+ average)

            is_bullish_rejection = False
            rejection_confidence = 0.0

            # Check structure
            if lower_wick > body * 2 and close > (high - total_range * 0.3):
                is_bullish_rejection = True
                rejection_confidence += 0.3

                # Check if at support
                if s2 and abs(low - s2) / s2 < 0.002:  # Tested S2
                    rejection_confidence += 0.3
                elif s1 and abs(low - s1) / s1 < 0.002:  # Tested S1
                    rejection_confidence += 0.25
                elif bc and abs(low - bc) / bc < 0.002:  # Tested BC
                    rejection_confidence += 0.2

                # Volume confirmation
                if volume_ratio >= 1.5:
                    rejection_confidence += 0.2
                elif volume_ratio >= 1.2:
                    rejection_confidence += 0.1

                # Price action confirmation (held above prev low)
                if close > prev['Low'] * 1.002:
                    rejection_confidence += 0.15

                if rejection_confidence >= 0.6:
                    self.log(
                        f"🟢 [REJECTION] Bullish rejection detected!\n"
                        f"  Low: {low:.2f} (tested support)\n"
                        f"  Close: {close:.2f} (bounced)\n"
                        f"  Lower wick: {lower_wick:.2f} ({lower_wick / body:.1f}x body)\n"
                        f"  Volume: {volume_ratio:.2f}x avg\n"
                        f"  Confidence: {rejection_confidence:.2f}",
                        False
                    )
                    return "BUY_REJECTION", rejection_confidence

            return None, 0.0

        except Exception as e:
            self.log(f"[REJECTION] Error detecting rejection candle: {e}", True)
            return None, 0.0

    def detect_volume_breakout(self, ohlc_df, ltp):
        """
        Detects early breakout attempts with volume surge
        Returns: ("VOLUME_BREAKOUT_BUY", confidence) or ("VOLUME_BREAKOUT_SELL", confidence) or (None, 0)
        """
        if ohlc_df is None or len(ohlc_df) < 10:
            return None, 0.0

        try:
            latest = ohlc_df.iloc[-1]

            # Calculate volume ratio
            avg_volume = ohlc_df['Volume'].rolling(10).mean().iloc[-1]
            if avg_volume <= 0:
                return None, 0.0

            volume_ratio = float(latest['Volume']) / avg_volume

            # Need significant volume surge (2x+)
            if volume_ratio < 2.0:
                return None, 0.0

            # Check price movement
            open_price = float(latest['Open'])
            close = float(latest['Close'])
            high = float(latest['High'])
            low = float(latest['Low'])

            body = abs(close - open_price)
            total_range = high - low

            # Bullish volume breakout
            if close > open_price and body / total_range > 0.6:  # Strong bullish body
                confidence = min(volume_ratio / 3.0, 0.9)  # Cap at 0.9

                self.log(
                    f"[START] [VOLUME] Bullish breakout attempt!\n"
                    f"  Volume: {volume_ratio:.2f}x average\n"
                    f"  Body: {body / total_range * 100:.1f}% of range\n"
                    f"  Confidence: {confidence:.2f}",
                    False
                )
                return "VOLUME_BREAKOUT_BUY", confidence

            # Bearish volume breakout
            elif close < open_price and body / total_range > 0.6:
                confidence = min(volume_ratio / 3.0, 0.9)

                self.log(
                    f"[START] [VOLUME] Bearish breakout attempt!\n"
                    f"  Volume: {volume_ratio:.2f}x average\n"
                    f"  Body: {body / total_range * 100:.1f}% of range\n"
                    f"  Confidence: {confidence:.2f}",
                    False
                )
                return "VOLUME_BREAKOUT_SELL", confidence

            return None, 0.0

        except Exception as e:
            self.log(f"[VOLUME] Error detecting volume breakout: {e}", True)
            return None, 0.0

    def detect_momentum_shift(self, ohlc_df):
        """
        Detects early momentum shifts (3 consecutive candles changing direction)
        Returns: ("MOMENTUM_SHIFT_BUY", confidence) or ("MOMENTUM_SHIFT_SELL", confidence) or (None, 0)
        """
        if ohlc_df is None or len(ohlc_df) < 5:
            return None, 0.0

        try:
            # Get last 5 candles
            recent = ohlc_df.iloc[-5:].copy()

            # Check for bearish to bullish shift
            bearish_count = 0
            bullish_count = 0

            for i in range(len(recent) - 3):
                if recent['Close'].iloc[i] < recent['Open'].iloc[i]:
                    bearish_count += 1

            for i in range(len(recent) - 3, len(recent)):
                if recent['Close'].iloc[i] > recent['Open'].iloc[i]:
                    bullish_count += 1

            # Shift from bearish to bullish (last 3 green after 2+ red)
            if bearish_count >= 2 and bullish_count == 3:
                confidence = 0.65
                self.log(
                    f"📈 [MOMENTUM] Shift to bullish detected!\n"
                    f"  Previous: {bearish_count} bearish candles\n"
                    f"  Current: 3 consecutive bullish\n"
                    f"  Confidence: {confidence:.2f}",
                    False
                )
                return "MOMENTUM_SHIFT_BUY", confidence

            # Shift from bullish to bearish (last 3 red after 2+ green)
            bearish_count = 0
            bullish_count = 0

            for i in range(len(recent) - 3):
                if recent['Close'].iloc[i] > recent['Open'].iloc[i]:
                    bullish_count += 1

            for i in range(len(recent) - 3, len(recent)):
                if recent['Close'].iloc[i] < recent['Open'].iloc[i]:
                    bearish_count += 1

            if bullish_count >= 2 and bearish_count == 3:
                confidence = 0.65
                self.log(
                    f"📉 [MOMENTUM] Shift to bearish detected!\n"
                    f"  Previous: {bullish_count} bullish candles\n"
                    f"  Current: 3 consecutive bearish\n"
                    f"  Confidence: {confidence:.2f}",
                    False
                )
                return "MOMENTUM_SHIFT_SELL", confidence

            return None, 0.0

        except Exception as e:
            self.log(f"[MOMENTUM] Error detecting momentum shift: {e}", True)
            return None, 0.0

    def detect_trend(self, df, i=-1):
        """
        Multi-timeframe trend detection optimized for 5-min intraday trading
        Uses EMA(5,9) for primary signals, EMA(5,21) for trend filtering
        """
        try:
            # ==========================================
            # VALIDATION
            # ==========================================
            if df is None:
                self.log("[TREND] No data available", True)
                return "Consolidation"

            if len(df) < 10:
                self.log(f"[TREND] Insufficient data ({len(df)} bars), need 10+", True)
                return "Consolidation"

            df = df.copy()
            df.columns = [c.lower() for c in df.columns]

            if 'close' not in df:
                self.log("[TREND] Missing 'close' column", True)
                return "Consolidation"

            # ==========================================
            # PRIMARY SIGNAL: EMA(5,9) - Fast Response
            # ==========================================
            ema5 = df['close'].ewm(span=5, adjust=False).mean()
            ema9 = df['close'].ewm(span=9, adjust=False).mean()

            last_5 = float(ema5.iloc[i])
            last_9 = float(ema9.iloc[i])

            ema_spread_59 = abs(last_5 - last_9) / last_9 * 100 if last_9 > 0 else 0

            # Determine short-term signal (0.15% threshold for 5-min data)
            short_signal = None
            if last_5 > last_9 * 1.0015:  # 0.15% gap
                short_signal = "Uptrend"
            elif last_5 < last_9 * 0.9985:
                short_signal = "Downtrend"
            else:
                short_signal = "Neutral"

            # ==========================================
            # FILTER: EMA(5,21) - Trend Strength
            # ==========================================
            long_filter = "Neutral"
            ema_spread_521 = 0

            if len(df) >= 21:
                ema21 = df['close'].ewm(span=21, adjust=False).mean()
                last_21 = float(ema21.iloc[i])

                ema_spread_521 = abs(last_5 - last_21) / last_21 * 100 if last_21 > 0 else 0

                # Determine long-term filter (0.3% threshold - relaxed)
                if last_5 > last_21 * 1.003:  # 0.3% gap
                    long_filter = "Uptrend"
                elif last_5 < last_21 * 0.997:
                    long_filter = "Downtrend"
                else:
                    long_filter = "Neutral"

                self.log(
                    f"[TREND-SIGNAL] EMA5/9: {short_signal} ({ema_spread_59:.3f}%) | "
                    f"EMA5/21: {long_filter} ({ema_spread_521:.3f}%)",
                    True
                )
            else:
                self.log(
                    f"[TREND-SIGNAL] EMA5/9: {short_signal} ({ema_spread_59:.3f}%) | "
                    f"EMA5/21: N/A (need 21+ bars)",
                    True
                )

            # ==========================================
            # ADDITIONAL CONFIRMATION: Momentum
            # ==========================================
            # Use last 5 candles for momentum check
            lookback = min(5, len(df))
            closes = df['close'].values[-lookback:]
            slope = (closes[-1] - closes[0]) / len(closes) if len(closes) > 1 else 0
            slope_pct = (slope / closes.mean()) * 100 if closes.mean() > 0 else 0

            # ==========================================
            # DECISION MATRIX
            # ==========================================
            result = None

            # CASE 1: Short-term UPTREND
            if short_signal == "Uptrend":
                if long_filter in ["Uptrend", "Neutral"]:
                    # [OK] Aligned or no conflict
                    result = "Uptrend"
                    self.log(
                        f"[TREND] [OK] UPTREND confirmed\n"
                        f"  EMA5: {last_5:.2f} > EMA9: {last_9:.2f} (gap: {ema_spread_59:.3f}%)\n"
                        f"  EMA5/21 filter: {long_filter} ({ema_spread_521:.3f}%)\n"
                        f"  Slope: {slope_pct:.3f}%",
                        False
                    )
                else:
                    # [WARNING] Counter-trend: EMA5>9 but EMA5<21 (short-term up, long-term down)
                    # Skip this - likely a pullback in downtrend
                    result = "Consolidation"
                    self.log(
                        f"[TREND] [WARNING] Conflicting signals - SKIP\n"
                        f"  Short: EMA5 > EMA9 (bullish)\n"
                        f"  Long: EMA5 < EMA21 (bearish)\n"
                        f"  Decision: Avoid counter-trend trade",
                        False
                    )

            # CASE 2: Short-term DOWNTREND
            elif short_signal == "Downtrend":
                if long_filter in ["Downtrend", "Neutral"]:
                    # [OK] Aligned or no conflict
                    result = "Downtrend"
                    self.log(
                        f"[TREND] [OK] DOWNTREND confirmed\n"
                        f"  EMA5: {last_5:.2f} < EMA9: {last_9:.2f} (gap: {ema_spread_59:.3f}%)\n"
                        f"  EMA5/21 filter: {long_filter} ({ema_spread_521:.3f}%)\n"
                        f"  Slope: {slope_pct:.3f}%",
                        False
                    )
                else:
                    # [WARNING] Counter-trend: EMA5<9 but EMA5>21 (short-term down, long-term up)
                    result = "Consolidation"
                    self.log(
                        f"[TREND] [WARNING] Conflicting signals - SKIP\n"
                        f"  Short: EMA5 < EMA9 (bearish)\n"
                        f"  Long: EMA5 > EMA21 (bullish)\n"
                        f"  Decision: Avoid counter-trend trade",
                        False
                    )

            # CASE 3: Short-term NEUTRAL (EMAs converged)
            else:
                # Check momentum to break the tie
                if abs(slope_pct) > 0.08:  # 0.08% movement threshold
                    result = "Uptrend" if slope_pct > 0 else "Downtrend"
                    self.log(
                        f"[TREND] {result} via MOMENTUM\n"
                        f"  EMAs flat (5/9 gap: {ema_spread_59:.3f}%)\n"
                        f"  But slope shows {slope_pct:.3f}% movement\n"
                        f"  Decision: Follow momentum",
                        False
                    )
                else:
                    # Truly flat market
                    result = "Consolidation"
                    self.log(
                        f"[TREND] 🔒 CONSOLIDATION\n"
                        f"  EMA5/9 gap: {ema_spread_59:.3f}% (< 0.15% threshold)\n"
                        f"  Slope: {slope_pct:.3f}% (< 0.08% threshold)\n"
                        f"  Decision: No clear direction",
                        False
                    )

            # ==========================================
            # VOLUME CONFIRMATION (Optional Enhancement)
            # ==========================================
            if result in ["Uptrend", "Downtrend"] and 'volume' in df.columns:
                try:
                    recent_vol = df['volume'].values[-5:]
                    avg_vol = recent_vol[:-1].mean() if len(recent_vol) > 1 else 0
                    current_vol = recent_vol[-1]

                    if avg_vol > 0:
                        vol_ratio = current_vol / avg_vol

                        if vol_ratio < 0.5:  # Very low volume
                            self.log(
                                f"[TREND] [WARNING] Low volume warning: {vol_ratio:.2f}x average\n"
                                f"  Signal may be weak",
                                True
                            )
                        elif vol_ratio > 1.5:  # High volume
                            self.log(
                                f"[TREND] [OK] Volume confirmation: {vol_ratio:.2f}x average",
                                True
                            )
                except Exception as e:
                    self.log(f"[TREND] Volume check error: {e}", True)

            return result

        except Exception as e:
            self.log(f"[TREND] [ERROR] Critical error: {e}", False)
            import traceback
            self.log(f"[TREND] Traceback:\n{traceback.format_exc()}", True)

            # ==========================================
            # EMERGENCY FALLBACK
            # ==========================================
            try:
                if df is not None and 'close' in df.columns and len(df) >= 5:
                    closes = df['close'].values[-5:]
                    slope = (closes[-1] - closes[0]) / len(closes)
                    slope_pct = (slope / closes.mean()) * 100

                    if slope_pct > 0.1:
                        self.log("[TREND] Emergency fallback: Uptrend (slope > 0.1%)", False)
                        return "Uptrend"
                    elif slope_pct < -0.1:
                        self.log("[TREND] Emergency fallback: Downtrend (slope < -0.1%)", False)
                        return "Downtrend"
            except:
                pass

            self.log("[TREND] Emergency fallback: Consolidation", False)
            return "Consolidation"

    # --- Helper: Cooldown management ---
    def _cooldown_active(self):
        """
        Returns True if cooldown is active (still waiting before next trade)
        """
        try:
            cd = int(self.position.get("_cooldown_bars", 0) or 0)
        except Exception:
            cd = 0

        if cd > 0:
            # reduce cooldown on each cycle
            self.position["_cooldown_bars"] = cd - 1
            self.log(f"[COOLDOWN] Waiting for {cd - 1} more bars before new trade", True)
            return True
        return False

    def _set_cooldown(self, bars: int = 1):
        """
        Activates cooldown after a trade to prevent overtrading
        """
        self.position["_cooldown_bars"] = bars
        self.log(f"[COOLDOWN] Set for {bars} bars", True)

    def _ai_place_order(self, ai_signal, reason="AI-CPR Signal"):
        """
        Safely execute AI-based orders with position awareness and cooldown checks.
        Handles both dict and string AI signals, and ensures correct order direction.
        """
        try:
            self.log("DEBUG: Entered Order Manager _ai_place_order", False)

            current_side = self.position.get("type")
            cooldown_active = self._cooldown_active()

            # --- Normalize AI Signal (handles dict or string) ---
            if isinstance(ai_signal, dict):
                ai_signal = ai_signal.get("action")

            if not ai_signal:
                self.log("[INFO] Empty AI signal — skipping order", True)
                return

            ai_signal = str(ai_signal).capitalize().strip()
            if ai_signal not in ("Bullish", "Bearish"):
                self.log(f"[INFO] Invalid AI signal ({ai_signal}) — skipping order", True)
                return

            desired_side = "BUY" if ai_signal == "Bullish" else "SELL"

            # --- Ensure we have a usable 'inds' in scope (fix for unresolved reference) ---
            inds = {}
            try:
                if getattr(self, "last_known_inds", None):
                    # If we have a primary timeframe saved, normalize to that TF
                    if getattr(self, "last_known_primary_tf", None):
                        inds = self._norm_tf(self.last_known_inds, self.last_known_primary_tf) or self.last_known_inds
                    else:
                        inds = self.last_known_inds
            except Exception:
                inds = self.last_known_inds or {}

            # Provide safe numeric defaults
            momentum_pct_val = abs(self._f(inds.get("momentum_pct"), 0))
            adx_val = self._f(inds.get("adx"), 0)

            # --- Cooldown enforcement ---
            if cooldown_active:
                self.log(f"[COOLDOWN] Skipping {desired_side} due to cooldown", True)
                return

            # --- Avoid duplicate entries ---
            if current_side == desired_side:
                self.log(f"[HOLD] Already in {current_side}, skipping re-entry.", True)
                return

            # --- Handle reversal (BUY → SELL or SELL → BUY) ---
            if current_side and current_side != desired_side:
                self.log(f"[REVERSAL] Closing {current_side} → Opening {desired_side} ({reason})", True)
                try:
                    self.bot.place_order(self.symbol, side="EXIT", qty=self.lot, reason="Reversal Exit")
                except Exception as e:
                    self.log(f"[ERROR] Failed to exit {current_side}: {e}", True)
                    return

                # Place new order
                try:
                    self.bot.place_order(self.symbol, side=desired_side, qty=self.lot, reason=reason)
                    self.position["side"] = desired_side
                    #self._set_cooldown(1)
                    # Zero cooldown if strong momentum, else 1 bar
                    cooldown_bars = 0 if (momentum_pct_val > 0.5 and adx_val > 25) else 1
                    self._set_cooldown(cooldown_bars)
                except Exception as e:
                    self.log(f"[ERROR] Failed to place {desired_side} order: {e}", True)
                return

            # --- New entry if flat (no current position) ---
            if not current_side:
                self.log(f"[ENTRY] Placing {desired_side} order for {self.symbol} ({reason})", True)
                self.bot.place_order(self.symbol, side=desired_side, qty=self.lot, reason=reason)
                self.position["side"] = desired_side
                # Zero cooldown if strong momentum, else 1 bar
                cooldown_bars = 0 if (momentum_pct_val > 0.5 and adx_val > 25) else 1
                self._set_cooldown(cooldown_bars)
                #self._set_cooldown(1)
                return

        except Exception as e:
            self.log(f"[ERROR] _ai_place_order() failed: {e}", True)

    # ---------- AI CPR STATE MANAGEMENT ----------
    def get_ai_state(self):
        """Get current AI CPR state for monitoring"""
        return {
            "last_ai_action": self.last_ai_action,
            "last_ai_confidence": self.last_ai_confidence,
            "ai_enabled": self.AI_CPR_ENABLED,
            "ai_gate_trades": self.AI_GATE_TRADES,
            "ai_min_confidence": self.AI_MIN_CONF
        }

    def update_ai_config(self, enabled=None, min_conf=None, gate_trades=None):
        """Update AI CPR configuration dynamically"""
        if enabled is not None:
            self.AI_CPR_ENABLED = enabled
            self.log(f"[AI-CPR] Strategy {'enabled' if enabled else 'disabled'}", False)

        if min_conf is not None:
            self.AI_MIN_CONF = min_conf
            self.log(f"[AI-CPR] Minimum confidence set to {min_conf}", False)

        if gate_trades is not None:
            self.AI_GATE_TRADES = gate_trades
            self.log(f"[AI-CPR] Trade gating {'enabled' if gate_trades else 'disabled'}", False)

    # ---------- CSV LOGGING METHODS ----------
    def _ensure_trade_csv(self):
        """Ensure trade CSV has AI-specific columns"""
        if not self.trades_csv:
            return

        if not os.path.exists(self.trades_csv):
            fields = [
                "trade_id", "symbol", "side", "event",
                "entry_time", "exit_time", "hold_seconds",
                "entry_ltp", "exit_ltp", "ltp_diff",
                "reason", "order_id", "bar_key",
                "adx", "macd_color", "ema5", "ema9", "ema21",
                "ai_confidence", "ai_action"
            ]
            with open(self.trades_csv, "w", newline="") as f:
                w = DictWriter(f, fieldnames=fields)
                w.writeheader()

    def _append_trade_csv(self, row: dict):
        """Append trade with AI data"""
        if not self.trades_csv:
            return

        fields = [
            "trade_id", "symbol", "side", "event",
            "entry_time", "exit_time", "hold_seconds",
            "entry_ltp", "exit_ltp", "ltp_diff",
            "reason", "order_id", "bar_key",
            "adx", "macd_color", "ema5", "ema9", "ema21",
            "ai_confidence", "ai_action"
        ]

        # Ensure all fields are present
        for k in fields:
            row.setdefault(k, "")

        with open(self.trades_csv, "a", newline="") as f:
            w = DictWriter(f, fieldnames=fields)
            w.writerow(row)

    # ---------- DASHBOARD EXPORT ----------
    def export_normalized_dashboard(self, all_inds, out_path, tfs=("1", "5", "15", "30")):
        dash = {}
        for tf in tfs:
            d = self._norm_tf(all_inds, tf)
            if d and any(k in d for k in ("ema_5", "ema_9", "ema_21", "close", "ATR", "atr")):
                dash[str(tf)] = {"inds": d, "ts": d.get("timestamp")}
        if not dash:
            self.log("[EXPORT] No TFs normalized; likely a raw list or wrong shape was passed.", False)
            return False
        robust_save_json({"Dashboard": dash}, out_path, self.log)
        self.log(f"[EXPORT] Wrote {out_path} with TFs: {', '.join(sorted(dash.keys()))}", False)
        return True


class PerformanceTracker:
    """Simple dashboard generator from CSV files"""

    def __init__(self, bot, dashboard_path="reports_bot/dashboard.html"):
        self.bot = bot
        self.dashboard_path = dashboard_path
        os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)

    def generate_html_dashboard(self):
        """Generate dashboard by reading CSV files"""
        try:
            all_stats = {}
            all_trades = []
            open_positions = {}

            # Read CSV files for each symbol
            for sym in self.bot.symbols:
                csv_path = self.bot.order_managers[sym].trades_csv

                if not os.path.exists(csv_path):
                    continue

                # Read CSV
                df = pd.read_csv(csv_path)

                if df.empty:
                    continue

                # Get completed trades (EXIT events)
                exits = df[df['event'] == 'EXIT'].copy()

                if not exits.empty:
                    # Calculate statistics
                    total_pnl = exits['ltp_diff'].fillna(0).sum()
                    win_trades = exits[exits['ltp_diff'] > 0]
                    loss_trades = exits[exits['ltp_diff'] < 0]

                    all_stats[sym] = {
                        'total_pnl': round(total_pnl, 2),
                        'total_trades': len(exits),
                        'win_rate': round(len(win_trades) / len(exits) * 100, 1) if len(exits) > 0 else 0,
                        'avg_win': round(win_trades['ltp_diff'].mean(), 2) if not win_trades.empty else 0,
                        'avg_loss': round(loss_trades['ltp_diff'].mean(), 2) if not loss_trades.empty else 0,
                        'largest_win': round(exits['ltp_diff'].max(), 2) if not exits.empty else 0
                    }

                    # Get last 10 trades
                    for _, row in exits.tail(10).iterrows():
                        all_trades.append({
                            'symbol': sym,
                            'side': row.get('side', 'N/A'),
                            'entry_time': row.get('entry_time', 'N/A'),
                            'exit_time': row.get('exit_time', 'N/A'),
                            'entry_ltp': row.get('entry_ltp', 0),
                            'exit_ltp': row.get('exit_ltp', 0),
                            'pnl': row.get('ltp_diff', 0),
                            'reason': row.get('reason', '')[:50]
                        })

                # Check for open positions
                om = self.bot.order_managers[sym]
                if om.position.get("type") != "FLAT":
                    ltp = self.bot.get_websocket_ltp(sym)
                    entry = om.position.get("entry_price", 0)

                    if ltp and entry:
                        side = om.position.get("type")
                        pnl = (ltp - entry) if side == "BUY" else (entry - ltp)

                        open_positions[sym] = {
                            'side': side,
                            'entry': entry,
                            'current': ltp,
                            'pnl': round(pnl, 2),
                            'pnl_pct': round((pnl / entry * 100), 2) if entry > 0 else 0,
                            'sl': om.position.get("stop_loss", 0)
                        }

            # Generate HTML
            self._write_html(all_stats, all_trades, open_positions)

            self.bot.log_message(f"[DASHBOARD] [OK] Updated: {self.dashboard_path}", True)

        except Exception as e:
            self.bot.log_message(f"[DASHBOARD] [ERROR] Error: {e}", False)
            import traceback
            self.bot.log_message(f"[DASHBOARD] {traceback.format_exc()}", True)

    def _write_html(self, stats, trades, positions):
        """Write HTML file"""
        now = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        # Calculate totals
        total_pnl = sum(s['total_pnl'] for s in stats.values())
        total_trades = sum(s['total_trades'] for s in stats.values())
        avg_win_rate = sum(s['win_rate'] for s in stats.values()) / len(stats) if stats else 0

        # Generate open positions HTML
        pos_html = self._generate_positions_html(positions)

        # Generate trades table HTML
        trades_html = self._generate_trades_html(trades)

        # Generate stats cards HTML
        stats_html = self._generate_stats_html(stats, total_pnl, total_trades, avg_win_rate)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="10">
    <title>Trading Bot Dashboard - Live</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{ max-width: 1400px; margin: 0 auto; }}

        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
            text-align: center;
        }}

        .header h1 {{ color: #2d3748; font-size: 2.5em; margin-bottom: 10px; }}
        .header .timestamp {{ color: #718096; font-size: 0.9em; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
            transition: transform 0.3s;
        }}

        .stat-card:hover {{ transform: translateY(-5px); }}

        .stat-card .label {{
            color: #718096;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #2d3748;
        }}

        .stat-card.positive .value {{ color: #48bb78; }}
        .stat-card.negative .value {{ color: #f56565; }}

        .card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }}

        .card h2 {{ color: #2d3748; margin-bottom: 20px; font-size: 1.5em; }}

        table {{ width: 100%; border-collapse: collapse; }}

        th {{
            background: #4a5568;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
        }}

        td {{ padding: 15px; border-bottom: 1px solid #e2e8f0; }}

        tr:hover {{ background: #f7fafc; }}

        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-buy {{ background: #c6f6d5; color: #22543d; }}
        .badge-sell {{ background: #fed7d7; color: #742a2a; }}

        .profit {{ color: #48bb78; font-weight: bold; }}
        .loss {{ color: #f56565; font-weight: bold; }}

        .position-row {{
            margin-bottom: 15px;
            padding: 15px;
            background: #f7fafc;
            border-radius: 10px;
        }}

        .no-data {{
            text-align: center;
            color: #718096;
            padding: 40px;
            font-size: 1.1em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Trading Bot Dashboard</h1>
            <div class="timestamp">Last Updated: {now}</div>
        </div>

        {stats_html}
        {pos_html}
        {trades_html}
    </div>
</body>
</html>
        """

        with open(self.dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html)

    def _generate_stats_html(self, stats, total_pnl, total_trades, avg_win_rate):
        """Generate statistics cards HTML"""
        pnl_class = 'positive' if total_pnl > 0 else 'negative' if total_pnl < 0 else ''
        wr_class = 'positive' if avg_win_rate >= 60 else 'negative' if avg_win_rate < 50 else ''

        return f"""
        <div class="stats-grid">
            <div class="stat-card {pnl_class}">
                <div class="label">Total P&L</div>
                <div class="value">₹{total_pnl:.2f}</div>
            </div>

            <div class="stat-card">
                <div class="label">Total Trades</div>
                <div class="value">{total_trades}</div>
            </div>

            <div class="stat-card {wr_class}">
                <div class="label">Avg Win Rate</div>
                <div class="value">{avg_win_rate:.1f}%</div>
            </div>

            <div class="stat-card">
                <div class="label">Active Symbols</div>
                <div class="value">{len(stats)}</div>
            </div>
        </div>
        """

    def _generate_positions_html(self, positions):
        """Generate open positions HTML"""
        if not positions:
            return """
            <div class="card">
                <h2>[POS] Open Positions</h2>
                <div class="no-data">No active positions</div>
            </div>
            """

        rows = ""
        for sym, pos in positions.items():
            pnl_color = "profit" if pos['pnl'] > 0 else "loss"
            side_badge = "badge-buy" if pos['side'] == 'BUY' else "badge-sell"

            rows += f"""
            <div class="position-row">
                <table>
                    <tr>
                        <td><strong>Symbol:</strong></td>
                        <td>{sym}</td>
                        <td><strong>Side:</strong></td>
                        <td><span class="badge {side_badge}">{pos['side']}</span></td>
                    </tr>
                    <tr>
                        <td><strong>Entry:</strong></td>
                        <td>₹{pos['entry']:.2f}</td>
                        <td><strong>Current:</strong></td>
                        <td>₹{pos['current']:.2f}</td>
                    </tr>
                    <tr>
                        <td><strong>Stop Loss:</strong></td>
                        <td>₹{pos['sl']:.2f}</td>
                        <td><strong>P&L:</strong></td>
                        <td class="{pnl_color}">₹{pos['pnl']:.2f} ({pos['pnl_pct']:.2f}%)</td>
                    </tr>
                </table>
            </div>
            """

        return f"""
        <div class="card">
            <h2>[POS] Open Positions ({len(positions)})</h2>
            {rows}
        </div>
        """

    def _generate_trades_html(self, trades):
        """Generate trades table HTML"""
        if not trades:
            return """
            <div class="card">
                <h2>[DATA] Recent Trades</h2>
                <div class="no-data">No trades recorded yet</div>
            </div>
            """

        # Sort by exit time (most recent first)
        trades_sorted = sorted(trades, key=lambda x: x.get('exit_time', ''), reverse=True)[:10]

        rows = ""
        for trade in trades_sorted:
            pnl = trade.get('pnl', 0)
            pnl_class = "profit" if pnl > 0 else "loss"
            side_badge = "badge-buy" if trade.get('side') == 'BUY' else "badge-sell"

            rows += f"""
            <tr>
                <td>{trade.get('symbol', 'N/A')}</td>
                <td><span class="badge {side_badge}">{trade.get('side', 'N/A')}</span></td>
                <td>₹{trade.get('entry_ltp', 0):.2f}</td>
                <td>₹{trade.get('exit_ltp', 0):.2f}</td>
                <td class="{pnl_class}">₹{pnl:.2f}</td>
                <td>{trade.get('exit_time', 'N/A')[:16]}</td>
                <td>{trade.get('reason', '')}</td>
            </tr>
            """

        return f"""
        <div class="card">
            <h2>[DATA] Recent Trades (Last 10)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>P&L</th>
                        <th>Exit Time</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """


import datetime as dt
import pandas as pd
from math import isfinite as _math_isfinite_unused


class UnifiedStopLossManager:
    """
    Single source of truth for all stop loss logic
    Runs before ANY other logic (entries, exits, trailing)
    """

    def __init__(self, order_manager):
        self.om = order_manager
        self.log = order_manager.log
        #self.position = order_manager.position
        self.symbol = order_manager.symbol

        # Natural Gas specific settings
        self.INITIAL_SL_ATR = 1.2
        self.EMERGENCY_SL_MULTIPLIER = 1.2
        self.MAX_LOSS_PERCENT = 0.015  # 1.5%
        self.BID_ASK_BUFFER = 0.10  # ₹25
        self.FLASH_CRASH_THRESHOLD = 300  # ₹300 in 2s

    def check_stop_loss(self, ltp, atr=None):
        """
        Check ALL stop loss conditions
        Returns: (should_exit, reason, loss_amount)
        """

        self.position = self.om.position
        # [OK] DEBUG
        self.log(f"[SL-CHECK] Starting check - LTP: {ltp}", True)

        # Check if position exists
        if not self.position or not isinstance(self.position, dict):
            self.log(f"[SL-CHECK] No position or invalid type: {type(self.position)}", True)
            return False, None, 0

        pos_type = self.position.get("type")
        if not pos_type or pos_type == "FLAT":
            self.log(f"[SL-CHECK] Position is FLAT", True)
            return False, None, 0

        side = pos_type
        entry = self._f(self.position.get("entry_price"))
        current_sl = self._f(self.position.get("stop_loss"))
        r_mult = self._f(self.position.get("r_mult", 0))

        self.log(
            f"[SL-CHECK] Position data:\n"
            f"  Side: {side}\n"
            f"  Entry: ₹{entry}\n"
            f"  Current SL: ₹{current_sl}\n"
            f"  R-mult: {r_mult}\n"
            f"  LTP: ₹{ltp}",
            True
        )

        if not all([entry, current_sl, ltp]):
            self.log(f"[SL-CHECK] [ERROR] Missing critical data", False)
            return False, None, 0

        # ========================================
        # CHECK 1: Regular Stop Loss
        # ========================================
        sl_with_buffer = self._apply_buffer(current_sl, side)

        self.log(
            f"[SL-CHECK-1] Regular SL:\n"
            f"  Original SL: ₹{current_sl:.2f}\n"
            f"  With buffer: ₹{sl_with_buffer:.2f}\n"
            f"  LTP: ₹{ltp:.2f}\n"
            f"  Will trigger: {self._is_sl_hit(ltp, sl_with_buffer, side)}",
            True
        )

        if self._is_sl_hit(ltp, sl_with_buffer, side):
            loss = self._calculate_loss(entry, ltp, side) * 250
            self.log(f"[SL-CHECK-1] [OK] TRIGGERED! Loss: ₹{loss:.2f}", False)
            return True, f"Regular SL Hit (₹{current_sl:.2f})", loss

        # ========================================
        # CHECK 2: Emergency Stop Loss
        # ========================================
        if r_mult > 0:
            if side == "BUY":
                current_pnl = (ltp - entry) * 250
            else:  # SELL
                current_pnl = (entry - ltp) * 250

            max_loss_allowed = r_mult * 250 * self.EMERGENCY_SL_MULTIPLIER

            # Trigger only if LOSING more than threshold
            if current_pnl < -max_loss_allowed:
                self.log(f"[SL-CHECK-2] [OK] TRIGGERED!", False)
                return True, f"Emergency SL (Loss > {self.EMERGENCY_SL_MULTIPLIER}x risk)", abs(current_pnl)

        # ========================================
        # CHECK 3: Max Loss %
        # ========================================
        loss_pct = abs((entry - ltp) / entry * 100) if side == "BUY" else abs((ltp - entry) / entry * 100)

        self.log(
            f"[SL-CHECK-3] Max Loss %:\n"
            f"  Current: {loss_pct:.2f}%\n"
            f"  Max allowed: {self.MAX_LOSS_PERCENT * 100}%\n"
            f"  Will trigger: {loss_pct > self.MAX_LOSS_PERCENT * 100}",
            True
        )

        if loss_pct > self.MAX_LOSS_PERCENT * 100:
            loss = self._calculate_loss(entry, ltp, side) * 250
            self.log(f"[SL-CHECK-3] [OK] TRIGGERED!", False)
            return True, f"Max Loss % Exceeded ({loss_pct:.2f}%)", loss

        # ========================================
        # CHECK 4: Flash Crash
        # ========================================
        flash_triggered, flash_loss = self._check_flash_crash(ltp, entry, side)

        if flash_triggered:
            self.log(f"[SL-CHECK-4] [OK] FLASH CRASH TRIGGERED!", False)
            return True, f"Flash Crash Protection (₹{flash_loss:.0f} in 2s)", flash_loss

        # All checks passed
        self.log(f"[SL-CHECK] [OK] All checks passed - position safe", True)
        return False, None, 0

    def _apply_buffer(self, sl, side):
        """Apply bid-ask buffer"""
        if side == "BUY":
            return sl - self.BID_ASK_BUFFER
        else:
            return sl + self.BID_ASK_BUFFER

    def _is_sl_hit(self, ltp, sl_with_buffer, side):
        """Check if SL triggered"""
        if side == "BUY":
            return ltp <= sl_with_buffer
        else:
            return ltp >= sl_with_buffer

    def _calculate_loss(self, entry, ltp, side):
        """Calculate loss in POINTS (multiply by 250 for rupees)"""
        if side == "BUY":
            profit = ltp - entry  # Positive = profit, Negative = loss
        else:  # SELL
            profit = entry - ltp  # [OK] CORRECT: Price drop = profit

        return -profit  # Return as LOSS (negative profit)

    def _check_flash_crash(self, ltp, entry, side):
        """Detect rapid movements"""
        try:
            import pytz
            IST = pytz.timezone("Asia/Kolkata")

            last_ltp = self.position.get("_last_checked_ltp")
            last_time = self.position.get("_last_check_time")

            if not last_ltp or not last_time:
                self.position["_last_checked_ltp"] = ltp
                self.position["_last_check_time"] = dt.datetime.now(IST)
                return False, 0

            if isinstance(last_time, str):
                last_time = pd.to_datetime(last_time).tz_localize(IST)

            current_time = dt.datetime.now(IST)
            time_diff = (current_time - last_time).total_seconds()

            if time_diff <= 2:
                if side == "BUY":
                    rapid_loss = (last_ltp - ltp) * 250
                else:
                    rapid_loss = (ltp - last_ltp) * 250

                if rapid_loss >= self.FLASH_CRASH_THRESHOLD:
                    self.log(
                        f"[FLASH-CRASH] Detected: ₹{rapid_loss:.0f} in {time_diff:.1f}s",
                        False
                    )
                    return True, rapid_loss

            self.position["_last_checked_ltp"] = ltp
            self.position["_last_check_time"] = current_time

        except Exception as e:
            self.log(f"[FLASH-CRASH] Error: {e}", True)

        return False, 0

    @staticmethod
    def _f(v, alt=None):
        try:
            if v is None: return alt
            x = float(v)
            return x if isfinite(x) else alt
        except:
            return alt


class UnifiedTrailingProfitManager:
    """Handles all trailing profit logic"""

    def __init__(self, order_manager):
        self.om = order_manager
        self.log = order_manager.log
        #self.position = order_manager.position

    def update_trailing_stops(self, ltp, atr, pivot_data=None):
        """
        Update trailing stops
        Returns: (should_exit, reason)
        """

        self.position = self.om.position

        # [OK] ADD DEBUG LOGGING
        self.log(
            f"\n{'=' * 60}\n"
            f"[TRAILING-DEBUG] Called\n"
            f"  LTP: ₹{ltp:.2f}\n"
            f"  ATR: {atr:.2f}\n"
            f"  Position type: {self.position.get('type')} [OK]\n"
            f"  Entry: ₹{self._f(self.position.get('entry_price')):.2f} [OK]\n"
            f"{'=' * 60}",
            False
        )
        
        if not self.position or self.position.get("type") == "FLAT":
            return False, None

        entry = self._f(self.position.get("entry_price"))
        side = self.position.get("type")

        if not entry or not ltp:
            return False, None

        # Calculate profit
        if side == "BUY":
            profit_points = ltp - entry
        else:
            profit_points = entry - ltp

        profit_rupees = profit_points * 250

        self.log(
            f"[TRAILING] Update check:\n"
            f"  Current profit: ₹{profit_rupees:.0f} ({profit_points:.2f} pts)",
            True
        )

        # Update stops based on profit level
        if profit_rupees >= 500 and atr:
            self._update_ultra_tight(ltp, profit_rupees, side)

        if atr and profit_rupees >= 0:
            self._update_tiered(ltp, atr, profit_rupees, side, entry)

        if pivot_data and atr:
            self._update_cpr_dynamic(ltp, atr, pivot_data, side)

        # Check if should exit
        return self._check_profit_protection(profit_rupees, side)

    def _update_ultra_tight(self, ltp, profit_rupees, side):
        """Ultra-tight trailing for ₹500+ profits"""

        current_sl = self._f(self.position.get("stop_loss"))
        if not current_sl:
            return

        # Determine trail distance
        if profit_rupees >= 2000:
            trail_rupees = 75
        elif profit_rupees >= 1500:
            trail_rupees = 100
        elif profit_rupees >= 1000:
            trail_rupees = 125
        else:
            trail_rupees = 150

        trail_points = trail_rupees / 250

        if side == "BUY":
            new_sl = ltp - trail_points
            should_update = new_sl > current_sl
        else:
            new_sl = ltp + trail_points
            should_update = new_sl < current_sl

        if should_update:
            self.position["stop_loss"] = round(new_sl, 2)
            locked = profit_rupees - trail_rupees

            self.log(
                f"[SIGNAL] [ULTRA-TIGHT] SL updated\n"
                f"  Old: ₹{current_sl:.2f} → New: ₹{new_sl:.2f}\n"
                f"  Trail: ₹{trail_rupees}\n"
                f"  Locked: ₹{locked:.0f}",
                False
            )

    def _update_tiered(self, ltp, atr, profit_rupees, side, entry):
        """Tiered trailing stops"""

        self.position = self.om.position
        current_sl = self._f(self.position.get("stop_loss"))
        if not current_sl:
            return

        # Breakeven at ₹250
        if 0 < profit_rupees < 500:
            if profit_rupees >= 250 and not self.position.get("breakeven_set"):
                self.position["stop_loss"] = round(entry, 2)
                self.position["breakeven_set"] = True
                self.log(f"[SIGNAL] [BREAKEVEN] SL → Entry: ₹{entry:.2f}", False)
                return

        # Tiered trailing
        if profit_rupees >= 500:
            if 500 <= profit_rupees < 1000:
                trail_dist = 1.2 * atr
            elif 1000 <= profit_rupees < 3000:
                trail_dist = 1.0 * atr
            else:
                trail_dist = 0.6 * atr

            if side == "BUY":
                new_sl = max(current_sl, ltp - trail_dist)
            else:
                new_sl = min(current_sl, ltp + trail_dist)

            if (side == "BUY" and new_sl > current_sl) or (side == "SELL" and new_sl < current_sl):
                self.position["stop_loss"] = round(new_sl, 2)
                self.log(f"[SIGNAL] [TIERED] SL updated: ₹{new_sl:.2f}", True)

    def _update_cpr_dynamic(self, ltp, atr, pivot_data, side):
        """CPR-based dynamic SL"""

        self.position = self.om.position
        current_sl = self._f(self.position.get("stop_loss"))
        if not current_sl:
            return

        buffer = 0.2 * atr

        tc = self._f(pivot_data.get("TC"))
        bc = self._f(pivot_data.get("BC"))
        r1 = self._f(pivot_data.get("R1"))
        r2 = self._f(pivot_data.get("R2"))
        r3 = self._f(pivot_data.get("R3"))
        s1 = self._f(pivot_data.get("S1"))
        s2 = self._f(pivot_data.get("S2"))
        s3 = self._f(pivot_data.get("S3"))

        new_sl = None

        if side == "BUY":
            if r3 and ltp > r3 and r2:
                new_sl = r2 - buffer
            elif r2 and ltp > r2 and r1:
                new_sl = r1 - buffer
            elif r1 and ltp > r1 and tc:
                new_sl = tc - buffer

            if new_sl and new_sl > current_sl:
                self.position["stop_loss"] = round(new_sl, 2)
                self.log(f"[SIGNAL] [CPR-SL] Updated: ₹{new_sl:.2f}", False)

        elif side == "SELL":
            if s3 and ltp < s3 and s2:
                new_sl = s2 + buffer
            elif s2 and ltp < s2 and s1:
                new_sl = s1 + buffer
            elif s1 and ltp < s1 and bc:
                new_sl = bc + buffer

            if new_sl and new_sl < current_sl:
                self.position["stop_loss"] = round(new_sl, 2)
                self.log(f"[SIGNAL] [CPR-SL] Updated: ₹{new_sl:.2f}", False)

    def _check_profit_protection(self, profit_rupees, side):
        """Exit if profit drops too much"""

        self.position = self.om.position

        max_profit = self.position.get("_max_profit", 0)

        if profit_rupees > max_profit:
            self.position["_max_profit"] = profit_rupees
            max_profit = profit_rupees

        if max_profit <= 0:
            return False, None

        drawdown = max_profit - profit_rupees
        drawdown_pct = (drawdown / max_profit * 100) if max_profit > 0 else 0

        # Tiered protection
        if max_profit >= 1500:
            max_dd = 20
        elif max_profit >= 1000:
            max_dd = 30
        else:
            max_dd = 40

        if drawdown_pct >= max_dd:
            return True, f"Profit Protection ({drawdown_pct:.1f}% drop from ₹{max_profit:.0f})"

        return False, None

    @staticmethod
    def _f(v, alt=None):
        try:
            if v is None: return alt
            x = float(v)
            return x if isfinite(x) else alt
        except:
            return alt

class TradingBot:
    # ---------- CORE HELPER METHODS ----------
    @staticmethod
    def _f(v, alt=None):
        try:
            if v is None: return alt
            x = float(v)
            return x if isfinite(x) else alt
        except Exception:
            return alt
    
    def get_dynamic_lot_size(self, symbol: str) -> int:
        """
        Fetch lot size dynamically from Fyers API.
        Falls back to intelligent defaults based on symbol type.
        
        Args:
            symbol: Trading symbol (e.g., "MCX:NATGASMINI26FEBFUT", "BSE:SENSEX-INDEX")
        
        Returns:
            lot_size: Integer lot size for the symbol
        """
        # Check cache first
        if hasattr(self, '_lot_size_cache'):
            if symbol in self._lot_size_cache:
                return self._lot_size_cache[symbol]
        else:
            self._lot_size_cache = {}
        
        try:
            # Try to fetch from Fyers quotes API
            resp = self.fyers_sdk_instance.quotes({"symbols": symbol})
            if resp.get("s") == "ok" and "d" in resp:
                for item in resp.get("d", []):
                    v = item.get("v", {})
                    lot = v.get("lot_size") or v.get("min_qty") or v.get("lotSize")
                    if lot and int(lot) > 0:
                        lot_size = int(lot)
                        self._lot_size_cache[symbol] = lot_size
                        self.log_message(f"[LOT-SIZE] Dynamic lot size for {symbol}: {lot_size}", True)
                        return lot_size
        except Exception as e:
            self.log_message(f"[LOT-SIZE] Error fetching lot size for {symbol}: {e}", True)
        
        # Intelligent defaults based on symbol type
        symbol_upper = symbol.upper()
        default_lots = {
            # Index Options
            "SENSEX": 20,
            "NIFTY50": 75,
            "NIFTY": 75,
            "BANKNIFTY": 30,
            "FINNIFTY": 65,
            "MIDCPNIFTY": 50,
            
            # Commodities (MCX)
            "NATGASMINI": 250,  # Natural Gas Mini
            "NATGAS": 1250,    # Natural Gas
            "CRUDEOIL": 100,
            "CRUDEOILM": 10,
            "GOLD": 100,
            "GOLDM": 10,
            "GOLDPETAL": 1,
            "SILVER": 30,
            "SILVERM": 5,
            "SILVERMIC": 1,
            "COPPER": 2500,
            
            # Default for equities
            "EQ": 1,
        }
        
        # Find matching lot size
        for key, lot in default_lots.items():
            if key in symbol_upper:
                self._lot_size_cache[symbol] = lot
                self.log_message(f"[LOT-SIZE] Using default lot size for {symbol}: {lot}", True)
                return lot
        
        # Ultimate fallback
        self._lot_size_cache[symbol] = 1
        self.log_message(f"[LOT-SIZE] Using fallback lot size of 1 for {symbol}", True)
        return 1



    def __init__(self, config_dir="config", run_websocket=True):
        self.IST = IST
        self.DEBUG = True

        # Trading configuration
        # Primary symbol: BSE:SENSEX-INDEX
        self.symbols = ["BSE:SENSEX-INDEX"]
        self.symbol = self.symbols[0]
        self.symbol_clean = self.symbol.replace(":", "_")

        # Strategy flags - ADD THESE NEW FLAGS
        self.USE_TREND_STRATEGY = True  # Use traditional trend-based strategy
        self.USE_AI_CPR_STRATEGY = True  # Use AI CPR-based strategy
        self.USE_COMBINED_STRATEGY = True  # Use both strategies together

        self._setup_paths()
        self._initialize_symbol_files()
        self._load_config(config_dir)

        # Initialize Fyers SDK
        self.fyers_sdk_instance = fyersModel.FyersModel(
            client_id=self.client_id,
            token=self.access_token,
            is_async=False,
            log_path=""
        )

        # Initialize AI Predictor - ENHANCED INITIALIZATION
        self.ai_predictor = CPR_AIPredictor(
            # model_path="ai_cpr_model.pkl",  # Specify model path
            # model_path="ai_cpr_model_v2.pkl",
            model_path="ai_cpr_model_final.pkl",
            logger=logger
        )

        # Websocket caches
        self.websocket_ltp = {}
        self.websocket_lock = threading.Lock()
        self._last_print_time = {}
        self.fresh_indicators_all_tfs = {}

        # Websocket OHLC accumulation
        # self.ohlc_timeframes = [5, 15, 30]  # Track 5min, 15min, 30min
        # Parse timeframe from argument (supports comma-separated for multiple timeframes)
        tf_list = [int(tf.strip()) for tf in str(tf_selected).split(',') if tf.strip()]
        self.ohlc_timeframes = tf_list
        self.ohlc_data = {tf: {} for tf in self.ohlc_timeframes}
        self.csv_data = {tf: {} for tf in self.ohlc_timeframes}
        self.timeframe_counters = {tf: 0 for tf in self.ohlc_timeframes}
        self.last_minute_seen = {tf: None for tf in self.ohlc_timeframes}
        self.ohlc_lock = threading.Lock()

        if run_websocket:
            self._setup_websocket()
        else:
            self.log_message("WebSocket disabled.", False)

        # Per-symbol order managers & candle tracking
        self.order_managers = {}
        self.last_bar_ts = {sym: None for sym in self.symbols}
        self.prev_st21 = {sym: None for sym in self.symbols}

        # ENHANCED ORDER MANAGER INITIALIZATION
        for sym in self.symbols:
            # Fetch dynamic lot size for this symbol
            dynamic_lot_size = self.get_dynamic_lot_size(sym)
            self.log_message(f"[INIT] Using lot size {dynamic_lot_size} for {sym}", False)
            
            fyers_service = FyersService(
                self.fyers_sdk_instance,
                self.data_paths[sym]['raw_api_log'],
                self.log_message,
                self.get_websocket_ltp
            )
            self.order_managers[sym] = OrderManager(
                fyers_service=fyers_service,
                symbol=sym,
                lot_size=dynamic_lot_size,  # Dynamic lot size instead of hardcoded 1
                log_fn=self.log_message,
                state_path=self.data_paths[sym]['om_state_path'],
                event_log=self.data_paths[sym]['om_event_log'],
                report_dir=self.data_paths[sym]['trade_report_dir'],
                ai_predictor=self.ai_predictor,  # Pass AI predictor
                bot=self  # Pass bot reference for fetch_ohlc
            )
            # Configure signal modes
            self.order_managers[sym].SIGNAL_MODE = "both"
            # Configure AI CPR settings
            self.order_managers[sym].AI_CPR_ENABLED = self.USE_AI_CPR_STRATEGY
            self.order_managers[sym].AI_MIN_CONF = 0.25
            self.order_managers[sym].AI_GATE_TRADES = True

        self.candle_analyzer = CandlestickAnalyzer(self)
        self.indicator_calculator = IndicatorCalculator(self)
        self.perf_tracker = PerformanceTracker(self)
        self.log_message("[PERF] Performance tracker initialized", True)
        self.previous_states = {tf: {} for tf in ['1', '5', '15', '30', '60']}
        
        # Strategy flag for option trading
        self.USE_OPTION_STRATEGY = True  # Enable/disable SENSEX option buying strategy
        
        # Initialize SENSEX Option Buyer (optional - for SENSEX option trading)
        self.sensex_option_buyer = None
        self.option_order_manager = None
        
        try:
            # Updated config for BSE SENSEX-INDEX
            crude_config = {
                "symbol_fut": "BSE:SENSEX26FEBFUT",
                "underlying_index": "BSE:SENSEX-INDEX", # Uses master for index data
            }
            self.sensex_option_buyer = SensexOptionBuyer(
                fyers_client=self.fyers_sdk_instance,
                log_fn=self.log_message,
                config=crude_config
            )
            self.log_message("[SENSEX-OPT] Option Buyer module initialized", False)
            
            # Initialize OptionOrderManager (bridge between SensexOptionBuyer and OrderManager)
            self.option_order_manager = OptionOrderManager(
                trading_bot=self,
                log_fn=self.log_message
            )
            self.log_message("[OPT-MGR] Option Order Manager initialized", False)

            # -------------------------------------------------------------
            # V3 Generic Option Buyer (supports FUT -> option selection)
            # Enables NATGASMINI FUT symbols like MCX:NATGASMINI26MARFUT to
            # automatically pick CE/PE from option chain using greeks/PCR/VIX.
            # -------------------------------------------------------------
            try:
                self._v3_mkt_adapter = FyersMarketDataAdapterV2(self.fyers_sdk_instance, bot=self)
                self._v3_option_buyer = GenericGreeksOptionBuyerV2(self._v3_mkt_adapter, configs=DEFAULT_CONFIGS_V2)
                self.log_message("[OPT-V3] GenericGreeksOptionBuyerV2 initialized", False)
            except Exception as e:
                self._v3_mkt_adapter = None
                self._v3_option_buyer = None
                self.log_message(f"[OPT-V3] Failed to init V3 option buyer: {e}", False)
            
        except Exception as e:
            self.log_message(f"[SENSEX-OPT] Failed to initialize option buyer/manager: {e}", False)
            import traceback
            traceback.print_exc()
    
    def run_sensex_option_analysis(self) -> Optional[OptionBuySignal]:
        """
        Convenience method to run SENSEX option buy analysis.
        
        Returns:
            OptionBuySignal if conditions are met, None otherwise
        """
        if self.sensex_option_buyer is None:
            self.log_message("[SENSEX-OPT] Option buyer not initialized", False)
            return None
        
        try:
            signal = self.sensex_option_buyer.analyze_and_get_signal()
            
            if signal.should_buy:
                self.log_message(
                    f"[SENSEX-OPT] [OK] BUY SIGNAL: {signal.symbol} @ {signal.entry_price} "
                    f"| Lot={signal.lot_size} | Delta={signal.greeks.get('delta', 0):.3f}",
                    False
                )
            
            return signal
            
        except Exception as e:
            self.log_message(f"[SENSEX-OPT] Analysis error: {e}", False)
            return None

    def execute_option_strategy(self) -> Optional[OptionBuySignal]:
        """
        Execute the SENSEX option buying strategy.
        
        This is the main method that combines:
        1. SensexOptionBuyer analysis (Greeks, PCR, VIX, Momentum)
        2. OptionOrderManager execution (Entry, Trailing, Exit)
        
        Flow:
        - If no active trade: Analyze market → Get signal → Execute entry
        - If active trade: Manage trailing stops → Check exit conditions
        
        Returns:
            OptionBuySignal if trade is active/executed, None otherwise
        """
        if not self.USE_OPTION_STRATEGY:
            return None
            
        if not self.option_order_manager:
            self.log_message("[OPT-STRATEGY] Option Order Manager not initialized", False)
            return None
            
        try:
            # Execute the analyze_and_trade flow
            result = self.option_order_manager.analyze_and_trade()
            
            # Log status
            status = self.option_order_manager.get_status()
            if status["in_option_trade"]:
                self.log_message(
                    f"[OPT-STRATEGY] Active Trade: {status['active_symbol']} ({status['option_type']}) | "
                    f"Entry: ₹{status['entry_price']:.2f} | Daily P&L: ₹{status['daily_pnl']:.0f}",
                    True
                )
            
            return result
            
        except Exception as e:
            self.log_message(f"[OPT-STRATEGY] Error: {e}", False)
            import traceback
            traceback.print_exc()
            return None

    def get_option_trading_status(self) -> Dict:
        """
        Get current option trading status for monitoring/dashboard.
        
        Returns:
            Dict with trading status information
        """
        if not self.option_order_manager:
            return {"enabled": False, "error": "Not initialized"}
            
        return {
            "enabled": self.USE_OPTION_STRATEGY,
            **self.option_order_manager.get_status()
        }

    def force_exit_option_trade(self, reason: str = "Manual Exit") -> None:
        """
        Force exit any active option trade.
        
        Args:
            reason: Exit reason for logging
        """
        if not self.option_order_manager:
            self.log_message("[OPT-STRATEGY] Option Order Manager not initialized", False)
            return
            
        self.option_order_manager.force_exit(reason)

    def get_cpr_levels_with_fallback(self, symbol):
        """
        Get CPR levels with multiple fallback strategies
        """
        # Try 1: Load from JSON
        pivot_json_path = self.data_paths[symbol]['pivot_json']
        pivot_data = robust_load_json(pivot_json_path, self.log_message, default={})

        cpr_levels = pivot_data.get(symbol, {})

        # Validate CPR levels
        required_keys = ["TC", "BC", "R1", "R2", "R3", "S1", "S2", "S3"]
        missing = [k for k in required_keys if k not in cpr_levels or cpr_levels[k] is None]

        if not missing:
            # Check if stale (older than 24 hours)
            pivot_ts = cpr_levels.get("ts", "1970-01-01")
            try:
                pivot_age = pd.to_datetime(pivot_ts) if pivot_ts != "1970-01-01" else pd.Timestamp.min
                hours_old = (pd.Timestamp.now(tz=IST) - pivot_age).total_seconds() / 3600

                if hours_old < 24:
                    self.log_message(f"[OK] Using valid CPR levels (age: {hours_old:.1f}h)", True)
                    return cpr_levels
                else:
                    self.log_message(f"[WARNING] CPR levels stale ({hours_old:.1f}h old), recalculating", False)
            except Exception as e:
                self.log_message(f"[WARNING] CPR age check failed: {e}", True)
        else:
            self.log_message(f"[WARNING] CPR missing keys: {missing}", False)

        # Try 2: Recalculate pivots
        self.log_message("🔄 Recalculating CPR levels...", False)
        try:
            new_pivots = self.process_pivots()

            if new_pivots and symbol in new_pivots:
                cpr_levels = new_pivots[symbol]

                # Validate again
                missing = [k for k in required_keys if k not in cpr_levels or cpr_levels[k] is None]
                if not missing:
                    self.log_message("[OK] CPR recalculation successful", False)
                    return cpr_levels
                else:
                    self.log_message(f"[ERROR] Recalculated CPR still missing: {missing}", False)

        except Exception as e:
            self.log_message(f"[ERROR] CPR recalculation failed: {e}", False)

        # Try 3: Use previous day's levels as emergency fallback
        self.log_message("[WARNING] Using emergency fallback CPR (may be inaccurate)", False)

        try:
            # Get current price
            ltp = self.get_websocket_ltp(symbol) or 350.0  # Default fallback

            # Generate approximate pivots based on LTP
            pp = round(ltp, 2)
            range_est = ltp * 0.02  # Estimate 2% range

            emergency_cpr = {
                "PP": pp,
                "TC": round(pp + range_est * 0.5, 2),
                "BC": round(pp - range_est * 0.5, 2),
                "R1": round(pp + range_est, 2),
                "R2": round(pp + range_est * 1.5, 2),
                "R3": round(pp + range_est * 2, 2),
                "S1": round(pp - range_est, 2),
                "S2": round(pp - range_est * 1.5, 2),
                "S3": round(pp - range_est * 2, 2),
                "High": round(ltp * 1.01, 2),
                "Low": round(ltp * 0.99, 2),
                "Close": ltp,
                "virgin_cpr": False,
                "ts": dt.datetime.now(IST).isoformat(),
                "_is_emergency_fallback": True
            }

            self.log_message(
                f"[WARNING] EMERGENCY CPR (approximate):\n"
                f"  TC={emergency_cpr['TC']}, BC={emergency_cpr['BC']}, PP={emergency_cpr['PP']}",
                False
            )

            return emergency_cpr

        except Exception as e:
            self.log_message(f"[ERROR] Emergency fallback failed: {e}", False)
            return {}

    def run(self, selected_tf=None):
        """
        Unified main trading loop:
        Combines AI CPR, Trend-based, and Combined strategy execution.
        """
        if selected_tf is None:
            selected_tf = tf_selected  # From sys.argv

        self.log_message(f"[SIGNAL] Trading decisions based on {selected_tf}m timeframe", False)
        self.initialize_pivots()
        # [OK] Verify pivots loaded successfully
        pivot_data = robust_load_json(self.pivot_json, self.log_message, default={})
        if self.symbol not in pivot_data:
            self.log_message("[ERROR] CRITICAL: Pivots not initialized! Cannot continue.", False)
            return

        symbol_pivots = pivot_data.get(self.symbol, {})
        required_keys = ["TC", "BC", "R1", "R2", "R3", "S1", "S2", "S3"]
        missing_keys = [k for k in required_keys if k not in symbol_pivots or symbol_pivots[k] is None]

        if missing_keys:
            self.log_message(f"[ERROR] CRITICAL: Pivots incomplete (missing: {', '.join(missing_keys)})", False)
            self.log_message("🔄 Attempting emergency pivot calculation...", False)
            self.process_pivots()

            # Re-check after calculation
            pivot_data = robust_load_json(self.pivot_json, self.log_message, default={})
            symbol_pivots = pivot_data.get(self.symbol, {})
            missing_keys = [k for k in required_keys if k not in symbol_pivots or symbol_pivots[k] is None]

            if missing_keys:
                self.log_message(f"[ERROR] FATAL: Cannot calculate pivots! Trading stopped.", False)
                return

        self.log_message(
            f"[OK] Pivots verified: TC={symbol_pivots['TC']}, BC={symbol_pivots['BC']}, "
            f"R1={symbol_pivots['R1']}, S1={symbol_pivots['S1']}",
            False
        )
        if hasattr(self, 'ai_predictor') and self.ai_predictor:
            health = self.check_ai_model_health()
            self.log_message(
                f"[AI-HEALTH] Model status: {health.get('status', 'unknown')} - "
                f"{health.get('message', 'N/A')}",
                False
            )
        #pivot_data = robust_load_json(self.pivot_json, self.log_message, default={})
        pivot_data = self.get_cpr_levels_with_fallback(self.symbol)
        if self.symbol in pivot_data:
            pivot_ts = pivot_data[self.symbol].get("ts", "1970-01-01")
            try:
                pivot_age = pd.to_datetime(pivot_ts) if pivot_ts != "1970-01-01" else pd.Timestamp.min
                if (pd.Timestamp.now(tz=IST) - pivot_age).total_seconds() > 86400:  # 24 hours
                    self.log_message("Pivots are stale (>24h), forcing recalculation", False)
                    self.process_pivots()  # Force recalculate
            except Exception as e:
                self.log_message(f"Error checking pivot age: {e}, forcing recalc", False)
                self.process_pivots()

        timeframes_to_process = ["1", "5", "15", "30"]

        while True:
            now = dt.datetime.now(self.IST)
            fresh_indicators_all_tfs = {}

            # ─────────────────────────── INDICATOR CALCULATION ───────────────────────────
            for sym in self.symbols:
                all_data_json_path = self.data_paths[sym]['all_data_json']
                dashboard_data = robust_load_json(all_data_json_path, self.log_message, default={})
                dashboard_data.setdefault("Dashboard", {})

                # ═══════════════════════════════════════════════════════════
                # CRITICAL: LOAD AND VALIDATE PIVOT DATA
                # ═══════════════════════════════════════════════════════════
                pivot_json_path = self.data_paths[sym]['pivot_json']

                # Load the JSON file
                pivot_json_data = robust_load_json(pivot_json_path, self.log_message, default={})

                # Extract the symbol-specific pivot data
                if isinstance(pivot_json_data, dict) and sym in pivot_json_data:
                    pivots = pivot_json_data[sym]
                else:
                    self.log_message(f"[WARNING] Pivot JSON structure invalid for {sym}", False)
                    self.log_message(f"   JSON type: {type(pivot_json_data)}", False)
                    self.log_message(
                        f"   JSON keys: {list(pivot_json_data.keys()) if isinstance(pivot_json_data, dict) else 'NOT A DICT'}",
                        False)
                    pivots = {}

                # Ensure pivots is a dictionary
                if not isinstance(pivots, dict):
                    self.log_message(f"[WARNING] Pivots for {sym} is {type(pivots)}, converting to dict", False)
                    pivots = {}

                # Validate required keys
                required_keys = ["TC", "BC", "R1", "R2", "R3", "S1", "S2", "S3"]
                missing_keys = [k for k in required_keys if k not in pivots or pivots[k] is None]

                if missing_keys:
                    self.log_message(
                        f"[ERROR] Pivots incomplete for {sym}: missing {', '.join(missing_keys)}\n"
                        f"   Attempting to recalculate...",
                        False
                    )

                    # Force recalculation
                    self.process_pivots()

                    # Reload after recalculation
                    pivot_json_data = robust_load_json(pivot_json_path, self.log_message, default={})
                    pivots = pivot_json_data.get(sym, {}) if isinstance(pivot_json_data, dict) else {}

                    # Re-validate
                    missing_keys = [k for k in required_keys if k not in pivots or pivots[k] is None]

                    if missing_keys:
                        self.log_message(
                            f"[ERROR] CRITICAL: Cannot calculate pivots for {sym}\n"
                            f"   Still missing: {', '.join(missing_keys)}\n"
                            f"   SKIPPING THIS SYMBOL",
                            False
                        )
                        continue  # Skip this symbol entirely

                #self.log_message(
                #    f"[OK] Pivots loaded for {sym}: "
                #    f"TC={pivots['TC']}, BC={pivots['BC']}, "
                #    f"R1={pivots['R1']}, S1={pivots['S1']}",
                #    True
                #)

                fresh_indicators_all_tfs[sym] = {}

                # Fetch OHLC once for candle pattern analysis
                ohlc_df_for_patterns = None
                if self.USE_AI_CPR_STRATEGY:
                    try:
                        ohlc_df_for_patterns = self.fetch_ohlc(sym, "5", 10)  # Fetch 10 days of data for patterns
                        #if ohlc_df_for_patterns is not None and not ohlc_df_for_patterns.empty:
                           # self.log_message(f"[DEBUG] Fetched OHLC for {sym} patterns: {len(ohlc_df_for_patterns)} rows", True)
                        #else:
                        #    self.log_message(f"[WARN] No OHLC data fetched for {sym} - candle features will use zeros", False)
                    except Exception as e:
                        self.log_message(f"[WARN] Failed to fetch OHLC for patterns {sym} 5m: {e}", True)

                for tf in timeframes_to_process:
                    # [OK] ENSURE pivots is a dictionary, not a list
                    if not isinstance(pivots, dict):
                        self.log_message(f"[WARNING] Pivots for {sym} is {type(pivots)}, using empty dict", False)
                        pivots_to_pass = {}
                    else:
                        pivots_to_pass = pivots

                    #self.log_message(
                    #    f"[DEBUG] Passing pivots to calculator for {sym} {tf}m: "
                    #    f"type={type(pivots_to_pass)}, "
                    #    f"keys={list(pivots_to_pass.keys())[:5] if isinstance(pivots_to_pass, dict) else 'N/A'}",
                    #    True
                    #)

                    indicators = self.indicator_calculator.calculate_indicators(
                        sym, tf, pivot_data=pivots_to_pass
                    )
                    # self.log_message(f"Calculating indicators for {sym} on {tf}min timeframe...", True)
                    #indicators = self.indicator_calculator.calculate_indicators(sym, tf, pivot_data=pivots)
                    fresh_indicators_all_tfs[sym][tf] = indicators

                    # Optional AI-CPR Analysis (Real-time friendly)
                    if ("error" not in indicators and self.USE_AI_CPR_STRATEGY):

                        try:
                            cpr_analysis = analyze_cpr_strategy(indicators, pivots, self.ai_predictor,
                                                                ohlc_df=ohlc_df_for_patterns)
                            indicators["cpr_analysis"] = cpr_analysis
                            trade_signal = cpr_analysis.get('trade_strategy', 'None')
                            ai_status = ""
                            if cpr_analysis.get('ai_cpr_label'):
                                ai_conf = cpr_analysis.get('ai_confidence', 0.0)
                                ai_status = f" | AI:{cpr_analysis['ai_cpr_label']}({ai_conf:.2f})"
                            self.log_message(
                                f"[AI-CPR] {sym} {tf}m → {trade_signal}{ai_status}",
                                True
                            )
                        except Exception as e:
                            self.log_message(f"[AI-CPR] Analysis failed for {sym} {tf}m: {e}", False)
                            indicators["cpr_analysis"] = {"error": str(e), "trade_strategy": "None"}
                    else:
                        # Ensure cpr_analysis exists even when skipped
                        indicators["cpr_analysis"] = {"trade_strategy": "None", "reason": "Insufficient data"}

                    if "error" in indicators:
                        dashboard_data["Dashboard"][tf] = {
                            "error": indicators.get("error"),
                            "ts": now.isoformat()
                        }
                    else:
                        dashboard_data["Dashboard"][tf] = {
                            "inds": convert_dict_to_serializable(indicators),
                            "pivots": pivots,
                            "ts": now.isoformat()
                        }

                robust_save_json(dashboard_data, all_data_json_path, self.log_message, debug_only=True)

            # ─────────────────────────── STRATEGY EXECUTION ───────────────────────────
            for sym in self.symbols:
                five_inds = fresh_indicators_all_tfs.get(sym, {}).get("5")

                # self.log_message(f"[DEBUG] 5m indicators for {sym}: {five_inds.keys() if five_inds else 'None'}", False)

                if not five_inds or "error" in five_inds:
                    self.log_message(f"[WARN] Using last known indicators for {sym} due to 5m calc error.", False)
                    five_inds = robust_load_json(self.data_paths[sym]['all_data_json'], self.log_message, default={}) \
                        .get("Dashboard", {}).get("5", {}).get("inds", {})
                    if not five_inds:
                        self.log_message(f"Skipping {sym}: no previous indicators found either.", False)
                        continue

                try:
                    #current_bar_ts_5 = pd.to_datetime(five_inds.get("timestamp")).tz_convert(IST)
                    current_bar_ts = pd.to_datetime(five_inds.get("timestamp")).tz_convert(IST)
                except Exception:
                    self.log_message(f"Bad 5m timestamp for {sym}, skipping.", False)
                    continue

                #new_candle_5 = (current_bar_ts_5 != self.last_bar_ts.get(sym))
                #intra_candle_entry_allowed = False

                selected_tf_int = int(selected_tf)
                #current_bar_ts = pd.to_datetime(five_inds.get("timestamp")).tz_convert(IST)

                # Check if we're within the SELECTED timeframe candle
                new_candle = (current_bar_ts != self.last_bar_ts.get(sym))
                intra_candle_entry_allowed = False

                if not new_candle:  # Still within current candle of selected timeframe
                    self.log_message(
                        f"⏳ Within {selected_tf}m candle - checking early entry conditions",
                        True
                    )

                    # Get current price movement
                    ltp = self.get_websocket_ltp(sym)
                    candle_open = self._f(five_inds.get("open"))
                    # 🔥 NEW: Get current candle high/low to detect false moves
                    candle_high = self._f(five_inds.get("high"))
                    candle_low = self._f(five_inds.get("low"))

                    if ltp and candle_open and candle_high and candle_low:
                        intra_candle_move_pct = abs((ltp - candle_open) / candle_open * 100)

                        # [SIGNAL] NEW: Detect current candle color (CRITICAL!)
                        current_candle_bullish = ltp > candle_open
                        current_candle_bearish = ltp < candle_open

                        # 🔥 NEW: Check if price is near candle extremes (not retracing)
                        candle_range = candle_high - candle_low
                        if candle_range > 0:
                            # For bullish: price should be in upper 30% of candle
                            bullish_position = (ltp - candle_low) / candle_range
                            # For bearish: price should be in lower 30% of candle
                            bearish_position = (candle_high - ltp) / candle_range
                        else:
                            bullish_position = 0.5
                            bearish_position = 0.5

                        # Get indicators
                        volume_ratio = self._f(five_inds.get("volume_ratio"), 1.0)
                        momentum_pct = self._f(five_inds.get("momentum_pct"), 0.0)
                        ema5 = self._f(five_inds.get("ema_5"))
                        ema9 = self._f(five_inds.get("ema_9"))
                        ema21 = self._f(five_inds.get("ema_21"))
                        st_main_trend = five_inds.get("st_main_trend", 0)

                        # [SIGNAL] ADAPTIVE THRESHOLDS based on timeframe
                        if selected_tf_int <= 5:
                            MOVE_THRESHOLD = 0.2
                            VOLUME_THRESHOLD = 1.2
                            MOMENTUM_THRESHOLD = 0.2
                            POSITION_THRESHOLD = 0.70  # Must be in top/bottom 30%
                        elif selected_tf_int <= 15:
                            MOVE_THRESHOLD = 0.3
                            VOLUME_THRESHOLD = 1.3
                            MOMENTUM_THRESHOLD = 0.3
                            POSITION_THRESHOLD = 0.65  # Slightly more lenient
                        else:  # 30+ min
                            MOVE_THRESHOLD = 0.5
                            VOLUME_THRESHOLD = 1.5
                            MOMENTUM_THRESHOLD = 0.5
                            POSITION_THRESHOLD = 0.60  # Most lenient for 30m

                        # Check trend alignment
                        trend_aligned = False
                        trend_direction = None

                        if ema5 and ema9 and ema21:
                            # 🟢 BULLISH CONDITIONS (ALL MUST BE TRUE)
                            if (ema5 > ema9 and ema9 > ema21 and
                                    momentum_pct > MOMENTUM_THRESHOLD and
                                    st_main_trend > 0 and
                                    current_candle_bullish and  # 🔥 NEW: Candle must be green
                                    bullish_position >= POSITION_THRESHOLD):  # 🔥 NEW: Price near high
                                trend_aligned = True
                                trend_direction = "BULLISH"

                            # 🔴 BEARISH CONDITIONS (ALL MUST BE TRUE)
                            elif (ema5 < ema9 and ema9 < ema21 and
                                  momentum_pct < -MOMENTUM_THRESHOLD and
                                  st_main_trend < 0 and
                                  current_candle_bearish and  # 🔥 NEW: Candle must be red
                                  bearish_position >= POSITION_THRESHOLD):  # 🔥 NEW: Price near low
                                trend_aligned = True
                                trend_direction = "BEARISH"

                        # [OK] ENTER IMMEDIATELY if conditions met
                        if (trend_aligned and
                                intra_candle_move_pct >= MOVE_THRESHOLD and
                                volume_ratio >= VOLUME_THRESHOLD and
                                abs(momentum_pct) >= MOMENTUM_THRESHOLD):

                            intra_candle_entry_allowed = True
                            st_status = "GREEN ✓" if st_main_trend > 0 else "RED ✓"

                            # 🔥 NEW: Calculate position in candle
                            position_pct = bullish_position if trend_direction == "BULLISH" else bearish_position

                            self.log_message(
                                f"[START] [EARLY-ENTRY-{selected_tf}m] {trend_direction} trend detected!\n"
                                f"  [DATA] Timeframe: {selected_tf}m\n"
                                f"  📈 Move: {intra_candle_move_pct:.2f}% (threshold: {MOVE_THRESHOLD}%)\n"
                                f"  [DATA] Volume: {volume_ratio:.2f}x (threshold: {VOLUME_THRESHOLD}x)\n"
                                f"  [SIGNAL] Momentum: {momentum_pct:.2f}% (threshold: ±{MOMENTUM_THRESHOLD}%)\n"
                                f"  [SIGNAL] SuperTrend: {st_status}\n"
                                f"  🕯️ Candle: {'GREEN' if current_candle_bullish else 'RED'} "
                                f"(O:{candle_open:.2f} → C:{ltp:.2f})\n"
                                f"  [POS] Position in candle: {position_pct * 100:.1f}% "
                                f"({'near HIGH' if trend_direction == 'BULLISH' else 'near LOW'})\n"
                                f"  [OK] EMA Alignment: {ema5:.2f} {'>' if trend_direction == 'BULLISH' else '<'} "
                                f"{ema9:.2f} {'>' if trend_direction == 'BULLISH' else '<'} {ema21:.2f}\n"
                                f"  [OK] Decision: ENTER NOW without waiting for candle close",
                                False
                            )
                        else:
                            # Log why we're NOT entering
                            reasons = []
                            if not trend_aligned:
                                if not current_candle_bullish and momentum_pct > 0:
                                    reasons.append("🚫 Candle is RED but momentum bullish (conflicting)")
                                elif not current_candle_bearish and momentum_pct < 0:
                                    reasons.append("🚫 Candle is GREEN but momentum bearish (conflicting)")
                                elif ema5 and ema9 and not (ema5 > ema9 if momentum_pct > 0 else ema5 < ema9):
                                    reasons.append(f"🚫 EMA not aligned (5:{ema5:.2f}, 9:{ema9:.2f})")
                                elif st_main_trend == 0:
                                    reasons.append("🚫 SuperTrend neutral")
                                elif st_main_trend > 0 and momentum_pct < 0:
                                    reasons.append("🚫 ST GREEN but momentum bearish")
                                elif st_main_trend < 0 and momentum_pct > 0:
                                    reasons.append("🚫 ST RED but momentum bullish")

                                # 🔥 NEW: Check position in candle
                                if momentum_pct > 0 and bullish_position < POSITION_THRESHOLD:
                                    reasons.append(
                                        f"🚫 Price too low in candle ({bullish_position * 100:.1f}% < {POSITION_THRESHOLD * 100:.1f}%)"
                                    )
                                elif momentum_pct < 0 and bearish_position < POSITION_THRESHOLD:
                                    reasons.append(
                                        f"🚫 Price too high in candle ({bearish_position * 100:.1f}% < {POSITION_THRESHOLD * 100:.1f}%)"
                                    )

                            if intra_candle_move_pct < MOVE_THRESHOLD:
                                reasons.append(f"Move {intra_candle_move_pct:.2f}% < {MOVE_THRESHOLD}%")
                            if volume_ratio < VOLUME_THRESHOLD:
                                reasons.append(f"Volume {volume_ratio:.2f}x < {VOLUME_THRESHOLD}x")
                            if abs(momentum_pct) < MOMENTUM_THRESHOLD:
                                reasons.append(f"Momentum {momentum_pct:.2f}% < ±{MOMENTUM_THRESHOLD}%")

                            self.log_message(
                                f"⏸️ [EARLY-ENTRY-{selected_tf}m] Move detected but NOT entering\n"
                                f"  Reasons: {' | '.join(reasons)}",
                                True
                            )

                    if not intra_candle_entry_allowed:
                        self.log_message(
                            f"⏳ Waiting for new {selected_tf}m candle for {sym}",
                            True
                        )
                        continue

                self.last_bar_ts[sym] = current_bar_ts
                self.log_message(f"New 5m candle for {sym} at {current_bar_ts.isoformat()}", False)

                # Get LTP (prefer WebSocket)
                ltp = self.get_websocket_ltp(sym) or five_inds.get("close")
                if not ltp:
                    self.log_message(f"Could not get LTP for {sym}, skipping strategies", False)
                    continue

                om = self.order_managers[sym]

                # ─────────────────────────── STRATEGY SELECTION ───────────────────────────
                om.execute_unified_strategy(
                    ltp=ltp,
                    all_inds=fresh_indicators_all_tfs.get(sym, {}),
                    primary_tf=selected_tf
                )

                # Log AI state for monitoring
                ai_state = om.get_ai_state()
                if ai_state.get("last_ai_action"):
                    self.log_message(f"[AI-CPR] {sym} State: {ai_state}", True)

            # ─────────────────────────── OPTION STRATEGY EXECUTION ───────────────────────────
            # Execute SENSEX option buying strategy if enabled
            if self.USE_OPTION_STRATEGY and self.option_order_manager:
                try:
                    option_result = self.execute_option_strategy()
                    
                    # Log option trading status
                    opt_status = self.get_option_trading_status()
                    if opt_status.get("in_option_trade"):
                        self.log_message(
                            f"[OPT-LIVE] {opt_status.get('active_symbol')} | "
                            f"Type: {opt_status.get('option_type')} | "
                            f"Entry: ₹{opt_status.get('entry_price', 0):.2f} | "
                            f"Daily P&L: ₹{opt_status.get('daily_pnl', 0):.0f}",
                            False
                        )
                except Exception as e:
                    self.log_message(f"[OPT-STRATEGY] Execution error: {e}", False)

            #self.log_message("Cycle complete, sleeping for 2 seconds...", True)
            if hasattr(self, 'ai_predictor') and self.ai_predictor:
                if hasattr(self.ai_predictor, 'last_prediction') and self.ai_predictor.last_prediction:
                    last_pred = self.ai_predictor.last_prediction
                    self.log_message(
                        f"[AI-LAST] {last_pred.get('label', 'N/A')} "
                        f"(conf: {last_pred.get('confidence', 0):.2f}) "
                        f"@ {last_pred.get('timestamp', 'N/A')}",
                        True
                    )
            time.sleep(2)
            if not hasattr(self, '_dashboard_counter'):
                self._dashboard_counter = 0

            self._dashboard_counter += 1

            if self._dashboard_counter >= 5:
                try:
                    self.perf_tracker.generate_html_dashboard()
                except Exception as e:
                    self.log_message(f"[DASHBOARD] Error updating: {e}", True)

                self._dashboard_counter = 0

    def _check_trend_alignment_for_early_entry(self, inds):
        """
        Check if EMAs are aligned for early entry
        Returns: (aligned: bool, direction: str)
        """
        ema5 = self._f(inds.get("ema_5"))
        ema9 = self._f(inds.get("ema_9"))
        ema21 = self._f(inds.get("ema_21"))
        momentum_pct = self._f(inds.get("momentum_pct"), 0.0)

        if not all([ema5, ema9, ema21]):
            return False, None

        # Bullish alignment
        if ema5 > ema9 and ema9 > ema21 and momentum_pct > 0.3:
            return True, "BULLISH"

        # Bearish alignment
        elif ema5 < ema9 and ema9 < ema21 and momentum_pct < -0.3:
            return True, "BEARISH"

        return False, None

    # ENHANCED: Add method to update AI strategy configuration dynamically
    def update_ai_strategy_config(self, symbol=None, enabled=None, min_conf=None, gate_trades=None):
        """
        Update AI CPR strategy configuration dynamically
        """
        symbols_to_update = [symbol] if symbol else self.symbols

        for sym in symbols_to_update:
            if sym in self.order_managers:
                self.order_managers[sym].update_ai_config(
                    enabled=enabled,
                    min_conf=min_conf,
                    gate_trades=gate_trades
                )
                self.log_message(
                    f"[AI-CPR] Updated config for {sym}: enabled={enabled}, min_conf={min_conf}, gate_trades={gate_trades}",
                    False)

    # ENHANCED: Add method to get AI status for monitoring
    def get_ai_status(self, symbol=None):
        """
        Get AI CPR status for monitoring/dashboard
        """
        status = {}
        symbols_to_check = [symbol] if symbol else self.symbols

        for sym in symbols_to_check:
            if sym in self.order_managers:
                status[sym] = {
                    "ai_state": self.order_managers[sym].get_ai_state(),
                    "position": self.order_managers[sym].position.get("type", "FLAT"),
                    "ai_enabled": self.order_managers[sym].AI_CPR_ENABLED
                }
        return status

    # ENHANCED: Add method to switch between strategy modes
    def set_strategy_mode(self, trend_strategy=None, ai_cpr_strategy=None, combined_strategy=None):
        """
        Switch between different strategy modes
        """
        if trend_strategy is not None:
            self.USE_TREND_STRATEGY = trend_strategy
            self.log_message(f"[STRATEGY] Trend strategy {'enabled' if trend_strategy else 'disabled'}", False)

        if ai_cpr_strategy is not None:
            self.USE_AI_CPR_STRATEGY = ai_cpr_strategy
            # Update all order managers
            for sym, om in self.order_managers.items():
                om.AI_CPR_ENABLED = ai_cpr_strategy
            self.log_message(f"[STRATEGY] AI CPR strategy {'enabled' if ai_cpr_strategy else 'disabled'}", False)

        if combined_strategy is not None:
            self.USE_COMBINED_STRATEGY = combined_strategy
            self.log_message(f"[STRATEGY] Combined strategy {'enabled' if combined_strategy else 'disabled'}", False)

    # ENHANCED: Add method for AI model health check
    def check_ai_model_health(self):
        """
        Check if AI model is loaded and healthy
        """
        if not hasattr(self, 'ai_predictor') or not self.ai_predictor:
            return {"status": "error", "message": "AI predictor not initialized"}

        try:
            if self.ai_predictor.model is None:
                return {"status": "error", "message": "AI model not loaded"}

            return {
                "status": "healthy",
                "model_loaded": True,
                "message": "AI model loaded successfully - test prediction skipped for live trading"
                # "test_prediction": prediction,  # Commented for live
                # "test_confidence": confidence,  # Commented for live
                # "feature_shape": features.shape if features is not None else None  # Commented for live
            }

        except Exception as e:
            return {"status": "error", "message": f"AI model test failed: {str(e)}"}

    # ENHANCED: Modify the existing analyze_setup_score to include AI analysis
    def analyze_setup_score(self, tf):
        """Enhanced setup analysis with AI CPR integration"""
        inds = self.indicator_calculator.calculate_indicators(self.symbol, tf)
        if "error" in inds:
            return {
                "type": "NO_DATA",
                "score": 0,
                "summary": inds.get("error"),
                "ts": dt.datetime.now(IST).isoformat()
            }

        score = 0
        reasons = []

        # Existing trend analysis
        if inds.get("st21Trend") == 1:
            score += 2
            reasons.append("ST21 Bull")

        # NEW: AI CPR analysis
        cpr_analysis = inds.get("cpr_analysis", {})
        if cpr_analysis and "error" not in cpr_analysis:
            ai_label = cpr_analysis.get("ai_cpr_label")
            ai_confidence = cpr_analysis.get("ai_confidence", 0)
            ai_filter_pass = cpr_analysis.get("ai_filter_pass", False)

            if ai_label and ai_confidence > 0.6:
                if ai_label.upper() == "BUY" and ai_filter_pass:
                    score += 3
                    reasons.append(f"AI CPR BUY (conf: {ai_confidence:.2f})")
                elif ai_label.upper() == "SELL" and ai_filter_pass:
                    score -= 3
                    reasons.append(f"AI CPR SELL (conf: {ai_confidence:.2f})")
                elif ai_label.upper() in ["HOLD", "NEUTRAL"]:
                    score += 0
                    reasons.append(f"AI CPR HOLD (conf: {ai_confidence:.2f})")

        reco = ("STRONG_BUY" if score > 3.5 else
                "STRONG_SELL" if score < -3.5 else
                "NEUTRAL")

        return {
            "type": reco,
            "score": round(score, 2),
            "reason": ", ".join(reasons),
            "ts": dt.datetime.now(IST).isoformat(),
            "ai_analysis": cpr_analysis  # Include full AI analysis
        }

    # ── FS paths ────────────────────────────────────────────────────────────────

    def _setup_paths(self):
        self.data_paths = {}
        self.log_paths = {}
        for sym in self.symbols:
            sym_clean = sym.replace(':', '_')
            data_dir = os.path.join(os.getcwd(), "data_bot")
            log_dir = os.path.join(os.getcwd(), "logs_bot")
            report_dir = os.path.join(os.getcwd(), "reports_bot", sym_clean)
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(report_dir, exist_ok=True)
            self.data_paths[sym] = {
                'raw_api_log': os.path.join(data_dir, f"fyers_raw_{sym_clean}.json"),
                'om_state_path': os.path.join(data_dir, f"om_state_{sym_clean}.json"),
                'om_event_log': os.path.join(data_dir, f"om_events_{sym_clean}.json"),
                'pivot_json': os.path.join(data_dir, f"pivot_{sym_clean}.json"),
                'all_data_json': os.path.join(data_dir, f"all_data_{sym_clean}.json"),
                'trade_report_dir': report_dir
            }
            self.log_paths[sym] = {
                'output_log': os.path.join(log_dir, f"bot_log_{sym_clean}.txt"),
                'status_change': os.path.join(log_dir, f"status_change_{sym_clean}.txt")
            }

        # Main symbol aliases
        main = self.symbol
        self.raw_api_log = self.data_paths[main]['raw_api_log']
        self.om_state_path = self.data_paths[main]['om_state_path']
        self.om_event_log = self.data_paths[main]['om_event_log']
        self.pivot_json = self.data_paths[main]['pivot_json']
        self.all_data_json = self.data_paths[main]['all_data_json']
        self.output_log = self.log_paths[main]['output_log']
        self.status_change = self.log_paths[main]['status_change']

    def _initialize_symbol_files(self):
        for sym in self.symbols:
            # Data files
            for k, path in self.data_paths[sym].items():
                if k == "trade_report_dir":
                    os.makedirs(path, exist_ok=True)
                    continue
                if not os.path.exists(path):
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump({}, f)
            # Log files
            for _, path in self.log_paths[sym].items():
                if not os.path.exists(path):
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("")

    # ── Dynamic symbol tracking (for V3 option symbols) ──────────────────────

    def ensure_symbol_tracking(self, sym: str) -> None:
        """Ensure filesystem paths + empty files exist for a symbol.

        V1 bot initializes paths only for self.symbols at startup (usually FUT).
        In V3 option buying we dynamically trade CE/PE symbols; we must create
        data/log/report folders for those option symbols at runtime.
        """
        if not sym:
            return
        if not hasattr(self, "data_paths") or not hasattr(self, "log_paths"):
            self._setup_paths()

        if sym in self.data_paths and sym in self.log_paths:
            return

        sym_clean = sym.replace(":", "_")
        data_dir = os.path.join(os.getcwd(), "data_bot")
        log_dir = os.path.join(os.getcwd(), "logs_bot")
        report_dir = os.path.join(os.getcwd(), "reports_bot", sym_clean)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        self.data_paths[sym] = {
            "raw_api_log": os.path.join(data_dir, f"fyers_raw_{sym_clean}.json"),
            "om_state_path": os.path.join(data_dir, f"om_state_{sym_clean}.json"),
            "om_event_log": os.path.join(data_dir, f"om_events_{sym_clean}.json"),
            "pivot_json": os.path.join(data_dir, f"pivot_{sym_clean}.json"),
            "all_data_json": os.path.join(data_dir, f"all_data_{sym_clean}.json"),
            "trade_report_dir": report_dir,
        }
        self.log_paths[sym] = {
            "output_log": os.path.join(log_dir, f"bot_log_{sym_clean}.txt"),
            "status_change": os.path.join(log_dir, f"status_change_{sym_clean}.txt"),
        }

        # create empty files
        for k, path in self.data_paths[sym].items():
            if k == "trade_report_dir":
                os.makedirs(path, exist_ok=True)
                continue
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
        for _, path in self.log_paths[sym].items():
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("")

    def activate_symbol_aliases(self, sym: str) -> None:
        """Switch 'main symbol' aliases so log_message/write helpers go to sym."""
        self.ensure_symbol_tracking(sym)
        if sym not in self.data_paths:
            return
        self.symbol = sym
        self.symbol_clean = sym.replace(":", "_")
        self.raw_api_log = self.data_paths[sym]["raw_api_log"]
        self.om_state_path = self.data_paths[sym]["om_state_path"]
        self.om_event_log = self.data_paths[sym]["om_event_log"]
        self.pivot_json = self.data_paths[sym]["pivot_json"]
        self.all_data_json = self.data_paths[sym]["all_data_json"]
        self.output_log = self.log_paths[sym]["output_log"]
        self.status_change = self.log_paths[sym]["status_change"]

    def _append_json_event(self, path: str, event: Dict[str, Any]) -> None:
        """Append event to a JSON list log (creating list if needed)."""
        try:
            data = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        existing = json.load(f)
                        if isinstance(existing, list):
                            data = existing
                        elif isinstance(existing, dict) and existing:
                            # previous versions sometimes used dict; keep a history list
                            data = [existing]
                    except Exception:
                        data = []
            data.append(event)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def persist_v3_entry(self, option_symbol: str, payload: Dict[str, Any]) -> None:
        """Persist V3 entry details under the option symbol.

        Also writes a per-symbol all_data_<OPTION>.json snapshot (same pattern as FUT).
        """
        payload = dict(payload or {})
        payload.setdefault("ts", dt.datetime.now(self.IST).isoformat())
        payload.setdefault("event", "ENTRY")
        # capture FUT snapshot BEFORE switching aliases
        fut = None
        try:
            if getattr(self, 'symbols', None):
                fut = self.symbols[0]
        except Exception:
            fut = None
        fut_snapshot = {}
        try:
            if fut and hasattr(self, 'data_paths') and fut in self.data_paths:
                fut_snapshot = robust_load_json(self.data_paths[fut]['all_data_json'], self.log_message, default={})
        except Exception:
            fut_snapshot = {}

        # switch active paths to option symbol
        self.activate_symbol_aliases(option_symbol)

        # write state + events
        try:
            with open(self.om_state_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass
        self._append_json_event(self.om_event_log, payload)

        # write all_data snapshot for option symbol
        try:
            option_all_data = {
                'base_symbol': fut,
                'option_symbol': option_symbol,
                'ts': payload.get('ts'),
                'snapshot': fut_snapshot,
                'trade': payload,
            }
            with open(self.all_data_json, 'w', encoding='utf-8') as f:
                json.dump(option_all_data, f, indent=2)
        except Exception:
            pass

        self.log_message(f"[V3-PERSIST] ENTRY saved for {option_symbol}", False)

    def persist_v3_exit(self, option_symbol: str, payload: Dict[str, Any]) -> None:
        """Persist V3 exit details + append a simple trade report.

        Also updates all_data_<OPTION>.json with the exit payload.
        """
        self.activate_symbol_aliases(option_symbol)
        payload = dict(payload or {})
        payload.setdefault('ts', dt.datetime.now(self.IST).isoformat())
        payload.setdefault('event', 'EXIT')
        try:
            with open(self.om_state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass
        self._append_json_event(self.om_event_log, payload)

        # update all_data snapshot with exit
        try:
            existing = robust_load_json(self.all_data_json, self.log_message, default={})
            if not isinstance(existing, dict):
                existing = {}
            existing['exit'] = payload
            existing['ts_exit'] = payload.get('ts')
            with open(self.all_data_json, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2)
        except Exception:
            pass

        # Append a minimal CSV-like report in reports_bot/<sym>/trades.csv
        try:
            report_dir = self.data_paths[option_symbol]["trade_report_dir"]
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "trades.csv")
            header = "ts,symbol,side,qty,entry,exit,pnl_points,pnl_rupees,reason\n"
            if not os.path.exists(report_path):
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(header)
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{payload.get('ts','')},{option_symbol},BUY,{payload.get('qty','')},"
                    f"{payload.get('entry_price','')},{payload.get('exit_price','')},"
                    f"{payload.get('pnl_points','')},{payload.get('pnl_rupees','')},"
                    f"{str(payload.get('reason','')).replace(',', ';')}\n"
                )
        except Exception:
            pass
        self.log_message(f"[V3-PERSIST] EXIT saved for {option_symbol}", False)

    def _load_config(self, config_dir):
        for p in [os.path.join(os.getcwd(), config_dir), os.getcwd(), config_dir]:
            cid = os.path.join(p, "client_id.txt")
            tok = os.path.join(p, "access_token.txt")
            if os.path.exists(cid) and os.path.exists(tok):
                with open(cid) as f:   self.client_id = f.read().strip()
                with open(tok) as f:   self.access_token = f.read().strip()
                self.log_message(f"Loaded config from {p}", False)
                return
        raise FileNotFoundError("Missing client_id.txt or access_token.txt")

    # ── Websocket ───────────────────────────────────────────────────────────────

    def _setup_websocket(self):
        try:
            self.fyers_websocket = data_ws.FyersDataSocket(
                access_token=f"{self.client_id}:{self.access_token}",
                log_path="",
                litemode=False,
                write_to_file=False,
                reconnect=True,
                on_connect=self._on_websocket_open,
                on_close=self._on_websocket_close,
                on_error=self._on_websocket_error,
                on_message=self._on_websocket_message
            )
            self.websocket_thread = threading.Thread(target=self._start_websocket, daemon=True)
            self.websocket_thread.start()
            self.log_message("WebSocket initialized", False)
        except Exception as e:
            self.log_message(f"WebSocket setup failed: {e}", False)

    def _start_websocket(self):
        try:
            self.fyers_websocket.connect()
        except Exception as e:
            self.log_message(f"WebSocket connection failed: {e}", False)

    def _on_websocket_open(self):
        try:
            data_type = "SymbolUpdate"
            self.fyers_websocket.subscribe(symbols=self.symbols, data_type=data_type)
            self.fyers_websocket.keep_running()
            self.log_message(f"WebSocket subscribed to {self.symbols}", False)
        except Exception as e:
            self.log_message(f"WebSocket subscription failed: {e}", False)

    def add_websocket_symbol(self, symbol):
        try:
            self.fyers_websocket.subscribe(symbols=[symbol], data_type="SymbolUpdate")
            self.log_message(f"Added symbol to WebSocket: {symbol}", False)
        except Exception as e:
            self.log_message(f"Failed to add symbol {symbol}: {e}", False)

    def remove_websocket_symbol(self, symbol):
        try:
            self.fyers_websocket.unsubscribe(symbols=[symbol])
            self.log_message(f"Removed symbol from WebSocket: {symbol}", False)
        except Exception as e:
            self.log_message(f"Failed to remove symbol {symbol}: {e}", False)

    def _on_websocket_message(self, message):
        try:
            if isinstance(message, dict) and 'symbol' in message and 'ltp' in message:
                symbol = message['symbol']
                ltp = message['ltp']
                now = dt.datetime.now(self.IST)

                # Update LTP cache
                with self.websocket_lock:
                    self.websocket_ltp[symbol] = {
                        'ltp': ltp,
                        'timestamp': now.isoformat()
                    }

                # Process OHLC accumulation
                self._process_ohlc_data(message)

                if self.DEBUG:
                    last_print = self._last_print_time.get(symbol)
                    if last_print is None or (now - last_print).total_seconds() > 5:
                        #print(f"WebSocket LTP: {symbol} = {ltp}")
                        self._last_print_time[symbol] = now
        except Exception as e:
            self.log_message(f"WebSocket message error: {e}", False)


    def _process_ohlc_data(self, message):
        """Enhanced to handle multiple timeframes independently"""
        try:
            ms = message.get('exch_feed_time')
            if ms is None:
                return

            # Handle ms or sec epoch
            ts_sec = ms / 1000.0 if ms > 1e11 else float(ms)
            curr_time = dt.datetime.fromtimestamp(ts_sec, tz=self.IST)

            symbol = message['symbol']
            ltp = float(message['ltp'])

            # [OK] Process each timeframe independently
            for tf in self.ohlc_timeframes:
                minute_key = curr_time.replace(second=0, microsecond=0)

                # Increment counter on new minute
                if self.last_minute_seen.get(tf) != minute_key:
                    self.last_minute_seen[tf] = minute_key
                    self.timeframe_counters[tf] += 1

                # [OK] Close bar when timeframe boundary reached
                if self.timeframe_counters[tf] >= tf:
                    with self.ohlc_lock:
                        # Process all symbols for this timeframe
                        for sym in list(self.ohlc_data[tf].keys()):
                            try:
                                if not self.ohlc_data[tf][sym]:
                                    continue

                                high = max(self.ohlc_data[tf][sym])
                                low = min(self.ohlc_data[tf][sym])
                                open_price = self.ohlc_data[tf][sym][0]
                                close_price = self.ohlc_data[tf][sym][-1]

                                csv_dict = {
                                    'minute': curr_time.strftime("%Y-%m-%d %H:%M:00"),
                                    'symbol': str(sym),
                                    'open': float(open_price),
                                    'high': float(high),
                                    'low': float(low),
                                    'close': float(close_price),
                                    'timeframe': str(tf)
                                }

                                self.csv_data[tf].setdefault(sym, []).append(csv_dict)

                                # [OK] Save to separate CSV file per timeframe
                                self._save_ohlc_csv(sym, csv_dict, timeframe=tf)

                                self.log_message(
                                    f"[OHLC-{tf}m] Updated CSV for {sym}: "
                                    f"O:{open_price:.2f} H:{high:.2f} L:{low:.2f} C:{close_price:.2f}",
                                    True
                                )
                            except Exception as e:
                                self.log_message(f"[OHLC-{tf}m] Error for {sym}: {e}", False)

                        # Clear data for this timeframe
                        self.ohlc_data[tf] = {}

                    # Reset counter for this timeframe
                    self.timeframe_counters[tf] = 0
                else:
                    # [OK] Append LTP to this timeframe's buffer
                    with self.ohlc_lock:
                        self.ohlc_data[tf].setdefault(symbol, []).append(ltp)

        except Exception as e:
            self.log_message(f"[OHLC] Processing error: {e}", False)
            import traceback
            self.log_message(f"[OHLC] Traceback: {traceback.format_exc()}", True)

    def _save_ohlc_csv(self, symbol, csv_dict, timeframe=5):
        """Enhanced with proper timeframe in filename"""
        try:
            # [OK] Include timeframe in filename
            csv_filename = f'{symbol.replace(":", "_")}_websocket_ohlc_{timeframe}min.csv'
            csv_path = os.path.join(os.getcwd(), "data_bot", csv_filename)

            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            file_exists = os.path.isfile(csv_path)

            # [OK] Ensure timeframe is in the dict
            csv_dict['timeframe'] = str(timeframe)

            with open(csv_path, 'a', newline='') as f:
                field_names = ['minute', 'symbol', 'open', 'high', 'low', 'close', 'timeframe']
                writer = DictWriter(f, fieldnames=field_names)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(csv_dict)

            #self.log_message(f"[OHLC-CSV] [OK] Saved to {csv_filename}", True)

        except Exception as e:
            self.log_message(f"[OHLC-CSV] [ERROR] Save error for {symbol}: {e}", False)

    def _on_websocket_error(self, message):
        self.log_message(f"WebSocket error: {message}", False)

    def _on_websocket_close(self, message):
        self.log_message(f"WebSocket closed: {message}", False)

    def get_websocket_ltp(self, symbol, timeout=5):
        """Get LTP from WebSocket cache only (no REST fallback)."""
        try:
            with self.websocket_lock:
                if symbol in self.websocket_ltp:
                    data = self.websocket_ltp[symbol]
                    if data['timestamp']:
                        ts = dt.datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                        if (dt.datetime.now(IST) - ts).total_seconds() <= timeout:
                            return data['ltp']
            return None
        except Exception as e:
            self.log_message(f"WebSocket LTP error for {symbol}: {e}", False)
            return None

    # ── Logging helpers ─────────────────────────────────────────────────────────

    def log_message(self, msg, debug_only=False):
        if debug_only and not self.DEBUG:
            return
        ts = dt.datetime.now(self.IST).strftime("%Y-%m-%d %H:%M:%S %Z")
        entry = f"{ts} - {msg}"
        try:
            print(entry)
        except UnicodeEncodeError:
            # Fallback for Windows terminal
            clean_entry = "".join(c for c in entry if ord(c) < 128)
            print(clean_entry)
        try:
            with open(self.output_log, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except:
            pass

    def log_status_change(self, tf, prev, curr, inds):
        ts = dt.datetime.now(self.IST).strftime("%Y-%m-%d %H:%M:%S %Z")
        lines = [f"[{ts}] TF={tf}"]
        for k in curr:
            if prev.get(k) != curr[k]:
                lines.append(f"  {k}: {prev.get(k)} -> {curr[k]}")
        if len(lines) > 1:
            try:
                with open(self.status_change, "a") as f:
                    f.write("\\n".join(lines) + "\\n\\n")
                self.log_message(f"Status change {tf}", True)
            except:
                pass

    # ── History fetch (with future-candle filter) ──────────────────────────────

    def fetch_ohlc(self, symbol, tf, days):
        try:
            to = dt.date.today()
            frm = to - dt.timedelta(days=int(days))
            resp = call_with_rate_limit_retry(
                self.fyers_sdk_instance.history,
                data={
                    "symbol": symbol,
                    "resolution": str(tf),
                    "date_format": "1",
                    "range_from": frm.strftime("%Y-%m-%d"),
                    "range_to": to.strftime("%Y-%m-%d"),
                    "cont_flag": "1"
                }
            )
            if resp and resp.get("s") == "ok" and resp.get("candles"):
                df = pd.DataFrame(resp["candles"], columns=["Ts", "Open", "High", "Low", "Close", "Volume"])
                df["Timestamp"] = pd.to_datetime(df["Ts"], unit="s", utc=True).dt.tz_convert(IST)
                df.set_index("Timestamp", inplace=True)

                now_ts = pd.Timestamp.now(tz=IST)
                original_rows = len(df)
                df = df[df.index <= now_ts]
                filtered_rows = len(df)
                if original_rows > filtered_rows:
                    self.log_message(
                        f"[DATA FIX] Removed {original_rows - filtered_rows} future-dated candles for {symbol}.", False)

                if df.empty:
                    self.log_message(f"[WARN] No historical data left for {symbol} after filtering future dates.",
                                     False)
                    return pd.DataFrame()

                for c in ["Open", "High", "Low", "Close", "Volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                return df.sort_index()

            self.log_message(f"OHLC error {symbol} {tf}: {resp.get('message', '') if resp else 'No response'}", False)
        except Exception as e:
            self.log_message(f"OHLC exception {symbol} {tf}: {e}", False)
        return pd.DataFrame()

    # ── Pivots & ADX snapshots (optional dashboard) ────────────────────────────

    def process_pivots(self):
        """
        Enhanced pivot processing with validation and AI context
        Combines robust validation from version 2 with AI analysis from version 1
        """
        self.log_message("[CPR] Calculating CPR pivot levels...", False)

        # ============================================
        # STEP 1: FETCH DAILY OHLC DATA
        # ============================================
        # Use 60 days for better historical context (from version 2)
        df = self.fetch_ohlc(self.symbol, "D", 60)

        if df.empty:
            self.log_message("[ERROR] No daily OHLC data available for pivot calculation!", False)
            return {}

        if len(df) < 2:
            self.log_message(f"[ERROR] Insufficient daily data: {len(df)} rows (need at least 2)", False)
            return {}

        # ============================================
        # STEP 2: CALCULATE PIVOT POINTS
        # ============================================
        raw = self.indicator_calculator.calculate_pivot_points(df)

        if not raw or not isinstance(raw, dict):
            self.log_message("[ERROR] Pivot calculation returned empty/invalid data!", False)
            return {}

        # ============================================
        # STEP 3: VALIDATION (from version 2)
        # ============================================
        required_keys = ["TC", "BC", "PP", "R1", "R2", "R3", "S1", "S2", "S3"]
        missing_keys = [k for k in required_keys if k not in raw or raw[k] is None]

        if missing_keys:
            self.log_message(
                f"[ERROR] Pivot calculation incomplete! Missing: {', '.join(missing_keys)}",
                False
            )
            self.log_message(f"   Raw data: {raw}", True)
            return {}

        # [OK] VALIDATION: Check TC > BC (sanity check)
        if raw["TC"] < raw["BC"]:
            self.log_message(
                f"[WARNING] WARNING: TC ({raw['TC']}) < BC ({raw['BC']}) - Inverted CPR!",
                False
            )

        # ============================================
        # STEP 4: CONVERT TO SERIALIZABLE FORMAT
        # ============================================
        clean = convert_dict_to_serializable(raw)

        # ============================================
        # STEP 5: ADD AI CONTEXT (from version 1)
        # ============================================
        if hasattr(self, 'ai_predictor') and self.ai_predictor:
            try:
                self.log_message("[AI-CPR] Adding AI context to pivots...", True)

                # Get current 5-min indicators for AI analysis
                current_inds = self.indicator_calculator.calculate_indicators(
                    self.symbol,
                    "5",
                    pivot_data=raw
                )

                if "error" not in current_inds:
                    # Run AI analysis on current market state
                    ai_analysis = analyze_cpr_strategy(
                        current_inds,
                        raw,
                        self.ai_predictor
                    )

                    # Add AI context to pivot data
                    clean["ai_context"] = {
                        "timestamp": dt.datetime.now(IST).isoformat(),
                        "analysis": ai_analysis,
                        "ai_signal": ai_analysis.get("trade_strategy", "None"),
                        "ai_confidence": ai_analysis.get("ai_confidence", 0.0),
                        "ai_label": ai_analysis.get("ai_cpr_label", "None")
                    }

                    self.log_message(
                        f"[OK] AI Context added: {ai_analysis.get('trade_strategy', 'None')} "
                        f"(conf: {ai_analysis.get('ai_confidence', 0.0):.2f})",
                        False
                    )
                else:
                    self.log_message(
                        f"[WARNING] Cannot add AI context - indicator error: {current_inds.get('error')}",
                        True
                    )

            except Exception as e:
                self.log_message(f"[AI-CPR] Pivot context error: {e}", True)
                # Don't fail pivot calculation if AI context fails
                import traceback
                self.log_message(f"[AI-CPR] Traceback: {traceback.format_exc()}", True)

        # ============================================
        # STEP 6: PREPARE FINAL DATA STRUCTURE
        # ============================================
        data = {
            self.symbol: {
                "ts": dt.datetime.now(self.IST).isoformat(),
                **clean
            }
        }

        # ============================================
        # OPTIONAL: STORE AI HISTORY FOR ANALYSIS
        # ============================================
        if "ai_context" in clean:
            try:
                # Append to AI history file for later analysis
                ai_history_path = self.pivot_json.replace("pivot_", "pivot_ai_history_")
                history = robust_load_json(ai_history_path, self.log_message, default=[])

                history.append({
                    "date": dt.datetime.now(self.IST).date().isoformat(),
                    "timestamp": dt.datetime.now(self.IST).isoformat(),
                    "symbol": self.symbol,
                    "pivots": {
                        "TC": raw["TC"],
                        "BC": raw["BC"],
                        "PP": raw["PP"]
                    },
                    "ai_prediction": clean["ai_context"]
                })

                # Keep only last 30 days
                history = history[-30:]

                robust_save_json(history, ai_history_path, self.log_message)
                self.log_message(f"   AI history updated ({len(history)} days)", True)

            except Exception as e:
                self.log_message(f"[AI-HISTORY] Failed to store: {e}", True)

        # ============================================
        # STEP 7: SAVE TO JSON
        # ============================================
        robust_save_json(data, self.pivot_json, self.log_message)

        # ============================================
        # STEP 8: LOG SUCCESS WITH DETAILS
        # ============================================
        self.log_message(
            f"[OK] CPR Pivots calculated successfully:\n"
            f"   TC={raw['TC']:.2f}, BC={raw['BC']:.2f}, PP={raw['PP']:.2f}\n"
            f"   R1={raw['R1']:.2f}, R2={raw['R2']:.2f}, R3={raw['R3']:.2f}\n"
            f"   S1={raw['S1']:.2f}, S2={raw['S2']:.2f}, S3={raw['S3']:.2f}\n"
            f"   Virgin CPR: {raw.get('virgin_cpr', False)}",
            False
        )

        # Log AI context if available
        if "ai_context" in clean:
            ai_ctx = clean["ai_context"]
            self.log_message(
                f"   AI Signal: {ai_ctx.get('ai_signal', 'None')} "
                f"(Label: {ai_ctx.get('ai_label', 'None')}, "
                f"Conf: {ai_ctx.get('ai_confidence', 0.0):.2f})",
                False
            )

        return data

    def initialize_pivots(self):
        """
        Initialize CPR pivot levels with validation
        Uses the enhanced merged process_pivots() method
        """
        piv_data = robust_load_json(self.pivot_json, self.log_message, default=None, debug_only=False)

        # [OK] VALIDATION: Check if pivots exist AND are valid
        needs_recalc = False

        if not isinstance(piv_data, dict) or self.symbol not in piv_data:
            self.log_message("[WARNING] No pivot data found - generating new pivots", False)
            needs_recalc = True
        else:
            # Check if all required CPR levels exist
            required_keys = ["TC", "BC", "R1", "R2", "R3", "S1", "S2", "S3"]
            symbol_pivots = piv_data.get(self.symbol, {})
            missing_keys = [k for k in required_keys if k not in symbol_pivots or symbol_pivots[k] is None]

            if missing_keys:
                self.log_message(f"[WARNING] Incomplete pivots (missing: {', '.join(missing_keys)}) - recalculating", False)
                needs_recalc = True
            else:
                # Check if pivots are stale (older than 24 hours)
                pivot_ts = symbol_pivots.get("ts", "1970-01-01")
                try:
                    pivot_age = pd.to_datetime(pivot_ts)
                    hours_old = (pd.Timestamp.now(tz=self.IST) - pivot_age).total_seconds() / 3600

                    if hours_old > 24:
                        self.log_message(f"[WARNING] Pivots are stale ({hours_old:.1f}h old) - recalculating", False)
                        needs_recalc = True
                    else:
                        self.log_message(f"[OK] Using existing pivots (age: {hours_old:.1f}h)", False)

                        # [OK] BONUS: Log AI context if available
                        if "ai_context" in symbol_pivots:
                            ai_ctx = symbol_pivots["ai_context"]
                            self.log_message(
                                f"   AI Context: {ai_ctx.get('ai_signal', 'None')} "
                                f"({ai_ctx.get('ai_confidence', 0.0):.2f})",
                                True
                            )

                except Exception as e:
                    self.log_message(f"[WARNING] Pivot age check failed: {e} - recalculating", False)
                    needs_recalc = True

        # Recalculate if needed
        if needs_recalc:
            self.process_pivots()  # [OK] Now uses merged enhanced version

            # [OK] VERIFY calculation succeeded
            piv_data = robust_load_json(self.pivot_json, self.log_message, default={})
            if self.symbol in piv_data:
                symbol_pivots = piv_data[self.symbol]
                self.log_message(
                    f"[OK] Pivots calculated successfully:\n"
                    f"   TC={symbol_pivots.get('TC')}, BC={symbol_pivots.get('BC')}\n"
                    f"   R1={symbol_pivots.get('R1')}, S1={symbol_pivots.get('S1')}",
                    False
                )
            else:
                self.log_message("[ERROR] CRITICAL: Pivot calculation failed!", False)

    def fetch_and_store_adx(self):
        df = fetchOHLC1(self.symbol, interval="5", duration=3)
        if df is not None and not df.empty:
            bundle = adx_efi_mom_trade_signal(df, self.symbol)
            payload = {
                "ts": dt.datetime.now(IST).isoformat(),
                "sig": bundle[0],
                "ADX": bundle[1],
                "DI+": bundle[2],
                "DI-": bundle[3],
                "Mom": bundle[4],
                "EFI": bundle[5],
                "RSI": bundle[6]
            }
            all_data = robust_load_json(self.all_data_json, self.log_message, default={})
            all_data["ADX"] = convert_dict_to_serializable(payload)
            robust_save_json(all_data, self.all_data_json, self.log_message)

    def analyze_setup_score(self, tf):
        inds = self.indicator_calculator.calculate_indicators(self.symbol, tf)
        if "error" in inds:
            return {
                "type": "NO_DATA",
                "score": 0,
                "summary": inds.get("error"),
                "ts": dt.datetime.now(IST).isoformat()
            }
        score = 0
        reasons = []
        if last(inds.get("st21Trend")) == 1:
            score += 2
            reasons.append("ST21 Bull")
        reco = ("STRONG_BUY" if score > 3.5 else
                "STRONG_SELL" if score < -3.5 else
                "NEUTRAL")
        return {
            "type": reco,
            "score": round(score, 2),
            "reason": ", ".join(reasons),
            "ts": dt.datetime.now(IST).isoformat()
        }

    def _ai_place_order(self, side, sym, indicators):
        print("DEBUG: Entered _ai_place_order")
        om = self.order_managers[sym]
        trade_qty = 1

        ema5 = last(indicators.get("ema_5"))
        ema21 = last(indicators.get("ema_21"))
        ltp = last(indicators.get("close"))
        trend = indicators.get("trend")  # ← optional: higher timeframe trend flag
        last_pos = om.position.get("last_type", None)
        current_pos = om.position.get("type", "FLAT")

        # [OK] Guard: If indicators are missing, skip
        if ema5 is None or ema21 is None or ltp is None:
            self.log_message(
                f"[AI-CPR] Skipping order for {sym}: indicators missing (ema5={ema5}, ema21={ema21}, ltp={ltp})", True
            )
            return "SKIP - Indicators missing"

        # [OK] Normalize side
        if side in ["Bullish", "BUY"]:
            side = "BUY"
        elif side in ["Bearish", "SELL"]:
            side = "SELL"
        else:
            if current_pos != "FLAT":
                return "HOLD"
            return "FLAT - No trade"

        # [OK] Optional Trend Filter (skip entries against trend)
        if trend:
            if side == "BUY" and trend != "UP":
                self.log_message(f"[AI-CPR] Skipping BUY for {sym}: Trend is {trend}", True)
                return "SKIP - Against trend"
            if side == "SELL" and trend != "DOWN":
                self.log_message(f"[AI-CPR] Skipping SELL for {sym}: Trend is {trend}", True)
                return "SKIP - Against trend"

        # [OK] New Entry
        if current_pos == "FLAT":
            if side == "BUY" and ema5 > ema21 and last_pos != "BUY":
                om.ai_buy(sym, trade_qty)
                return "BUY"
            elif side == "SELL" and ema5 < ema21 and last_pos != "SELL":
                om.ai_sell(sym, trade_qty)
                return "SELL"
            return "FLAT - No trade"

        # [OK] Exit on direction change
        if (current_pos == "BUY" and side == "SELL") or (current_pos == "SELL" and side == "BUY"):
            self.log_message(f"[AI-CPR] EXIT: Direction change for {sym}", True)
            om.ai_exit_all(sym)
            return "EXIT on direction change"

        # [OK] Hold if already in same side
        if (current_pos == side) or (current_pos == "FLAT" and last_pos == side):
            self.log_message(f"[AI-CPR] HOLD: Already in {side} or just exited {side} for {sym}", True)
            return "HOLD"

        return "HOLD"

    def _check_recent_trend(self, df, i, recent_candles):
        uptrend, downtrend = True, True
        for k in range(recent_candles):
            idx = i - k
            if not (df['Close'][idx] > df['Open'][idx]):
                uptrend = False
            if not (df['Close'][idx] < df['Open'][idx]):
                downtrend = False
            if k < recent_candles - 1:
                prev_idx = i - (k + 1)
                if not (df['High'][idx] > df['High'][prev_idx] and df['Low'][idx] > df['Low'][prev_idx]):
                    uptrend = False
                if not (df['High'][idx] < df['High'][prev_idx] and df['Low'][idx] < df['Low'][prev_idx]):
                    downtrend = False
        return uptrend, downtrend

    def _check_preceding_trend(self, df, i, recent_candles, preceding_candles):
        uptrend, downtrend = True, True
        start_idx = i - recent_candles
        for k in range(preceding_candles):
            idx = start_idx - k
            if not (df['Close'][idx] > df['Open'][idx]):
                downtrend = False
            if not (df['Close'][idx] < df['Open'][idx]):
                uptrend = False
            if k < preceding_candles - 1:
                prev_idx = start_idx - (k + 1)
                if not (df['High'][idx] < df['High'][prev_idx] and df['Low'][idx] < df['Low'][prev_idx]):
                    uptrend = False
                if not (df['High'][idx] > df['High'][prev_idx] and df['Low'][idx] > df['Low'][prev_idx]):
                    downtrend = False
        return uptrend, downtrend

    # (Optional) monitoring helpers
    def _log_ohlc_status(self):
        try:
            with self.ohlc_lock:
                status = {
                    'timeframe': self.timeframe,
                    'counter': self.timeframe_counter,
                    'active_symbols': list(self.ohlc_data.keys()),
                    'symbol_counts': {symbol: len(data) for symbol, data in self.ohlc_data.items()},
                    'csv_records': {symbol: len(data) for symbol, data in self.csv_data.items()}
                }
            self.log_message(f"OHLC Status: {status}", True)
        except Exception as e:
            self.log_message(f"OHLC status error: {e}", False)

    def get_websocket_status(self):
        try:
            with self.websocket_lock:
                ltp_status = {
                    symbol: {
                        'ltp': data['ltp'],
                        'age_seconds': (dt.datetime.now(IST) - dt.datetime.fromisoformat(
                            data['timestamp'].replace('Z', '+00:00'))).total_seconds()
                    }
                    for symbol, data in self.websocket_ltp.items()
                }

            with self.ohlc_lock:
                ohlc_status = {
                    'timeframe': self.timeframe,
                    'counter': self.timeframe_counter,
                    'active_symbols': list(self.ohlc_data.keys()),
                    'symbol_counts': {symbol: len(data) for symbol, data in self.ohlc_data.items()},
                    'csv_records': {symbol: len(data) for symbol, data in self.csv_data.items()}
                }

            return {
                'ltp_data': ltp_status,
                'ohlc_data': ohlc_status,
                'websocket_active': hasattr(self, 'fyers_websocket') and self.fyers_websocket is not None
            }
        except Exception as e:
            self.log_message(f"WebSocket status error: {e}", False)
            return {'error': str(e)}


# ───────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ───────────────────────────────────────────────────────────────────────────────




# =============================================================================
# REFACTORED OPTION BUYING ARCHITECTURE (V2 - NON-BREAKING ADDITION)
# =============================================================================
# This section adds a clean, end-to-end option buying workflow aligned to:
#
# TradingBotV2.run()
#   ├── run_underlying_option_analysis()
#   │     └── GenericGreeksOptionBuyerV2.analyze_and_get_signal()
#   │            - Greeks guardrails + scoring
#   │            - PCR gating
#   │            - VIX gating (optional per underlying)
#   │            - Momentum confirmation (EMA/trend from MarketDataAPIV2)
#   │          -> OptionBuySignalV2 (CE/PE + params)
#   └── OptionOrderManagerV2 (bridge) -> OrderManagerV2 (entry/SL/TP/trailing/exit)
#
# NOTE
# - All class names are suffixed with V2 to avoid clashing with your existing bot.
# - Plug-in points:
#     * BrokerAPIV2: adapt your existing FyersService (or broker) to this interface.
#     * MarketDataAPIV2: adapt your existing indicator + option chain fetch logic.
# - This section is self-contained and does not modify your legacy flow.



from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple, List, Iterable
import math as _math
import time as _time
import traceback as _traceback
import logging as _logging
import datetime as _dt
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    _IST_V3 = _ZoneInfo("Asia/Kolkata")
except Exception:
    # No zoneinfo available: fall back to a fixed +05:30 offset.
    _IST_V3 = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
_logger_v2 = _logging.getLogger("optionbuying.v2")

@dataclass
class ResolvedSymbolsV3:
    underlying_key: str
    analysis_symbol: str
    option_chain_symbol: str

class SymbolRouterV3:
    @staticmethod
    def resolve(symbol: str) -> Optional[ResolvedSymbolsV3]:
        s = (symbol or "").upper()
        if "SENSEX" in s:
            return ResolvedSymbolsV3("SENSEX", symbol, symbol)
        # MCX Futures that map to option chain on the same base symbol
        if "NATGASMINI" in s:
            return ResolvedSymbolsV3("NATGASMINI", symbol, symbol)
        if "CRUDEOILM" in s:
            return ResolvedSymbolsV3("CRUDEOILM", symbol, symbol)
        if "CRUDEOIL" in s:
            return ResolvedSymbolsV3("CRUDEOIL", symbol, symbol)
        return None


# ---------------------------------------------------------------------------
# V2 Data Models
# ---------------------------------------------------------------------------

@dataclass
class GreeksV2:
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float


@dataclass
class OptionCandidateV2:
    symbol: str
    option_type: str  # "CE" or "PE"
    strike: float
    expiry: str
    ltp: float
    bid: float
    ask: float
    oi: float
    volume: float
    greeks: GreeksV2
    pcr: float  # strike-level PCR
    expectancy: float = 0.0


@dataclass
class OptionBuySignalV2:
    underlying: str
    option_symbol: str
    option_type: str  # "CE" / "PE"
    strike: float
    expiry: str
    entry_price: float
    greeks: GreeksV2
    pcr: float
    vix: float
    momentum: str  # "BULLISH"/"BEARISH"
    score: float
    expectancy: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnderlyingConfigV2:
    key: str
    point_value: float
    default_sl_pct: float
    default_tp_pct: float

    vix_symbol: Optional[str] = None
    max_vix: Optional[float] = None

    max_trades_per_day: int = 3
    daily_loss_limit: float = 2000.0

    ema_fast: int = 9
    ema_slow: int = 21
    momentum_tf_min: int = 5
    momentum_lookback_min: int = 240

    delta_target_abs: float = 0.55
    delta_min_abs: float = 0.45
    delta_max_abs: float = 0.65
    pcr_bullish_min: float = 1.05
    pcr_bearish_max: float = 0.95
    max_spread_pct: float = 0.015
    strike_step: float = 100.0
    lot_size: int = 1

    # --- Option affordability + quality gates (defaults) ---
    # Premium filter helps avoid very expensive deep ITM buys and ultra-cheap illiquid far OTM.
    min_premium: float = 5.0
    max_premium: float = 25.0

    # High-confidence gate: if enabled, the bot will SKIP trades when conditions are weak.
    require_high_confidence: bool = True
    min_adx: float = 20.0
    min_votes: int = 3

    # Order sizing (for MCX options FYERS expects qty in lots; keep lots=1 unless you scale).
    order_lots: int = 1
    min_expectancy: float = 0.5
    time_stop_min: int = 12


# ---------------------------------------------------------------------------
# V2 Broker / Market Data Abstractions (ADAPT YOUR EXISTING CODE)
# ---------------------------------------------------------------------------

class BrokerAPIV2:
    """
    Minimal broker adapter interface.
    Implement by wrapping your FyersService / broker SDK.
    """
    def get_ltp(self, symbol: str) -> float:
        raise NotImplementedError

    def place_market_order(self, symbol: str, side: str, qty: int) -> str:
        raise NotImplementedError

    def exit_position_market(self, symbol: str, qty: int) -> str:
        raise NotImplementedError

    def get_lot_size(self, symbol: str) -> int:
        raise NotImplementedError


class MarketDataAPIV2:
    @property
    def bot(self) -> Any:
        raise NotImplementedError

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Returns raw quote dictionary with lp, v, oi, etc."""
        raise NotImplementedError

    def get_history_df(self, symbol: str, tf_min: int, lookback_min: int) -> Any:
        """Returns OHLC DataFrame."""
        raise NotImplementedError

    def get_option_chain_raw(self, symbol: str, strikecount: str = "") -> Dict[str, Any]:
        """Returns raw option chain dictionary."""
        raise NotImplementedError

    def get_pcr(self, symbol: str) -> float:
        """Returns the Put-Call Ratio for the underlying."""
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Fyers Adapters for V2/V3
# ---------------------------------------------------------------------------

class FyersMarketDataAdapterV2(MarketDataAPIV2):
    def __init__(self, fyers, bot=None) -> None:
        self.fyers = fyers
        self._bot = bot
        self._quote_cache: Dict[str, Tuple[float, float]] = {}  # {symbol: (ltp, timestamp)}

    @property
    def bot(self) -> Any:
        return self._bot

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        # Simple short-lived cache (1 second) to prevent hammering during sync/analysis/execution
        import time as _t
        now = _t.time()
        if symbol in self._quote_cache:
            ltp, ts = self._quote_cache[symbol]
            if now - ts < 1.0:
                return {"lp": ltp, "ltp": ltp}
                
        resp = self.fyers.quotes(data={"symbols": symbol})
        if resp.get("s") == "ok" and resp.get("d"):
            v = resp["d"][0].get("v", {}) or {}
            ltp = float(v.get("lp") or v.get("ltp") or 0.0)
            if ltp > 0:
                self._quote_cache[symbol] = (ltp, now)
            return v
        return {}
        
    def get_ltp(self, symbol: str) -> float:
        q = self.get_quote(symbol)
        return float(q.get("lp") or q.get("ltp") or 0.0)

    def get_history_df(self, symbol: str, tf_min: int, lookback_min: int):
        import pandas as pd
        now = _dt.datetime.now(_IST_V3)
        start = now - _dt.timedelta(minutes=int(lookback_min))
        data = {
            "symbol": symbol,
            "resolution": str(int(tf_min)),
            "date_format": "0",
            "range_from": str(int(start.timestamp())),
            "range_to": str(int(now.timestamp())),
            "cont_flag": "1",
        }
        resp = self.fyers.history(data=data)
        if resp.get("s") != "ok" or "candles" not in resp:
            return pd.DataFrame()

        df = pd.DataFrame(resp["candles"], columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(_IST_V3)
        df.set_index("dt", inplace=True)
        return df

    def get_option_chain_raw(self, symbol: str, strikecount: str = "") -> Dict[str, Any]:
        data = {"symbol": symbol, "strikecount": strikecount, "timestamp": ""}
        return self.fyers.optionchain(data=data)

    def get_pcr(self, symbol: str) -> float:
        """Fetch index-level PCR if available, or approximate from chain."""
        try:
            resolved = SymbolRouterV3.resolve(symbol)
            if not resolved: return 1.0
            
            raw_chain = self.get_option_chain_raw(resolved.option_chain_symbol, strikecount="1")
            if raw_chain.get("s") == "ok" and "data" in raw_chain:
                rows = raw_chain["data"].get("optionsChain") or []
                if rows:
                    # Most rows in Fyers chain carry the index-level PCR
                    return float(rows[0].get("pcr") or 1.0)
        except Exception:
            pass
        return 1.0



class MarginShortfallError(RuntimeError):
    """Raised when broker rejects order due to insufficient margin."""
    pass

class FyersBrokerAdapterV2(BrokerAPIV2):
    def __init__(self, fyers) -> None:
        self.fyers = fyers

    def get_ltp(self, symbol: str) -> float:
        q = self.fyers.quotes(data={"symbols": symbol})
        if q.get("s") == "ok" and q.get("d"):
            v = q["d"][0].get("v", {}) or {}
            return float(v.get("lp") or v.get("ltp") or 0.0)
        return 0.0

    def _get_best_prices(self, symbol: str) -> tuple[float, float, float]:
        """Returns (ltp, best_bid, best_ask) if available."""
        q = self.fyers.quotes(data={"symbols": symbol})
        if q.get("s") == "ok" and q.get("d"):
            v = q["d"][0].get("v", {}) or {}
            ltp = float(v.get("lp") or v.get("ltp") or 0.0)
            bid = float(v.get("bp") or v.get("bid") or 0.0)  # best bid
            ask = float(v.get("ap") or v.get("ask") or 0.0)  # best ask
            return ltp, bid, ask
        return 0.0, 0.0, 0.0

    @staticmethod
    def _round_to_tick(price: float, tick: float = 0.05, *, up: bool = True) -> float:
        if price <= 0:
            return 0.0
        steps = price / tick
        if up:
            steps = int(steps + 0.999999)
        else:
            steps = int(steps)
        return round(steps * tick, 2)

    def place_market_order(self, symbol: str, side: str, qty: int) -> str:
        """Place an order via FYERS.

        Notes:
        - MCX & BSE options generally need LIMIT orders (market may be rejected or set with high protection buffer).
        - FYERS validates limitPrice strictly (must be >= 0.0025 for LIMIT).
        """
        symbol = (symbol or "").strip()
        side_u = (side or "").upper().strip()

        # Treat both MCX and BSE options as Limit to control price better
        is_opt_limit = (
            (symbol.startswith("MCX:") or symbol.startswith("BSE:")) and 
            (symbol.upper().endswith("CE") or symbol.upper().endswith("PE"))
        )

        order_type = 2  # 2 = MARKET, 1 = LIMIT (FYERS)
        limit_price: float = 0.0

        if is_opt_limit:
            # Auto-switch to LIMIT using best prices to avoid FYERS validation errors & protection buffers.
            ltp, bid, ask = self._get_best_prices(symbol)
            
            # For BUY, use Ask or LTP. For SELL, use Bid or LTP.
            # Add a small buffer to ensure fill (Marketable Limit)
            ref = ask if side_u == "BUY" else bid
            if not ref or ref <= 0:
                ref = ltp
            
            if ref > 0:
                if side_u == "BUY":
                    # Buy LIMIT at slightly higher to ensure fill (e.g. +0.5%) 
                    # OR just round to tick if aggressive. User wants "LTP value".
                    # Let's use exact ref (Ask) rounded up to tick to be safe but not over-pay.
                    limit_price = self._round_to_tick(ref, up=True)
                else:
                    limit_price = self._round_to_tick(ref, up=False)
                
                order_type = 1
            else:
                # Fallback to market if no price found
                order_type = 2

        # Enforce FYERS limitPrice rules
        if order_type == 1:
            if not limit_price or limit_price < 0.0025:
                # safe minimum tick for MCX/BSE options; keeps request valid
                limit_price = 0.05
        else:
            # market order: keep limitPrice 0
            limit_price = 0.0

        # Strict type validation to avoid "must be real number, not str"
        try:
            qty_int = int(qty)
            type_int = int(order_type)
            side_int = 1 if side_u == "BUY" else -1
            lim_float = float(limit_price)
        except Exception as e:
            raise ValueError(f"Type conversion failed in place_market_order: {e}")

        payload = {
            "symbol": str(symbol),
            "qty": qty_int,
            "type": type_int,
            "side": side_int,
            "productType": "MARGIN",
            "limitPrice": lim_float,
            "stopPrice": 0.0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
        }

        # Debug log to catch issues
        # print(f"[DEBUG] Placing Order: {payload}") 

        resp = self.fyers.place_order(data=payload)
        if resp.get("s") == "ok":
            return str(resp.get("id") or resp.get("orderId") or "OK")

        code = resp.get("code")
        msg = str(resp.get("message") or "")
        if code in (-99, -100) or "Margin Shortfall" in msg or "margin" in msg.lower():
            raise MarginShortfallError(f"Order rejected (margin): {resp}")

        raise RuntimeError(f"place_market_order failed: {resp}")

    def exit_position_market(self, symbol: str, qty: int) -> str:
        return self.place_market_order(symbol, side="SELL", qty=int(qty))

    def get_lot_size(self, symbol: str) -> int:
        q = self.fyers.quotes(data={"symbols": symbol})
        if q.get("s") == "ok" and q.get("d"):
            v = q["d"][0].get("v", {}) or {}
            lot = v.get("lot_size") or v.get("lotSize") or v.get("min_qty")
            if lot:
                return int(lot)
        return 1

# ---------------------------------------------------------------------------
# V2 Unified Stop Loss Manager
# ---------------------------------------------------------------------------

# =============================================================================
# TRADE JOURNAL
# =============================================================================
class TradeJournalV5:
    def __init__(self, path="trade_journal_v3_5.csv"):
        self.path = path
        if not os.path.exists(path):
            try:
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "time","symbol","type","strike","entry","exit","pnl","delta","gamma","theta","expectancy","reason"
                    ])
            except Exception as e:
                print(f"[WARNING] [JOURNAL] Could not create {path}: {e}")

    def log_trade(self, symbol, opt_type, strike, entry, exit_price, delta, gamma, theta, expectancy, reason):
        try:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow([
                    dt.datetime.now(IST).isoformat(),
                    symbol, opt_type, strike, entry, exit_price,
                    round(exit_price - entry, 2),
                    round(delta, 4), round(gamma, 6), round(theta, 4),
                    round(expectancy, 2), reason
                ])
        except Exception as e:
            print(f"[WARNING] [JOURNAL] Log failed: {e}")

# ==========================
# TRAILING SL + PROFIT LOCK (Point Based)
# ==========================
class TrailingStopManagerV3:
    def __init__(self, sl_points: float):
        self.sl_points = sl_points
        self.entry_price = None
        self.max_price = None
        self.sl = None

    def init(self, entry_price: float):
        self.entry_price = entry_price
        self.max_price = entry_price
        self.sl = entry_price - self.sl_points

    def update(self, ltp: float) -> float:
        if self.max_price is None or ltp > self.max_price:
            self.max_price = ltp

        profit = self.max_price - self.entry_price

        # Breakeven: if profit >= sl_points, sl = entry_price
        if profit >= self.sl_points:
            self.sl = max(self.sl, self.entry_price)

        # Profit lock: if profit >= sl_points * 2, sl = entry_price + profit * 0.5
        if profit >= self.sl_points * 2:
            self.sl = max(self.sl, self.entry_price + profit * 0.5)

        return self.sl

# ==========================
# EXPECTANCY CALCULATION
# ==========================
def calculate_expectancy_v3(trades: List[Dict[str, float]]) -> float:
    if not trades:
        return 0.0

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]

    if not trades: return 0.0
    win_rate = len(wins) / len(trades)
    avg_win = sum(t["pnl"] for t in wins) / max(len(wins), 1)
    avg_loss = abs(sum(t["pnl"] for t in losses)) / max(len(losses), 1)

    return win_rate * avg_win - (1 - win_rate) * avg_loss

# ==========================
# BACKTEST ENGINE
# ==========================
class BacktestEngineV3:
    def __init__(self, bot, historical_data: List[Dict[str, Any]]):
        self.bot = bot
        self.data = historical_data
        self.trades = []

    def run(self):
        print(f"[BACKTEST] Running on {len(self.data)} historical candles...")
        for candle in self.data:
            # Check decision engine gates
            allow, bias, opt_type, inds = self.bot.should_enter_trade(
                candle["symbol"],
                candle["pcr"],
                candle["greeks"]
            )
            if allow:
                self.trades.append({
                    "pnl": candle.get("simulated_pnl", 0.0)
                })

        exp_val = calculate_expectancy_v3(self.trades)
        print(f"[BACKTEST] Exp={exp_val:.2f} | Trades={len(self.trades)}")
        return exp_val

class UnifiedStopLossManagerV2:
    """
    Stop-loss decision engine for BUY trades.
    Returns: should_exit, reason, loss_rupees
    """
    def __init__(
        self,
        *,
        max_loss_pct_default: float = 0.30,
        max_loss_abs: Optional[float] = None,
        hard_stop_buffer_pct: float = 0.0,
    ) -> None:
        self.max_loss_pct_default = float(max_loss_pct_default)
        self.max_loss_abs = max_loss_abs
        self.hard_stop_buffer_pct = float(hard_stop_buffer_pct)

    def check_stop_loss(
        self,
        *,
        entry_price: float,
        ltp: float,
        point_value: float,
        qty: int,
        sl_pct_override: Optional[float] = None,
    ) -> Tuple[bool, str, float]:
        if entry_price <= 0 or qty <= 0 or point_value <= 0:
            return False, "INVALID_STATE", 0.0

        sl_pct = self.max_loss_pct_default if sl_pct_override is None else float(sl_pct_override)
        effective_sl_pct = sl_pct + self.hard_stop_buffer_pct

        pnl_points = (ltp - entry_price)
        pnl_rupees = pnl_points * point_value * qty

        loss_rupees = -pnl_rupees if pnl_rupees < 0 else 0.0
        denom = (entry_price * point_value * qty)
        loss_pct = (loss_rupees / denom) if denom > 0 else 0.0

        if loss_pct >= effective_sl_pct:
            return True, f"STOP_LOSS_PCT({loss_pct:.2%} >= {effective_sl_pct:.2%})", loss_rupees

        if self.max_loss_abs is not None and loss_rupees >= float(self.max_loss_abs):
            return True, f"STOP_LOSS_ABS({loss_rupees:.2f} >= {self.max_loss_abs:.2f})", loss_rupees

        return False, "NO_STOP_LOSS", loss_rupees


# ---------------------------------------------------------------------------
# ATR-BASED RISK MANAGEMENT (SENSEX SPECIALIZED)
# ---------------------------------------------------------------------------

class ATRRiskManager:
    """
    Dynamic ATR-based SL / TP for option buying
    """
    def __init__(
        self,
        point_value: float,
        atr_sl_mult: float = 0.6,     # SL = ATR * 0.6
        atr_tp_mult: float = 1.2,     # TP = ATR * 1.2
        min_sl_pct: float = 0.30,     # 30% minimum SL
        max_sl_pct: float = 0.45      # safety cap
    ):
        self.point_value = point_value
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.min_sl_pct = min_sl_pct
        self.max_sl_pct = max_sl_pct

    def compute_sl_tp(
        self,
        premium: float,
        atr_points: float
    ) -> tuple[float, float, float]:
        """
        Returns (sl_price, tp_price, sl_pct)
        """
        if premium <= 0 or atr_points <= 0:
            return 0.0, 0.0, self.min_sl_pct

        # ATR converted to option premium move
        # (ATR Points / Underlyer Step) -> No, user said (atr_points * mult) / point_value
        # Actually SENSEX point value is 1.0 for premium usually, but user spec says 20.0 (Lot size?)
        # Let's follow their logic: atr_premium_move = (atr_points * mult) / self.point_value
        # Wait, if SENSEX moves 100 points, the option delta 0.5 moves 50 points.
        # ATR is in index points.
        
        atr_premium_move = (atr_points * self.atr_sl_mult)
        sl_pct = (atr_premium_move / premium) if premium > 0 else self.min_sl_pct
        sl_pct = min(max(sl_pct, self.min_sl_pct), self.max_sl_pct)
        sl_price = premium * (1 - sl_pct)

        tp_premium_move = (atr_points * self.atr_tp_mult)
        tp_price = premium + tp_premium_move

        return round(sl_price, 2), round(tp_price, 2), sl_pct

class DynamicTrailingSL:
    """
    ATR-aware trailing stop with profit locking
    """
    def __init__(self, entry_price: float):
        self.entry = entry_price
        self.max_price = entry_price
        self.sl: Optional[float] = None

    def update(self, ltp: float) -> Optional[float]:
        if ltp > self.max_price:
            self.max_price = ltp

        profit_pct = (self.max_price - self.entry) / self.entry

        # Breakeven
        if profit_pct >= 0.20:
            self.sl = max(self.sl or 0.0, self.entry)

        # Lock partial
        if profit_pct >= 0.35:
            self.sl = max(self.sl or 0.0, self.entry * 1.15)

        # Trail aggressively
        if profit_pct >= 0.50:
            self.sl = max(self.sl or 0.0, self.max_price * 0.80)

        return round(self.sl, 2) if self.sl is not None else None


# ---------------------------------------------------------------------------
# V2 Order Manager
# ---------------------------------------------------------------------------

class OrderManagerV2:
    """
    Execution + lifecycle for one active option trade.
    Priority:
      1) Stop loss (UnifiedStopLossManagerV2)
      2) Take profit
      3) Trailing exit
    """
    def __init__(self, broker: BrokerAPIV2, *, sl_manager: Optional[UnifiedStopLossManagerV2] = None) -> None:
        self.broker = broker
        self.sl_manager = sl_manager or UnifiedStopLossManagerV2()

        # Active state
        self.active: bool = False
        self.symbol: Optional[str] = None
        self.qty: int = 0
        self.entry_price: float = 0.0
        self.point_value: float = 1.0

        # Risk
        self.sl_pct: float = 0.30
        self.tp_pct: float = 0.50

        self.entry_time: Optional[dt.datetime] = None
        self.time_stop_min: int = 12

        # Trailing
        self.trailing_active: bool = False
        self.trailing_start_pct: float = 0.20
        self.trailing_lock_pct: float = 0.10
        self.peak_price: float = 0.0

        # Dynamic ATR-based Risk
        self.tp_price: Optional[float] = None
        self.atr_trail: Optional[DynamicTrailingSL] = None

        # Context for debugging
        self.indsP: Dict[str, Any] = {}

    def _process_entry(
        self,
        *,
        symbol: str,
        qty: int,
        side: str,
        point_value: float,
        sl_pct: float,
        tp_pct: float,
        trailing_start_pct: float,
        trailing_lock_pct: float,
        indsP: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self.active:
            _logger_v2.warning("ENTRY_IGNORED | already active | symbol=%s", self.symbol)
            return False

        if side.upper() != "BUY":
            raise ValueError("OrderManagerV2 expects BUY for option buying flow.")
        if qty <= 0:
            raise ValueError("qty must be > 0")

        try:
            order_id = self.broker.place_market_order(symbol, side="BUY", qty=int(qty))
        except MarginShortfallError as e:
            log(f"🚫 [ENTRY] Skipped due to margin shortfall: {e}")
            return False

        ltp = float(self.broker.get_ltp(symbol))

        self.active = True
        self.symbol = symbol
        self.qty = int(qty)
        self.entry_price = ltp
        self.point_value = float(point_value)

        self.sl_pct = float(sl_pct)
        self.tp_pct = float(tp_pct)

        # Dynamic ATR-based Risk Params
        self.tp_price = float(getattr(self, 'tp_price_override', ltp * (1 + tp_pct)))
        atr_val = float(indsP.get("atr", 0)) if indsP else 0
        
        self.trailing_start_pct = float(trailing_start_pct)
        self.trailing_lock_pct = float(trailing_lock_pct)
        self.trailing_active = False
        self.peak_price = self.entry_price

        # Professional Profit-Lock + Trailing
        self.atr_trail = DynamicTrailingSL(self.entry_price)
        # Initial SL from caller (OptionOrderManagerV2.execute_signal)
        self.atr_trail.sl = float(getattr(self, 'sl_price_override', ltp * (1 - sl_pct)))

        # New: Point-based trail from user request
        self.point_trail = TrailingStopManagerV3(sl_points=10) # Default 10 points
        self.point_trail.init(self.entry_price)

        self.indsP = indsP or {}
        self.entry_time = dt.datetime.now(IST)
        # Try to extract time_stop_min from indsP/signal if possible
        if self.indsP.get("time_stop_min"):
            self.time_stop_min = int(self.indsP.get("time_stop_min"))

        _logger_v2.info(
            "ENTRY | symbol=%s | qty=%s | entry=%.2f | order_id=%s | sl=%.2f%% | tp=%.2f%%",
            symbol, qty, self.entry_price, order_id, self.sl_pct * 100, self.tp_pct * 100
        )
        return True

    def _check_trailing_profit(self, ltp: float) -> Tuple[bool, str]:
        if not self.active or self.entry_price <= 0:
            return False, "NO_ACTIVE_TRADE"

        if ltp > self.peak_price:
            self.peak_price = ltp

        gain_pct = (ltp - self.entry_price) / self.entry_price

        if (not self.trailing_active) and gain_pct >= self.trailing_start_pct:
            self.trailing_active = True
            _logger_v2.info("TRAILING_ACTIVATED | gain=%.2f%% | peak=%.2f", gain_pct * 100, self.peak_price)

        if not self.trailing_active:
            return False, "TRAILING_NOT_ACTIVE"

        drawdown_pct = (self.peak_price - ltp) / self.peak_price if self.peak_price > 0 else 0.0
        if drawdown_pct >= self.trailing_lock_pct:
            return True, f"TRAILING_EXIT(drawdown={drawdown_pct:.2%})"

        return False, "TRAILING_OK"

    def execute_unified_strategy(self, ltp: float) -> Tuple[bool, str]:
        if not self.active or self.entry_price <= 0:
            return False, "NO_ACTIVE_TRADE"

        # 1) STOP LOSS FIRST
        should_exit_sl, sl_reason, _loss_rupees = self.sl_manager.check_stop_loss(
            entry_price=self.entry_price,
            ltp=float(ltp),
            point_value=self.point_value,
            qty=self.qty,
            sl_pct_override=self.sl_pct,
        )
        if should_exit_sl:
            return True, sl_reason

        # 2) TAKE PROFIT
        gain_pct = (ltp - self.entry_price) / self.entry_price
        if self.tp_price and ltp >= self.tp_price:
            return True, f"TAKE_PROFIT_ATR(ltp={ltp:.2f} >= tp={self.tp_price:.2f})"
        if gain_pct >= self.tp_pct:
            return True, f"TAKE_PROFIT_PCT(gain={gain_pct:.2%})"

        # 3) TRAILING (Percentage-based legacy fallback)
        should_exit_trail, trail_reason = self._check_trailing_profit(float(ltp))
        if should_exit_trail:
            return True, trail_reason

        # 3.5) PROFESSIONAL PROFIT-LOCK + TRAILING (New)
        if self.atr_trail:
            new_sl = self.atr_trail.update(ltp)
            if new_sl and ltp <= new_sl:
                return True, f"PROFIT_LOCK_TRAIL_SL({new_sl:.2f})"

        # 3.6) POINT-BASED TRAIL (Old point fallback)
        if hasattr(self, "point_trail") and self.point_trail.sl is not None:
            sl_point = self.point_trail.update(ltp)
            if ltp <= sl_point:
                return True, f"POINT_TRAIL_SL({sl_point:.2f})"

        # 4) TIME STOP
        if self.entry_time:
            elapsed = (dt.datetime.now(IST) - self.entry_time).total_seconds() / 60
            if elapsed >= self.time_stop_min:
                return True, f"TIME_STOP({elapsed:.1f}m)"

        return False, "HOLD"

    def _process_exit(self, reason: str, ltp: Optional[float] = None) -> Dict[str, Any]:
        if not self.active or not self.symbol:
            return {"exited": False, "reason": "NO_ACTIVE_TRADE"}

        symbol = self.symbol
        qty = self.qty
        entry = self.entry_price
        pv = self.point_value

        exit_id = self.broker.exit_position_market(symbol, qty=int(qty))
        exit_ltp = float(self.broker.get_ltp(symbol) if ltp is None else ltp)

        pnl_points = (exit_ltp - entry)
        pnl_rupees = pnl_points * pv * qty

        _logger_v2.info(
            "EXIT | symbol=%s | qty=%s | exit=%.2f | pnl_points=%.2f | pnl_rupees=%.2f | reason=%s | exit_id=%s",
            symbol, qty, exit_ltp, pnl_points, pnl_rupees, reason, exit_id
        )

        # cleanup
        self.active = False
        self.symbol = None
        self.qty = 0
        self.entry_price = 0.0
        self.point_value = 1.0
        self.trailing_active = False
        self.peak_price = 0.0
        self.indsP = {}

        return {
            "exited": True,
            "symbol": symbol,
            "qty": int(qty),
            "entry_price": float(entry),
            "exit_price": float(exit_ltp),
            "pnl_points": float(pnl_points),
            "pnl_rupees": float(pnl_rupees),
            "reason": reason,
        }


# ---------------------------------------------------------------------------
# V2 Option Buyers
# ---------------------------------------------------------------------------

class OptionEngineV3:
    def __init__(self, mkt: MarketDataAPIV2, configs: Dict[str, UnderlyingConfigV2]) -> None:
        self.mkt = mkt
        self.configs = configs
        self.fallbacks_today = 0
        self.last_fallback_date = None

    def get_atm_greeks(self, symbol: str, raw_chain: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Fetch ATM Greeks for entry validation.
        """
        try:
            resolved = SymbolRouterV3.resolve(symbol)
            if not resolved: return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "iv": 0.0}
            
            # Fetch a small chain around ATM if not provided
            if raw_chain is None:
                raw_chain = self.mkt.get_option_chain_raw(resolved.option_chain_symbol, strikecount="1")
            
            if not raw_chain or raw_chain.get("s") != "ok" or "data" not in raw_chain:
                return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "iv": 0.0}
                
            rows = raw_chain["data"].get("optionsChain") or []
            if not rows: return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "iv": 0.0}
            
            cfg = self.configs.get(resolved.underlying_key)
            underlying_ltp = self._get_underlying_ltp(resolved.analysis_symbol)
            if underlying_ltp <= 0:
                # Fallback to strike price from first row if quote fails
                underlying_ltp = float(rows[0].get("strike_price") or rows[0].get("strike") or 0.0)

            # Pick the first row (strikecount 1 usually returns ATM CE/PE)
            g = self._calculate_greeks_for_row(rows[0], cfg, underlying_ltp)
            return {
                "delta": g.delta,
                "gamma": g.gamma,
                "theta": g.theta,
                "vega": g.vega,
                "iv": g.iv
            }
        except Exception as e:
            print(f"[ERROR] get_atm_greeks failed: {e}")
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "iv": 0.0}

    def select_strike(self, symbol: str, option_type: str, greeks: Optional[Dict[str, float]] = None, force_atm: bool = False, raw_chain: Optional[Dict[str, Any]] = None) -> str:
        """
        Select the specific strike symbol for the given option type.
        Incorporates auto-shift logic for high delta/gamma.
        """
        resolved = SymbolRouterV3.resolve(symbol)
        if not resolved: return ""

        # Logic for auto-shift (User request)
        if greeks and abs(greeks.get("delta", 0)) > 0.55 and greeks.get("gamma", 0) > 0:
            print(f"[ENGINE] High Delta/Gamma detected. Shifting strike for {option_type}")
            # In analyze(), this translates to forcing a deeper strike if desired.
            # For now, we rely on the scoring in analyze() which already targets cfg.delta_target_abs (usually 0.55).
            pass

        signal = self.analyze(
            resolved, 
            desired_type=option_type, 
            force_atm=force_atm,
            extra_meta={"precomputed_greeks": greeks},
            raw_chain=raw_chain
        )
        if signal:
            return signal.option_symbol
        return ""

    def _get_vix(self, cfg: UnderlyingConfigV2) -> float:
        if not cfg.vix_symbol:
            return 0.0
        q = self.mkt.get_quote(cfg.vix_symbol)
        return float(q.get("lp") or q.get("ltp") or 0.0)

    def _momentum(self, cfg: UnderlyingConfigV2, symbol: str) -> Tuple[str, Dict[str, Any]]:
        target_tf = str(cfg.momentum_tf_min)
        lookback = int(cfg.momentum_lookback_min)
        df = self.mkt.get_history_df(symbol, int(target_tf), lookback)
        if df.empty or len(df) < cfg.ema_slow:
            return "NEUTRAL", {}

        close = df["Close"]
        ema_f = close.ewm(span=cfg.ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=cfg.ema_slow, adjust=False).mean()

        last_f = ema_f.iloc[-1]
        last_s = ema_s.iloc[-1]
        prev_f = ema_f.iloc[-2]
        prev_s = ema_s.iloc[-2]

        info = {"ema_fast": round(last_f, 4), "ema_slow": round(last_s, 4)}
        if last_f > last_s:
            return "BULLISH", info
        if last_f < last_s:
            return "BEARISH", info
        return "NEUTRAL", info

    def _get_underlying_ltp(self, symbol: str) -> float:
        """Best-effort LTP fetch for the underlying."""
        try:
            q = self.mkt.get_quote(symbol)
            return float(q.get("lp") or q.get("ltp") or 0.0)
        except Exception:
            return 0.0

    def _row_option_type(self, r: Dict[str, Any]) -> Optional[str]:
        """Normalize option type from option-chain rows.

        Fyers payloads vary across segments/updates. We accept:
        - option_type: "Call"/"Put" OR "CE"/"PE"
        - optionType / right: similar variants
        """
        val = r.get("option_type") or r.get("optionType") or r.get("right") or r.get("type")
        if not isinstance(val, str):
            return None
        v = val.strip().upper()
        if v in ("CE", "PE"):
            return v
        if v in ("CALL", "C") or v.startswith("CAL"):
            return "CE"
        if v in ("PUT", "P") or v.startswith("PUT"):
            return "PE"
        return None

    def _row_delta(self, r: Dict[str, Any]) -> float:
        """Best-effort delta extraction (supports nested greeks dicts)."""
        try:
            if r.get("delta") is not None:
                return float(r.get("delta") or 0.0)
            g = r.get("greeks")
            if isinstance(g, dict):
                return float(g.get("delta") or 0.0)
            g2 = r.get("Greeks")
            if isinstance(g2, dict):
                return float(g2.get("delta") or 0.0)
        except Exception:
            pass
        return 0.0

    def _time_to_expiry_years(self, expiry_val: Any) -> float:
        """Best-effort time-to-expiry in years.

        Supports:
        - Unix epoch seconds/ms
        - ISO date strings (YYYY-MM-DD / YYYY-MM-DDTHH:MM)
        - Compact MCX/BSE style strings (e.g., '26JAN2026', '26JAN', '26JANFUT')

        Falls back to 2 days if parsing fails.
        """
        try:
            now = _dt.datetime.now(_IST_V3)
            # epoch
            if isinstance(expiry_val, (int, float)):
                ts = float(expiry_val)
                if ts > 1e12:
                    ts = ts / 1000.0
                exp = _dt.datetime.fromtimestamp(ts, tz=_IST_V3)
                return max((exp - now).total_seconds(), 0.0) / (365.0 * 24 * 3600)
            s = str(expiry_val or "").strip()
            if not s:
                raise ValueError("empty")
            # ISO-ish
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y"):
                try:
                    exp = _dt.datetime.strptime(s, fmt)
                    exp = exp.replace(tzinfo=_IST_V3)
                    return max((exp - now).total_seconds(), 0.0) / (365.0 * 24 * 3600)
                except Exception:
                    continue
            # If above didn't work, try dateutil if available
            try:
                from dateutil.parser import parse as _parse
                exp = _parse(s)
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=_IST_V3)
                else:
                    exp = exp.astimezone(_IST_V3)
                return max((exp - now).total_seconds(), 0.0) / (365.0 * 24 * 3600)
            except Exception:
                pass
        except Exception:
            pass
        # fallback: 2 trading days
        return 2.0 / 365.0

    def _calculate_greeks_for_row(self, r: Dict[str, Any], cfg: UnderlyingConfigV2, underlying_ltp: float) -> GreeksV2:
        """Centralized Greek calculation with Black-76 fallback."""
        delta_val = float(self._row_delta(r) or 0.0)
        gamma_val = float(r.get("gamma") or 0.0)
        theta_val = float(r.get("theta") or 0.0)
        vega_val = float(r.get("vega") or 0.0)
        iv_val = float(r.get("iv") or 0.0)
        ltp = float(r.get("ltp") or r.get("lp") or 0.0)
        K = float(r.get("strike_price") or r.get("strike") or 0.0)
        opt_type = self._row_option_type(r) or "CE"

        # If API gives 0.0, calculate manually
        if (abs(delta_val) < 1e-9) and (underlying_ltp > 0) and (K > 0) and (ltp > 0):
            try:
                T = float(self._time_to_expiry_years(r.get("expiry") or r.get("expDate") or ""))
                rfr = float(getattr(cfg, "risk_free_rate", 0.07) or 0.07)
                opt_kind = 'C' if opt_type == 'CE' else 'P'
                sigma = iv_val if iv_val > 0 else float(Black76Greeks.implied_vol(underlying_ltp, K, rfr, T, opt_kind, ltp))
                gg = Black76Greeks.greeks(underlying_ltp, K, rfr, sigma, T, opt_kind)
                
                delta_val = float(gg.get("delta") or 0.0)
                gamma_val = float(gg.get("gamma") or 0.0)
                theta_val = float(gg.get("theta_day") or 0.0)
                vega_val = float(gg.get("vega") or 0.0)
                iv_val = sigma
            except Exception: pass

        return GreeksV2(
            delta=abs(float(delta_val)),
            gamma=float(gamma_val),
            theta=float(theta_val),
            vega=float(vega_val),
            iv=float(iv_val)
        )

    def _pick_atm_from_chain(
        self,
        cfg: UnderlyingConfigV2,
        resolved: ResolvedSymbolsV3,
        raw_chain: Dict[str, Any],
        desired_type: str,
        *,
        reason: str,
        vix: float,
        momentum: str,
        mom_info: Dict[str, Any],
        extra_meta: Dict[str, Any],
    ) -> Optional[OptionBuySignalV2]:
        """Fallback: pick ATM option (closest strike) for desired_type.

        This is used when:
        - Greeks are missing/zero (e.g., delta=0 for all rows), causing filters to drop everything.
        - The user wants strong voting to always choose ATM CE/PE.
        """
        rows = (raw_chain.get("data") or {}).get("optionsChain") or []
        if not rows:
            return None

        underlying_ltp = self._get_underlying_ltp(resolved.analysis_symbol)
        # If we cannot fetch LTP, still pick by nearest strike in the chain (median-ish)
        if underlying_ltp <= 0:
            try:
                strikes = [float(r.get("strike_price") or r.get("strike") or 0.0) for r in rows]
                strikes = [s for s in strikes if s > 0]
                underlying_ltp = float(np.median(strikes)) if strikes else 0.0
            except Exception:
                underlying_ltp = 0.0

        best_row = None
        best_dist = float("inf")
        for r in rows:
            opt_type = self._row_option_type(r) or ""
            if opt_type != desired_type:
                continue
            strike = float(r.get("strike_price") or r.get("strike") or 0.0)
            if strike <= 0:
                continue
            dist = abs(strike - underlying_ltp) if underlying_ltp > 0 else 0.0
            if dist < best_dist:
                best_dist = dist
                best_row = r

        if not best_row or not best_row.get("symbol"):
            return None

        ltp = float(best_row.get("ltp") or best_row.get("lp") or 0.0)
        bid = float(best_row.get("bid") or best_row.get("bid_price") or ltp)
        ask = float(best_row.get("ask") or best_row.get("ask_price") or ltp)
        entry = ltp if ltp > 0 else (ask if ask > 0 else bid)

        delta_val = float(self._row_delta(best_row) or 0.0)
        g = self._calculate_greeks_for_row(best_row, cfg, underlying_ltp)
        pcr = float(best_row.get("pcr") or 1.0)

        meta = {
            "mom": mom_info,
            "bid": bid,
            "ask": ask,
            "oi": float(best_row.get("oi") or 0.0),
            "volume": float(best_row.get("volume") or 0.0),
            "fallback": "ATM",
            "fallback_reason": reason,
            "underlying_ltp": underlying_ltp,
        }
        meta.update(extra_meta or {})

        sig = OptionBuySignalV2(
            underlying=cfg.key,
            option_symbol=str(best_row["symbol"]),
            option_type=desired_type,
            strike=float(best_row.get("strike_price") or best_row.get("strike") or 0.0),
            expiry=str(best_row.get("expiry", "")),
            entry_price=float(entry or 0.0),
            greeks=g,
            pcr=pcr,
            vix=vix,
            momentum=momentum,
            score=0.0,
            expectancy=float(g.vega * g.iv * 100 - abs(g.theta)),
            meta=meta,
        )
        print(f"[FALLBACK] Picked ATM {desired_type}: {sig.option_symbol} | strike={sig.strike} | entry={sig.entry_price}")
        return sig

    def analyze(
        self,
        resolved: ResolvedSymbolsV3,
        *,
        desired_type: Optional[str] = None,
        force_atm: bool = False,
        extra_meta: Optional[Dict[str, Any]] = None,
        raw_chain: Optional[Dict[str, Any]] = None,
        underlying_ltp: Optional[float] = None,
    ) -> Optional[OptionBuySignalV2]:
        cfg = self.configs.get(resolved.underlying_key)
        if not cfg:
            return None

        vix = self._get_vix(cfg)
        if cfg.max_vix is not None and vix > float(cfg.max_vix):
            _logger_v2.info("NO_SIGNAL | %s | VIX high: %.2f", cfg.key, vix)
            return None

        momentum, mom_info = self._momentum(cfg, resolved.analysis_symbol)

        # Allow caller (voting) to force the direction.
        if desired_type in ("CE", "PE"):
            momentum = "BULLISH" if desired_type == "CE" else "BEARISH"
        else:
            if momentum not in ("BULLISH", "BEARISH"):
                _logger_v2.info("NO_SIGNAL | %s | momentum=%s", cfg.key, momentum)
                return None
            desired_type = "CE" if momentum == "BULLISH" else "PE"

        # Fetch chain if not provided
        if raw_chain is None:
            raw_chain = self.mkt.get_option_chain_raw(resolved.option_chain_symbol, strikecount="10")
            
        if not raw_chain or raw_chain.get("s") != "ok" or "data" not in raw_chain or "optionsChain" not in raw_chain["data"]:
            print(f"[WARNING] [ENGINE] No option chain data for {resolved.option_chain_symbol}")
            return None

        candidates = []
        rows = raw_chain["data"]["optionsChain"]
        print(f"[ENGINE] Processing {len(rows)} chain rows for {desired_type}...")

        # Underlying futures LTP for greeks fallback (Black-76)
        underlying_ltp_for_greeks = underlying_ltp or self._get_underlying_ltp(resolved.analysis_symbol)
        
        # Pre-fetch limits for speed
        d_min = float(getattr(cfg, 'delta_min_abs', 0.10)) # Reduced default min delta
        d_max = float(getattr(cfg, 'delta_max_abs', 0.65))
        min_p = float(getattr(cfg, 'min_premium', 5.0) or 5.0) # Reduced default min premium
        max_p = float(getattr(cfg, 'max_premium', 12000.0) or 12000.0)
        max_spr = float(getattr(cfg, 'max_spread_pct', 0.05) or 0.05)
        
        for r in rows:
            opt_type = self._row_option_type(r) or ""
            if opt_type != desired_type:
                continue

            # Fyers might use 'ltp' or 'lp' or 'last_price'
            ltp = float(r.get("ltp") or r.get("lp") or 0.0)
            
            # USE PRE-FETCHED LTP (FIXED: removed redundant api call inside loop)
            g = self._calculate_greeks_for_row(r, cfg, underlying_ltp_for_greeks)

            pcr = float(r.get("pcr") or 1.0)
            
            c = OptionCandidateV2(
                symbol=r["symbol"],
                option_type=opt_type,
                strike=float(r.get("strike_price") or r.get("strike") or 0.0),
                expiry=str(r.get("expiry", "")),
                ltp=ltp,
                bid=float(r.get("bid") or r.get("bid_price") or ltp),
                ask=float(r.get("ask") or r.get("ask_price") or ltp),
                oi=float(r.get("oi") or 0.0),
                volume=float(r.get("volume") or 0.0),
                greeks=g,
                pcr=pcr,
                expectancy=float(g.vega * g.iv * 100 - abs(g.theta))
            )
            
            if self._passes_filters(cfg, c, momentum):
                candidates.append(c)
            else:
                # Debug why filtered
                spread = (c.ask - c.bid) / c.ltp if c.ltp > 0 else 1.0
                print(f"   Rejected {c.symbol}: ltp={c.ltp}, delta={c.greeks.delta:.2f}, exp={c.expectancy:.2f}, spr={spread:.2%}, pcr={c.pcr:.2f}")
                print(f"     (Limits: d={d_min}-{d_max}, exp_min={cfg.min_expectancy}, spr_max={max_spr:.2%}, p={min_p}-{max_p})")
                pass

        if not candidates:
            print(f"[ERROR] [ENGINE] No candidates passed filters for {cfg.key} {desired_type}")
            # Fallback Detection (Fix 3: More robust)
            all_candidate_deltas = [self._calculate_greeks_for_row(r, cfg, underlying_ltp_for_greeks).delta for r in rows]
            valid_deltas = [d for d in all_candidate_deltas if d is not None and np.isfinite(d)]
            no_data = len(valid_deltas) == 0 or max(valid_deltas or [0.0]) < 0.05
            
            adx_val = float((extra_meta or {}).get("adx") or 0.0)
            
            # Reset daily fallbacks
            today = dt.date.today()
            if self.last_fallback_date != today:
                self.fallbacks_today = 0
                self.last_fallback_date = today

            do_fallback = False
            if no_data:
                # Advice: Never fallback if ADX > 30 (trend day)
                if adx_val > 30:
                    print(f"[SKIP] Greek fallback denied: ADX {adx_val:.1f} > 30 (Strong Trend)")
                elif self.fallbacks_today >= 1:
                    print(f"[SKIP] Greek fallback denied: Daily limit reached ({self.fallbacks_today})")
                else:
                    print("[ERROR] [ENGINE] Greeks missing/zero. Triggering fallback approximation...")
                    do_fallback = True
                    self.fallbacks_today += 1

            if force_atm or do_fallback:
                return self._pick_atm_from_chain(
                    cfg,
                    resolved,
                    raw_chain,
                    desired_type,
                    reason=("FORCE_ATM" if force_atm else "GREEKS_MISSING"),
                    vix=vix,
                    momentum=momentum,
                    mom_info=mom_info,
                    extra_meta=(extra_meta or {}),
                )
            return None

        # FORCE ATM check: If user wants ATM only, bypass scoring
        if force_atm:
             return self._pick_atm_from_chain(
                    cfg,
                    resolved,
                    raw_chain,
                    desired_type,
                    reason="FORCE_ATM",
                    vix=vix,
                    momentum=momentum,
                    mom_info=mom_info,
                    extra_meta=(extra_meta or {}),
                )

        best = max(candidates, key=lambda x: self._score(cfg, x))
        score = self._score(cfg, best)

        if best.ltp <= 0:
            print(f"[ERROR] [ENGINE] Best candidate has 0 LTP: {best.symbol}")
            return None

        sig = OptionBuySignalV2(
            underlying=cfg.key,
            option_symbol=best.symbol,
            option_type=best.option_type,
            strike=best.strike,
            expiry=best.expiry,
            entry_price=best.ltp,
            greeks=best.greeks,
            pcr=best.pcr,
            vix=vix,
            momentum=momentum,
            score=score,
            expectancy=best.expectancy,
            meta={"mom": mom_info, "bid": best.bid, "ask": best.ask, "oi": best.oi, "volume": best.volume, **(extra_meta or {})},
        )

        print(
            f"[SIGNAL] {sig.underlying} | {sig.option_symbol} {sig.option_type} | strike={sig.strike} | "
            f"ltp={sig.entry_price} | delta={sig.greeks.delta:.2f} | expectancy={sig.expectancy:.2f} | score={sig.score:.2f}"
        )
        return sig

    def _passes_filters(self, cfg: UnderlyingConfigV2, c: OptionCandidateV2, momentum: str) -> bool:
        if c.ltp <= 0: return False

        # Premium affordability filter
        try:
            min_p = float(getattr(cfg, 'min_premium', 0.0) or 0.0)
            max_p = float(getattr(cfg, 'max_premium', 0.0) or 0.0)
        except Exception:
            min_p, max_p = 0.0, 0.0
        if min_p > 0 and c.ltp < min_p: return False
        if max_p > 0 and c.ltp > max_p: return False
        
        # ✅ Delta Window (Relaxed default range 0.10 - 0.85)
        d_min = float(getattr(cfg, 'delta_min_abs', 0.10))
        d_max = float(getattr(cfg, 'delta_max_abs', 0.85))
        if not (d_min <= abs(c.greeks.delta) <= d_max):
            return False
            
        spread = (c.ask - c.bid) / c.ltp if c.ltp > 0 else 1.0
        # Check against config max spread or default to 5%
        max_spr = float(getattr(cfg, 'max_spread_pct', 0.05) or 0.05)
        if spread > max_spr: return False

        # ✅ Correct PCR Logic (Fix 1: REMOVED HARD FILTER)
        # PCR is now a soft bias in _score()
        # if not str(c.symbol).startswith("MCX:"): ... REMOVED
        
        # [OK] Expectancy Gate
        if c.expectancy < cfg.min_expectancy:
            return False
        
        return True

    def _score(self, cfg: UnderlyingConfigV2, c: OptionCandidateV2) -> float:
        delta_err = abs(c.greeks.delta - cfg.delta_target_abs)
        spread = (c.ask - c.bid) / c.ltp if c.ltp > 0 else 1.0
        
        score = 10.0
        score -= delta_err * 20.0
        score -= spread * 100.0
        score += c.expectancy * 0.5

        # ✅ PCR Score (Bias not Gate)
        # PCR Logic: CE -> <0.9 (+1), 0.9-1.1 (+0.5), >1.1 (-0.5)
        #            PE -> >1.1 (+1), 0.9-1.1 (+0.5), <0.9 (-0.5)
        pcr_score = 0.0
        if not str(c.symbol).startswith("MCX:"):
             if c.pcr > 0 and abs(c.pcr - 1.0) > 1e-4:
                if c.option_type == "CE":
                    if c.pcr < 0.90: pcr_score = 1.0
                    elif c.pcr <= 1.10: pcr_score = 0.5
                    else: pcr_score = -0.5
                else: # PE
                    if c.pcr > 1.10: pcr_score = 1.0
                    elif c.pcr >= 0.90: pcr_score = 0.5
                    else: pcr_score = -0.5
        
        score += pcr_score
        return score


# ============================================================================
# ZERO TO HERO EXPIRY DAY STRATEGY MODULE
# ============================================================================

class TradingMode(Enum):
    """Trading mode based on day type"""
    HERO_TO_ZERO_PREVENTION = "hero_to_zero"  # Daily option buying
    ZERO_TO_HERO_EXPIRY = "zero_to_hero"      # Expiry day OTM buying
    GREEKS_EDGE_V3 = "greeks_edge_v3"         # Original v3 mode
    NO_TRADE = "no_trade"                     # Risk too high

class SignalStrength(Enum):
    """Signal confidence levels"""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"

class OptionType(Enum):
    """Option types - ONLY BUYING"""
    CE = "CE"  # Call Buy
    PE = "PE"  # Put Buy
    NONE = "NONE"

@dataclass
class MarketData:
    """Current market state"""
    timestamp: dt.datetime
    price: float
    open: float
    high: float
    low: float
    volume: float
    vwap: float
    ema_9: float
    ema_21: float
    sma_20: float
    rsi: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    bb_middle: float
    bb_width: float
    vol_ratio: float
    atr: float = 0.0
    day_range_pct: float = 0.0
    adx: float = 0.0
    di_plus: float = 0.0
    di_minus: float = 0.0

@dataclass
class ZeroToHeroOptionData:
    """Option chain data"""
    strike: float
    option_type: str  # CE or PE
    ltp: float
    bid: float
    ask: float
    volume: int
    oi: int
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    symbol: str = ""
    expiry: str = ""

@dataclass
class TradingSignal:
    """Trading signal output - OPTION BUYING ONLY"""
    mode: TradingMode
    action: str  # BUY/EXIT/WAIT
    direction: OptionType  # CE/PE/NONE
    strength: SignalStrength
    entry_price: float
    stop_loss: float
    targets: List[float]
    position_size_pct: float
    risk_level: str
    reasons: List[str]
    time_sl_minutes: Optional[int] = None
    strike_otm: Optional[int] = None  # For expiry day: 0=ATM, 1=1-step OTM
    option_data: Optional[ZeroToHeroOptionData] = None
    signal_score: float = 0.0
    
@dataclass
class PositionZeroToHero:
    """Current position - OPTION BUYING ONLY"""
    symbol: str
    option_type: str  # CE or PE
    strike: float
    entry_price: float
    current_price: float
    quantity: int
    entry_time: dt.datetime
    stop_loss: float
    trailing_sl: Optional[float] = None
    targets: List[float] = field(default_factory=list)
    targets_hit: List[bool] = field(default_factory=list)
    pnl: float = 0.0
    pnl_pct: float = 0.0
    time_sl: Optional[dt.datetime] = None
    mode: str = "NORMAL"  # NORMAL or EXPIRY
    
    def update_pnl(self):
        """Update P&L"""
        self.pnl = (self.current_price - self.entry_price) * self.quantity
        if self.entry_price > 0:
            self.pnl_pct = ((self.current_price - self.entry_price) / self.entry_price) * 100

class TradingConfigZeroToHero:
    """Trading configuration - Unified for all modes"""
    
    # General Settings
    TIMEZONE = "Asia/Kolkata"
    MARKET_OPEN = dt_time(9, 15)
    MARKET_CLOSE = dt_time(15, 30)
    
    # Hero to Zero Prevention Settings (Normal Days)
    MIN_BB_WIDTH = 2.0  # Minimum BB width percentage
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    MAX_CONSECUTIVE_DAYS = 3
    MIN_VOLUME_RATIO = 0.7
    
    # Position Sizing (OPTION BUYING)
    NORMAL_DAY_RISK_PCT = 5.0      # 5% of capital per trade
    EXPIRY_DAY_RISK_PCT = 2.0      # 2% of capital on expiry day
    HIGH_RISK_DAY_PCT = 2.0        # 2% on high risk days
    MAX_LOSS_PER_TRADE_PCT = 40.0  # Max 40% loss on option premium
    
    # Stop Loss Settings
    OPTION_SL_PCT = 40.0           # 40% loss on option premium
    PROFIT_LOCK_AT_PCT = 30.0      # Lock profit when +30%
    TRAIL_SL_PCT = 85.0            # Trail at 85% of peak
    BREAKEVEN_AT_PCT = 150.0       # Move to breakeven at +150%
    
    # Zero to Hero Expiry Day Settings
    EXPIRY_MIN_TRADE_TIME = dt_time(10, 30)  # Don't trade before 10:30 AM
    EXPIRY_EXIT_TIME = dt_time(15, 15)       # Exit all by 3:15 PM
    EXPIRY_MIN_RANGE_PCT = 0.9               # Minimum 0.9% range to trade
    EXPIRY_MIN_FIRST_75MIN_RANGE_PCT = 0.5   # Minimum first 75min range
    EXPIRY_MAX_VWAP_CROSSES = 4              # Max VWAP crosses allowed
    EXPIRY_TIME_SL_MINUTES = 10              # Time-based stop loss
    EXPIRY_MIN_PROFIT_IN_TIME_SL_PCT = 20.0  # Min profit in 10 min
    
    # OTM Settings for Expiry Day
    MAX_OTM_STEPS = 1              # ONLY 1-step OTM
    MIN_PREMIUM = 70.0             # Minimum premium ₹70
    MAX_PREMIUM = 200.0            # Maximum premium ₹200
    MIN_DELTA_FOR_OTM = 0.25       # Minimum delta
    MAX_THETA_FOR_OTM = 15.0       # Maximum theta decay
    
    # Target Settings (OPTION BUYING)
    TARGET_1_MULT = 2.0    # 100% profit - Book 50%
    TARGET_2_MULT = 3.0    # 200% profit - Book 30%
    TARGET_3_MULT = 5.0    # 400% profit - Book 20%
    
    TARGET_1_BOOK_PCT = 50.0
    TARGET_2_BOOK_PCT = 30.0
    TARGET_3_BOOK_PCT = 20.0
    
    # Score Thresholds
    MIN_BULLISH_SCORE = 8
    MIN_BEARISH_SCORE = 8
    STRONG_SIGNAL_SCORE = 10
    
    # Greeks-based V3 Settings
    MIN_DELTA = 0.20
    MAX_DELTA = 0.80
    MIN_IV_PERCENTILE = 30
    THETA_DECAY_LIMIT = -25.0

class ExpiryDayFilters:
    """
    Master filters for expiry day trading
    These filters eliminate 70-75% of losing days
    """
    
    def __init__(self, config: TradingConfigZeroToHero):
        self.config = config
        self.vwap_cross_count = 0
        self.first_75min_range = 0.0
        self.current_range = 0.0
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def check_time_filter(self, current_time: dt_time) -> Tuple[bool, str]:
        """Check if time is appropriate for expiry day trading"""
        if current_time < self.config.EXPIRY_MIN_TRADE_TIME:
            return False, f"Too early: {current_time} < {self.config.EXPIRY_MIN_TRADE_TIME}"
        
        if current_time >= self.config.EXPIRY_EXIT_TIME:
            return False, f"Too late: {current_time} >= {self.config.EXPIRY_EXIT_TIME}"
            
        return True, "Time filter passed"
    
    def check_range_filter(self, first_75min_range_pct: float, current_range_pct: float) -> Tuple[bool, str]:
        """Check if range is sufficient for hero move"""
        if first_75min_range_pct < self.config.EXPIRY_MIN_FIRST_75MIN_RANGE_PCT:
            return False, f"First 75min range too small: {first_75min_range_pct:.2f}%"
        
        if current_range_pct < self.config.EXPIRY_MIN_RANGE_PCT:
            return False, f"Current range too small: {current_range_pct:.2f}%"
        
        return True, "Range filter passed"
    
    def check_vwap_filter(self, vwap_crosses: int) -> Tuple[bool, str]:
        """Check if market is not too choppy"""
        if vwap_crosses >= self.config.EXPIRY_MAX_VWAP_CROSSES:
            return False, f"Too choppy: {vwap_crosses} VWAP crosses"
        
        return True, "VWAP filter passed"
    
    def check_ema_flat_filter(self, ema9: float, ema21: float, price: float) -> Tuple[bool, str]:
        """Check if EMAs are not flat (momentum exists)"""
        ema_gap_pct = abs(ema9 - ema21) / price * 100
        
        if ema_gap_pct < 0.1:  # EMAs too close = no momentum
            return False, f"EMAs too flat: gap {ema_gap_pct:.3f}%"
        
        return True, "EMA filter passed"
    
    def check_all_filters(self, current_time: dt_time, first_75min_range_pct: float,
                         current_range_pct: float, vwap_crosses: int,
                         ema9: float, ema21: float, price: float) -> Tuple[bool, List[str]]:
        """Check all master filters"""
        reasons = []
        all_passed = True
        
        # Time filter
        passed, reason = self.check_time_filter(current_time)
        if not passed:
            all_passed = False
            reasons.append(f"❌ {reason}")
        else:
            reasons.append(f"✓ {reason}")
        
        # Range filter
        passed, reason = self.check_range_filter(first_75min_range_pct, current_range_pct)
        if not passed:
            all_passed = False
            reasons.append(f"❌ {reason}")
        else:
            reasons.append(f"✓ {reason}")
        
        # VWAP filter
        passed, reason = self.check_vwap_filter(vwap_crosses)
        if not passed:
            all_passed = False
            reasons.append(f"❌ {reason}")
        else:
            reasons.append(f"✓ {reason}")
        
        # EMA filter
        passed, reason = self.check_ema_flat_filter(ema9, ema21, price)
        if not passed:
            all_passed = False
            reasons.append(f"❌ {reason}")
        else:
            reasons.append(f"✓ {reason}")
        
        return all_passed, reasons

class TrendConfirmation:
    """Confirm strong directional trend for expiry day OTM buying"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.candles_history = []
        
    def add_candle(self, timestamp: dt.datetime, price: float, vwap: float, 
                   ema9: float, ema21: float, volume: float):
        """Add candle to history (keep last 12 candles = 1 hour for 5-min candles)"""
        self.candles_history.append({
            'timestamp': timestamp,
            'price': price,
            'vwap': vwap,
            'ema9': ema9,
            'ema21': ema21,
            'volume': volume
        })
        
        # Keep only last 12 candles
        if len(self.candles_history) > 12:
            self.candles_history.pop(0)
    
    def check_ce_trend(self) -> Tuple[bool, List[str]]:
        """Check for strong CE (bullish) trend"""
        reasons = []
        
        if len(self.candles_history) < 6:
            return False, ["Not enough candles yet"]
        
        # Check last 6 candles
        last_6 = self.candles_history[-6:]
        above_vwap = sum(1 for c in last_6 if c['price'] > c['vwap'])
        
        if above_vwap >= 5:  # At least 5 of 6 above VWAP
            reasons.append("✓ Price above VWAP (5+ of 6 candles)")
            
            # Check EMA alignment
            if self.candles_history[-1]['ema9'] > self.candles_history[-1]['ema21']:
                reasons.append("✓ EMA9 > EMA21")
                
                # Check gap widening
                ema_gap_now = self.candles_history[-1]['ema9'] - self.candles_history[-1]['ema21']
                ema_gap_prev = self.candles_history[-2]['ema9'] - self.candles_history[-2]['ema21']
                
                if ema_gap_now > ema_gap_prev:
                    reasons.append("✓ EMA gap widening")
                    return True, reasons
        
        return False, reasons or ["No strong CE trend"]
    
    def check_pe_trend(self) -> Tuple[bool, List[str]]:
        """Check for strong PE (bearish) trend"""
        reasons = []
        
        if len(self.candles_history) < 6:
            return False, ["Not enough candles yet"]
        
        # Check last 6 candles
        last_6 = self.candles_history[-6:]
        below_vwap = sum(1 for c in last_6 if c['price'] < c['vwap'])
        
        if below_vwap >= 5:  # At least 5 of 6 below VWAP
            reasons.append("✓ Price below VWAP (5+ of 6 candles)")
            
            # Check EMA alignment
            if self.candles_history[-1]['ema9'] < self.candles_history[-1]['ema21']:
                reasons.append("✓ EMA9 < EMA21")
                
                # Check gap widening
                ema_gap_now = self.candles_history[-1]['ema21'] - self.candles_history[-1]['ema9']
                ema_gap_prev = self.candles_history[-2]['ema21'] - self.candles_history[-2]['ema9']
                
                if ema_gap_now > ema_gap_prev:
                    reasons.append("✓ EMA gap widening (downward)")
                    return True, reasons
        
        return False, reasons or ["No strong PE trend"]

class HeroToZeroPrevention:
    """Daily option buying strategy with strict risk management"""
    
    def __init__(self, config: TradingConfigZeroToHero):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.consecutive_days_traded = 0
        self.last_exit_reason = ""
        
    def should_trade_today(self) -> Tuple[bool, str]:
        """Check if should trade today based on consecutive days"""
        if self.consecutive_days_traded >= self.config.MAX_CONSECUTIVE_DAYS:
            return False, f"Max consecutive days reached: {self.consecutive_days_traded}"
        return True, "Can trade today"
    
    def calculate_bullish_score(self, market: MarketData) -> float:
        """Calculate bullish score for signal generation"""
        score = 0.0
        
        # BB Width (wider = better)
        if market.bb_width >= self.config.MIN_BB_WIDTH:
            score += 3.0
        
        # RSI (30-70 range is normal)
        if market.rsi < self.config.RSI_OVERSOLD:
            score += 2.0  # Oversold = bullish
        elif market.rsi > self.config.RSI_OVERBOUGHT:
            score -= 2.0  # Overbought = less bullish
        else:
            score += 1.0  # Normal range
        
        # MACD
        if market.macd > market.macd_signal:
            score += 2.0
        
        # Price position in BB
        bb_range = market.bb_upper - market.bb_lower
        if bb_range > 0:
            price_position = (market.price - market.bb_lower) / bb_range
            if 0.3 < price_position < 0.7:  # In middle = stable
                score += 1.0
        
        # Volume
        if market.vol_ratio >= self.config.MIN_VOLUME_RATIO:
            score += 2.0
        
        return min(score, 10.0)  # Cap at 10
    
    def calculate_bearish_score(self, market: MarketData) -> float:
        """Calculate bearish score for signal generation"""
        score = 0.0
        
        # BB Width (wider = better)
        if market.bb_width >= self.config.MIN_BB_WIDTH:
            score += 3.0
        
        # RSI (opposite of bullish)
        if market.rsi > self.config.RSI_OVERBOUGHT:
            score += 2.0  # Overbought = bearish
        elif market.rsi < self.config.RSI_OVERSOLD:
            score -= 2.0  # Oversold = less bearish
        else:
            score += 1.0  # Normal range
        
        # MACD
        if market.macd < market.macd_signal:
            score += 2.0
        
        # Volume
        if market.vol_ratio >= self.config.MIN_VOLUME_RATIO:
            score += 2.0
        
        return min(score, 10.0)  # Cap at 10

class OptionSelector:
    """Select best option based on Greeks and filters"""
    
    def __init__(self, config: TradingConfigZeroToHero):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def select_best_option_atm(self, option_chain: List[ZeroToHeroOptionData], 
                               spot_price: float, direction: str) -> Optional[ZeroToHeroOptionData]:
        """Select ATM or nearest ITM option for normal days"""
        
        if direction == "CE":
            # For calls, prefer ATM or slightly ITM
            candidates = [opt for opt in option_chain 
                         if opt.option_type == "CE" and 
                         opt.strike <= spot_price and
                         self.config.MIN_DELTA <= opt.delta <= self.config.MAX_DELTA]
        else:  # PE
            # For puts, prefer ATM or slightly ITM
            candidates = [opt for opt in option_chain 
                         if opt.option_type == "PE" and 
                         opt.strike >= spot_price and
                         -self.config.MAX_DELTA <= opt.delta <= -self.config.MIN_DELTA]
        
        if not candidates:
            return None
        
        # Return the one closest to spot
        return min(candidates, key=lambda x: abs(x.strike - spot_price))
    
    def select_best_option_otm(self, option_chain: List[ZeroToHeroOptionData], 
                               spot_price: float, direction: str) -> Optional[ZeroToHeroOptionData]:
        """Select 1-step OTM option for expiry days"""
        
        if direction == "CE":
            # For calls, 1-step OTM = strike 1 step above current
            candidates = [opt for opt in option_chain 
                         if opt.option_type == "CE" and 
                         opt.strike > spot_price and
                         opt.delta >= self.config.MIN_DELTA_FOR_OTM and
                         self.config.MIN_PREMIUM <= opt.ltp <= self.config.MAX_PREMIUM]
        else:  # PE
            # For puts, 1-step OTM = strike 1 step below current
            candidates = [opt for opt in option_chain 
                         if opt.option_type == "PE" and 
                         opt.strike < spot_price and
                         opt.delta <= -self.config.MIN_DELTA_FOR_OTM and
                         self.config.MIN_PREMIUM <= opt.ltp <= self.config.MAX_PREMIUM]
        
        if not candidates:
            return None
        
        # Prefer the one with best delta (highest for CE, lowest for PE)
        if direction == "CE":
            return max(candidates, key=lambda x: x.delta)
        else:
            return min(candidates, key=lambda x: x.delta)  # Most negative

class SignalGenerator:
    """Generate trading signals based on market conditions"""
    
    def __init__(self, config: TradingConfigZeroToHero):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hero_to_zero = HeroToZeroPrevention(config)
        self.expiry_filters = ExpiryDayFilters(config)
        self.trend_confirmation = TrendConfirmation()
        self.option_selector = OptionSelector(config)
    
    def generate_signal(self, market: MarketData, is_expiry_day: bool) -> Optional[TradingSignal]:
        """Generate trading signal based on mode"""
        
        if is_expiry_day:
            return self._generate_expiry_day_signal(market)
        else:
            return self._generate_normal_day_signal(market)
    
    def _generate_normal_day_signal(self, market: MarketData) -> Optional[TradingSignal]:
        """Generate signal for normal trading days"""
        
        bullish_score = self.hero_to_zero.calculate_bullish_score(market)
        bearish_score = self.hero_to_zero.calculate_bearish_score(market)
        
        # Determine direction and strength
        if bullish_score >= self.config.MIN_BULLISH_SCORE:
            direction = OptionType.CE
            score = bullish_score
            strength = SignalStrength.STRONG if bullish_score >= self.config.STRONG_SIGNAL_SCORE else SignalStrength.MODERATE
        elif bearish_score >= self.config.MIN_BEARISH_SCORE:
            direction = OptionType.PE
            score = bearish_score
            strength = SignalStrength.STRONG if bearish_score >= self.config.STRONG_SIGNAL_SCORE else SignalStrength.MODERATE
        else:
            return None  # No signal
        
        # Set targets
        entry_price = market.price
        targets = [
            entry_price * self.config.TARGET_1_MULT,
            entry_price * self.config.TARGET_2_MULT,
            entry_price * self.config.TARGET_3_MULT
        ]
        
        return TradingSignal(
            mode=TradingMode.HERO_TO_ZERO_PREVENTION,
            action="BUY",
            direction=direction,
            strength=strength,
            entry_price=market.price,
            stop_loss=entry_price * (1 - self.config.OPTION_SL_PCT / 100),
            targets=targets,
            position_size_pct=self.config.NORMAL_DAY_RISK_PCT,
            risk_level="NORMAL",
            reasons=[f"Score: {score:.1f}/10", "Good volatility", "Volume confirmed"],
            signal_score=score
        )
    
    def _generate_expiry_day_signal(self, market: MarketData) -> Optional[TradingSignal]:
        """Generate signal for expiry day OTM buying"""
        
        # Check trend first
        ce_trend_ok, ce_reasons = self.trend_confirmation.check_ce_trend()
        pe_trend_ok, pe_reasons = self.trend_confirmation.check_pe_trend()
        
        if ce_trend_ok:
            direction = OptionType.CE
            reasons = ce_reasons
        elif pe_trend_ok:
            direction = OptionType.PE
            reasons = pe_reasons
        else:
            return None  # No strong trend
        
        entry_price = market.price
        targets = [
            entry_price * self.config.TARGET_1_MULT,
            entry_price * self.config.TARGET_2_MULT,
            entry_price * self.config.TARGET_3_MULT
        ]
        
        return TradingSignal(
            mode=TradingMode.ZERO_TO_HERO_EXPIRY,
            action="BUY",
            direction=direction,
            strength=SignalStrength.STRONG,
            entry_price=entry_price,
            stop_loss=entry_price * (1 - self.config.OPTION_SL_PCT / 100),
            targets=targets,
            position_size_pct=self.config.EXPIRY_DAY_RISK_PCT,
            risk_level="EXPIRY_HIGH_REWARD",
            time_sl_minutes=self.config.EXPIRY_TIME_SL_MINUTES,
            strike_otm=1,
            reasons=reasons,
            signal_score=9.5
        )

# ============================================================================
# END ZERO TO HERO EXPIRY DAY STRATEGY MODULE
# ============================================================================

# ---------------------------------------------------------------------------
# V2 Bridge + Safety Guards
# ---------------------------------------------------------------------------

class OptionOrderManagerV2:
    def __init__(self, broker: BrokerAPIV2, order_manager: OrderManagerV2) -> None:
        self.broker = broker
        self.om = order_manager
        self.trades_today: int = 0
        self.day_pnl_rupees: float = 0.0
        self.journal = TradeJournalV5()
        self.active_expectancy: float = 0.0
        self.current_day = dt.date.today()

    def reset_day_if_needed(self):
        if dt.date.today() != self.current_day:
            self.trades_today = 0
            self.day_pnl_rupees = 0.0
            self.current_day = dt.date.today()
            print("📅 [RESET] Daily trade count and PnL reset.")

    def can_trade(self, cfg: UnderlyingConfigV2) -> Tuple[bool, str]:
        if self.trades_today >= cfg.max_trades_per_day:
            return False, "MAX_TRADES_REACHED"
        if self.day_pnl_rupees <= -abs(cfg.daily_loss_limit):
            return False, "DAILY_LOSS_LIMIT_REACHED"
        return True, "OK"

    def execute_signal(self, signal: OptionBuySignalV2, cfg: UnderlyingConfigV2) -> bool:
        # 1. Parameter Preparation (Moved to top for FORCE_TRADE)
        # Lot-size resolution order: signal.meta -> cfg.lot_size -> broker quote
        lot = int((signal.meta or {}).get('lot_size') or getattr(cfg, 'lot_size', 1) or self.broker.get_lot_size(signal.option_symbol) or 1)
        
        # FYERS behaviour note:
        # For MCX options, FYERS rejects MARKET orders and order sizing is typically done in LOTS.
        # So we send qty as number of lots (default 1) for MCX option symbols.
        is_mcx_opt = str(signal.option_symbol).startswith('MCX:') and (str(signal.option_symbol).endswith('CE') or str(signal.option_symbol).endswith('PE'))
        lots = int((signal.meta or {}).get('lots') or int(getattr(cfg, 'order_lots', 1) or 1))
        
        if is_mcx_opt:
            qty = max(1, lots)
            _logger_v2.info('ORDER_SIZING | symbol=%s | lots=%s | (lot_size=%s ignored for MCX opts) | qty=%s', signal.option_symbol, lots, lot, qty)
        else:
            qty = max(1, lot * lots)
            _logger_v2.info('ORDER_SIZING | symbol=%s | lot_size=%s | lots=%s (from order_lots) | qty=%s (Total Units)', signal.option_symbol, lot, lots, qty)
            print(f"[ORDER SIZING] Lot Size: {lot} x Order Lots: {lots} = Final Qty: {qty} units")

        # Dynamic SL/TP tuned for option buying
        # Goal: avoid big premium drawdown; take quicker profits when delta is strong.
        delta = float(signal.greeks.delta or 0.0)
        gamma = float(signal.greeks.gamma or 0.0)
        theta = float(signal.greeks.theta or 0.0)
        
        # Base SL/TP from cfg
        sl_pct = float(getattr(cfg, 'default_sl_pct', 0.25))
        tp_pct = float(getattr(cfg, 'default_tp_pct', 0.40))

        # If delta is strong, prefer tighter SL and faster TP (higher hit rate).
        if delta >= 0.60:
            sl_pct = min(sl_pct, 0.12)
            tp_pct = max(tp_pct, 0.18)
        elif delta >= 0.50:
            sl_pct = min(sl_pct, 0.14)
            tp_pct = max(tp_pct, 0.20)
        else:
            # lower delta trades are riskier: do NOT widen SL; keep tight and rely on trailing/time exit
            sl_pct = min(sl_pct, 0.16)
            tp_pct = max(tp_pct, 0.22)

        # High gamma => moves fast: allow a slightly larger TP and earlier trailing
        if gamma >= 0.02:
            tp_pct = max(tp_pct, 0.22)

        # High negative theta => decay risk: take profits faster and reduce holding time
        if theta < -0.20:
            tp_pct = min(tp_pct, 0.18)
            sl_pct = min(sl_pct, 0.14)

        # Trailing: start earlier for safer exits
        trailing_start_pct = 0.12 if delta >= 0.55 else 0.15
        trailing_lock_pct = 0.06 if delta >= 0.55 else 0.08

        # Prepare context
        votes_buy = int((signal.meta or {}).get('votes_buy') or (signal.meta or {}).get('buy') or 0)
        votes_sell = int((signal.meta or {}).get('votes_sell') or (signal.meta or {}).get('sell') or 0)
        adx_val = float((signal.meta or {}).get('adx') or 0.0)
        atr_val = float((signal.meta or {}).get('atr') or (signal.meta or {}).get('atr_5') or 0.0)

        # ✅ PROFESSIONAL ATR-BASED RISK (New)
        risk_mgr = ATRRiskManager(point_value=cfg.point_value)
        sl_price_target, tp_price_target, atr_sl_pct = risk_mgr.compute_sl_tp(
            premium=signal.entry_price,
            atr_points=atr_val
        )
        
        # Override the defaults with ATR-calculated ones if ATR is valid
        if atr_val > 0:
            sl_pct = atr_sl_pct
            _logger_v2.info("ATR_RISK | atr=%.2f | sl_price=%.2f | tp_price=%.2f | sl_pct=%.2f%%", 
                            atr_val, sl_price_target, tp_price_target, sl_pct * 100)

        indsP = {
            "underlying": str(signal.underlying),
            "option_type": str(signal.option_type),
            "strike": float(signal.strike or 0.0),
            "expiry": str(signal.expiry),
            "delta": float(signal.greeks.delta or 0.0),
            "gamma": float(signal.greeks.gamma or 0.0),
            "theta": float(signal.greeks.theta or 0.0),
            "vega": float(signal.greeks.vega or 0.0),
            "iv": float(signal.greeks.iv or 0.0),
            "pcr": float(signal.pcr or 0.0),
            "vix": float(signal.vix or 0.0),
            "momentum": str(signal.momentum),
            "score": float(signal.score or 0.0),
            "expectancy": float(signal.expectancy or 0.0),
            "time_stop_min": int(getattr(cfg, 'time_stop_min', 12)),
            "meta": signal.meta,
            "votes_buy": votes_buy,
            "votes_sell": votes_sell,
            "adx": adx_val,
            "atr": atr_val,
            "sl_pct_dynamic": sl_pct,
            "tp_pct_dynamic": tp_pct
        }
        self.active_expectancy = float(signal.expectancy or 0.0)

        # ✅ FIX 4: Detailed debug log (Requested)
        print(f"""
[ENTRY CHECK]
signal={bool(signal)} 
symbol={signal.option_symbol if signal else 'None'}
trades_today={self.trades_today}
max_trades={cfg.max_trades_per_day}
day_pnl={self.day_pnl_rupees}
limit={cfg.daily_loss_limit}
active_pos={self.om.active}
atr={atr_val}
sl_price={sl_price_target}
tp_price={tp_price_target}
""")

        # Inject overrides into OrderManager before process_entry
        self.om.sl_price_override = sl_price_target
        self.om.tp_price_override = tp_price_target

        # ✅ FIX 3 & 5: Force Entry Logic (Implementation)
        FORCE_TRADE = False  # Respect all safety checks including vote
        
        if FORCE_TRADE and signal and not self.om.active:
            print(f"[FORCE-TRADE] EXECUTION for {signal.option_symbol}")
            print(f"[EXECUTION] Placing FORCED {signal.option_type} order for {signal.option_symbol} (Qty: {qty})")
            
            # Explicit execution call bypassing other checks
            entered = self.om._process_entry(
                symbol=signal.option_symbol,
                qty=qty,
                side="BUY",
                point_value=cfg.point_value,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                trailing_start_pct=trailing_start_pct,
                trailing_lock_pct=trailing_lock_pct,
                indsP=indsP,
            )
            if entered:
                self.trades_today += 1
                return True
            else:
                print(f"[ERROR] FORCE-TRADE failed in _process_entry")
                return False

        # -------------------------
        # Standard Clean Execution Flow (Guardrails applied)
        # -------------------------
        ok, why = self.can_trade(cfg)
        if not ok:
            _logger_v2.info("TRADE_BLOCKED | %s | %s", signal.underlying, why)
            return False

        # High-confidence gate
        min_votes = int(getattr(cfg, 'min_votes', 0) or 0)
        min_adx = float(getattr(cfg, 'min_adx', 0.0) or 0.0)
        require_hq = bool(getattr(cfg, 'require_high_confidence', False))

        if require_hq:
            if signal.option_type == 'CE' and votes_buy < min_votes:
                _logger_v2.info('HQ_SKIP | %s | votes_buy=%s < %s', signal.option_symbol, votes_buy, min_votes)
                return False
            if signal.option_type == 'PE' and votes_sell < min_votes:
                _logger_v2.info('HQ_SKIP | %s | votes_sell=%s < %s', signal.option_symbol, votes_sell, min_votes)
                return False
            if adx_val and adx_val < min_adx:
                _logger_v2.info('HQ_SKIP | %s | adx=%.2f < %.2f', signal.option_symbol, adx_val, min_adx)
                return False

        entered = self.om._process_entry(
            symbol=signal.option_symbol,
            qty=qty,
            side="BUY",
            point_value=cfg.point_value,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            trailing_start_pct=trailing_start_pct,
            trailing_lock_pct=trailing_lock_pct,
            indsP=indsP,
        )
        if entered:
            self.trades_today += 1
            # Telegram Entry Alert
            send_telegram(f"🚀 [ENTRY] {signal.option_symbol} | Side: {signal.option_type} | Strike: {signal.strike} | Price: {self.om.entry_price:.2f}")
        return entered

    def on_trade_closed(self, pnl_rupees: float, exit_price: float, reason: str) -> None:
        self.day_pnl_rupees += float(pnl_rupees)
        if self.om.symbol:
            # Prepare Greeks for log
            inds = self.om.indsP or {}
            self.journal.log_trade(
                self.om.symbol, 
                inds.get("option_type", "N/A"),
                inds.get("strike", 0),
                self.om.entry_price,
                exit_price,
                inds.get("delta", 0),
                inds.get("gamma", 0),
                inds.get("theta", 0),
                self.active_expectancy,
                reason
            )
            # Telegram Exit Alert
            send_telegram(f"📉 [EXIT] {self.om.symbol} | Price: {exit_price:.2f} | PnL: {pnl_rupees:+.2f} | Reason: {reason}")


# ---------------------------------------------------------------------------
# V2 Trading Bot
# ---------------------------------------------------------------------------

class TradingBotV2:
    """
    Minimal orchestrator for the refactored option buying flow.
    """
    WEIGHTS = {
        "supertrend": 0.30,
        "ema": 0.20,
        "vwap": 0.15,
        "rsi": 0.15,
        "volume": 0.20,
    }

    def get_weighted_bias(self, symbol, inds):
        """
        Decide TREND ONLY (BULLISH / BEARISH / NEUTRAL)
        """
        score = 0.0
        direction = 0
        details = []

        # Helper for safe extraction
        def _get_val(k, default=0):
            v = inds.get(k, default)
            if hasattr(v, 'iloc'): return float(v.iloc[-1])
            return float(v) if v is not None else default

        # ADX gate
        adx = _get_val("adx", 0)
        if adx < 15: # Relaxed ADX slightly from 18 to 15
            # print(f"[DEBUG] {symbol} ADX too low: {adx:.1f}")
            return "NEUTRAL", inds

        # SuperTrend (index trend)
        st14 = _get_val("st_14_2_signal", 0)
        st21 = _get_val("st_21_1_signal", 0)
        
        if st14 == 1 and st21 == 1:
            score += self.WEIGHTS["supertrend"]
            direction += 1
            details.append(f"ST BULL (+{self.WEIGHTS['supertrend']})")
        elif st14 == -1 and st21 == -1: # Fixed: was checking for 0
            score += self.WEIGHTS["supertrend"]
            direction -= 1
            details.append(f"ST BEAR (+{self.WEIGHTS['supertrend']})")

        # EMA alignment
        ema5 = _get_val("ema_5", 0)
        ema9 = _get_val("ema_9", 0)
        if ema5 > ema9:
            score += self.WEIGHTS["ema"]
            direction += 1
            details.append(f"EMA BULL (+{self.WEIGHTS['ema']})")
        elif ema5 < ema9:
            score += self.WEIGHTS["ema"]
            direction -= 1
            details.append(f"EMA BEAR (+{self.WEIGHTS['ema']})")

        # VWAP
        ltp = self.broker.get_ltp(symbol)
        vwap_val = _get_val("vwap", 0) or _get_val("VWAP", 0)
        if ltp > vwap_val * 1.001:
            score += self.WEIGHTS["vwap"]
            direction += 1
            details.append(f"VWAP BULL (+{self.WEIGHTS['vwap']})")
        elif ltp < vwap_val * 0.999:
            score += self.WEIGHTS["vwap"]
            direction -= 1
            details.append(f"VWAP BEAR (+{self.WEIGHTS['vwap']})")

        # RSI (Added)
        rsi = _get_val("rsi", 50)
        if rsi > 55:
            score += self.WEIGHTS["rsi"]
            if direction >= 0: direction += 1
            details.append(f"RSI BULL (+{self.WEIGHTS['rsi']})")
        elif rsi < 45:
            score += self.WEIGHTS["rsi"]
            if direction <= 0: direction -= 1
            details.append(f"RSI BEAR (+{self.WEIGHTS['rsi']})")

        # Volume (Added)
        vol_ratio = _get_val("volume_ratio", 1.0)
        if vol_ratio > 1.2:
            score += self.WEIGHTS["volume"]
            details.append(f"VOL BULL (+{self.WEIGHTS['volume']})")

        if score >= 0.65:
            res = "BULLISH" if direction > 0 else "BEARISH"
            print(f"[BIAS] {symbol} | {res} | Score: {score:.2f} | Details: {', '.join(details)}")
            return res, inds

        # print(f"[DEBUG] {symbol} NEUTRAL | Score: {score:.2f} | Need: 0.65 | Details: {', '.join(details)}")
        return "NEUTRAL", inds

    def option_buying_vote(self, fut_inds, opt_inds, option_type):
        """
        OPTION BUYING VOTING ENGINE - EXPIRY DAY VERSION
        
        fut_inds: Indicators for Underlying/Future (Check TREND + EMA + ATR here)
        opt_inds: Indicators for Option (Greeks validation)
        option_type: "CE" or "PE"
        
        EXPIRY DAY STRATEGY:
        - Uses EMA 200/20 alignment instead of lagging EFI/Momentum
        - Uses ATR ratio to detect market EXPANSION (trending) vs COMPRESSION (ranging)
        - Synchronized with MERGED.py institutional logic
        """

        score = 0

        # FUTURE (Trend Context)
        st_14_2 = fut_inds.get("st_14_2_signal", 0)
        st_21_1 = fut_inds.get("st_21_1_signal", 0)
        
        # Get EMA and ATR from future for new logic
        close_fut = fut_inds.get("close", 0)
        ema_5 = fut_inds.get("ema_5", fut_inds.get("ema5", 0))
        ema_20 = fut_inds.get("ema_20", fut_inds.get("ema20", 0))
        ema_200 = fut_inds.get("ema_200", fut_inds.get("ema200", 0))

        # ATR regime: prefer precomputed atr_ratio (ATR / ATR_MA50)
        atr_fut = fut_inds.get("atr", fut_inds.get("ATR", 0))
        atr_ratio = fut_inds.get("atr_ratio", None)
        if atr_ratio is None:
            atr_base = fut_inds.get("atr_ma50", 0) or atr_fut
            atr_ratio = (atr_fut / atr_base) if atr_fut and atr_base else 1.0

        print(f"\n{'='*70}")
        print(f"[OPTION BUY VOTE - EXPIRY DAY] Type: {option_type}")
        print(f"  Future/Index Trend : ST14={st_14_2}, ST21={st_21_1}")
        print(f"  EMA Alignment      : Close={close_fut:.1f}, EMA5={ema_5:.1f}, EMA20={ema_20:.1f} (EMA200={ema_200:.1f})")
        print(f"  Market Regime      : ATR Ratio={atr_ratio:.2f} ({'EXPANSION' if atr_ratio > 1.2 else 'COMPRESSION' if atr_ratio < 0.8 else 'NEUTRAL'})")

        # Debug: capture and show original ATR sources to diagnose unexpected zeros
        try:
            atr_ratio_value = float(atr_ratio)
        except Exception:
            atr_ratio_value = 1.0
        print(f"  [DEBUG] fut_inds.atr={fut_inds.get('atr', None)!r}, fut_inds.atr_ratio={fut_inds.get('atr_ratio', None)!r}, computed_atr_ratio={atr_ratio_value!r}")
        print(f"\n[VOTE SCORING]")

        # ─────────────────────────────────────────
        # 1️⃣ TREND DIRECTION (FROM FUTURE/INDEX)
        # ─────────────────────────────────────────
        if option_type == "CE":
            if st_14_2 == 1 and st_21_1 == 1:
                score += 2
                print(f"  ✅ FUT BULLISH TREND (CE BUY): +2")
            else:
                print(f"  ❌ FUT Trend not bullish (st14={st_14_2}, st21={st_21_1})")

        elif option_type == "PE":
            if st_14_2 == -1 and st_21_1 == -1:
                score += 2
                print(f"  ✅ FUT BEARISH TREND (PE BUY): +2")
            else:
                print(f"  ❌ FUT Trend not bearish (st14={st_14_2}, st21={st_21_1})")

        # ─────────────────────────────────────────
        # 2️⃣ TREND ALIGNMENT (EMA 200/20) + EXPANSION (ATR RATIO)
        # ─────────────────────────────────────────
        # EXPIRY DAY STRATEGY: Use EMA alignment + ATR ratio instead of EFI/Momentum
        # Synchronized with MERGED.py institutional logic
        
        # Get EMA and ATR data from FUTURE/INDEX indicators (not option)
        close_fut = fut_inds.get("close", 0)
        ema_20 = fut_inds.get("ema_20", fut_inds.get("ema20", 0))
        ema_200 = fut_inds.get("ema_200", fut_inds.get("ema200", 0))
        atr_fut = fut_inds.get("atr", 0)
        
        # ATR ratio is computed earlier from fut_inds (atr_ratio) or atr_ma50 fallback.
        # Do NOT overwrite it by dividing by `close_fut` (that makes it nearly zero).
        # Keep the previously computed `atr_ratio` above.
        # Bind to a local immutable copy used for scoring to avoid accidental later mutation.
        try:
            atr_ratio = float(atr_ratio)
        except Exception:
            atr_ratio = 1.0
        atr_ratio_value = atr_ratio
        
        if option_type == "CE":
            # CE (Call) requires: close > ema_20 > ema_200 (bullish alignment)
            if close_fut and ema_20 and ema_200:
                if close_fut > ema_5 > ema_20:
                    score += 1
                    print(f"  ✅ FUT EMA BULLISH (Close {close_fut:.1f} > EMA5 {ema_5:.1f} > EMA20 {ema_20:.1f}): +1")
                else:
                    print(f"  ❌ FUT EMA NOT BULLISH (Close={close_fut:.1f}, EMA5={ema_5:.1f}, EMA20={ema_20:.1f})")
            else:
                print(f"  ⚠️  FUT EMA data missing (Close={close_fut}, EMA5={ema_5}, EMA20={ema_20})")
            
            # ATR ratio > 1.2 = market in EXPANSION (trending)
            if atr_ratio_value > 1.2:
                score += 1
                print(f"  ✅ FUT ATR EXPANSION (ATR Ratio {atr_ratio_value:.2f} > 1.2): +1")
            else:
                print(f"  ❌ FUT ATR weak (ATR Ratio {atr_ratio_value:.2f} <= 1.2)")
        
        elif option_type == "PE":
            # PE (Put) requires: close < ema_20 < ema_200 (bearish alignment)
            if close_fut and ema_20 and ema_200:
                if close_fut < ema_5 < ema_20:
                    score += 1
                    print(f"  ✅ FUT EMA BEARISH (Close {close_fut:.1f} < EMA5 {ema_5:.1f} < EMA20 {ema_20:.1f}): +1")
                else:
                    print(f"  ❌ FUT EMA NOT BEARISH (Close={close_fut:.1f}, EMA5={ema_5:.1f}, EMA20={ema_20:.1f})")
            else:
                print(f"  ⚠️  FUT EMA data missing (Close={close_fut}, EMA5={ema_5}, EMA20={ema_20})")
            
            # ATR ratio > 1.2 = market in EXPANSION (trending)
            if atr_ratio_value > 1.2:
                score += 1
                print(f"  ✅ FUT ATR EXPANSION (ATR Ratio {atr_ratio_value:.2f} > 1.2): +1")
            else:
                print(f"  ❌ FUT ATR weak (ATR Ratio {atr_ratio_value:.2f} <= 1.2)")

        # ─────────────────────────────────────────
        # FINAL DECISION
        # ─────────────────────────────────────────
        print(f"\n[VOTE RESULT] FINAL SCORE: {score}/4 (PASS = 4)")
        print(f"[VOTE RESULT] {'✅ BUY OPTION' if score == 4 else '❌ NO TRADE'}")
        print(f"{'='*70}\n")

        return score


    # def option_buying_vote(self, inds, bias="BULLISH"):
    #     """
    #     Entry confirmation AFTER strike selection
    #     Uses SuperTrend + Elder Force + Momentum
    #     """
    #     score = 0
    #     ef = inds.get("elder_force_now", 0)
    #     mom = inds.get("momentum_now", 0)
    #     st_14_2 = inds.get("st_14_2_signal", 0)
    #     st_21_1 = inds.get("st_21_1_signal", 0)

    #     # DEBUG LOGGING
    #     print(f"\n{'='*70}")
    #     print(f"[VOTE DEBUG] Bias: {bias}")
    #     print(f"[VOTE DEBUG] Elder Force: {ef:,.0f}")
    #     print(f"[VOTE DEBUG] Momentum: {mom:.2f}")
    #     print(f"[VOTE DEBUG] SuperTrend 14,2: {st_14_2}")
    #     print(f"[VOTE DEBUG] SuperTrend 21,1: {st_21_1}")
    #     print(f"\n[VOTE SCORING]")

    #     # SuperTrend alignment
    #     if "BULL" in bias:
    #         if st_14_2 == 1 and st_21_1 == 1:
    #             score += 2
    #             print(f"  ✅ SuperTrend aligned (BULL): +2 → score={score}")
    #         else:
    #             print(f"  ❌ SuperTrend NOT aligned (st_14_2={st_14_2}, st_21_1={st_21_1})")
            
    #         # Thresholds for BULLISH move
    #         # Sensex index needs large EF, typically > 10M-20M. Using 10M as more reachable.
    #         #ef_threshold = 100000 if "SENSEX" in str(inds.get("symbol", "")) else 250000
    #         ef_threshold = 1 if "SENSEX" in str(inds.get("symbol", "")) else 2
    #         if ef > ef_threshold:
    #             score += 1
    #             print(f"  ✅ Elder Force bullish ({ef:,.0f} > {ef_threshold/1e6}M): +1 → score={score}")
    #         else:
    #             print(f"  ❌ Elder Force too low ({ef:,.0f} <= {ef_threshold/1e6}M)")
                
    #         if mom > 5: # Relaxed momentum from 100 to 50
    #             score += 1
    #             print(f"  ✅ Momentum bullish ({mom:.2f} > 5): +1 → score={score}")
    #         else:
    #             print(f"  ❌ Momentum too low ({mom:.2f} <= 5)")
                
    #     elif "BEAR" in bias:
    #         if st_14_2 == -1 and st_21_1 == -1: # Fixed: match Bear signal
    #             score += 2
    #             print(f"  ✅ SuperTrend aligned (BEAR): +2 → score={score}")
    #         else:
    #             print(f"  ❌ SuperTrend NOT aligned (st_14_2={st_14_2}, st_21_1={st_21_1})")
            
    #         # Thresholds for BEARISH move (SENSEX index needs large EF, typically > 10M-20M)
    #         # Make thresholds symbol-dependent or slightly more relaxed if needed.
    #         # Using 10M as a more reachable threshold for Sensex unless volume is massive.
    #         ef_threshold = 10 if "SENSEX" in str(inds.get("symbol", "")) else 2
    #         if ef < -ef_threshold:
    #             score += 1
    #             print(f"  ✅ Elder Force bearish ({ef:,.0f} < -{ef_threshold/1e6}M): +1 → score={score}")
    #         else:
    #             print(f"  ❌ Elder Force too high ({ef:,.0f} >= -{ef_threshold/1e6}M)")
                
    #         if mom < -50: # Relaxed momentum from -100 to -50
    #             score += 1
    #             print(f"  ✅ Momentum bearish ({mom:.2f} < -50): +1 → score={score}")
    #         else:
    #             print(f"  ❌ Momentum too high ({mom:.2f} >= -50)")

    #     print(f"\n[VOTE RESULT] FINAL SCORE: {score}/4 (threshold: 3)")
    #     print(f"[VOTE RESULT] {'✅ PASS - Trade Allowed' if score >= 3 else '❌ FAIL - Trade Blocked'}")
    #     print(f"{'='*70}\n")
        
    #     return score

    def calculate_indicators(self, symbol, timeframe, pivot_data=None):
        if self.mkt.bot and hasattr(self.mkt.bot, "indicator_calculator"):
            # If pivot_data not provided, try to get from cache (especially for options)
            if not pivot_data:
                # Check directly for symbol
                if symbol in self.pivots_cache:
                    pivot_data = self.pivots_cache[symbol]
                
                # If still None and it's an option, try underlying
                if not pivot_data:
                    resolved = SymbolRouterV3.resolve(symbol)
                    if hasattr(resolved, 'analysis_symbol') and resolved.analysis_symbol in self.pivots_cache:
                        pivot_data = self.pivots_cache[resolved.analysis_symbol]

            # Ensure we have some base indicators from the calculator
            inds = self.mkt.bot.indicator_calculator.calculate_indicators(symbol, timeframe, pivot_data=pivot_data)
            
            # Add Elder Force and Momentum indicators for option_buying_vote
            try:
                # Synchronize with optionbuying_sensex.py: duration=10 for EMA convergence
                ohlc_df = fetchOHLC1(symbol, timeframe, duration=10)
                
                if ohlc_df is not None and not ohlc_df.empty and len(ohlc_df) >= 20:
                    inds["symbol"] = symbol
                    inds["close"] = float(ohlc_df["Close"].iloc[-1])
                    
                    # Diagnostic: Print the OHLC data the bot is actually seeing
                    print(f"[DIAG] Data tail for {symbol}:\n{ohlc_df[['Close','Volume']].tail(3)}")
                    
                    # Use smoothed helpers
                    ef_series = elder_force_index(ohlc_df, period=13)
                    mom_series = momentum_indicator(ohlc_df, period=10)
                    
                    inds["elder_force_now"] = float(ef_series.iloc[-1]) if pd.notna(ef_series.iloc[-1]) else 0.0
                    inds["momentum_now"] = float(mom_series.iloc[-1]) if pd.notna(mom_series.iloc[-1]) else 0.0

                    # Diagnostics: Matches screen?
                    print(f"[IND-DIAG] {symbol} | Price={inds['close']:.2f} | EFI={inds['elder_force_now']:,.0f} | MOM={inds['momentum_now']:.2f}")
                    if len(ohlc_df) < 50:
                        print(f"  [WARN] Low OHLC count: {len(ohlc_df)} bars. EFI may need more data.")

                    # Ensure ATR is available for risk management
                    if "atr" not in inds or not inds["atr"]:
                        atr_series = atr(ohlc_df, period=14)
                        inds["atr"] = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
                    
                    # FIX 6: Log actual numeric values (debug)
                    # logger.info(...)
                    print(f"[IND] {symbol} | EFI={inds['elder_force_now']:.2f} | MOM={inds['momentum_now']:.2f} | VOL={ohlc_df['Volume'].iloc[-1]}")
                    
                else:
                    # Set defaults if OHLC data unavailable
                    inds["elder_force_now"] = 0
                    inds["momentum_now"] = 0
                    if "atr" not in inds: inds["atr"] = 0
                    print(f"[INDICATORS] OHLC data unavailable (strict fetch) for {symbol}")
            except Exception as e:
                # If calculation fails, set to 0 to avoid blocking
                inds["elder_force_now"] = 0
                inds["momentum_now"] = 0
                if "atr" not in inds: inds["atr"] = 0
                print(f"[ERROR] Failed to calculate indicators for {symbol}: {e}")
            
            return inds
        return {"error": "Indicator calculator not available"}

    def get_atm_strike(self, symbol):
        ltp = self.broker.get_ltp(symbol)
        if not ltp or ltp <= 0:
            return 0
        resolved = SymbolRouterV3.resolve(symbol)
        cfg = self.underlying_configs.get(resolved.underlying_key)
        step = cfg.strike_step if cfg else 100
        return round(ltp / step) * step

    def __init__(
        self,
        broker: BrokerAPIV2,
        mkt: MarketDataAPIV2,
        *,
        underlying_configs: Dict[str, UnderlyingConfigV2],
        poll_seconds: float = 5,
    ) -> None:
        self.broker = broker
        self.mkt = mkt
        self.underlying_configs = underlying_configs
        self.poll_seconds = float(poll_seconds)

        self.engine = OptionEngineV3(self.mkt, self.underlying_configs)
        self.om = OrderManagerV2(self.broker, sl_manager=UnifiedStopLossManagerV2())
        self.bridge = OptionOrderManagerV2(self.broker, self.om)
        self.pivots_cache: Dict[str, Any] = {}


    def _get_voting_signal(self, symbol: str) -> Tuple[str, Dict[str, Any]]:
        """
        Directional voting system for intraday option buying.
        Determines MARKET BIAS only (CE vs PE).
        """

        try:
            # --- Load CPR pivots ---
            pivots = self.pivots_cache.get(symbol)
            if not pivots:
                pivots = self.mkt.bot.get_cpr_levels_with_fallback(symbol).get(symbol, {})
                if pivots:
                    self.pivots_cache[symbol] = pivots

            # --- Indicators ---
            inds = self.mkt.bot.indicator_calculator.calculate_indicators(
                symbol, "5", pivot_data=pivots
            )
            if "error" in inds:
                return "NEUTRAL", {}

            # Helper for safe extraction
            def _get_val(k, default=0):
                v = inds.get(k, default)
                if hasattr(v, 'iloc'): return float(v.iloc[-1])
                return float(v) if v is not None else default

            buy_votes = 0
            sell_votes = 0

            # -------------------------
            # 1️⃣ TREND STRENGTH GATE
            # -------------------------
            adx = _get_val("adx", 0)
            if adx < 18:
                return "NEUTRAL", inds   # no trend → no trade

            # -------------------------
            # 2️⃣ EMA MOMENTUM
            # -------------------------
            ema5 = _get_val("ema_5")
            ema9 = _get_val("ema_9")
            if ema5 and ema9:
                buy_votes += ema5 > ema9
                sell_votes += ema5 < ema9

            # -------------------------
            # 3️⃣ VWAP POSITION
            # -------------------------
            ltp = self.broker.get_ltp(symbol)
            vwap = _get_val("vwap")
            if ltp and vwap:
                buy_votes += ltp > vwap * 1.001
                sell_votes += ltp < vwap * 0.999

            # -------------------------
            # 4️⃣ SUPERTREND DIRECTION
            # -------------------------
            st = _get_val("st_main_trend")
            buy_votes += st == 1
            sell_votes += st == -1

            # -------------------------
            # 5️⃣ MOMENTUM CONFIRMATION (RSI)
            # -------------------------
            rsi = _get_val("rsi", 50)
            buy_votes += rsi > 50
            sell_votes += rsi < 50

            print(
                f"[VOTING] {symbol} | BUY={buy_votes} | SELL={sell_votes} | ADX={adx:.1f}"
            )

            # -------------------------
            # FINAL DECISION (5 signals)
            # -------------------------
            if buy_votes >= 4 and buy_votes > sell_votes:
                return "BULLISH_FORCE", inds
            if sell_votes >= 4 and sell_votes > buy_votes:
                return "BEARISH_FORCE", inds

            if buy_votes == 3 and buy_votes > sell_votes:
                return "BULLISH", inds
            if sell_votes == 3 and sell_votes > buy_votes:
                return "BEARISH", inds

            return "NEUTRAL", inds

        except Exception as e:
            print(f"[ERROR] Voting failed: {e}")
            return "NEUTRAL", {}


    # def _get_voting_signal(self, symbol: str) -> Tuple[str, Dict[str, Any]]:
        """
        Directional voting system.
        Determines MARKET BIAS only (CE vs PE).
        """
        try:
            pivots = self.pivots_cache.get(symbol)
            if not pivots:
                pivots = self.mkt.bot.get_cpr_levels_with_fallback(symbol).get(symbol, {})
                if pivots:
                    self.pivots_cache[symbol] = pivots

            inds = self.mkt.bot.indicator_calculator.calculate_indicators(
                symbol, "5", pivot_data=pivots
            )
            if "error" in inds:
                return "NEUTRAL", {}

            buy_votes = 0
            sell_votes = 0

            # EMA alignment
            ema5 = last(inds.get("ema_5"))
            ema_9 = last(inds.get("ema_9"))
            if ema5 and ema_9:
                buy_votes += ema5 > ema_9
                sell_votes += ema5 < ema_9

            # # RSI
            # rsi = last(inds.get("rsi"))
            # if rsi:
            #     buy_votes += rsi > 55
            #     sell_votes += rsi < 45

            # VWAP
            ltp = self.broker.get_ltp(symbol)
            vwap_val = last(inds.get("vwap"))
            if vwap_val and ltp:
                buy_votes += ltp > vwap_val * 1.001
                sell_votes += ltp < vwap_val * 0.999

            # SuperTrend
            st_trend = last(inds.get("st_main_trend"))
            buy_votes += st_trend == 1
            sell_votes += st_trend == -1

            print(
                f"[VOTING] {symbol} | BUY={buy_votes} | SELL={sell_votes} | ADX={last(inds.get('adx',0)):.1f}"
            )

            # Final bias decision
            if buy_votes >= 5 and buy_votes > sell_votes:
                return "BULLISH_FORCE", inds
            if sell_votes >= 5 and sell_votes > buy_votes:
                return "BEARISH_FORCE", inds

            if buy_votes >= 4 and buy_votes > sell_votes:
                return "BULLISH", inds
            if sell_votes >= 4 and sell_votes > buy_votes:
                return "BEARISH", inds

            return "NEUTRAL", inds

        except Exception as e:
            print(f"[ERROR] Voting failed: {e}")
            return "NEUTRAL", {}

    def _adx_ok(self, inds: Dict[str, Any]) -> bool:
        return inds.get("adx", 0) >= 20

    def _pcr_ok(self, bias: str, pcr: float) -> bool:
        if bias.startswith("BULLISH") and pcr > 1.2:
            return False
        if bias.startswith("BEARISH") and pcr < 0.7:
            return False
        return True

    def _greeks_ok(self, greeks: Dict[str, float]) -> bool:
        delta = abs(greeks.get("delta", 0))
        gamma = greeks.get("gamma", 0)
        theta = abs(greeks.get("theta", 0))

        if delta < 0.40 or delta > 0.65:
            return False
        if gamma <= 0:
            return False
        if theta > 1.5:
            return False
        return True

    def _expectancy_ok(self, inds: Dict[str, Any], greeks: Dict[str, float]) -> bool:
        score = 0
        # Trend strength
        if inds.get("adx", 0) >= 25:
            score += 1
        # Delta quality
        if 0.45 <= abs(greeks.get("delta", 0)) <= 0.60:
            score += 1
        # Low decay
        if abs(greeks.get("theta", 0)) < 1.0:
            score += 1
        return score >= 2

    def should_enter_trade(
        self,
        symbol: str,
        pcr: float,
        greeks: Dict[str, float],
    ) -> Tuple[bool, str, str, Dict[str, Any]]:
        """
        FINAL decision engine.
        Returns: (allow_trade, bias, CE_or_PE, indicators)
        """
        bias, inds = self._get_voting_signal(symbol)

        if bias == "NEUTRAL":
            return False, bias, "", inds

        # Pass extra context to gates
        adx = inds.get("adx", 0)

        if not self._adx_ok(inds):
            print(f"[GATE] {symbol} | ADX < 20 ({adx:.1f}) | SKIP")
            return False, bias, "", inds

        if not self._pcr_ok(bias, pcr):
            print(f"[GATE] {symbol} | PCR Context {pcr:.2f} weak for {bias} | SKIP")
            return False, bias, "", inds

        if not self._greeks_ok(greeks):
            print(f"[GATE] {symbol} | Greeks weak (D={greeks.get('delta'):.2f}, T={greeks.get('theta'):.2f}) | SKIP")
            return False, bias, "", inds

        if not self._expectancy_ok(inds, greeks):
            # We log but allow if bias is FORCE? No, strict discipline.
            print(f"[GATE] {symbol} | Expectancy Proxy weak | SKIP")
            return False, bias, "", inds

        option_type = "CE" if bias.startswith("BULLISH") else "PE"
        return True, bias, option_type, inds

    def run(self, symbol: str, *, max_cycles: Optional[int] = None) -> None:
        print(f"[BOT] Option Buying Engine Started for {symbol}")
        send_telegram(f"🚀 Option Buying Engine Started for {symbol}")
        resolved = SymbolRouterV3.resolve(symbol)
        if not resolved:
            raise ValueError(f"Symbol not resolved: {symbol}")

        cycles = 0
        while True:
            cycles += 1
            if max_cycles is not None and cycles > max_cycles:
                _logger_v2.info("STOP | max_cycles=%s reached", max_cycles)
                return

            try:
                cfg = self.underlying_configs.get(resolved.underlying_key)
                if not cfg:
                    _logger_v2.error("CONFIG_MISSING | %s", resolved.underlying_key)
                    break
                
                self.bridge.reset_day_if_needed()

                if not self.om.active:
                    # 1️⃣ Calculate indicators on INDEX
                    timeframe = "5" # Default
                    # Try to get cached pivots
                    cached_pivots = self.pivots_cache.get(resolved.analysis_symbol)
                    inds = self.calculate_indicators(resolved.analysis_symbol, timeframe, pivot_data=cached_pivots)
                    
                    # Update cache if new pivots found
                    if inds.get("pivot_data"):
                        self.pivots_cache[resolved.analysis_symbol] = inds["pivot_data"]
                    if "error" in inds:
                        _time.sleep(self.poll_seconds)
                        continue

                    # 2️⃣ Decide trend
                    bias_str, inds = self.get_weighted_bias(resolved.analysis_symbol, inds)
                    if bias_str == "NEUTRAL":
                        _time.sleep(self.poll_seconds)
                        continue

                    # 3️⃣ Select option type
                    desired_type = "CE" if bias_str == "BULLISH" else "PE"

                    # 4️⃣ Proposed Signal Selection (Consolidated)
                    # Fetch chain ONCE to be used by all methods (FIX for rate limits)
                    raw_chain = self.mkt.get_option_chain_raw(resolved.option_chain_symbol, strikecount="10")
                    
                    # Extract index-level PCR from chain
                    pcr = 1.0
                    if raw_chain and raw_chain.get("s") == "ok":
                         ch_rows = (raw_chain.get("data") or {}).get("optionsChain") or []
                         if ch_rows: pcr = float(ch_rows[0].get("pcr") or 1.0)
                    
                    is_sensex = ("SENSEX" in resolved.underlying_key)
                    
                    # GET PROPOSED SIGNAL ONCE
                    proposed_signal = self.engine.analyze(
                        resolved,
                        desired_type=desired_type,
                        force_atm=is_sensex,
                        raw_chain=raw_chain,
                        underlying_ltp=inds["close"]
                    )
                    
                    if not proposed_signal:
                        print(f"[SKIP] No suitable {desired_type} strike found for {resolved.analysis_symbol}")
                        _time.sleep(self.poll_seconds)
                        continue

                    option_symbol = proposed_signal.option_symbol

                    # 5️⃣ Entry Confirmation (Sync Option Momentum only)
                    print(f"[CHECK] Synchronizing indicators for Option {option_symbol}")
                    
                    pivots = self.pivots_cache.get(resolved.analysis_symbol)
                    
                    from concurrent.futures import ThreadPoolExecutor
                    def fetch_inds(s, tf, p=None):
                        return self.calculate_indicators(s, tf, pivot_data=p)
                        
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        # We reuse 'inds' for future, only fetch option Indicators
                        option_task = executor.submit(fetch_inds, option_symbol, timeframe, pivots)
                        
                        inds_fut = inds # Reuse from 1️⃣
                        inds_opt = option_task.result()
                        
                        efi_fut = inds_fut.get("elder_force_now", 0)
                        efi_opt = inds_opt.get("elder_force_now", 0)
                        
                        print(f"[SYNC] FUTURE {resolved.analysis_symbol}: EFI={efi_fut:,.0f} | MOM={inds_fut.get('momentum_now', 0):.2f}")
                        print(f"[SYNC] OPTION {option_symbol}: EFI={efi_opt:,.0f} | MOM={inds_opt.get('momentum_now', 0):.2f}")

                        if (not efi_opt and not inds_opt.get("momentum_now")):
                            print(f"[SKIP] OHLC/Indicators NOT READY (Zero EFI/Mom) for OPTION {option_symbol} — skipping cycle")
                            _time.sleep(self.poll_seconds)
                            continue

                        print(f"[SYNC] Barrier Passed. Proceeding to Vote.")

                    vote = self.option_buying_vote(inds_fut, inds_opt, option_type=desired_type)
                    if vote < 4:
                        print(f"[VOTE] {resolved.analysis_symbol} | Bias={bias_str} | Opt-Vote={vote} < 4 | SKIP")
                        _time.sleep(self.poll_seconds)
                        continue

                    # 6️⃣ PLACE ORDER
                    # Use our proposed_signal since it's already validated
                    signal = proposed_signal
                    # Inject latest meta
                    signal.meta.update({
                        "voting": bias_str, 
                        "vote_score": vote,
                        "votes_buy": vote if desired_type == "CE" else 0,
                        "votes_sell": vote if desired_type == "PE" else 0,
                        "adx": inds_fut.get("adx", 0),
                        "atr": inds_opt.get("atr", 0), 
                        "pcr_index": pcr,
                    })

                    if signal:
                        # Safety: ensure the analysis returned the selected strike
                        if signal.option_symbol != option_symbol:
                            _logger_v2.warning("MISMATCH | Selected %s but analyze returned %s", option_symbol, signal.option_symbol)

                        # TRACKING (Existing legacy logic)
                        if getattr(self.mkt, 'bot', None) is not None:
                            try:
                                self.mkt.bot.ensure_symbol_tracking(signal.option_symbol)
                                self.mkt.bot.activate_symbol_aliases(signal.option_symbol)
                            except Exception: pass

                        # EXECUTION
                        entered = self.bridge.execute_signal(signal, cfg)
                        
                        if entered:
                            print(f"[OK] [ENTRY] {signal.option_symbol} | Bias={bias_str} | Greek-D={signal.greeks.delta:.2f}")


                            # Persist entry under option symbol (data_bot/logs_bot/reports_bot)
# if entered:

                    #     if getattr(self.mkt, 'bot', None) is not None:
                            try:
                                self.mkt.bot.persist_v3_entry(
                                    signal.option_symbol,
                                    {
                                        "event": "ENTRY",
                                        "underlying": signal.underlying,
                                        "symbol": signal.option_symbol,
                                        "qty": int(self.om.qty or 0),
                                        "entry_price": float(self.om.entry_price or 0.0),
                                        "sl_pct": float(self.om.sl_pct),
                                        "tp_pct": float(self.om.tp_pct),
                                        "trailing_start_pct": float(self.om.trailing_start_pct),
                                        "trailing_lock_pct": float(self.om.trailing_lock_pct),
                                        "indsP": dict(self.om.indsP or {}),
                                    },
                                )
                            except Exception as e:
                                _logger_v2.exception(
                                    "[PERSIST] Failed to persist ENTRY for %s: %s",
                                    signal.option_symbol,
                                    e
                                )

                        # --------------------------------------------------
                        # HARD LOOP TRANSITION (CRITICAL)
                        # --------------------------------------------------
                        print(
                            f"[ENGINE] Trade entered for {signal.option_symbol}. "
                            f"Switching to Position Management."
                        )

                        _time.sleep(1.5)   # short cooldown only
                        continue


                        # # Persist entry under option symbol (data_bot/logs_bot/reports_bot)
                        # if entered and getattr(self.mkt, 'bot', None) is not None:
                        #     try:
                        #         self.mkt.bot.persist_v3_entry(
                        #             signal.option_symbol,
                        #             {
                        #                 "event": "ENTRY",
                        #                 "underlying": signal.underlying,
                        #                 "symbol": signal.option_symbol,
                        #                 "qty": int(self.om.qty or 0),
                        #                 "entry_price": float(self.om.entry_price or 0.0),
                        #                 "sl_pct": float(self.om.sl_pct),
                        #                 "tp_pct": float(self.om.tp_pct),
                        #                 "trailing_start_pct": float(self.om.trailing_start_pct),
                        #                 "trailing_lock_pct": float(self.om.trailing_lock_pct),
                        #                 "indsP": dict(self.om.indsP or {}),
                        #             },
                        #         )
                        #         pass
                            
                        #     # CRITICAL FIX: HARD STOP (LOOP TRANSITION)
                        #     # Once a trade is entered, we continue to the next loop iteration.
                        #     # The next iteration will see self.om.active = True and switch to Position Management (else block).
                        #     print(f"[ENGINE] Trade entered for {signal.option_symbol}. Switching to Position Management.")
                        #     _time.sleep(6) 
                        #     continue

                else:
                    assert self.om.symbol is not None
                    # Route persistence/logs to the active option symbol while trade is open
                    if getattr(self.mkt, 'bot', None) is not None:
                        try:
                            self.mkt.bot.activate_symbol_aliases(self.om.symbol)
                        except Exception:
                            pass
                    ltp = float(self.broker.get_ltp(self.om.symbol))
                    try:
                        ep=float(self.om.entry_price or 0.0)
                        sl_price=ep*(1.0-float(self.om.sl_pct or 0.0)) if ep>0 else 0.0
                        tp_price=ep*(1.0+float(self.om.tp_pct or 0.0)) if ep>0 else 0.0
                        peak=float(getattr(self.om,'peak_price',ep) or ep)
                        print(f"[POSITION] {self.om.symbol} | LTP={ltp:.2f} | EP={ep:.2f} | SL≈{sl_price:.2f} | TP≈{tp_price:.2f} | PEAK={peak:.2f} | trailing={getattr(self.om,'trailing_active',False)}")
                    except Exception:
                        pass
                    should_exit, reason = self.om.execute_unified_strategy(ltp)
                    if should_exit:
                        pnl_points = (ltp - self.om.entry_price)
                        pnl_rupees = pnl_points * self.om.point_value * self.om.qty
                        result = self.om._process_exit(reason=reason, ltp=ltp)
                        if result.get("exited"):
                            # Persist exit under option symbol
                            if getattr(self.mkt, 'bot', None) is not None:
                                try:
                                    self.mkt.bot.persist_v3_exit(
                                        result.get("symbol") or "",
                                        {
                                            "event": "EXIT",
                                            "symbol": result.get("symbol"),
                                            "qty": int(result.get("qty") or 0) if isinstance(result.get("qty"), (int, float, str)) else int(getattr(self.om, 'qty', 0) or 0),
                                            "entry_price": float(result.get("entry_price") or 0.0) if result.get("entry_price") is not None else float(ltp - pnl_points),
                                            "exit_price": float(result.get("exit_price") or 0.0),
                                            "pnl_points": float((result.get("exit_price") or ltp) - (ltp - pnl_points)),
                                            "pnl_rupees": float(result.get("pnl_rupees") or pnl_rupees),
                                            "reason": str(result.get("reason") or reason),
                                        },
                                    )
                                except Exception:
                                    pass
                            self.bridge.on_trade_closed(float(pnl_rupees), float(result.get("exit_price") or ltp), str(reason))

                _time.sleep(self.poll_seconds)

            except Exception as e:
                _logger_v2.error("BOT_ERROR | %s\n%s", e, _traceback.format_exc())
                _time.sleep(max(10, self.poll_seconds))


# ---------------------------------------------------------------------------
# V2 Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS_V2: Dict[str, UnderlyingConfigV2] = {
    "SENSEX": UnderlyingConfigV2(
        key="SENSEX",
        point_value=20.0,
        default_sl_pct=0.35,     # fallback only
        default_tp_pct=0.60,
        vix_symbol="NSE:INDIAVIX-INDEX",
        max_vix=22.0,
        max_trades_per_day=2,
        daily_loss_limit=3000.0,
        delta_target_abs=0.50, # Lowered from 0.55
        delta_min_abs=0.01,    # Extremely inclusive (was 0.10)
        delta_max_abs=0.95,    # Increased (was 0.85)
        pcr_bullish_min=1.05,
        pcr_bearish_max=0.95,
        max_spread_pct=0.05,   # Increased (was 0.035)
        strike_step=100.0,
        lot_size=20,
        min_premium=0.5,       # Extremely inclusive (was 5.0)
        max_premium=12000.0,
        time_stop_min=20,
        order_lots=1,
    ),
    "NATGASMINI": UnderlyingConfigV2(
        key="NATGASMINI",
        point_value=250.0,
        # Tighter SL to avoid big premium drawdown; target uses R-multiple via dynamic logic below.
        default_sl_pct=0.15,
        default_tp_pct=0.22,
        vix_symbol="NSE:INDIAVIX-INDEX",
        max_vix=25.0,
        max_trades_per_day=5,
        daily_loss_limit=3000.0,
        # Prefer closer-to-ATM but not too expensive: delta 0.45+ improves hit-rate vs very low delta.
        delta_target_abs=0.55,
        delta_min_abs=0.45,
        # PCR disabled for MCX options anyway (kept for non-MCX).
        pcr_bullish_min=1.02,
        pcr_bearish_max=0.98,
        max_spread_pct=0.02,
        strike_step=0.05,
        lot_size=250,
        # Premium range for NATGASMINI options (tune as you like)
        min_premium=8.0,
        max_premium=25.0,
        require_high_confidence=True,
        min_adx=20.0,
        min_votes=3,
        order_lots=1,
    ),
        "CRUDEOILM": UnderlyingConfigV2(
        key="CRUDEOILM",
        point_value=1.0,
        default_sl_pct=0.12,
        default_tp_pct=0.18,
        vix_symbol=None,
        max_vix=None,
        max_trades_per_day=5,
        daily_loss_limit=3000.0,
        delta_target_abs=0.55,
        delta_min_abs=0.45,
        pcr_bullish_min=1.02,
        pcr_bearish_max=0.98,
        max_spread_pct=0.02,
        strike_step=50.0,
        lot_size=10,
        min_premium=25.0,
        max_premium=120.0,
        require_high_confidence=True,
        min_adx=20.0,
        min_votes=3,
        order_lots=1,
    ),
        "CRUDEOIL": UnderlyingConfigV2(
        key="CRUDEOIL",
        point_value=1.0,
        default_sl_pct=0.12,
        default_tp_pct=0.18,
        vix_symbol=None,
        max_vix=None,
        max_trades_per_day=5,
        daily_loss_limit=5000.0,
        delta_target_abs=0.55,
        delta_min_abs=0.45,
        pcr_bullish_min=1.02,
        pcr_bearish_max=0.98,
        max_spread_pct=0.02,
        strike_step=50.0,
        lot_size=100,
        min_premium=50.0,
        max_premium=250.0,
        require_high_confidence=True,
        min_adx=20.0,
        min_votes=3,
        order_lots=1,
    ),
}

def run_v3_bot(symbol: str, simulate: bool = False):
    # Pass empty symbols initially to prevent hardcoded logs
    bot_v1 = TradingBot(run_websocket=False)
    bot_v1.symbols = [symbol]
    bot_v1.symbol = symbol
    bot_v1._setup_paths()
    bot_v1._initialize_symbol_files()
    
    # Pre-calculate pivots for the correct symbol
    bot_v1.process_pivots()
    
    # Cache pivots for V2 bot
    pivots = bot_v1.get_cpr_levels_with_fallback(symbol).get(symbol, {})

    mkt = FyersMarketDataAdapterV2(bot_v1.fyers_sdk_instance, bot=bot_v1)
    register_ohlc_provider(bot_v1)
    broker = FyersBrokerAdapterV2(bot_v1.fyers_sdk_instance)
    
    bot_v3 = TradingBotV2(
        broker=broker,
        mkt=mkt,
        underlying_configs=DEFAULT_CONFIGS_V2,
        poll_seconds=5
    )
    bot_v3.pivots_cache[symbol] = pivots

    if simulate:
        print(f"[SIM] [SIMULATION] Starting dry-run for {symbol}...")
        # Optional: Override poll_seconds for fast simulation if it were a loop
        bot_v3.poll_seconds = 5
        bot_v3.run(symbol, max_cycles=5)
        print("[OK] [SIMULATION] Completed successfully.")
    else:
        print(f"[START] [BOT-V3] Starting for {symbol}...")
        bot_v3.run(symbol)

# ---------------------------------------------------------------------------

def main():
    try:
        # Args: [timeframe] [symbol] [mode]
        # mode: live (default) or simulate
        tf = sys.argv[1] if len(sys.argv) > 1 else "5"
        symbol = sys.argv[2] if len(sys.argv) > 2 else "BSE:SENSEX-INDEX"
        mode = sys.argv[3] if len(sys.argv) > 3 else "live"
        
        simulate_requested = (mode.lower() == "simulate")

        # Determine which bot to run
        if ("NATGAS" in symbol.upper() or "SENSEX" in symbol.upper() or "CRUDEOIL" in symbol.upper()):
            run_v3_bot(symbol, simulate=simulate_requested)
        else:
            bot = TradingBot()
            register_ohlc_provider(bot)
            bot.run(selected_tf=tf)
            
    except Exception as e:
        ts = _dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z")
        msg = f"[{ts}] FATAL: {e}\n{_traceback.format_exc()}"
        try:
            print(msg)
        except UnicodeEncodeError:
            print("".join(c for c in msg if ord(c) < 128))
        raise


if __name__ == "__main__":
    main()
