                                                                   #!/usr/bin/env python3
# trading_bot.py
from __future__ import annotations

import logging
import os
import json
import time
import sys
import threading
import warnings
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
from modules.Fyers.adx_efi_mom.service import fetchOHLC1
from math import isfinite


# =========================
# Embedded AI CPR Predictor
# (Merged from cpr_ai_predictor_ultimate_refactor_nextcandle.py)
# =========================
"""
cpr_ai_predictor_ultimate.py

Drop-in refactor of CPR_AIPredictor with safer model loading, better probability handling,
optional calibration, feature-drift gating, and predictable logging.

Design goals
- Backward-compatible predict() signature: (label, confidence, distribution, features)
- Works with model packages saved as dict via joblib/pickle (model/scaler/feature_names/...)
- Adds optional components if present in PKL:
    * calibrator: object exposing predict_proba(X) for calibrated probabilities
    * feature_mean, feature_std: arrays for drift checks (z-score gating)
    * metadata: free-form dict (version, trained_date, training_symbol, etc.)

Notes
- This file does NOT attempt to "beat all models". It makes the runtime inference path
  more robust, safer, and easier to diagnose.
"""

import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import joblib
import numpy as np

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


LABEL_MAP_DEFAULT: Dict[int, str] = {
    -2: "STRONG_SELL",
    -1: "SELL",
    0: "HOLD",
    1: "BUY",
    2: "STRONG_BUY",
}


@dataclass
class DriftConfig:
    enabled: bool = False
    z_threshold: float = 6.0           # flag if any feature exceeds this abs(z)
    max_bad_features: int = 3          # allow small number of outliers
    min_std: float = 1e-9              # avoid div by zero


@dataclass
class PredictGates:
    min_confidence: float = 0.0        # hard minimum acceptance for any non-HOLD
    abstain_label: str = "HOLD"        # if gates fail, return this label
    require_ohlc_for_candles: bool = True
    min_ohlc_rows_for_candles: int = 12


class CPR_AIPredictorUltimate:
    """
    Advanced, safer inference wrapper.

    Backward compatible with your integration in kubera bot:
      ai_label, ai_conf, ai_dist, feature_array = predictor.predict(...)
    """

    def __init__(
        self,
        model_path: str = "ai_cpr_jan_model_v4.pkl",
        logger: Optional[logging.Logger] = None,
        label_map: Optional[Dict[int, str]] = None,
        gates: Optional[PredictGates] = None,
        drift_cfg: Optional[DriftConfig] = None,
    ):
        self.model_path = model_path
        self.logger = logger or logging.getLogger(__name__)

        self.model: Any = None
        self.scaler: Any = None
        self.calibrator: Any = None

        self.feature_names: list[str] = []
        self.n_features: int = 30  # default: 18 technical + 12 candle patterns

        # Optional drift statistics
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

        # Metadata
        self.metadata: Dict[str, Any] = {}

        # Configuration
        self.label_map = label_map or dict(LABEL_MAP_DEFAULT)
        self.gates = gates or PredictGates()
        self.drift_cfg = drift_cfg or DriftConfig()

        # Diagnostics
        self.prediction_history: list[Dict[str, Any]] = []
        self.last_prediction: Optional[Dict[str, Any]] = None

        self._load_model()

    # ---------------------------------------------------------------------
    # Loading
    # ---------------------------------------------------------------------
    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            self.logger.warning(f"[AI-CPR] Model not found: {self.model_path}")
            return

        try:
            package = joblib.load(self.model_path)

            if isinstance(package, dict):
                self.model = package.get("model")
                self.scaler = package.get("scaler")
                self.calibrator = package.get("calibrator") or package.get("calibration_model")

                # next-candle direction model (optional)
                self.next_model = package.get("next_model") or package.get("next_candle_model") or package.get("nextcandle_model")
                self.next_scaler = package.get("next_scaler") or None
                self.next_calibrator = package.get("next_calibrator") or package.get("next_candle_calibrator") or None
                self.next_feature_names = package.get("next_feature_names", []) or []

                self.feature_names = package.get("feature_names", []) or []
                self.n_features = int(package.get("n_features", self.n_features) or self.n_features)

                # drift stats (optional)
                fm = package.get("feature_mean") if package.get("feature_mean") is not None else package.get("feature_mean_")
                fs = package.get("feature_std") if package.get("feature_std") is not None else package.get("feature_std_")
                if fm is not None and fs is not None:
                    self.feature_mean = np.asarray(fm, dtype=float)
                    self.feature_std = np.asarray(fs, dtype=float)

                self.metadata = package.get("metadata", {}) or {}
                # Back-compat common keys
                for k in ("version", "trained_date", "training_symbol", "training_tf"):
                    if k in package and k not in self.metadata:
                        self.metadata[k] = package.get(k)

            else:
                # Direct model fallback
                self.model = package
                self.metadata = {"version": "Direct", "trained_date": "Unknown"}

            if self.model is None:
                self.logger.error(f"[AI-CPR] Loaded package contains no model: {self.model_path}")
                return

            self.logger.info(
                "\n" + "=" * 70 + "\n"
                "🤖 AI CPR MODEL LOADED (Ultimate)\n"
                + "=" * 70 + "\n"
                f"Path: {self.model_path}\n"
                f"Version: {self.metadata.get('version', 'Unknown')}\n"
                f"Trained: {self.metadata.get('trained_date', 'Unknown')}\n"
                f"Model Type: {type(self.model).__name__}\n"
                f"Scaler: {type(self.scaler).__name__ if self.scaler is not None else 'None'}\n"
                f"Calibrator: {type(self.calibrator).__name__ if self.calibrator is not None else 'None'}\n"
                f"n_features: {self.n_features}\n"
                + "=" * 70
            )
        except Exception as e:
            self.logger.error(f"[AI-CPR] Failed to load model: {e}")
            self.model = None

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _to_2d(features: Any) -> Optional[np.ndarray]:
        if features is None:
            return None
        arr = np.asarray(features, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr if arr.ndim == 2 else None

    def _validate_ohlc_for_candles(self, ohlc_df: Any) -> Tuple[bool, str]:
        if not self.gates.require_ohlc_for_candles:
            return True, "ohlc_check_disabled"

        # If your feature builder uses candle features, you need at least N rows.
        # We cannot enforce schema here (pandas df vs list), only length.
        if ohlc_df is None:
            return False, "ohlc_df_missing"
        try:
            n = len(ohlc_df)
        except Exception:
            return False, "ohlc_df_unmeasurable"

        if n < self.gates.min_ohlc_rows_for_candles:
            return False, f"ohlc_df_too_short({n}<{self.gates.min_ohlc_rows_for_candles})"
        return True, "ohlc_ok"

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X
        try:
            return self.scaler.transform(X)
        except Exception as e:
            self.logger.error(f"[AI-CPR] Scaler transform failed: {e}")
            return X

    def _predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Priority:
          calibrator.predict_proba -> model.predict_proba -> None
        """
        try:
            if self.calibrator is not None and hasattr(self.calibrator, "predict_proba"):
                return np.asarray(self.calibrator.predict_proba(X), dtype=float)
        except Exception as e:
            self.logger.warning(f"[AI-CPR] Calibrator predict_proba failed (fallback to model): {e}")

        try:
            if hasattr(self.model, "predict_proba"):
                return np.asarray(self.model.predict_proba(X), dtype=float)
        except Exception as e:
            self.logger.error(f"[AI-CPR] Model predict_proba failed: {e}")

        return None

    def _drift_check(self, raw_features: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Simple z-score drift check against training mean/std if present.
        """
        if not self.drift_cfg.enabled:
            return True, {"enabled": False}

        if self.feature_mean is None or self.feature_std is None:
            return True, {"enabled": True, "skipped": "no_feature_stats"}

        x = raw_features.reshape(-1)
        if x.size != self.feature_mean.size or x.size != self.feature_std.size:
            return True, {"enabled": True, "skipped": "stats_size_mismatch"}

        std = np.maximum(self.feature_std, self.drift_cfg.min_std)
        z = (x - self.feature_mean) / std
        bad = np.where(np.abs(z) > self.drift_cfg.z_threshold)[0]
        ok = len(bad) <= self.drift_cfg.max_bad_features

        details = {
            "enabled": True,
            "z_threshold": self.drift_cfg.z_threshold,
            "max_bad_features": self.drift_cfg.max_bad_features,
            "bad_count": int(len(bad)),
            "bad_indices": bad[:20].tolist(),  # cap to avoid huge logs
        }
        return ok, details

    def _classes(self) -> Optional[np.ndarray]:
        cls = getattr(self.model, "classes_", None)
        if cls is None:
            return None
        try:
            return np.asarray(cls)
        except Exception:
            return None

    def _build_distribution(self, proba_row: np.ndarray, classes: Optional[np.ndarray]) -> Dict[str, float]:
        dist: Dict[str, float] = {}
        if classes is not None and len(classes) == len(proba_row):
            for c, p in zip(classes, proba_row):
                dist[self.label_map.get(int(c), str(int(c)))] = float(p)
        else:
            # fallback: map by sorted label keys if possible
            known = sorted(self.label_map.keys())
            if len(known) == len(proba_row):
                for c, p in zip(known, proba_row):
                    dist[self.label_map.get(int(c), str(int(c)))] = float(p)
            else:
                for i, p in enumerate(proba_row):
                    dist[f"class_{i}"] = float(p)
        return dist

    # ---------------------------------------------------------------------
    # Interpretations (kept lightweight; your main bot already logs a lot)
    # ---------------------------------------------------------------------
    def _interpret_features_quick(self, features_flat: np.ndarray) -> Dict[str, Any]:
        """
        Minimal interpretation useful for gating and debugging.
        Assumes first 18 are technical, last 12 are candle features (as in v3a).
        """
        out: Dict[str, Any] = {}
        if features_flat.size >= 18:
            rsi = float(features_flat[5])
            adx = float(features_flat[12])
            out["rsi"] = rsi
            out["adx"] = adx
            out["trend_fast_vs_slow"] = "UP" if float(features_flat[0]) > float(features_flat[2]) else "DOWN"
        if features_flat.size >= 12:
            candle = features_flat[-12:]
            out["candle_dir"] = "BULL" if candle[1] > 0 else "BEAR" if candle[1] < 0 else "NEUTRAL"
            out["engulfing"] = float(candle[4])
            out["reversal"] = float(candle[5])
            out["gap"] = float(candle[9])
        return out

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    
    def predict_next_candle_direction(
        self,
        indicators: Dict[str, Any],
        pivot_data: Dict[str, Any],
        feature_builder: Callable[..., Any],
        *,
        ohlc_df: Any = None,
    ) -> Dict[str, Any]:
        """Return next candle direction as UP/DOWN/HOLD with probabilities.

        If a dedicated next-candle model is available in the model package (next_model),
        it will be used. Otherwise, direction is derived from the main model distribution.
        """
        out: Dict[str, Any] = {
            "direction": "HOLD",
            "candle": "NEUTRAL",
            "confidence": 0.0,
            "margin": 0.0,
            "p_up": None,
            "p_down": None,
            "source": "fallback",
        }

        if self.model is None:
            return out

        X = self._build_features(indicators, pivot_data, feature_builder, ohlc_df=ohlc_df)
        if X is None:
            return out

        # Dedicated next-candle model path
        if getattr(self, "next_model", None) is not None:
            try:
                Xn = X
                if getattr(self, "next_scaler", None) is not None:
                    Xn = self.next_scaler.transform(Xn)

                if getattr(self, "next_calibrator", None) is not None:
                    proba = self.next_calibrator.predict_proba(Xn)
                else:
                    proba = self.next_model.predict_proba(Xn)

                p = np.asarray(proba, dtype=float)[0]
                if p.shape[0] >= 2:
                    p_down, p_up = float(p[0]), float(p[1])
                else:
                    p_up = float(p[0]); p_down = 1.0 - p_up

                conf = float(max(p_up, p_down))
                out.update({
                    "p_up": p_up,
                    "p_down": p_down,
                    "confidence": conf,
                    "margin": float(abs(p_up - p_down)),
                    "source": "next_model",
                })

                if conf >= 0.55:
                    if p_up > p_down:
                        out["direction"] = "UP"
                        out["candle"] = "GREEN"
                    else:
                        out["direction"] = "DOWN"
                        out["candle"] = "RED"
                return out
            except Exception:
                # fall back to proxy
                pass

        # Proxy path from main model distribution
        try:
            label, conf, dist, _ = self.predict(indicators, pivot_data, feature_builder, ohlc_df=ohlc_df)
            if isinstance(dist, dict) and dist:
                p_buy = float(dist.get("BUY", dist.get("BULLISH", 0.0)) or 0.0)
                p_sell = float(dist.get("SELL", dist.get("BEARISH", 0.0)) or 0.0)
                p_hold = float(dist.get("HOLD", dist.get("NEUTRAL", 0.0)) or 0.0)

                if p_buy == 0.0 and p_sell == 0.0 and p_hold == 0.0:
                    items = sorted([(k, float(v)) for k, v in dist.items()], key=lambda kv: kv[1], reverse=True)
                    top1 = items[0]
                    top2 = items[1] if len(items) > 1 else (None, 0.0)
                    out["confidence"] = float(top1[1])
                    out["margin"] = float(top1[1] - top2[1])
                    out["source"] = "main_model_top1"
                    k = str(top1[0]).upper()
                    if "BUY" in k or "BULL" in k or "UP" in k:
                        out["direction"] = "UP"; out["candle"] = "GREEN"
                    elif "SELL" in k or "BEAR" in k or "DOWN" in k:
                        out["direction"] = "DOWN"; out["candle"] = "RED"
                    return out

                out.update({
                    "p_up": p_buy,
                    "p_down": p_sell,
                    "confidence": float(max(p_buy, p_sell, p_hold)),
                    "margin": float(abs(p_buy - p_sell)),
                    "source": "main_model_proxy",
                })
                if max(p_buy, p_sell) >= 0.55 and abs(p_buy - p_sell) >= 0.05:
                    if p_buy > p_sell:
                        out["direction"] = "UP"; out["candle"] = "GREEN"
                    else:
                        out["direction"] = "DOWN"; out["candle"] = "RED"
        except Exception:
            pass

        return out

    def predict(
        self,
        indicators: Dict[str, Any],
        pivot_data: Dict[str, Any],
        feature_builder: Callable[..., Any],
        ohlc_df: Any = None,
    ) -> Tuple[Optional[str], float, Optional[Dict[str, float]], Optional[np.ndarray]]:
        """
        Returns:
          label (str|None), confidence (float), distribution (dict|None), features (np.ndarray|None)

        Backward compatible with your current caller in kubera bot.
        """
        if self.model is None:
            self.logger.warning("[AI-CPR] Model not loaded - skipping prediction")
            return None, 0.0, None, None

        # Validate LTP
        ltp = indicators.get("close")
        if ltp is None:
            self.logger.warning("[AI-CPR] No LTP available (indicators['close'])")
            return None, 0.0, None, None

        # Optional OHLC check (for candle features)
        ohlc_ok, ohlc_reason = self._validate_ohlc_for_candles(ohlc_df)
        if not ohlc_ok:
            self.logger.warning(f"[AI-CPR] Skipping AI: {ohlc_reason}")
            return None, 0.0, None, None

        # Build features
        try:
            raw = feature_builder(ltp, indicators, pivot_data, ohlc_df=ohlc_df)
        except TypeError:
            # Some builders may not accept ohlc_df; fall back.
            raw = feature_builder(ltp, indicators, pivot_data)

        X = self._to_2d(raw)
        if X is None:
            self.logger.error("[AI-CPR] Feature error: builder returned invalid shape")
            return None, 0.0, None, None

        if X.shape[1] != self.n_features:
            self.logger.error(f"[AI-CPR] Feature count mismatch: got {X.shape[1]} expected {self.n_features}")
            return None, 0.0, None, X

        # Drift check on raw features (pre-scale)
        drift_ok, drift_meta = self._drift_check(X)
        if not drift_ok:
            self.logger.warning(f"[AI-CPR] Drift gate failed: {drift_meta}")
            # Abstain (return HOLD with low confidence, but still return features for debugging)
            return self.gates.abstain_label, 0.0, {"DRIFT_ABSTAIN": 1.0}, X

        # Scale then predict
        Xs = self._scale(X)

        try:
            pred = self.model.predict(Xs)[0]
            pred_class = int(pred)
        except Exception as e:
            self.logger.error(f"[AI-CPR] Model predict failed: {e}")
            return None, 0.0, None, X

        proba = self._predict_proba(Xs)
        distribution: Optional[Dict[str, float]] = None
        confidence: float = 0.0

        if proba is not None and proba.ndim == 2 and proba.shape[0] == 1:
            p_row = proba[0]
            classes = self._classes()
            distribution = self._build_distribution(p_row, classes)

            # confidence = prob of predicted class if possible; else max
            if classes is not None and len(classes) == len(p_row):
                try:
                    idx = list(classes).index(pred_class)
                    confidence = float(p_row[idx])
                except Exception:
                    confidence = float(np.max(p_row))
            else:
                confidence = float(np.max(p_row))
        else:
            # If no probabilities, leave confidence at 0.0 and no distribution.
            distribution = None
            confidence = 0.0

        label = self.label_map.get(pred_class, "UNKNOWN")

        # Gate: if non-HOLD but confidence too low -> abstain to HOLD
        if label != self.gates.abstain_label and confidence < self.gates.min_confidence:
            self.logger.info(
                f"[AI-CPR] Confidence gate: {label} conf={confidence:.3f} < min={self.gates.min_confidence:.3f} -> {self.gates.abstain_label}"
            )
            label = self.gates.abstain_label

        # Store history (cap 200)
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ltp": float(ltp) if ltp is not None else None,
            "label": label,
            "pred_class": pred_class,
            "confidence": float(confidence),
            "distribution": distribution,
            "ohlc_reason": ohlc_reason,
            "drift": drift_meta,
            "quick": self._interpret_features_quick(X.reshape(-1)),
        }
        self.last_prediction = meta
        self.prediction_history.append(meta)
        if len(self.prediction_history) > 200:
            self.prediction_history = self.prediction_history[-200:]

        # Concise log line (your main bot already prints full details)
        self.logger.info(
            f"[AI-CPR] {label} conf={confidence:.3f} ltp={float(ltp):.2f} drift_ok={drift_ok} ohlc={ohlc_reason}"
        )

        return label, confidence, distribution, X

def get_prediction_summary(self, last_n: int = 10) -> str:
        if not self.prediction_history:
            return "No predictions yet"
        recent = self.prediction_history[-max(1, int(last_n)):]
        lines = ["\n📊 RECENT AI-CPR PREDICTIONS:"]
        for p in recent:
            lines.append(
                f"  [{p['timestamp']}] {p['label']} (conf={p['confidence']:.2f}) @ {p.get('ltp', 0):.2f}"
            )
        return "\n".join(lines)

def get_feature_importance(self, top_k: int = 10) -> Optional[list[Tuple[str, float]]]:
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return None
        try:
            importances = np.asarray(self.model.feature_importances_, dtype=float)
            if self.feature_names and len(self.feature_names) == len(importances):
                names = self.feature_names
            else:
                names = [f"Feature_{i}" for i in range(len(importances))]

            pairs = list(zip(names, importances.tolist()))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[: max(1, int(top_k))]
        except Exception as e:
            self.logger.error(f"[AI-CPR] Error getting feature importance: {e}")
            return None

# Backward-compatible alias
CPR_AIPredictor = CPR_AIPredictorUltimate


# Backward compatible alias used by the bot
CPR_AIPredictor = CPR_AIPredictorUltimate

from typing import Optional

# =========================
# Fine-tuning configuration
# =========================
# Goal: Improve profitability by reducing chop trades, tightening entry validation,
# and adapting thresholds by timeframe (especially for NATGASMINI).
#
# You can override key knobs with environment variables:
#   AI_CPR_TF=15m
#   AI_CPR_PROFILE=balanced|conservative|aggressive
#   AI_CPR_POINT_VALUE=250
#
# Notes:
# - "conservative" = fewer trades, higher quality
# - "aggressive"   = more trades, lower selectivity
#
DEFAULT_TUNING = {
    "1m":  {"chop_adx": 18.0, "bbw_squeeze": 0.020, "cpr_width": 0.0022, "vwap_dev": 0.0018,
            "vol_breakout_z": 1.9, "breakout_conf": 0.80, "rejection_conf": 0.82,
            "impulse_body_atr": 1.6, "cooldown_min": 2, "cooldown_max": 5,
            "early_entry_conf": 0.70, "st_bypass_in_chop": False},
    "5m":  {"chop_adx": 19.0, "bbw_squeeze": 0.018, "cpr_width": 0.0020, "vwap_dev": 0.0016,
            "vol_breakout_z": 1.8, "breakout_conf": 0.79, "rejection_conf": 0.81,
            "impulse_body_atr": 1.7, "cooldown_min": 2, "cooldown_max": 6,
            "early_entry_conf": 0.70, "st_bypass_in_chop": False},
    "15m": {"chop_adx": 20.0, "bbw_squeeze": 0.016, "cpr_width": 0.0018, "vwap_dev": 0.0014,
            "vol_breakout_z": 1.7, "breakout_conf": 0.78, "rejection_conf": 0.82,
            "impulse_body_atr": 1.8, "cooldown_min": 3, "cooldown_max": 8,
            "early_entry_conf": 0.72, "st_bypass_in_chop": False},
    "30m": {"chop_adx": 21.0, "bbw_squeeze": 0.015, "cpr_width": 0.0016, "vwap_dev": 0.0012,
            "vol_breakout_z": 1.6, "breakout_conf": 0.77, "rejection_conf": 0.82,
            "impulse_body_atr": 1.9, "cooldown_min": 3, "cooldown_max": 10,
            "early_entry_conf": 0.73, "st_bypass_in_chop": False},
    "1h":  {"chop_adx": 22.0, "bbw_squeeze": 0.014, "cpr_width": 0.0014, "vwap_dev": 0.0010,
            "vol_breakout_z": 1.5, "breakout_conf": 0.76, "rejection_conf": 0.83,
            "impulse_body_atr": 2.0, "cooldown_min": 2, "cooldown_max": 8,
            "early_entry_conf": 0.74, "st_bypass_in_chop": True},
}

PROFILE_ADJUST = {
    "conservative": {"chop_adx": +1.0, "bbw_squeeze": -0.001, "vol_breakout_z": +0.15,
                     "breakout_conf": +0.03, "rejection_conf": +0.02, "early_entry_conf": +0.03,
                     "impulse_body_atr": -0.05},
    "balanced":     {},
    "aggressive":   {"chop_adx": -1.0, "bbw_squeeze": +0.001, "vol_breakout_z": -0.10,
                     "breakout_conf": -0.02, "rejection_conf": -0.02, "early_entry_conf": -0.03,
                     "impulse_body_atr": +0.05},
}

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

DEFAULT_TIMEZONE = "Asia/Kolkata"
IST = pytz.timezone(DEFAULT_TIMEZONE)
tf_selected = sys.argv[1] if len(sys.argv) > 1 else "5"
print(f"🕒 Selected timeframe: {tf_selected}m")

def get_now_iso():
    """Global function to get current timestamp in ISO format"""
    return dt.datetime.now(IST).isoformat()

AI_GATE_TRADES = True # Set to True to make AI signal a requirement for entry, False to just log it.
logger = logging.getLogger(__name__)


def _build_ai_cpr_features(ltp: float, indicators: dict, pivot_data: dict, ohlc_df=None) -> np.ndarray:
    """
    FIXED: Matches training data exactly
    Total: 30 features (18 technical + 12 candles)

    ⚠️ CRITICAL: Feature order MUST match train_ai_model.py FEATURE_COLUMNS
    """
    logger = logging.getLogger(__name__)
    features = []

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
            f"[AI-CPR] ❌ FEATURE COUNT MISMATCH! "
            f"Expected={expected_count}, Got={actual_count}"
        )
        return None  # 🔧 FIX: Return None to prevent prediction with invalid features

    # Check for NaN/Inf
    features_array = np.array(features, dtype=float)
    if np.isnan(features_array).any():
        nan_count = np.isnan(features_array).sum()
        logger.error(f"[AI-CPR] ❌ Found {nan_count} NaN values - replacing with 0")
        features_array = np.nan_to_num(features_array, nan=0.0)

    if np.isinf(features_array).any():
        inf_count = np.isinf(features_array).sum()
        logger.error(f"[AI-CPR] ❌ Found {inf_count} Inf values - replacing with 0")
        features_array = np.nan_to_num(features_array, posinf=0.0, neginf=0.0)

    logger.debug(f"[AI-CPR] ✅ Built {len(features_array)} valid features")

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
        patterns_detected.append("🚀 3-CANDLE BULLISH MOMENTUM")
    elif momentum_3 == -1:
        patterns_detected.append("🔻 3-CANDLE BEARISH MOMENTUM")

    # Feature 9: Range Expansion
    if range_expansion > 1.5:
        patterns_detected.append(f"📊 RANGE EXPANSION ({range_expansion:.2f}x)")

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
        patterns_detected.append(f"⚠️ LOW VOLUME ({vol_ratio:.2f}x)")

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
# Configuration Constants
# ───────────────────────────────────────────────────────────────────────────────
# 🔥 FIX #2: Set primary timeframe to 30m (not 5m)
PRIMARY_TF = str(tf_selected)  # All ADX/RSI/EFI calculations will use selected TF

# ───────────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────────

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
        if df.empty or len(df) < 14:
            return ("NO TRADE",) + (None,)*6
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
        prev, cur = ind[i-1], ind[i]
        if df2['Close'].iloc[i-1] <= df2['FinalUp'].iloc[i-1]:
            df2.at[cur,'FinalUp'] = min(df2['BasicUp'].iloc[i], df2['FinalUp'].iloc[i-1])
        else:
            df2.at[cur,'FinalUp'] = df2['BasicUp'].iloc[i]
        if df2['Close'].iloc[i-1] >= df2['FinalDown'].iloc[i-1]:
            df2.at[cur,'FinalDown'] = max(df2['BasicDown'].iloc[i], df2['FinalDown'].iloc[i-1])
        else:
            df2.at[cur,'FinalDown'] = df2['BasicDown'].iloc[i]

    flip = None
    for i in range(period, len(df2)):
        prev, cur = ind[i-1], ind[i]
        if df2['Close'].iloc[i-1] <= df2['FinalUp'].iloc[i-1] and df2['Close'].iloc[i] > df2['FinalUp'].iloc[i]:
            df2.at[cur,'Strend'], df2.at[cur,'Trend'] = df2['FinalDown'].iloc[i], 1
            flip = i; break
        if df2['Close'].iloc[i-1] >= df2['FinalDown'].iloc[i-1] and df2['Close'].iloc[i] < df2['FinalDown'].iloc[i]:
            df2.at[cur,'Strend'], df2.at[cur,'Trend'] = df2['FinalUp'].iloc[i], -1
            flip = i; break
    if flip is None:
        flip = period

    for i in range(flip+1, len(df2)):
        prev, cur = ind[i-1], ind[i]
        if df2.at[prev,'Strend'] == df2.at[prev,'FinalUp']:
            if df2['Close'].iloc[i] <= df2['FinalUp'].iloc[i]:
                df2.at[cur,'Strend'], df2.at[cur,'Trend'] = df2['FinalUp'].iloc[i], -1
            else:
                df2.at[cur,'Strend'], df2.at[cur,'Trend'] = df2['FinalDown'].iloc[i], 1
        else:
            if df2['Close'].iloc[i] >= df2['FinalDown'].iloc[i]:
                df2.at[cur,'Strend'], df2.at[cur,'Trend'] = df2['FinalDown'].iloc[i], 1
            else:
                df2.at[cur,'Strend'], df2.at[cur,'Trend'] = df2['FinalUp'].iloc[i], -1

    return df2['Strend'], df2['FinalUp'], df2['FinalDown'], df2['Trend']

# ───────────────────────────────────────────────────────────────────────────────
# EMA50/200 crossover helper
# ───────────────────────────────────────────────────────────────────────────────

class EMA50_200:
    def __init__(self, fyers_client, ticker, interval, duration=60):
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

    # ✅ HANDLE NUMERIC LABELS FROM YOUR MODEL:
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
    # ✅ NEW: Comprehensive CPR validation
    # ✅ TF context for logging (avoid NameError when called without explicit TF)
    primary_tf = (indicators.get('timeframe') or indicators.get('tf') or pivot_data.get('timeframe') or pivot_data.get('tf'))
    if primary_tf is None:
        primary_tf = getattr(ai_predictor, 'last_known_primary_tf', None)
    if primary_tf is None:
        primary_tf = 'NA'
    try:
        primary_tf = str(int(float(primary_tf))) if str(primary_tf).replace('.','',1).isdigit() else str(primary_tf)
    except Exception:
        primary_tf = str(primary_tf)

    required_cpr_keys = ["TC", "BC", "R1", "R2", "R3", "S1", "S2", "S3"]
    missing_keys = [key for key in required_cpr_keys
                    if key not in pivot_data or pivot_data[key] is None]

    if missing_keys:
        error_msg = f"Missing CPR levels: {', '.join(missing_keys)}"
        logger.error(f"[CPR-ERROR] {error_msg}")
        logger.error(f"[CPR-ERROR] Pivot data received: {list(pivot_data.keys())}")
        logger.error(f"[CPR-ERROR] Sample values: TC={pivot_data.get('TC')}, BC={pivot_data.get('BC')}")
        return {
            "error": error_msg,
            "trade_strategy": "None",
            "reason": "Incomplete CPR data",
            "cpr_levels": pivot_data
        }

    # Validate TC > BC
    tc = pivot_data.get("TC")
    bc = pivot_data.get("BC")

    if tc is None or bc is None:
        return {"error": "TC or BC is None", "trade_strategy": "None"}

    try:
        if float(tc) < float(bc):
            logger.warning(f"[CPR-WARNING] TC ({tc}) < BC ({bc}) - Inverted CPR!")
    except (ValueError, TypeError):
        return {"error": f"Invalid TC/BC: {tc}/{bc}", "trade_strategy": "None"}

    # ✅ Check for minimal required indicators before proceeding
    has_minimal_data = all([
        indicators.get("ema_5"),
        indicators.get("ema_21"),
        indicators.get("close")
    ])

    if not has_minimal_data:
        return {"error": f"Insufficient indicators for CPR analysis: ema_5={indicators.get('ema_5')}, ema_21={indicators.get('ema_21')}, close={indicators.get('close')}",
                "trade_strategy": "None"}

    ema_21 = convert_to_serializable(indicators.get("ema_21", 0))
    ema_5 = convert_to_serializable(indicators.get("ema_5", 0))
    ema_20 = convert_to_serializable(indicators.get("ema_20", 0))
    ema_50 = convert_to_serializable(indicators.get("ema_50", 0))
    ema_200 = convert_to_serializable(indicators.get("ema_200", 0))
    st21Trend = convert_to_serializable(indicators.get("st21Trend", 0))
    adx = convert_to_serializable(indicators.get("adx", 0))
    close = convert_to_serializable(indicators.get("close", 0))

    # CPR/Pivot levels
    tc = convert_to_serializable(pivot_data.get("TC", 0))
    bc = convert_to_serializable(pivot_data.get("BC", 0))

    r1 = convert_to_serializable(pivot_data.get("R1", 0))
    r2 = convert_to_serializable(pivot_data.get("R2", 0))
    r3 = convert_to_serializable(pivot_data.get("R3", 0))
    s1 = convert_to_serializable(pivot_data.get("S1", 0))
    s2 = convert_to_serializable(pivot_data.get("S2", 0))
    s3 = convert_to_serializable(pivot_data.get("S3", 0))

    # Check if essential data is available and valid
    if not (tc and bc and close and tc > 0 and bc > 0 and close > 0):
        return {"error": f"Invalid CPR data: TC={tc or 0}, BC={bc or 0}, Close={close or 0}",
                "trade_strategy": "None"}

    # Key Price Action Levels
    pdh = convert_to_serializable(pivot_data.get("High", 0))
    pdl = convert_to_serializable(pivot_data.get("Low", 0))
    pwh = convert_to_serializable(pivot_data.get("PWH", 0))
    pwl = convert_to_serializable(pivot_data.get("PWL", 0))
    pmh = convert_to_serializable(pivot_data.get("PMH", 0))
    pml = convert_to_serializable(pivot_data.get("PML", 0))
    wh_52 = convert_to_serializable(pivot_data.get("52WH", 0))
    wl_52 = convert_to_serializable(pivot_data.get("52WL", 0))

    # Previous day CPR for comparison (if available)
    prev_tc = convert_to_serializable(pivot_data.get("prev_TC"))
    prev_bc = convert_to_serializable(pivot_data.get("prev_BC"))

    if not all([tc, bc, close]):
        return {"error": "Invalid CPR or price data"}

    # Key Price Action View Formulation (Market Bias)
    key_price_action_view = "NEUTRAL"
    position_sizing = "CONSERVATIVE"

    # AGGRESSIVE BULLISH MOMENTUM: Break above key levels with follow-through
    if close > pdh * 1.002 and adx > 35 and ema_5 > ema_21:
        key_price_action_view = "AGGRESSIVE_BULLISH"
        position_sizing = "AGGRESSIVE"
    elif close > pmh * 1.002 and adx > 30:  # Break above monthly high
        key_price_action_view = "AGGRESSIVE_BULLISH"
        position_sizing = "AGGRESSIVE"
    elif close > wh_52 * 1.002 and adx > 25:  # Break above 52-week high
        key_price_action_view = "AGGRESSIVE_BULLISH"
        position_sizing = "AGGRESSIVE"

    # AGGRESSIVE BEARISH REVERSAL: Rejection at resistance or break below support
    elif close < pdl * 0.998 and adx > 35 and ema_5 < ema_21:
        key_price_action_view = "AGGRESSIVE_BEARISH"
        position_sizing = "AGGRESSIVE"
    elif close < pml * 0.998 and adx > 30:  # Break below monthly low
        key_price_action_view = "AGGRESSIVE_BEARISH"
        position_sizing = "AGGRESSIVE"
    elif close < wl_52 * 0.998 and adx > 25:  # Break below 52-week low
        key_price_action_view = "AGGRESSIVE_BEARISH"
        position_sizing = "AGGRESSIVE"

    # DEFENSIVE RETRACEMENT: Break then reverse (caution mode)
    elif close > pdh * 1.001 and close < pdh * 1.005 and adx < 30:
        key_price_action_view = "DEFENSIVE_RETRACEMENT"
        position_sizing = "DEFENSIVE"
    elif close < pdl * 0.999 and close > pdl * 0.995 and adx < 30:
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
    
    # ✅ FIX: AI Confidence Gate (Layer-2)
    # NOTE: Do NOT overwrite the model's label. Only suppress directional use when confidence is low.
    AI_DIR_MIN_CONF = 0.60
    ai_label_raw = ai_label
    ai_dir = _ai_dir_from_label(ai_label_raw)
    suppressed = False
    if ai_conf is not None and float(ai_conf) < AI_DIR_MIN_CONF:
        suppressed = True
        ai_dir = 0
        try:
            logger.info(f"[AI-GATE] Low confidence (conf={float(ai_conf):.3f} < {AI_DIR_MIN_CONF:.2f}). Suppressing direction; preserving label={ai_label_raw}.")
        except Exception:
            pass
    ai_label = ai_label_raw

    
    # Ensure ai_dir is numeric for comparisons later
    if ai_dir is None:
        ai_dir = 0
    elif isinstance(ai_dir, str):
        # Emergency recovery if somehow a string leaked
        ai_dir = _ai_dir_from_label(ai_dir) or 0

    # === NEXT CANDLE BIAS (Human-friendly) ===
    try:
        _label_u = str(ai_label_raw).upper() if ai_label_raw is not None else ""
        if "BUY" in _label_u:
            next_bias = "UP"
        elif "SELL" in _label_u:
            next_bias = "DOWN"
        else:
            next_bias = "NEUTRAL"
        bias_note = " (direction suppressed by gate)" if suppressed and next_bias != "NEUTRAL" else ""
        _conf = float(ai_conf) if ai_conf is not None else 0.0
        logger.info(f"🧭 [NEXT-CANDLE] TF={primary_tf}m | Bias={next_bias}{bias_note} | Label={ai_label_raw} | Conf={_conf:.3f}")
    except Exception:
        pass

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
                f"  • 3-Candle Momentum: {candle_features[7]} {'🚀' if candle_features[7] > 0 else '🔻' if candle_features[7] < 0 else ''}\n"
                f"  • Range Expansion: {candle_features[8]:.2f}x\n"
                f"  • Gap: {candle_features[9]} {'⬆️' if candle_features[9] > 0 else '⬇️' if candle_features[9] < 0 else ''}\n"
                f"  • Close Position: {candle_features[10]:.1f}% {'(near high)' if candle_features[10] > 70 else '(near low)' if candle_features[10] < 30 else '(mid)'}\n"
                f"  • Volume: {candle_features[11]:.2f}x {'💪' if candle_features[11] > 1.5 else '⚠️' if candle_features[11] < 0.7 else ''}\n"
                f"\n"
                f"KEY TECHNICAL FEATURES:\n"
                f"  • EMA5: {features_flat[0]:.2f}\n"
                f"  • EMA21: {features_flat[2]:.2f}\n"
                f"  • RSI: {features_flat[5]:.1f}\n"
                f"  • ADX: {features_flat[12]:.1f}\n"
                f"  • CPR Distance: {features_flat[15] * 100:.2f}%\n"
                f"{'=' * 60}"
            )
    else:
        # User wants to see prediction even if breakdown is skipped
        logger.info(f"🤖 AI PREDICTION: {ai_label} (confidence: {ai_conf:.3f}) | Model Breakdown: SKIPPED", True)

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
    try:
        # Cast to avoid "str vs int" TypeErrors
        n_ai_dir = int(ai_dir) if ai_dir is not None else 0
        n_ai_conf = float(ai_conf) if ai_conf is not None else 0.0
        n_min_conf = float(AI_MIN_CONF)
        
        if trade_strategy.startswith("Buy") and n_ai_dir < 0 and n_ai_conf >= n_min_conf:
            ai_filter_pass = False
            reason += f" | AI disagrees (label={ai_label}, conf={round(n_ai_conf, 2)})"
        elif trade_strategy.startswith("Sell") and n_ai_dir > 0 and n_ai_conf >= n_min_conf:
            ai_filter_pass = False
            reason += f" | AI disagrees (label={ai_label}, conf={round(n_ai_conf, 2)})"
        elif trade_strategy == "Exit" and n_ai_dir < 0 and n_ai_conf >= n_min_conf:
            # AI confirms exit
            reason += f" | AI confirms exit (label={ai_label}, conf={round(n_ai_conf, 2)})"
        else:
            reason += f" | AI:{ai_label}({round(n_ai_conf, 2)})"
    except (ValueError, TypeError) as e:
        logger.error(f"[AI-GATE-ERROR] Numeric cast failed: {e}")
        reason += f" | AI:ERROR({ai_label})"

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

    def analyze(self, symbol, ltp, indicators, pivot_data, ohlc_df):
        """
        Main Price Action Analysis
        Returns: (signal, confidence, reason, levels)
        """

        # Get all key levels
        levels = self._get_all_levels(indicators, pivot_data, ohlc_df)

        # Check which level we're at
        level_interaction = self._check_level_interaction(ltp, levels)

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
            "signal": signal,  # "BUY", "SELL", or None
            "confidence": confidence,  # 0.0 to 1.0
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

    def _check_level_interaction(self, ltp, levels, tolerance=0.003):
        """
        Check if price is near any key level (within 0.3%)
        Returns: {"at_level": bool, "level_type": str, "price": float, "direction": str}
        """

        # Check resistance levels
        for r_level in levels["resistance"]:
            price = r_level["price"]
            if abs(ltp - price) / price <= tolerance:
                return {
                    "at_level": True,
                    "level_type": r_level["type"],
                    "price": price,
                    "direction": "resistance",
                    "strength": r_level["strength"],
                    "distance_pct": ((ltp - price) / price * 100)
                }

        # Check support levels
        for s_level in levels["support"]:
            price = s_level["price"]
            if abs(ltp - price) / price <= tolerance:
                return {
                    "at_level": True,
                    "level_type": s_level["type"],
                    "price": price,
                    "direction": "support",
                    "strength": s_level["strength"],
                    "distance_pct": ((ltp - price) / price * 100)
                }

        # Not at any key level
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
        Returns: {"pattern": str, "confidence": float}
        """

        if not level_interaction.get("at_level"):
            return {"pattern": None, "confidence": 0.0}

        if ohlc_df is None or len(ohlc_df) < 3:
            return {"pattern": None, "confidence": 0.0}

        latest = ohlc_df.iloc[-1]

        body = abs(latest['Close'] - latest['Open'])
        total_range = latest['High'] - latest['Low']
        upper_wick = latest['High'] - max(latest['Open'], latest['Close'])
        lower_wick = min(latest['Open'], latest['Close']) - latest['Low']

        direction = level_interaction.get("direction")

        # 🔴 At RESISTANCE - Look for bearish rejection
        if direction == "resistance":
            # Large upper wick = rejection at resistance
            if upper_wick > body * 2 and latest['Close'] < (latest['Low'] + total_range * 0.3):
                return {
                    "pattern": "Bearish Rejection",
                    "confidence": 0.85,
                    "signal": "SELL"
                }

            # Shooting star
            if upper_wick > body * 2.5 and lower_wick < body * 0.3:
                return {
                    "pattern": "Shooting Star",
                    "confidence": 0.80,
                    "signal": "SELL"
                }

        # 🟢 At SUPPORT - Look for bullish rejection
        elif direction == "support":
            # Large lower wick = rejection at support
            if lower_wick > body * 2 and latest['Close'] > (latest['High'] - total_range * 0.3):
                return {
                    "pattern": "Bullish Rejection",
                    "confidence": 0.85,
                    "signal": "BUY"
                }

            # Hammer
            if lower_wick > body * 2.5 and upper_wick < body * 0.3:
                return {
                    "pattern": "Hammer",
                    "confidence": 0.80,
                    "signal": "BUY"
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
        Generate final trading signal based on price action
        """

        if not level_interaction.get("at_level"):
            return None, 0.0, "Not at key level"

        level_type = level_interaction.get("level_type")
        level_price = level_interaction.get("price")
        direction = level_interaction.get("direction")
        strength = level_interaction.get("strength")

        # Base confidence from level strength
        confidence = {
            "strong": 0.70,
            "medium": 0.60,
            "weak": 0.50
        }.get(strength, 0.50)

        # Check candle pattern
        pattern = candle_signal.get("pattern")
        pattern_conf = candle_signal.get("confidence", 0.0)
        signal = candle_signal.get("signal")

        if not pattern:
            return None, 0.0, f"At {level_type} but no clear candle pattern"

        # Add candle pattern confidence
        confidence = min(0.95, confidence + (pattern_conf * 0.3))

        # Add Fibonacci confluence boost
        if fib_signal.get("confluence"):
            confidence += fib_signal.get("confidence_boost", 0.0)
            confluence_msg = f" + Fib {fib_signal.get('fib_level')}"
        else:
            confluence_msg = ""

        # Check volume confirmation
        volume_ratio = indicators.get("volume_ratio", 1.0)
        if volume_ratio >= 1.5:
            confidence = min(0.95, confidence + 0.10)
            volume_msg = f" + Volume {volume_ratio:.1f}x"
        else:
            volume_msg = " (low volume)"

        # Generate reason
        reason = (
            f"Price Action: {pattern} at {level_type} "
            f"({level_price:.2f}){confluence_msg}{volume_msg}"
        )

        # Final confidence check
        if confidence < 0.65:
            return None, confidence, f"{reason} - Confidence too low"

        return signal, confidence, reason
# ───────────────────────────────────────────────────────────────────────────────
# IndicatorCalculator.calculate_indicators (full replacement)
# ───────────────────────────────────────────────────────────────────────────────
class IndicatorCalculator:
    def __init__(self, bot):
        self.bot = bot

    def calculate_pivot_points(self, df_day):
        """Enhanced CPR calculation with comprehensive validation"""

        # ✅ VALIDATION 1: Check DataFrame
        if df_day is None or not isinstance(df_day, pd.DataFrame):
            self.bot.log_message("❌ Pivot: Invalid DataFrame type", False)
            return {}

        if df_day.empty:
            self.bot.log_message("❌ Pivot: Empty DataFrame", False)
            return {}

        # ✅ VALIDATION 2: Check required columns
        required_cols = ["High", "Low", "Close"]
        for c in required_cols:
            if c not in df_day.columns:
                self.bot.log_message(f"❌ Pivot: Missing column '{c}'", False)
                return {}

        # ✅ VALIDATION 3: Check row count
        if len(df_day) < 2:
            self.bot.log_message(f"❌ Pivot: Need at least 2 days, got {len(df_day)}", False)
            return {}

        # Use previous day's data (most recent complete day)
        prev_day = df_day.iloc[-2]

        high = float(prev_day["High"])
        low = float(prev_day["Low"])
        close = float(prev_day["Close"])

        # ✅ VALIDATION 4: Check values are positive
        if high <= 0 or low <= 0 or close <= 0:
            self.bot.log_message(f"❌ Pivot: Invalid OHLC - H:{high}, L:{low}, C:{close}", False)
            return {}

        # ✅ VALIDATION 5: Check high >= low
        if high < low:
            self.bot.log_message(f"❌ Pivot: High ({high}) < Low ({low}) - Invalid data!", False)
            return {}

        # Calculate pivot levels
        PP = round((high + low + close) / 3, 2)
        BC = round((high + low) / 2, 2)
        TC = round((PP - BC) + PP, 2)

        # ✅ FIX: Ensure TC > BC
        if TC < BC:
            TC, BC = BC, TC
            self.bot.log_message(f"⚠️ Swapped TC/BC: TC={TC}, BC={BC}", True)

        # Calculate support/resistance levels
        R1 = round(2 * PP - low, 2)
        S1 = round(2 * PP - high, 2)
        R2 = round(PP + (high - low), 2)
        S2 = round(PP - (high - low), 2)
        R3 = round(high + 2 * (PP - low), 2)
        S3 = round(low - 2 * (high - PP), 2)

        # ✅ Check Virgin CPR
        virgin_cpr = False
        if prev_day["High"] < BC or prev_day["Low"] > TC:
            virgin_cpr = True

        # ✅ Calculate additional levels
        try:
            PWH = round(df_day['High'].tail(5).max(), 2) if len(df_day) >= 5 else high
            PWL = round(df_day['Low'].tail(5).min(), 2) if len(df_day) >= 5 else low
            PMH = round(df_day['High'].tail(20).max(), 2) if len(df_day) >= 20 else high
            PML = round(df_day['Low'].tail(20).min(), 2) if len(df_day) >= 20 else low
            WH_52 = round(df_day['High'].tail(252).max(), 2) if len(df_day) >= 252 else high
            WL_52 = round(df_day['Low'].tail(252).min(), 2) if len(df_day) >= 252 else low
        except Exception as e:
            self.bot.log_message(f"⚠️ Weekly/Monthly levels calc error: {e}", True)
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

        # ✅ FINAL VALIDATION: Ensure all core levels exist
        required_result_keys = ["TC", "BC", "PP", "R1", "R2", "R3", "S1", "S2", "S3"]
        missing_result_keys = [k for k in required_result_keys if k not in result or result[k] is None]

        if missing_result_keys:
            self.bot.log_message(f"❌ Pivot result incomplete! Missing: {', '.join(missing_result_keys)}", False)
            return {}

        # Log successful calculation
        self.bot.log_message(
            f"\n📊 CPR LEVELS CALCULATED:\n"
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
        # ✅ CRITICAL: Validate pivot_data received
        if isinstance(pivot_data, list):
            self.bot.log_message(
                f"❌ [INDICATORS] pivot_data is a LIST for {symbol} - converting to dict",
                False
            )
            pivot_data = {}

            # Check if pivot_data is None
        if pivot_data is None:
            self.bot.log_message(
                f"⚠️ [INDICATORS] No pivot data for {symbol} - attempting emergency load",
                False
            )
            pivot_data = {}

            # Check if pivot_data is not a dictionary
        if not isinstance(pivot_data, dict):
            self.bot.log_message(
                f"❌ [INDICATORS] pivot_data is {type(pivot_data)} for {symbol} - using empty dict",
                False
            )
            pivot_data = {}

            # If empty, try to load from file
        if not pivot_data:
            try:
                pivot_json_path = self.bot.data_paths[symbol]['pivot_json']
                loaded_data = robust_load_json(pivot_json_path, self.bot.log_message, default={})

                # Extract symbol-specific data
                if isinstance(loaded_data, dict) and symbol in loaded_data:
                    pivot_data = loaded_data[symbol]
                    self.bot.log_message(f"✅ [INDICATORS] Loaded pivots from file for {symbol}", True)
                else:
                    self.bot.log_message(
                        f"❌ [INDICATORS] Cannot find {symbol} in pivot file\n"
                        f"   File keys: {list(loaded_data.keys()) if isinstance(loaded_data, dict) else 'NOT A DICT'}",
                        False
                    )
            except Exception as e:
                self.bot.log_message(f"❌ [INDICATORS] Emergency pivot load failed: {e}", False)

            # Final validation
        required_keys = ["TC", "BC", "R1", "R2", "R3", "S1", "S2", "S3"]
        missing_keys = [k for k in required_keys if k not in pivot_data or pivot_data[k] is None]

        if missing_keys:
            self.bot.log_message(
                f"⚠️ [INDICATORS] Pivot data incomplete for {symbol}: missing {', '.join(missing_keys)}",
                False
            )
            # Continue anyway - CPR analysis will handle gracefully
        else:
            self.bot.log_message(
                f"✅ [INDICATORS] Valid pivots for {symbol}: TC={pivot_data['TC']}, BC={pivot_data['BC']}",
                True
            )
        if ohlc_df is not None and not ohlc_df.empty:
            ohlc = ohlc_df.copy()
            self.bot.log_message("IndicatorCalc: Using pre-fetched OHLC for backtesting.", True)
        else:
            ohlc = self.bot.fetch_ohlc(symbol, timeframe, 60)

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

        # ✅ NEW: Volume Ratio (Volume / Average)
        ohlc["volume_ratio"] = ohlc["Volume"] / ohlc["VolSMA20"]

        # ✅ NEW: Volume Surge Detection
        ohlc["volume_surge"] = ohlc["volume_ratio"] > 1.5  # 50% above average

        # ✅ NEW: Extreme Volume (for very strong moves)
        ohlc["volume_extreme"] = ohlc["volume_ratio"] > 2.0  # 100% above average

        # ==========================================
        # 🚀 MOMENTUM INDICATORS
        # ==========================================

        # ✅ NEW: 10-Period Momentum (Price - Price[10])
        ohlc["momentum_10"] = ohlc["Close"].diff(10)

        # ✅ NEW: Momentum Percentage
        ohlc["momentum_pct"] = ((ohlc["Close"] - ohlc["Close"].shift(10)) /
                                ohlc["Close"].shift(10) * 100)

        # ✅ NEW: Rate of Change (ROC) - Alternative momentum
        ohlc["roc_10"] = ((ohlc["Close"] - ohlc["Close"].shift(10)) /
                          ohlc["Close"].shift(10) * 100)

        # ✅ NEW: Acceleration (momentum of momentum)
        ohlc["acceleration"] = ohlc["momentum_10"].diff(3)

        # ==========================================
        # 🎯 COMBINED SIGNALS
        # ==========================================

        # ✅ NEW: Strong Bearish Signal
        ohlc["strong_bearish"] = (
                (ohlc["momentum_pct"] < -0.5) &  # Dropping fast
                (ohlc["volume_ratio"] > 1.3)  # With volume
        )

        # ✅ NEW: Strong Bullish Signal
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
        # 🔧 FIX: More stable Supertrend configuration to reduce false signals
        st_main, _, _, tr_main = supertrend(ohlc, period=10, multiplier=3.0)  # More stable, less noise
        st_fast, _, _, tr_fast = supertrend(ohlc, period=7, multiplier=2.5)   # Quick exits, balanced
        #st_main, _, _, tr_main = supertrend(ohlc, period=7, multiplier=2.5)
        #st7,  _, _, tr7  = supertrend(ohlc, 7,  3)

        # Extras (unchanged)
        bull_div, bear_div = rsi_divergence(ohlc)
        patterns   = self.bot.candle_analyzer.detect_patterns(ohlc)
        adx_bundle = adx_efi_mom_trade_signal(fetchOHLC1(symbol, interval=str(timeframe), duration=60), symbol)
        fib        = fibonacci_retracement(ohlc, logger=self.bot.log_message)

        # --- NEW: CPR & AI Analysis ---
        #cpr_analysis = analyze_cpr_strategy(latest.to_dict(), pivot_data or {}, self.bot.ai_predictor)
        # NEW CALL (add ohlc_df):
        cpr_analysis = analyze_cpr_strategy(
            indicators=latest.to_dict(),
            pivot_data=pivot_data or {},
            ai_predictor=self.bot.ai_predictor,
            ohlc_df=ohlc  # ✅ Pass full OHLC dataframe for candle pattern detection
        )

        return {
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
            # ✅ NEW: Volume indicators
            "volume": float(latest["Volume"]),
            "volume_prev": float(prev["Volume"]),
            "volume_sma_20": float(ohlc["VolSMA20"].iloc[-1]),
            "volume_ratio": float(latest["volume_ratio"]) if pd.notna(latest["volume_ratio"]) else 1.0,
            "volume_surge": bool(latest["volume_surge"]) if pd.notna(latest["volume_surge"]) else False,
            "volume_extreme": bool(latest["volume_extreme"]) if pd.notna(latest["volume_extreme"]) else False,

            # ✅ NEW: Momentum indicators
            "momentum_10": float(latest["momentum_10"]) if pd.notna(latest["momentum_10"]) else 0.0,
            "momentum_pct": float(latest["momentum_pct"]) if pd.notna(latest["momentum_pct"]) else 0.0,
            "roc_10": float(latest["roc_10"]) if pd.notna(latest["roc_10"]) else 0.0,
            "acceleration": float(latest["acceleration"]) if pd.notna(latest["acceleration"]) else 0.0,

            # ✅ NEW: Combined signals
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
            "support": float(latest["Support"]), "resistance": float(latest["Resistance"]),
            # --- CORRECTED: Bollinger Bandwidth for volatility filter ---
            "bb_bandwidth": ((latest.get("BB_upper", 0) - latest.get("BB_lower", 0)) / latest.get("BB_mid", 1))
                            if (latest.get("BB_mid") and latest.get("BB_mid") > 0 and
                                latest.get("BB_upper") and latest.get("BB_lower"))
                            else 0.0,

            "ATR": float(latest["ATR"]),

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
            "productType": "INTRADAY",
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

    # -------------------------
    # Fine-tuning helpers
    # -------------------------
    def _normalize_tf(self, tf: str) -> str:
        if not tf:
            return "15m"
        tf = str(tf).strip().lower()
        # normalize common variants
        tf = tf.replace("min", "m").replace("minutes", "m").replace("minute", "m")
        tf = tf.replace("hour", "h").replace("hours", "h")
        if tf in ("1", "1m"): return "1m"
        if tf in ("5", "5m"): return "5m"
        if tf in ("15", "15m"): return "15m"
        if tf in ("30", "30m"): return "30m"
        if tf in ("60", "1h", "60m"): return "1h"
        return tf

    def _get_tuning(self, tf: str = None) -> dict:
        # timeframe resolution priority:
        # 1) explicit tf arg
        # 2) self.primary_tf / self.timeframe (if present)
        # 3) env AI_CPR_TF
        # 4) default 15m
        tf0 = tf
        if not tf0:
            tf0 = getattr(self, "primary_tf", None) or getattr(self, "timeframe", None)
        if not tf0:
            tf0 = os.environ.get("AI_CPR_TF", "15m")
        tf0 = self._normalize_tf(tf0)

        base = dict(DEFAULT_TUNING.get(tf0, DEFAULT_TUNING["15m"]))
        profile = (os.environ.get("AI_CPR_PROFILE", "balanced") or "balanced").strip().lower()
        adj = PROFILE_ADJUST.get(profile, {})
        for k, v in adj.items():
            if k in base and isinstance(base[k], (int, float)) and isinstance(v, (int, float)):
                base[k] = float(base[k]) + float(v)
        base["tf"] = tf0
        base["profile"] = profile
        return base

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

    def __init__(self, fyers_service, symbol, lot_size, log_fn, state_path, ai_predictor, event_log=None, bot=None, **_):
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
        
        # 🔥 FIX #3: Minimum hold time to prevent rapid flipping (30m aligned)
        self.position.setdefault("_bars_since_entry", 0)
        self.position.setdefault("_last_evaluated_bar", None)
        self.position.setdefault("_entry_bar_key", None)
        self.MIN_HOLD_BARS = 1  # Hold for at least 1 candle (30 min)

        # ✅ NEW: Production Safety Features
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
            self.INITIAL_SL_ATR = 2.0  # Wider initial SL for NATGAS (was 1.2/1.5)

            self.log("[NATGAS-SL] Using optimized stop loss settings", False)

        # ✅ NEW: AI Stability & Cooldown State
        from collections import deque
        self._ai_dir_hist = deque(maxlen=3)   # last 3 AI directions
        self._ai_last_dir = None
        self._ai_last_flip_ts = 0
        self.AI_STABLE_COUNT = 2              # need 2 of last 3 to match
        self.AI_FLIP_COOLDOWN_SEC = 120       # ignore opposite flips for 2 minutes

        # 🔧 FIX: Signal weights for priority-based voting (higher = more trusted)
        self.SIGNAL_WEIGHTS = {
            "rejection_candle": 1.0,      # High confidence pivot rejections
            "volume_breakout": 0.95,      # Strong volume + price action
            "ai_cpr": 0.90,               # AI predictions (data-driven)
            "early_breakout": 0.85,       # Early momentum detection
            "cpr_strategy": 0.80,         # CPR-based signals
            "momentum_shift": 0.75,       # Momentum changes
            "vwap": 0.70,                 # VWAP crossovers
            "volume_momentum": 0.70,      # Volume + momentum combo
            "price_action": 0.65,         # Price action patterns
            "momentum_direct": 0.65,      # Direct momentum signals
            "bid_ask_pressure": 0.60,     # Order flow
        }

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

    # 🔧 FIX: Signal conflict detection and priority-based voting
    def _check_signal_conflicts(self, signals, confidences):
        """
        Detect conflicting signals and determine if entry should be allowed
        Returns: (can_trade: bool, final_signal: str, conflict_reason: str)
        """
        buy_signals = []
        sell_signals = []
        
        # Collect signals with their weights
        for source, signal in signals.items():
            if signal == "BUY":
                weight = self.SIGNAL_WEIGHTS.get(source, 0.5)  # Default 0.5 if not defined
                weighted_conf = confidences[source] * weight
                buy_signals.append((source, confidences[source], weighted_conf))
            elif signal == "SELL":
                weight = self.SIGNAL_WEIGHTS.get(source, 0.5)
                weighted_conf = confidences[source] * weight
                sell_signals.append((source, confidences[source], weighted_conf))
        
        buy_count = len(buy_signals)
        sell_count = len(sell_signals)
        
        # Calculate weighted scores
        buy_score = sum(weighted_conf for _, _, weighted_conf in buy_signals)
        sell_score = sum(weighted_conf for _, _, weighted_conf in sell_signals)
        
        # Log signal details
        if buy_signals:
            self.log(f"[SIGNALS] BUY ({buy_count}): {[(s, f'{c:.2f}', f'{w:.2f}') for s, c, w in buy_signals]}", True)
        if sell_signals:
            self.log(f"[SIGNALS] SELL ({sell_count}): {[(s, f'{c:.2f}', f'{w:.2f}') for s, c, w in sell_signals]}", True)
        
        # 🔴 CONFLICT DETECTION: Block if significant disagreement
        if buy_count > 0 and sell_count > 0:
            conflict_ratio = min(buy_score, sell_score) / max(buy_score, sell_score) if max(buy_score, sell_score) > 0 else 0
            
            if conflict_ratio > 0.6:  # High conflict (60%+ opposing strength)
                return False, None, f"⛔ Major conflict: {buy_count} BUY (score: {buy_score:.2f}) vs {sell_count} SELL (score: {sell_score:.2f})"
        
        # 🟢 DETERMINE FINAL SIGNAL: Require 50% higher score to win
        min_score_threshold = 0.4  # Minimum score to consider
        conviction_multiplier = 1.5  # Score must be 1.5x opposing side
        
        if buy_score > sell_score * conviction_multiplier and buy_score >= min_score_threshold:
            return True, "BUY", f"✅ BUY score {buy_score:.2f} dominates SELL {sell_score:.2f}"
        elif sell_score > buy_score * conviction_multiplier and sell_score >= min_score_threshold:
            return True, "SELL", f"✅ SELL score {sell_score:.2f} dominates BUY {buy_score:.2f}"
        
        return False, None, f"⚠️ Insufficient conviction: BUY {buy_score:.2f} vs SELL {sell_score:.2f} (need {conviction_multiplier}x advantage)"

    def _safe_cpr_compare(self, val1, op, val2):
        """Safe comparison for CPR levels to avoid None/invalid errors"""
        try:
            if val1 is None or val2 is None:
                return False
            v1 = float(val1)
            v2 = float(val2)
            if op == ">": return v1 > v2
            if op == "<": return v1 < v2
            if op == ">=": return v1 >= v2
            if op == "<=": return v1 <= v2
            return False
        except (ValueError, TypeError):
            return False


    def _load_state(self):
        self.position = robust_load_json(self.state_path, self.log, default={})

        # ✅ ADD THESE VALIDATION CHECKS:
        # Ensure position has valid type
        if "type" not in self.position or self.position["type"] not in ["FLAT", "BUY", "SELL"]:
            self.log(f"⚠️ [STATE-FIX] Invalid position type: {self.position.get('type')}, resetting to FLAT", False)
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

    def _compute_dynamic_sl(self, entry_price, atr, direction, tf="5", adx=None, ai_conf=None):
        """
        Compute dynamic stop-loss distance based on volatility, timeframe, and signal confidence.
        """
        if not entry_price or not atr:
            return None, None

        # Base SL multiplier
        base_mult = self.INITIAL_SL_ATR if hasattr(self, 'INITIAL_SL_ATR') else 1.5

        # Timeframe adjustments
        tf_str = str(tf).lower()
        if "15" in tf_str:
            base_mult = max(base_mult, 1.8)
        elif "30" in tf_str:
            base_mult = max(base_mult, 2.0)
        elif "60" in tf_str or "1h" in tf_str:
            base_mult = max(base_mult, 2.5)
        
        # Volatility dampener: If ATR is huge relative to price (>0.5% per bar), reduce multiplier
        # This prevents stops from being ridiculously wide in crash scenarios
        atr_pct = (atr / entry_price) * 100
        if atr_pct > 0.5:
             base_mult *= 0.8
        
        # AI Confidence boost: Tighten SL if confidence is very high (sniper entry)
        if ai_conf and ai_conf >= 0.85:
             base_mult *= 0.9
        
        # ADX Adjustment: Relax SL in strong trends to avoid noise wicks
        if adx and adx > 30:
             base_mult *= 1.2

        sl_dist = atr * base_mult
        
        # Enforce min/max risk limits relative to price
        # Max risk: 1.5% of entry price
        # Min risk: 0.25% of entry price (to prevent stop hunt on low vol)
        max_risk_pts = entry_price * 0.015
        min_risk_pts = entry_price * 0.0025
        
        sl_dist = max(min_risk_pts, min(sl_dist, max_risk_pts))
        
        if direction == "BUY":
            return entry_price - sl_dist, sl_dist
        else:
            return entry_price + sl_dist, sl_dist

    def _process_entry(self, side, reason, ltp, atr, bar_key=None, indsP=None, ai_conf=0.0):

        # ✅ NEW: Reset daily counters if new day
        today = dt.datetime.now(self.IST).date()
        if today != self.last_reset_date:
            self.daily_loss = 0
            self.trades_today = 0
            self.TRADING_HALTED = False
            self.last_reset_date = today
            self.log("[SAFETY] Daily counters reset for new trading day", False)

        # ✅ NEW: Check if trading halted
        if self.TRADING_HALTED:
            self.log(
                f"🚨 [SAFETY] Trading HALTED - Circuit breaker active\n"
                f"  Daily Loss: ₹{self.daily_loss}\n"
                f"  Trades Today: {self.trades_today}",
                False
            )
            return False

        # ✅ NEW: Check daily loss limit
        if self.daily_loss <= -self.daily_loss_limit:
            self.log(
                f"🚨 [SAFETY] Daily loss limit reached\n"
                f"  Loss: ₹{self.daily_loss} / ₹{self.daily_loss_limit}\n"
                f"  No more entries today",
                False
            )
            return False

        # ✅ NEW: Check max trades per day
        if self.trades_today >= self.max_trades_per_day:
            self.log(
                f"⚠️ [SAFETY] Max trades per day reached\n"
                f"  Trades: {self.trades_today} / {self.max_trades_per_day}\n"
                f"  No more entries today",
                False
            )
            return False

        # ✅ FIX: Normalize position type first
        current_pos = self.position.get("type", "FLAT")
        if current_pos not in ["FLAT", "BUY", "SELL"]:
            self.log(
                f"⚠️ [STATE-CORRUPTION] Invalid position type: {current_pos}, resetting to FLAT",
                False
            )
            self.position["type"] = "FLAT"
            current_pos = "FLAT"

        # Check 1: Already in position?
        if current_pos != "FLAT":
            self.log(
                f"⚠️ [ENTRY-BLOCKED] Already in {current_pos} position | "
                f"Order ID: {self.position.get('order_id')}",
                False
            )
            return False

        # Check 2: Just exited on this bar?
        if self.position.get("_last_action_bar") == bar_key:
            self.log(
                f"⚠️ [ENTRY-BLOCKED] Just exited on bar {bar_key} | "
                f"Wait for next candle (cooldown active)",
                False
            )
            return False

        # Check 3: Valid ATR?
        if atr is None or atr <= 0:
            self.log(
                f"⚠️ [ENTRY-BLOCKED] Invalid ATR: {atr} | Cannot calculate stop loss",
                False
            )
            return False

        # Check 4: Valid LTP?
        if ltp is None or ltp <= 0:
            self.log(
                f"⚠️ [ENTRY-BLOCKED] Invalid LTP: {ltp} | Cannot place order",
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

            # 🔥 DYNAMIC STOP LOSS CALCULATION
            tf_from_key = "5"
            if bar_key and ":" in str(bar_key):
                tf_from_key = str(bar_key).split(":")[0]
            
            adx_val = self._f(indsP.get("adx")) if indsP else None
            
            initial_sl, r_amount = self._compute_dynamic_sl(
                entry_price=ltp,
                atr=atr,
                direction=side,
                tf=tf_from_key,
                adx=adx_val,
                ai_conf=ai_conf
            )
            
            r_mult = r_amount if r_amount else (self.INITIAL_SL_ATR * atr)
            if not initial_sl:
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

            # Save position state
            self.position = {
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
                "_exits_this_bar": 0,
                "_global_last_trade_bar": bar_key,  # 🔥 FIX: Set shared lock
                "_reversal_history": self.position.get("_reversal_history", [])  # 🔥 FIX: Preserve history
            }

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
        # ✅ ADD: Store entry details for later PnL calculation
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
                # ✅ NEW: Update daily loss tracking
                self.daily_loss += ltp_diff
                self.trades_today += 1

                self.log(
                    f"[SAFETY] Daily Stats Updated:\n"
                    f"  Daily Profit/Loss (points): {self.daily_loss:+.2f}\n"
                    f"  Trade P&L: ₹{ltp_diff:.2f}\n"
                    f"  Daily Total: ₹{self.daily_loss:.2f} / ₹{self.daily_loss_limit}\n"
                    f"  Trades Today: {self.trades_today} / {self.max_trades_per_day}",
                    False
                )
            
            # ✅ FIX: Store last exit timestamp for cooldown logic
            self.position["last_exit_ts"] = time.time()

            # ✅ NEW: Circuit breaker check (3 consecutive losses totaling ₹300)
            if self.check_circuit_breaker():
                self.TRADING_HALTED = True
                self.log("🚨🚨🚨 CIRCUIT BREAKER ACTIVATED 🚨🚨🚨", False)

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
                    f"🚨 CIRCUIT BREAKER TRIGGERED 🚨\n"
                    f"Last 3 trades total: ₹{total_loss:.2f}\n"
                    f"Threshold: ₹-300\n"
                    f"HALTING ALL TRADING UNTIL MANUAL REVIEW",
                    False
                )
                return True
        return False

    def _update_ultra_tight_trailing(self, ltp, atr):
        """
        Layer 3: Ultra-tight trailing for big winners
        ⚠️ Only activates when profit >= ₹3,000
        """
        if not self.position or self.position.get("type") == "FLAT":
            return

        entry = self._f(self.position.get("entry_price"))
        side = self.position.get("type")
        current_sl = self._f(self.position.get("stop_loss"))

        if not entry or not side or not current_sl:
            return

        # Calculate profit
        if side == "BUY":
            profit_points = ltp - entry
        else:
            profit_points = entry - ltp

        profit_rupees = profit_points * 250

        # ⚠️ CRITICAL: Only activate for big winners (₹3,000+)
        # Let Layer 2 (tiered trailing) handle ₹0-3,000
        #if profit_rupees < 3000:
        #    return
        if profit_rupees <= 0 or profit_rupees < 3000:
            if profit_rupees <= 0:
                self.log(f"[ULTRA-TIGHT] Skipping - in loss: ₹{profit_rupees:.0f}", True)
            return

        # 🎯 HYBRID APPROACH: Best of both versions
        # More conservative early, aggressive later

        if profit_points < 16:  # ₹3,000-4,000 (12-16 pts)
            trail_distance = 1.2  # Wide breathing room
            tier = "Early Big Win"

        elif profit_points < 24:  # ₹4,000-6,000 (16-24 pts)
            trail_distance = 1.0  # Medium
            tier = "Good Big Win"

        elif profit_points < 32:  # ₹6,000-8,000 (24-32 pts)
            trail_distance = 0.8  # Getting tight
            tier = "Great Big Win"

        elif profit_points < 40:  # ₹8,000-10,000 (32-40 pts)
            trail_distance = 0.6  # Tight - your target zone
            tier = "Excellent Big Win"

        else:  # ₹10,000+ (40+ pts)
            trail_distance = 0.4  # Ultra-tight
            tier = "Epic Big Win"

        # Calculate new SL
        if side == "BUY":
            new_sl = ltp - trail_distance
        else:
            new_sl = ltp + trail_distance

        # Only move SL in favorable direction
        should_update = False
        if side == "BUY" and new_sl > current_sl:
            should_update = True
        elif side == "SELL" and new_sl < current_sl:
            should_update = True

        if should_update:
            locked_profit = new_sl - entry if side == "BUY" else entry - new_sl
            locked_rupees = locked_profit * 250

            self.position["stop_loss"] = round(new_sl, 2)

            self.log(
                f"🎯 [ULTRA-TIGHT] {tier}\n"
                f"  Profit: ₹{profit_rupees:.0f} ({profit_points:.1f} pts)\n"
                f"  Trail: {trail_distance:.1f}pts (₹{trail_distance * 250:.0f})\n"
                f"  New SL: ₹{new_sl:.2f}\n"
                f"  Locked: ₹{locked_rupees:.0f}",
                False
            )

            self._save_state()

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

        # ✅ CHECK 1: On candle close (patient exit)
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

                    # ✅ Check 15-min trend alignment
                    trend_15m_confirms = (trend_15m == "GREEN")

                    if is_red_candle and closed_above_ema5:
                        current_profit = entry - ltp
                        profit_rupees = current_profit * 250

                        confirmation_msg = ""
                        if trend_15m_confirms:
                            confirmation_msg = " ✅ 15m trend confirms"
                        else:
                            confirmation_msg = " ⚠️ 15m trend mixed"

                        self.log(
                            f"🚨 [EMA5-EXIT] Exiting SELL position\n"
                            f"  ⏰ Candle CLOSED: {current_bar_ts}\n"
                            f"  📊 30m Trend: GREEN | 15m Trend: {trend_15m}{confirmation_msg}\n"
                            f"  🕯️ Red candle: O:{open_price:.2f} → C:{close:.2f}\n"
                            f"  ✅ Close {close:.2f} > EMA-5 {ema5:.2f}\n"
                            f"  💰 Profit: ₹{profit_rupees:.0f}\n"
                            f"  🎯 EXIT NOW",
                            False
                        )
                        return self._process_exit("EMA-5 Reversal (candle closed)", ltp)

                    elif is_red_candle and not closed_above_ema5:
                        self.log(
                            f"⏸️ [EMA5-HOLD] Holding SELL\n"
                            f"  📊 30m: GREEN, 15m: {trend_15m}\n"
                            f"  Close {close:.2f} ≤ EMA-5 {ema5:.2f}\n"
                            f"  🎯 HOLD - no confirmation",
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
                            confirmation_msg = " ✅ 15m trend confirms"
                        else:
                            confirmation_msg = " ⚠️ 15m trend mixed"

                        self.log(
                            f"🚨 [EMA5-EXIT] Exiting BUY position\n"
                            f"  ⏰ Candle CLOSED: {current_bar_ts}\n"
                            f"  📊 30m Trend: RED | 15m Trend: {trend_15m}{confirmation_msg}\n"
                            f"  🕯️ Green candle: O:{open_price:.2f} → C:{close:.2f}\n"
                            f"  ✅ Close {close:.2f} < EMA-5 {ema5:.2f}\n"
                            f"  💰 Profit: ₹{profit_rupees:.0f}\n"
                            f"  🎯 EXIT NOW",
                            False
                        )
                        return self._process_exit("EMA-5 Reversal (candle closed)", ltp)

                    elif is_green_candle and not closed_below_ema5:
                        self.log(
                            f"⏸️ [EMA5-HOLD] Holding BUY\n"
                            f"  📊 30m: RED, 15m: {trend_15m}\n"
                            f"  Close {close:.2f} ≥ EMA-5 {ema5:.2f}\n"
                            f"  🎯 HOLD - no confirmation",
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

                # 🚨 Emergency: Losing AND price far above EMA-5 AND 15m confirms
                emergency_condition = (
                        loss_rupees < -250 and
                        ema5_distance_pct > 1.0 and
                        (trend_15m == "GREEN" or trend_15m == "NEUTRAL")  # 15m not opposing
                )

                if emergency_condition:
                    self.log(
                        f"⚡ [EMERGENCY-EMA5-EXIT] Fast exit - large reversal\n"
                        f"  💥 Loss: ₹{loss_rupees:.0f}\n"
                        f"  📊 Price {ema5_distance_pct:.2f}% above EMA-5\n"
                        f"  🔴 30m: GREEN, 15m: {trend_15m}\n"
                        f"  🎯 EXIT NOW (not waiting for candle close)",
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
                        f"  📊 Price {ema5_distance_pct:.2f}% below EMA-5\n"
                        f"  🔴 30m: RED, 15m: {trend_15m}\n"
                        f"  🎯 EXIT NOW (not waiting for candle close)",
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
                        # ✅ FIX: Increased from 2 to 80 candles for sufficient AI context
                        _ptf = getattr(self, 'last_known_primary_tf', None) or getattr(self, 'tf_selected', None) or globals().get('tf_selected', None) or '15'
                        try:
                            primary_tf = str(int(float(_ptf)))
                        except Exception:
                            primary_tf = str(_ptf)
                        ohlc_for_ai_exit = self.bot.fetch_ohlc(self.symbol, str(primary_tf), 80)
                    except Exception as e:
                        self.log(f"[AI-EXIT] Failed to fetch OHLC: {e}", True)
                # ✅ FIX: Use the inds parameter that's passed to the function
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
                    f"📊 [SUPERTREND-FLIP] Multi-TF flip to SELL (conf: {conf:.2f})\n"
                    f"  Profit: ₹{current_profit * 250:.0f}\n"
                    f"  Action: Exit before breakdown",
                False
                )
                return self._process_exit("Multi-TF SuperTrend Flip", ltp)

            if side == "SELL" and aligned == "BUY" and conf >= 0.66 and max_profit > 1:
                self.log(
                    f"📊 [SUPERTREND-FLIP] Multi-TF flip to BUY (conf: {conf:.2f})\n"
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
        🎯 Predict if we should exit in next 1-3 candles
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
        profit = (ltp - entry) if side == "BUY" else (entry - ltp)
        profit_rupees = profit * 250

        # 🔥 NEW: Don't exit if profit < ₹1,500 and trend is strong
        if profit_rupees < 1100:
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
        MAX_LOSS_PCT = 1.0  # Exit at -2.5% loss

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
        # ✅ Don't exit too early - wait for bigger profit first
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

            # ✅ More defensive price_above_st check
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

            # ✅ More lenient trend detection (was failing too often)
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
                f"🎯 TIER UP! Profit ₹{profit:.2f} hit {profit_target:.2f}. "
                f"Next target ₹{next_target:.2f} (Trend: {tier_label})", False
            )
            profit_target = next_target
            current_tier += 1

        # ==========================================
        # ✅ Exit logic
        # ==========================================
        if profit >= profit_target:
            if trend_strong or trend_very_strong:
                if drawdown >= trailing_pct * max_profit:
                    self.log(
                        f"⚠️ {tier_label} trend trailing stop: Profit dropped {drawdown:.2f} "
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
                    f"⚠️ [PROFIT-PROTECTION] Exiting - Dropped {drawdown:.2f} ({drawdown_pct:.0f}%) from peak ₹{max_profit:.2f} | "
                    f"Max allowed: {max_drawdown_pct}% | Current profit: ₹{profit:.2f}",
                    False
                )
                try:
                    self._process_exit("Early Profit Protection", ltp)
                except Exception as e:
                    self.log(f"[ERROR] Profit protection exit failed: {e}")
                return

        # ✅ Improved trend reversal detection
        if inds and isinstance(inds, dict) and max_profit > 0:
            e5 = self._f(inds.get("ema_5"))
            e21 = self._f(inds.get("ema_21"))
            macd_color = str(inds.get("macd_color", "")).strip().lower()
            adx = self._f(inds.get("adx"))

            # ✅ More robust reversal check
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

    def _update_trailing_and_breakeven(self, ltp, atr):
        """
        Adaptive trailing stop based on R-multiples and volatility (ATR).

        Key improvements:
          - Monotonic tightening (never loosens SL)
          - Step/hysteresis to avoid frequent micro-updates
          - Uses ultra-tight trailing in high-profit conditions
        """
        if not self.position or self.position.get("type") in (None, "FLAT"):
            return

        side = self.position.get("type")
        if side not in ("BUY", "SELL"):
            return

        ltp = self._f(ltp)
        atr = self._f(atr)
        cur_sl = self._f(self.position.get("stop_loss"))
        entry = self._f(self.position.get("entry_price"))
        r_dist = self._f(self.position.get("r_mult"))  # risk distance

        if not ltp or entry is None or cur_sl is None or not atr or atr <= 0 or not r_dist:
            return

        # Profit
        profit_points = (ltp - entry) if side == "BUY" else (entry - ltp)
        profit_r = profit_points / r_dist if r_dist > 0 else 0.0

        # Throttle SL updates per bar (avoid thrash)
        bar_key = self.position.get("_last_bar_key")
        last_action_bar = self.position.get("_last_trail_action_bar")
        if bar_key is not None and last_action_bar == bar_key:
            # already adjusted this bar
            return

        min_step = max(0.02, 0.08 * atr)

        # Tiered trailing
        new_sl = cur_sl
        reason = None

        # Tier 0: <0.5R => no action
        if profit_r < 0.5:
            return

        # Tier 1: 0.5R..1R => reduce loss size (protect partial)
        if 0.5 <= profit_r < 1.0:
            # Allow max loss of ~0.25R from entry
            if side == "BUY":
                cand = entry - 0.25 * r_dist
                new_sl = max(new_sl, cand)
            else:
                cand = entry + 0.25 * r_dist
                new_sl = min(new_sl, cand)
            reason = f"TIER-0.5 protect (profit={profit_r:.2f}R)"

        # Tier 1.5: 1R..2R => move to breakeven (+ small buffer)
        elif 1.0 <= profit_r < 2.0:
            be_buf = 0.02  # small to cover costs
            if side == "BUY":
                new_sl = max(new_sl, entry + be_buf)
            else:
                new_sl = min(new_sl, entry - be_buf)
            self.position["breakeven_set"] = True
            reason = f"TIER-1 BE set (profit={profit_r:.2f}R)"

        # Tier 2: 2R..4R => trail by 1.2 ATR
        elif 2.0 <= profit_r < 4.0:
            trail_dist = 1.2 * atr
            if side == "BUY":
                new_sl = max(new_sl, ltp - trail_dist)
                new_sl = max(new_sl, entry)  # never below BE
            else:
                new_sl = min(new_sl, ltp + trail_dist)
                new_sl = min(new_sl, entry)
            reason = f"TIER-2 1.2ATR trail (profit={profit_r:.2f}R)"

        # Tier 3: >=4R => ultra-tight trailing + ATR trail
        else:
            # First, apply ultra tight trailing (if implemented)
            try:
                self._update_ultra_tight_trailing(ltp, atr)
                # It may have updated stop_loss already
                cur_sl2 = self._f(self.position.get("stop_loss"))
                if cur_sl2 is not None:
                    cur_sl = cur_sl2
            except Exception as e:
                self.log(f"[TRAIL] Ultra tight trailing error: {e}", True)

            # Then enforce an ATR-based hard trail as safety
            trail_dist = 0.9 * atr
            if side == "BUY":
                new_sl = max(cur_sl, ltp - trail_dist)
            else:
                new_sl = min(cur_sl, ltp + trail_dist)

            reason = f"TIER-3 ultra trail (profit={profit_r:.2f}R)"

        # Enforce monotonic tightening
        if side == "BUY":
            if new_sl <= cur_sl + min_step:
                return
        else:
            if new_sl >= cur_sl - min_step:
                return

        # Commit update
        self.position["stop_loss"] = float(round(new_sl, 4))
        self.position["_last_trail_action_bar"] = bar_key
        self.log(f"[TRAIL] ✅ {side} SL updated: {cur_sl:.2f} -> {new_sl:.2f} | {reason}", True)
        self._save_state()

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

    def execute_ai_cpr_strategy(self, ltp, all_inds, primary_tf=None):
        """
        Execute AI CPR-based trading strategy independently
        🔥 FIX #1: Default to 30m, not 5m
        """
        t = self._get_tuning(primary_tf)

        if not self.AI_CPR_ENABLED:
            return


        # Resolve primary timeframe dynamically (avoid hard-coded 30m defaults)
        if primary_tf is None:
            primary_tf = (
                getattr(self, "last_known_primary_tf", None)
                or getattr(self, "last_primary_tf", None)
                or getattr(getattr(self, "bot", None), "tf_selected", None)
                or globals().get("tf_selected", None)
                or "15"
            )
        primary_tf = str(primary_tf).strip()
        self.last_known_ltp = ltp
        self.last_known_inds = all_inds
        self.last_known_primary_tf = primary_tf

        def _tf(tf):
            return self._norm_tf(all_inds, str(tf))

        inds = _tf(primary_tf)
        if not isinstance(inds, dict) or not inds.get("timestamp"):
            self.log(f"[AI-CPR] No indicators for TF={primary_tf}.", True)
            return

        # Post-impulse cooldown (prevents whipsaw entries immediately after large candles)
        self._apply_impulse_cooldown(inds, str(primary_tf))

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

        # 🔥 FIX #4: Shared lock key across AI and Unified strategies
        if self.position.get("_global_last_trade_bar") == bar_key:
            self.log(f"[SHARED-LOCK] Trade already executed this bar {bar_key} - blocking AI", True)
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
        t = self._get_tuning()

        if ohlc_df is None or len(ohlc_df) < 20:
            return "UNKNOWN", 0.0

        try:
            # Get indicators for regime detection
            adx = self._f(indicators.get("adx"), 0)
            bb_bandwidth = self._f(indicators.get("bb_bandwidth"), 0)
            atr = self._f(indicators.get("ATR"), 0)

            close = self._f(indicators.get("close"), self._f(indicators.get("ltp"), 0))
            atr_pct = (atr / close) if (close and atr) else 0.0
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
            # SQUEEZE / COMPRESSION (treat as CHOPPY with high confidence)
            # ═══════════════════════════════════════════════════════════
            squeeze_score = 0.0
            if bb_bandwidth > 0 and bb_bandwidth < 0.009:
                squeeze_score += 0.5
            if atr_pct > 0 and atr_pct < 0.0032:
                squeeze_score += 0.3
            if volume_ratio < 1.15:
                squeeze_score += 0.2

            if squeeze_score >= 0.7:
                self.log(
                    f"[REGIME] 🧊 SQUEEZE/COMPRESSION detected (score: {squeeze_score:.2f})\n"
                    f"  BB Width: {bb_bandwidth:.4f}, ATR%: {atr_pct:.4f}, VolRatio: {volume_ratio:.2f}\n"
                    f"  → SuperTrend will be STRICT (breakouts only)",
                    False
                )
                return "CHOPPY", min(1.0, squeeze_score)

            # ═══════════════════════════════════════════════════════════
            # CHOPPY MARKET (Sideways, no clear direction)
            # ═══════════════════════════════════════════════════════════
            choppy_score = 0

            if adx < 20:
                choppy_score += 0.4  # Weak trend

            if bb_bandwidth < 0.008:  # Very narrow bands
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

            if bb_bandwidth > 0.018:  # Wide bands
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
        # ✅ BYPASS RULE 2.5: EXTREME VOLUME BREAKOUT
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

    def _update_dynamic_cpr_stop_loss(self, ltp, atr, pivot_data):
        """
        Dynamic CPR-based stop loss adjustment.

        Design goals (live-safe):
          1) Only *tighten* risk (never widen SL).
          2) Avoid SL thrashing: apply time + step hysteresis.
          3) Prefer CPR levels (BC/TC/S/R) with ATR buffers.
          4) Stay conservative while price is inside CPR (chop zone).

        Notes:
          - Uses self.position fields for throttling:
              _last_cpr_sl_update_ts
              _last_cpr_sl_value
        """
        # --- Validation ---
        if not self.position or self.position.get("type") == "FLAT":
            return

        side = self.position.get("type")
        if side not in ("BUY", "SELL"):
            return

        ltp = self._f(ltp)
        atr = self._f(atr)

        if not ltp or ltp <= 0:
            self.log(f"[CPR-SL] Invalid LTP: {ltp} - skipping dynamic SL", True)
            return
        if not atr or atr <= 0:
            self.log(f"[CPR-SL] Invalid ATR: {atr} - skipping dynamic SL", True)
            return
        if not pivot_data or not isinstance(pivot_data, dict):
            self.log("[CPR-SL] Invalid pivot data - skipping dynamic SL", True)
            return

        entry = self._f(self.position.get("entry_price"))
        cur_sl = self._f(self.position.get("stop_loss"))
        if entry is None or cur_sl is None:
            self.log("[CPR-SL] Missing entry/SL - skipping dynamic SL", True)
            return

        tc = self._f(pivot_data.get("TC"))
        bc = self._f(pivot_data.get("BC"))
        r1 = self._f(pivot_data.get("R1")); r2 = self._f(pivot_data.get("R2")); r3 = self._f(pivot_data.get("R3"))
        s1 = self._f(pivot_data.get("S1")); s2 = self._f(pivot_data.get("S2")); s3 = self._f(pivot_data.get("S3"))

        if tc is None or bc is None:
            self.log("[CPR-SL] TC/BC missing - skipping dynamic SL", True)
            return

        # --- Throttle / hysteresis ---
        now_ts = time.time()
        last_ts = self._f(self.position.get("_last_cpr_sl_update_ts"), 0.0) or 0.0
        last_val = self._f(self.position.get("_last_cpr_sl_value"), cur_sl) or cur_sl

        # Do not update more frequently than every 20 seconds (tick-safe)
        if now_ts - last_ts < 20:
            return

        # Minimum meaningful improvement step
        min_step = max(0.02, 0.10 * atr)  # NG noise handling

        # Determine if we are inside CPR (chop zone)
        low_cpr = min(bc, tc)
        high_cpr = max(bc, tc)
        inside_cpr = (low_cpr <= ltp <= high_cpr)

        # Profit context
        r_dist = self._f(self.position.get("r_mult"))
        profit_points = (ltp - entry) if side == "BUY" else (entry - ltp)
        profit_r = (profit_points / r_dist) if (r_dist and r_dist > 0) else 0.0

        # ATR buffers around pivot levels (reduce stop hunts)
        buff_lo = 0.12 * atr
        buff_hi = 0.18 * atr

        candidate_sl = None
        reason = None

        # --- Ruleset ---
        if side == "BUY":
            # Conservative while inside CPR unless already decent profit
            if inside_cpr and profit_r < 0.8:
                return

            # As price advances, raise SL to key floors
            # Priority order: BC -> TC -> R1 -> R2 (each with buffer)
            if ltp > high_cpr + buff_lo:
                # above CPR band: lock to BC or TC depending on position
                base = high_cpr if profit_r >= 1.0 else low_cpr
                candidate_sl = base - buff_lo
                reason = f"BUY above CPR band => SL to {base:.2f}-buf"

            if r1 and ltp >= r1 + buff_lo:
                cand = r1 - buff_hi
                if candidate_sl is None or cand > candidate_sl:
                    candidate_sl = cand
                    reason = f"BUY above R1 => SL to R1-buf"

            if r2 and ltp >= r2 + buff_lo:
                cand = r2 - buff_hi
                if candidate_sl is None or cand > candidate_sl:
                    candidate_sl = cand
                    reason = f"BUY above R2 => SL to R2-buf"

            if r3 and ltp >= r3 + buff_lo:
                cand = r3 - buff_hi
                if candidate_sl is None or cand > candidate_sl:
                    candidate_sl = cand
                    reason = f"BUY above R3 => SL to R3-buf"

            # Always at least breakeven once >1R
            if profit_r >= 1.0:
                be_cand = entry + 0.02  # tiny BE+ to cover costs
                if candidate_sl is None or be_cand > candidate_sl:
                    candidate_sl = be_cand
                    reason = "BUY >=1R => SL to BE+"

            # Tighten only (never lower for BUY)
            if candidate_sl is None:
                return
            new_sl = max(cur_sl, candidate_sl)

            # Hysteresis: avoid tiny changes
            if (new_sl - last_val) < min_step:
                return

            if new_sl > cur_sl:
                self.position["stop_loss"] = float(round(new_sl, 4))
                self.position["_last_cpr_sl_update_ts"] = now_ts
                self.position["_last_cpr_sl_value"] = float(round(new_sl, 4))
                self.log(f"[CPR-SL] ✅ BUY SL raised: {cur_sl:.2f} -> {new_sl:.2f} | {reason}", False)
                self._save_state()
            return

        # --- SELL side ---
        if inside_cpr and profit_r < 0.8:
            return

        # As price declines, lower SL to key ceilings (for shorts SL is above price)
        if ltp < low_cpr - buff_lo:
            base = low_cpr if profit_r >= 1.0 else high_cpr
            candidate_sl = base + buff_lo
            reason = f"SELL below CPR band => SL to {base:.2f}+buf"

        if s1 and ltp <= s1 - buff_lo:
            cand = s1 + buff_hi
            if candidate_sl is None or cand < candidate_sl:
                candidate_sl = cand
                reason = f"SELL below S1 => SL to S1+buf"

        if s2 and ltp <= s2 - buff_lo:
            cand = s2 + buff_hi
            if candidate_sl is None or cand < candidate_sl:
                candidate_sl = cand
                reason = f"SELL below S2 => SL to S2+buf"

        if s3 and ltp <= s3 - buff_lo:
            cand = s3 + buff_hi
            if candidate_sl is None or cand < candidate_sl:
                candidate_sl = cand
                reason = f"SELL below S3 => SL to S3+buf"

        if profit_r >= 1.0:
            be_cand = entry - 0.02
            if candidate_sl is None or be_cand < candidate_sl:
                candidate_sl = be_cand
                reason = "SELL >=1R => SL to BE-"

        if candidate_sl is None:
            return

        new_sl = min(cur_sl, candidate_sl)  # tighten for SELL

        if (last_val - new_sl) < min_step:
            return

        if new_sl < cur_sl:
            self.position["stop_loss"] = float(round(new_sl, 4))
            self.position["_last_cpr_sl_update_ts"] = now_ts
            self.position["_last_cpr_sl_value"] = float(round(new_sl, 4))
            self.log(f"[CPR-SL] ✅ SELL SL lowered: {cur_sl:.2f} -> {new_sl:.2f} | {reason}", False)
            self._save_state()

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
        Detect if the market is in a *strong* directional trend.

        This is used as a *context* signal:
          - Strong trend => avoid micro-exits on minor pullbacks.
          - No strong trend => allow tighter exits / reversal exits.

        Returns:
            ("STRONG_UPTREND", confidence) or ("STRONG_DOWNTREND", confidence) or (None, 0.0)
        """
        if ohlc_df is None or len(ohlc_df) < 25:
            return None, 0.0

        try:
            recent = ohlc_df.iloc[-25:].copy()

            highs = recent["High"].astype(float).values
            lows = recent["Low"].astype(float).values
            closes = recent["Close"].astype(float).values

            # Higher-high / lower-low structure (simple but robust)
            higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
            higher_lows  = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])
            lower_lows   = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i - 1])
            lower_highs  = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i - 1])

            # Trend slope (linear regression) - normalized
            # Use numpy only if already imported; else do manual slope (safe)
            n = len(closes)
            x_mean = (n - 1) / 2.0
            y_mean = float(sum(closes)) / n
            num = sum((i - x_mean) * (closes[i] - y_mean) for i in range(n))
            den = sum((i - x_mean) ** 2 for i in range(n)) or 1.0
            slope = num / den  # points per candle

            atr = self._f(indicators.get("atr"))
            if not atr or atr <= 0:
                atr = None

            slope_norm = (slope / atr) if atr else (slope / max(1e-6, closes[-1] * 0.002))
            momentum_pct = ((closes[-1] - closes[0]) / max(1e-6, closes[0])) * 100.0

            adx = self._f(indicators.get("adx"), 0)
            plus_di = self._f(indicators.get("plus_di"), None)
            minus_di = self._f(indicators.get("minus_di"), None)
            volume_ratio = self._f(indicators.get("volume_ratio"), 1.0)

            # Confidence building blocks (bounded)
            def clip(v, lo=0.0, hi=1.0):
                return max(lo, min(hi, v))

            adx_score = clip((adx - 18.0) / 22.0)  # 18->0, 40->1
            vol_score = clip((volume_ratio - 1.0) / 1.0)  # 1.0->0, 2.0->1
            mom_score = clip(abs(momentum_pct) / 2.0)  # 2% in 25 bars => strong
            slope_score = clip(abs(slope_norm) / 0.35)  # ~0.35 ATR per candle = very strong

            # Direction voting
            up_struct = (higher_highs + higher_lows) / (2 * (n - 1))
            dn_struct = (lower_lows + lower_highs) / (2 * (n - 1))

            di_bias = 0.0
            if plus_di is not None and minus_di is not None:
                di_bias = (plus_di - minus_di) / max(1e-6, (plus_di + minus_di))
            # di_bias in [-1, +1]

            # Decide direction
            up_votes = 0
            dn_votes = 0

            if slope > 0: up_votes += 1
            if slope < 0: dn_votes += 1

            if momentum_pct > 0.0: up_votes += 1
            if momentum_pct < 0.0: dn_votes += 1

            if di_bias > 0.05: up_votes += 1
            if di_bias < -0.05: dn_votes += 1

            if up_struct > dn_struct + 0.10: up_votes += 1
            if dn_struct > up_struct + 0.10: dn_votes += 1

            direction = None
            if up_votes >= 3 and up_votes > dn_votes:
                direction = "UP"
            elif dn_votes >= 3 and dn_votes > up_votes:
                direction = "DOWN"

            # Strength gate
            # Require ADX and at least one of slope/momentum/structure
            strength_gate = (adx >= 22) and (slope_score >= 0.35 or mom_score >= 0.35 or max(up_struct, dn_struct) >= 0.65)

            if not direction or not strength_gate:
                return None, 0.0

            base_conf = 0.45 + (0.20 * adx_score) + (0.15 * slope_score) + (0.10 * mom_score) + (0.10 * vol_score)
            struct_bonus = 0.10 * clip((max(up_struct, dn_struct) - 0.55) / 0.25)  # 0.55->0, 0.80->1
            di_bonus = 0.05 * clip(abs(di_bias) / 0.35)

            confidence = clip(base_conf + struct_bonus + di_bonus, 0.0, 0.92)

            if direction == "UP":
                self.log(
                    f"[TREND] STRONG UPTREND detected | conf={confidence:.2f} | "
                    f"ADX={adx:.1f} slope={slope:.4f} mom={momentum_pct:+.2f}% vol={volume_ratio:.2f}x",
                    True
                )
                return "STRONG_UPTREND", confidence

            self.log(
                f"[TREND] STRONG DOWNTREND detected | conf={confidence:.2f} | "
                f"ADX={adx:.1f} slope={slope:.4f} mom={momentum_pct:+.2f}% vol={volume_ratio:.2f}x",
                True
            )
            return "STRONG_DOWNTREND", confidence

        except Exception as e:
            self.log(f"[TREND-STRENGTH] Error: {e}", True)
            return None, 0.0

    def _detect_early_breakout(self, ltp, indicators, ohlc_df):
        """
        Detect breakout EARLY (before full voting completes).

        Upgrades vs. basic version:
          - Requires range compression + directional candle body
          - Incorporates VWAP displacement + orderbook pressure (if available)
          - Uses ATR-based tolerance when available (reduces false triggers on noise)

        Returns: ("BUY", confidence) or ("SELL", confidence) or (None, 0.0)
        """
        try:
            if ohlc_df is None or len(ohlc_df) < 8 or ltp is None:
                return None, 0.0

            recent = ohlc_df.iloc[-8:].copy()

            # Indicators
            volume_ratio = self._f(indicators.get("volume_ratio"), 1.0)
            momentum_pct = self._f(indicators.get("momentum_pct"), 0.0)
            vwap = self._f(indicators.get("VWAP"))
            atr = self._f(indicators.get("atr"))
            bidask_sig, bidask_conf = self._calculate_bid_ask_pressure(indicators)

            # Range compression (last 6 candles)
            last6 = recent.iloc[-6:]
            high6 = float(last6["High"].max())
            low6 = float(last6["Low"].min())
            mid6 = (high6 + low6) / 2.0 if (high6 and low6) else None
            range_pct = ((high6 - low6) / max(1e-6, low6)) * 100.0

            is_tight_range = range_pct < 0.90  # slightly tighter than 1% (NG noise is high)

            # Directional intent candle (last candle)
            c = recent.iloc[-1]
            o = float(c["Open"]); h = float(c["High"]); l = float(c["Low"]); cl = float(c["Close"])
            rng = max(1e-6, h - l)
            body = abs(cl - o)
            body_ratio = body / rng

            bullish_body = (cl > o) and (body_ratio >= 0.55) and ((h - cl) / rng <= 0.25)
            bearish_body = (cl < o) and (body_ratio >= 0.55) and ((cl - l) / rng <= 0.25)

            # Break level (recent micro-range)
            prev_high = float(recent.iloc[-7:-1]["High"].max())
            prev_low = float(recent.iloc[-7:-1]["Low"].min())

            # ATR tolerance
            tol = 0.0
            if atr and atr > 0:
                tol = 0.08 * atr  # 8% ATR breakout tolerance

            # VWAP displacement
            vwap_bonus = 0.0
            vwap_dist = 0.0
            if vwap:
                vwap_dist = ((ltp - vwap) / max(1e-6, vwap)) * 100.0
                if abs(vwap_dist) >= 0.25:
                    vwap_bonus = 0.08

            # Core breakout conditions
            has_volume = volume_ratio >= 1.4
            has_momentum = abs(momentum_pct) >= 0.35

            # Orderbook pressure optional confirmation
            ob_bonus = 0.0
            if bidask_sig:
                ob_bonus = 0.06 * min(1.0, bidask_conf / 0.8)

            # Bullish breakout
            if is_tight_range and has_volume and (momentum_pct > 0.35) and bullish_body and (ltp >= prev_high + tol):
                confidence = 0.62 + 0.12 * min(1.0, (volume_ratio - 1.0)) + 0.10 * min(1.0, abs(momentum_pct) / 0.8)
                confidence += vwap_bonus + (ob_bonus if bidask_sig == "BUY" else 0.0)
                confidence = min(0.90, confidence)

                self.log(
                    f"🚀 [EARLY-BREAKOUT] BUY breakout | conf={confidence:.2f} | "
                    f"range={range_pct:.2f}% vol={volume_ratio:.2f}x mom={momentum_pct:+.2f}% "
                    f"vwap={vwap_dist:+.2f}% ob={bidask_sig or '-'}",
                    True
                )
                return "BUY", confidence

            # Bearish breakout
            if is_tight_range and has_volume and (momentum_pct < -0.35) and bearish_body and (ltp <= prev_low - tol):
                confidence = 0.62 + 0.12 * min(1.0, (volume_ratio - 1.0)) + 0.10 * min(1.0, abs(momentum_pct) / 0.8)
                confidence += vwap_bonus + (ob_bonus if bidask_sig == "SELL" else 0.0)
                confidence = min(0.90, confidence)

                self.log(
                    f"🚀 [EARLY-BREAKOUT] SELL breakout | conf={confidence:.2f} | "
                    f"range={range_pct:.2f}% vol={volume_ratio:.2f}x mom={momentum_pct:+.2f}% "
                    f"vwap={vwap_dist:+.2f}% ob={bidask_sig or '-'}",
                    True
                )
                return "SELL", confidence

            # Fallback: allow earlier trigger without candle-body if price displacement is large
            if is_tight_range and has_volume and has_momentum and vwap and abs(vwap_dist) >= 0.40:
                direction = "BUY" if vwap_dist > 0 else "SELL"
                confidence = min(0.82, 0.62 + 0.10 + 0.08 + 0.06)  # deterministic
                self.log(
                    f"[EARLY-BREAKOUT] VWAP displacement fallback => {direction} | conf={confidence:.2f} | vwap={vwap_dist:+.2f}%",
                    True
                )
                return direction, confidence

            return None, 0.0

        except Exception as e:
            self.log(f"[EARLY-BREAKOUT] Error: {e}", True)
            return None, 0.0

    def _should_exit_on_stop_loss(self, ltp, current_pos):
        """
        Simple check: Should we exit when SL is hit?

        Returns: (should_exit: bool, reason: str)
        """
        current_sl = self._f(self.position.get("stop_loss"))
        entry_price = self._f(self.position.get("entry_price"))

        if not (current_sl and entry_price and ltp):
            return True, "Invalid data"

        # Calculate loss
        if current_pos == "BUY":
            loss_amount = entry_price - ltp
        else:
            loss_amount = ltp - entry_price

        loss_pct = (loss_amount / entry_price * 100) if entry_price > 0 else 0

        # ==========================================
        # CHECK 1: Emergency - Exit immediately
        # ==========================================
        if loss_pct >= 1.5:
            return True, f"Emergency exit - loss {loss_pct:.2f}%"

        # ==========================================
        # CHECK 2: SuperTrend on 5min, 15min, 30min
        # ==========================================
        try:
            # Check 3 timeframes
            bullish_count = 0
            bearish_count = 0

            for tf in ["5", "15", "30"]:
                inds = self.bot.indicator_calculator.calculate_indicators(
                    self.symbol, tf, pivot_data={}
                )

                if "error" not in inds:
                    st_trend = inds.get("st_main_trend", 0)

                    if st_trend > 0:
                        bullish_count += 1
                    elif st_trend < 0:
                        bearish_count += 1

            # If 2 or 3 SuperTrends are in our favor, DON'T exit (unless loss is big)
            if current_pos == "BUY" and bullish_count >= 2 and loss_pct < 1.0:
                return False, f"SuperTrend protection: {bullish_count}/3 bullish"

            if current_pos == "SELL" and bearish_count >= 2 and loss_pct < 1.0:
                return False, f"SuperTrend protection: {bearish_count}/3 bearish"

        except Exception as e:
            self.log(f"[SL-CHECK] SuperTrend check error: {e}", True)

        # ==========================================
        # CHECK 3: Candle closing below/above previous
        # ==========================================
        try:
            # SAFE: never reference primary_tf here
            tf = getattr(self, "last_primary_tf", "5")
            ohlc = self.bot.fetch_ohlc(self.symbol, tf, 10)

            if ohlc is not None and len(ohlc) >= 2:
                latest = ohlc.iloc[-1]
                prev = ohlc.iloc[-2]

                # For BUY: Check if closed below previous candle low
                if current_pos == "BUY":
                    if latest['Close'] > prev['Low']:  # NOT below prev low
                        return False, "Candle not closed below prev low - holding"

                # For SELL: Check if closed above previous candle high
                elif current_pos == "SELL":
                    if latest['Close'] < prev['High']:  # NOT above prev high
                        return False, "Candle not closed above prev high - holding"

        except Exception as e:
            self.log(f"[SL-CHECK] Candle check error: {e}", True)

        # ==========================================
        # Default: Allow exit
        # ==========================================
        return True, "Stop loss conditions met"

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

                # ✅ SuperTrend confirms direction
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
                    self.log(f"⚠️ [EARLY-ENTRY] No SuperTrend data - using only EMAs", True)

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

            # ✅ Bonus if SuperTrend strongly confirms
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
                f"✅ [EARLY-ENTRY] {signal} signal detected!\n"
                f"  📈 Move from open: {abs(move_from_open):.2f}%\n"
                f"  📊 Volume: {volume_ratio:.2f}x average\n"
                f"  💪 ADX: {adx:.1f}\n"
                f"  📐 EMA: 5={ema5:.2f}, 9={ema9:.2f}\n"
                f"  🎯 SuperTrend: {st_status}\n"
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
        t = self._get_tuning(primary_tf)


        # Resolve primary timeframe dynamically (avoid hard-coded 30m defaults)
        # Priority: argument > last known > bot selected TF > global tf_selected > fallback 15m
        if primary_tf is None:
            primary_tf = (
                getattr(self, "last_known_primary_tf", None)
                or getattr(self, "last_primary_tf", None)
                or getattr(getattr(self, "bot", None), "tf_selected", None)
                or globals().get("tf_selected", None)
                or "15"
            )
        primary_tf = str(primary_tf).strip()

        # Store primary TF once (authoritative source)
        self.last_primary_tf = str(primary_tf)

        # Get pivot data early (needed for both exit and entry logic)
        pivot_json_path = self.state_path.replace("om_state", "pivot")
        pivot_data = robust_load_json(pivot_json_path, self.log, default={})
        pivot_levels = pivot_data.get(self.symbol, {})

        # ==========================================
        # 🛑 PRIORITY #1: ENHANCED STOP LOSS CHECK
        # ==========================================
        current_pos = self.position.get("type", "FLAT")
        if current_pos != "FLAT":
            current_sl = self._f(self.position.get("stop_loss"))
            entry_price = self._f(self.position.get("entry_price"))
            r_mult = self._f(self.position.get("r_mult", 0))

            # 🔥 NEW: Get current profit tier for context
            if current_pos == "BUY":
                current_profit_points = ltp - entry_price if (ltp and entry_price) else 0
            else:
                current_profit_points = entry_price - ltp if (ltp and entry_price) else 0

            current_profit_rupees = current_profit_points * 250

            if current_sl and ltp and entry_price:
                sl_triggered = False
                exit_reason = ""

                # 🎯 NATURAL GAS: Add bid-ask spread buffer (0.10 points = ₹25)
                BID_ASK_BUFFER = 0.10  # Natural Gas typical spread

                # ==========================================
                # Check 1: Regular Stop Loss (with buffer)
                # ==========================================
                if current_pos == "BUY":
                    # Add buffer to prevent premature stops on spread
                    sl_with_buffer = current_sl - BID_ASK_BUFFER

                    if ltp <= sl_with_buffer:
                        sl_triggered = True
                        loss_amount = entry_price - ltp
                        loss_rupees = loss_amount * 250
                        loss_pct = (loss_amount / entry_price) * 100
                        exit_reason = (
                            f"STOP LOSS HIT | "
                            f"Entry:₹{entry_price:.2f}, SL:₹{current_sl:.2f}, Current:₹{ltp:.2f} | "
                            f"Loss: ₹{loss_rupees:.0f} ({loss_pct:.2f}%)"
                        )

                elif current_pos == "SELL":
                    sl_with_buffer = current_sl + BID_ASK_BUFFER

                    if ltp >= sl_with_buffer:
                        sl_triggered = True
                        loss_amount = ltp - entry_price
                        loss_rupees = loss_amount * 250
                        loss_pct = (loss_amount / entry_price) * 100
                        exit_reason = (
                            f"STOP LOSS HIT | "
                            f"Entry:₹{entry_price:.2f}, SL:₹{current_sl:.2f}, Current:₹{ltp:.2f} | "
                            f"Loss: ₹{loss_rupees:.0f} ({loss_pct:.2f}%)"
                        )

                # ==========================================
                # Check 2: Tiered Emergency Stop Loss
                # ==========================================
                if not sl_triggered and r_mult > 0:
                    if current_pos == "BUY":
                        current_loss = entry_price - ltp
                    else:
                        current_loss = ltp - entry_price

                    current_loss_rupees = current_loss * 250

                    # 🎯 TIERED EMERGENCY SL (based on profit history)
                    max_profit_achieved = self.position.get("_max_profit", 0)
                    max_profit_rupees = max_profit_achieved * 250

                    # Tier 1: Never made profit - strict emergency SL
                    if max_profit_rupees < 250:  # Never went into profit
                        emergency_multiplier = 1.2  # Allow only 1.2x initial risk
                        max_loss_rupees = r_mult * 250 * emergency_multiplier

                    # Tier 2: Made small profit but now losing - medium SL
                    elif max_profit_rupees < 1000:
                        emergency_multiplier = 1.4
                        max_loss_rupees = r_mult * 250 * emergency_multiplier

                    # Tier 3: Made good profit but now losing - wide SL
                    else:
                        emergency_multiplier = 1.6
                        max_loss_rupees = r_mult * 250 * emergency_multiplier

                    if current_loss_rupees > max_loss_rupees:
                        sl_triggered = True
                        exit_reason = (
                            f"EMERGENCY STOP | Fast adverse move detected | "
                            f"Loss: ₹{current_loss_rupees:.0f} exceeds {emergency_multiplier}x risk (₹{max_loss_rupees:.0f}) | "
                            f"Max profit was: ₹{max_profit_rupees:.0f}"
                        )

                # ==========================================
                # Check 3: Profit-Based Maximum Loss
                # ==========================================
                if not sl_triggered:
                    if current_pos == "BUY":
                        loss_amount = entry_price - ltp
                    else:
                        loss_amount = ltp - entry_price

                    loss_rupees = loss_amount * 250
                    loss_pct = (loss_amount / entry_price) * 100 if entry_price > 0 else 0

                    # 🎯 ADAPTIVE: Max loss based on profit tier
                    max_profit_achieved = self.position.get("_max_profit", 0)
                    max_profit_rupees = max_profit_achieved * 250

                    # If we made significant profit before, allow bigger loss
                    if max_profit_rupees >= 2000:  # Made ₹2,000+ before
                        max_loss_pct = 1.8  # Allow 1.8% loss
                    elif max_profit_rupees >= 1000:  # Made ₹1,000+ before
                        max_loss_pct = 1.5  # Allow 1.5% loss
                    else:  # Never made much profit
                        max_loss_pct = 1.2  # Strict 1.2% loss limit

                    if loss_pct > max_loss_pct:
                        sl_triggered = True
                        exit_reason = (
                            f"MAX LOSS PERCENT | "
                            f"Loss: ₹{loss_rupees:.0f} ({loss_pct:.2f}%) > {max_loss_pct:.1f}% threshold | "
                            f"Max profit was: ₹{max_profit_rupees:.0f}"
                        )

                # ==========================================
                # Check 4: 🔥 NEW - Rapid Loss Detection
                # ==========================================
                if not sl_triggered:
                    # Check if losing ₹500+ in last 2 seconds
                    last_ltp = self.position.get("_last_checked_ltp")
                    last_check_time = self.position.get("_last_check_time")
                    current_time = dt.datetime.now(self.IST)

                    if last_ltp and last_check_time:
                        # Convert string to datetime if needed
                        if isinstance(last_check_time, str):
                            try:
                                last_check_time = pd.to_datetime(last_check_time).tz_localize(self.IST)
                            except Exception:
                                last_check_time = current_time  # Fallback

                        time_diff = (current_time - last_check_time).total_seconds()

                        if time_diff <= 2:  # Within 2 seconds
                            if current_pos == "BUY":
                                rapid_loss = (last_ltp - ltp) * 250
                            else:
                                rapid_loss = (ltp - last_ltp) * 250

                            # Exit if losing ₹500+ in 2 seconds (flash crash protection)
                            if rapid_loss >= 500:
                                sl_triggered = True
                                exit_reason = (
                                    f"FLASH CRASH PROTECTION | "
                                    f"Lost ₹{rapid_loss:.0f} in {time_diff:.1f}s | "
                                    f"Price: ₹{last_ltp:.2f} → ₹{ltp:.2f}"
                                )

                    # Update tracking
                    self.position["_last_checked_ltp"] = ltp
                    self.position["_last_check_time"] = current_time

                # ==========================================
                # Execute Stop Loss Exit
                # ==========================================
                if sl_triggered:
                    should_exit, check_reason = self._should_exit_on_stop_loss(ltp, current_pos)

                    if should_exit:
                        # Calculate final P&L
                        if current_pos == "BUY":
                            final_loss = (entry_price - ltp) * 250
                        else:
                            final_loss = (ltp - entry_price) * 250

                        max_profit_achieved = self.position.get("_max_profit", 0) * 250

                        self.log(
                            f"\n🛑🛑🛑 STOP LOSS TRIGGERED 🛑🛑🛑\n"
                            f"{'=' * 60}\n"
                            f"Position: {current_pos}\n"
                            f"Entry: ₹{entry_price:.2f}\n"
                            f"Stop Loss: ₹{current_sl:.2f}\n"
                            f"Current Price: ₹{ltp:.2f}\n"
                            f"{'=' * 60}\n"
                            f"LOSS DETAILS:\n"
                            f"  Final Loss: ₹{final_loss:.0f}\n"
                            f"  Max Profit Achieved: ₹{max_profit_achieved:.0f}\n"
                            f"  Total Drawdown: ₹{final_loss + max_profit_achieved:.0f}\n"
                            f"{'=' * 60}\n"
                            f"REASON: {exit_reason}\n"
                            f"Check Result: {check_reason}\n"
                            f"{'=' * 60}\n"
                            f"⚠️ Exiting immediately...\n",
                            False
                        )

                        self._process_exit(exit_reason, ltp)
                        return

                    else:
                        # Stop loss hit but NOT exiting
                        self.log(
                            f"\n🛡️🛡️🛡️ STOP LOSS PROTECTED 🛡️🛡️🛡️\n"
                            f"{'=' * 60}\n"
                            f"Position: {current_pos}\n"
                            f"Entry: ₹{entry_price:.2f}\n"
                            f"Stop Loss: ₹{current_sl:.2f}\n"
                            f"Current Price: ₹{ltp:.2f}\n"
                            f"{'=' * 60}\n"
                            f"Protection Reason: {check_reason}\n"
                            f"{'=' * 60}\n"
                            f"✅ Continuing to hold position...\n",
                            False
                        )

        # ==========================================
        # REJECTION CANDLE EXIT CHECK
        # ==========================================
        if current_pos != "FLAT":
            ohlc_for_exit = None
            try:
                ohlc_for_exit = self.bot.fetch_ohlc(self.symbol, str(getattr(self, "last_primary_tf", "15")), 1)
            except Exception as e:
                self.log(f"[EXIT-CHECK] Failed to fetch OHLC: {e}", True)

            if ohlc_for_exit is not None and not ohlc_for_exit.empty and len(ohlc_for_exit) >= 3:
                rejection_signal, rejection_conf = self.detect_rejection_candle(
                    ohlc_for_exit,
                    ltp,
                    pivot_levels
                )

                if rejection_signal and rejection_conf >= 0.70:
                    if current_pos == "BUY" and rejection_signal == "SELL_REJECTION":
                        entry_price = self._f(self.position.get("entry_price"))
                        profit = ltp - entry_price if entry_price else 0

                        self.log(
                            f"\n🔴 [REJECTION-EXIT] Exiting LONG position\n"
                            f"  Reason: Bearish rejection at resistance (conf: {rejection_conf:.2f})\n"
                            f"  Entry: ₹{entry_price:.2f}, Current: ₹{ltp:.2f}\n"
                            f"  P&L: ₹{profit:.2f}",
                            False
                        )

                        if self._process_exit(f"Rejection at resistance (conf: {rejection_conf:.2f})", ltp):
                            # Zero cooldown if strong momentum, else 1 bar
                            cooldown_bars = 0 if (
                                        abs(self._f(all_inds.get("momentum_pct"), 0)) > 0.5 and self._f(all_inds.get("adx"),
                                                                                                    0) > 25) else 1
                            self._set_cooldown(cooldown_bars)
                            #self._set_cooldown(1)
                            self._save_state()
                            return

                    elif current_pos == "SELL" and rejection_signal == "BUY_REJECTION":
                        entry_price = self._f(self.position.get("entry_price"))
                        profit = entry_price - ltp if entry_price else 0

                        self.log(
                            f"\n🟢 [REJECTION-EXIT] Exiting SHORT position\n"
                            f"  Reason: Bullish rejection at support (conf: {rejection_conf:.2f})\n"
                            f"  Entry: ₹{entry_price:.2f}, Current: ₹{ltp:.2f}\n"
                            f"  P&L: ₹{profit:.2f}",
                            False
                        )

                        if self._process_exit(f"Rejection at support (conf: {rejection_conf:.2f})", ltp):
                            # Zero cooldown if strong momentum, else 1 bar
                            cooldown_bars = 0 if (
                                        abs(self._f(all_inds.get("momentum_pct"), 0)) > 0.5 and self._f(all_inds.get("adx"),
                                                                                                    0) > 25) else 1
                            self._set_cooldown(cooldown_bars)
                            #self._set_cooldown(1)
                            self._save_state()
                            return

        # ==========================================
        # VALIDATE INPUTS
        # ==========================================
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

        # Post-impulse cooldown (prevents whipsaw entries immediately after large candles)
        if current_pos == "FLAT":
            self._apply_impulse_cooldown(inds, str(primary_tf))

        # ✅ FIX: Safety Gating (Cooldown & ADX Chop Filter)
        if current_pos == "FLAT":
            now = time.time()
            last_exit = self.position.get("last_exit_ts", 0)
            cooldown_sec = 180  # 3 minutes
            
            # 1. Cooldown Gate
            if now - last_exit < cooldown_sec:
                self.log(f"⛔ [COOLDOWN] Skipping entry - {int(cooldown_sec - (now - last_exit))}s remaining", True)
                return
            
            # 2. ADX Gate (Layer-1: Market Regime)
            adx_val = self._f(inds.get("adx"), 0)
            if adx_val < 20:
                self.log(f"⛔ [ADX-GATE] Entry blocked - Weak trend (ADX {adx_val:.1f} < 20)", True)
                return

        if not isinstance(inds, dict) or not inds.get("timestamp"):
            # 🔧 FIX: Block all trading when indicators are stale
            self.log(
                f"⛔ [UNIFIED] No valid indicators for TF={primary_tf} - BLOCKING all actions\n"
                f"  Cannot trade or manage positions with stale/missing data\n"
                f"  Position: {current_pos}\n"
                f"  Reason: Safety measure to prevent wrong signals from outdated indicators",
                False
            )
            return  # Don't proceed - wait for fresh data

        bar_key = f"{primary_tf}:{inds['timestamp']}"

        # ==========================================
        # 🔧 FIX #7: SINGLE SOURCE OF TRUTH FOR BAR DEDUPLICATION
        # ==========================================
        
        # Rule 1: Already evaluated this bar? Skip new signals but manage existing position.
        if self.position.get("_last_evaluated_bar") == bar_key:
            self.log(f"[DEDUP] Already evaluated bar {bar_key} - skipping new entries", True)
            if current_pos != "FLAT":
                atr_here = self._get_atr_with_fallback(inds, ltp)
                if atr_here:
                    self._update_trailing_and_breakeven(ltp, atr_here)
                    cpr_analysis = inds.get("cpr_analysis", {})
                    pivot_data_guard = cpr_analysis.get("cpr_levels", {}) if cpr_analysis else {}
                    if pivot_data_guard and "TC" in pivot_data_guard:
                        self._update_dynamic_cpr_stop_loss(ltp, atr_here, pivot_data_guard)
                self._check_trailing_profit(ltp, inds)
            return

        # Rule 2: Just exited this bar? Don't re-enter immediately (prevent whipping)
        if current_pos == "FLAT":
            last_action_bar = self.position.get("_last_action_bar")
            if last_action_bar == bar_key:
                self.log(f"[REENTRY-GUARD] Just exited bar {bar_key} - must wait for next bar", False)
                return

        # Mark this bar as evaluated to lock it for new entries
        self.position["_last_evaluated_bar"] = bar_key
        
        # Rule 3: Minimum hold period for existing positions
        if current_pos != "FLAT":
            entry_bar = self.position.get("_entry_bar_key")
            if entry_bar and entry_bar == bar_key:
                self.log(f"[MIN-HOLD] Still in entry bar {bar_key} - cannot exit yet", True)
                # Update management but block exits
                atr_here = self._get_atr_with_fallback(inds, ltp)
                if atr_here:
                    self._update_trailing_and_breakeven(ltp, atr_here)
                    cpr_analysis = inds.get("cpr_analysis", {})
                    pivot_data_hold = cpr_analysis.get("cpr_levels", {}) if cpr_analysis else {}
                    if pivot_data_hold and "TC" in pivot_data_hold:
                        self._update_dynamic_cpr_stop_loss(ltp, atr_here, pivot_data_hold)
                self._check_trailing_profit(ltp, inds)
                self._save_state()
                return

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # BEYOND THIS POINT: New signals can be evaluated
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # ==========================================
        # 🔥 FIX #2: DETECT TREND using primary_tf, not hardcoded "5"
        # ==========================================
        # 🔧 FIX: Safe trend detection with fallback
        detected_trend = "CONSOLIDATION"  # Default to safe state
        try:
            df_tf = None
            if hasattr(self, 'bot'):
                try:
                    df_tf = self.bot.fetch_ohlc(self.symbol, str(primary_tf), 30)
                    self.log(f"[TREND-DETECT] Using {primary_tf}m timeframe for trend", True)
                except Exception as e:
                    self.log(f"[TREND-DETECT] Failed to fetch {primary_tf}m OHLC: {e}", True)

            if df_tf is not None and isinstance(df_tf, pd.DataFrame) and not df_tf.empty:
                df_tf = df_tf.copy()
                df_tf.columns = [c.lower() for c in df_tf.columns]
                if "close" in df_tf.columns and len(df_tf) > 10:
                    i = len(df_tf) - 1
                    try:
                        detected_trend = self.detect_trend(df=df_tf, i=i)
                        self.log(f"[TREND-DETECT] ✅ Result on {primary_tf}m: {detected_trend}", True)
                    except Exception as e:
                        self.log(f"[TREND-DETECT] Detection error: {e}, using CONSOLIDATION", True)
                        detected_trend = "CONSOLIDATION"
            else:
                self.log(f"[TREND-DETECT] ⚠️ No OHLC data - defaulting to CONSOLIDATION", True)
        except Exception as e:
            self.log(f"[TREND-DETECT] ❌ Critical error: {e} - blocking entries (CONSOLIDATION)", False)
            detected_trend = "CONSOLIDATION"

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
                    # ✅ FIX: Increased from 2 to 80 candles for sufficient AI context
                    ohlc_for_ai = self.bot.fetch_ohlc(self.symbol, str(primary_tf), 80)
                except Exception as e:
                    self.log(f"[AI-OHLC] Failed to fetch: {e}", True)
                    ohlc_for_ai = None

                # Then pass it to predictor
                self.log(f"[DEBUG] Calling AI Predictor with OHLC size={len(ohlc_for_ai) if ohlc_for_ai is not None else 0}", True)
                ai_label, ai_conf, ai_dist, feature_array = self.ai_predictor.predict(
                    indicators=inds,
                    pivot_data=pivot_data,
                    feature_builder=_build_ai_cpr_features,
                    ohlc_df=ohlc_for_ai  # 🔥 ADD THIS
                )
                self.log(f"[DEBUG] AI Result: Label={ai_label}, Conf={ai_conf}", True)
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
                    f"🤖 AI NEXT CANDLE PREDICTION\n"
                    f"{'=' * 60}",
                    False
                )
                self.log(
                    f"Label: {ai_label}\n"
                    f"Confidence: {ai_confidence_raw:.4f} ({ai_confidence_raw * 100:.2f}%)\n"
                    f"Threshold (Min): {self.AI_MIN_CONF:.4f}\n"
                    f"Threshold (Override): 0.75\n"
                    f"Status: {'✅ ACCEPTED' if ai_confidence_raw >= self.AI_MIN_CONF else '❌ REJECTED'}",
                    False
                )
                if ai_distribution:
                    self.log(f"Distribution: {ai_distribution}", True)
                self.log(f"{'=' * 60}", False)

                # ------------------------------
                # NEXT CANDLE DIRECTION (LIVE)
                # ------------------------------
                try:
                    next_candle = self._predict_next_candle_signal(
                        ai_label=ai_label,
                        ai_conf=ai_confidence_raw,
                        ai_dist=ai_distribution,
                        inds=inds,
                        pivot_data=pivot_data,
                        ohlc_df=ohlc_for_ai,
                        current_pos=current_pos,
                        ltp=ltp,
                    )
                    self.position["_next_candle_prediction"] = {
                        **next_candle,
                        "timestamp": dt.datetime.now(self.IST).isoformat()
                    }
                    self.log(
                        f"🕯️ [NEXT-CANDLE] {next_candle['direction']} ({next_candle['candle_color']}) | "
                        f"Action: {next_candle['action']} | "
                        f"Conf: {next_candle['confidence']:.2f} | "
                        f"Quality: {next_candle.get('quality')} | "
                        f"Chop={next_candle.get('in_chop')} CPR={next_candle.get('inside_cpr')} | "
                        f"Reason: {next_candle.get('reason')}",
                        False
                    )
                except Exception as e:
                    self.log(f"[NEXT-CANDLE] Prediction error: {e}", True)


                AI_OVERRIDE_MIN_CONFIDENCE = 0.85 # Increased from 0.75 for better reliability
                
                if ai_label and ai_confidence_raw >= AI_OVERRIDE_MIN_CONFIDENCE:
                    ai_label_upper = str(ai_label).upper()
                    
                    # 🔧 AI override requires volume, momentum, trend, and regime support
                    volume_ratio = self._f(inds.get("volume_ratio"), 1.0)
                    momentum_pct = self._f(inds.get("momentum_pct"), 0.0)
                    adx = self._f(inds.get("adx"), self._f(inds.get("ADX"), 0.0))
                    in_chop = self._is_chop_regime(inds)
                    
                    # CPR zone check
                    bc = self._f(pivot_data.get("BC"), None)
                    tc = self._f(pivot_data.get("TC"), None)
                    inside_cpr = (bc is not None and tc is not None and min(bc, tc) <= ltp <= max(bc, tc))

                    confirmation_ok = (
                        volume_ratio > 1.3 
                        and abs(momentum_pct) > 0.3 
                        and adx >= 20.0 
                        and (not in_chop) 
                        and (not inside_cpr)
                    )

                    if any(keyword in ai_label_upper for keyword in ["BUY", "BULLISH", "LONG", "UP"]) and confirmation_ok:
                        ai_override_signal = "BUY"
                        ai_override_allowed = True
                        self.log(
                            f"\n🚀 [AI-OVERRIDE] HIGH CONFIDENCE BUY DETECTED!\n"
                            f"   Confidence: {ai_confidence_raw:.2f} (≥ {AI_OVERRIDE_MIN_CONFIDENCE})\n"
                            f"   Confirmation: Vol={volume_ratio:.2f}x, Mom={momentum_pct:.2f}%\n"
                            f"   Can bypass consolidation if needed\n",
                            False
                        )

                    elif any(keyword in ai_label_upper for keyword in ["SELL", "BEARISH", "SHORT", "DOWN"]) and confirmation_ok:
                        ai_override_signal = "SELL"
                        ai_override_allowed = True
                        self.log(
                            f"\n🚀 [AI-OVERRIDE] HIGH CONFIDENCE SELL DETECTED!\n"
                            f"   Confidence: {ai_confidence_raw:.2f} (≥ {AI_OVERRIDE_MIN_CONFIDENCE})\n"
                            f"   Confirmation: Vol={volume_ratio:.2f}x, Mom={momentum_pct:.2f}%\n"
                            f"   Can bypass consolidation if needed\n",
                            False
                        )
                    else:
                        self.log(
                            f"[AI-OVERRIDE] Rejecting override - missing confirmation (Vol={volume_ratio:.2f}x, Mom={momentum_pct:.2f}%)",
                            True
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
                    f"[AI-CPR] ⚠️ Pivot data incomplete - skipping prediction\n"
                    f"   TC: {pivot_data.get('TC')}, BC: {pivot_data.get('BC')}",
                    True
                )

        except Exception as e:
            self.log(f"[AI-CPR] ❌ Prediction error: {e}", False)
            import traceback
            self.log(f"[AI-CPR] Traceback:\n{traceback.format_exc()}", True)

        # ==========================================
        # CONSOLIDATION BLOCKER
        # ==========================================

        ohlc_for_breakout = None
        try:
            ohlc_for_breakout = self.bot.fetch_ohlc(self.symbol, str(primary_tf), 30)
        except Exception as e:
            self.log(f"[BREAKOUT] Failed to fetch OHLC: {e}", True)

        early_breakout_signal, early_breakout_conf = self._detect_early_breakout(
            ltp, inds, ohlc_for_breakout
        )
        # Trend-strength context (helps bypass consolidation and avoid premature exits)
        trend_strength_label, trend_strength_conf = self._detect_trend_strength(ohlc_for_breakout, inds)
        self.last_trend_strength = trend_strength_label
        self.last_trend_strength_conf = trend_strength_conf

        # ==========================================
        # 🔥 FIX 1: SMART CONSOLIDATION CHECK (with breakout detection)
        # ==========================================
        consolidation_blocks_entry = False

        if detected_trend and "consol" in str(detected_trend).lower():

            # ✅ Check for BREAKOUT signals
            volume_ratio = self._f(inds.get("volume_ratio"), 1.0)
            momentum_pct = self._f(inds.get("momentum_pct"), 0.0)

            # Get VWAP distance
            vwap = self._f(inds.get("VWAP"))
            vwap_distance = abs((ltp - vwap) / vwap * 100) if vwap else 0

            # ✅ BREAKOUT DETECTED = Allow entry
            is_breakout = (
                    early_breakout_signal is not None or  # Early breakout detector
                    (trend_strength_label is not None and trend_strength_conf >= 0.70) or  # Strong trend context
                    (volume_ratio >= 0.6 and abs(momentum_pct) >= 0.2) or  # Volume + Momentum
                    (vwap_distance >= 0.2) or  # Strong VWAP breakout
                    (ai_override_allowed and ai_override_signal)  # AI confirms direction
            )


            # 🔧 FIX #8: Actually block consolidation entries unless clear breakout is detected
            consolidation_blocks_entry = False
            
            # ✅ BREAKOUT DETECTED = Allow entry
            is_breakout = (
                    early_breakout_signal is not None or  # Early breakout detector
                    (trend_strength_label is not None and trend_strength_conf >= 0.70) or  # Strong trend context
                    (volume_ratio >= 1.5 and abs(momentum_pct) >= 0.4) or  # Stronger Volume + Momentum req
                    (vwap_distance >= 0.25) or  # Stronger VWAP breakout req
                    (ai_override_allowed and ai_override_signal)  # AI confirms direction with high confidence
            )

            if is_breakout:
                self.log(
                    f"✅ [BREAKOUT] Consolidation breakout detected - ALLOWING entry\n"
                    f"  Early signal: {early_breakout_signal} ({early_breakout_conf:.2f})\n"
                    f"  Volume: {volume_ratio:.2f}x | Momentum: {momentum_pct:.2f}%\n"
                    f"  VWAP distance: {vwap_distance:.2f}%\n"
                    f"  AI Override: {ai_override_allowed}\n"
                    f"  Decision: ALLOW ENTRY during consolidation",
                    False
                )
                consolidation_blocks_entry = False
            else:
                self.log(
                    f"⛔ [CONSOLIDATION] No clear breakout - BLOCKING entry\n"
                    f"  Volume: {volume_ratio:.2f}x (need 1.5x+)\n"
                    f"  Momentum: {momentum_pct:.2f}% (need 0.4%+)\n"
                    f"  VWAP distance: {vwap_distance:.2f}% (need 0.25%+)\n"
                    f"  Early breakout: {early_breakout_signal or 'None'}",
                    False
                )
                consolidation_blocks_entry = True

        # If consolidation blocks entry, manage existing position and return
        if consolidation_blocks_entry:
            if current_pos != "FLAT":
                self.log("[CONSOLIDATION] Managing existing position during consolidation", True)
                try:
                    atr_here = self._get_atr_with_fallback(inds, ltp)
                    if atr_here:
                        self._update_trailing_and_breakeven(ltp, atr_here)
                        cpr_analysis_inner = inds.get("cpr_analysis", {})
                        pivot_data_inner = cpr_analysis_inner.get("cpr_levels", {}) if cpr_analysis_inner else {}
                        if pivot_data_inner and "TC" in pivot_data_inner:
                            self._update_dynamic_cpr_stop_loss(ltp, atr_here, pivot_data_inner)
                    self._check_trailing_profit(ltp, inds)
                except Exception as e:
                    self.log(f"[CONSOLIDATION] Error managing position: {e}", False)
            return

        # ==========================================
        # 🔥 FIX #3: EARLY SIGNALS using correct timeframe OHLC
        # ==========================================
        early_entry_signal = None
        early_entry_confidence = 0.0

        ohlc_for_early = None
        try:
            # 🔥 Use primary_tf for early entry detection, not hardcoded "5"
            ohlc_for_early = self.bot.fetch_ohlc(self.symbol, str(primary_tf), 2)
            self.log(f"[EARLY] Using {primary_tf}m OHLC for early entry detection", True)
        except Exception as e:
            self.log(f"[EARLY] Failed to fetch {primary_tf}m OHLC: {e}", True)

        if ohlc_for_early is not None and not ohlc_for_early.empty:
            # 🔥 Get candle open/high/low from correct timeframe
            # Use the OHLC frame we just fetched (authoritative) instead of relying on indicator dict fields,
            # which may not carry open/high/low for the current bar yet.
            try:
                last_row = ohlc_for_early.iloc[-1]
                candle_open = self._f(last_row.get("Open") or last_row.get("open"))
                candle_high = self._f(last_row.get("High") or last_row.get("high"))
                candle_low = self._f(last_row.get("Low") or last_row.get("low"))
            except Exception:
                candle_open = candle_high = candle_low = None
            
            self.log(
                f"[EARLY] Using {primary_tf}m candle OHLC: O:{candle_open} H:{candle_high} L:{candle_low}",
                True
            )
            
            volume_signal, volume_conf = self.detect_volume_breakout(ohlc_for_early, ltp)

            if volume_signal and volume_conf >= 0.75:
                early_entry_signal = "BUY" if volume_signal == "VOLUME_BREAKOUT_BUY" else "SELL"
                early_entry_confidence = volume_conf

                self.log(
                    f"🚀 [EARLY-ENTRY] Volume breakout {early_entry_signal}!\n"
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
                        f"🎯 [EARLY-ENTRY] Rejection {early_entry_signal}!\n"
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
                        indsP=inds,
                        ai_conf=early_entry_confidence
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

        # Fetch OHLC data for patterns using primary_tf instead of hardcoded 15m
        ohlc_for_patterns = None
        try:
            ohlc_for_patterns = self.bot.fetch_ohlc(self.symbol, str(primary_tf), 5)
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
                self.log("[CHOPPY] Managing existing position in choppy market", True)
                atr_here = self._get_atr_with_fallback(inds, ltp)
                if atr_here:
                    self._update_trailing_and_breakeven(ltp, atr_here)
                    cpr_analysis = inds.get("cpr_analysis", {})
                    pivot_data = cpr_analysis.get("cpr_levels", {}) if cpr_analysis else {}
                    if pivot_data and "TC" in pivot_data:
                        self._update_dynamic_cpr_stop_loss(ltp, atr_here, pivot_data)
                self._check_trailing_profit(ltp, inds)
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
                f"🎯 [PRIORITY] Early breakout: {early_breakout_signal} ({early_breakout_conf:.2f})",
                False
            )

        # ✅ NEW: Direct momentum signal
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
                    f"[VOLUME-MOMENTUM] 🚀 EXTREME BUY - Volume: {volume_ratio:.2f}x, Momentum: {momentum_pct:.2f}%",
                    False
                )
            else:
                confidences["volume_momentum"] = 0.75
                self.log(
                    f"[VOLUME-MOMENTUM] ✅ Strong BUY - Volume: {volume_ratio:.2f}x, Momentum: {momentum_pct:.2f}%",
                    True
                )

        elif volume_surge and momentum_pct < -0.5:
            signals["volume_momentum"] = "SELL"
            if volume_extreme and momentum_pct < -0.8:
                confidences["volume_momentum"] = 0.85
                self.log(
                    f"[VOLUME-MOMENTUM] 🚀 EXTREME SELL - Volume: {volume_ratio:.2f}x, Momentum: {momentum_pct:.2f}%",
                    False
                )
            else:
                confidences["volume_momentum"] = 0.75
                self.log(
                    f"[VOLUME-MOMENTUM] ✅ Strong SELL - Volume: {volume_ratio:.2f}x, Momentum: {momentum_pct:.2f}%",
                    True
                )

        elif strong_bullish:
            signals["volume_momentum"] = "BUY"
            confidences["volume_momentum"] = 0.70
            self.log(f"[VOLUME-MOMENTUM] ✅ BUY signal (combined)", True)

        elif strong_bearish:
            signals["volume_momentum"] = "SELL"
            confidences["volume_momentum"] = 0.70
            self.log(f"[VOLUME-MOMENTUM] ✅ SELL signal (combined)", True)

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
                    f"[AI-CPR] Threshold: {self.AI_MIN_CONF:.4f} | Status: {'✅ ACCEPTED' if ai_confidence_raw >= self.AI_MIN_CONF else '❌ REJECTED'}",
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
                                    f"⚠️ [AI-COUNTER-TREND] AI suggests BUY but trend is {detected_trend}",
                                    False
                                )

                        elif any(keyword in ai_label_upper for keyword in ["SELL", "BEARISH", "SHORT", "DOWN"]):
                            if "up" in trend_str or "bull" in trend_str:
                                is_counter_trend = True
                                self.log(
                                    f"⚠️ [AI-COUNTER-TREND] AI suggests SELL but trend is {detected_trend}",
                                    False
                                )

                    min_confidence_required = 0.85 if is_counter_trend else self.AI_MIN_CONF

                    # ✅ NEW: Multi-Layer Stability & Chop Gating
                    ai_dir = self._ai_direction(ai_label)
                    
                    # 1. Track direction history (for stability)
                    self._ai_dir_hist.append(ai_dir)
                    
                    # 2. Stability check: require direction to appear at least AI_STABLE_COUNT times in last 3
                    stable = (ai_dir in ["BUY", "SELL"]) and (list(self._ai_dir_hist).count(ai_dir) >= self.AI_STABLE_COUNT)
                    
                    # 3. Cooldown check: if direction flipped recently, block it
                    now_ts = time.time()
                    if self._ai_last_dir and ai_dir in ["BUY", "SELL"] and ai_dir != self._ai_last_dir:
                        if (now_ts - self._ai_last_flip_ts) < self.AI_FLIP_COOLDOWN_SEC:
                            stable = False
                            self.log(f"⛔ [AI-COOLDOWN] Signal blocked due to recent flip ({int(self.AI_FLIP_COOLDOWN_SEC - (now_ts - self._ai_last_flip_ts))}s remaining)", True)
                        else:
                            self._ai_last_flip_ts = now_ts
                            self._ai_last_dir = ai_dir
                    elif self._ai_last_dir is None and ai_dir in ["BUY", "SELL"]:
                        self._ai_last_dir = ai_dir

                    # 4. Chop regime tightening
                    if self._is_chop_regime(inds):
                        # In chop, only allow very high confidence “strong” signals
                        if ai_dir in ["BUY", "SELL"] and ai_confidence_raw < 0.90:
                            stable = False
                            self.log(f"⛔ [AI-CHOP] Entry tightened - blocking medium signal ({ai_confidence_raw:.2f} < 0.90) in chop regime", True)

                    # Final gate for signal assignment
                    if stable:
                        if ai_dir == "BUY":
                            signals["ai_cpr"] = "BUY"
                            confidences["ai_cpr"] = ai_confidence_raw
                            if is_counter_trend:
                                self.log(f"[AI-CPR] ✅ Stable Counter-trend BUY accepted (conf: {ai_confidence_raw:.3f})", False)
                            else:
                                self.log(f"[AI-CPR] ✅ Stable BUY signal accepted (conf: {ai_confidence_raw:.3f})", True)
                        elif ai_dir == "SELL":
                            signals["ai_cpr"] = "SELL"
                            confidences["ai_cpr"] = ai_confidence_raw
                            if is_counter_trend:
                                self.log(f"[AI-CPR] ✅ Stable Counter-trend SELL accepted (conf: {ai_confidence_raw:.3f})", False)
                            else:
                                self.log(f"[AI-CPR] ✅ Stable SELL signal accepted (conf: {ai_confidence_raw:.3f})", True)
                    else:
                        if ai_dir in ["BUY", "SELL"]:
                            self.log(f"⚠️ [AI-UNSTABLE] Signal {ai_dir} rejected (needs count={self.AI_STABLE_COUNT} or passed cooldown/chop)", True)
                        else:
                            self.log(f"[AI-CPR] ⚠️ Neutral/Hold signal: {ai_label}", True)

                else:
                    if ai_label:
                        self.log(
                            f"[AI-CPR] ❌ Signal rejected: confidence {ai_confidence_raw:.3f} < {self.AI_MIN_CONF}",
                            True
                        )

        except Exception as e:
            self.log(f"[AI-CPR] ⚠️ Prediction error: {e}", False)
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
            ohlc_for_patterns = self.bot.fetch_ohlc(self.symbol, str(getattr(self, "last_primary_tf", "15")), 2)
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
                        f"🚀 [EARLY] BUY volume breakout (conf: {volume_conf:.2f})",
                        False
                    )
                elif volume_signal == "VOLUME_BREAKOUT_SELL":
                    signals["volume_breakout"] = "SELL"
                    confidences["volume_breakout"] = volume_conf
                    self.log(
                        f"🚀 [EARLY] SELL volume breakout (conf: {volume_conf:.2f})",
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
                            f"📊 PRICE ACTION SIGNAL DETECTED\n"
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
        # 🔧 FIX: SIGNAL CONFLICT DETECTION
        # ==========================================
        can_trade, final_signal_check, conflict_reason = self._check_signal_conflicts(signals, confidences)

        self.log(
            f"\n{'='*60}\n"
            f"📊 SIGNAL CONFLICT CHECK\n"
            f"{'='*60}\n"
            f"  Result: {'✅ CLEAR TO TRADE' if can_trade else '⛔ BLOCKED'}\n"
            f"  Final Signal: {final_signal_check or 'NONE'}\n"
            f"  Reason: {conflict_reason}\n"
            f"{'='*60}",
            False if not can_trade else True
        )

        if not can_trade and current_pos == "FLAT":
            self.log(
                f"[CONFLICT] ⛔ Blocking entry due to signal conflict\n"
                f"  BUY votes: {buy_count} (score: {buy_score:.2f})\n"
                f"  SELL votes: {sell_count} (score: {sell_score:.2f})\n"
                f"  Decision: Wait for clearer signal",
                False
            )
            # Still manage existing positions
            if current_pos != "FLAT":
                atr_here = self._get_atr_with_fallback(inds, ltp)
                if atr_here:
                    self._update_trailing_and_breakeven(ltp, atr_here)
                    self._check_trailing_profit(ltp, inds)
            return

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

        if pa_signal and pa_conf >= 0.75:  # High confidence threshold
            final_signal = pa_signal
            reason = f"PRICE ACTION (High Confidence {pa_conf:.2%})"

            self.log(
                f"\n{'=' * 60}\n"
                f"🎯 PRIORITY-0: PRICE ACTION SIGNAL SELECTED\n"
                f"{'=' * 60}\n"
                f"Signal: {final_signal}\n"
                f"Confidence: {pa_conf:.2%}\n"
                f"This is a KEY LEVEL trade\n"
                f"Overriding all other signals\n"
                f"{'=' * 60}",
                False
            )

        # PRIORITY 1: High Confidence Rejection
        elif rejection_in_sell and confidences["rejection_candle"] >= 0.7:
            if sell_count >= 2 or (sell_count == 1 and confidences["rejection_candle"] >= 0.8):
                final_signal = "SELL"
                reason = f"EARLY REJECTION at resistance (conf: {confidences['rejection_candle']:.2f})"

        elif rejection_in_buy and confidences["rejection_candle"] >= 0.7:
            if buy_count >= 2 or (buy_count == 1 and confidences["rejection_candle"] >= 0.8):
                final_signal = "BUY"
                reason = f"EARLY REJECTION at support (conf: {confidences['rejection_candle']:.2f})"

        # PRIORITY 2: Volume Breakout
        elif volume_breakout_in_buy and confidences["volume_breakout"] >= 0.7:
            if buy_count >= 2:
                final_signal = "BUY"
                reason = f"VOLUME BREAKOUT BUY (conf: {confidences['volume_breakout']:.2f}) + {buy_count - 1} confirmations"

        elif volume_breakout_in_sell and confidences["volume_breakout"] >= 0.7:
            if sell_count >= 2:
                final_signal = "SELL"
                reason = f"VOLUME BREAKOUT SELL (conf: {confidences['volume_breakout']:.2f}) + {sell_count - 1} confirmations"

        # PRIORITY 3: High Confidence AI
        elif ai_in_buy and confidences["ai_cpr"] >= 0.75:
            opposing_signals = sell_count
            supporting_signals = buy_count - 1

            if supporting_signals >= 1:
                final_signal = "BUY"
                reason = f"HIGH CONFIDENCE AI BUY ({confidences['ai_cpr']:.2f}) + {supporting_signals} support"
            elif opposing_signals <= 2:
                final_signal = "BUY"
                reason = f"HIGH CONFIDENCE AI BUY ({confidences['ai_cpr']:.2f}) solo (weak opposition: {opposing_signals})"
            else:
                self.log(
                    f"[AI-BLOCK] AI BUY confidence {confidences['ai_cpr']:.2f} but {opposing_signals} strong opposing signals - too risky",
                    False
                )

        elif ai_in_sell and confidences["ai_cpr"] >= 0.75:
            opposing_signals = buy_count
            supporting_signals = sell_count - 1

            if supporting_signals >= 1:
                final_signal = "SELL"
                reason = f"HIGH CONFIDENCE AI SELL ({confidences['ai_cpr']:.2f}) + {supporting_signals} support"
            elif opposing_signals <= 2:
                final_signal = "SELL"
                reason = f"HIGH CONFIDENCE AI SELL ({confidences['ai_cpr']:.2f}) solo (weak opposition: {opposing_signals})"
            else:
                self.log(
                    f"[AI-BLOCK] AI SELL confidence {confidences['ai_cpr']:.2f} but {opposing_signals} strong opposing signals - too risky",
                    False
                )

        # PRIORITY 4: Strong Confluence (3+ votes)
        elif buy_count >= 3:
            final_signal = "BUY"
            has_vwap_confirm = signals.get("vwap") == "BUY"
            has_bid_ask_confirm = signals.get("bid_ask_pressure") == "BUY"

            if has_vwap_confirm or has_bid_ask_confirm:
                final_signal = "BUY"
                ai_part = f" + AI({confidences['ai_cpr']:.2f})" if ai_in_buy else ""
                vwap_part = " + VWAP" if has_vwap_confirm else ""
                bid_ask_part = " + BidAsk" if has_bid_ask_confirm else ""
                reason = f"Strong BUY ({buy_count}/10 votes, score: {buy_score:.2f}){ai_part}{vwap_part}{bid_ask_part}"
            else:
                self.log(
                    f"[ENTRY-BLOCK] 3 BUY votes but missing VWAP/Bid-Ask confirmation - waiting",
                    False
                )
            ai_part = f" + AI({confidences['ai_cpr']:.2f})" if ai_in_buy else ""
            reason = f"Strong BUY confluence ({buy_count}/10 votes, score: {buy_score:.2f}){ai_part}"

        elif sell_count >= 3:
            final_signal = "SELL"
            has_vwap_confirm = signals.get("vwap") == "SELL"
            has_bid_ask_confirm = signals.get("bid_ask_pressure") == "SELL"

            if has_vwap_confirm or has_bid_ask_confirm:
                final_signal = "SELL"
                ai_part = f" + AI({confidences['ai_cpr']:.2f})" if ai_in_sell else ""
                vwap_part = " + VWAP" if has_vwap_confirm else ""
                bid_ask_part = " + BidAsk" if has_bid_ask_confirm else ""
                reason = f"Strong SELL ({sell_count}/10 votes, score: {sell_score:.2f}){ai_part}{vwap_part}{bid_ask_part}"
            else:
                self.log(
                    f"[ENTRY-BLOCK] 3 SELL votes but missing VWAP/Bid-Ask confirmation - waiting",
                    False
                )
            ai_part = f" + AI({confidences['ai_cpr']:.2f})" if ai_in_sell else ""
            reason = f"Strong SELL confluence ({sell_count}/10 votes, score: {sell_score:.2f}){ai_part}"

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

            # ✅ FIX: Handle None/missing trend gracefully
            if not detected_trend or detected_trend == "None" or str(detected_trend).lower() == "none":
                self.log(
                    f"⚠️ [TREND-WARNING] No trend detected, using EMA fallback",
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
                                f"✅ [EMA-FALLBACK] BUY allowed - EMA9({e9:.2f}) > EMA21({e21:.2f})",
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
                                f"✅ [EMA-FALLBACK] SELL allowed - EMA9({e9:.2f}) < EMA21({e21:.2f})",
                                False
                            )
                        else:
                            self.log(
                                f"⛔ [EMA-FALLBACK] SELL blocked - EMA9({e9:.2f}) > EMA21({e21:.2f})",
                                False
                            )
                            final_signal = None
                else:
                    # ✅ EMERGENCY FALLBACK: No EMAs available
                    vote_count = buy_count if final_signal == "BUY" else sell_count
                    vote_score = buy_score if final_signal == "BUY" else sell_score
                    ai_conf = confidences.get("ai_cpr", 0)

                    # Allow if very strong signal
                    if vote_count >= 4 and vote_score >= 2.5 and ai_conf >= 0.35:
                        self.log(
                            f"✅ [EMERGENCY-OVERRIDE] {final_signal} allowed without trend data\n"
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
                # ✅ NORMAL CASE: Trend detected
                trend_str = str(detected_trend).lower()
                trend_validated = False

                if final_signal == "BUY":
                    # Allow BUY in uptrend
                    if "up" in trend_str or "bull" in trend_str:
                        trend_validated = True
                        self.log(f"✅ [TREND] BUY allowed - {detected_trend}", True)

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
                                f"✅ [BREAKOUT] BUY allowed from consolidation\n"
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
                                f"✅ [COUNTER-TREND] BUY allowed against downtrend\n"
                                f"  Votes: {buy_count} | Score: {buy_score:.2f}\n"
                                f"  AI: {confidences['ai_cpr']:.2f} | Momentum: {momentum_pct:.2f}%\n"
                                f"  Volume: {volume_ratio:.2f}x\n"
                                f"  ⚠️ WARNING: Counter-trend trade - higher risk!",
                                False
                            )

                elif final_signal == "SELL":
                    if "down" in trend_str or "bear" in trend_str:
                        trend_validated = True
                        self.log(f"✅ [TREND] SELL allowed - {detected_trend}", True)

                    elif "consol" in trend_str:
                        breakout_valid = (
                                (volume_ratio >= 1.0 and momentum_score >= 1.2) or
                                (sell_count >= 4) or
                                (ai_in_sell and confidences["ai_cpr"] >= 0.75 and sell_count >= 2)
                        )

                        if breakout_valid:
                            trend_validated = True
                            self.log(
                                f"✅ [BREAKOUT] SELL allowed from consolidation\n"
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
                                f"✅ [COUNTER-TREND] SELL allowed against uptrend\n"
                                f"  Votes: {sell_count} | Score: {sell_score:.2f}\n"
                                f"  AI: {confidences['ai_cpr']:.2f} | Momentum: {momentum_pct:.2f}%\n"
                                f"  Volume: {volume_ratio:.2f}x\n"
                                f"  ⚠️ WARNING: Counter-trend trade - higher risk!",
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
                        f"✅ [MOMENTUM-SKIP] {skip_reason.upper()} bypasses momentum check",
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
            # LAYER 2.5: CHOP / SQUEEZE FILTER (avoid over-trading)
            # ==========================================
            if final_signal:
                try:
                    regime, regime_conf = self.detect_market_regime(ohlc_for_patterns, inds)
                except Exception:
                    regime, regime_conf = "UNKNOWN", 0.0

                # Allow trades in CHOP only if it's a strong breakout/rejection with high confidence
                sigs = signals if isinstance(signals, dict) else {}
                confs = confidences if isinstance(confidences, dict) else {}
                allow_breakout = (
                    float(confs.get("volume_breakout", 0.0)) >= float(t.get("breakout_conf", 0.78))
                ) or (
                    float(confs.get("rejection_candle", 0.0)) >= float(t.get("rejection_conf", 0.82))
                )

                if regime == "CHOPPY" and regime_conf >= 0.65 and not allow_breakout:
                    self.log(
                        f"[FILTER] 🧊 CHOP/SQUEEZE detected → skipping trade\n"
                        f"  Regime conf: {regime_conf:.2f} | Signal: {final_signal}\n"
                        f"  (Breakout/Rejection not strong enough)",
                        False
                    )
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
                            f"🚀 [ST-ADAPTIVE] SuperTrend check BYPASSED\n"
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

                            # ✅ CORRECT LOGIC FOR SELL
                            if ltp < supertrend:
                                self.log(
                                    f"✅ [SUPERTREND] SELL allowed - Price in bearish zone | "
                                    f"LTP:{ltp:.2f} < ST:{supertrend:.2f}",
                                    False
                                )

                            elif gap_pct < 0.3:
                                if sell_count >= 4:
                                    self.log(
                                        f"✅ [SUPERTREND] SELL allowed - Imminent breakdown + strong signals | "
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
                                    f"✅ [SUPERTREND] SELL allowed - Rejection override | "
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
                                    f"✅ [SUPERTREND] BUY allowed - Price in bullish zone | "
                                    f"LTP:{ltp:.2f} > ST:{supertrend:.2f}",
                                    False
                                )

                            elif gap_pct < 0.3:
                                if buy_count >= 4:
                                    self.log(
                                        f"✅ [SUPERTREND] BUY allowed - Imminent breakout + strong signals | "
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
                                    f"✅ [SUPERTREND] BUY allowed - Rejection override | "
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
                    self.log("⚠️ [SUPERTREND] No SuperTrend data - skipping check", True)

            # ==========================================
            # LAYER 4: RSI EXTREME CONDITIONS FILTER (FIXED)
            # ==========================================
            if final_signal:
                rsi = self._f(inds.get("adx_efi", {}).get("RSI"))

                if rsi is not None:
                    # ✅ CORRECT LOGIC: Block BUY when overbought, block SELL when oversold
                    if final_signal == "BUY" and rsi > 70:
                        self.log(
                            f"⛔ [RSI-FILTER] BUY BLOCKED - RSI too high | "
                            f"RSI:{rsi:.1f} > 70 (overbought, pullback risk)",
                            False
                        )
                        final_signal = None

                    elif final_signal == "SELL" and rsi < 30:
                        self.log(
                            f"⛔ [RSI-FILTER] SELL BLOCKED - RSI too low | "
                            f"RSI:{rsi:.1f} < 30 (oversold, bounce risk)",
                            False
                        )
                        final_signal = None
                else:
                    self.log("⚠️ [RSI-FILTER] No RSI data - skipping check", True)

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
                                f"⚠️ [ADX-FILTER] {final_signal} allowed despite weak ADX:{adx_val:.1f} "
                                f"(strong confluence: {vote_count} votes)",
                                False
                            )
                else:
                    self.log("⚠️ [ADX-FILTER] No ADX data - skipping check", True)

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
                        self.log(f"⚠️ [VOLATILITY] Low volatility but allowing morning breakout", True)
                else:
                    self.log("⚠️ [VOLATILITY] No BB data - skipping check", True)

            # ==========================================
            # ✅ ALL FILTERS PASSED - EXECUTE ENTRY
            # ==========================================
            if final_signal:
                self.log(
                    f"✅ [FILTER-SUMMARY] {final_signal} APPROVED - Passed all 6 filters | "
                    f"Trend✓ Momentum✓ SuperTrend✓ RSI✓ ADX✓ Volatility✓",
                    False
                )

                atr_here = self._get_atr_with_fallback(inds, ltp)
                if atr_here:
                    sources = [f"{s}({c:.2f})" for s, c in (buy_votes if final_signal == "BUY" else sell_votes)]
                    full_reason = f"{reason} | Sources: {', '.join(sources)}"

                    if self._process_entry(final_signal, full_reason, ltp, atr_here, bar_key=bar_key, indsP=inds, ai_conf=confidences.get("ai_cpr", 0.0)):
                        self.position["_last_action_bar"] = bar_key
                        self.position["_entry_bar_key"] = bar_key  # 🔥 FIX: Track entry bar for min hold
                        self.position["_bars_since_entry"] = 0  # Reset counter
                        self.position["ai_entry_confidence"] = confidences.get("ai_cpr", 0.0)
                        self.position["ai_distribution"] = ai_distribution
                        self._set_cooldown(self.FLIP_COOLDOWN_BARS)
                        self._save_state()
                        self.log(
                            f"[UNIFIED] ✅ Entry executed: {final_signal} | AI Confidence: {confidences.get('ai_cpr', 'N/A')}",
                            False
                        )
                        return

        # ==========================================
        # 🚀 FAST REVERSAL ENTRY
        # ==========================================
        if current_pos != "FLAT":
            # ✅ FIX: DICT-SAFE reversal history init
            if "_reversal_history" not in self.position or not isinstance(self.position["_reversal_history"], list):
                self.position["_reversal_history"] = []

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
                            # ✅ FIX: Robust trend alignment check
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
                            f"⚠️ [REVERSAL-VOLUME] Low volume ({volume_ratio:.2f}x) but strong score ({opposing_score:.2f})",
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
                            f"  ✅ Trend: {detected_trend}\n"
                            f"  ✅ Momentum: {momentum_pct:.2f}%\n"
                            f"  ✅ Volume: {volume_ratio:.2f}x\n"
                            f"  ✅ Opposing votes: {len(opposing_votes)} (score: {opposing_score:.2f})\n"
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
                            self.log(f"✅ [REVERSAL] Exit completed", False)

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
                                        f"📊 [REVERSAL-HISTORY] Total reversals in last 30min: {len(self.position['_reversal_history'])}",
                                        True
                                    )

                                    self._save_state()

                                    self.log(
                                        f"✅✅✅ [REVERSAL-COMPLETE] Now in {opposing_signal} position\n",
                                        False
                                    )
                                    return
                            else:
                                self.log(f"❌ [REVERSAL-FAILED] ATR unavailable for re-entry", False)
                        else:
                            self.log(f"❌ [REVERSAL-FAILED] Exit failed", False)

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
                            f"⚠️ [REVERSAL-SKIP] {opposing_signal} signal but reversal not validated\n"
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
                        f"⚠️ [AI-EXIT-BLOCKED] AI says SELL but trend NOT bearish\n"
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
                        f"⚠️ [AI-EXIT-BLOCKED] AI says BUY but trend NOT bullish\n"
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
            atr_here = self._get_atr_with_fallback(inds, ltp)
            if atr_here:
                self._update_trailing_and_breakeven(ltp, atr_here)
                self._update_ultra_tight_trailing(ltp, atr_here)
                cpr_analysis = inds.get("cpr_analysis", {})
                pivot_data = cpr_analysis.get("cpr_levels", {}) if cpr_analysis else {}
                if pivot_data and "TC" in pivot_data:
                    self._update_dynamic_cpr_stop_loss(ltp, atr_here, pivot_data)
            self._check_trailing_profit(ltp, inds)

        # ==========================================
        # EXIT PREDICTION (for open positions)
        # ==========================================
        if current_pos != "FLAT":
            try:
                # Get OHLC for exit prediction
                ohlc_for_exit = None
                try:
                    # ✅ FIX: Increased from 2 to 80 candles for sufficient AI context
                    ohlc_for_exit = self.bot.fetch_ohlc(self.symbol, str(primary_tf), 80)
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
                                f"🚨 [EXIT-PRED] Immediate exit signal\n"
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
                                    f"⚠️ [EXIT-PRED] Exit signal ignored - currently losing ₹{abs(profit_rupees):.0f}",
                                    False
                                )

                    # Tighten stops on exhaustion warning
                    elif exit_action == "EXIT_SOON" and exit_conf >= 0.60:
                        self.log(
                            f"⚠️ [EXIT-PRED] Exhaustion warning\n"
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
                                        f"🎯 [TIGHT-SL] Stop loss tightened (exhaustion warning)\n"
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
                    f"🚀 [VOLUME] Bullish breakout attempt!\n"
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
                    f"🚀 [VOLUME] Bearish breakout attempt!\n"
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
                    # ✅ Aligned or no conflict
                    result = "Uptrend"
                    self.log(
                        f"[TREND] ✅ UPTREND confirmed\n"
                        f"  EMA5: {last_5:.2f} > EMA9: {last_9:.2f} (gap: {ema_spread_59:.3f}%)\n"
                        f"  EMA5/21 filter: {long_filter} ({ema_spread_521:.3f}%)\n"
                        f"  Slope: {slope_pct:.3f}%",
                        False
                    )
                else:
                    # ⚠️ Counter-trend: EMA5>9 but EMA5<21 (short-term up, long-term down)
                    # Skip this - likely a pullback in downtrend
                    result = "Consolidation"
                    self.log(
                        f"[TREND] ⚠️ Conflicting signals - SKIP\n"
                        f"  Short: EMA5 > EMA9 (bullish)\n"
                        f"  Long: EMA5 < EMA21 (bearish)\n"
                        f"  Decision: Avoid counter-trend trade",
                        False
                    )

            # CASE 2: Short-term DOWNTREND
            elif short_signal == "Downtrend":
                if long_filter in ["Downtrend", "Neutral"]:
                    # ✅ Aligned or no conflict
                    result = "Downtrend"
                    self.log(
                        f"[TREND] ✅ DOWNTREND confirmed\n"
                        f"  EMA5: {last_5:.2f} < EMA9: {last_9:.2f} (gap: {ema_spread_59:.3f}%)\n"
                        f"  EMA5/21 filter: {long_filter} ({ema_spread_521:.3f}%)\n"
                        f"  Slope: {slope_pct:.3f}%",
                        False
                    )
                else:
                    # ⚠️ Counter-trend: EMA5<9 but EMA5>21 (short-term down, long-term up)
                    result = "Consolidation"
                    self.log(
                        f"[TREND] ⚠️ Conflicting signals - SKIP\n"
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
                                f"[TREND] ⚠️ Low volume warning: {vol_ratio:.2f}x average\n"
                                f"  Signal may be weak",
                                True
                            )
                        elif vol_ratio > 1.5:  # High volume
                            self.log(
                                f"[TREND] ✅ Volume confirmation: {vol_ratio:.2f}x average",
                                True
                            )
                except Exception as e:
                    self.log(f"[TREND] Volume check error: {e}", True)

            return result

        except Exception as e:
            self.log(f"[TREND] ❌ Critical error: {e}", False)
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


    def _apply_impulse_cooldown(self, inds: dict, primary_tf: str):
        t = self._get_tuning(primary_tf if 'primary_tf' in locals() else None)
        """
        Detects abnormally large impulse candles (often followed by chop/mean-revert)
        and applies a short cooldown to avoid low-quality entries.

        This is intentionally conservative and will NEVER interfere with exits.
        """
        try:
            ts = inds.get("timestamp")
            bar_key = f"{primary_tf}:{ts}"
            # Avoid re-applying cooldown multiple times for same bar
            if self.position.get("_last_impulse_bar_key") == bar_key:
                return False

            o = float(inds.get("open", 0) or 0)
            h = float(inds.get("high", 0) or 0)
            l = float(inds.get("low", 0) or 0)
            c = float(inds.get("close", 0) or 0)
            atr = float(inds.get("atr", 0) or 0)
            vol = float(inds.get("volume", 0) or 0)
            vol_ma = float(inds.get("vol_ma", 0) or 0)

            if c <= 0:
                return False

            body = abs(c - o)
            rng = max(0.0, h - l)
            # Fallback ATR proxy if ATR missing
            atr_eff = atr if atr and atr > 0 else max(rng, body, 0.0)

            # Impulse definition (tuned for intraday futures):
            # - body is large vs ATR
            # - candle is directional (body vs range)
            # - optional volume expansion
            body_vs_atr = body / atr_eff if atr_eff > 0 else 0.0
            body_vs_range = body / rng if rng > 0 else 0.0
            vol_spike = (vol / vol_ma) if (vol_ma and vol_ma > 0) else 1.0

            is_impulse = (body_vs_atr >= 1.15 and body_vs_range >= 0.55) or (body_vs_atr >= 1.40)
            if is_impulse:
                # Cooldown severity (bars) based on impulse strength
                bars = 2
                if body_vs_atr >= 1.40:
                    bars = 3
                if body_vs_atr >= 1.70:
                    bars = 4
                # If volume also spikes, we keep it slightly higher (exhaustion risk)
                if vol_spike >= 1.8:
                    bars = min(5, bars + 1)

                self.position["_last_impulse_bar_key"] = bar_key
                self._set_cooldown(bars)
                self.log(
                    f"[IMPULSE] Cooldown={bars} bars | body/ATR={body_vs_atr:.2f} body/range={body_vs_range:.2f} volSpike={vol_spike:.2f}",
                    True
                )
                return True
        except Exception as e:
            self.log(f"[IMPULSE] Detection error: {e}", True)
        return False

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

    def _ai_direction(self, ai_label: str) -> str:
        if not ai_label:
            return "NONE"
        u = str(ai_label).upper()
        if any(k in u for k in ["STRONG_BUY", "BUY", "BULLISH", "LONG", "UP"]):
            return "BUY"
        if any(k in u for k in ["STRONG_SELL", "SELL", "BEARISH", "SHORT", "DOWN"]):
            return "SELL"
        return "HOLD"
    def _ai_direction(self, ai_label: str) -> str:
        if not ai_label:
            return "NONE"
        u = str(ai_label).upper()
        if any(k in u for k in ["STRONG_BUY", "BUY", "BULLISH", "LONG", "UP"]):
            return "BUY"
        if any(k in u for k in ["STRONG_SELL", "SELL", "BEARISH", "SHORT", "DOWN"]):
            return "SELL"
        return "HOLD"
    def _predict_next_candle_signal(
        self,
        ai_label: str,
        ai_conf: float,
        ai_dist: Optional[dict],
        inds: dict,
        pivot_data: Optional[dict],
        ohlc_df: Optional[object],
        current_pos: str,
        ltp: float,
    ) -> dict:
        """Return a concise 'next candle' view for live logs/UI.

        Output keys:
          - direction: BULLISH / BEARISH / HOLD
          - candle_color: GREEN / RED / NEUTRAL
          - action: BUY / SELL / HOLD / EXIT
          - confidence: float (0-1)
          - reason: short text

        Note: this is a *probabilistic* hint. Execution must remain gated
        by your main strategy checks (regime, supertrend, CPR zone, etc.).
        """
        # Normalize inputs
        ai_label_u = str(ai_label or "HOLD").upper()
        conf = float(ai_conf or 0.0)

        # Prefer predictor's dedicated next-candle model if available
        next_pred = None
        try:
            if getattr(self, "ai_predictor", None) is not None and hasattr(self.ai_predictor, "predict_next_candle_direction"):
                next_pred = self.ai_predictor.predict_next_candle_direction(
                    indicators=inds or {},
                    pivot_data=pivot_data or {},
                    feature_builder=_build_ai_cpr_features,
                    ohlc_df=ohlc_df,
                )
        except Exception as _e:
            next_pred = None

        if isinstance(next_pred, dict) and next_pred.get("direction") in ("UP", "DOWN", "HOLD"):
            # Map UP/DOWN/HOLD -> BULLISH/BEARISH/HOLD used by the bot
            nc_dir = next_pred.get("direction")
            if nc_dir == "UP":
                direction = "BULLISH"
            elif nc_dir == "DOWN":
                direction = "BEARISH"
            else:
                direction = "HOLD"

            candle_color = next_pred.get("candle") or ("GREEN" if direction == "BULLISH" else "RED" if direction == "BEARISH" else "NEUTRAL")
            confidence = float(next_pred.get("confidence") or 0.0)
            margin = float(next_pred.get("margin") or 0.0)

            # Derive action suggestion (entry hint only; actual entries are still gated later)
            action = "WAIT"
            if current_pos == "FLAT" and confidence >= 0.55 and direction in ("BULLISH", "BEARISH"):
                action = "ENTRY_CANDIDATE"
            elif current_pos != "FLAT" and confidence >= 0.55 and direction == "HOLD":
                action = "HOLD_OR_EXIT_CHECK"

            return {
                "direction": direction,
                "candle_color": str(candle_color).upper(),
                "confidence": confidence,
                "margin": margin,
                "action": action,
                "quality": next_pred.get("source"),
                "reason": "predictor_next_candle",
                "in_chop": None,
                "inside_cpr": None,
                "p_up": next_pred.get("p_up"),
                "p_down": next_pred.get("p_down"),
            }


        # Base direction from label
        if any(k in ai_label_u for k in ("STRONG_BUY", "BUY", "BULL")):
            direction = "BULLISH"
        elif any(k in ai_label_u for k in ("STRONG_SELL", "SELL", "BEAR")):
            direction = "BEARISH"
        else:
            direction = "HOLD"

        # Optional margin (how decisive the distribution is)
        margin = 0.0
        top2 = None
        if isinstance(ai_dist, dict) and ai_dist:
            try:
                probs = sorted([float(v) for v in ai_dist.values()], reverse=True)
                if len(probs) >= 2:
                    top2 = (probs[0], probs[1])
                    margin = max(0.0, probs[0] - probs[1])
            except Exception:
                margin = 0.0

        # Regime gating: if chop OR inside CPR -> HOLD unless very strong
        in_chop = False
        inside_cpr = False
        try:
            in_chop = self._is_chop_regime(inds)
        except Exception:
            in_chop = False

        if isinstance(pivot_data, dict):
            bc = self._f(pivot_data.get("BC"), None)
            tc = self._f(pivot_data.get("TC"), None)
            if bc is not None and tc is not None and ltp is not None:
                inside_cpr = (min(bc, tc) <= float(ltp) <= max(bc, tc))

        # If very noisy -> force HOLD unless confidence extremely high
        if (in_chop or inside_cpr) and conf < 0.90:
            direction = "HOLD"

        candle_color = "GREEN" if direction == "BULLISH" else "RED" if direction == "BEARISH" else "NEUTRAL"

        # Action mapping relative to current position
        action = "HOLD"
        reason = ""

        if current_pos in ("BUY", "LONG") and direction == "BEARISH" and conf >= 0.75:
            action = "EXIT"
            reason = "AI suggests bearish next candle vs current LONG"
        elif current_pos in ("SELL", "SHORT") and direction == "BULLISH" and conf >= 0.75:
            action = "EXIT"
            reason = "AI suggests bullish next candle vs current SHORT"
        elif current_pos == "FLAT":
            if direction == "BULLISH" and conf >= self.AI_MIN_CONF:
                action = "BUY"
                reason = "AI bullish next candle"
            elif direction == "BEARISH" and conf >= self.AI_MIN_CONF:
                action = "SELL"
                reason = "AI bearish next candle"
            else:
                action = "HOLD"
                reason = "AI not strong enough"
        else:
            action = "HOLD"
            reason = "In-position: follow exit+trail logic"

        # Add a compact quality tag
        quality = "HIGH" if conf >= 0.85 and margin >= 0.15 else "MED" if conf >= 0.70 else "LOW"
        if not reason:
            reason = f"quality={quality}"

        return {
            "direction": direction,
            "candle_color": candle_color,
            "action": action,
            "confidence": round(conf, 4),
            "margin": round(margin, 4),
            "quality": quality,
            "in_chop": bool(in_chop),
            "inside_cpr": bool(inside_cpr),
            "reason": reason,
        }

    def _is_chop_regime(self, inds: dict) -> bool:
        """Return True when conditions are likely sideways/choppy.

        This is used as a *risk filter* to reduce over-trading during compression.
        The method is intentionally conservative: if key inputs are missing, it will
        fall back to a simple ADX + BBW check.

        Inputs expected in inds (when available):
          - adx/ADX
          - bb_bandwidth
          - atr/ATR
          - close/ltp
          - cpr_analysis.cpr_levels.cpr_width
          - vwap
        """
        t = self._get_tuning()
        adx = self._f(inds.get("adx"), self._f(inds.get("ADX"), 0.0))
        bbw = self._f(inds.get("bb_bandwidth"), 0.0)

        # Close / ATR (as % of price) – helps for NATGAS where BBW alone can be noisy
        close = self._f(inds.get("close"), self._f(inds.get("ltp"), 0.0))
        atr = self._f(inds.get("atr"), self._f(inds.get("ATR"), 0.0))
        atr_pct = (atr / close) if (close and atr) else 0.0

        # CPR width (if available)
        cpr_analysis = inds.get("cpr_analysis", {})
        cpr_levels = cpr_analysis.get("cpr_levels", {}) if isinstance(cpr_analysis, dict) else {}
        cprw = self._f(cpr_levels.get("cpr_width"), 0.0)

        # VWAP proximity (if very close, often mean-reverting)
        vwap = self._f(inds.get("vwap"), 0.0)
        vwap_dist_pct = abs(close - vwap) / close if (close and vwap) else 0.0

        # Core chop conditions (tuned for intraday 15m)
        # - weak trend (ADX)
        # - compression (BBW) and/or low ATR%
        # - narrow CPR zone
        # - price hugging VWAP
        weak_trend = adx > 0 and adx < float(t.get('chop_adx', 18.0)) + 0.5
        compression = (bbw > 0 and bbw < 0.009) or (atr_pct > 0 and atr_pct < 0.0032)
        narrow_cpr = (cprw > 0 and cprw < 0.0025)
        near_vwap = (vwap_dist_pct > 0 and vwap_dist_pct < 0.0015)

        # Fallback when only ADX/BBW present
        if (adx > 0) and (bbw > 0) and (atr_pct == 0.0) and (cprw == 0.0) and (vwap == 0.0):
            return (adx < float(t.get('chop_adx', 18.0))) and (bbw < float(t.get('bbw_squeeze', 0.015)))

        # Final decision
        score = 0
        score += 1 if weak_trend else 0
        score += 1 if compression else 0
        score += 1 if narrow_cpr else 0
        score += 1 if near_vwap else 0

        return score >= 3



class PerformanceTracker:
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

            self.bot.log_message(f"[DASHBOARD] ✅ Updated: {self.dashboard_path}", True)

        except Exception as e:
            self.bot.log_message(f"[DASHBOARD] ❌ Error: {e}", False)
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
                <h2>📍 Open Positions</h2>
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
            <h2>📍 Open Positions ({len(positions)})</h2>
            {rows}
        </div>
        """

    def _generate_trades_html(self, trades):
        """Generate trades table HTML"""
        if not trades:
            return """
            <div class="card">
                <h2>📊 Recent Trades</h2>
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
            <h2>📊 Recent Trades (Last 10)</h2>
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

    def __init__(self, config_dir="config", run_websocket=True):
        self.IST = IST
        self.DEBUG = True

        # Trading configuration
        #self.symbols = ["MCX:NATGASMINI26JANFUT"]  # Your symbol
        #self.symbols = ["MCX:NATGASMINI26FEBFUT"]
        self.symbols = ["MCX:GOLDPETAL26JANFUT"]
        # self.symbols = ["NSE:CHOLAFIN-EQ"]
        #self.symbols = ["NSE:MANAPPURAM-EQ"]
        # self.symbols = ["NSE:MAZDOCK-EQ"]
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
            model_path="ai_cpr_model_v3_advanced.pkl",
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
            fyers_service = FyersService(
                self.fyers_sdk_instance,
                self.data_paths[sym]['raw_api_log'],
                self.log_message,
                self.get_websocket_ltp
            )
            self.order_managers[sym] = OrderManager(
                fyers_service=fyers_service,
                symbol=sym,
                lot_size=1,
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
                    self.log_message(f"✅ Using valid CPR levels (age: {hours_old:.1f}h)", True)
                    return cpr_levels
                else:
                    self.log_message(f"⚠️ CPR levels stale ({hours_old:.1f}h old), recalculating", False)
            except Exception as e:
                self.log_message(f"⚠️ CPR age check failed: {e}", True)
        else:
            self.log_message(f"⚠️ CPR missing keys: {missing}", False)

        # Try 2: Recalculate pivots
        self.log_message("🔄 Recalculating CPR levels...", False)
        try:
            new_pivots = self.process_pivots()

            if new_pivots and symbol in new_pivots:
                cpr_levels = new_pivots[symbol]

                # Validate again
                missing = [k for k in required_keys if k not in cpr_levels or cpr_levels[k] is None]
                if not missing:
                    self.log_message("✅ CPR recalculation successful", False)
                    return cpr_levels
                else:
                    self.log_message(f"❌ Recalculated CPR still missing: {missing}", False)

        except Exception as e:
            self.log_message(f"❌ CPR recalculation failed: {e}", False)

        # Try 3: Use previous day's levels as emergency fallback
        self.log_message("⚠️ Using emergency fallback CPR (may be inaccurate)", False)

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
                f"⚠️ EMERGENCY CPR (approximate):\n"
                f"  TC={emergency_cpr['TC']}, BC={emergency_cpr['BC']}, PP={emergency_cpr['PP']}",
                False
            )

            return emergency_cpr

        except Exception as e:
            self.log_message(f"❌ Emergency fallback failed: {e}", False)
            return {}

    def run(self, selected_tf=None):
        """
        Unified main trading loop:
        Combines AI CPR, Trend-based, and Combined strategy execution.
        """
        if selected_tf is None:
            selected_tf = tf_selected  # From sys.argv

        self.log_message(f"🎯 Trading decisions based on {selected_tf}m timeframe", False)
        self.initialize_pivots()
        # ✅ Verify pivots loaded successfully
        pivot_data = robust_load_json(self.pivot_json, self.log_message, default={})
        if self.symbol not in pivot_data:
            self.log_message("❌ CRITICAL: Pivots not initialized! Cannot continue.", False)
            return

        symbol_pivots = pivot_data.get(self.symbol, {})
        required_keys = ["TC", "BC", "R1", "R2", "R3", "S1", "S2", "S3"]
        missing_keys = [k for k in required_keys if k not in symbol_pivots or symbol_pivots[k] is None]

        if missing_keys:
            self.log_message(f"❌ CRITICAL: Pivots incomplete (missing: {', '.join(missing_keys)})", False)
            self.log_message("🔄 Attempting emergency pivot calculation...", False)
            self.process_pivots()

            # Re-check after calculation
            pivot_data = robust_load_json(self.pivot_json, self.log_message, default={})
            symbol_pivots = pivot_data.get(self.symbol, {})
            missing_keys = [k for k in required_keys if k not in symbol_pivots or symbol_pivots[k] is None]

            if missing_keys:
                self.log_message(f"❌ FATAL: Cannot calculate pivots! Trading stopped.", False)
                return

        self.log_message(
            f"✅ Pivots verified: TC={symbol_pivots['TC']}, BC={symbol_pivots['BC']}, "
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
                    self.log_message(f"⚠️ Pivot JSON structure invalid for {sym}", False)
                    self.log_message(f"   JSON type: {type(pivot_json_data)}", False)
                    self.log_message(
                        f"   JSON keys: {list(pivot_json_data.keys()) if isinstance(pivot_json_data, dict) else 'NOT A DICT'}",
                        False)
                    pivots = {}

                # Ensure pivots is a dictionary
                if not isinstance(pivots, dict):
                    self.log_message(f"⚠️ Pivots for {sym} is {type(pivots)}, converting to dict", False)
                    pivots = {}

                # Validate required keys
                required_keys = ["TC", "BC", "R1", "R2", "R3", "S1", "S2", "S3"]
                missing_keys = [k for k in required_keys if k not in pivots or pivots[k] is None]

                if missing_keys:
                    self.log_message(
                        f"❌ Pivots incomplete for {sym}: missing {', '.join(missing_keys)}\n"
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
                            f"❌ CRITICAL: Cannot calculate pivots for {sym}\n"
                            f"   Still missing: {', '.join(missing_keys)}\n"
                            f"   SKIPPING THIS SYMBOL",
                            False
                        )
                        continue  # Skip this symbol entirely

                self.log_message(
                    f"✅ Pivots loaded for {sym}: "
                    f"TC={pivots['TC']}, BC={pivots['BC']}, "
                    f"R1={pivots['R1']}, S1={pivots['S1']}",
                    True
                )

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
                    # ✅ ENSURE pivots is a dictionary, not a list
                    if not isinstance(pivots, dict):
                        self.log_message(f"⚠️ Pivots for {sym} is {type(pivots)}, using empty dict", False)
                        pivots_to_pass = {}
                    else:
                        pivots_to_pass = pivots

                    indicators = self.indicator_calculator.calculate_indicators(
                        sym, tf, pivot_data=pivots_to_pass
                    )
                    fresh_indicators_all_tfs[sym][tf] = indicators
                    # 🔥 FIX: Inject timeframe into indicators for consistent downstream usage
                    indicators["timeframe"] = str(tf)

                    # Optional AI-CPR Analysis (Real-time friendly)
                    if "error" not in indicators and self.USE_AI_CPR_STRATEGY:
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

                        # 🎯 NEW: Detect current candle color (CRITICAL!)
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

                        # 🎯 ADAPTIVE THRESHOLDS based on timeframe
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

                        # ✅ ENTER IMMEDIATELY if conditions met
                        if (trend_aligned and
                                intra_candle_move_pct >= MOVE_THRESHOLD and
                                volume_ratio >= VOLUME_THRESHOLD and
                                abs(momentum_pct) >= MOMENTUM_THRESHOLD):

                            intra_candle_entry_allowed = True
                            st_status = "GREEN ✓" if st_main_trend > 0 else "RED ✓"

                            # 🔥 NEW: Calculate position in candle
                            position_pct = bullish_position if trend_direction == "BULLISH" else bearish_position

                            self.log_message(
                                f"🚀 [EARLY-ENTRY-{selected_tf}m] {trend_direction} trend detected!\n"
                                f"  📊 Timeframe: {selected_tf}m\n"
                                f"  📈 Move: {intra_candle_move_pct:.2f}% (threshold: {MOVE_THRESHOLD}%)\n"
                                f"  📊 Volume: {volume_ratio:.2f}x (threshold: {VOLUME_THRESHOLD}x)\n"
                                f"  🎯 Momentum: {momentum_pct:.2f}% (threshold: ±{MOMENTUM_THRESHOLD}%)\n"
                                f"  🎯 SuperTrend: {st_status}\n"
                                f"  🕯️ Candle: {'GREEN' if current_candle_bullish else 'RED'} "
                                f"(O:{candle_open:.2f} → C:{ltp:.2f})\n"
                                f"  📍 Position in candle: {position_pct * 100:.1f}% "
                                f"({'near HIGH' if trend_direction == 'BULLISH' else 'near LOW'})\n"
                                f"  ✅ EMA Alignment: {ema5:.2f} {'>' if trend_direction == 'BULLISH' else '<'} "
                                f"{ema9:.2f} {'>' if trend_direction == 'BULLISH' else '<'} {ema21:.2f}\n"
                                f"  ✅ Decision: ENTER NOW without waiting for candle close",
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

            # ✅ Process each timeframe independently
            for tf in self.ohlc_timeframes:
                minute_key = curr_time.replace(second=0, microsecond=0)

                # Increment counter on new minute
                if self.last_minute_seen.get(tf) != minute_key:
                    self.last_minute_seen[tf] = minute_key
                    self.timeframe_counters[tf] += 1

                # ✅ Close bar when timeframe boundary reached
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

                                # ✅ Save to separate CSV file per timeframe
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
                    # ✅ Append LTP to this timeframe's buffer
                    with self.ohlc_lock:
                        self.ohlc_data[tf].setdefault(symbol, []).append(ltp)

        except Exception as e:
            self.log_message(f"[OHLC] Processing error: {e}", False)
            import traceback
            self.log_message(f"[OHLC] Traceback: {traceback.format_exc()}", True)

    def _save_ohlc_csv(self, symbol, csv_dict, timeframe=5):
        """Enhanced with proper timeframe in filename"""
        try:
            # ✅ Include timeframe in filename
            csv_filename = f'{symbol.replace(":", "_")}_websocket_ohlc_{timeframe}min.csv'
            csv_path = os.path.join(os.getcwd(), "data_bot", csv_filename)

            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            file_exists = os.path.isfile(csv_path)

            # ✅ Ensure timeframe is in the dict
            csv_dict['timeframe'] = str(timeframe)

            with open(csv_path, 'a', newline='') as f:
                field_names = ['minute', 'symbol', 'open', 'high', 'low', 'close', 'timeframe']
                writer = DictWriter(f, fieldnames=field_names)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(csv_dict)

            #self.log_message(f"[OHLC-CSV] ✅ Saved to {csv_filename}", True)

        except Exception as e:
            self.log_message(f"[OHLC-CSV] ❌ Save error for {symbol}: {e}", False)

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
        print(entry)
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
        self.log_message("📊 Calculating CPR pivot levels...", False)

        # ============================================
        # STEP 1: FETCH DAILY OHLC DATA
        # ============================================
        # Use 60 days for better historical context (from version 2)
        df = self.fetch_ohlc(self.symbol, "D", 60)

        if df.empty:
            self.log_message("❌ No daily OHLC data available for pivot calculation!", False)
            return {}

        if len(df) < 2:
            self.log_message(f"❌ Insufficient daily data: {len(df)} rows (need at least 2)", False)
            return {}

        # ============================================
        # STEP 2: CALCULATE PIVOT POINTS
        # ============================================
        raw = self.indicator_calculator.calculate_pivot_points(df)

        if not raw or not isinstance(raw, dict):
            self.log_message("❌ Pivot calculation returned empty/invalid data!", False)
            return {}

        # ============================================
        # STEP 3: VALIDATION (from version 2)
        # ============================================
        required_keys = ["TC", "BC", "PP", "R1", "R2", "R3", "S1", "S2", "S3"]
        missing_keys = [k for k in required_keys if k not in raw or raw[k] is None]

        if missing_keys:
            self.log_message(
                f"❌ Pivot calculation incomplete! Missing: {', '.join(missing_keys)}",
                False
            )
            self.log_message(f"   Raw data: {raw}", True)
            return {}

        # ✅ VALIDATION: Check TC > BC (sanity check)
        if raw["TC"] < raw["BC"]:
            self.log_message(
                f"⚠️ WARNING: TC ({raw['TC']}) < BC ({raw['BC']}) - Inverted CPR!",
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

                # Get indicators for AI analysis (match active primary timeframe when available)
                # Prefer the bot's current primary timeframe; fallback to 5m if unknown
                try:
                    tf_for_ai = str(getattr(getattr(self, "bot", None), "last_known_primary_tf", None) or getattr(self, "primary_tf", None) or "5")
                except Exception:
                    tf_for_ai = "5"

                current_inds = self.indicator_calculator.calculate_indicators(
                    self.symbol,
                    tf_for_ai,
                    pivot_data=raw
                )
                current_inds = self.indicator_calculator.calculate_indicators(
                    self.symbol,
                    tf_for_ai,
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
                        f"✅ AI Context added: {ai_analysis.get('trade_strategy', 'None')} "
                        f"(conf: {ai_analysis.get('ai_confidence', 0.0):.2f})",
                        False
                    )
                else:
                    self.log_message(
                        f"⚠️ Cannot add AI context - indicator error: {current_inds.get('error')}",
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
            f"✅ CPR Pivots calculated successfully:\n"
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

        # ✅ VALIDATION: Check if pivots exist AND are valid
        needs_recalc = False

        if not isinstance(piv_data, dict) or self.symbol not in piv_data:
            self.log_message("⚠️ No pivot data found - generating new pivots", False)
            needs_recalc = True
        else:
            # Check if all required CPR levels exist
            required_keys = ["TC", "BC", "R1", "R2", "R3", "S1", "S2", "S3"]
            symbol_pivots = piv_data.get(self.symbol, {})
            missing_keys = [k for k in required_keys if k not in symbol_pivots or symbol_pivots[k] is None]

            if missing_keys:
                self.log_message(f"⚠️ Incomplete pivots (missing: {', '.join(missing_keys)}) - recalculating", False)
                needs_recalc = True
            else:
                # Check if pivots are stale (older than 24 hours)
                pivot_ts = symbol_pivots.get("ts", "1970-01-01")
                try:
                    pivot_age = pd.to_datetime(pivot_ts)
                    hours_old = (pd.Timestamp.now(tz=self.IST) - pivot_age).total_seconds() / 3600

                    if hours_old > 24:
                        self.log_message(f"⚠️ Pivots are stale ({hours_old:.1f}h old) - recalculating", False)
                        needs_recalc = True
                    else:
                        self.log_message(f"✅ Using existing pivots (age: {hours_old:.1f}h)", False)

                        # ✅ BONUS: Log AI context if available
                        if "ai_context" in symbol_pivots:
                            ai_ctx = symbol_pivots["ai_context"]
                            self.log_message(
                                f"   AI Context: {ai_ctx.get('ai_signal', 'None')} "
                                f"({ai_ctx.get('ai_confidence', 0.0):.2f})",
                                True
                            )

                except Exception as e:
                    self.log_message(f"⚠️ Pivot age check failed: {e} - recalculating", False)
                    needs_recalc = True

        # Recalculate if needed
        if needs_recalc:
            self.process_pivots()  # ✅ Now uses merged enhanced version

            # ✅ VERIFY calculation succeeded
            piv_data = robust_load_json(self.pivot_json, self.log_message, default={})
            if self.symbol in piv_data:
                symbol_pivots = piv_data[self.symbol]
                self.log_message(
                    f"✅ Pivots calculated successfully:\n"
                    f"   TC={symbol_pivots.get('TC')}, BC={symbol_pivots.get('BC')}\n"
                    f"   R1={symbol_pivots.get('R1')}, S1={symbol_pivots.get('S1')}",
                    False
                )
            else:
                self.log_message("❌ CRITICAL: Pivot calculation failed!", False)

    def fetch_and_store_adx(self):
        # 🔥 FIX #2: Use 30m timeframe instead of hardcoded "5"
        df = fetchOHLC1(self.symbol, interval="30", duration=10)  # 10 days for better ADX stability
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
        if inds.get("st21Trend") == 1:
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

        ema5 = indicators.get("ema_5")
        ema21 = indicators.get("ema_21")
        ltp = indicators.get("close")
        trend = indicators.get("trend")  # ← optional: higher timeframe trend flag
        last_pos = om.position.get("last_type", None)
        current_pos = om.position.get("type", "FLAT")

        # ✅ Guard: If indicators are missing, skip
        if ema5 is None or ema21 is None or ltp is None:
            self.log_message(
                f"[AI-CPR] Skipping order for {sym}: indicators missing (ema5={ema5}, ema21={ema21}, ltp={ltp})", True
            )
            return "SKIP - Indicators missing"

        # ✅ Normalize side
        if side in ["Bullish", "BUY"]:
            side = "BUY"
        elif side in ["Bearish", "SELL"]:
            side = "SELL"
        else:
            if current_pos != "FLAT":
                return "HOLD"
            return "FLAT - No trade"

        # ✅ Optional Trend Filter (skip entries against trend)
        if trend:
            if side == "BUY" and trend != "UP":
                self.log_message(f"[AI-CPR] Skipping BUY for {sym}: Trend is {trend}", True)
                return "SKIP - Against trend"
            if side == "SELL" and trend != "DOWN":
                self.log_message(f"[AI-CPR] Skipping SELL for {sym}: Trend is {trend}", True)
                return "SKIP - Against trend"

        # ✅ New Entry
        if current_pos == "FLAT":
            if side == "BUY" and ema5 > ema21 and last_pos != "BUY":
                om.ai_buy(sym, trade_qty)
                return "BUY"
            elif side == "SELL" and ema5 < ema21 and last_pos != "SELL":
                om.ai_sell(sym, trade_qty)
                return "SELL"
            return "FLAT - No trade"

        # ✅ Exit on direction change (🔥 FIX #5: Add trend confirmation)
        if (current_pos == "BUY" and side == "SELL") or (current_pos == "SELL" and side == "BUY"):
            # 🔥 Require trend confirmation before AI direction-change exit
            if not (ema5 and ema21):
                self.log_message(f"[AI-CPR] Direction change but missing EMA data - HOLDING", True)
                return "HOLD - Missing EMA confirmation"
            
            # BUY->SELL: Require EMA5 < EMA21 (bearish)
            if current_pos == "BUY" and side == "SELL":
                if ema5 < ema21:
                    self.log_message(
                        f"[AI-CPR] EXIT LONG: AI flip to SELL confirmed by EMA trend | "
                        f"EMA5:{ema5:.2f} < EMA21:{ema21:.2f}", 
                        False
                    )
                    om.ai_exit_all(sym)
                    return "EXIT on confirmed direction change"
                else:
                    self.log_message(
                        f"[AI-CPR] HOLD LONG: AI flip to SELL but EMA5 still > EMA21 | "
                        f"EMA5:{ema5:.2f} > EMA21:{ema21:.2f} - ignoring AI flip",
                        True
                    )
                    return "HOLD - AI flip not confirmed by trend"
            
            # SELL->BUY: Require EMA5 > EMA21 (bullish)
            elif current_pos == "SELL" and side == "BUY":
                if ema5 > ema21:
                    self.log_message(
                        f"[AI-CPR] EXIT SHORT: AI flip to BUY confirmed by EMA trend | "
                        f"EMA5:{ema5:.2f} > EMA21:{ema21:.2f}",
                        False
                    )
                    om.ai_exit_all(sym)
                    return "EXIT on confirmed direction change"
                else:
                    self.log_message(
                        f"[AI-CPR] HOLD SHORT: AI flip to BUY but EMA5 still < EMA21 | "
                        f"EMA5:{ema5:.2f} < EMA21:{ema21:.2f} - ignoring AI flip",
                        True
                    )
                    return "HOLD - AI flip not confirmed by trend"

        # ✅ Hold if already in same side
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

def main():
    try:
        bot = TradingBot()
        # Trade based on 1-min selected TF for decision cycle; 5m/15m used inside OM
        bot.run(selected_tf=tf_selected)
    except Exception as e:
        ts = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{ts}] FATAL: {e}\\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
