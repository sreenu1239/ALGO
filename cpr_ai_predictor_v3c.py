 

import joblib
import numpy as np
import logging
import pickle
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd


class CPR_AIPredictor:
    """
    AI-powered CPR (Central Pivot Range) prediction system.
    
    Combines technical indicators and candlestick patterns to predict market direction.
    """
    
    def __init__(self, model_path: str = "ai_cpr_jan_model_v4.pkl", logger: Optional[logging.Logger] = None):
        """
        Initialize the CPR AI Predictor.
        
        Args:
            model_path: Path to the trained pickle model file
            logger: Optional logger instance
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.n_features = 30  # 18 technical + 12 candle patterns
        self.logger = logger or logging.getLogger(__name__)
        self.order_manager = None

        # Track predictions for analysis
        self.prediction_history = []
        self.last_prediction = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from pickle file."""
        if not os.path.exists(self.model_path):
            self.logger.warning(f"[AI-CPR] Model not found: {self.model_path}")
            return

        try:
            with open(self.model_path, 'rb') as f:
                package = pickle.load(f)

            self.model = package.get('model')
            self.scaler = package.get('scaler')
            self.feature_names = package.get('feature_names', [])
            self.n_features = package.get('n_features', 30)
            version = package.get('version', '1.0')
            trained_date = package.get('trained_date', 'Unknown')

            self.logger.info(
                f"\n{'=' * 60}\n"
                f"AI CPR MODEL LOADED\n"
                f"{'=' * 60}\n"
                f"Path: {self.model_path}\n"
                f"Version: {version}\n"
                f"Features: {self.n_features}\n"
                f"Trained: {trained_date}\n"
                f"Model Type: {type(self.model).__name__}\n"
                f"{'=' * 60}"
            )

        except Exception as e:
            self.logger.error(f"[AI-CPR] Failed to load model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.model = None

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """
        Safely convert value to float.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            Float value or default
        """
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value.replace(',', ''))
            return default
        except (ValueError, TypeError, AttributeError):
            return default

    def _interpret_candle_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Interpret the 12 candle pattern features.
        
        Args:
            features: Flattened feature array
            
        Returns:
            Dictionary with candle pattern interpretation
        """
        if len(features) < 12:
            self.logger.warning(f"[AI-CPR] Insufficient features for candle analysis: {len(features)} < 12")
            return {}

        candle_features = features[-12:]

        # Safely unpack features with bounds checking
        try:
            (body_pct, body_direction, upper_wick_pct, lower_wick_pct,
             engulfing_score, reversal_pattern, marubozu_score, momentum_3,
             range_expansion, gap_score, close_position, vol_ratio) = candle_features
        except ValueError as e:
            self.logger.error(f"[AI-CPR] Error unpacking candle features: {e}")
            return {}

        # Convert to safe floats
        body_pct = self._safe_float(body_pct)
        body_direction = self._safe_float(body_direction)
        upper_wick_pct = self._safe_float(upper_wick_pct)
        lower_wick_pct = self._safe_float(lower_wick_pct)
        vol_ratio = self._safe_float(vol_ratio, 1.0)
        close_position = self._safe_float(close_position, 50.0)

        interpretation = {
            "body_strength": "STRONG" if body_pct > 80 else "MODERATE" if body_pct > 60 else "WEAK",
            "body_pct": body_pct,
            "direction": "BULLISH" if body_direction > 0 else "BEARISH" if body_direction < 0 else "NEUTRAL",
            "upper_wick": upper_wick_pct,
            "lower_wick": lower_wick_pct,
            "patterns": []
        }

        # Detect patterns with safety checks
        if self._safe_float(engulfing_score) > 0:
            interpretation["patterns"].append("Bullish Engulfing")
        elif self._safe_float(engulfing_score) < 0:
            interpretation["patterns"].append("Bearish Engulfing")

        if self._safe_float(reversal_pattern) > 0:
            interpretation["patterns"].append("Hammer (bullish reversal)")
        elif self._safe_float(reversal_pattern) < 0:
            interpretation["patterns"].append("Shooting Star (bearish reversal)")

        if self._safe_float(marubozu_score) > 0:
            interpretation["patterns"].append("Bullish Marubozu")
        elif self._safe_float(marubozu_score) < 0:
            interpretation["patterns"].append("Bearish Marubozu")

        if self._safe_float(momentum_3) > 0:
            interpretation["patterns"].append("3-Candle Bull Momentum")
        elif self._safe_float(momentum_3) < 0:
            interpretation["patterns"].append("3-Candle Bear Momentum")

        if self._safe_float(gap_score) > 0:
            interpretation["patterns"].append("Gap Up")
        elif self._safe_float(gap_score) < 0:
            interpretation["patterns"].append("Gap Down")

        # Volume analysis
        if vol_ratio > 1.5:
            interpretation["volume"] = "HIGH"
            interpretation["patterns"].append(f"High Volume ({vol_ratio:.1f}x)")
        elif vol_ratio < 0.7:
            interpretation["volume"] = "LOW"
            interpretation["patterns"].append(f"Low Volume ({vol_ratio:.1f}x)")
        else:
            interpretation["volume"] = "NORMAL"

        # Close position analysis
        if close_position > 80:
            interpretation["close_strength"] = "Strong (near high)"
        elif close_position < 20:
            interpretation["close_strength"] = "Weak (near low)"
        else:
            interpretation["close_strength"] = "Neutral (mid-range)"

        return interpretation

    def _interpret_technical_features(self, features: np.ndarray) -> Dict[str, float]:
        """
        Interpret the first 18 technical indicator features.
        
        Args:
            features: Flattened feature array
            
        Returns:
            Dictionary with technical indicator values
        """
        if len(features) < 18:
            self.logger.warning(f"[AI-CPR] Insufficient features for technical analysis: {len(features)} < 18")
            return {}

        tech_features = features[:18]

        return {
            "ema_5": self._safe_float(tech_features[0]),
            "ema_9": self._safe_float(tech_features[1]),
            "ema_21": self._safe_float(tech_features[2]),
            "ema_50": self._safe_float(tech_features[3]),
            "ema_200": self._safe_float(tech_features[4]),
            "rsi": self._safe_float(tech_features[5]),
            "momentum": self._safe_float(tech_features[6]),
            "roc_10": self._safe_float(tech_features[7]),
            "atr": self._safe_float(tech_features[8]),
            "bb_bandwidth": self._safe_float(tech_features[9]),
            "volume_ratio": self._safe_float(tech_features[10]),
            "efi": self._safe_float(tech_features[11]),
            "adx": self._safe_float(tech_features[12]),
            "plus_di": self._safe_float(tech_features[13]),
            "minus_di": self._safe_float(tech_features[14]),
            "cpr_distance_tc": self._safe_float(tech_features[15]),
            "cpr_distance_bc": self._safe_float(tech_features[16]),
            "cpr_width": self._safe_float(tech_features[17]),
        }

    def _log_prediction_details(self, features: np.ndarray, label: str, 
                                confidence: float, distribution: Dict[str, float]) -> Dict[str, Any]:
        """
        Log detailed prediction information.
        
        Args:
            features: Feature array
            label: Predicted label
            confidence: Prediction confidence
            distribution: Probability distribution
            
        Returns:
            Analysis dictionary
        """
        features_flat = features.flatten()
        candle_interp = self._interpret_candle_features(features_flat)
        tech_interp = self._interpret_technical_features(features_flat)

        parts = []
        parts.append(
            f"\n{'=' * 70}\n"
            f"AI CPR PREDICTION - {datetime.now().strftime('%H:%M:%S')}\n"
            f"{'=' * 70}\n\n"
            f"PREDICTION: {label} (Confidence: {confidence:.1%})\n\n"
        )

        # Candle analysis
        if candle_interp:
            parts.append(
                "CANDLE ANALYSIS:\n"
                f"  Direction: {candle_interp.get('direction', 'N/A')} "
                f"(Body: {candle_interp.get('body_strength', 'N/A')} {candle_interp.get('body_pct', 0):.1f}%)\n"
                f"  Upper Wick: {candle_interp.get('upper_wick', 0):.1f}% "
                f"{'(rejection)' if candle_interp.get('upper_wick', 0) > 30 else ''}\n"
                f"  Lower Wick: {candle_interp.get('lower_wick', 0):.1f}% "
                f"{'(support)' if candle_interp.get('lower_wick', 0) > 30 else ''}\n"
                f"  Close Position: {candle_interp.get('close_strength', 'N/A')}\n"
                f"  Volume: {candle_interp.get('volume', 'N/A')}\n\n"
            )

        # Technical indicators
        if tech_interp:
            ema_5 = tech_interp.get('ema_5', 0)
            ema_21 = tech_interp.get('ema_21', 0)
            rsi = tech_interp.get('rsi', 0)
            adx = tech_interp.get('adx', 0)
            
            parts.append(
                "TECHNICAL INDICATORS:\n"
                f"  Trend: EMA5({ema_5:.2f}) "
                f"{'>' if ema_5 > ema_21 else '<'} "
                f"EMA21({ema_21:.2f})\n"
                f"  RSI: {rsi:.1f} "
                f"{'(Overbought)' if rsi > 70 else '(Oversold)' if rsi < 30 else ''}\n"
                f"  ADX: {adx:.1f} "
                f"{'(Strong trend)' if adx > 25 else '(Weak trend)'}\n"
                f"  Momentum: {tech_interp.get('momentum', 0):.3f}\n"
                f"  Volume Ratio: {tech_interp.get('volume_ratio', 0):.2f}x\n\n"
                f"CPR POSITION:\n"
                f"  Distance to TC: {tech_interp.get('cpr_distance_tc', 0) * 100:+.2f}%\n"
                f"  Distance to BC: {tech_interp.get('cpr_distance_bc', 0) * 100:+.2f}%\n"
                f"  CPR Width: {tech_interp.get('cpr_width', 0) * 100:.2f}%\n"
            )
    
        # Patterns
        if candle_interp and candle_interp.get('patterns'):
            parts.append("\n  Patterns Detected:\n")
            for pattern in candle_interp['patterns']:
                parts.append(f"    - {pattern}\n")
        else:
            parts.append("  Patterns: None detected\n")

        # Probability distribution
        parts.append("\n PROBABILITY DISTRIBUTION:\n")
        for label_name, prob in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(prob * 30)
            bar = '=' * bar_length
            parts.append(f"  {label_name:12s}: {prob:6.1%} {bar}\n")

        parts.append(f"{'=' * 70}\n")

        log_msg = ''.join(parts)
        
        # Safe logging (handle Unicode errors)
        try:
            self.logger.info(log_msg)
        except UnicodeEncodeError:
            # Fallback: strip non-ASCII characters
            clean_msg = log_msg.encode('ascii', 'ignore').decode('ascii')
            self.logger.info(clean_msg)

        return {
            "candle_analysis": candle_interp,
            "technical_analysis": tech_interp
        }

    def predict(self, indicators: Dict[str, Any], pivot_data: Dict[str, Any], 
                feature_builder: callable, ohlc_df: Optional[pd.DataFrame] = None) -> Tuple[Optional[str], float, Optional[Dict], Optional[np.ndarray]]:
        """
        Make prediction with enhanced validation and logging.
        
        Args:
            indicators: Dictionary of technical indicators
            pivot_data: CPR pivot data (TC, BC, Pivot)
            feature_builder: Function to build feature array
            ohlc_df: Optional OHLC dataframe for candle pattern analysis
            
        Returns:
            Tuple of (label, confidence, distribution, features)
        """
        if self.model is None:
            self.logger.warning("[AI-CPR] Model not loaded - skipping prediction")
            return None, 0.0, None, None

        try:
            # Validate inputs
            if not isinstance(indicators, dict):
                self.logger.error(f"[AI-CPR] Invalid indicators type: {type(indicators)}")
                return None, 0.0, None, None
                
            if not isinstance(pivot_data, dict):
                self.logger.error(f"[AI-CPR] Invalid pivot_data type: {type(pivot_data)}")
                return None, 0.0, None, None

            # Get LTP with validation
            ltp = indicators.get("close")
            if ltp is None:
                ltp = indicators.get("ltp")
            if ltp is None:
                self.logger.warning("[AI-CPR] No LTP/close available in indicators")
                return None, 0.0, None, None

            ltp = self._safe_float(ltp)
            if ltp <= 0:
                self.logger.warning(f"[AI-CPR] Invalid LTP value: {ltp}")
                return None, 0.0, None, None

            # Validate OHLC dataframe if provided
            if ohlc_df is not None:
                if not isinstance(ohlc_df, pd.DataFrame):
                    self.logger.warning(f"[AI-CPR] Invalid OHLC type: {type(ohlc_df)}, expected DataFrame")
                    ohlc_df = None
                elif ohlc_df.empty:
                    self.logger.warning("[AI-CPR] Empty OHLC dataframe provided")
                    ohlc_df = None
                elif len(ohlc_df) < 12:
                    self.logger.warning(f"[AI-CPR] Insufficient OHLC data: {len(ohlc_df)} rows (need 12+ for candle patterns)")
                    # Continue with limited features instead of failing

            # Build features
            try:
                features = feature_builder(ltp, indicators, pivot_data, ohlc_df=ohlc_df)
            except Exception as e:
                self.logger.error(f"[AI-CPR] Feature builder failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return None, 0.0, None, None

            # Validate features
            if features is None:
                self.logger.error("[AI-CPR] Feature builder returned None")
                return None, 0.0, None, None
                
            if not isinstance(features, np.ndarray):
                self.logger.error(f"[AI-CPR] Invalid features type: {type(features)}")
                return None, 0.0, None, None

            # Check feature shape
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            if features.shape[1] != self.n_features:
                self.logger.error(
                    f"[AI-CPR] Feature mismatch: got {features.shape[1]}, expected {self.n_features}"
                )
                return None, 0.0, None, None

            # Check for NaN or infinite values
            if np.any(~np.isfinite(features)):
                self.logger.warning("[AI-CPR] Features contain NaN or infinite values, replacing with 0")
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features
            if self.scaler:
                try:
                    features_scaled = self.scaler.transform(features)
                except Exception as e:
                    self.logger.error(f"[AI-CPR] Scaling failed: {e}")
                    return None, 0.0, None, None
            else:
                features_scaled = features

            # Predict
            try:
                prediction = self.model.predict(features_scaled)[0]
                probabilities = self.model.predict_proba(features_scaled)[0]
            except Exception as e:
                self.logger.error(f"[AI-CPR] Model prediction failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return None, 0.0, None, None

            # Map prediction to label
            label_map = {
                -2: "STRONG_SELL",
                -1: "SELL",
                0: "HOLD",
                1: "BUY",
                2: "STRONG_BUY"
            }

            label = label_map.get(int(prediction), "UNKNOWN")
            confidence = float(probabilities.max())

            # Build distribution
            distribution = {}
            for i, prob in enumerate(probabilities):
                map_key = i - 2  # Maps index 0->-2, 1->-1, 2->0, 3->1, 4->2
                distribution[label_map.get(map_key, "UNKNOWN")] = float(prob)

            # Log detailed prediction
            analysis = self._log_prediction_details(features, label, confidence, distribution)

            # Store prediction history
            self.last_prediction = {
                "timestamp": datetime.now(),
                "label": label,
                "confidence": confidence,
                "distribution": distribution,
                "ltp": ltp,
                "analysis": analysis
            }
            self.prediction_history.append(self.last_prediction)

            # Keep last 100 predictions
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]

            # Return features as 4th value for logging in main script
            return label, confidence, distribution, features

        except Exception as e:
            self.logger.error(f"[AI-CPR] Prediction error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, 0.0, None, None

    def get_prediction_summary(self) -> str:
        """
        Get summary of recent predictions.
        
        Returns:
            Formatted string with recent prediction history
        """
        if not self.prediction_history:
            return "No predictions yet"

        recent = self.prediction_history[-10:]

        summary_parts = ["\nRECENT PREDICTIONS:\n"]
        for pred in recent:
            summary_parts.append(
                f"  [{pred['timestamp'].strftime('%H:%M:%S')}] "
                f"{pred['label']} (conf: {pred['confidence']:.2f}) "
                f"@ LTP {pred['ltp']:.2f}\n"
            )

        return ''.join(summary_parts)

    def get_feature_importance(self) -> Optional[List[Tuple[str, float]]]:
        """
        Get feature importance if model supports it.
        
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        if self.model is None:
            return None

        try:
            # Check if model has feature_importances_ attribute
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_

                # Get feature names (if available)
                if self.feature_names and len(self.feature_names) == len(importances):
                    feature_dict = dict(zip(self.feature_names, importances))
                else:
                    # Use generic names
                    feature_dict = {
                        f"Feature_{i}": float(imp)
                        for i, imp in enumerate(importances)
                    }

                # Sort by importance (descending)
                sorted_features = sorted(
                    feature_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                return sorted_features[:10]  # Top 10
                
        except Exception as e:
            self.logger.error(f"[AI-CPR] Error getting feature importance: {e}")

        return None

    def get_last_prediction(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent prediction.
        
        Returns:
            Dictionary with last prediction details or None
        """
        return self.last_prediction

    def clear_history(self) -> None:
        """Clear prediction history."""
        self.prediction_history.clear()
        self.last_prediction = None
        self.logger.info("[AI-CPR] Prediction history cleared")


# Standalone test function
def test_predictor():
    """Test the CPR AI Predictor with sample data."""
    logging.basicConfig(level=logging.INFO)
    
    predictor = CPR_AIPredictor()
    
    # Sample indicators
    indicators = {
        "close": 322.65,
        "ema_5": 322.0,
        "ema_21": 321.5,
        "rsi": 65.0,
        "momentum": 2.45,
        "adx": 25.0,
        "efi": 16.0
    }
    
    # Sample pivot data
    pivot_data = {
        "TC": 323.0,
        "BC": 321.0,
        "Pivot": 322.0
    }
    
    print("\nTest completed. Check logs for details.")


if __name__ == "__main__":
    test_predictor()