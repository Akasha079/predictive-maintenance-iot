"""
Feature Engineering for IoT Sensor Data.

Extracts time-domain, frequency-domain, and statistical features
from raw sensor readings for use in anomaly detection and RUL prediction.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.fft import fft, fftfreq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SensorConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extracts predictive maintenance features from raw IoT sensor data."""

    def __init__(self, window_size: int = 60, step_size: int = 30):
        """
        Initialize feature engineer.

        Args:
            window_size: Number of samples per rolling window.
            step_size: Step size for rolling window computation.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.sensor_config = SensorConfig()
        self.sensor_names = self.sensor_config.SENSOR_NAMES

    def compute_time_domain_features(self, signal: np.ndarray) -> dict:
        """
        Compute time-domain statistical features from a signal window.

        Args:
            signal: 1D array of sensor readings.

        Returns:
            Dictionary of time-domain features.
        """
        if len(signal) == 0:
            return {}

        features = {}
        features["mean"] = np.mean(signal)
        features["std"] = np.std(signal)
        features["rms"] = np.sqrt(np.mean(signal ** 2))
        features["max"] = np.max(signal)
        features["min"] = np.min(signal)
        features["peak_to_peak"] = np.max(signal) - np.min(signal)
        features["median"] = np.median(signal)
        features["variance"] = np.var(signal)

        # Higher-order statistics
        if features["std"] > 1e-10:
            features["kurtosis"] = float(scipy_stats.kurtosis(signal, fisher=True))
            features["skewness"] = float(scipy_stats.skew(signal))
        else:
            features["kurtosis"] = 0.0
            features["skewness"] = 0.0

        # Crest factor and shape factor
        if features["rms"] > 1e-10:
            features["crest_factor"] = features["max"] / features["rms"]
        else:
            features["crest_factor"] = 0.0

        abs_mean = np.mean(np.abs(signal))
        if abs_mean > 1e-10:
            features["shape_factor"] = features["rms"] / abs_mean
        else:
            features["shape_factor"] = 0.0

        # Impulse factor
        if abs_mean > 1e-10:
            features["impulse_factor"] = features["max"] / abs_mean
        else:
            features["impulse_factor"] = 0.0

        return features

    def compute_frequency_domain_features(
        self, signal: np.ndarray, sampling_rate: float = 1.0
    ) -> dict:
        """
        Compute frequency-domain features using FFT.

        Args:
            signal: 1D array of sensor readings.
            sampling_rate: Sampling rate in Hz.

        Returns:
            Dictionary of frequency-domain features.
        """
        if len(signal) < 4:
            return {
                "dominant_freq": 0.0,
                "spectral_energy": 0.0,
                "spectral_entropy": 0.0,
                "spectral_centroid": 0.0,
                "bandwidth": 0.0,
            }

        n = len(signal)
        yf = fft(signal - np.mean(signal))  # Remove DC component
        xf = fftfreq(n, 1.0 / sampling_rate)

        # Only positive frequencies
        positive_mask = xf > 0
        xf_pos = xf[positive_mask]
        power_spectrum = np.abs(yf[positive_mask]) ** 2

        features = {}

        if len(power_spectrum) == 0 or np.sum(power_spectrum) < 1e-10:
            features["dominant_freq"] = 0.0
            features["spectral_energy"] = 0.0
            features["spectral_entropy"] = 0.0
            features["spectral_centroid"] = 0.0
            features["bandwidth"] = 0.0
        else:
            # Dominant frequency
            features["dominant_freq"] = float(xf_pos[np.argmax(power_spectrum)])

            # Total spectral energy
            features["spectral_energy"] = float(np.sum(power_spectrum))

            # Spectral entropy
            psd_norm = power_spectrum / np.sum(power_spectrum)
            psd_norm = psd_norm[psd_norm > 0]
            features["spectral_entropy"] = float(-np.sum(psd_norm * np.log2(psd_norm)))

            # Spectral centroid
            features["spectral_centroid"] = float(
                np.sum(xf_pos * power_spectrum) / np.sum(power_spectrum)
            )

            # Bandwidth (spectral spread)
            centroid = features["spectral_centroid"]
            features["bandwidth"] = float(
                np.sqrt(
                    np.sum(((xf_pos - centroid) ** 2) * power_spectrum)
                    / np.sum(power_spectrum)
                )
            )

        return features

    def compute_rolling_statistics(self, df: pd.DataFrame, sensor: str) -> pd.DataFrame:
        """
        Compute rolling window statistics for a sensor column.

        Args:
            df: DataFrame containing sensor data.
            sensor: Name of the sensor column.

        Returns:
            DataFrame with rolling statistics columns added.
        """
        result = pd.DataFrame(index=df.index)
        col = df[sensor]

        for window in [10, 30, 60]:
            prefix = f"{sensor}_roll{window}"
            rolling = col.rolling(window=window, min_periods=1)
            result[f"{prefix}_mean"] = rolling.mean()
            result[f"{prefix}_std"] = rolling.std().fillna(0)
            result[f"{prefix}_max"] = rolling.max()
            result[f"{prefix}_min"] = rolling.min()

        return result

    def compute_rate_of_change(self, df: pd.DataFrame, sensor: str) -> pd.DataFrame:
        """
        Compute rate of change features for a sensor.

        Args:
            df: DataFrame containing sensor data.
            sensor: Name of the sensor column.

        Returns:
            DataFrame with rate of change columns.
        """
        result = pd.DataFrame(index=df.index)
        col = df[sensor]

        result[f"{sensor}_diff1"] = col.diff().fillna(0)
        result[f"{sensor}_diff2"] = col.diff().diff().fillna(0)
        result[f"{sensor}_pct_change"] = col.pct_change().fillna(0).replace(
            [np.inf, -np.inf], 0
        )

        return result

    def compute_health_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite health indicators from multiple sensors.

        Args:
            df: DataFrame with all sensor readings.

        Returns:
            DataFrame with health indicator columns.
        """
        result = pd.DataFrame(index=df.index)

        # Normalized distance from normal range center for each sensor
        for sensor in self.sensor_names:
            if sensor not in df.columns:
                continue
            config = self.sensor_config.SENSOR_TYPES[sensor]
            low, high = config["normal_range"]
            center = (low + high) / 2
            span = (high - low) / 2
            if span > 0:
                result[f"{sensor}_deviation"] = np.abs(df[sensor] - center) / span
            else:
                result[f"{sensor}_deviation"] = 0.0

        # Combined health indicator (average deviation)
        deviation_cols = [c for c in result.columns if c.endswith("_deviation")]
        if deviation_cols:
            result["combined_deviation"] = result[deviation_cols].mean(axis=1)
        else:
            result["combined_deviation"] = 0.0

        return result

    def compute_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute multi-sensor correlation features using rolling windows.

        Args:
            df: DataFrame with sensor readings.

        Returns:
            DataFrame with correlation features.
        """
        result = pd.DataFrame(index=df.index)
        available_sensors = [s for s in self.sensor_names if s in df.columns]

        # Pairwise rolling correlations
        window = min(30, len(df))
        for i, s1 in enumerate(available_sensors):
            for s2 in available_sensors[i + 1:]:
                col_name = f"corr_{s1}_{s2}"
                result[col_name] = (
                    df[s1]
                    .rolling(window=window, min_periods=2)
                    .corr(df[s2])
                    .fillna(0)
                )

        # Vibration-temperature ratio (common indicator)
        if "vibration" in df.columns and "temperature" in df.columns:
            temp_safe = df["temperature"].replace(0, 1e-6)
            result["vib_temp_ratio"] = df["vibration"] / temp_safe

        # Current-voltage ratio (power indicator)
        if "current" in df.columns and "voltage" in df.columns:
            voltage_safe = df["voltage"].replace(0, 1e-6)
            result["current_voltage_ratio"] = df["current"] / voltage_safe

        return result

    def extract_features_for_window(self, window_data: pd.DataFrame) -> dict:
        """
        Extract all features from a single window of data.

        Args:
            window_data: DataFrame slice representing one window.

        Returns:
            Dictionary of all extracted features.
        """
        features = {}

        for sensor in self.sensor_names:
            if sensor not in window_data.columns:
                continue

            signal = window_data[sensor].values
            config = self.sensor_config.SENSOR_TYPES[sensor]

            # Time-domain features
            td_features = self.compute_time_domain_features(signal)
            for key, value in td_features.items():
                features[f"{sensor}_{key}"] = value

            # Frequency-domain features
            fd_features = self.compute_frequency_domain_features(
                signal, config["sampling_rate_hz"]
            )
            for key, value in fd_features.items():
                features[f"{sensor}_{key}"] = value

        return features

    def extract_all_features(
        self,
        df: pd.DataFrame,
        device_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extract comprehensive features from a device's sensor data.

        Args:
            df: Raw sensor data DataFrame.
            device_id: Optional device ID to filter data.

        Returns:
            DataFrame with all engineered features.
        """
        if device_id is not None:
            df = df[df["device_id"] == device_id].copy()

        if len(df) == 0:
            logger.warning("No data found for feature extraction")
            return pd.DataFrame()

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Collect all feature DataFrames
        feature_frames = [df[["device_id", "timestamp"]].copy()]

        # Preserve target columns if they exist
        for col in ["health_status", "degradation_progress", "rul_hours", "failure_mode"]:
            if col in df.columns:
                feature_frames.append(df[[col]])

        # Raw sensor values
        available_sensors = [s for s in self.sensor_names if s in df.columns]
        if available_sensors:
            feature_frames.append(df[available_sensors])

        # Rolling statistics
        for sensor in available_sensors:
            rolling_df = self.compute_rolling_statistics(df, sensor)
            feature_frames.append(rolling_df)

        # Rate of change
        for sensor in available_sensors:
            roc_df = self.compute_rate_of_change(df, sensor)
            feature_frames.append(roc_df)

        # Health indicators
        health_df = self.compute_health_indicators(df)
        feature_frames.append(health_df)

        # Correlation features
        corr_df = self.compute_correlation_features(df)
        feature_frames.append(corr_df)

        # Combine all features
        result = pd.concat(feature_frames, axis=1)

        # Remove duplicate columns
        result = result.loc[:, ~result.columns.duplicated()]

        # Replace infinities and fill NaN
        result = result.replace([np.inf, -np.inf], np.nan)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)

        logger.info(
            "Extracted %d features from %d samples", len(result.columns), len(result)
        )
        return result

    def get_feature_names(self) -> List[str]:
        """
        Get the list of all feature names that will be generated.

        Returns:
            List of feature column names.
        """
        # Generate a small sample to get feature names
        dummy_data = {}
        for sensor in self.sensor_names:
            dummy_data[sensor] = np.random.randn(100)
        dummy_data["device_id"] = "dummy"
        dummy_data["timestamp"] = pd.date_range("2024-01-01", periods=100, freq="min")

        dummy_df = pd.DataFrame(dummy_data)
        features_df = self.extract_all_features(dummy_df)

        exclude = {"device_id", "timestamp", "health_status", "degradation_progress",
                    "rul_hours", "failure_mode"}
        return [c for c in features_df.columns if c not in exclude]
