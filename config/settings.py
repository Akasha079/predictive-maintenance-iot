"""
Configuration settings for the Predictive Maintenance IoT System.

Defines sensor types, failure modes, thresholds, model paths,
and alert levels used throughout the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class SensorConfig:
    """Configuration for IoT sensor types and their operating parameters."""

    SENSOR_TYPES = {
        "vibration": {
            "unit": "mm/s",
            "normal_range": (0.5, 4.5),
            "warning_threshold": 7.0,
            "critical_threshold": 11.0,
            "sampling_rate_hz": 100,
            "noise_std": 0.3,
        },
        "temperature": {
            "unit": "celsius",
            "normal_range": (35.0, 75.0),
            "warning_threshold": 85.0,
            "critical_threshold": 95.0,
            "sampling_rate_hz": 10,
            "noise_std": 1.5,
        },
        "pressure": {
            "unit": "bar",
            "normal_range": (2.0, 6.0),
            "warning_threshold": 7.5,
            "critical_threshold": 9.0,
            "sampling_rate_hz": 10,
            "noise_std": 0.2,
        },
        "rpm": {
            "unit": "rev/min",
            "normal_range": (1400, 1600),
            "warning_threshold": 1750,
            "critical_threshold": 1900,
            "sampling_rate_hz": 50,
            "noise_std": 15.0,
        },
        "voltage": {
            "unit": "volts",
            "normal_range": (380, 420),
            "warning_threshold": 440,
            "critical_threshold": 460,
            "sampling_rate_hz": 50,
            "noise_std": 3.0,
        },
        "current": {
            "unit": "amps",
            "normal_range": (10, 25),
            "warning_threshold": 30,
            "critical_threshold": 35,
            "sampling_rate_hz": 50,
            "noise_std": 1.0,
        },
    }

    SENSOR_NAMES = list(SENSOR_TYPES.keys())


class FailureConfig:
    """Configuration for equipment failure modes and their characteristics."""

    FAILURE_MODES = {
        "bearing_failure": {
            "description": "Bearing degradation leading to increased vibration and temperature",
            "affected_sensors": ["vibration", "temperature", "current"],
            "degradation_rate": 0.02,
            "vibration_multiplier": 3.5,
            "temperature_offset": 20.0,
            "current_multiplier": 1.4,
            "mean_time_to_failure_hours": 500,
            "onset_percentage": 0.6,
        },
        "overheating": {
            "description": "Thermal runaway due to cooling system failure or overload",
            "affected_sensors": ["temperature", "current", "voltage"],
            "degradation_rate": 0.03,
            "temperature_multiplier": 1.8,
            "current_multiplier": 1.3,
            "voltage_drop": 15.0,
            "mean_time_to_failure_hours": 300,
            "onset_percentage": 0.5,
        },
        "electrical_fault": {
            "description": "Electrical insulation breakdown or winding fault",
            "affected_sensors": ["voltage", "current", "temperature"],
            "degradation_rate": 0.04,
            "voltage_variance_multiplier": 4.0,
            "current_spike_probability": 0.15,
            "temperature_offset": 10.0,
            "mean_time_to_failure_hours": 200,
            "onset_percentage": 0.7,
        },
        "mechanical_wear": {
            "description": "General mechanical wear on rotating components",
            "affected_sensors": ["vibration", "rpm", "pressure", "current"],
            "degradation_rate": 0.015,
            "vibration_multiplier": 2.5,
            "rpm_variance_multiplier": 2.0,
            "pressure_drop": 1.5,
            "current_multiplier": 1.2,
            "mean_time_to_failure_hours": 700,
            "onset_percentage": 0.4,
        },
    }

    FAILURE_NAMES = list(FAILURE_MODES.keys())


class ModelConfig:
    """Configuration for ML model paths and hyperparameters."""

    MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models")))

    ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_detector.joblib"
    ANOMALY_SCALER_PATH = MODEL_DIR / "anomaly_scaler.joblib"
    RF_RUL_MODEL_PATH = MODEL_DIR / "rf_rul_model.joblib"
    XGB_RUL_MODEL_PATH = MODEL_DIR / "xgb_rul_model.joblib"
    LSTM_RUL_MODEL_PATH = MODEL_DIR / "lstm_rul_model.keras"
    FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"
    RUL_SCALER_PATH = MODEL_DIR / "rul_scaler.joblib"

    # Isolation Forest parameters
    ISOLATION_FOREST_PARAMS = {
        "n_estimators": 200,
        "contamination": 0.05,
        "max_samples": "auto",
        "random_state": 42,
    }

    # Random Forest RUL parameters
    RF_RUL_PARAMS = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    }

    # XGBoost RUL parameters
    XGB_RUL_PARAMS = {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "objective": "reg:squarederror",
    }

    # LSTM parameters
    LSTM_PARAMS = {
        "sequence_length": 50,
        "lstm_units_1": 64,
        "lstm_units_2": 32,
        "dense_units": 16,
        "dropout_rate": 0.2,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
    }


class AlertConfig:
    """Configuration for alert levels and notification settings."""

    ALERT_LEVELS = {
        "info": {
            "priority": 0,
            "color": "#17a2b8",
            "icon": "info-circle",
            "description": "Informational - no action required",
        },
        "warning": {
            "priority": 1,
            "color": "#ffc107",
            "icon": "exclamation-triangle",
            "description": "Warning - schedule maintenance within 2 weeks",
        },
        "critical": {
            "priority": 2,
            "color": "#fd7e14",
            "icon": "exclamation-circle",
            "description": "Critical - schedule maintenance within 48 hours",
        },
        "emergency": {
            "priority": 3,
            "color": "#dc3545",
            "icon": "times-circle",
            "description": "Emergency - immediate shutdown and inspection required",
        },
    }

    ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.7"))

    # RUL thresholds for urgency mapping (in hours)
    RUL_THRESHOLDS = {
        "normal": 200,      # > 200 hours remaining
        "warning": 100,     # 100-200 hours remaining
        "critical": 48,     # 48-100 hours remaining
        "immediate": 0,     # < 48 hours remaining
    }

    # Health score thresholds
    HEALTH_THRESHOLDS = {
        "excellent": 90,
        "good": 70,
        "fair": 50,
        "poor": 30,
        "critical": 0,
    }


class DataConfig:
    """Configuration for data storage and simulation parameters."""

    DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    LOG_DIR = BASE_DIR / "logs"

    # Simulation defaults
    DEFAULT_NUM_DEVICES = 10
    DEFAULT_DURATION_HOURS = 720  # 30 days
    DEFAULT_SAMPLING_INTERVAL_SECONDS = 60
    FLEET_DEVICE_PREFIX = "IOT-DEVICE"


class AppConfig:
    """Flask application configuration."""

    SECRET_KEY = os.getenv("SECRET_KEY", "predictive-maintenance-dev-key-2024")
    DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT = int(os.getenv("FLASK_PORT", "5000"))
    POLLING_INTERVAL = int(os.getenv("POLLING_INTERVAL", "5000"))  # milliseconds


class Settings:
    """Aggregated settings for the entire application."""

    sensor = SensorConfig()
    failure = FailureConfig()
    model = ModelConfig()
    alert = AlertConfig()
    data = DataConfig()
    app = AppConfig()

    @classmethod
    def ensure_directories(cls):
        """Create all required directories if they do not exist."""
        dirs = [
            cls.data.DATA_DIR,
            cls.data.RAW_DATA_DIR,
            cls.data.PROCESSED_DATA_DIR,
            cls.data.LOG_DIR,
            cls.model.MODEL_DIR,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
