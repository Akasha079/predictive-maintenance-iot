"""
IoT Sensor Data Simulator for Predictive Maintenance.

Generates synthetic time-series data that mimics real industrial IoT sensors,
including normal operation, gradual degradation, and various failure modes.
"""

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SensorConfig, FailureConfig, DataConfig

logger = logging.getLogger(__name__)


class SensorDataSimulator:
    """Simulates realistic IoT sensor data with degradation and failure patterns."""

    def __init__(self, seed: int = 42):
        """
        Initialize the simulator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)
        self.sensor_config = SensorConfig()
        self.failure_config = FailureConfig()
        self.data_config = DataConfig()

    def generate_device_id(self, index: int) -> str:
        """Generate a unique device identifier."""
        return f"{self.data_config.FLEET_DEVICE_PREFIX}-{index:04d}"

    def _generate_normal_readings(
        self, sensor_type: str, num_samples: int
    ) -> np.ndarray:
        """
        Generate normal operating sensor readings with realistic noise.

        Args:
            sensor_type: Type of sensor from SensorConfig.
            num_samples: Number of data points to generate.

        Returns:
            Array of sensor readings under normal conditions.
        """
        config = self.sensor_config.SENSOR_TYPES[sensor_type]
        low, high = config["normal_range"]
        center = (low + high) / 2
        amplitude = (high - low) / 2

        # Base signal with slight sinusoidal pattern (operational cycles)
        t = np.linspace(0, num_samples / 60, num_samples)
        cycle_period = self.rng.uniform(4, 12)  # hours
        base_signal = center + amplitude * 0.3 * np.sin(2 * np.pi * t / cycle_period)

        # Add Gaussian noise
        noise = self.rng.normal(0, config["noise_std"], num_samples)

        # Add occasional small transients
        transient_mask = self.rng.random(num_samples) < 0.005
        transients = transient_mask * self.rng.normal(0, config["noise_std"] * 3, num_samples)

        readings = base_signal + noise + transients
        return np.clip(readings, low * 0.8, high * 1.2)

    def _apply_degradation(
        self,
        readings: dict,
        failure_mode: str,
        num_samples: int,
        onset_idx: int,
    ) -> dict:
        """
        Apply degradation pattern to sensor readings based on failure mode.

        Args:
            readings: Dictionary of sensor arrays keyed by sensor type.
            failure_mode: The failure mode to simulate.
            num_samples: Total number of samples.
            onset_idx: Index where degradation begins.

        Returns:
            Modified readings dictionary with degradation applied.
        """
        failure_cfg = self.failure_config.FAILURE_MODES[failure_mode]
        degradation_length = num_samples - onset_idx

        # Create degradation curve (exponential ramp)
        degradation_curve = np.zeros(num_samples)
        if degradation_length > 0:
            t = np.linspace(0, 1, degradation_length)
            degradation_curve[onset_idx:] = np.exp(3 * t) - 1
            degradation_curve[onset_idx:] /= degradation_curve[-1]  # Normalize to [0, 1]

        if failure_mode == "bearing_failure":
            if "vibration" in readings:
                multiplier = failure_cfg["vibration_multiplier"]
                config = self.sensor_config.SENSOR_TYPES["vibration"]
                center = sum(config["normal_range"]) / 2
                readings["vibration"] += degradation_curve * center * (multiplier - 1)
                # Add high-frequency spikes in late degradation
                spike_region = degradation_curve > 0.7
                spike_noise = spike_region * self.rng.exponential(2.0, num_samples)
                readings["vibration"] += spike_noise

            if "temperature" in readings:
                readings["temperature"] += degradation_curve * failure_cfg["temperature_offset"]

            if "current" in readings:
                readings["current"] *= 1 + degradation_curve * (failure_cfg["current_multiplier"] - 1)

        elif failure_mode == "overheating":
            if "temperature" in readings:
                config = self.sensor_config.SENSOR_TYPES["temperature"]
                center = sum(config["normal_range"]) / 2
                readings["temperature"] *= 1 + degradation_curve * (failure_cfg["temperature_multiplier"] - 1)

            if "current" in readings:
                readings["current"] *= 1 + degradation_curve * (failure_cfg["current_multiplier"] - 1)

            if "voltage" in readings:
                readings["voltage"] -= degradation_curve * failure_cfg["voltage_drop"]

        elif failure_mode == "electrical_fault":
            if "voltage" in readings:
                variance_mult = failure_cfg["voltage_variance_multiplier"]
                voltage_noise = self.rng.normal(0, 5, num_samples) * degradation_curve * variance_mult
                readings["voltage"] += voltage_noise

            if "current" in readings:
                spike_prob = failure_cfg["current_spike_probability"]
                for i in range(onset_idx, num_samples):
                    if self.rng.random() < spike_prob * degradation_curve[i]:
                        spike_duration = self.rng.randint(1, 5)
                        spike_magnitude = self.rng.uniform(5, 15)
                        end_idx = min(i + spike_duration, num_samples)
                        readings["current"][i:end_idx] += spike_magnitude

            if "temperature" in readings:
                readings["temperature"] += degradation_curve * failure_cfg["temperature_offset"]

        elif failure_mode == "mechanical_wear":
            if "vibration" in readings:
                multiplier = failure_cfg["vibration_multiplier"]
                config = self.sensor_config.SENSOR_TYPES["vibration"]
                center = sum(config["normal_range"]) / 2
                readings["vibration"] += degradation_curve * center * (multiplier - 1)

            if "rpm" in readings:
                variance_mult = failure_cfg["rpm_variance_multiplier"]
                rpm_noise = self.rng.normal(0, 20, num_samples) * degradation_curve * variance_mult
                readings["rpm"] += rpm_noise

            if "pressure" in readings:
                readings["pressure"] -= degradation_curve * failure_cfg["pressure_drop"]

            if "current" in readings:
                readings["current"] *= 1 + degradation_curve * (failure_cfg["current_multiplier"] - 1)

        return readings

    def _compute_health_status(
        self, readings: dict, degradation_progress: np.ndarray
    ) -> np.ndarray:
        """
        Compute health status labels based on degradation progress.

        Args:
            readings: Dictionary of sensor readings.
            degradation_progress: Array of degradation level [0, 1].

        Returns:
            Array of health status labels.
        """
        num_samples = len(degradation_progress)
        status = np.full(num_samples, "normal", dtype=object)
        status[degradation_progress > 0.3] = "degrading"
        status[degradation_progress > 0.6] = "warning"
        status[degradation_progress > 0.85] = "critical"
        status[degradation_progress > 0.95] = "failure"
        return status

    def generate_device_data(
        self,
        device_id: str,
        duration_hours: int = 720,
        sampling_interval_seconds: int = 60,
        failure_mode: Optional[str] = None,
        include_failure: bool = True,
    ) -> pd.DataFrame:
        """
        Generate complete sensor data for a single device.

        Args:
            device_id: Unique device identifier.
            duration_hours: Total simulation duration in hours.
            sampling_interval_seconds: Time between readings in seconds.
            failure_mode: Specific failure mode; random if None.
            include_failure: Whether to include degradation/failure.

        Returns:
            DataFrame with timestamps, sensor readings, and health labels.
        """
        num_samples = int(duration_hours * 3600 / sampling_interval_seconds)
        start_time = datetime.now() - timedelta(hours=duration_hours)
        timestamps = [
            start_time + timedelta(seconds=i * sampling_interval_seconds)
            for i in range(num_samples)
        ]

        # Generate normal readings for all sensors
        readings = {}
        for sensor_type in self.sensor_config.SENSOR_NAMES:
            readings[sensor_type] = self._generate_normal_readings(sensor_type, num_samples)

        # Degradation progress tracker
        degradation_progress = np.zeros(num_samples)

        if include_failure:
            if failure_mode is None:
                failure_mode = self.rng.choice(self.failure_config.FAILURE_NAMES)

            failure_cfg = self.failure_config.FAILURE_MODES[failure_mode]
            onset_pct = failure_cfg["onset_percentage"]
            onset_idx = int(num_samples * onset_pct)

            # Apply degradation
            readings = self._apply_degradation(readings, failure_mode, num_samples, onset_idx)

            # Compute degradation progress
            degradation_length = num_samples - onset_idx
            if degradation_length > 0:
                t = np.linspace(0, 1, degradation_length)
                degradation_progress[onset_idx:] = t

            # Compute RUL (Remaining Useful Life) in hours
            rul = np.zeros(num_samples)
            total_failure_hours = duration_hours * (1 - onset_pct)
            for i in range(num_samples):
                if i < onset_idx:
                    rul[i] = total_failure_hours + (onset_idx - i) * sampling_interval_seconds / 3600
                else:
                    remaining_samples = num_samples - i
                    rul[i] = max(0, remaining_samples * sampling_interval_seconds / 3600)
        else:
            failure_mode = "none"
            rul = np.full(num_samples, duration_hours)

        # Compute health status
        health_status = self._compute_health_status(readings, degradation_progress)

        # Build DataFrame
        data = {
            "device_id": device_id,
            "timestamp": timestamps,
            "failure_mode": failure_mode,
        }
        for sensor_type in self.sensor_config.SENSOR_NAMES:
            data[sensor_type] = np.round(readings[sensor_type], 4)

        data["health_status"] = health_status
        data["degradation_progress"] = np.round(degradation_progress, 4)
        data["rul_hours"] = np.round(rul, 2)

        df = pd.DataFrame(data)
        logger.info(
            "Generated %d samples for device %s (failure_mode=%s)",
            num_samples, device_id, failure_mode,
        )
        return df

    def generate_fleet_data(
        self,
        num_devices: int = 10,
        duration_hours: int = 720,
        sampling_interval_seconds: int = 60,
        failure_ratio: float = 0.7,
    ) -> pd.DataFrame:
        """
        Generate sensor data for an entire fleet of devices.

        Args:
            num_devices: Number of devices in the fleet.
            duration_hours: Simulation duration per device.
            sampling_interval_seconds: Time between readings.
            failure_ratio: Fraction of devices that will experience failures.

        Returns:
            Combined DataFrame for all devices.
        """
        all_data = []
        num_failing = int(num_devices * failure_ratio)

        for i in range(num_devices):
            device_id = self.generate_device_id(i + 1)
            include_failure = i < num_failing
            failure_mode = None
            if include_failure:
                failure_mode = self.failure_config.FAILURE_NAMES[
                    i % len(self.failure_config.FAILURE_NAMES)
                ]

            df = self.generate_device_data(
                device_id=device_id,
                duration_hours=duration_hours,
                sampling_interval_seconds=sampling_interval_seconds,
                failure_mode=failure_mode,
                include_failure=include_failure,
            )
            all_data.append(df)

        fleet_df = pd.concat(all_data, ignore_index=True)
        logger.info(
            "Generated fleet data: %d devices, %d total records",
            num_devices, len(fleet_df),
        )
        return fleet_df

    def save_data(self, df: pd.DataFrame, filename: str = "sensor_data.csv") -> Path:
        """
        Save generated data to CSV.

        Args:
            df: DataFrame to save.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        self.data_config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = self.data_config.RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info("Saved sensor data to %s (%d records)", filepath, len(df))
        return filepath
