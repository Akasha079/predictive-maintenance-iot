import os
import logging
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.data_simulator import SensorDataSimulator
from config.settings import DataConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "iot-maintenance-secret"

# Global simulator
simulator = SensorDataSimulator()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/sim/device/<device_id>")
def simulate_device(device_id):
    """Simulate real-time-like data for a specific device."""
    # We'll generate a small slice for "live" feel
    hours = request.args.get("hours", 24, type=int)
    df = simulator.generate_device_data(device_id, duration_hours=hours)
    
    # Return as list of dicts for the frontend
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/sim/fleet")
def simulate_fleet():
    """Get status summary for the entire fleet."""
    num_devices = request.args.get("num", 10, type=int)
    fleet_df = simulator.generate_fleet_data(num_devices=num_devices)
    
    # Get the latest record for each device
    latest = fleet_df.groupby("device_id").tail(1)
    return jsonify(latest.to_dict(orient="records"))

@app.route("/api/failure_modes")
def get_failure_modes():
    return jsonify(list(simulator.failure_config.FAILURE_NAMES))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
