import os
import time
import pickle
import threading
import random
import pandas as pd
import numpy as np
from fastapi import FastAPI
from sklearn.ensemble import IsolationForest
import flwr as fl

# --- Configuration & Paths ---
app = FastAPI(title=f"Federated Pi Client: {os.getenv('CLIENT_NAME', 'Unknown')}")
DATA_PATH = "/app/local_data/sensor.csv"
SERVER_ADDR = os.getenv("SERVER_ADDRESS", "fl-server:8080")

# Global model state
# Initialized with a placeholder; will be replaced by the Global Forest from Server
local_model = IsolationForest(n_estimators=100, contamination=0.1)
is_trained = False

# --- 1. Background Sensor Simulator ---
def sensor_simulator():
    """Simulates a physical sensor writing to a local private CSV."""
    if not os.path.exists("/app/local_data"):
        os.makedirs("/app/local_data")

    while True:
        # 95% Normal data, 5% Anomaly
        is_anomaly = random.random() < 0.05
        if is_anomaly:
            new_row = {
                'temp': round(random.uniform(80.0, 100.0), 2),
                'vibration': round(random.uniform(120.0, 150.0), 2),
                'timestamp': time.time()
            }
        else:
            new_row = {
                'temp': round(random.uniform(22.0, 28.0), 2),
                'vibration': round(random.uniform(45.0, 55.0), 2),
                'timestamp': time.time()
            }

        df = pd.DataFrame([new_row])
        # Data never leaves this container's mapped volume
        df.to_csv(DATA_PATH, mode='a', header=not os.path.exists(DATA_PATH), index=False)
        time.sleep(5)

# --- 2. Flower Federated Client ---
class AnomalyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # We don't send the full model to the server initially
        return []

    def fit(self, parameters, config):
        global local_model
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH).tail(500) # Train on last 500 readings
            features = df[['temp', 'vibration']]
            
            # Local Training
            local_model.fit(features)
            
            # Serialize local trees to send to Server for aggregation
            trees_bytes = pickle.dumps(local_model.estimators_)
            return [trees_bytes], len(df), {}
        return [], 0, {}

    def set_parameters(self, parameters, config):
        global local_model, is_trained
        # Receive PRUNED Global Forest from Server
        global_trees = pickle.loads(parameters[0])
        local_model.estimators_ = global_trees
        local_model.n_estimators = len(global_trees)
        is_trained = True
        print(f"[{os.getenv('CLIENT_NAME')}] Global Model Updated: {len(global_trees)} trees.")

# --- 3. FastAPI Endpoints ---

@app.get("/status")
def get_status():
    """Check if the Pi is online and if the Global Model is synchronized."""
    return {
        "client_name": os.getenv("CLIENT_NAME"),
        "model_ready": is_trained,
        "tree_count": len(local_model.estimators_) if is_trained else 0,
        "data_points_collected": len(pd.read_csv(DATA_PATH)) if os.path.exists(DATA_PATH) else 0
    }

@app.get("/predict/latest")
def predict_latest():
    """Runs anomaly detection on the most recent sensor reading."""
    if not is_trained or not os.path.exists(DATA_PATH):
        return {"error": "Model not ready or no data available"}
    
    df = pd.read_csv(DATA_PATH).tail(1)
    features = df[['temp', 'vibration']]
    
    # 1 = Normal, -1 = Anomaly
    prediction = local_model.predict(features)[0]
    return {
        "data": df.to_dict(orient='records')[0],
        "result": "Normal" if prediction == 1 else "ANOMALY DETECTED"
    }

@app.get("/data/raw")
def get_raw_data(limit: int = 10):
    """View the last few lines of the private local CSV."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH).tail(limit).to_dict(orient='records')
    return {"error": "No data"}

# --- 4. Execution Entrypoint ---
def start_fl_client():
    # Wait for server to be fully ready in the star topology
    time.sleep(5)
    fl.client.start_numpy_client(server_address=SERVER_ADDR, client=AnomalyClient())

if __name__ == "__main__":
    # Start Sensor Simulator
    threading.Thread(target=sensor_simulator, daemon=True).start()
    # Start Flower Client
    threading.Thread(target=start_fl_client, daemon=True).start()
    
    # Run FastAPI (usually handled by uvicorn in Dockerfile)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
