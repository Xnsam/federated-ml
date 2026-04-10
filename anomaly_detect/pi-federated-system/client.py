import os
import requests
import time
import json

import numpy as np

from model.factory import ModelFactory

LOCAL_STORAGE = "/app/data/local_model.json"
model_name = os.getenv("MODEL_TYPE", "jax_light")
client_id = os.getenv("CLIENT_ID", "unknown-client")
server_url = os.getenv("SERVER_URL", "http://fed_server:8000")
THRESHOLD = 2

def get_realtime_sensor_data(num_features=8):
    """
    randomised sensor data
    """
    print(f"Fetching data...{client_id}")
    is_glitching = False
    try:
        r = requests.get(
            f"{server_url}/chaos_status/{client_id}", 
            timeout=0.5
        )
        is_glitching = r.json().get("chaos_mode", False)
    except:
        pass

    if is_glitching:
        return np.random.uniform(5, 10, num_features).astype(np.float32)

    return np.random.normal(0, 1, num_features).astype(np.float32)


# client control logic
def run_client():
    print('Running model')
    model = ModelFactory.get_model(model_name)
    model.load(LOCAL_STORAGE)

    raw_buffer = []
    WINDOW_SIZE = 20

    while True:
        reading = get_realtime_sensor_data()
        raw_buffer.append(reading)

        if len(raw_buffer) >= WINDOW_SIZE:

            current_window = np.array(
                raw_buffer[-WINDOW_SIZE:]).flatten().reshape(1, -1)
            
            anomaly_score = model.predict(current_window)

            if anomaly_score > THRESHOLD:
                print(f" Anomaly detect: {anomaly_score: .4f}")
            else:
                print(f"No Anomaly detected :: System Stable :: {anomaly_score}")
            
            if len(raw_buffer) >= 120:
                training_batch = np.array(raw_buffer).reshape(-1, 160)

                loss = model.train(training_batch, client_id=client_id)
                print(f"Loss :: {loss}")
                model.save(LOCAL_STORAGE)
                raw_buffer = raw_buffer[-WINDOW_SIZE:]
        time.sleep(0.01) # 100 hz Pie

if __name__ == "__main__":
    try:
        print(f"Running Client {client_id}")
        run_client()
    except Exception as e:
        print(f"Client Crashed: {e}")
        time.sleep(60)