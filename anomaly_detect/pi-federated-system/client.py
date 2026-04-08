import os
import requests
import time
import json

import numpy as np

from model.factory import ModelFactory

LOCAL_STORAGE = "/app/data/"
model_name = os.getenv("MODEL_TYPE", "jax_light")

def get_realtime_sensor_data(num_features=32):
    return np.random.randn(num_features).astype(np.float32)


# client control logic
def run_client():
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

            if anomaly_score > 0.5:
                print(f" Anomaly detect: {anomaly_score: .4f}")
            
            if len(raw_buffer) >= 120:
                training_batch = np.array(raw_buffer).reshape(-1, 160)

                loss = model.train(training_batch)
                print(f"Loss :: {loss}")
                model.save(LOCAL_STORAGE)
                raw_buffer = raw_buffer[-WINDOW_SIZE:]
        time.sleep(0.01) # 100 hz Pie
    