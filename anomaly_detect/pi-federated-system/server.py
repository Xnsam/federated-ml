import os
import uvicorn
import numpy as np
from fastapi import FastAPI, Body
from typing import List, Optional, Dict
import json


app = FastAPI(title="Federated Aggregator Server")

global_weights: Optional[List[List[float]]] = None
client_chaos_registry: Dict[str, bool] = {}
updates_buffer = []

MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", 3))
STORAGE_PATH = "/app/data/global_model.json"


def save_weights_to_disk(weights):
    """
    Saves the global model to a JSON file.
    """
    try:
        with open(STORAGE_PATH, 'w') as f:
            json.dump(weights, f)
        
        print(f"Back up saved to {STORAGE_PATH}")
    except Exception as e:
        print(f"Backup failed: {e}")

def load_weights_from_disk():
    """
    loads weights on start up if they exist
    """
    if os.path.exists(STORAGE_PATH):
        try:
            with open(STORAGE_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def aggregate_and_update():
    """
    Performs federated averaging across all buffered client updates.
    """
    global global_weights, updates_buffer

    print("--- Starting Aggregation Round ---")

    new_weights = []
    num_layers = len(updates_buffer[0])

    for i in range(num_layers):
        layer_average = np.mean([client[i] for client in updates_buffer], axis=0)
        new_weights.append(layer_average.tolis())
    
    global_weights = new_weights
    updates_buffer = []
    print("---- Global model updated successfully --- ")
 

@app.get("/status")
def get_status():
    return {
        "status": "online",
        "updates_in_buffer": len(updates_buffer), 
        "required_for_aggregation": MIN_CLIENTS,
        "global_model_ready": global_weights is not None
    }

@app.post("/push_update")
async def push_update(payload: dict = Body(...)):
    global updates_buffer

    client_weights = [np.array(w) for w in payload["weights"]]
    updates_buffer.append(client_weights)

    print(f"Received update from {payload.get('client_id', 'unknown')}. Buffer: {len(updates_buffer)}/{MIN_CLIENTS}")

    if len(updates_buffer) >= MIN_CLIENTS:
        aggregate_and_update()
    return {"message": "Update received"}

@app.get("/pull_global")
async def pull_global():
    """
    Endpoint for clients to download the latest global model
    """
    if global_weights is None:
        return {"weights": None, "info": "Model not initialized yet"}
    
    return {"weights": global_weights}

@app.get("/trigger_anomaly/{client_id}")
async def trigger_client_anomaly(client_id: str, status: bool):
    """
    Sets the anomaly status for a specific client.
    """
    client_chaos_registry[client_id] = status
    print(f"Chaos registry : ", client_chaos_registry)
    state = "ENABLED" if status else "DISABLED"
    return {"message": f"Chaos mode {state} for {client_id}"}

@app.get("/chaos_status/{client_id}")
async def get_client_status(client_id: str):
    """
    Endpoint for clients to poll their specific status.
    """
    return {"chaos_mode": client_chaos_registry.get(client_id, False)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
