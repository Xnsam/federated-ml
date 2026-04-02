import flwr as fl
import pickle
import random
import os
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import Metrics, Parameters, Scalar

# --- Configuration ---
MAX_GLOBAL_TREES = 100  # Pruning limit for Raspberry Pi memory
SERVER_ADDR = "0.0.0.0:8080"

class IsolationForestStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        all_trees = []
        
        # 1. Collect trees from all clients (Pi 1 and Pi 2)
        for _, fit_res in results:
            # Deserialize the list of trees sent by the client
            client_trees = pickle.loads(fit_res.parameters.tensors[0])
            all_trees.extend(client_trees)
        
        print(f">>> Round {server_round}: Received {len(all_trees)} total trees from clients.")

        # 2. PRUNING: If we have too many trees, pick a random subset
        # This prevents "tree bloat" on the Raspberry Pis
        if len(all_trees) > MAX_GLOBAL_TREES:
            print(f">>> Pruning forest from {len(all_trees)} down to {MAX_GLOBAL_TREES} trees.")
            pruned_trees = random.sample(all_trees, MAX_GLOBAL_TREES)
        else:
            pruned_trees = all_trees

        # 3. Serialize the pruned Global Forest to send back to the star arms
        parameters_aggregated = fl.common.ndarrays_to_parameters([pickle.dumps(pruned_trees)])
        
        return parameters_aggregated, {}

    def aggregate_evaluate(self, server_round, results, failures):
        # Optional: Aggregate accuracy metrics if clients provide them
        return super().aggregate_evaluate(server_round, results, failures)

# --- Execution ---
if __name__ == "__main__":
    print(f"Starting Federated Server on {SERVER_ADDR}...")
    
    # Define the strategy
    strategy = IsolationForestStrategy(
        min_fit_clients=2,          # Wait for both Pis to connect before starting
        min_available_clients=2,    # Ensure the star topology is complete
    )

    # Start the Flower server
    fl.server.start_server(
        server_address=SERVER_ADDR,
        config=fl.server.ServerConfig(num_rounds=5), # Run 5 rounds of federated learning
        strategy=strategy,
    )
