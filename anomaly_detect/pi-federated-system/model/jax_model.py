import json
import os

import jax
import jax.numpy as jnp
from jax import grad, jit

from .model_adapter import ModelAdapter

class JaxAD(ModelAdapter):
    def __init__(self, input_dim=160, latent_dim=32):
        self.key = jax.random.PRNGKey(42)
        self.params = self._init_params(input_dim, latent_dim)

        self._compiled_train_step = self._get_compiled_train_step()
    
    def _init_params(self, in_d, lat_d):
        """
        weight initialization using Xavier initialization
        """
        k1, k2 = jax.random.split(self.key)
        w1 = jax.random.normal(k1, (in_d, lat_d)) * jnp.sqrt(2 / (in_d + lat_d))
        w2 = jax.random.normal(k2, (lat_d, in_d)) * jnp.sqrt(2 / (in_d + lat_d))
        return [w1, w2]
    

    @staticmethod
    @jit
    def _forward(params, x):
        """
        pure function for model inference
        """
        w1, w2 = params
        latent = jax.nn.relu(jnp.dot(x, w1))
        return jnp.dot(latent, w2)
    
    def _get_compiled_train_step(self):
        """
        define and JIT-compile the training update logic
        """

        def loss_fn(p, x):
            recon = self._forward(p, x)
            return jnp.mean(jnp.square(recon - x))
        
        @jit
        def step(params, batch, lr):
            current_loss = loss_fn(params, batch)
            grads = grad(loss_fn)(params, batch)

            new_params = [p - lr * g for p, g in zip(params, grads)]
            return new_params, current_loss
        
        return step
    
    def predict(self, x):
        """
        calculates reconstruction error
        higher values indicate a higher probability of an anomaly
        """
        reconstruction = self._forward(self.params, x)
        return jnp.mean(jnp.square(reconstruction - x))
    
    def train(self, batch, lr=0.006, client_id=None):
        """
        executes the JIT-compiled training step and updates internal weights
        """
        print(f"performing training locally : {client_id}")
        self.param, loss = self._compiled_train_step(
            self.params, batch, lr
        )
        return float(loss)
    
    def get_weights(self):
        return [p.tolist() for p in self.params]
    
    def set_weights(self, weights):
        self.params = [jnp.array(w) for w in weights]
    
    def save(self, path):
        print('Saving the model')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.get_weights(), f)
    
    def load(self, path: str):
        path += "/local_model.json"
        if os.path.exists(path):
            print("loading local model")
            try:
                with open(path, 'r') as f:
                    self.set_weights(json.load(f))
                return True
            except Exception as e:
                print(f"Failed to load weights: {e}")
        return False