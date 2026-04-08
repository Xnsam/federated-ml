import os
from .jax_model import JaxAD

class ModelFactory:

    @staticmethod
    def get_model(model_type: str = "jax_light"):
        input_dim = 160 # 20 window * 8 features

        if model_type == "jax_light":
            return JaxAD(
                input_dim=input_dim,
                latent_dim=32
            )
        else:
            raise NotImplementedError("Model not found.")