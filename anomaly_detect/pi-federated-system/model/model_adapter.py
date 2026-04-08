class ModelAdapter:
    def train(self, batch):
        raise NotImplementedError
    
    def predict(self, x):
        raise NotImplementedError
    
    def get_weights(self):
        raise NotImplementedError
    
    def set_weights(self, weights):
        raise NotImplementedError
    
    def save(self, path):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError
    
