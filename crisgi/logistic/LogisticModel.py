import torch
import numpy as np
from crisgi.logistic.train import train as train_logistic
from sklearn.linear_model import LogisticRegression
import joblib

class LogisticModel:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load(model_path)

    def train(self, train_loader):
        self.model, metrics = train_logistic(
            train_loader=train_loader,
        )
        return metrics
    
    def predict(self, test_loader):
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        X_test = []
        with torch.no_grad():
            for x, _ in test_loader:
                X_test.append(x.view(x.size(0), -1).cpu().numpy())
        X_test = np.vstack(X_test)
        return self.model.predict(X_test)
    
    def save(self, path):
        if self.model is None:
            raise ValueError("No model to save.")
        torch.save(self.model, path)
        print(f"✅ Logistic model saved to: {path}")

    def load(self, path):
        self.model = torch.load(path,map_location='cpu',weights_only=False)
        print(f"✅ Logistic model loaded from: {path}")
