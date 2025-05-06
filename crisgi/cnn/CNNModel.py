import sys
import torch
from crisgi.cnn.autoEncoder import AE
import crisgi.cnn.autoEncoder
from crisgi.cnn.mlp import MLP
import crisgi.cnn.mlp
from crisgi.cnn.train import train as train_loop

sys.modules['autoEncoder'] = crisgi.cnn.autoEncoder
sys.modules['mlp'] = crisgi.cnn.mlp


class CNNModel:
    def __init__(self, device, ae_path=None, mlp_path=None):
        self.device = device
        self.ae = AE().to(device)
        self.mlp = MLP().to(device)
        self.ae_loss_fn = torch.nn.MSELoss().to(device)
        self.ce_loss_fn = torch.nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.Adam(
            list(self.ae.parameters()) + list(self.mlp.parameters()), 
            lr=0.001, weight_decay=1e-4
        )

        if ae_path and mlp_path:
            print(f"Loading pretrained models:\n - AE: {ae_path}\n - MLP: {mlp_path}")
            self.ae = torch.load(ae_path, map_location=device, weights_only=False)
            self.mlp = torch.load(mlp_path, map_location=device, weights_only=False)


    def train(self, train_loader,epochs=10):
        final_metrics = None
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            final_metrics = train_loop(
                self.ae,
                self.mlp,
                train_loader,
                self.ae_loss_fn,
                self.ce_loss_fn,
                self.optimizer,
                self.device
            )
        return final_metrics

    def predict(self, data_loader):
        self.ae.eval()
        self.mlp.eval()
        all_predictions = []

        with torch.no_grad():
            for x, _ in data_loader:  
                x = x.to(self.device)
                en, _ = self.ae(x)
                out = self.mlp(en)
                predicted = out.argmax(dim=1)
                all_predictions.extend(predicted.cpu().numpy())

        return all_predictions
    
    def save(self, ae_path, mlp_path):
        torch.save(self.ae.state_dict(), ae_path)
        torch.save(self.mlp.state_dict(), mlp_path)
        print(f"Models saved to:\n - {ae_path}\n - {mlp_path}")
