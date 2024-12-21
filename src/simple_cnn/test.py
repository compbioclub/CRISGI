import torch
from evalution_metrics import calculate_pred_metric
import numpy as np

def test(ae, mlp, val_loader, AE_loss_function, CE_loss_function, device):
    ae.eval()
    mlp.eval()
    total_loss = 0
    correct = 0
    size = len(val_loader.dataset)
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for x, y, dataset_name in val_loader:
            x, y = x.to(device), y.to(device)
            en, de = ae(x)
            classification_res = mlp(en)

            AE_loss = AE_loss_function(de, x)
            CE_loss = CE_loss_function(classification_res, y)
            loss = AE_loss + CE_loss

            total_loss += loss.item()
            predicted = classification_res.argmax(1)
            correct += (predicted == y).sum().item()

            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / size
    accuracy = 100 * correct / size

    print(f"Total Test Loss: {avg_loss}")
    print(f"Test Accuracy: {accuracy}%")

    metrics = calculate_pred_metric(all_labels, all_predictions)

    return metrics

