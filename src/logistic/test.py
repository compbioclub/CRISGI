import torch
from evalution_metrics import calculate_pred_metric
import numpy as np


def test(logistic_regression, val_loader):
    X_test = []
    y_test = []

    with torch.no_grad():
        for x, y, dataset_name in val_loader:
            X_test.append(x.view(x.size(0), -1).cpu().numpy())
            y_test.extend(y.cpu().numpy())

    X_test = np.vstack(X_test)
    y_test = np.array(y_test)

    predictions = logistic_regression.predict(X_test)
    accuracy = (predictions == y_test).mean() * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    metrics = calculate_pred_metric(y_test, predictions)

    return metrics