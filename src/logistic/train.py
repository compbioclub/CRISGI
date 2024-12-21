from evalution_metrics import calculate_pred_metric
from sklearn.linear_model import LogisticRegression
import numpy as np


def train(ae, mlp, train_loader, AE_loss_function, CE_loss_function, optimizer, device):
    X_train = []
    y_train = []

    for x, y, dataset_name in train_loader:
        X_train.append(x.view(x.size(0), -1).cpu().numpy())
        y_train.extend(y.cpu().numpy())

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(X_train, y_train)

    predictions = logistic_regression.predict(X_train)

    accuracy = (predictions == y_train).mean() * 100
    print(f"Train Accuracy: {accuracy:.2f}%")

    metrics = calculate_pred_metric(y_train, predictions)

    return logistic_regression, metrics