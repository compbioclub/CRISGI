from crisgi.logistic.evalution_metrics import calculate_pred_metric
from sklearn.linear_model import LogisticRegression
import numpy as np



def train(train_loader):
    # Prepare data for logistic regression
    X_train = []
    y_train = []

    for x, y in train_loader:
        X_train.append(x.view(x.size(0), -1).cpu().numpy())  # Flatten and convert to numpy
        y_train.extend(y.cpu().numpy())

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    # Define and train logistic regression model
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(X_train, y_train)

    # Predict on training set to evaluate performance
    predictions = logistic_regression.predict(X_train)

    accuracy = (predictions == y_train).mean() * 100
    print(f"Train Accuracy: {accuracy:.2f}%")

    # Calculate additional metrics
    metrics = calculate_pred_metric(y_train, predictions)

    return logistic_regression, metrics