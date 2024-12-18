from evalution_metrics import calculate_pred_metric


def train(ae, mlp, train_loader, AE_loss_function, CE_loss_function, optimizer, device):
    ae.train()
    mlp.train()
    total_loss = 0
    correct = 0
    size = len(train_loader.dataset)
    all_predictions = []
    all_labels = []

    for x, y, dataset_name in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        en, de = ae(x)
        classification_res = mlp(en)

        AE_loss = AE_loss_function(de, x)
        CE_loss = CE_loss_function(classification_res, y)
        loss = AE_loss + CE_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = classification_res.argmax(1)
        correct += (predicted == y).sum().item()

        all_labels.extend(y.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / size
    accuracy = 100 * correct / size

    print(f"Total Train Loss: {avg_loss}")
    print(f"Train Accuracy: {accuracy}%")

    metrics = calculate_pred_metric(all_labels, all_predictions)

    return metrics