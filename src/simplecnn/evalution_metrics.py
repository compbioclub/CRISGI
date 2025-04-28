from sklearn.metrics import auc, brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


def calculate_pred_metric(label, pred):
    # label: ground truth
    # pred: prediction

    # Check if both classes are present in the labels
    if len(set(label)) < 2:
        roc_auc = None
        avg_pr_auc = None
        # Set default values for confusion matrix-based metrics
        tn = fp = fn = tp = 0
    else:
        # Calculate ROC AUC
        roc_auc = roc_auc_score(label, pred)

        # Calculate PR curve and its AUC
        precision_vals, recall_vals, _ = precision_recall_curve(label, pred)
        avg_pr_auc = auc(recall_vals, precision_vals)

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(label, pred).ravel()

    # other metrics
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(label, pred) if len(set(label)) == 2 else 0
    kappa = cohen_kappa_score(label, pred) if len(set(label)) == 2 else 0
    brier = brier_score_loss(label, pred)

    # Store the result into dictionary
    res = {
        'AUROC': roc_auc,
        'AUPRC': avg_pr_auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'F1_Score': f1,
        'Kappa': kappa,
        'Brier_Score': brier
    }

    return res