import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
import numpy as np

def evaluate(model, loader, target_col_idx):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    n_batches = 0

    total_correct = 0
    total_samples = 0

    probA_all = []
    orig_all = []
    pred_all = []

    with torch.no_grad():
        for orig, masked in loader:
            orig = orig.long().to(device)       # [B, L]
            masked = masked.long().to(device)   # [B, L]

            logits = model(masked)              # [B, L, C]
            probs = F.softmax(logits, dim=-1)   # [B, L, C]
            preds = logits.argmax(dim=-1)       # [B, L]

            # --- Focus only on last element (target_col_idx) ---
            logits_target = logits[:, target_col_idx]     # [B, C]
            targets = orig[:, target_col_idx]             # [B]

            loss = F.cross_entropy(logits_target, targets)
            total_loss += loss.item()
            n_batches += 1

            pred_labels = preds[:, target_col_idx]
            total_correct += (pred_labels == targets).sum().item()
            total_samples += targets.numel()

            # Collect predictions and probabilities for metrics
            probA_all.extend(probs[:, target_col_idx, 1].cpu().numpy())
            orig_all.extend(targets.cpu().numpy())
            pred_all.extend(pred_labels.cpu().numpy())

    # --- Compute metrics ---
    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_samples, 1)

    y_true = np.array(orig_all)
    y_pred = np.array(pred_all)
    y_prob = np.array(probA_all)

    auc = roc_auc_score(y_true, y_prob)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return avg_loss, accuracy, bal_acc, auc, cm