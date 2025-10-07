import torch
import torch.nn.functional as F
from evaluate import evaluate
def train_model(model, optimizer, train_loader, val_loader, test_loader, target_idx=16, num_epochs=30):
    best_val_auc = 0.0
    device = next(model.parameters()).device
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for orig, masked in train_loader:
            orig = orig.long().to(device)
            masked = masked.long().to(device)

            logits = model(masked)
            logits_target = logits[:, target_idx]
            targets = orig[:, target_idx]

            if logits_target.numel() > 0:
                loss = F.cross_entropy(logits_target, targets)
            else:
                loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # --- Evaluate on validation and test sets ---
        val_loss, val_acc, val_bal_acc, val_auc, val_cm = evaluate(model, val_loader, target_idx)
        test_loss, test_acc, test_bal_acc, test_auc,test_cm= evaluate(model, test_loader, target_idx)

        # --- Save model if it improves validation AUC ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_infill_model_B.pth")
            print("✅ Saved new best model (AUC improved)")

        # --- Logging ---
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val BalAcc: {val_bal_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Test AUC: {test_auc:.4f} | "
              f"Test BalAcc: {test_bal_acc:.4f}")
        print("Test Confusion Matrix:\n", test_cm[:2, :2])
        print("-" * 90)

    print("Training complete.")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    return model
