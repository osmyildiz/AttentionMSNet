"""
Shared training utilities used by all step scripts.
Keeps training logic DRY - change once, applies everywhere.
"""
import os, sys, time, copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import compute_metrics, print_results


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, f1_score(all_labels, all_preds, average="macro")


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    running_loss, total = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in tqdm(loader, desc="  Eval", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        probs = torch.softmax(outputs, dim=1)
        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds),
                               y_prob=np.array(all_probs), class_names=class_names)
    metrics["loss"] = running_loss / total
    return metrics, np.array(all_probs)


def train_model(model, name, train_loader, val_loader, criterion, device,
                class_names, num_epochs=50, lr=3e-4, weight_decay=1e-3,
                patience=10, log_dir="logs"):
    """
    Standard training loop. Same for ALL experiments.
    Returns: model (best), history dict, best_f1
    """
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_state, patience_counter = 0.0, None, 0

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Training: {name}")
    print(f"  Epochs: {num_epochs}, LR: {lr}, Train size: {len(train_loader.dataset):,}")
    print(sep)
    start = time.time()

    for epoch in range(num_epochs):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _ = evaluate(model, val_loader, criterion, device, class_names)
        va_loss, va_acc, va_f1 = val_metrics["loss"], val_metrics["accuracy"], val_metrics["f1_macro"]
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        marker = ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " *BEST*"
        else:
            patience_counter += 1

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:>2}/{num_epochs}  "
              f"TrL:{tr_loss:.4f} TrA:{tr_acc:.4f} TrF1:{tr_f1:.4f} | "
              f"VaL:{va_loss:.4f} VaA:{va_acc:.4f} VaF1:{va_f1:.4f} "
              f"LR:{lr_now:.2e}{marker}")

        if patience_counter >= patience:
            print(f"  >> Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start
    print(f"  Time: {elapsed/60:.1f} min | Best val F1: {best_f1:.4f}")

    model.load_state_dict(best_state)

    safe_name = name.lower().replace("-", "_").replace("+", "_").replace(" ", "_").replace("(", "").replace(")", "")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{safe_name}_history.json"), "w") as f:
        json.dump(history, f)

    return model, history, best_f1
