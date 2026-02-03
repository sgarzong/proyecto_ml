import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from furniture_classifier.pipeline import ModelSpec, ClassBalancer


def train_one(model, train_loader, val_loader, criterion, device, epochs=5, lr=5e-4):
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
    best_acc = 0.0
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()

        model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                loss_sum += loss.item() * x.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_loss = loss_sum / max(total, 1)
        val_acc = correct / max(total, 1)
        best_val = min(best_val, val_loss)
        best_acc = max(best_acc, val_acc)
    return best_val, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--arch", default="mobilenet_v3_small")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="outputs/reports/learning_curve")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_tf, va_tf = ModelSpec.default_transforms(args.image_size)
    train_ds = datasets.ImageFolder(Path(args.data_dir) / "train", transform=tr_tf)
    val_ds = datasets.ImageFolder(Path(args.data_dir) / "val", transform=va_tf)

    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    indices = np.arange(len(train_ds))

    weights, counts = ClassBalancer.class_weights(train_ds, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    for frac in tqdm(fractions, desc="Learning curve"):
        np.random.shuffle(indices)
        n = max(1, int(len(indices) * frac))
        subset = Subset(train_ds, indices[:n])
        train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = ModelSpec.build_model(len(train_ds.classes), arch=args.arch, pretrained=True).to(device)
        val_loss, val_acc = train_one(model, train_loader, val_loader, criterion, device, epochs=args.epochs)
        rows.append({"fraction": frac, "val_loss": val_loss, "val_acc": val_acc})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "learning_curve.csv", index=False)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(df["fraction"], df["val_acc"], marker="o")
    plt.xlabel("Train Fraction")
    plt.ylabel("Val Accuracy")
    plt.title("Learning Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve.png", dpi=160)
    plt.close()

    print(f"Learning curve saved in {out_dir}")


if __name__ == "__main__":
    main()
