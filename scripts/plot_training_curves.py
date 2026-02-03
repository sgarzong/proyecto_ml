import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--out_dir", default="outputs/reports")
    args = ap.parse_args()

    hist = pd.read_csv(args.history)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(hist["epoch"], hist["train_acc"], label="train_acc")
    plt.plot(hist["epoch"], hist["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(hist["epoch"], hist["train_loss"] - hist["val_loss"], label="gap_loss")
    plt.plot(hist["epoch"], hist["train_acc"] - hist["val_acc"], label="gap_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Gap (train - val)")
    plt.title("Bias-Variance Proxy (Gaps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bias_variance_gap.png", dpi=160)
    plt.close()

    print(f"Saved curves in {out_dir}")


if __name__ == "__main__":
    main()
