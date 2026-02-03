import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, average_precision_score)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

from furniture_classifier.pipeline import ModelSpec
from tqdm import tqdm


def _plot_curves(y_true_bin, y_prob, classes, out_path, kind="roc"):
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(classes):
        if kind == "roc":
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=cls)
        else:
            from sklearn.metrics import precision_recall_curve
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            plt.plot(recall, precision, label=cls)
    plt.xlabel("False Positive Rate" if kind == "roc" else "Recall")
    plt.ylabel("True Positive Rate" if kind == "roc" else "Precision")
    plt.title("ROC Curves" if kind == "roc" else "PR Curves")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_spec", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="outputs/reports")
    args = ap.parse_args()

    model, tf, classes = ModelSpec.load(args.model_spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    split_dir = Path(args.data_dir) / args.split
    ds = datasets.ImageFolder(split_dir, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    y_true = []
    y_prob = []
    y_pred = []
    filenames = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Evaluando {args.split}"):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_true.extend(y.numpy().tolist())
            y_prob.append(probs)
            y_pred.extend(preds.tolist())
            filenames.extend([p[0] for p in ds.samples[len(filenames):len(filenames)+len(y)]])

    y_true = np.array(y_true)
    y_prob = np.vstack(y_prob)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    y_true_bin = label_binarize(y_true, classes=list(range(len(classes))))

    roc_auc = float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
    pr_auc = float(average_precision_score(y_true_bin, y_prob, average="macro"))

    out_dir = Path(args.out_dir) / Path(args.model_spec).parent.name / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc_macro_ovr": roc_auc,
        "pr_auc_macro": pr_auc,
        "num_samples": int(len(y_true)),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
    cm_df.to_csv(out_dir / "confusion_matrix.csv", index=True)

    _plot_curves(y_true_bin, y_prob, classes, out_dir / "roc_curves.png", kind="roc")
    _plot_curves(y_true_bin, y_prob, classes, out_dir / "pr_curves.png", kind="pr")

    preds_df = pd.DataFrame({
        "filepath": filenames,
        "true": [classes[i] for i in y_true],
        "pred": [classes[i] for i in y_pred],
        "conf": y_prob.max(axis=1),
    })
    preds_df.to_csv(out_dir / "predictions.csv", index=False)

    print(f"Metrics saved in {out_dir}")


if __name__ == "__main__":
    main()
