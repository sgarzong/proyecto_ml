import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from scipy.stats import chi2
from tqdm import tqdm

from furniture_classifier.pipeline import ModelSpec


def predict(model, loader, device):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Predicciones"):
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            y_true.extend(y.numpy().tolist())
            y_pred.extend(preds.tolist())
    return np.array(y_true), np.array(y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True)
    ap.add_argument("--model_b", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="outputs/reports/model_compare")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a, tf, classes = ModelSpec.load(args.model_a)
    model_b, tf2, classes_b = ModelSpec.load(args.model_b)
    if classes != classes_b:
        raise SystemExit("Model classes do not match.")

    ds = datasets.ImageFolder(Path(args.data_dir) / args.split, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model_a.to(device)
    model_b.to(device)

    y_true, y_pred_a = predict(model_a, loader, device)
    _, y_pred_b = predict(model_b, loader, device)

    # McNemar test (chi-square with continuity correction)
    both_correct = (y_pred_a == y_true) & (y_pred_b == y_true)
    a_correct_b_wrong = (y_pred_a == y_true) & (y_pred_b != y_true)
    b_correct_a_wrong = (y_pred_a != y_true) & (y_pred_b == y_true)
    both_wrong = (y_pred_a != y_true) & (y_pred_b != y_true)

    table = np.array([
        [both_correct.sum(), a_correct_b_wrong.sum()],
        [b_correct_a_wrong.sum(), both_wrong.sum()],
    ])

    n01 = table[1, 0]
    n10 = table[0, 1]
    if (n01 + n10) == 0:
        stat = 0.0
        pval = 1.0
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        pval = 1.0 - chi2.cdf(stat, df=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table, index=["A_correct", "A_wrong"], columns=["B_correct", "B_wrong"]).to_csv(out_dir / "mcnemar_table.csv")

    report = {
        "n_samples": int(len(y_true)),
        "mcnemar_stat": float(stat),
        "mcnemar_pvalue": float(pval),
    }
    with open(out_dir / "mcnemar_report.json", "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, indent=2)

    print(f"McNemar test saved in {out_dir}")


if __name__ == "__main__":
    main()
