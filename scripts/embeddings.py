import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False


def get_feature_extractor(arch: str):
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat = torch.nn.Sequential(*list(m.children())[:-1])
        out_dim = m.fc.in_features
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        feat = torch.nn.Sequential(m.features, torch.nn.AdaptiveAvgPool2d(1))
        out_dim = m.classifier[-1].in_features
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat = torch.nn.Sequential(m.features, torch.nn.AdaptiveAvgPool2d(1))
        out_dim = m.classifier[-1].in_features
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m, feat, out_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_images", type=int, default=2000)
    ap.add_argument("--dataset_master", default="data/meta/dataset_master.csv")
    ap.add_argument("--excel", default="data/raw/base_muebles.xlsx")
    ap.add_argument("--url_col", default="LINK GITHUB")
    ap.add_argument("--label_col", default="MUEBLES")
    ap.add_argument("--chain_col", default="CADENA")
    ap.add_argument("--brand_col", default="MARCA")
    ap.add_argument("--out_dir", default="outputs/reports/embeddings")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feat, _ = get_feature_extractor(args.arch)
    model.to(device)
    feat.to(device)
    feat.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = datasets.ImageFolder(Path(args.data_dir) / args.split, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    embeddings = []
    labels = []
    filepaths = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extrayendo embeddings"):
            x = x.to(device)
            z = feat(x)
            z = z.view(z.size(0), -1).cpu().numpy()
            embeddings.append(z)
            labels.extend(y.numpy().tolist())
            filepaths.extend([p[0] for p in ds.samples[len(filepaths):len(filepaths)+len(y)]])
            if len(filepaths) >= args.max_images:
                break

    emb = np.vstack(embeddings)[:args.max_images]
    labels = labels[:args.max_images]
    filepaths = filepaths[:args.max_images]

    df_master = pd.read_csv(args.dataset_master)
    df_excel = pd.read_excel(args.excel)
    df_excel = df_excel[[args.url_col, args.chain_col, args.brand_col]].rename(
        columns={args.url_col: "url", args.chain_col: "chain", args.brand_col: "brand"}
    )
    df = pd.DataFrame({"filepath": filepaths, "label_idx": labels})
    df = df.merge(df_master[["filepath", "url", "label"]], on="filepath", how="left")
    df = df.merge(df_excel, on="url", how="left")

    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        proj = reducer.fit_transform(emb)
        method = "umap"
    else:
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(emb)
        method = "tsne"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df["x"] = proj[:, 0]
    df["y"] = proj[:, 1]
    df.to_csv(out_dir / f"embeddings_{method}.csv", index=False)

    for col in ["label", "chain", "brand"]:
        plt.figure(figsize=(7, 6))
        for name, g in df.groupby(col):
            plt.scatter(g["x"], g["y"], s=6, alpha=0.6, label=str(name))
        plt.title(f"{method.upper()} by {col}")
        plt.legend(markerscale=2, fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"{method}_{col}.png", dpi=160)
        plt.close()

    print(f"Embedding projections saved in {out_dir}")


if __name__ == "__main__":
    main()
