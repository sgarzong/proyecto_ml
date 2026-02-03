import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def safe_name(s: str) -> str:
    return (s or "").strip().replace(" ", "_")


def load_excel(path, url_col, label_col, chain_col, store_col, brand_col):
    df = pd.read_excel(path)
    cols = {url_col: "url", label_col: "label", chain_col: "chain", store_col: "store_id", brand_col: "brand"}
    out = df[list(cols.keys())].rename(columns=cols)
    for c in ["label", "chain", "store_id", "brand"]:
        out[c] = out[c].astype(str)
    out["url"] = out["url"].astype(str)
    return out


def load_master(path):
    df = pd.read_csv(path)
    if "url" not in df.columns:
        df["url"] = ""
    return df


def a_hash(img, size=8):
    im = img.convert("L").resize((size, size))
    arr = np.array(im)
    mean = arr.mean()
    bits = (arr > mean).astype(np.uint8).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming(a, b):
    return (a ^ b).bit_count()


def save_grid(image_paths, title, out_path, ncols=5, max_images=25):
    paths = image_paths[:max_images]
    n = len(paths)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(ncols * 2.2, nrows * 2.2))
    for i, p in enumerate(paths, 1):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        ax = plt.subplot(nrows, ncols, i)
        ax.imshow(img)
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--dataset_master", default="data/meta/dataset_master.csv")
    ap.add_argument("--url_col", default="LINK GITHUB")
    ap.add_argument("--label_col", default="MUEBLES")
    ap.add_argument("--chain_col", default="CADENA")
    ap.add_argument("--store_col", default="COD LOCAL/Ales")
    ap.add_argument("--brand_col", default="MARCA")
    ap.add_argument("--out_dir", default="outputs/reports/eda")
    ap.add_argument("--near_dup_max", type=int, default=5000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_excel = load_excel(args.excel, args.url_col, args.label_col, args.chain_col, args.store_col, args.brand_col)
    df_master = load_master(args.dataset_master)

    df = df_master.merge(df_excel, on="url", how="left")
    if "label" not in df.columns:
        if "label_x" in df.columns:
            df["label"] = df["label_x"]
        elif "label_y" in df.columns:
            df["label"] = df["label_y"]

    # 1. Dataset structure & balance
    total_images = len(df)
    class_counts = df["label"].value_counts().rename("count")
    chain_counts = df["chain"].value_counts().rename("count")
    store_counts = df["store_id"].value_counts().rename("count")

    class_counts.to_csv(out_dir / "class_distribution.csv")
    chain_counts.to_csv(out_dir / "chain_distribution.csv")
    store_counts.to_csv(out_dir / "store_distribution.csv")

    # Dominance analysis
    def dominance_table(group_col):
        rows = []
        for cls, g in df.groupby("label"):
            total = len(g)
            vc = g[group_col].value_counts()
            top_val = vc.index[0] if len(vc) else ""
            top_share = float(vc.iloc[0] / total) if total else 0.0
            rows.append({"label": cls, f"top_{group_col}": top_val, f"top_{group_col}_share": top_share, f"unique_{group_col}": vc.size})
        return pd.DataFrame(rows)

    dom_chain = dominance_table("chain")
    dom_brand = dominance_table("brand")
    dom_store = dominance_table("store_id")
    dom_chain.to_csv(out_dir / "dominance_chain.csv", index=False)
    dom_brand.to_csv(out_dir / "dominance_brand.csv", index=False)
    dom_store.to_csv(out_dir / "dominance_store.csv", index=False)

    # Flag imbalance & concentration
    imbalance_flags = class_counts[class_counts < class_counts.mean() * 0.3]
    conc_flags = dom_chain[dom_chain["top_chain_share"] > 0.7]
    imbalance_flags.to_csv(out_dir / "flags_class_imbalance.csv")
    conc_flags.to_csv(out_dir / "flags_chain_concentration.csv", index=False)

    # 2. Cross-feature bias analysis
    for feature in ["chain", "brand", "store_id"]:
        ct = pd.crosstab(df["label"], df[feature], normalize="index")
        ct.to_csv(out_dir / f"crosstab_label_{feature}.csv")

    # 3. Visual inspection (stratified)
    grids_dir = out_dir / "grids"
    grids_dir.mkdir(exist_ok=True)

    for cls, g in tqdm(list(df.groupby("label")), desc="Grids por clase"):
        paths = g["filepath"].dropna().tolist()
        save_grid(paths, f"Label: {cls}", grids_dir / f"grid_label_{safe_name(cls)}.png")

        # by top chains within class
        top_chains = g["chain"].value_counts().head(2).index.tolist()
        for ch in top_chains:
            paths_ch = g[g["chain"] == ch]["filepath"].dropna().tolist()
            save_grid(paths_ch, f"{cls} | Chain: {ch}", grids_dir / f"grid_{safe_name(cls)}_chain_{safe_name(ch)}.png")

        # by top brands within class
        top_brands = g["brand"].value_counts().head(2).index.tolist()
        for br in top_brands:
            paths_br = g[g["brand"] == br]["filepath"].dropna().tolist()
            save_grid(paths_br, f"{cls} | Brand: {br}", grids_dir / f"grid_{safe_name(cls)}_brand_{safe_name(br)}.png")

    # 4. Image geometry & quality analysis
    stats = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Stats de imágenes"):
        p = r.get("filepath", "")
        if not p or not Path(p).exists():
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        arr = np.asarray(img).astype(np.float32) / 255.0
        gray = arr.mean(axis=2)
        bright = float(gray.mean())
        contrast = float(gray.std())
        sx = sobel(gray, axis=0)
        sy = sobel(gray, axis=1)
        edge = np.hypot(sx, sy)
        edge_density = float((edge > np.percentile(edge, 75)).mean())
        stats.append({
            "filepath": p,
            "label": r.get("label", ""),
            "chain": r.get("chain", ""),
            "store_id": r.get("store_id", ""),
            "brand": r.get("brand", ""),
            "width": w,
            "height": h,
            "aspect_ratio": w / max(h, 1),
            "megapixels": (w * h) / 1e6,
            "brightness": bright,
            "contrast": contrast,
            "edge_density": edge_density,
        })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out_dir / "image_stats.csv", index=False)

    # plots
    plt.figure(figsize=(7, 4))
    sns.histplot(stats_df["aspect_ratio"], bins=40)
    plt.title("Aspect Ratio Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "aspect_ratio_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(stats_df["megapixels"], bins=40)
    plt.title("Megapixels Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "megapixels_hist.png", dpi=160)
    plt.close()

    # brightness/contrast by label
    plt.figure(figsize=(9, 4))
    sns.boxplot(data=stats_df, x="label", y="brightness")
    plt.xticks(rotation=45, ha="right")
    plt.title("Brightness by Label")
    plt.tight_layout()
    plt.savefig(out_dir / "brightness_by_label.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    sns.boxplot(data=stats_df, x="label", y="contrast")
    plt.xticks(rotation=45, ha="right")
    plt.title("Contrast by Label")
    plt.tight_layout()
    plt.savefig(out_dir / "contrast_by_label.png", dpi=160)
    plt.close()

    # 6. Duplicate & leakage detection (exact + near)
    hashes = []
    for p in tqdm(df["filepath"].dropna().tolist(), desc="aHash imágenes"):
        if not Path(p).exists():
            continue
        try:
            h = a_hash(Image.open(p).convert("RGB"))
            hashes.append((p, h))
        except Exception:
            continue

    hash_df = pd.DataFrame(hashes, columns=["filepath", "ahash"])
    hash_df.to_csv(out_dir / "ahash.csv", index=False)

    exact_dups = hash_df.groupby("ahash").filter(lambda g: len(g) > 1)
    exact_dups.to_csv(out_dir / "duplicates_exact_ahash.csv", index=False)

    # near-duplicates within buckets (prefix)
    near = []
    bucket = {}
    for p, h in hashes:
        key = h >> 48
        bucket.setdefault(key, []).append((p, h))

    comparisons = 0
    for items in tqdm(list(bucket.values()), desc="Near-duplicates"):
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                if comparisons >= args.near_dup_max:
                    break
                d = hamming(items[i][1], items[j][1])
                if d <= 5:
                    near.append({"filepath_a": items[i][0], "filepath_b": items[j][0], "hamming": d})
                comparisons += 1
            if comparisons >= args.near_dup_max:
                break
        if comparisons >= args.near_dup_max:
            break

    pd.DataFrame(near).to_csv(out_dir / "duplicates_near_ahash.csv", index=False)

    # 7. Store-level dependency risk
    # Leave-one-store-out proxy: worst remaining fraction after removing any single store
    loso_rows = []
    for cls, g in df.groupby("label"):
        total = len(g)
        worst_remaining = 1.0
        if total > 0:
            for store, sg in g.groupby("store_id"):
                remaining = (total - len(sg)) / total
                worst_remaining = min(worst_remaining, remaining)
        loso_rows.append({"label": cls, "worst_remaining_fraction": worst_remaining})

    loso_df = pd.DataFrame(loso_rows)
    store_risk = dom_store.merge(loso_df, on="label", how="left")
    store_risk["risk_few_stores"] = store_risk["unique_store_id"] < 5
    store_risk["risk_top_store_gt_50pct"] = store_risk["top_store_id_share"] > 0.5
    store_risk["risk_loso_remaining_lt_50pct"] = store_risk["worst_remaining_fraction"] < 0.5
    store_risk.to_csv(out_dir / "store_dependency_risk.csv", index=False)

    summary = {
        "total_images": int(total_images),
        "num_classes": int(class_counts.size),
        "num_chains": int(chain_counts.size),
        "num_stores": int(store_counts.size),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"EDA outputs saved in {out_dir}")


if __name__ == "__main__":
    main()
