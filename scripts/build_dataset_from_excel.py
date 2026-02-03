import argparse
from pathlib import Path

from furniture_classifier.pipeline import DataIngestor, ImageStore, DatasetBuilder, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--url_col", default="LINK GITHUB")
    ap.add_argument("--label_col", default="MUEBLES")
    ap.add_argument("--group_col", default="COD LOCAL/Ales")
    ap.add_argument("--group_from_url", action="store_true")
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--out_processed", default="data/processed/furniture_type_split")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedup", action="store_true", default=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_processed = Path(args.out_processed)

    ing = DataIngestor(
        excel_paths=[args.excel],
        url_col=args.url_col,
        label_col=args.label_col,
        group_col=args.group_col or None,
        group_from_url=args.group_from_url,
    )
    df = ing.read()

    store = ImageStore(out_root=str(out_root))
    csv_master = store.materialize(df, dedup=args.dedup)

    ensure_dir(out_processed)
    builder = DatasetBuilder(val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    builder.build_from_csv(csv_path=str(csv_master), out_dir=str(out_processed))

    print(f"dataset_master.csv: {csv_master}")
    print(f"processed dataset: {out_processed}")


if __name__ == "__main__":
    main()
