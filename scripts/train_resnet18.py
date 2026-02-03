from furniture_classifier.pipeline import Trainer, TrainerConfig

def main():
    cfg = TrainerConfig(
        data_dir="data/processed/furniture_type_split",
        out_dir="models/resnet18_furniture_type",
        epochs=20,
        batch_size=16,
        lr=5e-4,
        image_size=256,
        architecture="resnet18",
        use_sampler=True,
        loss="focal",
        pin_memory=False,
        patience=5,
        seed=42,
        eval_test=True,
    )
    Trainer(cfg).fit()

if __name__ == "__main__":
    main()
