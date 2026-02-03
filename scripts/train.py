from furniture_classifier.pipeline import Trainer, TrainerConfig

if __name__ == "__main__":
    cfg = TrainerConfig(
        data_dir="data/processed/furniture_type_split",
        out_dir="models/mobilenet_v3_small_furniture_type",
        epochs=20, batch_size=16, lr=5e-4, image_size=256,
        architecture="mobilenet_v3_small", use_sampler=True, loss="focal",
        pin_memory=False, patience=5, seed=42, eval_test=True
    )
    Trainer(cfg).fit()
