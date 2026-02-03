from furniture_classifier.pipeline import InferenceEngine

if __name__ == "__main__":
    eng = InferenceEngine("models/mobilenet_v3_small_furniture_type/model_spec.json")
    eng.classify_zip(
        zip_path=r"C:\Users\sebas\Downloads\OneDrive_2025-10-17.zip",
        out_dir="outputs/inference/lote_fusion_oop",
        keep_names=True, save_csv=True
    )
