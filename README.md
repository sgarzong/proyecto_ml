# Proyecto OOP Clasificador de Muebles

## Estructura
- `src/furniture_classifier/pipeline.py`: Pipeline por clases (ingesta, dataset, entrenamiento, evaluación, inferencia)
- `scripts/build_dataset_gui.py`: GUI para construir dataset desde Excel (split train/val/test y grupos)
- `scripts/train.py`: Entrenamiento
- `scripts/infer.py`: Inferencia
- `data/`: raw/interim/processed/meta
- `models/`: pesos y specs
- `outputs/`: reportes, gráficos e inferencias
- `requirements.txt`: dependencias

## Entrenamiento
Edita `data_dir` en `scripts/train.py` y ejecuta:
```
python -m venv .venv
./.venv/Scripts/activate 
pip install -r requirements.txt
set PYTHONPATH=src
python scripts/train.py
```

## Inferencia
Asegúrate de que `model_spec.json` existe (salida del entrenamiento) y ejecuta:
```
set PYTHONPATH=src
python scripts/infer.py
```

## Notas
- El pipeline usa Focal Loss + WeightedRandomSampler para desbalance de clases.
- Se elimina duplicado exacto de imágenes (hash md5) y se guardan duplicados en `data/meta/duplicates.csv`.
- El split es estratificado si no hay grupos; si hay columna de grupo, se hace split por grupos para evitar leakage.
- Se crea `train/val/test` si se define `test_ratio`.
- Guarda `weights.pt`, `model_spec.json`, `history.csv`, `confusion_matrix_val.csv` y `confusion_matrix_test.csv` (si existe test).

## Reportes / Métricas
Ejemplos (ejecutar con `set PYTHONPATH=src`):
- Evaluación y curvas ROC/PR: `python scripts/evaluate.py --model_spec models/<run>/model_spec.json --data_dir data/processed/<dataset> --split test`
- Curvas de entrenamiento: `python scripts/plot_training_curves.py --history models/<run>/history.csv --out_dir outputs/reports/<run>`
- EDA completo: `python scripts/eda.py --excel data/raw/base_muebles.xlsx`
- Embeddings + UMAP/t-SNE: `python scripts/embeddings.py --data_dir data/processed/<dataset> --split val`
- Learning curve: `python scripts/learning_curve.py --data_dir data/processed/<dataset>`
- Comparación estadística de modelos: `python scripts/model_compare.py --model_a models/<run_a>/model_spec.json --model_b models/<run_b>/model_spec.json --data_dir data/processed/<dataset>`
