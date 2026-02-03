from __future__ import annotations
import io, os, csv, json, math, shutil, zipfile, random, re, hashlib, urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms, datasets
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


# ---------- Utilidades ----------
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_\-]+")

def safe_name(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    s = SAFE_NAME_RE.sub("", s)
    return s or "NA"

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _extract_group_from_url(url: str) -> str:
    try:
        u = urllib.parse.urlparse(url)
        parts = [p for p in u.path.split("/") if p]
        if len(parts) >= 2:
            return safe_name(f"{parts[0]}_{parts[1]}")
    except Exception:
        pass
    return "NA"


# 1) Ingesta de datos
@dataclass  #decorador para evitar tener que poner el código tantas veces, se define en TrainerConfig
class DataIngestor:
    excel_paths: List[str]
    url_col: str
    label_col: str  # “Tipo de mueble”
    group_col: Optional[str] = None
    group_from_url: bool = False

    def read(self) -> pd.DataFrame:
        frames = []
        for p in self.excel_paths:
            df = pd.read_excel(p, engine="openpyxl")
            df["__source_excel__"] = p
            frames.append(df)
        if not frames:
            raise ValueError("No se cargaron Excels.")
        df = pd.concat(frames, ignore_index=True)
        if self.url_col not in df.columns or self.label_col not in df.columns:
            raise ValueError("Columnas URL/Etiqueta no existen en los Excels.")
        out = df[[self.url_col, self.label_col]].rename(
            columns={self.url_col: "url", self.label_col: "label"}
        )
        if self.group_col and self.group_col in df.columns:
            out["group"] = df[self.group_col].astype(str).map(safe_name)
        elif self.group_from_url:
            out["group"] = out["url"].astype(str).map(_extract_group_from_url)
        return out


# 2) Almacenamiento de imágenes
class ImageStore:
    def __init__(self, out_root: str):
        self.raw_dir = ensure_dir(Path(out_root) / "data" / "raw")
        self.meta_dir = ensure_dir(Path(out_root) / "data" / "meta")

    def _download_or_copy(self, url_or_path: str, dst_path: Path, timeout: int = 15) -> bool:
        try:
            s = (url_or_path or "").strip()
            if s.lower().startswith(("http://", "https://")):
                import requests
                r = requests.get(s, timeout=timeout, stream=True)
                if r.status_code != 200:
                    return False
                with open(dst_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            else:
                src = Path(s)
                if not src.exists():
                    return False
                shutil.copy2(src, dst_path)
            return True
        except Exception:
            return False

    def _file_md5(self, path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def materialize(self, df: pd.DataFrame, dedup: bool = True) -> Path:
        """Crea dataset_master.csv con filepath,label."""
        rows = []
        seen_hashes = {}
        dup_rows = []
        for i, r in tqdm(df.iterrows(), total=len(df), desc="Descargando/copiendo"):
            label = safe_name(str(r["label"]))
            url = str(r["url"]).strip()
            fname = f"img_{i}.jpg"
            dst = self.raw_dir / label / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            ok = self._download_or_copy(url, dst)
            if ok:
                file_hash = self._file_md5(dst)
                if dedup and file_hash in seen_hashes:
                    dst.unlink(missing_ok=True)
                    dup_rows.append({"filepath": str(dst), "label": label, "url": url, "dup_of": seen_hashes[file_hash]})
                    continue
                seen_hashes[file_hash] = str(dst)
                row = {"filepath": str(dst), "label": label, "url": url}
                if "group" in df.columns:
                    row["group"] = safe_name(str(r["group"]))
                rows.append(row)
        if not rows:
            raise RuntimeError("No se pudo materializar ninguna imagen.")
        csv_path = self.meta_dir / "dataset_master.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        if dup_rows:
            pd.DataFrame(dup_rows).to_csv(self.meta_dir / "duplicates.csv", index=False)
        return csv_path


# 3) Construcción de ImageFolder
class DatasetBuilder:
    def __init__(self, val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def _split_stratified(self, labels: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.arange(len(labels))
        labels_arr = np.array(labels)

        if self.test_ratio and self.test_ratio > 0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.seed)
            trainval_idx, test_idx = next(sss.split(idx, labels_arr))
            val_ratio_adj = self.val_ratio / max(1.0 - self.test_ratio, 1e-6)
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adj, random_state=self.seed + 1)
            train_idx, val_idx = next(sss2.split(trainval_idx, labels_arr[trainval_idx]))
            train_idx = trainval_idx[train_idx]
            val_idx = trainval_idx[val_idx]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_ratio, random_state=self.seed)
            train_idx, val_idx = next(sss.split(idx, labels_arr))
            test_idx = np.array([], dtype=int)
        return train_idx, val_idx, test_idx

    def _split_grouped(self, labels: List[str], groups: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.arange(len(labels))
        labels_arr = np.array(labels)
        groups_arr = np.array(groups)

        def has_all_classes(split_idx: np.ndarray) -> bool:
            return set(labels_arr[split_idx]) == set(labels_arr)

        max_attempts = 50
        for k in range(max_attempts):
            seed = self.seed + k
            if self.test_ratio and self.test_ratio > 0:
                gss = GroupShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=seed)
                trainval_idx, test_idx = next(gss.split(idx, labels_arr, groups_arr))
                val_ratio_adj = self.val_ratio / max(1.0 - self.test_ratio, 1e-6)
                gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio_adj, random_state=seed + 1)
                train_idx, val_idx = next(gss2.split(trainval_idx, labels_arr[trainval_idx], groups_arr[trainval_idx]))
                train_idx = trainval_idx[train_idx]
                val_idx = trainval_idx[val_idx]
            else:
                gss = GroupShuffleSplit(n_splits=1, test_size=self.val_ratio, random_state=seed)
                train_idx, val_idx = next(gss.split(idx, labels_arr, groups_arr))
                test_idx = np.array([], dtype=int)

            if has_all_classes(train_idx) and has_all_classes(val_idx):
                return train_idx, val_idx, test_idx

        print("Advertencia: no se logró mantener todas las clases en train/val con el split por grupos.")
        return train_idx, val_idx, test_idx

    def build_from_csv(self, csv_path: str, out_dir: str) -> Dict[str, int]:
        df = pd.read_csv(csv_path)
        assert {"filepath", "label"}.issubset(df.columns)
        out_dir = ensure_dir(out_dir)
        train_dir = ensure_dir(Path(out_dir) / "train")
        val_dir = ensure_dir(Path(out_dir) / "val")
        test_dir = ensure_dir(Path(out_dir) / "test") if self.test_ratio and self.test_ratio > 0 else None

        labels = [safe_name(str(r["label"])) for _, r in df.iterrows()]
        groups = [safe_name(str(r["group"])) for _, r in df.iterrows()] if "group" in df.columns else None
        filepaths = df["filepath"].astype(str).tolist()
        items = list(zip(filepaths, labels))

        if groups:
            train_idx, val_idx, test_idx = self._split_grouped(labels, groups)
        else:
            train_idx, val_idx, test_idx = self._split_stratified(labels)

        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]
        test_items = [items[i] for i in test_idx] if len(test_idx) else []

        classes = sorted({lbl for _, lbl in items})
        class_to_idx = {c: i for i, c in enumerate(classes)}

        splits = [("train", train_items, train_dir), ("val", val_items, val_dir)]
        if test_dir and test_items:
            splits.append(("test", test_items, test_dir))

        for split_name, split_items, split_root in splits:
            for src_path, label in tqdm(split_items, desc=f"Copiando {split_name}"):
                dst_dir = ensure_dir(Path(split_root) / label)
                dst = dst_dir / Path(src_path).name
                try:
                    shutil.copy2(src_path, dst)
                except Exception:
                    try:
                        Image.open(src_path).convert("RGB").save(dst)
                    except Exception:
                        pass

        with open(Path(out_dir) / "class_to_idx.json", "w", encoding="utf-8") as f:  #carga los tipos de fotos en el json
            json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
        return class_to_idx


# 4) Balanceo
class ClassBalancer:
    @staticmethod  #ya que la funcion solo usa sus propios argumentos, se pone este decorador
    def class_weights(train_ds: datasets.ImageFolder, device) -> Tuple[torch.Tensor, np.ndarray]:
        counts = np.bincount(train_ds.targets, minlength=len(train_ds.classes)).astype(np.float64)
        inv = 1.0 / (counts + 1e-6)
        inv = inv / inv.sum() * len(counts)
        return torch.tensor(inv, dtype=torch.float32, device=device), counts

    @staticmethod
    def sampler(train_ds: datasets.ImageFolder, counts: np.ndarray) -> WeightedRandomSampler:
        sample_weights = [1.0 / max(counts[y], 1.0) for y in train_ds.targets]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# 5) Modelo / Especificación
class ModelSpec:
    @staticmethod
    def default_transforms(image_size: int = 224):  #les estandariza a las imágenes y las prepara para el entrenamiento
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return train_tf, val_tf

    @staticmethod    #función para elegir entre varios modelos
    def build_model(num_classes: int, arch: str = "resnet18", pretrained: bool = True) -> nn.Module:
        if arch == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            in_f = m.fc.in_features
            m.fc = nn.Linear(in_f, num_classes)
            return m
        elif arch == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            in_f = m.fc.in_features
            m.fc = nn.Linear(in_f, num_classes)
            return m
        elif arch == "mobilenet_v3_small":
            m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            in_f = m.classifier[-1].in_features
            m.classifier[-1] = nn.Linear(in_f, num_classes)
            return m
        elif arch == "mobilenet_v2":
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            in_f = m.classifier[-1].in_features
            m.classifier[-1] = nn.Linear(in_f, num_classes)
            return m
        elif arch == "efficientnet_b0":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            in_f = m.classifier[-1].in_features
            m.classifier[-1] = nn.Linear(in_f, num_classes)
            return m
        elif arch == "squeezenet1_0":
            m = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1 if pretrained else None)
            m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
            m.num_classes = num_classes
            return m
        else:
            raise ValueError(f"Arquitectura no soportada: {arch}")

    @staticmethod
    def save(spec_path: str, spec: Dict):
        ensure_dir(Path(spec_path).parent)
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(spec_path: str) -> Tuple[nn.Module, transforms.Compose, List[str]]:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        classes = spec["classes"]
        arch = spec["architecture"]
        img_size = spec.get("image_size", 224)
        _, val_tf = ModelSpec.default_transforms(img_size)
        model = ModelSpec.build_model(len(classes), arch=arch, pretrained=False)
        state = torch.load(spec["weights_path"], map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, val_tf, classes


# 6) Entrenamiento
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"): #inicialización de parámetros de función de costo
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")
        self.reduction = reduction

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce).clamp_min(1e-8)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean() if self.reduction == "mean" else focal.sum()

from dataclasses import dataclass
@dataclass
class TrainerConfig: #función para tener configuración predeterminada del decorador dataclass
    data_dir: str
    out_dir: str
    epochs: int = 10
    batch_size: int = 32
    lr: float = 5e-4
    image_size: int = 224
    architecture: str = "resnet18"
    use_sampler: bool = True
    loss: str = "focal"  # "ce" o "focal"
    gamma: float = 2.0
    label_smoothing: float = 0.1
    pin_memory: bool = False
    step_size: int = 5
    step_gamma: float = 0.5
    patience: int = 5
    seed: int = 42
    eval_test: bool = True

class Evaluator:  #función para evaluar desempeño del modelo y elegir el mejor
    @staticmethod
    @torch.no_grad()  #no carga el autograd, solo calcula métricas
    def evaluate(model, loader, criterion, device, classes: List[str], out_dir: Path, split_name: str = "val") -> Tuple[float, float, float]:
        model.eval()
        total, correct, running = 0, 0, 0.0
        all_preds, all_labels = [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

        val_loss = running / max(total, 1)
        val_acc = correct / max(total, 1)

        bal_acc = balanced_accuracy_score(all_labels, all_preds) if len(set(all_labels)) > 1 else val_acc
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                                columns=[f"pred_{c}" for c in classes])
        cm_df.to_csv(Path(out_dir) / f"confusion_matrix_{split_name}.csv", index=True)
        return val_loss, val_acc, bal_acc

class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #permite usar cpu si está disponible


    def _dataloaders(self):
        tr_tf, va_tf = ModelSpec.default_transforms(self.cfg.image_size)
        train_ds = datasets.ImageFolder(Path(self.cfg.data_dir) / "train", transform=tr_tf)
        val_ds   = datasets.ImageFolder(Path(self.cfg.data_dir) / "val",   transform=va_tf)
        test_path = Path(self.cfg.data_dir) / "test"
        test_ds = datasets.ImageFolder(test_path, transform=va_tf) if test_path.exists() else None

        # validación de que las clases son las mismas
        if train_ds.classes != val_ds.classes:
            raise SystemExit(f"Clases distintas entre train y val.\ntrain: {train_ds.classes}\nval: {val_ds.classes}")
        if test_ds is not None and train_ds.classes != test_ds.classes:
            raise SystemExit(f"Clases distintas entre train y test.\ntrain: {train_ds.classes}\ntest: {test_ds.classes}")

        # permite determinar qué función de costo utilizar (si no es la de focal, usa cross entropy)
        weights, counts = ClassBalancer.class_weights(train_ds, device=self.device)
        if self.cfg.loss == "focal":
            criterion = FocalLoss(alpha=weights, gamma=self.cfg.gamma)
        else:
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=self.cfg.label_smoothing)

        if self.cfg.use_sampler:
            sampler = ClassBalancer.sampler(train_ds, counts)
            train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, sampler=sampler,
                                      num_workers=2, pin_memory=self.cfg.pin_memory)
        else:
            train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True,
                                      num_workers=2, pin_memory=self.cfg.pin_memory)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False,
                                num_workers=2, pin_memory=self.cfg.pin_memory)
        test_loader = None
        if test_ds is not None:
            test_loader = DataLoader(test_ds, batch_size=self.cfg.batch_size, shuffle=False,
                                     num_workers=2, pin_memory=self.cfg.pin_memory)
        return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, criterion

    # función para realizar el ajuste de parámetros y seleccionar mejor modelo
    def fit(self):
        set_global_seed(self.cfg.seed)
        out_dir = ensure_dir(self.cfg.out_dir)
        train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, criterion = self._dataloaders()

        model = ModelSpec.build_model(len(train_ds.classes), arch=self.cfg.architecture, pretrained=True).to(self.device)
        optim = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=self.cfg.step_size, gamma=self.cfg.step_gamma)

        best_bal = -1.0
        bad_epochs = 0
        weights_path = Path(out_dir) / "weights.pt"
        history = []

        for epoch in range(1, self.cfg.epochs + 1):
            # train
            model.train()
            tot, ok, loss_sum = 0, 0, 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optim.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optim.step()

                loss_sum += loss.item() * x.size(0)
                ok += (logits.argmax(1) == y).sum().item()
                tot += y.size(0)
            tr_loss, tr_acc = loss_sum / max(tot, 1), ok / max(tot, 1)

            # eval
            va_loss, va_acc, va_bal = Evaluator.evaluate(model, val_loader, criterion, self.device, train_ds.classes, out_dir, split_name="val")

            print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f} bal_acc={va_bal:.3f}")
            history.append({
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "val_bal_acc": va_bal,
                "lr": optim.param_groups[0]["lr"],
            })

            if va_bal > best_bal:
                best_bal = va_bal
                torch.save(model.state_dict(), weights_path)
                bad_epochs = 0
                print(f"* Mejor modelo (según balanced acc) guardado en: {weights_path}")
            else:
                bad_epochs += 1

            sched.step()
            if bad_epochs >= self.cfg.patience:
                print(f"Early stopping (sin mejora en {self.cfg.patience} épocas).")
                break

        # guardar spec
        spec = {
            "architecture": self.cfg.architecture,
            "image_size": self.cfg.image_size,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "classes": train_ds.classes,
            "class_to_idx": train_ds.class_to_idx,
            "weights_path": str(weights_path),
            "pytorch_version": torch.__version__,
        }
        ModelSpec.save(str(Path(out_dir) / "model_spec.json"), spec)
        pd.DataFrame(history).to_csv(Path(out_dir) / "history.csv", index=False)

        if self.cfg.eval_test and test_loader is not None:
            best_state = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(best_state)
            te_loss, te_acc, te_bal = Evaluator.evaluate(
                model, test_loader, criterion, self.device, train_ds.classes, out_dir, split_name="test"
            )
            print(f"Test | loss={te_loss:.4f} acc={te_acc:.3f} bal_acc={te_bal:.3f}")


# 7) Inferencia
class InferenceEngine: #clase para realizar la clasificación del zip a carpetas
    def __init__(self, model_spec_path: str):
        self.model, self.tf, self.classes = ModelSpec.load(model_spec_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _list_images_in_zip(self, zip_path: str) -> List[str]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        with zipfile.ZipFile(zip_path, "r") as zf:
            return [n for n in zf.namelist() if Path(n).suffix.lower() in exts]

    def classify_zip(self, zip_path: str, out_dir: str, keep_names: bool = True, save_csv: bool = True):
        out_dir = ensure_dir(out_dir)
        rows = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = self._list_images_in_zip(zip_path)
            print(f"{len(names)} imágenes detectadas.")
            for i, name in enumerate(names, 1):
                with zf.open(name) as f:
                    im = Image.open(io.BytesIO(f.read())).convert("RGB")
                x = self.tf(im).unsqueeze(0).to(self.device)
                with torch.no_grad(): 
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    idx = int(np.argmax(probs))
                    label = self.classes[idx]
                    conf = float(probs[idx])

                class_dir = ensure_dir(Path(out_dir) / label)
                stem = Path(name).name if keep_names else f"img_{i:05d}.jpg"
                im.save(class_dir / stem)
                rows.append({"filename": stem, "pred": label, "conf": conf})

        if save_csv:
            pd.DataFrame(rows).to_csv(Path(out_dir) / "predicciones.csv", index=False)
            print("CSV de predicciones guardado.")
