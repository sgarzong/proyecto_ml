import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

# Importa tu pipeline OOP
from furniture_classifier.pipeline import DataIngestor, ImageStore, DatasetBuilder, ensure_dir


class GuiBuildFromExcel(tk.Tk):
    """
    GUI para:
      1) Seleccionar varios Excels
      2) Elegir columnas (URL y etiqueta)
      3) Definir carpeta de salida, val_ratio y seed
      4) Construir dataset ImageFolder (train/val/<clase>)
    """

    def __init__(self):
        super().__init__()
        self.title("Dataset Builder desde Excel (Tkinter)")
        self.geometry("820x600")
        self.minsize(820, 600)

        # Paths base del proyecto
        self.project_root = Path(__file__).resolve().parents[1]

        # Estado
        self.excel_paths = []       # lista de rutas a excels seleccionados
        self.columns_cache = []     # columnas detectadas del primer excel (para combos)
        self.url_col = tk.StringVar()
        self.label_col = tk.StringVar()
        self.group_col = tk.StringVar()
        self.out_root = tk.StringVar(value=str(self.project_root / "data"))
        self.out_processed = tk.StringVar(value=str(self.project_root / "data" / "processed" / "furniture_type_split"))
        self.val_ratio = tk.DoubleVar(value=0.20)
        self.test_ratio = tk.DoubleVar(value=0.10)
        self.seed = tk.IntVar(value=42)
        self.group_from_url = tk.BooleanVar(value=False)
        self.dedup = tk.BooleanVar(value=True)

        # UI
        self._build_ui()

    # -------------------------
    # Construcción de interfaz
    # -------------------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        frame_files = ttk.LabelFrame(self, text="1) Archivos Excel")
        frame_files.pack(fill="x", **pad)

        btn_add = ttk.Button(frame_files, text="Agregar Excels...", command=self.on_add_excels)
        btn_add.pack(side="left", **pad)

        btn_clear = ttk.Button(frame_files, text="Limpiar lista", command=self.on_clear_excels)
        btn_clear.pack(side="left", **pad)

        self.list_files = tk.Listbox(frame_files, height=4)
        self.list_files.pack(fill="x", padx=10, pady=8)

        frame_cols = ttk.LabelFrame(self, text="2) Selección de columnas")
        frame_cols.pack(fill="x", **pad)

        row_cols = ttk.Frame(frame_cols)
        row_cols.pack(fill="x", **pad)

        ttk.Label(row_cols, text="Columna de URL:").grid(row=0, column=0, sticky="w")
        self.cbo_url = ttk.Combobox(row_cols, textvariable=self.url_col, state="readonly", width=40)
        self.cbo_url.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(row_cols, text="Columna de Etiqueta (tipo):").grid(row=0, column=2, sticky="w")
        self.cbo_lbl = ttk.Combobox(row_cols, textvariable=self.label_col, state="readonly", width=40)
        self.cbo_lbl.grid(row=0, column=3, sticky="w", padx=6)

        ttk.Label(row_cols, text="Columna de Grupo (opcional):").grid(row=1, column=0, sticky="w", pady=(8,0))
        self.cbo_group = ttk.Combobox(row_cols, textvariable=self.group_col, state="readonly", width=40)
        self.cbo_group.grid(row=1, column=1, sticky="w", padx=6, pady=(8,0))
        chk_group_url = ttk.Checkbutton(row_cols, text="Derivar grupo desde URL (repo)", variable=self.group_from_url)
        chk_group_url.grid(row=1, column=2, columnspan=2, sticky="w", padx=6, pady=(8,0))

        btn_inspect = ttk.Button(frame_cols, text="Inspeccionar columnas", command=self.on_inspect_columns)
        btn_inspect.pack(anchor="w", **pad)

        # Salidas / configuración
        frame_out = ttk.LabelFrame(self, text="3) Salida y parámetros")
        frame_out.pack(fill="x", **pad)

        row_out1 = ttk.Frame(frame_out)
        row_out1.pack(fill="x", **pad)

        ttk.Label(row_out1, text="Carpeta raíz de salida (se creará data/raw y data/meta):").grid(row=0, column=0, sticky="w")
        ent_out = ttk.Entry(row_out1, textvariable=self.out_root, width=70)
        ent_out.grid(row=1, column=0, sticky="we", pady=4)
        btn_browse_out = ttk.Button(row_out1, text="Examinar...", command=self.on_browse_out_root)
        btn_browse_out.grid(row=1, column=1, padx=6)

        row_out2 = ttk.Frame(frame_out)
        row_out2.pack(fill="x", **pad)

        ttk.Label(row_out2, text="Carpeta del dataset procesado (train/val):").grid(row=0, column=0, sticky="w")
        ent_proc = ttk.Entry(row_out2, textvariable=self.out_processed, width=70)
        ent_proc.grid(row=1, column=0, sticky="we", pady=4)
        btn_browse_proc = ttk.Button(row_out2, text="Examinar...", command=self.on_browse_out_processed)
        btn_browse_proc.grid(row=1, column=1, padx=6)

        row_params = ttk.Frame(frame_out)
        row_params.pack(fill="x", **pad)

        ttk.Label(row_params, text="Porcentaje de validación (0.0 - 0.9):").grid(row=0, column=0, sticky="w")
        spn_val = ttk.Spinbox(row_params, from_=0.05, to=0.9, increment=0.05, textvariable=self.val_ratio, width=6)
        spn_val.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(row_params, text="Porcentaje de test (0.0 - 0.5):").grid(row=0, column=2, sticky="w", padx=(18,6))
        spn_test = ttk.Spinbox(row_params, from_=0.0, to=0.5, increment=0.05, textvariable=self.test_ratio, width=6)
        spn_test.grid(row=0, column=3, sticky="w")

        ttk.Label(row_params, text="Seed (reproducibilidad):").grid(row=1, column=0, sticky="w", pady=(8,0))
        spn_seed = ttk.Spinbox(row_params, from_=0, to=999999, increment=1, textvariable=self.seed, width=8)
        spn_seed.grid(row=1, column=1, sticky="w", pady=(8,0))
        chk_dedup = ttk.Checkbutton(row_params, text="Eliminar duplicados exactos", variable=self.dedup)
        chk_dedup.grid(row=1, column=2, columnspan=2, sticky="w", padx=6, pady=(8,0))

        # Acciones
        frame_actions = ttk.Frame(self)
        frame_actions.pack(fill="x", **pad)

        self.btn_build = ttk.Button(frame_actions, text="4) Construir dataset", command=self.on_build_clicked)
        self.btn_build.pack(side="left", padx=8)

        btn_close = ttk.Button(frame_actions, text="Cerrar", command=self.destroy)
        btn_close.pack(side="right", padx=8)

        # Log
        frame_log = ttk.LabelFrame(self, text="Registro")
        frame_log.pack(fill="both", expand=True, **pad)

        self.txt_log = tk.Text(frame_log, height=14)
        self.txt_log.pack(fill="both", expand=True, padx=8, pady=8)
        self._log("Bienvenido. Agrega Excels, elige columnas y construye tu dataset.")

    # -------------------------
    # Handlers UI
    # -------------------------
    def _log(self, msg: str):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.update_idletasks()

    def on_add_excels(self):
        paths = filedialog.askopenfilenames(
            title="Selecciona archivos Excel",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if not paths:
            return
        for p in paths:
            if p not in self.excel_paths:
                self.excel_paths.append(p)
                self.list_files.insert("end", p)
        self._log(f"{len(paths)} archivo(s) agregado(s). Total: {len(self.excel_paths)}")

    def on_clear_excels(self):
        self.excel_paths.clear()
        self.list_files.delete(0, "end")
        self.cbo_url.set("")
        self.cbo_lbl.set("")
        self.columns_cache = []
        self._log("Lista de Excels limpiada.")

    def on_inspect_columns(self):
        if not self.excel_paths:
            messagebox.showwarning("Falta Excel", "Primero agrega uno o más archivos de Excel.")
            return
        first = self.excel_paths[0]
        try:
            self._log(f"Inspeccionando columnas de: {first}")
            # Leemos solo encabezados (n=0 filas) para evitar cargar todo
            df_head = pd.read_excel(first, engine="openpyxl", nrows=0)
            cols = list(df_head.columns)
            if not cols:
                raise ValueError("No se detectaron columnas en el Excel.")
            self.columns_cache = cols
            self.cbo_url["values"] = cols
            self.cbo_lbl["values"] = cols
            self.cbo_group["values"] = cols
            # intento autoselección heurística
            guess_url = next((c for c in cols if c.strip().lower() in ("url", "image_url", "link", "foto", "imagen", "url_imagen", "link github")), "")
            guess_lbl = next((c for c in cols if "tipo" in c.strip().lower() or "mueble" in c.strip().lower() or "clase" in c.strip().lower() or "label" in c.strip().lower()), "")
            guess_group = next((c for c in cols if "cod" in c.strip().lower() or "local" in c.strip().lower() or "id" in c.strip().lower()), "")
            if guess_url: self.url_col.set(guess_url)
            if guess_lbl: self.label_col.set(guess_lbl)
            if guess_group: self.group_col.set(guess_group)
            self._log(f"Columnas detectadas: {cols}")
            if guess_url or guess_lbl:
                self._log(f"Sugerencias -> URL: '{guess_url or '-'}' | Etiqueta: '{guess_lbl or '-'}' | Grupo: '{guess_group or '-'}'")
        except Exception as e:
            messagebox.showerror("Error leyendo columnas", str(e))
            self._log(traceback.format_exc())

    def on_browse_out_root(self):
        path = filedialog.askdirectory(title="Selecciona carpeta raíz para data/raw y data/meta")
        if path:
            self.out_root.set(path)
            self._log(f"Carpeta raíz establecida: {path}")

    def on_browse_out_processed(self):
        path = filedialog.askdirectory(title="Selecciona carpeta para el dataset procesado (train/val)")
        if path:
            self.out_processed.set(path)
            self._log(f"Carpeta procesada establecida: {path}")

    def on_build_clicked(self):
        # Validaciones mínimas
        if not self.excel_paths:
            messagebox.showwarning("Faltan Excels", "Agrega al menos un archivo Excel.")
            return
        if not self.url_col.get() or not self.label_col.get():
            messagebox.showwarning("Faltan columnas", "Selecciona columna de URL y columna de Etiqueta.")
            return
        if not self.out_root.get():
            self.out_root.set(str(self.project_root / "data"))
        if not self.out_processed.get():
            self.out_processed.set(str(self.project_root / "data" / "processed" / "furniture_type_split"))

        # Corre en hilo aparte para no congelar la UI
        self.btn_build.config(state="disabled")
        t = threading.Thread(target=self._build_pipeline_safe, daemon=True)
        t.start()

    # -------------------------
    # Lógica de construcción
    # -------------------------
    def _build_pipeline_safe(self):
        try:
            self._build_pipeline()
            messagebox.showinfo("Éxito", "Dataset construido correctamente.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._log(traceback.format_exc())
        finally:
            self.btn_build.config(state="normal")

    def _build_pipeline(self):
        excel_paths = list(self.excel_paths)
        url_col = self.url_col.get()
        label_col = self.label_col.get()
        group_col = self.group_col.get()
        out_root = self.out_root.get()
        out_processed = self.out_processed.get()
        val_ratio = float(self.val_ratio.get())
        test_ratio = float(self.test_ratio.get())
        seed = int(self.seed.get())
        group_from_url = bool(self.group_from_url.get())
        dedup = bool(self.dedup.get())

        # 1) Ingesta
        self._log("Leyendo Excels y unificando columnas...")
        ing = DataIngestor(
            excel_paths=excel_paths,
            url_col=url_col,
            label_col=label_col,
            group_col=group_col or None,
            group_from_url=group_from_url
        )
        df = ing.read()
        self._log(f"Filas leídas: {len(df)}")

        # 2) Materialización (descarga / copia) + CSV maestro
        self._log("Descargando / copiando imágenes y generando dataset_master.csv ...")
        store = ImageStore(out_root=out_root)
        csv_master = store.materialize(df, dedup=dedup)
        self._log(f"CSV maestro generado en: {csv_master}")

        # 3) Construir ImageFolder (train/val/test)
        self._log("Construyendo estructura ImageFolder (train/val/test/<clase>) ...")
        ensure_dir(out_processed)
        builder = DatasetBuilder(val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
        class_to_idx = builder.build_from_csv(csv_path=str(csv_master), out_dir=out_processed)

        clases = ", ".join(class_to_idx.keys())
        self._log(f"¡Listo! Dataset en: {out_processed}")
        self._log(f"Clases detectadas ({len(class_to_idx)}): {clases}")


if __name__ == "__main__":
    app = GuiBuildFromExcel()
    app.mainloop()
