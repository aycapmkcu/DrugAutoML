from DrugAutoML.data_preprocess import DataPreprocessor
from DrugAutoML.featurization import Featurizer
from DrugAutoML.data_split import DataSplitter
from DrugAutoML.model_selection import ModelSelector
from DrugAutoML.model_finalization import ModelFinalization
from DrugAutoML.explainability import Explainability
from DrugAutoML.prediction import Predictor


config_preprocessing = {
    "path": "BACE1.csv",
    "smiles_col": "Smiles",
    "activity_col": "class",
    "activity_type": "binary",
    "standardize_smiles": True,
    "save_results": True,
    "results_dir": "results"
}

config_featurization = {
    "fp_type": "ECFP",
    "radius": 2,
    "nbits": 2048,
    "use_counts": False,
    "drop_constant": True,
    "save_results": True,
    "results_dir": "results"
}

config_split = {
    "method": "scaffold",  # veya "random"
    "frac_train": 0.8,
    "random_state": 42,
    "save_results": True,
    "results_dir": "results"
}

config_model_selection = {
    # "train_X_path": "results/split_train_X_20250813_132955.csv",  # opsiyonel
    # "train_y_path": "results/split_train_y_20250813_132955.csv",  # opsiyonel
    "results_dir": "results",
    "algos": "auto",  # veya ["LR", "RF", "XGB"]
    "scoring": "MCC",  # balanced_accuracy, recall, specificity, precision, roc_auc, f1_macro
    "n_trials_per_algo": 20,
    "cv_splits": 5,
    "cv_repeats": 1,
    "random_state": 42
}

config_model_finalization = {
        "results_dir": "results",
        # "algo": "LGBM",
        # "params": {...},
        "calibration": "isotonic",        # default; good for >1000 samples
        "cv_folds": 5,
        "threshold_strategy": "max_mcc",  # "fixed" | "youden" | "max_f1" | "max_mcc" | "target_sensitivity" | "target_specificity" | "cost_ratio"
        "threshold_kwargs": {},           # e.g. {"value":0.5} or {"target":0.9}
        "save_model": True
}

config_explainability = {
    "results_dir": "results",        # pipeline çıktı klasörü
    "sample_limit": 1000,            # SHAP için örnek sınırı (None = tüm test)
    "gallery_top_bits": 6,           # galeriye alınacak en önemli bit sayısı
    "gallery_examples_per_bit": 6,   # her bit için kaç molekül gösterilecek
    "gallery_cols": 3,               # galeri sütun sayısı
    "gallery_sub_img": [280, 240],   # hücre boyutu (genişlik, yükseklik)

    # çizim stili
    "bw_base": True,                 # siyah-beyaz iskelet
    "hide_atom_labels": True,        # atom etiketlerini gizle (sadece iskelet)
    "highlight_rgb": [0.0, 0.75, 1.0],  # highlight rengi (0..1 arası RGB)
    "highlight_radius": 0.60,        # highlight baloncuk yarıçapı
    "bond_line_width": 1.0,          # bağ çizgisi kalınlığı

    # SHAP özetleri
    "also_beeswarm": True,           # beeswarm + bar + signed bar kaydet

    "random_seed": 42
}



preproc = DataPreprocessor(config_preprocessing).run()
featurizer = Featurizer(config_featurization).run()
splitter = DataSplitter(config_split).run()
selector = ModelSelector(config_model_selection).run()
finalizer = ModelFinalization(config_model_finalization).run()
explainer = Explainability(config_explainability).run()


cfg = {
    "results_dir": "results",
    "input_path": "/Users/aycabeyhan/PycharmProjects/bioactivity_prediction/example/data/BACE1_external.csv",
    "smiles_col": "mol",
    "label_col": "Class",              # 0/1
    "activity_type": "binary"
}
pred = Predictor(cfg).run()

