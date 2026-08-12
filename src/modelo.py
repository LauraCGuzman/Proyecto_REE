# src/modelo.py
"""Carga y guardado del modelo champion serializado (`modelo_v1`).

No entrena ni reajusta nada: el objeto que se guarda con `guardar_modelo` es
el `modelo_nivel_final` ya entrenado en notebooks/modelo_demanda.ipynb,
celda 28 (`DecisionTreeRegressor(random_state=42, max_depth=10,
min_samples_leaf=5)`, ajustado sobre `train_model_clean[features_5f]`). Ver
`scripts/serializar_modelo.py` para cómo se produjo `modelos/modelo_v1.pkl`.

`modelo_v1` es el champion: una vez serializado, congelado para siempre
(pliego Fase 1.5, sección "Qué NO hacer"). Los challengers de la Fase 5 se
guardan con su propio nombre (`challenger_YYYYMM.pkl`) y no pasan por aquí.
"""
import json
from pathlib import Path

import joblib

from src.paths import DIR_MODELOS

RUTA_MODELO_V1 = DIR_MODELOS / "modelo_v1.pkl"
RUTA_METADATOS_V1 = DIR_MODELOS / "modelo_v1.json"


def guardar_modelo(modelo, ruta: Path = RUTA_MODELO_V1) -> None:
    """Serializa `modelo` a `ruta` con joblib. No sobrescribe en silencio:
    quien llame a esto sobre modelo_v1.pkl ya existente lo hace a sabiendas
    (el champion está congelado; ver docstring del módulo)."""
    joblib.dump(modelo, ruta)


def cargar_modelo(ruta: Path = RUTA_MODELO_V1):
    """Carga un modelo serializado con joblib. Lanza si `ruta` no existe --
    sin fallback silencioso a un modelo sin entrenar."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Genera el modelo con scripts/serializar_modelo.py "
            "antes de cargarlo."
        )
    return joblib.load(ruta)


def guardar_metadatos(metadatos: dict, ruta: Path = RUTA_METADATOS_V1) -> None:
    """Escribe los metadatos del modelo (fecha de entrenamiento, rango de
    datos, MAE test, features) como JSON legible."""
    ruta.write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def cargar_metadatos(ruta: Path = RUTA_METADATOS_V1) -> dict:
    if not ruta.exists():
        raise FileNotFoundError(f"No existen metadatos en {ruta}.")
    return json.loads(ruta.read_text(encoding="utf-8"))
