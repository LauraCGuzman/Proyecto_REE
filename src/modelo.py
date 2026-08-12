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

`cargar_modelo` exige que la versión de scikit-learn instalada coincida
exactamente con la que grabó `scripts/serializar_modelo.py` en los
metadatos. Un `DecisionTreeRegressor` no tiene garantía de compatibilidad
binaria entre versiones de sklearn -- el propio pickle emite
`InconsistentVersionWarning` en ese caso, pero es solo un warning: no para
la ejecución. Aquí sí para, con la versión exacta que falta en el mensaje.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.paths import DIR_MODELOS

RUTA_MODELO_V1 = DIR_MODELOS / "modelo_v1.pkl"
RUTA_METADATOS_V1 = DIR_MODELOS / "modelo_v1.json"


def guardar_modelo(modelo, ruta: Path = RUTA_MODELO_V1) -> None:
    """Serializa `modelo` a `ruta` con joblib. No sobrescribe en silencio:
    quien llame a esto sobre modelo_v1.pkl ya existente lo hace a sabiendas
    (el champion está congelado; ver docstring del módulo)."""
    joblib.dump(modelo, ruta)


def _ruta_metadatos_de(ruta_modelo: Path) -> Path:
    """modelo_v1.pkl -> modelo_v1.json (misma convención para los
    challengers de la Fase 5: challenger_YYYYMM.pkl -> challenger_YYYYMM.json)."""
    return ruta_modelo.with_suffix(".json")


def cargar_modelo(ruta: Path = RUTA_MODELO_V1):
    """Carga un modelo serializado con joblib. Lanza si `ruta` no existe --
    sin fallback silencioso a un modelo sin entrenar -- y lanza también si
    la versión de scikit-learn instalada no es exactamente la que grabaron
    los metadatos (ver docstring del módulo)."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Genera el modelo con scripts/serializar_modelo.py "
            "antes de cargarlo."
        )

    metadatos = cargar_metadatos(_ruta_metadatos_de(ruta))
    version_esperada = metadatos.get("version_sklearn")
    if version_esperada is not None and version_esperada != sklearn.__version__:
        raise RuntimeError(
            f"scikit-learn instalado ({sklearn.__version__}) no coincide con la "
            f"versión con la que se serializó {ruta.name} ({version_esperada}). "
            "Un DecisionTreeRegressor no tiene garantía de compatibilidad binaria "
            "entre versiones de sklearn: instala la versión exacta "
            f"(pip install scikit-learn=={version_esperada}) o vuelve a serializar "
            "el modelo con la versión instalada (scripts/serializar_modelo.py --forzar)."
        )

    return joblib.load(ruta)


def predecir(modelo, X: pd.DataFrame):
    """`modelo.predict(X)` defensivo: exige que las columnas de `X`
    coincidan en nombre Y ORDEN exactos con `modelo.feature_names_in_` (las
    que vio al entrenar). sklearn no siempre lo garantiza por sí solo en
    `.predict()` sobre un DataFrame; en un pipeline desatendido, un
    DataFrame con las columnas en otro orden no debe devolver una
    predicción silenciosamente equivocada."""
    columnas_modelo = list(modelo.feature_names_in_)
    columnas_x = list(X.columns)
    assert columnas_x == columnas_modelo, (
        "Las columnas de X no coinciden (en nombre u orden) con las del "
        f"modelo. Esperado: {columnas_modelo} | Recibido: {columnas_x}"
    )
    return modelo.predict(X)


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
