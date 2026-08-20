# tests/test_paridad_v2.py
"""Test de paridad de v2 (pliego `PLIEGO_Fase5bis_v2_lag168.md`, Parte 2 — PR A):
`src/features_v2.py` debe reproducir EXACTAMENTE (tolerancia 0) lo que produjo
`notebooks/modelo_v2_lag168.ipynb`, sobre un conjunto fijo de fechas.

Mismo patrón que `tests/test_paridad.py` para v1 -- ese fichero y su fixture
(`tests/fixtures/paridad_golden.parquet`) no se tocan aquí. v2 lleva la suya:
`tests/fixtures/paridad_v2_golden.parquet`, generada por
`scripts/serializar_modelo_v2.py`.

Diferencia deliberada con v1: la fixture de v1 sale de `test_model` (2026,
test). La de v2 sale de `train_clean_6f` (2023-2025, train) porque este PR no
lee el conjunto de test ni una vez -- el oráculo de paridad es igual de
válido sobre filas de train, ya que solo comprueba que `src/` reproduce
aritméticamente lo que hizo el notebook, no la capacidad de generalización
del modelo.

Se puede ejecutar como script (`python tests/test_paridad_v2.py`) o recolectar
con pytest: las funciones `test_*` usan solo `assert` plano, sin fixtures ni
marks de pytest, para que ambas formas de ejecutarlo hagan exactamente lo mismo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.features_v2 import FEATURES_6F, construir_features_6f
from src.modelo import cargar_modelo, predecir
from src.paths import DIR_MODELOS, DIR_PROCESSED

RUTA_FIXTURE = Path(__file__).parent / "fixtures" / "paridad_v2_golden.parquet"
RUTA_DEMANDA_HORARIA = DIR_PROCESSED / "demanda_horaria.parquet"
RUTA_MODELO_V2 = DIR_MODELOS / "modelo_v2.pkl"


def _construir_features_sobre_historico() -> pd.DataFrame:
    """src/features_v2.py necesita la serie horaria completa y contigua para
    calcular demanda_lag_24 y demanda_lag_168 (shift posicional) -- no se
    puede construir solo sobre las fechas de la fixture."""
    df = pd.read_parquet(RUTA_DEMANDA_HORARIA)
    return construir_features_6f(df)


def test_features_v2_reproducen_notebook():
    fixture = pd.read_parquet(RUTA_FIXTURE)
    df_features = _construir_features_sobre_historico().set_index("datetime_utc")

    faltantes = set(fixture["datetime_utc"]) - set(df_features.index)
    assert not faltantes, f"Fechas de la fixture ausentes en src/features_v2.py: {faltantes}"

    for _, fila in fixture.iterrows():
        actual = df_features.loc[fila["datetime_utc"]]
        for col in FEATURES_6F:
            valor_actual = actual[col]
            valor_esperado = fila[col]
            assert valor_actual == valor_esperado, (
                f"{fila['datetime_utc']} / {col}: "
                f"src/features_v2.py={valor_actual!r} != notebook={valor_esperado!r}"
            )
        assert bool(actual["es_evento"]) == bool(fila["es_evento"]), (
            f"{fila['datetime_utc']}: es_evento no coincide -- ¿es la misma fila?"
        )


def test_predicciones_v2_reproducen_notebook():
    fixture = pd.read_parquet(RUTA_FIXTURE)
    df_features = _construir_features_sobre_historico().set_index("datetime_utc")
    modelo = cargar_modelo(ruta=RUTA_MODELO_V2)

    X = df_features.loc[fixture["datetime_utc"], FEATURES_6F]
    predicciones = predecir(modelo, X)

    for (_, fila), pred in zip(fixture.iterrows(), predicciones):
        esperado = fila["prediccion_modelo_v2"]
        assert pred == esperado, (
            f"{fila['datetime_utc']}: predicción src/={pred!r} != notebook={esperado!r}"
        )


def main() -> int:
    test_features_v2_reproducen_notebook()
    print("✔ test_features_v2_reproducen_notebook — src/features_v2.py reproduce el notebook.")
    test_predicciones_v2_reproducen_notebook()
    print("✔ test_predicciones_v2_reproducen_notebook — modelo_v2.pkl reproduce el notebook.")
    n_filas = len(pd.read_parquet(RUTA_FIXTURE))
    print(f"\nParidad v2 OK — tolerancia 0 sobre {n_filas} filas fijas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
