# tests/test_paridad.py
"""Test de paridad obligatorio (Fase 1.5, pliego): src/ debe reproducir
EXACTAMENTE (tolerancia 0) lo que produjo notebooks/modelo_demanda.ipynb
sobre un conjunto fijo de fechas.

La fixture dorada (`tests/fixtures/paridad_golden.parquet`) no es una
reimplementación aparte: es un volcado directo de los objetos en memoria del
propio notebook, generado por `scripts/serializar_modelo.py`. Si este test
falla, `src/features.py` o `src/modelo.py` divergen del notebook -- no se
"arregla" ajustando la fixture, se para y se reporta (misma regla que el
gate de `scripts/gate_numeros.py`).

Se puede ejecutar como script (sin pytest, no está en el entorno del
proyecto hoy) o recolectar con pytest si se añade más adelante: las
funciones `test_*` usan solo `assert` plano.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# La consola de Windows usa cp1252 por defecto, que no representa "✔" (mismo
# problema que scripts/gate_numeros.py).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.features import FEATURES_5F, construir_features_5f
from src.modelo import cargar_modelo
from src.paths import DIR_PROCESSED

RUTA_FIXTURE = Path(__file__).parent / "fixtures" / "paridad_golden.parquet"
RUTA_DEMANDA_HORARIA = DIR_PROCESSED / "demanda_horaria.parquet"


def _construir_features_sobre_historico() -> pd.DataFrame:
    """src/features.py necesita la serie horaria completa y contigua para
    calcular demanda_lag_24 (shift posicional, celda 20) -- no se puede
    construir solo sobre las fechas de la fixture."""
    df = pd.read_parquet(RUTA_DEMANDA_HORARIA)
    return construir_features_5f(df)


def test_features_reproducen_notebook():
    fixture = pd.read_parquet(RUTA_FIXTURE)
    df_features = _construir_features_sobre_historico().set_index("datetime_utc")

    faltantes = set(fixture["datetime_utc"]) - set(df_features.index)
    assert not faltantes, f"Fechas de la fixture ausentes en src/features.py: {faltantes}"

    for _, fila in fixture.iterrows():
        actual = df_features.loc[fila["datetime_utc"]]
        for col in FEATURES_5F:
            valor_actual = actual[col]
            valor_esperado = fila[col]
            assert valor_actual == valor_esperado, (
                f"{fila['datetime_utc']} / {col}: "
                f"src/features.py={valor_actual!r} != notebook={valor_esperado!r}"
            )
        # es_evento no es una feature del modelo (no está en FEATURES_5F): se
        # comprueba aparte como ancla de que la fila es la fila correcta.
        assert bool(actual["es_evento"]) == bool(fila["es_evento"]), (
            f"{fila['datetime_utc']}: es_evento no coincide -- ¿es la misma fila?"
        )


def test_predicciones_reproducen_notebook():
    fixture = pd.read_parquet(RUTA_FIXTURE)
    df_features = _construir_features_sobre_historico().set_index("datetime_utc")
    modelo = cargar_modelo()

    X = df_features.loc[fixture["datetime_utc"], FEATURES_5F]
    predicciones = modelo.predict(X)

    for (_, fila), pred in zip(fixture.iterrows(), predicciones):
        esperado = fila["prediccion_modelo_v1"]
        assert pred == esperado, (
            f"{fila['datetime_utc']}: predicción src/={pred!r} != notebook={esperado!r}"
        )


def main() -> int:
    test_features_reproducen_notebook()
    print("✔ test_features_reproducen_notebook — src/features.py reproduce el notebook.")
    test_predicciones_reproducen_notebook()
    print("✔ test_predicciones_reproducen_notebook — modelo_v1.pkl reproduce el notebook.")
    print("\nParidad OK — tolerancia 0 sobre 28 filas fijas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
