# tests/test_predecir.py
"""Tests de `pipeline/predecir.py` (Fase 5bis, PR B: emisión paralela).
Solo funciones puras y de E/S sobre ficheros temporales -- datos sintéticos
en memoria, nunca e·sios ni ficheros del repo (mismo criterio que
`tests/test_evaluar.py`, pliego Fase 2 §5).

Se puede ejecutar como script (`python tests/test_predecir.py`) o recolectar
con pytest: las funciones `test_*` usan solo `assert` plano, sin fixtures ni
marks de pytest, mismo patrón que el resto de `tests/`."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import pipeline.predecir as predecir_mod
from pipeline.predecir import (
    COLUMNAS_ANCLA_USADA,
    COLUMNAS_PREDICCIONES,
    anadir_lag_168_prediccion,
    ancla_ya_registrada,
    guardar_ancla_usada,
    guardar_predicciones,
)


def _serie_referencia(n_horas: int, inicio="2026-08-01T00:00:00Z") -> pd.Series:
    base = pd.Timestamp(inicio, tz="UTC")
    idx = [base + pd.Timedelta(hours=i) for i in range(n_horas)]
    return pd.Series([30_000.0 + i for i in range(n_horas)], index=idx)


def _df_target_sintetico(n_horas: int, inicio="2026-08-19T00:00:00Z") -> pd.DataFrame:
    base = pd.Timestamp(inicio, tz="UTC")
    return pd.DataFrame({"datetime_utc": [base + pd.Timedelta(hours=i) for i in range(n_horas)]})


def test_anadir_lag_168_prediccion_ok():
    """Serie de referencia con las 168h+24h de margen cubiertas: sin NaN,
    no salta ningún assert."""
    df_target = _df_target_sintetico(24, inicio="2026-08-19T00:00:00Z")
    # T - 168h del primer horizonte (2026-08-19T00:00Z) es 2026-08-12T00:00Z:
    # la serie sintética arranca el 1/8, así que cubre de sobra.
    serie = _serie_referencia(24 * 25, inicio="2026-08-01T00:00:00Z")

    resultado = anadir_lag_168_prediccion(df_target, serie)

    assert "demanda_lag_168" in resultado.columns
    assert resultado["demanda_lag_168"].isna().sum() == 0
    # Búsqueda por timestamp, no por posición: el valor de la primera fila
    # debe ser exactamente el de la serie en T-168h, no el n-ésimo por índice.
    esperado = serie.loc[df_target["datetime_utc"].iloc[0] - pd.Timedelta(hours=168)]
    assert resultado["demanda_lag_168"].iloc[0] == esperado


def test_anadir_lag_168_prediccion_nan_salta():
    """Serie de referencia sin el margen de 168h (caso que el pliego pide
    vigilar, §3.2 punto 5): el assert de n_nan == 0 tiene que saltar, no
    colar una predicción parcial."""
    df_target = _df_target_sintetico(24, inicio="2026-08-19T00:00:00Z")
    # Serie que solo llega hasta 2026-08-15: T-168h (2026-08-12) no está.
    serie = _serie_referencia(24 * 3, inicio="2026-08-13T00:00:00Z")

    salto = False
    try:
        anadir_lag_168_prediccion(df_target, serie)
    except AssertionError:
        salto = True
    assert salto, "Debía saltar el assert de demanda_lag_168 con NaN"


def _predicciones_sinteticas(n: int, modelo: str, inicio="2026-08-19T00:00:00Z") -> pd.DataFrame:
    base = pd.Timestamp(inicio, tz="UTC")
    horizontes = [(base + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n)]
    return pd.DataFrame(
        {
            "fecha_emision": ["2026-08-19T05:45:00Z"] * n,
            "horizonte": horizontes,
            "valor_predicho": [30_000.0 + i for i in range(n)],
            "ancla_ultimo_dia_real": ["2026-08-18"] * n,
            "modelo": [modelo] * n,
        }
    )[COLUMNAS_PREDICCIONES]


def test_guardar_predicciones_dos_modelos_mismo_horizonte_no_colisionan():
    """Clave real de idempotencia: (horizonte, modelo) (pliego §3.2 punto 3).
    v1 y v2 escriben para los MISMOS horizontes el mismo día -- deben
    convivir sin pisarse, y un rerun de cada uno por separado no duplica."""
    with tempfile.TemporaryDirectory() as tmp:
        ruta_original = predecir_mod.RUTA_PREDICCIONES
        predecir_mod.RUTA_PREDICCIONES = Path(tmp) / "predicciones.csv"
        try:
            filas_v1 = _predicciones_sinteticas(24, "v1")
            filas_v2 = _predicciones_sinteticas(24, "v2")

            assert guardar_predicciones(filas_v1) is True
            assert guardar_predicciones(filas_v2) is True, (
                "v2 no debe colisionar con v1 aunque comparta todos los horizontes"
            )

            escritas = pd.read_csv(predecir_mod.RUTA_PREDICCIONES, dtype=str)
            assert len(escritas) == 48, f"Se esperaban 48 filas (24+24), hay {len(escritas)}"
            assert set(escritas["modelo"]) == {"v1", "v2"}

            # Rerun de cada modelo por separado: idempotente, no duplica.
            assert guardar_predicciones(filas_v1) is False
            assert guardar_predicciones(filas_v2) is False
            escritas_tras_rerun = pd.read_csv(predecir_mod.RUTA_PREDICCIONES, dtype=str)
            assert len(escritas_tras_rerun) == 48, "predicciones.csv no debe crecer en el rerun"
        finally:
            predecir_mod.RUTA_PREDICCIONES = ruta_original


def _ancla_sintetica(fecha_emision: str, ancla_ultimo_dia_real: str, n=24) -> pd.DataFrame:
    base = pd.Timestamp(f"{ancla_ultimo_dia_real}T00:00:00Z") - pd.Timedelta(hours=24)
    return pd.DataFrame(
        {
            "fecha_emision": [fecha_emision] * n,
            "datetime_utc_ancla": [
                (base + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n)
            ],
            "demanda_real": [30_000.0] * n,
            "n_lecturas": [12] * n,
            "ancla_ultimo_dia_real": [ancla_ultimo_dia_real] * n,
        }
    )[COLUMNAS_ANCLA_USADA]


def test_ancla_ya_registrada_sin_fichero():
    with tempfile.TemporaryDirectory() as tmp:
        ruta_original = predecir_mod.RUTA_ANCLA_USADA
        predecir_mod.RUTA_ANCLA_USADA = Path(tmp) / "ancla_usada.csv"
        try:
            dia_ancla = pd.Timestamp("2026-08-18T00:00:00+00:00")
            assert ancla_ya_registrada(dia_ancla) is False
        finally:
            predecir_mod.RUTA_ANCLA_USADA = ruta_original


def test_ancla_se_escribe_una_sola_vez_por_corrida_con_dos_modelos():
    """Simula la corrida de dos modelos el mismo día (pliego §3.2 punto 2):
    v1 escribe el ancla; cuando "v2" llega después con el mismo
    `dia_ancla`, `ancla_ya_registrada` debe verla y el llamador (main) no
    debe volver a escribirla -- si los dos la escribieran, se duplicaría."""
    with tempfile.TemporaryDirectory() as tmp:
        ruta_original = predecir_mod.RUTA_ANCLA_USADA
        predecir_mod.RUTA_ANCLA_USADA = Path(tmp) / "ancla_usada.csv"
        try:
            dia_ancla = pd.Timestamp("2026-08-18T00:00:00+00:00")

            # --- "v1" predice primero: escribe el ancla ---
            assert ancla_ya_registrada(dia_ancla) is False
            guardar_ancla_usada(_ancla_sintetica("2026-08-19T05:45:00Z", "2026-08-18"))
            assert ancla_ya_registrada(dia_ancla) is True

            filas_tras_v1 = pd.read_csv(predecir_mod.RUTA_ANCLA_USADA, dtype=str)
            assert len(filas_tras_v1) == 24

            # --- "v2" predice después, mismo ancla: main() NO debe volver a
            # llamar a guardar_ancla_usada porque ancla_ya_registrada ya es
            # True -- se comprueba aquí la condición, no una segunda escritura.
            assert ancla_ya_registrada(dia_ancla) is True, (
                "v2 debe ver el ancla ya escrita por v1 y no reescribirla"
            )
            filas_tras_v2 = pd.read_csv(predecir_mod.RUTA_ANCLA_USADA, dtype=str)
            assert len(filas_tras_v2) == 24, "ancla_usada.csv no debe crecer con el segundo modelo"
        finally:
            predecir_mod.RUTA_ANCLA_USADA = ruta_original


def main() -> int:
    test_anadir_lag_168_prediccion_ok()
    print("✔ test_anadir_lag_168_prediccion_ok")
    test_anadir_lag_168_prediccion_nan_salta()
    print("✔ test_anadir_lag_168_prediccion_nan_salta")
    test_guardar_predicciones_dos_modelos_mismo_horizonte_no_colisionan()
    print("✔ test_guardar_predicciones_dos_modelos_mismo_horizonte_no_colisionan")
    test_ancla_ya_registrada_sin_fichero()
    print("✔ test_ancla_ya_registrada_sin_fichero")
    test_ancla_se_escribe_una_sola_vez_por_corrida_con_dos_modelos()
    print("✔ test_ancla_se_escribe_una_sola_vez_por_corrida_con_dos_modelos")
    print("\nTodos los tests de predecir.py en verde.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
