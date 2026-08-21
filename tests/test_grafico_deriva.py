# tests/test_grafico_deriva.py
"""Tests de `scripts/grafico_deriva.py` (Fase 4, pliego
`PLIEGO_Fase4_drift.md`). Solo funciones puras, datos sintéticos en
memoria -- mismo criterio que `tests/test_evaluar.py`."""
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

from scripts.grafico_deriva import (
    _serie_con_huecos_explicitos,
    _texto_cobertura_v2,
    cobertura_v2,
    dias_no_laborable_a_laborable,
    serie_diaria_v1,
)


def _fila_error(horizonte, modelo, error, h_adelanto_h):
    return {
        "fecha_evaluacion": "2026-08-21T05:31:00Z",
        "horizonte": horizonte,
        "modelo": modelo,
        "fecha_emision": "2026-08-20T05:45:00Z",
        "valor_predicho": 30_000.0,
        "valor_real": 30_000.0 + error,
        "error": error,
        "h_adelanto_h": h_adelanto_h,
    }


def test_serie_diaria_v1_solo_publicado_y_solo_v1():
    """Excluye diagnóstico (h_adelanto_h <= 0) y excluye otros modelos --
    pliego §3.1 y §3.6 regla 1 (no agregar publicado con diagnóstico)."""
    errores = pd.DataFrame(
        [
            _fila_error("2026-08-14T06:00:00Z", "v1", 100.0, 12.0),   # publicado
            _fila_error("2026-08-14T02:00:00Z", "v1", 900.0, -4.0),   # diagnóstico -- excluido
            _fila_error("2026-08-14T07:00:00Z", "v2", 500.0, 12.0),   # otro modelo -- excluido
            _fila_error("2026-08-15T06:00:00Z", "v1", -200.0, 12.0),
            _fila_error("2026-08-15T07:00:00Z", "v1", 200.0, 12.0),
        ]
    )

    serie = serie_diaria_v1(errores)

    assert list(serie["fecha"].astype(str)) == ["2026-08-14", "2026-08-15"]
    fila_14 = serie[serie["fecha"].astype(str) == "2026-08-14"].iloc[0]
    assert fila_14["n"] == 1
    assert fila_14["mae"] == 100.0
    assert fila_14["sesgo"] == 100.0

    fila_15 = serie[serie["fecha"].astype(str) == "2026-08-15"].iloc[0]
    assert fila_15["n"] == 2
    assert fila_15["mae"] == 200.0  # mean(|-200|, |200|)
    assert fila_15["sesgo"] == 0.0  # mean(-200, 200)


def test_serie_diaria_v1_vacia_sin_horas_publicadas():
    errores = pd.DataFrame([_fila_error("2026-08-14T02:00:00Z", "v1", 900.0, -4.0)])
    serie = serie_diaria_v1(errores)
    assert serie.empty


def test_dias_no_laborable_a_laborable_marca_el_lunes_17_8():
    """Semana de ejemplo del pliego (§3.1): el único día que transiciona de
    no_laborable a laborable es el lunes 17/8/2026 (16/8 es domingo)."""
    dias = dias_no_laborable_a_laborable("2026-08-14", "2026-08-20")
    import datetime

    assert dias == {datetime.date(2026, 8, 17)}


def test_serie_con_huecos_explicitos_no_rellena_con_cero():
    """Un día sin fila en la serie original queda como NaN al reindexar, no
    como 0 -- pliego §4, "un día sin horas evaluadas es un hueco, no un
    cero"."""
    serie = pd.DataFrame(
        {
            "fecha": [pd.Timestamp("2026-08-14").date(), pd.Timestamp("2026-08-16").date()],
            "n": [16, 16],
            "mae": [1000.0, 2000.0],
            "sesgo": [100.0, -200.0],
        }
    )
    con_huecos = _serie_con_huecos_explicitos(serie)

    assert len(con_huecos) == 3  # 14, 15 (hueco), 16
    fila_15 = con_huecos[con_huecos["fecha"].astype(str) == "2026-08-15"].iloc[0]
    assert pd.isna(fila_15["mae"])
    assert pd.isna(fila_15["n"])
    assert fila_15["mae"] != 0.0 and not (fila_15["mae"] == 0)  # explícitamente no es un cero


def test_cobertura_v2_sin_filas():
    """Caso real hasta el 22/8/2026 (pliego §4): v2 sin ninguna fila en
    errores.csv -- cobertura_v2 no debe lanzar, y el texto no debe sugerir
    una serie ni un cero como si fuera una medida."""
    metricas = {
        "v1": {"publicado": {"ventanas": {"90d": {"dias_cubiertos": 7, "n_horas": 100}}}, "muestra_insuficiente": False},
        "v2": {"publicado": {"ventanas": {"90d": {"dias_cubiertos": 0, "n_horas": 0}}}, "muestra_insuficiente": True},
    }
    cobertura = cobertura_v2(metricas)
    assert cobertura["n_horas"] == 0
    texto = _texto_cobertura_v2(cobertura)
    assert "sin corridas evaluadas" in texto
    assert "MAE" not in texto and "sesgo" not in texto.lower()


def test_cobertura_v2_con_filas():
    metricas = {
        "v1": {"publicado": {"ventanas": {"90d": {"dias_cubiertos": 7, "n_horas": 100}}}, "muestra_insuficiente": False},
        "v2": {"publicado": {"ventanas": {"90d": {"dias_cubiertos": 3, "n_horas": 40}}}, "muestra_insuficiente": True},
    }
    cobertura = cobertura_v2(metricas)
    texto = _texto_cobertura_v2(cobertura)
    assert "3 fecha" in texto
    assert "40 horas" in texto
    # Nunca un MAE ni un sesgo de v2 en el texto (pliego §3.5).
    assert "MAE" not in texto and "Sesgo" not in texto and "sesgo" not in texto.lower()


def main() -> int:
    test_serie_diaria_v1_solo_publicado_y_solo_v1()
    print("✔ test_serie_diaria_v1_solo_publicado_y_solo_v1")
    test_serie_diaria_v1_vacia_sin_horas_publicadas()
    print("✔ test_serie_diaria_v1_vacia_sin_horas_publicadas")
    test_dias_no_laborable_a_laborable_marca_el_lunes_17_8()
    print("✔ test_dias_no_laborable_a_laborable_marca_el_lunes_17_8")
    test_serie_con_huecos_explicitos_no_rellena_con_cero()
    print("✔ test_serie_con_huecos_explicitos_no_rellena_con_cero")
    test_cobertura_v2_sin_filas()
    print("✔ test_cobertura_v2_sin_filas")
    test_cobertura_v2_con_filas()
    print("✔ test_cobertura_v2_con_filas")
    print("\nTodos los tests de grafico_deriva.py en verde.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
