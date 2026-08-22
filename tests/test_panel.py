# tests/test_panel.py
"""Tests de las funciones puras de `panel/panel.py` (Fase 6, pliego
`PLIEGO_Fase6_streamlit.md`, PR 2). Solo funciones puras, datos sintéticos
en memoria -- mismo criterio que `tests/test_grafico_deriva.py`.

No se testea la página en sí (`main`/`_pagina`, todo llamadas a `st.*`):
esa parte está bajo `if __name__ == "__main__":` precisamente para que
importar este módulo no la ejecute. Se verificó a mano con
`streamlit run panel/panel.py` (log del PR 2)."""
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

from panel.panel import (
    V1,
    V2,
    construir_grafico_curva,
    construir_grafico_deriva,
    dia_con_mayor_mae,
    extraer_fechas_presentes,
    horas_publicadas,
    techo_modelo,
    texto_v2_seguro,
)


def _fila_error(horizonte, modelo, valor_real, valor_predicho, h_adelanto_h):
    error = valor_real - valor_predicho
    return {
        "fecha_evaluacion": "2026-08-21T05:31:00Z",
        "horizonte": horizonte,
        "modelo": modelo,
        "fecha_emision": "2026-08-20T05:45:00Z",
        "valor_predicho": valor_predicho,
        "valor_real": valor_real,
        "error": error,
        "h_adelanto_h": h_adelanto_h,
    }


def test_horas_publicadas_solo_modelo_y_solo_publicado():
    errores = pd.DataFrame(
        [
            _fila_error("2026-08-14T06:00:00Z", "v1", 30_100.0, 30_000.0, 12.0),  # publicado
            _fila_error("2026-08-14T02:00:00Z", "v1", 30_900.0, 30_000.0, -4.0),  # diagnóstico
            _fila_error("2026-08-14T07:00:00Z", "v2", 30_500.0, 30_000.0, 12.0),  # otro modelo
        ]
    )
    publicado = horas_publicadas(errores, "v1")
    assert len(publicado) == 1
    assert publicado.iloc[0]["valor_real"] == 30_100.0
    assert list(publicado.columns).count("fecha") == 1


def test_horas_publicadas_vacio_si_no_hay_filas():
    errores = pd.DataFrame(
        [_fila_error("2026-08-14T02:00:00Z", "v2", 30_900.0, 30_000.0, -4.0)]
    )
    publicado = horas_publicadas(errores, "v2")
    assert publicado.empty


def test_dia_con_mayor_mae():
    """Semana de ejemplo del pliego: el 17/8 tiene el MAE más alto -- debe
    ganar aunque no sea ni el primer ni el último día de la serie."""
    errores = pd.DataFrame(
        [
            _fila_error("2026-08-16T06:00:00Z", "v1", 30_100.0, 30_000.0, 12.0),  # |100|
            _fila_error("2026-08-17T06:00:00Z", "v1", 35_000.0, 30_000.0, 12.0),  # |5000|
            _fila_error("2026-08-18T06:00:00Z", "v1", 30_050.0, 30_000.0, 12.0),  # |50|
        ]
    )
    publicado = horas_publicadas(errores, "v1")
    import datetime

    assert dia_con_mayor_mae(publicado) == datetime.date(2026, 8, 17)


def test_dia_con_mayor_mae_vacio():
    assert dia_con_mayor_mae(pd.DataFrame(columns=["fecha", "error"])) is None


def test_techo_modelo_calculado_en_vivo_por_modelo():
    predicciones = pd.DataFrame(
        {
            "modelo": ["v1", "v1", "v2"],
            "valor_predicho": [38_000.0, 38_861.1, 30_000.0],
        }
    )
    assert techo_modelo(predicciones, "v1") == 38_861.1
    assert techo_modelo(predicciones, "v2") == 30_000.0


def test_techo_modelo_none_si_falta_o_vacio():
    assert techo_modelo(None, "v1") is None
    predicciones = pd.DataFrame({"modelo": ["v2"], "valor_predicho": [1.0]})
    assert techo_modelo(predicciones, "v1") is None


def test_construir_grafico_curva_dos_series_y_techo():
    """Pliego (PR gráficos interactivos): un modelo, dos líneas (real y
    predicho), el techo como línea horizontal aparte -- nunca una tercera
    serie de datos."""
    dia_df = pd.DataFrame(
        {
            "horizonte_madrid": pd.to_datetime(
                ["2026-08-17 10:00", "2026-08-17 11:00"]
            ).tz_localize("Europe/Madrid"),
            "valor_real": [30_000.0, 31_000.0],
            "valor_predicho": [29_500.0, 30_400.0],
            "fecha": [pd.Timestamp("2026-08-17").date()] * 2,
        }
    )
    fig = construir_grafico_curva(dia_df, "v1", techo=38_861.1)
    nombres = [t.name for t in fig.data]
    assert nombres == ["Real", "Predicho"]  # exactamente dos series, en ese orden
    assert len(fig.layout.shapes) == 1  # la línea del techo, no una tercera serie


def test_construir_grafico_curva_sin_techo_no_dibuja_linea():
    dia_df = pd.DataFrame(
        {
            "horizonte_madrid": pd.to_datetime(["2026-08-17 10:00"]).tz_localize("Europe/Madrid"),
            "valor_real": [30_000.0],
            "valor_predicho": [29_500.0],
            "fecha": [pd.Timestamp("2026-08-17").date()],
        }
    )
    fig = construir_grafico_curva(dia_df, "v1", techo=None)
    assert not fig.layout.shapes


def test_construir_grafico_deriva_dos_paneles():
    """Dos series (MAE, sesgo) repartidas en dos filas de subplots -- el
    mismo dato que ya calcula scripts/grafico_deriva.py, dibujado aparte."""
    serie = pd.DataFrame(
        {
            "fecha": [pd.Timestamp("2026-08-16").date(), pd.Timestamp("2026-08-17").date()],
            "n": [16, 16],
            "mae": [2000.0, 5650.0],
            "sesgo": [500.0, -300.0],
        }
    )
    import datetime

    fig = construir_grafico_deriva(serie, dias_marcados={datetime.date(2026, 8, 17)})
    nombres = [t.name for t in fig.data]
    assert nombres == ["MAE diario", "Sesgo diario"]


def test_extraer_fechas_presentes():
    estado_md = (
        "## Modelo: v1\n\n"
        "Fechas presentes en `data/errores.csv` para este modelo "
        "(publicadas + diagnóstico): 10\n"
    )
    assert extraer_fechas_presentes(estado_md, V1) == 10
    assert extraer_fechas_presentes(estado_md, V2) is None


def test_texto_v2_seguro_sin_bloque_v2():
    """pliego §2.6: metricas.json sin el bloque de v2 -- el panel funciona
    igual, no revienta."""
    metricas = {"v1": {"publicado": {"ventanas": {"90d": {"dias_cubiertos": 7, "n_horas": 100}}}}}
    texto = texto_v2_seguro(metricas)
    assert "v2" in texto
    assert "MAE" not in texto and "sesgo" not in texto.lower()


def test_texto_v2_seguro_con_bloque_v2():
    metricas = {
        "v1": {"publicado": {"ventanas": {"90d": {"dias_cubiertos": 7, "n_horas": 100}}}, "muestra_insuficiente": False},
        "v2": {"publicado": {"ventanas": {"90d": {"dias_cubiertos": 0, "n_horas": 0}}}, "muestra_insuficiente": True},
    }
    texto = texto_v2_seguro(metricas)
    assert "sin corridas evaluadas" in texto


def main() -> int:
    test_horas_publicadas_solo_modelo_y_solo_publicado()
    print("✔ test_horas_publicadas_solo_modelo_y_solo_publicado")
    test_horas_publicadas_vacio_si_no_hay_filas()
    print("✔ test_horas_publicadas_vacio_si_no_hay_filas")
    test_dia_con_mayor_mae()
    print("✔ test_dia_con_mayor_mae")
    test_dia_con_mayor_mae_vacio()
    print("✔ test_dia_con_mayor_mae_vacio")
    test_techo_modelo_calculado_en_vivo_por_modelo()
    print("✔ test_techo_modelo_calculado_en_vivo_por_modelo")
    test_techo_modelo_none_si_falta_o_vacio()
    print("✔ test_techo_modelo_none_si_falta_o_vacio")
    test_construir_grafico_curva_dos_series_y_techo()
    print("✔ test_construir_grafico_curva_dos_series_y_techo")
    test_construir_grafico_curva_sin_techo_no_dibuja_linea()
    print("✔ test_construir_grafico_curva_sin_techo_no_dibuja_linea")
    test_construir_grafico_deriva_dos_paneles()
    print("✔ test_construir_grafico_deriva_dos_paneles")
    test_extraer_fechas_presentes()
    print("✔ test_extraer_fechas_presentes")
    test_texto_v2_seguro_sin_bloque_v2()
    print("✔ test_texto_v2_seguro_sin_bloque_v2")
    test_texto_v2_seguro_con_bloque_v2()
    print("✔ test_texto_v2_seguro_con_bloque_v2")
    print("\nTodos los tests de panel.py en verde.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
