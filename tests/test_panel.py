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
    dia_con_mayor_mae,
    extraer_fechas_presentes,
    extraer_seccion_notebook,
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


def test_extraer_seccion_notebook_v1_presente_v2_ausente():
    """v1 tiene la sección "Por qué el MAE..."; v2 (sin referencia de
    notebook propia) tiene un encabezado distinto ("Referencia de
    notebook") -- extraer_seccion_notebook debe devolver None para v2, no
    inventar ni recortar mal."""
    estado_md = (
        "## Modelo: v1\n\n"
        "### Por qué el MAE de producción no coincide con el del notebook\n\n"
        "Texto de v1.\n\n"
        "---\n\n"
        "## Modelo: v2\n\n"
        "### Referencia de notebook\n\n"
        "Texto de v2, sin sección de notebook.\n"
    )
    seccion_v1 = extraer_seccion_notebook(estado_md, V1)
    assert seccion_v1 is not None
    assert "Texto de v1." in seccion_v1
    assert "Texto de v2" not in seccion_v1  # no se cuela el bloque de v2

    assert extraer_seccion_notebook(estado_md, V2) is None


def test_extraer_seccion_notebook_modelo_ausente():
    assert extraer_seccion_notebook("## Modelo: v1\n\ncontenido", "v2") is None


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
    test_extraer_seccion_notebook_v1_presente_v2_ausente()
    print("✔ test_extraer_seccion_notebook_v1_presente_v2_ausente")
    test_extraer_seccion_notebook_modelo_ausente()
    print("✔ test_extraer_seccion_notebook_modelo_ausente")
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
