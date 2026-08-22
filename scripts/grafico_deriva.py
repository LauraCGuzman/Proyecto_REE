"""Genera el PNG de deriva del error diario de v1 (Fase 4, pliego
`PLIEGO_Fase4_drift.md`).

Script de SOLO LECTURA (pliego, §0): no toca `pipeline/predecir.py`,
`pipeline/evaluar.py`, `data/errores.csv`, `data/metricas.json` ni
`reports/estado_pipeline.md`. Lee los dos primeros, escribe un PNG.

`data/metricas.json` se lee con el esquema de dos bloques que dejó el PR B
de Fase 5bis (`{"v1": {...}, "v2": {...}}`) -- si el fichero real todavía
no tiene esa forma (porque `evaluar.py` con el cambio no ha corrido en
producción ni una vez), este script para en rojo con un mensaje claro en
vez de adivinar un esquema antiguo.

La referencia de notebook (1.263,02 MW) NO se lee de `metricas.json`: es la
misma constante congelada que `pipeline/evaluar.py::MAE_TEST_NOTEBOOK_MW`
(gate de re-entrenamiento, `scripts/gate_numeros.py`), copiada aquí con el
mismo criterio -- no es una dependencia en tiempo de ejecución del pipeline,
y depender de `metricas.json` para un número congelado ataría este script a
que el pipeline ya hubiera corrido con el esquema nuevo.

Uso:
    python scripts/grafico_deriva.py                     # normal: escribe reports/deriva_error_diario.png
    python scripts/grafico_deriva.py --salida ruta.png    # verificación: no toca reports/ real
    python scripts/grafico_deriva.py --metricas ruta.json # verificación: no depende del metricas.json real
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import anadir_calendario, anadir_festivos_puentes, mapear_tipo_efectivo
from src.paths import DIR_DATA, DIR_REPORTS

TZ_MADRID = "Europe/Madrid"

NOMBRE_MODELO = "v1"
NOMBRE_MODELO_V2 = "v2"

RUTA_ERRORES = DIR_DATA / "errores.csv"
RUTA_METRICAS = DIR_DATA / "metricas.json"
RUTA_PNG_DEFECTO = DIR_REPORTS / "deriva_error_diario.png"

# Ver docstring del módulo: congelada, no leída de metricas.json.
MAE_TEST_NOTEBOOK_MW = 1263.02

# Mismo umbral que pipeline/evaluar.py::COBERTURA_MINIMA_DIAS, para
# interpretar "muestra_insuficiente" del bloque de v2 si algún día se usa.
COBERTURA_MINIMA_DIAS = 7


def _leer_csv(ruta: Path) -> pd.DataFrame:
    """Mismo criterio que `pipeline/evaluar.py::_leer_csv`:
    `float_precision='round_trip'`, sin eso el parser de floats de pandas
    pierde 1 ULP."""
    return pd.read_csv(ruta, float_precision="round_trip")


def _leer_metricas(ruta: Path) -> dict:
    metricas = json.loads(Path(ruta).read_text(encoding="utf-8"))
    assert NOMBRE_MODELO in metricas and NOMBRE_MODELO_V2 in metricas, (
        f"{ruta} no tiene el esquema de dos bloques del PR B de Fase5bis "
        f'(se esperaban las claves "{NOMBRE_MODELO}" y "{NOMBRE_MODELO_V2}", '
        f"hay {sorted(metricas.keys())}). Probablemente evaluar.py con el "
        "cambio de emisión paralela todavía no ha corrido en producción."
    )
    return metricas


def serie_diaria_v1(errores: pd.DataFrame) -> pd.DataFrame:
    """MAE y sesgo diarios de v1, SOLO horas publicadas (`h_adelanto_h > 0`,
    pliego §3.1) -- las de diagnóstico no entran, ni sumadas ni en una
    segunda serie (regla de no agregar publicado con diagnóstico, §3.6
    regla 1 de Fase 2, que este PR no relaja).

    Un día sin ninguna hora publicada evaluada NO aparece en el resultado:
    es un hueco (pliego §4, "caso de día vacío"), no una fila con `n=0`, y
    el llamador es quien decide cómo dibujar ese hueco (ver
    `_serie_con_huecos_explicitos`)."""
    publicado = errores[
        (errores["modelo"] == NOMBRE_MODELO) & (errores["h_adelanto_h"] > 0)
    ]
    if publicado.empty:
        return pd.DataFrame(columns=["fecha", "n", "mae", "sesgo"])

    horizonte_madrid = pd.to_datetime(publicado["horizonte"], utc=True).dt.tz_convert(
        TZ_MADRID
    )
    con_fecha = publicado.assign(fecha=horizonte_madrid.dt.date)
    agregado = (
        con_fecha.groupby("fecha")["error"]
        .agg(n="count", mae=lambda s: s.abs().mean(), sesgo="mean")
        .reset_index()
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    return agregado


def _serie_con_huecos_explicitos(serie: pd.DataFrame) -> pd.DataFrame:
    """Reindexa `serie` (salida de `serie_diaria_v1`) al rango continuo de
    fechas [min, max], con NaN -- no 0 -- en las fechas ausentes. Un `plot`
    de matplotlib no traza segmento sobre un NaN: el hueco se ve como hueco,
    no como una interpolación silenciosa entre los días que sí hay dato
    (pliego §4: "un día sin horas evaluadas es un hueco, no un cero")."""
    if serie.empty:
        return serie
    rango_completo = pd.date_range(serie["fecha"].min(), serie["fecha"].max(), freq="D").date
    return (
        serie.set_index("fecha")
        .reindex(rango_completo)
        .rename_axis("fecha")
        .reset_index()
    )


def dias_no_laborable_a_laborable(fecha_min, fecha_max) -> set:
    """Fechas de calendario en [`fecha_min`, `fecha_max`] cuyo
    `tipo_efectivo` pasa de no_laborable (1) a laborable (0) respecto al día
    de calendario anterior -- el régimen que describe
    `pipeline/evaluar.py` ("El régimen de transición no laborable →
    laborable (los lunes)..."). En la semana de ejemplo del pliego coincide
    exactamente con el lunes 17/8 (16/8 domingo -> 17/8 laborable), pero la
    definición es la del régimen, no el nombre del día -- un martes tras un
    lunes festivo es el mismo caso.

    Reutiliza `anadir_calendario`, `anadir_festivos_puentes` y
    `mapear_tipo_efectivo` de `src/features.py` (pliego §2.3/§3.4): NO
    reimplementa el cálculo de `tipo_efectivo`. Se alimenta con el
    mediodía local de cada fecha (no medianoche) para no rozar ninguna
    frontera de cambio de hora al convertir a UTC."""
    rango_local = pd.date_range(
        pd.Timestamp(fecha_min) - pd.Timedelta(days=1), fecha_max, freq="D", tz=TZ_MADRID
    )
    mediodia_utc = (rango_local + pd.Timedelta(hours=12)).tz_convert("UTC")

    calendario = pd.DataFrame({"datetime_utc": mediodia_utc})
    calendario = anadir_calendario(calendario)
    calendario = anadir_festivos_puentes(calendario)
    calendario = mapear_tipo_efectivo(calendario)
    calendario["fecha"] = rango_local.date

    transicion = (calendario["tipo_efectivo"].shift(1) == 1) & (
        calendario["tipo_efectivo"] == 0
    )
    return set(calendario.loc[transicion, "fecha"])


def cobertura_v2(metricas: dict) -> dict:
    """Resumen de cobertura de v2 a partir de `metricas.json` (única fuente
    permitida junto a `errores.csv`, pliego §0) -- NUNCA una serie de error:
    v2 no tiene ni MAE ni sesgo dibujado en este PR (pliego §3.5). Usa la
    ventana de 90 días como aproximación de "toda la cobertura hasta ahora"
    -- la ventana de promoción de Fase5bis son 6 semanas (42 días), muy por
    debajo de 90, así que no recorta nada mientras esa ventana siga abierta."""
    bloque_v2 = metricas[NOMBRE_MODELO_V2]
    ventana_90d = bloque_v2["publicado"]["ventanas"]["90d"]
    return {
        "dias_cubiertos": ventana_90d["dias_cubiertos"],
        "n_horas": ventana_90d["n_horas"],
        "muestra_insuficiente": bloque_v2["muestra_insuficiente"],
    }


def _texto_cobertura_v2(cobertura: dict) -> str:
    if cobertura["n_horas"] == 0:
        return "v2: sin corridas evaluadas todavía (estado real hasta el 22/8/2026)."
    estado = "muestra insuficiente todavía" if cobertura["muestra_insuficiente"] else "en régimen"
    return (
        f"v2: {cobertura['dias_cubiertos']} fecha(s) con horas publicadas, "
        f"{cobertura['n_horas']} horas evaluadas -- {estado}. "
        "Sin serie de error: no se evalúa hasta el 3/10/2026 (pliego Fase5bis)."
    )


def construir_grafico(serie: pd.DataFrame, dias_marcados: set, texto_v2: str):
    """Dos paneles apilados, eje de tiempo compartido (pliego §3.2): MAE
    arriba (con la referencia de notebook, §3.3, como línea horizontal
    etiquetada -- nunca como umbral), sesgo abajo (con el cero marcado,
    porque el signo es la información). Días de transición no_laborable ->
    laborable resaltados en los dos paneles (§3.4). Días con `n` por debajo
    de la moda de la serie, anotados con su `n` -- no ocultados ni tratados
    como los demás (§3.1, "El 14/8 tiene 15 horas, no 16")."""
    fig, (ax_mae, ax_sesgo) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    x = pd.to_datetime(serie["fecha"])

    for dia in dias_marcados:
        dia_ts = pd.Timestamp(dia)
        if x.min() <= dia_ts <= x.max():
            for ax in (ax_mae, ax_sesgo):
                ax.axvspan(
                    dia_ts - pd.Timedelta(hours=12),
                    dia_ts + pd.Timedelta(hours=12),
                    color="gold",
                    alpha=0.25,
                    zorder=0,
                )

    ax_mae.plot(x, serie["mae"], marker="o", color="tab:blue")
    ax_mae.axhline(
        MAE_TEST_NOTEBOOK_MW,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=(
            f"{MAE_TEST_NOTEBOOK_MW:,.2f} MW -- MAE de TEST del notebook "
            "(4.336 h, 2026 H1). No es una meta de producción."
        ),
    )
    ax_mae.set_ylabel("MAE diario (MW)")
    # Fuera de los ejes (pliego Fase6 §1.1): dentro de la caja de datos, en
    # cualquier esquina, la leyenda tapa parte de la serie o del pico del
    # máximo -- la caja se ancla por encima del panel, en el hueco entre el
    # título y los datos, donde no hay ninguna serie que cubrir.
    ax_mae.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8,
        frameon=False,
    )
    ax_mae.set_title(
        "Deriva del error diario -- v1 (solo horas publicadas)", pad=28
    )

    ax_sesgo.plot(x, serie["sesgo"], marker="o", color="tab:red")
    ax_sesgo.axhline(0, color="black", linewidth=1)
    ax_sesgo.set_ylabel("Sesgo diario (MW)\n+ infrapredicción / − sobrepredicción")
    ax_sesgo.set_xlabel("Fecha del horizonte (Madrid)")

    n_moda = serie["n"].mode()
    n_moda = n_moda.iloc[0] if not n_moda.empty else None
    for _, fila in serie.iterrows():
        if pd.isna(fila["n"]):
            continue
        if n_moda is not None and fila["n"] != n_moda:
            # Ancla explícita al punto (pliego Fase6 §1.2): un desplazamiento
            # en puntos de pantalla, sin más, se lee en unidades de datos muy
            # distintas según el rango del eje -- en este eje (cientos a
            # miles de MW) un offset "pequeño" en pantalla puede aterrizar
            # la etiqueta junto a la línea de referencia del notebook en vez
            # de junto al punto. La flecha corta quita la ambigüedad
            # independientemente de la escala.
            ax_mae.annotate(
                f"n={int(fila['n'])}",
                (pd.Timestamp(fila["fecha"]), fila["mae"]),
                textcoords="offset points",
                xytext=(18, -14),
                fontsize=7,
                ha="left",
                va="top",
                arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.6),
            )

    fig.autofmt_xdate()
    fig.text(0.01, 0.01, texto_v2, fontsize=8, color="dimgray")
    # Techo del rect recortado a 0.90 (antes 1): dejar hueco para el título
    # y, encima de él, la leyenda anclada fuera de los ejes (§1.1) -- si no,
    # tight_layout la comprime contra el título o la corta al guardar.
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--salida",
        type=Path,
        default=RUTA_PNG_DEFECTO,
        help=f"Ruta del PNG a escribir (default: {RUTA_PNG_DEFECTO}).",
    )
    parser.add_argument(
        "--metricas",
        type=Path,
        default=RUTA_METRICAS,
        help=(
            "Ruta de metricas.json a leer (default: data/metricas.json). "
            "Solo para verificación -- el pliego prohíbe depender de nada "
            "que no sea el fichero real en la corrida normal."
        ),
    )
    parser.add_argument(
        "--errores",
        type=Path,
        default=RUTA_ERRORES,
        help=(
            "Ruta de errores.csv a leer (default: data/errores.csv). Solo "
            "para verificación, mismo criterio que --metricas."
        ),
    )
    args = parser.parse_args()

    errores = _leer_csv(args.errores)
    metricas = _leer_metricas(args.metricas)

    serie = serie_diaria_v1(errores)
    if serie.empty:
        raise RuntimeError(
            "Ninguna hora publicada de v1 en errores.csv -- nada que dibujar."
        )

    print("Valores diarios calculados (v1, solo horas publicadas):")
    print(serie.to_string(index=False))

    serie_con_huecos = _serie_con_huecos_explicitos(serie)
    dias_marcados = dias_no_laborable_a_laborable(serie["fecha"].min(), serie["fecha"].max())
    cobertura = cobertura_v2(metricas)
    texto_v2 = _texto_cobertura_v2(cobertura)
    print(f"\n{texto_v2}")

    fig = construir_grafico(serie_con_huecos, dias_marcados, texto_v2)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.salida, dpi=150)
    plt.close(fig)

    print(f"\nPNG escrito en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
