"""Evaluación diaria de las predicciones emitidas por `pipeline/predecir.py`
(Fase 2, cierra el pliego de Fase 1.5).

Convención de signo del error -- verificada, no asumida (pliego §3.3): igual
que `SESGO_NIVEL` en `notebooks/modelo_demanda.ipynb`, celda 28
(`SESGO_NIVEL = (y_test_nivel - pred_test_nivel).mean()`, +755,15 MW
publicado, `modelos/baseline_numeros.json`): **`error = valor_real -
valor_predicho`**. Positivo significa infrapredicción (el modelo se queda
corto), igual que el sesgo del modelo publicado.

Qué filas se evalúan (§3.1): pendiente = fila de `predicciones.csv` sin
entrada en `errores.csv` para la misma clave (`horizonte`, `modelo`). Se
buscan huecos, no se asume D-1 -- si el job murió en rojo un día, el
siguiente se autocura. Una hora cuyo dato real todavía no está completo
(`n_lecturas < 12`) sigue pendiente: no es error, no se escribe nada para
ella.

Descarga del dato real (§3.2): `resample_horario_con_conteo` (el permisivo),
nunca `resample_horario` (el estricto) -- una descarga en vivo siempre tiene
la hora en curso incompleta, y el estricto fallaría en rojo cada día sin
excepción.

Escritura (§3.8), siempre en este orden: `errores.csv` (append) ->
recalcular métricas desde `errores.csv` completo -> `metricas.json` ->
`reports/estado_pipeline.md`. Sin try/except amplios, sin `|| true`, sin
`continue-on-error`: si e·sios no responde, el job muere en rojo y no
escribe nada parcial.

Uso:
    evaluar.py                        # normal (cron): evalúa pendientes.
    evaluar.py --ignorar-antiguas     # desbloquea pendientes > 15 días,
                                       # dejándolas permanentemente sin
                                       # evaluar (uso manual).
    evaluar.py --reevaluar D1 D2      # revisión REE de una ventana ya
                                       # evaluada (uso manual, nunca cron).
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

from src.datos import descargar_demanda_cruda, resample_horario_con_conteo
from src.paths import DIR_DATA, DIR_REPORTS
from src.quality import assert_rango_fisico

NOMBRE_MODELO = "v1"

RUTA_PREDICCIONES = DIR_DATA / "predicciones.csv"
RUTA_ERRORES = DIR_DATA / "errores.csv"
RUTA_METRICAS = DIR_DATA / "metricas.json"
RUTA_ESTADO = DIR_REPORTS / "estado_pipeline.md"

COLUMNAS_ERRORES = [
    "fecha_evaluacion",
    "horizonte",
    "modelo",
    "fecha_emision",
    "valor_predicho",
    "valor_real",
    "error",
    "h_adelanto_h",
]

# Dos semanas de pipeline muerto merecen una mirada, no una descarga
# silenciosa de seis meses (§3.1).
DIAS_VENTANA_MAX_EVALUACION = 15

# `mae: null` mientras la ventana no cubra al menos esto (§3.6, regla 3):
# lo que no se puede afirmar todavía no se escribe.
COBERTURA_MINIMA_DIAS = 7

VENTANAS_DIAS = {"7d": 7, "30d": 30, "90d": 90}

# Referencia del modelo publicado, notebooks/modelo_demanda.ipynb celda 28
# (`MAE_NIVEL_4336`, `SESGO_NIVEL`) y `modelos/baseline_numeros.json`
# (`MAE_PERSISTENCIA`). No se leen en caliente de esos ficheros porque son
# la verdad congelada del gate de re-entrenamiento (`scripts/
# gate_numeros.py`), no una dependencia en tiempo de ejecución del pipeline.
MAE_TEST_NOTEBOOK_MW = 1263.02
SESGO_TEST_NOTEBOOK_MW = 755.15
MAE_BASELINE_PERSISTENCIA_MW = 1843


def _leer_csv(ruta: Path) -> pd.DataFrame:
    """Todos los CSV del pipeline se leen con `float_precision='round_trip'`:
    sin eso el parser de floats de pandas pierde 1 ULP (hallazgo del 14/8,
    `3,637978807e-12`) y aparecen diferencias que el modelo no puede
    producir."""
    return pd.read_csv(ruta, float_precision="round_trip")


def filas_pendientes(df_predicciones: pd.DataFrame, df_errores: pd.DataFrame) -> pd.DataFrame:
    """Filas de `df_predicciones` sin entrada en `df_errores` para la misma
    clave (`horizonte`, `modelo`). No asume D-1 (§3.1): busca huecos, no una
    fecha fija."""
    if df_errores.empty:
        return df_predicciones.copy()
    clave_evaluada = set(zip(df_errores["horizonte"], df_errores["modelo"]))
    ya_evaluadas = df_predicciones.apply(
        lambda f: (f["horizonte"], f["modelo"]) in clave_evaluada, axis=1
    )
    return df_predicciones[~ya_evaluadas].copy()


def filtrar_pendientes_por_ventana(
    pendientes: pd.DataFrame, ahora_utc: pd.Timestamp, ignorar_antiguas: bool
) -> pd.DataFrame:
    """Aplica el tope de `DIAS_VENTANA_MAX_EVALUACION` (§3.1). Sin
    `--ignorar-antiguas`, para en rojo si hay pendientes más antiguas que el
    tope. Con el flag, las descarta (quedan permanentemente sin evaluar) y
    lo dice en el log -- nunca en silencio."""
    if pendientes.empty:
        return pendientes

    limite = ahora_utc - pd.Timedelta(days=DIAS_VENTANA_MAX_EVALUACION)
    horizontes = pd.to_datetime(pendientes["horizonte"], utc=True)
    antiguas = horizontes < limite

    if not antiguas.any():
        return pendientes

    n_antiguas = int(antiguas.sum())
    desde = horizontes[antiguas].min()

    if not ignorar_antiguas:
        raise RuntimeError(
            f"{n_antiguas} pendiente(s) más antigua(s) que "
            f"{DIAS_VENTANA_MAX_EVALUACION} días (desde {desde}). No se "
            "avanza -- revisa manualmente o relanza con --ignorar-antiguas."
        )

    print(
        f"--ignorar-antiguas: {n_antiguas} pendiente(s) anteriores a "
        f"{DIAS_VENTANA_MAX_EVALUACION} días (desde {desde}) quedan "
        "permanentemente sin evaluar."
    )
    return pendientes[~antiguas]


def construir_filas_error(
    pendientes: pd.DataFrame, reales: pd.Series, fecha_evaluacion: str
) -> pd.DataFrame:
    """De las `pendientes`, solo las que ya tienen dato real completo
    (`n_lecturas == 12`, ver `reales`). Una hora sin dato real completo
    sigue pendiente -- no es error, no se escribe nada para ella y no se
    lanza ninguna excepción (§3.1, caso 2)."""
    horizonte_ts = pd.to_datetime(pendientes["horizonte"], utc=True)
    fecha_emision_ts = pd.to_datetime(pendientes["fecha_emision"], utc=True)

    valor_real = reales.reindex(horizonte_ts.to_numpy())
    valor_real.index = pendientes.index  # alinear por posición, no por timestamp
    disponible = valor_real.notna()

    if not disponible.any():
        return pd.DataFrame(columns=COLUMNAS_ERRORES)

    filas = pendientes.loc[disponible]
    valor_predicho = filas["valor_predicho"].astype(float)
    valor_real_disp = valor_real.loc[disponible].astype(float)

    # Convención verificada en el docstring del módulo: real - predicho.
    error = valor_real_disp - valor_predicho

    h_adelanto_h = (
        (horizonte_ts.loc[disponible] - fecha_emision_ts.loc[disponible]).dt.total_seconds()
        / 3600.0
    ).round(2)

    return pd.DataFrame(
        {
            "fecha_evaluacion": fecha_evaluacion,
            "horizonte": filas["horizonte"].to_numpy(),
            "modelo": filas["modelo"].to_numpy(),
            "fecha_emision": filas["fecha_emision"].to_numpy(),
            "valor_predicho": valor_predicho.to_numpy(),
            "valor_real": valor_real_disp.to_numpy(),
            "error": error.to_numpy(),
            "h_adelanto_h": h_adelanto_h.to_numpy(),
        }
    )[COLUMNAS_ERRORES]


def guardar_errores(filas_nuevas: pd.DataFrame) -> None:
    """Append-only. Nunca edición (§3.4)."""
    if filas_nuevas.empty:
        return
    if RUTA_ERRORES.exists():
        filas_nuevas.to_csv(RUTA_ERRORES, mode="a", header=False, index=False)
    else:
        RUTA_ERRORES.parent.mkdir(parents=True, exist_ok=True)
        filas_nuevas.to_csv(RUTA_ERRORES, mode="w", header=True, index=False)


def _metrica_ventana(df_ventana: pd.DataFrame) -> dict:
    """`mae`/`sesgo_medio` en `null` mientras `cobertura_dias <
    COBERTURA_MINIMA_DIAS` (§3.6, regla 3): lo que no se puede afirmar
    todavía no se escribe, no un número con asterisco. `cobertura_dias` es
    el span real (en días) entre el primer y el último horizonte evaluado
    de la ventana -- no el tamaño nominal de la ventana (7/30/90): un
    `mae_90d` calculado sobre 3 días reales de historia es una mentira con
    formato de métrica."""
    n_horas = len(df_ventana)
    if n_horas == 0:
        cobertura_dias = 0.0
    else:
        horizonte_ts = pd.to_datetime(df_ventana["horizonte"], utc=True)
        span = horizonte_ts.max() - horizonte_ts.min()
        cobertura_dias = round(span.total_seconds() / 86400.0, 1)

    suficiente = cobertura_dias >= COBERTURA_MINIMA_DIAS
    return {
        "mae": round(float(df_ventana["error"].abs().mean()), 2) if suficiente else None,
        "sesgo_medio": round(float(df_ventana["error"].mean()), 2) if suficiente else None,
        "n_horas": n_horas,
        "cobertura_dias": cobertura_dias,
    }


def calcular_metricas(df_errores: pd.DataFrame, modelo: str, ahora_utc: pd.Timestamp) -> dict:
    """Recalcula `metricas.json` entero leyendo `df_errores` completo, nunca
    incrementando el valor anterior (§3.6): una sola fuente de verdad.

    Publicado (`h_adelanto_h > 0`) y diagnóstico (`h_adelanto_h <= 0`) nunca
    se agregan en un mismo número (§3.6, regla 1) -- se guarda el signo de
    `h_adelanto_h` en cada fila de `errores.csv`, no el resultado del
    filtro, así que el corte se recalcula aquí cada vez, no se congela."""
    df_modelo = df_errores[df_errores["modelo"] == modelo] if not df_errores.empty else df_errores

    def _bloque(mascara: pd.Series) -> dict:
        df_criterio = df_modelo[mascara] if not df_modelo.empty else df_modelo
        ventanas = {}
        for nombre, dias in VENTANAS_DIAS.items():
            corte = ahora_utc - pd.Timedelta(days=dias)
            if df_criterio.empty:
                df_ventana = df_criterio
            else:
                horizonte_ts = pd.to_datetime(df_criterio["horizonte"], utc=True)
                df_ventana = df_criterio[horizonte_ts >= corte]
            ventanas[nombre] = _metrica_ventana(df_ventana)
        return ventanas

    if df_modelo.empty:
        h_adelanto = pd.Series([], dtype=float)
    else:
        h_adelanto = df_modelo["h_adelanto_h"].astype(float)

    publicado = _bloque(h_adelanto > 0)
    diagnostico = _bloque(h_adelanto <= 0)

    cobertura_maxima = max(v["cobertura_dias"] for v in publicado.values())
    muestra_insuficiente = cobertura_maxima < COBERTURA_MINIMA_DIAS

    return {
        "actualizado_utc": ahora_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modelo": modelo,
        "muestra_insuficiente": muestra_insuficiente,
        "publicado": {"criterio": "h_adelanto_h > 0", "ventanas": publicado},
        "diagnostico": {"criterio": "h_adelanto_h <= 0", "ventanas": diagnostico},
        "referencia": {
            "mae_test_notebook": MAE_TEST_NOTEBOOK_MW,
            "sesgo_test_notebook": SESGO_TEST_NOTEBOOK_MW,
            "mae_baseline_persistencia": MAE_BASELINE_PERSISTENCIA_MW,
        },
    }


def escribir_metricas(metricas: dict) -> None:
    """Fichero derivado: se reescribe entero en cada corrida (§3.6)."""
    RUTA_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_METRICAS.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _formatear_mae(valor) -> str:
    return f"{valor:.2f}" if valor is not None else "—"


def _formatear_sesgo(valor) -> str:
    return f"{valor:+.2f}" if valor is not None else "—"


def construir_estado_pipeline_md(metricas: dict, n_dias_serie: int) -> str:
    """Fichero generado, se reescribe entero (§3.7). Primera línea marca el
    fichero como no editable a mano. Métrica publicada solamente -- nunca
    agregada con la diagnóstica (§3.6, regla 1)."""
    filas_tabla = []
    for ventana, valores in metricas["publicado"]["ventanas"].items():
        filas_tabla.append(
            f"| {ventana} | {_formatear_mae(valores['mae'])} | "
            f"{_formatear_sesgo(valores['sesgo_medio'])} | {valores['n_horas']} | "
            f"{valores['cobertura_dias']} | "
            f"{metricas['referencia']['mae_test_notebook']:.2f} |"
        )
    tabla = "\n".join(filas_tabla)

    lineas = [
        "<!-- GENERADO POR pipeline/evaluar.py -- NO EDITAR A MANO -->",
        "",
        "# Estado del pipeline REE",
        "",
        f"Última corrida: {metricas['actualizado_utc']}",
        f"Modelo: {metricas['modelo']}",
        f"Días de serie acumulados en `data/errores.csv`: {n_dias_serie}",
        "",
        "## Métrica publicada (`h_adelanto_h > 0`)",
        "",
        "| Ventana | MAE (MW) | Sesgo medio (MW) | n horas | Cobertura (días) | "
        "MAE test notebook (MW) |",
        "|---|---|---|---|---|---|",
        tabla,
        "",
    ]

    if metricas["muestra_insuficiente"]:
        lineas += [
            "> **Muestra insuficiente todavía.** Ninguna ventana alcanza los "
            f"{COBERTURA_MINIMA_DIAS} días de cobertura mínima -- el MAE se "
            "muestra como `—` (`null` en `data/metricas.json`) porque lo que "
            "no se puede afirmar todavía no se escribe.",
            "",
        ]

    return "\n".join(lineas) + "\n"


def _recalcular_y_escribir(ahora_utc: pd.Timestamp) -> None:
    """Último tramo de §3.8: recalcular desde `errores.csv` completo ->
    `metricas.json` -> `estado_pipeline.md`. Se llama tanto desde el flujo
    normal como desde `--reevaluar`."""
    errores_completo = _leer_csv(RUTA_ERRORES) if RUTA_ERRORES.exists() else pd.DataFrame(
        columns=COLUMNAS_ERRORES
    )
    metricas = calcular_metricas(errores_completo, NOMBRE_MODELO, ahora_utc)
    escribir_metricas(metricas)

    if errores_completo.empty:
        n_dias_serie = 0
    else:
        n_dias_serie = int(
            pd.to_datetime(errores_completo["horizonte"], utc=True).dt.date.nunique()
        )

    RUTA_ESTADO.parent.mkdir(parents=True, exist_ok=True)
    RUTA_ESTADO.write_text(construir_estado_pipeline_md(metricas, n_dias_serie), encoding="utf-8")

    print(f"metricas.json y {RUTA_ESTADO.name} actualizados.")


def _reevaluar(desde: str, hasta: str, ahora_utc: pd.Timestamp) -> int:
    """Modo manual (§3.5), nunca en el cron: re-descarga la ventana
    [`desde`, `hasta`] y escribe fila NUEVA (con `fecha_evaluacion` nueva)
    solo si `|valor_real_nuevo - valor_real_guardado| > 1 MW` -- umbral para
    no confundir una revisión de telemetría de REE con ruido de parser. La
    revisión queda visible (la misma hora aparece dos veces, con dos marcas
    de tiempo) en vez de silenciosa."""
    if not RUTA_ERRORES.exists():
        print("No existe data/errores.csv todavía -- nada que reevaluar.")
        return 0

    errores = _leer_csv(RUTA_ERRORES)
    inicio = pd.Timestamp(desde, tz="UTC")
    fin = pd.Timestamp(hasta, tz="UTC") + pd.Timedelta(days=1)  # [desde, hasta] inclusive

    horizonte_ts = pd.to_datetime(errores["horizonte"], utc=True)
    en_ventana = (horizonte_ts >= inicio) & (horizonte_ts < fin)
    objetivo = errores.loc[en_ventana]

    if objetivo.empty:
        print(f"Ninguna fila de errores.csv cae en [{desde}, {hasta}] -- nada que reevaluar.")
        return 0

    df_crudo = descargar_demanda_cruda(inicio - pd.Timedelta(hours=1), fin)
    df_horario = resample_horario_con_conteo(df_crudo)
    reales = df_horario.loc[df_horario["n_lecturas"] == 12].set_index("datetime_utc")[
        "demanda_real"
    ]
    assert_rango_fisico(reales, "evaluar.py: valor_real recién descargado (--reevaluar)")

    fecha_evaluacion = ahora_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    filas_revision = []
    for _, fila in objetivo.iterrows():
        ts = pd.Timestamp(fila["horizonte"])
        if ts not in reales.index:
            continue  # sigue sin dato completo -- no se puede revisar todavía
        valor_real_nuevo = float(reales.loc[ts])
        if abs(valor_real_nuevo - float(fila["valor_real"])) <= 1.0:
            continue  # dentro del umbral de 1 MW -- ruido de parser, no revisión real
        filas_revision.append(
            {
                "fecha_evaluacion": fecha_evaluacion,
                "horizonte": fila["horizonte"],
                "modelo": fila["modelo"],
                "fecha_emision": fila["fecha_emision"],
                "valor_predicho": float(fila["valor_predicho"]),
                "valor_real": valor_real_nuevo,
                "error": valor_real_nuevo - float(fila["valor_predicho"]),
                "h_adelanto_h": float(fila["h_adelanto_h"]),
            }
        )

    if not filas_revision:
        print("Ninguna revisión supera el umbral de 1 MW -- nada que escribir.")
        return 0

    df_revision = pd.DataFrame(filas_revision)[COLUMNAS_ERRORES]
    guardar_errores(df_revision)
    print(f"{len(df_revision)} revisión(es) de REE escrita(s) como fila(s) nueva(s) en errores.csv.")

    _recalcular_y_escribir(ahora_utc)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ignorar-antiguas",
        action="store_true",
        help=(
            "Desbloquea pendientes más antiguas que "
            f"{DIAS_VENTANA_MAX_EVALUACION} días, dejándolas "
            "permanentemente sin evaluar. Uso manual -- NUNCA en el cron."
        ),
    )
    parser.add_argument(
        "--reevaluar",
        nargs=2,
        metavar=("DESDE", "HASTA"),
        help=(
            "Modo manual (YYYY-MM-DD YYYY-MM-DD): re-descarga esa ventana y "
            "escribe fila nueva solo si la revisión supera 1 MW. NUNCA en "
            "el cron."
        ),
    )
    args = parser.parse_args()

    ahora_utc = pd.Timestamp.now(tz="UTC")

    if args.reevaluar:
        return _reevaluar(args.reevaluar[0], args.reevaluar[1], ahora_utc)

    if not RUTA_PREDICCIONES.exists():
        print("No existe data/predicciones.csv todavía -- nada que evaluar.")
        return 0

    predicciones = _leer_csv(RUTA_PREDICCIONES)
    errores = _leer_csv(RUTA_ERRORES) if RUTA_ERRORES.exists() else pd.DataFrame(
        columns=COLUMNAS_ERRORES
    )

    pendientes = filas_pendientes(predicciones, errores)
    pendientes = filtrar_pendientes_por_ventana(pendientes, ahora_utc, args.ignorar_antiguas)

    if pendientes.empty:
        print("Sin pendientes -- nada que evaluar.")
    else:
        horizonte_min = pd.to_datetime(pendientes["horizonte"], utc=True).min()
        inicio_descarga = horizonte_min - pd.Timedelta(hours=1)

        df_crudo = descargar_demanda_cruda(inicio_descarga, ahora_utc)
        df_horario = resample_horario_con_conteo(df_crudo)
        reales = df_horario.loc[df_horario["n_lecturas"] == 12].set_index("datetime_utc")[
            "demanda_real"
        ]

        # Recién descargado, antes de calcular ningún error (§1, punto 1):
        # una hora corrupta escrita en errores.csv contamina el histórico de
        # forma permanente (append-only, no se puede borrar).
        assert_rango_fisico(reales, "evaluar.py: valor_real recién descargado (main)")

        fecha_evaluacion = ahora_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        filas_nuevas = construir_filas_error(pendientes, reales, fecha_evaluacion)
        guardar_errores(filas_nuevas)
        print(f"{len(filas_nuevas)} fila(s) nueva(s) en errores.csv.")

    _recalcular_y_escribir(ahora_utc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
