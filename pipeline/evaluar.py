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

Emisión paralela (Fase 5bis, PR B): a diferencia de `predecir.py`, este
script corre UNA sola vez por corrida, no una por modelo -- evalúa las
pendientes de todos los modelos presentes en `predicciones.csv` en la misma
pasada (`filas_pendientes`/`construir_filas_error` ya trabajan por la clave
`(horizonte, modelo)`, sin cambios en este PR). Solo el tramo de métricas
(`_recalcular_y_escribir`) recorre `MODELOS_ACTIVOS` explícitamente, un
bloque por modelo en `metricas.json` y en `reports/estado_pipeline.md` --
nunca agregados entre sí (§3.5 del pliego del PR B).

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

# Emisión paralela (Fase 5bis, PR B): evaluar.py corre una sola vez por
# corrida (a diferencia de predecir.py) y recorre los modelos presentes en
# este orden -- v1 primero, mismo orden que los pasos del pliego §3.1.
NOMBRE_MODELO_V2 = "v2"
MODELOS_ACTIVOS = [NOMBRE_MODELO, NOMBRE_MODELO_V2]

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
# lo que no se puede afirmar todavía no se escribe. Se compara contra
# `dias_cubiertos` (fechas de calendario distintas con al menos una hora en
# la ventana), no contra un span en días -- ver `_metrica_ventana`.
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

# v2 (Fase 5bis, PR A) NO tiene una referencia de notebook propia del
# modelo serializado -- y no es solo que no sea comparable con la de v1
# (test), es que no es DE ESTE modelo (revisión de Laura, PR B, 21/8):
# `modelo_v2.pkl` se entrena con `train_clean_6f` completo (2023-2025,
# 26.136 filas, ver `modelos/modelo_v2.json`), así que su propia validación
# (2025) forma parte de su entrenamiento y no tiene una medida en validación
# posible. El `mae_val`/`sesgo_val` que sí existe en `modelos/modelo_v2.json`
# es de un árbol de COMPARACIÓN distinto (entrenado solo con 2023-2024, para
# medir el efecto de `demanda_lag_168` con val como conjunto de decisión) --
# ese número tiene su sitio propio, con su nota, en `modelos/modelo_v2.json`
# (PR A); no se repite aquí. Ver la nota de `_referencia_de_modelo` abajo.


def _referencia_de_modelo(modelo: str) -> dict:
    """Bloque `referencia` de `calcular_metricas`, por modelo -- antes era
    fijo (siempre las constantes de v1). v1 conserva su `mae_test_notebook`
    de siempre (modelo serializado y número medido son el mismo objeto). v2
    no lleva ningún MAE/sesgo de notebook aquí -- ver comentario de arriba:
    no es que no sea comparable con la de v1, es que la única cifra
    disponible (`modelos/modelo_v2.json`) no es del modelo serializado. Una
    columna en una tabla se lee sin leer la nota de debajo; el hueco (sin
    columna) es la forma honesta de no invitar a esa comparación."""
    if modelo == NOMBRE_MODELO:
        return {
            "mae_test_notebook": MAE_TEST_NOTEBOOK_MW,
            "sesgo_test_notebook": SESGO_TEST_NOTEBOOK_MW,
            "mae_baseline_persistencia": MAE_BASELINE_PERSISTENCIA_MW,
        }
    if modelo == NOMBRE_MODELO_V2:
        return {
            "mae_baseline_persistencia": MAE_BASELINE_PERSISTENCIA_MW,
            "nota": (
                "v2 no tiene una referencia de notebook para el modelo "
                "SERIALIZADO (entrena con validación incluida, 2023-2025 "
                "completo): no hay una medida en validación posible para "
                "ese modelo. El mae_val/sesgo_val de modelos/modelo_v2.json "
                "es de un árbol de comparación distinto, entrenado solo con "
                "2023-2024 -- se queda documentado ahí, con su propia nota, "
                "y no se repite aquí para no sugerir que es del serializado."
            ),
        }
    raise ValueError(f"Sin referencia declarada para el modelo {modelo!r}.")


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
    """`mae`/`sesgo_medio` en `null` mientras `dias_cubiertos <
    COBERTURA_MINIMA_DIAS` (§3.6, regla 3): lo que no se puede afirmar
    todavía no se escribe, no un número con asterisco. `cobertura_dias` es
    el span real (en días) entre el primer y el último horizonte evaluado
    de la ventana -- no el tamaño nominal de la ventana (7/30/90): un
    `mae_90d` calculado sobre 3 días reales de historia es una mentira con
    formato de métrica. Se conserva en el JSON con el mismo cálculo, pero
    ya no gobierna la guarda.

    `dias_cubiertos` es la magnitud que sí gobierna la guarda: fechas de
    calendario distintas con al menos una hora en la ventana, no un span.
    Un span cuenta las 06:00Z-21:00Z de siete días consecutivos como 6,6
    días (6 días y 15 horas entre el primer y el último horizonte, porque
    cada día objetivo solo publica horas desde las 06:00Z) -- la guarda por
    span nunca se satisface con la ventana llena. Fechas distintas cuenta
    esos mismos siete días como 7."""
    n_horas = len(df_ventana)
    if n_horas == 0:
        cobertura_dias = 0.0
        dias_cubiertos = 0
    else:
        horizonte_ts = pd.to_datetime(df_ventana["horizonte"], utc=True)
        span = horizonte_ts.max() - horizonte_ts.min()
        cobertura_dias = round(span.total_seconds() / 86400.0, 1)
        dias_cubiertos = horizonte_ts.dt.date.nunique()

    # Fechas distintas, no span: un span cuenta 7 días consecutivos de horas
    # 06:00Z-21:00Z como 6,6 días (6 días y 15 horas de primer a último
    # horizonte) y la guarda nunca se satisface con la ventana llena.
    suficiente = dias_cubiertos >= COBERTURA_MINIMA_DIAS
    return {
        "mae": round(float(df_ventana["error"].abs().mean()), 2) if suficiente else None,
        "sesgo_medio": round(float(df_ventana["error"].mean()), 2) if suficiente else None,
        "n_horas": n_horas,
        "cobertura_dias": cobertura_dias,
        "dias_cubiertos": dias_cubiertos,
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

    # Misma magnitud que gobierna la guarda por ventana (dias_cubiertos), no
    # el span: si comparara cobertura_dias aquí, el aviso podría desaparecer
    # sin que ninguna ventana tenga aún mae escrito, o al revés.
    dias_cubiertos_maximo = max(v["dias_cubiertos"] for v in publicado.values())
    muestra_insuficiente = dias_cubiertos_maximo < COBERTURA_MINIMA_DIAS

    return {
        "actualizado_utc": ahora_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modelo": modelo,
        "muestra_insuficiente": muestra_insuficiente,
        "publicado": {"criterio": "h_adelanto_h > 0", "ventanas": publicado},
        "diagnostico": {"criterio": "h_adelanto_h <= 0", "ventanas": diagnostico},
        "referencia": _referencia_de_modelo(modelo),
    }


def escribir_metricas(metricas_por_modelo: dict[str, dict]) -> None:
    """Fichero derivado: se reescribe entero en cada corrida (§3.6).

    Emisión paralela (Fase 5bis PR B): antes `metricas.json` era un único
    bloque (el de v1, con `modelo` dentro); ahora es un objeto con una clave
    por modelo (`{"v1": {...}, "v2": {...}}`), cada valor el mismo bloque de
    siempre -- dos bloques, no una fusión de los dos en un número (§3.5)."""
    RUTA_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_METRICAS.write_text(
        json.dumps(metricas_por_modelo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _formatear_mae(valor) -> str:
    return f"{valor:.2f}" if valor is not None else "—"


def _formatear_sesgo(valor) -> str:
    return f"{valor:+.2f}" if valor is not None else "—"


def _bloque_md_modelo(metricas: dict, n_dias_serie: int) -> list[str]:
    """Bloque de un solo modelo, mismo contenido que producía
    `construir_estado_pipeline_md` antes del PR B (Fase5bis, emisión
    paralela) cuando solo existía v1 -- ahora se llama una vez por modelo
    presente en `MODELOS_ACTIVOS` y `construir_estado_pipeline_md` concatena
    los bloques (§3.6 del pliego del PR B: "los dos modelos, pegados
    enteros"). Métrica publicada solamente -- nunca agregada con la
    diagnóstica (§3.6, regla 1 del pliego de Fase 2).

    `n_dias_serie` es ya SOLO de este modelo (columna `modelo` de
    `errores.csv` filtrada aguas arriba, ver `_recalcular_y_escribir`): antes
    de que existiera v2 esto no hacía falta decirlo porque solo había un
    modelo en el fichero; con dos, contar fechas sin filtrar por modelo
    mezclaría los históricos de v1 y v2 en un solo número -- justo la
    agregación entre modelos que el pliego del PR B prohíbe (§3.5).

    Referencia de la tabla (columna final): SOLO se añade si
    `metricas["referencia"]` trae un número de notebook propio del modelo
    serializado (`mae_test_notebook`, hoy solo v1). No se añade una columna
    "aproximada" con nota al pie explicando que no vale: una columna en una
    tabla se lee sin leer la nota de debajo, y esa cifra terminaría
    comparada con la de v1 tarde o temprano (revisión de Laura, PR B,
    21/8) -- v2 no tiene esa referencia hoy (ver `_referencia_de_modelo`),
    así que su tabla sale sin columna final, no con una engañosa."""
    referencia = metricas["referencia"]
    tiene_referencia_propia = "mae_test_notebook" in referencia

    filas_tabla = []
    for ventana, valores in metricas["publicado"]["ventanas"].items():
        fila = (
            f"| {ventana} | {_formatear_mae(valores['mae'])} | "
            f"{_formatear_sesgo(valores['sesgo_medio'])} | {valores['n_horas']} | "
            f"{valores['dias_cubiertos']} | {valores['cobertura_dias']} |"
        )
        if tiene_referencia_propia:
            fila += f" {referencia['mae_test_notebook']:.2f} |"
        filas_tabla.append(fila)
    tabla = "\n".join(filas_tabla)

    cabecera = (
        "| Ventana | MAE (MW) | Sesgo medio (MW) | n horas | "
        "Fechas cubiertas (gobierna) | Span (días) |"
    )
    separador = "|---|---|---|---|---|---|"
    if tiene_referencia_propia:
        cabecera += " MAE test notebook (MW) |"
        separador += "---|"

    lineas = [
        f"## Modelo: {metricas['modelo']}",
        "",
        f"Última corrida: {metricas['actualizado_utc']}",
        f"Fechas presentes en `data/errores.csv` para este modelo "
        f"(publicadas + diagnóstico): {n_dias_serie}",
        "",
        "### Métrica publicada (`h_adelanto_h > 0`)",
        "",
        "`Fechas cubiertas` cuenta fechas de calendario distintas con al menos "
        "una hora publicada en la ventana: es la magnitud que decide si el MAE "
        "se escribe o se muestra `—`. `Span (días)` es la distancia en días "
        "entre el primer y el último horizonte publicado de la ventana; se "
        "conserva como dato informativo, pero no gobierna nada. Es menor que "
        "`Fechas cubiertas` cuando las fechas son contiguas (el span mide días "
        "transcurridos entre extremos, no fechas contadas), y mayor cuando hay "
        "huecos entre ellas (fechas dispersas en el tiempo estiran el span sin "
        "sumar fechas cubiertas).",
        "",
        cabecera,
        separador,
        tabla,
        "",
    ]

    if metricas["muestra_insuficiente"]:
        lineas += [
            "> **Muestra insuficiente todavía.** Ninguna ventana alcanza las "
            f"{COBERTURA_MINIMA_DIAS} fechas de calendario distintas con al "
            "menos una hora publicada que exige la cobertura mínima -- el MAE "
            "se muestra como `—` (`null` en `data/metricas.json`) porque lo "
            "que no se puede afirmar todavía no se escribe.",
            "",
        ]

    if tiene_referencia_propia:
        lineas += [
            "### Por qué el MAE de producción no coincide con el del notebook",
            "",
            "El MAE de producción de la tabla de arriba no es directamente "
            f"comparable al {referencia['mae_test_notebook']:.2f} MW medido en "
            "el conjunto de test del notebook "
            "(`notebooks/modelo_demanda.ipynb`, celda 28). Dos limitaciones "
            "conocidas y diagnosticadas del modelo -- no fallos del pipeline "
            "-- explican buena parte de la diferencia:",
            "",
            "- **Arranque de la semana.** El modelo condiciona el nivel de la "
            "predicción en `tipo_efectivo(D)` y en `demanda_lag_24`, pero nunca "
            "en `tipo_efectivo(D-1)`: no sabe de qué tipo de día viene su punto "
            "de partida. El régimen de transición `no laborable → laborable` "
            "(los lunes) es donde más se nota, y toda ventana de 7 días contiene "
            "exactamente un lunes -- esta limitación es estado estacionario de "
            "cualquier ventana de producción, no un transitorio que vaya a "
            "desaparecer con más datos.",
            "- **Techo de extrapolación.** El modelo (`DecisionTreeRegressor`) no "
            "extrapola por encima del rango de entrenamiento: su predicción "
            "máxima estaba clavada en 38.861,1 MW en agosto de 2026, y la demanda "
            "real superó los 40.000 MW dos veces esa misma semana del 14/8/2026 -- "
            "en esas horas la desviación entre predicción y demanda real es "
            "grande por construcción del modelo, no por un fallo del pipeline.",
            "",
        ]
    else:
        lineas += [
            "### Referencia de notebook",
            "",
            "Este modelo no tiene todavía una referencia de notebook para el "
            "propio artefacto serializado -- sin columna en la tabla de "
            "arriba a propósito, para no invitar a compararla con la de v1. "
            + referencia.get("nota", ""),
            "",
            "No hay tampoco una sección de \"limitaciones diagnosticadas\" "
            "propia de este modelo todavía: las de v1 (arranque de la semana, "
            "techo de extrapolación) son observaciones medidas sobre su "
            "árbol, no se asumen iguales aquí sin medirlas -- queda para "
            "cuando la ventana de seis semanas (pliego Fase5bis, 0.1) dé "
            "señal suficiente.",
            "",
        ]

    return lineas


def construir_estado_pipeline_md(
    metricas_por_modelo: dict[str, dict], n_dias_por_modelo: dict[str, int]
) -> str:
    """Fichero generado, se reescribe entero (§3.7 del pliego de Fase 2).
    Primera línea marca el fichero como no editable a mano.

    Emisión paralela (Fase 5bis PR B, §3.6): un bloque `## Modelo: vX` por
    modelo presente en `metricas_por_modelo`, en el orden de
    `MODELOS_ACTIVOS` (v1 primero) -- "los dos modelos, pegados enteros", no
    resumidos ni comparados entre sí en este fichero (eso es la Parte 4 del
    pliego, el panel de Streamlit, todavía sin construir)."""
    lineas = [
        "<!-- GENERADO POR pipeline/evaluar.py -- NO EDITAR A MANO -->",
        "",
        "# Estado del pipeline REE",
        "",
    ]

    for modelo in MODELOS_ACTIVOS:
        if modelo not in metricas_por_modelo:
            continue
        lineas += _bloque_md_modelo(metricas_por_modelo[modelo], n_dias_por_modelo[modelo])
        lineas += ["---", ""]

    if lineas[-2:] == ["---", ""]:
        lineas = lineas[:-2]

    return "\n".join(lineas) + "\n"


def _recalcular_y_escribir(ahora_utc: pd.Timestamp) -> None:
    """Último tramo de §3.8: recalcular desde `errores.csv` completo ->
    `metricas.json` -> `estado_pipeline.md`. Se llama tanto desde el flujo
    normal como desde `--reevaluar`.

    Emisión paralela (Fase 5bis PR B, §3.2 punto 4): recorre
    `MODELOS_ACTIVOS` -- `calcular_metricas` ya filtra `df_errores` por
    modelo internamente (línea `df_modelo = df_errores[df_errores["modelo"]
    == modelo]`), así que se le pasa `errores_completo` sin prefiltrar y
    cada modelo recibe su propio bloque, incluido v2 el primer día que corre
    (con `errores_completo` sin ninguna fila `modelo == "v2"` todavía --
    `calcular_metricas` da `mae: null` en todas las ventanas, no un bloque
    ausente: v2 tiene que aparecer con `—` desde el día uno, pliego §3.5
    último punto)."""
    errores_completo = _leer_csv(RUTA_ERRORES) if RUTA_ERRORES.exists() else pd.DataFrame(
        columns=COLUMNAS_ERRORES
    )

    metricas_por_modelo = {}
    n_dias_por_modelo = {}
    for modelo in MODELOS_ACTIVOS:
        metricas_por_modelo[modelo] = calcular_metricas(errores_completo, modelo, ahora_utc)

        # n_dias_serie por modelo, no global (§3.5): mezclar las fechas de
        # v1 y v2 en una sola cuenta agregaría los dos históricos en un
        # número, justo lo que el pliego prohíbe.
        errores_modelo = (
            errores_completo[errores_completo["modelo"] == modelo]
            if not errores_completo.empty
            else errores_completo
        )
        if errores_modelo.empty:
            n_dias_por_modelo[modelo] = 0
        else:
            n_dias_por_modelo[modelo] = int(
                pd.to_datetime(errores_modelo["horizonte"], utc=True).dt.date.nunique()
            )

    escribir_metricas(metricas_por_modelo)

    RUTA_ESTADO.parent.mkdir(parents=True, exist_ok=True)
    RUTA_ESTADO.write_text(
        construir_estado_pipeline_md(metricas_por_modelo, n_dias_por_modelo), encoding="utf-8"
    )

    print(f"metricas.json y {RUTA_ESTADO.name} actualizados ({', '.join(MODELOS_ACTIVOS)}).")


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
