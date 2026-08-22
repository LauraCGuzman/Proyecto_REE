"""Descarga la demanda horaria posterior al 30/6/2026 (pliego
`PLIEGO_descarga_2026H2.md`, prerrequisito de la Fase 5).

Script de SOLO ESCRITURA en un fichero nuevo. NUNCA toca
`data/processed/demanda_horaria.parquet` (§1 del pliego: ese fichero
reproduce los números publicados -- 30.648 filas, los asserts de
`notebooks/modelo_demanda.ipynb` y `notebooks/modelo_v2_lag168.ipynb`, y
`scripts/gate_numeros.py` -- y se queda congelado). Escribe
`data/processed/demanda_horaria_extendida.parquet`: el histórico completo
(2023-01-01 -> último día natural completo disponible), para que la Fase 5
pueda entrenar un challenger con datos más recientes que los de v1. Este
script no entrena ni despliega nada.

Deliberadamente NO usa `src/esios_client.py::descargar_rango` (§1 punto 4,
verificado y documentado en el log del PR): esa función exige que
`fecha_inicio`/`fecha_fin` sean primeros de mes (bloquea un corte a mitad de
mes) y, peor, llama a `src/quality.py::validar_rejilla`, que exige la
rejilla de 5 minutos del mes ENTERO sin un solo hueco -- si e·sios tuviera
una sola lectura de 5' ausente en julio o agosto, ese mes completo saldría
con cero filas en vez del filtrado hora a hora que pide este pliego. En su
lugar, reutiliza las piezas de más bajo nivel que ya usa
`pipeline/predecir.py` en producción para exactamente este caso ("decidir
por sí mismo qué horas son fiables"):
  - `src.datos.descargar_demanda_cruda` (sin restricción de mes, sin gate
    de rejilla -- solo `raise_for_status`).
  - `src.datos.resample_horario_con_conteo` (agrega 5' -> horario con
    `n_lecturas`, sin assert de completitud -- la MISMA función que pide
    reutilizar el pliego §2, no reimplementada).
  - `pipeline.predecir.encontrar_ultimo_dia_completo` (el "último día
    natural completo" del pliego es literalmente esta función: recorre los
    días naturales de Madrid, más reciente primero, y devuelve el primero
    con TODAS sus horas UTC presentes y `n_lecturas == 12` -- ya maneja DST
    con `DateOffset`, no se reimplementa).

Tres columnas -- `datetime_utc`, `demanda_real`, `demanda_prevista` --
revisado tras la primera versión de este script (que solo traía
`demanda_real`, indicador 1293). `demanda_prevista` (indicador 544) no es
input de ningún entrenamiento -- `src/features.py` no la toca -- pero es la
previsión oficial de REE, la misma que sostiene el `mae_baseline` de REE
(`notebooks/descarga_historico.ipynb`, mismo patrón de descarga que
`demanda_real`); sin ella no hay con qué comparar al modelo contra REE en
el rango nuevo (decisión de Laura, log de este PR).

`demanda_real` sigue gobernando qué horas entran en el fichero (pliego §2:
`n_lecturas == 12`, corte en el último día natural completo). Para
`demanda_prevista` se aplica el mismo filtro de calidad, mismo criterio,
pero de forma independiente -- si una hora de `demanda_prevista` viniera
incompleta sin que le pasara lo mismo a `demanda_real` (algo que no ha
ocurrido en la ventana descargada, ver log), esa hora se queda en el
fichero con `demanda_real` real y `demanda_prevista` en NaN, en vez de
descartar la fila entera por un problema de una sola columna -- mismo
comportamiento que ya tiene el fichero original (`demanda_prevista` no
gobierna nunca qué filas hay).

`resample_horario_con_conteo` (`src/datos.py`) está fijado a la columna
`demanda_real` -- para reutilizarla tal cual con `demanda_prevista` (pliego
§2: "no reimplementar la agregación") se renombra la columna antes de
llamarla y se deshace el renombrado después (`_resamplear_reutilizando`,
más abajo). La alternativa (una segunda función de resample solo para
cambiar el nombre de columna) sería la reimplementación que el pliego
pide evitar.

`es_evento` se queda FUERA, a propósito (Laura, log de este PR: "puedes
dejarla fuera, o incluirla si sale gratis del mismo camino"). No sale
gratis: no es un indicador de e·sios, es una anotación manual de dos
ventanas históricas conocidas (28-A y 11-J, ver
`notebooks/analisis_historico_2_demandas.ipynb` celda 13) que alguien
tendría que revisar a mano fecha a fecha sobre el rango nuevo -- eso es
curación editorial, no descarga, y no entra en el alcance de este script.

Uso:
    python scripts/descargar_2026h2.py                  # normal: escribe demanda_horaria_extendida.parquet
    python scripts/descargar_2026h2.py --salida ruta.parquet   # verificación: no toca data/processed/ real
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from pipeline.predecir import encontrar_ultimo_dia_completo
from src.datos import descargar_demanda_cruda, obtener_token, resample_horario_con_conteo
from src.esios_client import crear_sesion, descargar_indicador
from src.paths import DIR_DATA, DIR_PROCESSED

TZ_MADRID = "Europe/Madrid"

# Mismo indicador que usa notebooks/descarga_historico.ipynb para construir
# la columna demanda_prevista del fichero congelado (misma llamada,
# `descargar_rango(sesion, 544, ..., 'demanda_prevista')`).
INDICADOR_DEMANDA_PREVISTA = 544

RUTA_ORIGINAL = DIR_PROCESSED / "demanda_horaria.parquet"
RUTA_ERRORES = DIR_DATA / "errores.csv"
RUTA_EXTENDIDA_DEFECTO = DIR_PROCESSED / "demanda_horaria_extendida.parquet"

INICIO_DESCARGA = pd.Timestamp("2026-07-01T00:00:00Z")

# Margen de solape GENUINO (pliego §4): sin este margen, "comparar el tramo
# común con el original" sería tautológico -- el tramo nuevo (>= 1/7) no
# comparte ni una sola hora con el fichero congelado (que termina el 30/6),
# así que un pd.concat sin re-descargar nada pasaría el check por
# construcción, sin haber verificado nada de e·sios. Se descargan también
# estos días de junio, ya presentes en el fichero congelado, para
# comprobarlos de verdad contra una respuesta fresca de la API -- y se
# descartan antes de escribir el fichero final, que sigue empezando el 1/7
# tal como pide §2.
MARGEN_SOLAPE_DIAS = 3


def _descargar_por_meses(inicio: pd.Timestamp, fin: pd.Timestamp) -> pd.DataFrame:
    """Descarga demanda_real cruda (5') entre `inicio` y `fin` (semiabierto),
    troceada en bloques mensuales -- mismo patrón de `descargar_rango`
    (pliego §2: "Descarga en bloques mensuales... replicar el patrón
    existente"), pero llamando a `descargar_demanda_cruda` en cada tramo en
    vez de a `descargar_rango` (ver docstring del módulo, motivo en el
    log). Los bloques dan puntos de control más pequeños; ninguno se salta
    ni se atrapa en un `except` amplio -- si e·sios falla, esto para en
    rojo."""
    cortes = list(pd.date_range(inicio, fin, freq="MS", tz="UTC"))
    if not cortes or cortes[0] != inicio:
        cortes = [inicio] + cortes
    if cortes[-1] != fin:
        cortes = cortes + [fin]

    trozos = []
    for tramo_inicio, tramo_fin in zip(cortes[:-1], cortes[1:]):
        if tramo_inicio >= tramo_fin:
            continue
        print(f"Descargando {tramo_inicio} -> {tramo_fin} ...")
        df_tramo = descargar_demanda_cruda(tramo_inicio, tramo_fin)
        print(f"  {len(df_tramo)} lecturas de 5' recibidas.")
        trozos.append(df_tramo)

    return pd.concat(trozos, ignore_index=True).sort_values("datetime_utc").reset_index(drop=True)


def _descargar_prevista_por_meses(inicio: pd.Timestamp, fin: pd.Timestamp) -> pd.DataFrame:
    """Igual que `_descargar_por_meses`, pero para `demanda_prevista`
    (indicador 544): `src.datos.descargar_demanda_cruda` está fijada al
    indicador 1293 (demanda_real), así que aquí se llama a
    `descargar_indicador` directamente -- el mismo primitivo de bajo nivel
    que usa `descargar_demanda_cruda` por dentro, ni un HTTP ni una
    normalización reimplementados."""
    sesion = crear_sesion(obtener_token())
    cortes = list(pd.date_range(inicio, fin, freq="MS", tz="UTC"))
    if not cortes or cortes[0] != inicio:
        cortes = [inicio] + cortes
    if cortes[-1] != fin:
        cortes = cortes + [fin]

    trozos = []
    for tramo_inicio, tramo_fin in zip(cortes[:-1], cortes[1:]):
        if tramo_inicio >= tramo_fin:
            continue
        print(f"Descargando demanda_prevista {tramo_inicio} -> {tramo_fin} ...")
        df_tramo = descargar_indicador(
            sesion, INDICADOR_DEMANDA_PREVISTA, tramo_inicio, tramo_fin, "demanda_prevista"
        )
        print(f"  {len(df_tramo)} lecturas de 5' recibidas.")
        trozos.append(df_tramo)

    return pd.concat(trozos, ignore_index=True).sort_values("datetime_utc").reset_index(drop=True)


def _resamplear_reutilizando(crudo: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Agrega `crudo[columna]` a horario con conteo reutilizando
    `resample_horario_con_conteo` TAL CUAL -- esa función está fijada a la
    columna `demanda_real` (ver `src/datos.py`), así que se renombra antes
    de llamarla y se deshace el renombrado después. La alternativa (una
    segunda función de resample que solo cambia el nombre de columna) sería
    la reimplementación de la agregación que el pliego pide evitar."""
    renombrado = crudo.rename(columns={columna: "demanda_real"})
    resultado = resample_horario_con_conteo(renombrado)
    return resultado.rename(columns={"demanda_real": columna, "n_lecturas": f"n_lecturas_{columna}"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--salida",
        type=Path,
        default=RUTA_EXTENDIDA_DEFECTO,
        help=f"Ruta del parquet a escribir (default: {RUTA_EXTENDIDA_DEFECTO}).",
    )
    args = parser.parse_args()

    # Ventana de descarga generosa hacia "ahora": el corte real lo decide
    # encontrar_ultimo_dia_completo() sobre el dato, no el reloj (mismo
    # criterio que pipeline/predecir.py). Un margen de sobra no hace daño:
    # las horas que aún no existen simplemente no vienen en la respuesta.
    ahora_utc = pd.Timestamp.now(tz="UTC")
    fin_sondeo = (ahora_utc + pd.Timedelta(days=1)).normalize()
    inicio_con_margen = INICIO_DESCARGA - pd.Timedelta(days=MARGEN_SOLAPE_DIAS)

    print(f"Sondeando {inicio_con_margen} -> {fin_sondeo} (ahora: {ahora_utc}) ...")
    crudo = _descargar_por_meses(inicio_con_margen, fin_sondeo)
    if crudo.empty:
        raise RuntimeError("e·sios no devolvió ninguna lectura para el rango solicitado.")

    horario_con_conteo = resample_horario_con_conteo(crudo)

    # demanda_prevista (544): mismo rango, mismo margen de solape -- para
    # poder verificarla también contra el fichero congelado. demanda_real
    # sigue gobernando el corte y qué horas entran (más abajo); esta serie
    # se agrega en paralelo y se cruza por datetime_utc, nunca al revés.
    print(f"\nSondeando demanda_prevista {inicio_con_margen} -> {fin_sondeo} ...")
    crudo_prevista = _descargar_prevista_por_meses(inicio_con_margen, fin_sondeo)
    if crudo_prevista.empty:
        raise RuntimeError("e·sios no devolvió ninguna lectura de demanda_prevista.")
    horario_prevista_con_conteo = _resamplear_reutilizando(crudo_prevista, "demanda_prevista")

    dia_ancla_madrid = encontrar_ultimo_dia_completo(horario_con_conteo)
    if dia_ancla_madrid is None:
        raise RuntimeError(
            "Ningún día natural de Madrid completo en la ventana descargada -- "
            "nada que escribir."
        )
    # DateOffset, no Timedelta -- mismo motivo que en encontrar_ultimo_dia_completo:
    # el corte debe seguir cayendo en medianoche local pase lo que pase con el DST.
    fin_utc_exclusivo = (dia_ancla_madrid + pd.DateOffset(days=1)).tz_convert("UTC")
    print(
        f"Último día natural de Madrid completo: {dia_ancla_madrid.date()} "
        f"(corte UTC exclusivo: {fin_utc_exclusivo})."
    )

    tramo = horario_con_conteo[
        (horario_con_conteo["datetime_utc"] >= INICIO_DESCARGA)
        & (horario_con_conteo["datetime_utc"] < fin_utc_exclusivo)
    ].sort_values("datetime_utc").reset_index(drop=True)

    incompletas = tramo[tramo["n_lecturas"] != 12]
    completas = tramo[tramo["n_lecturas"] == 12].drop(columns="n_lecturas").reset_index(drop=True)

    print(f"\nHoras en el tramo [{INICIO_DESCARGA}, {fin_utc_exclusivo}): {len(tramo)}")
    print(f"Horas completas de demanda_real (n_lecturas == 12): {len(completas)}")
    print(f"Horas incompletas descartadas (n_lecturas != 12): {len(incompletas)}")
    if not incompletas.empty:
        print(incompletas[["datetime_utc", "n_lecturas"]].to_string(index=False))

    # demanda_prevista: mismo filtro de calidad, aplicado por separado --
    # una hora incompleta de demanda_prevista NO tira la fila (demanda_real
    # sigue gobernando qué horas hay), la deja con demanda_prevista en NaN,
    # igual que ya se comporta el fichero original (nunca se descarta una
    # fila por un problema solo en esa columna).
    tramo_prevista = horario_prevista_con_conteo[
        (horario_prevista_con_conteo["datetime_utc"] >= INICIO_DESCARGA)
        & (horario_prevista_con_conteo["datetime_utc"] < fin_utc_exclusivo)
    ].sort_values("datetime_utc").reset_index(drop=True)
    incompletas_prevista = tramo_prevista[tramo_prevista["n_lecturas_demanda_prevista"] != 12]
    completas_prevista = (
        tramo_prevista[tramo_prevista["n_lecturas_demanda_prevista"] == 12]
        .drop(columns="n_lecturas_demanda_prevista")
        .reset_index(drop=True)
    )
    print(f"\nHoras completas de demanda_prevista (n_lecturas == 12): {len(completas_prevista)}")
    print(f"Horas incompletas de demanda_prevista descartadas: {len(incompletas_prevista)}")
    if not incompletas_prevista.empty:
        print(incompletas_prevista[["datetime_utc", "n_lecturas_demanda_prevista"]].to_string(index=False))

    # Cruce: demanda_prevista se añade por datetime_utc, con NaN donde no
    # haya hora completa de prevista para esa hora de demanda_real.
    completas = completas.merge(
        completas_prevista[["datetime_utc", "demanda_prevista"]], on="datetime_utc", how="left"
    )
    nulos_prevista = int(completas["demanda_prevista"].isna().sum())
    print(f"Horas de demanda_real sin demanda_prevista correspondiente: {nulos_prevista}")

    # Continuidad: dentro del tramo completo, ninguna hora ausente aparte de
    # las incompletas ya declaradas arriba (pliego §4).
    rango_horas_esperado = pd.date_range(INICIO_DESCARGA, fin_utc_exclusivo, freq="h", inclusive="left")
    horas_presentes = set(tramo["datetime_utc"])
    huecos = sorted(set(rango_horas_esperado) - horas_presentes)
    print(f"Huecos de continuidad (horas ni completas ni incompletas -- ausentes del todo): {len(huecos)}")
    if huecos:
        print([str(h) for h in huecos])

    # Solape GENUINO (pliego §4, ver MARGEN_SOLAPE_DIAS arriba): re-descarga
    # independiente de los últimos días de junio, ya en el fichero
    # congelado, comparada hora a hora contra ese fichero. Esto sí puede
    # fallar si e·sios ha revisado datos históricos -- a diferencia de
    # comparar el original contra sí mismo, que sería tautológico.
    original = pd.read_parquet(RUTA_ORIGINAL)[["datetime_utc", "demanda_real", "demanda_prevista"]]
    margen = horario_con_conteo[
        (horario_con_conteo["datetime_utc"] >= inicio_con_margen)
        & (horario_con_conteo["datetime_utc"] < INICIO_DESCARGA)
    ].sort_values("datetime_utc").reset_index(drop=True)
    original_margen = original[
        original["datetime_utc"].isin(margen["datetime_utc"])
    ].sort_values("datetime_utc").reset_index(drop=True)

    assert len(margen) == len(original_margen), (
        f"El margen de solape trajo {len(margen)} horas, el original tiene "
        f"{len(original_margen)} para ese mismo rango -- no son directamente "
        "comparables."
    )
    assert (margen["n_lecturas"] == 12).all(), (
        "El margen de solape (ya publicado hace semanas) tiene horas "
        "incompletas -- inesperado, revisar antes de seguir."
    )
    diferencias_margen = (margen["demanda_real"] - original_margen["demanda_real"]).abs()
    diferencia_maxima_margen = float(diferencias_margen.max())
    print(
        f"\nSolape genuino, demanda_real ({inicio_con_margen} -> {INICIO_DESCARGA}, "
        f"{len(margen)} horas re-descargadas de e·sios): "
        f"diferencia máxima = {diferencia_maxima_margen}"
    )
    assert diferencia_maxima_margen == 0.0, (
        "El margen de solape no reproduce el original exactamente -- posible "
        "revisión de datos históricos por parte de e·sios. NO se escribe el "
        "fichero."
    )

    # Mismo solape genuino, para demanda_prevista.
    margen_prevista = horario_prevista_con_conteo[
        (horario_prevista_con_conteo["datetime_utc"] >= inicio_con_margen)
        & (horario_prevista_con_conteo["datetime_utc"] < INICIO_DESCARGA)
    ].sort_values("datetime_utc").reset_index(drop=True)
    assert len(margen_prevista) == len(original_margen), (
        f"El margen de solape de demanda_prevista trajo {len(margen_prevista)} horas, "
        f"el original tiene {len(original_margen)} -- no son directamente comparables."
    )
    assert (margen_prevista["n_lecturas_demanda_prevista"] == 12).all(), (
        "El margen de solape de demanda_prevista (ya publicado hace semanas) tiene "
        "horas incompletas -- inesperado, revisar antes de seguir."
    )
    diferencias_margen_prevista = (
        margen_prevista["demanda_prevista"] - original_margen["demanda_prevista"]
    ).abs()
    diferencia_maxima_margen_prevista = float(diferencias_margen_prevista.max())
    print(
        f"Solape genuino, demanda_prevista: diferencia máxima = "
        f"{diferencia_maxima_margen_prevista}"
    )
    assert diferencia_maxima_margen_prevista == 0.0, (
        "El margen de solape de demanda_prevista no reproduce el original "
        "exactamente. NO se escribe el fichero."
    )

    # Concatenación con el histórico congelado -- se lee, nunca se escribe.
    extendida = (
        pd.concat([original, completas], ignore_index=True)
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )

    print(f"\nFilas en el fichero extendido: {len(extendida)} "
          f"({len(original)} originales + {len(completas)} nuevas)")
    print(f"Rango: {extendida['datetime_utc'].min()} -> {extendida['datetime_utc'].max()}")

    # Construcción: el tramo común (>= 1/7, si acaso) sigue debiendo
    # reproducir el original exacto -- por construcción (pd.concat de las
    # mismas filas), pero se verifica igual como red de seguridad frente a
    # bugs de la propia concatenación (dtype, orden, duplicados).
    comun = extendida[extendida["datetime_utc"].isin(original["datetime_utc"])].reset_index(drop=True)
    original_ordenado = original.sort_values("datetime_utc").reset_index(drop=True)
    assert len(comun) == len(original_ordenado), (
        f"El tramo común tiene {len(comun)} filas, se esperaban {len(original_ordenado)}."
    )
    diferencia_maxima_construccion = float(
        (comun["demanda_real"] - original_ordenado["demanda_real"]).abs().max()
    )
    diferencia_maxima_construccion_prevista = float(
        (comun["demanda_prevista"] - original_ordenado["demanda_prevista"]).abs().max()
    )
    print(
        f"Solape por construcción (autoconsistencia): demanda_real="
        f"{diferencia_maxima_construccion}, demanda_prevista="
        f"{diferencia_maxima_construccion_prevista}"
    )
    assert diferencia_maxima_construccion == 0.0
    assert diferencia_maxima_construccion_prevista == 0.0

    # Cruce con producción: valor_real de agosto en errores.csv debe coincidir.
    if RUTA_ERRORES.exists():
        errores = pd.read_csv(RUTA_ERRORES, float_precision="round_trip")
        errores_horario = (
            errores[["horizonte", "valor_real"]]
            .drop_duplicates(subset="horizonte")
            .assign(datetime_utc=lambda d: pd.to_datetime(d["horizonte"], utc=True))
        )
        cruce = extendida.merge(
            errores_horario[["datetime_utc", "valor_real"]], on="datetime_utc", how="inner"
        )
        if cruce.empty:
            print("\nCruce con errores.csv: sin horas en común todavía.")
        else:
            dif_cruce = (cruce["demanda_real"] - cruce["valor_real"]).abs()
            print(
                f"\nCruce con errores.csv: {len(cruce)} horas en común, "
                f"diferencia máxima = {float(dif_cruce.max())}"
            )
            assert dif_cruce.max() == 0.0, (
                "demanda_real (descarga nueva) y valor_real (errores.csv, producción) "
                "no coinciden en alguna hora de agosto -- hay que entenderlo antes de seguir."
            )
    else:
        print("\ndata/errores.csv no existe: sin cruce con producción que hacer.")

    maximo = extendida["demanda_real"].max()
    fila_maximo = extendida.loc[extendida["demanda_real"].idxmax()]
    print(
        f"\nMáximo de la serie nueva: {maximo:,.1f} MW "
        f"({fila_maximo['datetime_utc']}). Techo de v1 (predicción máxima): 38.861,1 MW."
    )
    veces_superado = int((extendida["demanda_real"] > 38_861.1).sum())
    print(f"Horas con demanda_real > 38.861,1 MW: {veces_superado}")

    nulos_prevista_total = int(extendida["demanda_prevista"].isna().sum())
    print(
        f"\ndemanda_prevista: {nulos_prevista_total} horas en NaN sobre "
        f"{len(extendida)} totales (0 esperadas en el tramo original, congelado)."
    )

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    extendida.to_parquet(args.salida, index=False)
    print(f"\nParquet escrito en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
