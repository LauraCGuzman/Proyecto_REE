"""Predicción diaria de demanda (Fase 1.5 / Fase 2).

El ancla sale del dato, no del reloj (decisión del 12/8): se busca en la
serie descargada el último día natural de Madrid con TODAS sus horas
completas (`n_lecturas == 12` en cada una, ver `src.datos.
resample_horario_con_conteo`) y se predice el día natural de Madrid
siguiente a ese ancla.

"Día natural de Madrid" no son 24 horas fijas: el nº de horas UTC que caen
dentro de un día local varía (23/24/25) en los cambios de hora. El cálculo
de qué horas tocan predecir se hace siempre en UTC tz-aware a partir del
calendario, nunca fijando `24`; el propio `demanda_lag_24` (un desplazamiento
de 24 horas UTC, no "misma hora local del día anterior") ya es correcto en
UTC sin importar el DST -- lo único que varía con el DST es cuántas de esas
horas UTC entran en un día natural de Madrid dado.

Si el último día completo no es ayer (Madrid): el job falla en rojo y no
escribe nada. Predecir el día siguiente a un ancla más antigua que ayer
sería predecir un día cuyo dato real ya existe -- no es una previsión, y
contaminaría el MAE publicado en `data/metricas.json` (Fase 2, evaluar.py --
"datos/" del pliego original renombrado a "data/" aquí para no tener dos
convenciones de nombre de carpeta, ver `data/predicciones.csv` más abajo).
`--recuperar` permite explícitamente ese caso para uso manual; no debe
usarse nunca desde el cron de Fase 3.

Fase 2 (16/8): `data/ancla_usada.csv` persiste la entrada real que produjo
cada predicción (append-only, condicionado a que `guardar_predicciones`
haya escrito de verdad -- ver su docstring), y `src.quality.
assert_rango_fisico` valida tanto `demanda_lag_24` como las predicciones
del modelo contra el rango físicamente posible de la demanda peninsular.

Emisión paralela (Fase 5bis, PR B): `--modelo` es obligatorio, sin default.
El cron invoca este script una vez por modelo, en pasos y commits separados
(pliego §3.1) -- si v2 falla, v1 ya emitió y commiteó antes, y el paso de v2
falla en rojo sin tocar lo de v1 (nada de `try/except` ni `continue-on-error`
alrededor: el aislamiento lo da el orden de los pasos, no una guarda).

Uso:
    predecir.py --modelo v1                # normal (cron): ancla == ayer (Madrid).
    predecir.py --modelo v2                # normal (cron): mismo ancla, más demanda_lag_168.
    predecir.py --modelo v1 --recuperar    # manual: acepta un ancla más antigua.
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

from src.datos import descargar_demanda_cruda, resample_horario_con_conteo
from src.features import (
    FEATURES_5F,
    anadir_calendario,
    anadir_festivos_puentes,
    mapear_tipo_efectivo,
)
from src.features_v2 import FEATURES_6F
from src.modelo import RUTA_METADATOS_V1, RUTA_MODELO_V1, cargar_metadatos, cargar_modelo
from src.modelo import predecir as predecir_con_modelo
from src.paths import DIR_DATA, DIR_MODELOS
from src.quality import assert_rango_fisico

TZ_MADRID = "Europe/Madrid"

# Emisión paralela (Fase 5bis, PR B): cada modelo declara aquí su pkl, su
# JSON de metadatos y la lista de features con la que predice. v1 no cambia
# de ruta -- son las mismas constantes con las que ya predecía
# (RUTA_MODELO_V1/RUTA_METADATOS_V1 de src/modelo.py), citadas aquí de forma
# explícita en vez de dejarlas implícitas en el default de `cargar_modelo`.
# El orden del dict es el orden de los pasos del pliego (§3.1): v1 primero.
MODELOS = {
    "v1": {
        "ruta_modelo": RUTA_MODELO_V1,
        "ruta_metadatos": RUTA_METADATOS_V1,
        "features": FEATURES_5F,
    },
    "v2": {
        "ruta_modelo": DIR_MODELOS / "modelo_v2.pkl",
        "ruta_metadatos": DIR_MODELOS / "modelo_v2.json",
        "features": FEATURES_6F,
    },
}

# Margen de descarga: no es "cuántos días hace falta para el lag" (el
# lag_24 de las predicciones solo necesita el propio día ancla) sino
# "cuánto retrocedo buscando un día completo si ayer no lo está" -- el 28-A
# y el 11-J son el precedente de que la telemetría puede fallar más de un día.
#
# 9 y no 8 (Bloque A, preparación de `lag_168`): `lag_168` necesita 7 días
# completos hacia atrás desde la primera hora del horizonte; el horizonte
# arranca a las 22:00Z del día anterior al objetivo, y el `.normalize()` de
# `inicio_descarga` (más abajo) recorta al comienzo del día -- 8 deja el
# margen a cero y cualquier retraso de publicación de e·sios lo convierte en
# un NaN; 9 deja un día de holgura. v1 solo usa `lag_24`, así que el margen
# extra no le cambia ninguna predicción -- verificado de nuevo en el PR B
# (emisión paralela): paridad de v1 a tolerancia 0 con la ventana ya a 9
# días, ver log del PR B. v2 (este PR) es quien empieza a usar el margen
# entero, para `demanda_lag_168`.
DIAS_VENTANA_DESCARGA = 9

RUTA_PREDICCIONES = DIR_DATA / "predicciones.csv"
COLUMNAS_PREDICCIONES = [
    "fecha_emision",
    "horizonte",
    "valor_predicho",
    "ancla_ultimo_dia_real",
    "modelo",
]

# `data/ancla_usada.csv`: la entrada que produjo cada predicción, no solo el
# número emitido (decisión de Laura, 16/8). Append-only, igual que
# `predicciones.csv`. Sin `modelo`: el dato de entrada no depende del modelo
# -- el join con `predicciones.csv` va por `fecha_emision` + `ancla_
# ultimo_dia_real`.
RUTA_ANCLA_USADA = DIR_DATA / "ancla_usada.csv"
COLUMNAS_ANCLA_USADA = [
    "fecha_emision",
    "datetime_utc_ancla",
    "demanda_real",
    "n_lecturas",
    "ancla_ultimo_dia_real",
]


def encontrar_ultimo_dia_completo(df_horario_conteo: pd.DataFrame) -> pd.Timestamp | None:
    """Recorre los días naturales de Madrid presentes en `df_horario_conteo`
    (más reciente primero) y devuelve la medianoche Madrid (tz-aware) del
    último cuyas horas UTC están TODAS presentes con `n_lecturas == 12`.
    `None` si ninguno lo está."""
    df = df_horario_conteo.copy()
    df["fecha_madrid"] = df["datetime_utc"].dt.tz_convert(TZ_MADRID).dt.normalize()

    for dia in sorted(df["fecha_madrid"].unique(), reverse=True):
        inicio_utc = dia.tz_convert("UTC")
        # DateOffset, no Timedelta: sumar un día natural de Madrid a una
        # medianoche local debe seguir siendo medianoche local al día
        # siguiente. Timedelta(days=1) es una duración fija de 24h
        # absolutas -- en un día de cambio de hora deja de caer a
        # medianoche (ver commit del 12/8, se comprobó con un test DST).
        fin_utc = (dia + pd.DateOffset(days=1)).tz_convert("UTC")
        horas_esperadas = int((fin_utc - inicio_utc) / pd.Timedelta(hours=1))

        filas_dia = df[(df["datetime_utc"] >= inicio_utc) & (df["datetime_utc"] < fin_utc)]
        if len(filas_dia) == horas_esperadas and (filas_dia["n_lecturas"] == 12).all():
            return dia
    return None


def construir_target_df(
    dia_ancla_madrid: pd.Timestamp, serie_referencia: pd.Series
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Filas a predecir para el día natural de Madrid siguiente a
    `dia_ancla_madrid`. `serie_referencia` es demanda_real horaria indexada
    por datetime_utc, restringida a horas ya confirmadas completas --
    `demanda_lag_24` sale de ahí por búsqueda directa (T - 24h), no por
    `shift` posicional: un shift asume contigüidad que aquí no está
    garantizada más allá del propio día ancla."""
    # DateOffset, no Timedelta -- ver nota en encontrar_ultimo_dia_completo.
    dia_target_madrid = dia_ancla_madrid + pd.DateOffset(days=1)
    inicio_utc = dia_target_madrid.tz_convert("UTC")
    fin_utc = (dia_target_madrid + pd.DateOffset(days=1)).tz_convert("UTC")

    timestamps_utc = pd.date_range(inicio_utc, fin_utc, freq="1h", inclusive="left")
    assert len(timestamps_utc) in (23, 24, 25), (
        f"Nº de horas del día objetivo fuera de lo físicamente posible en un "
        f"cambio de hora: {len(timestamps_utc)}"
    )

    df_target = pd.DataFrame({"datetime_utc": timestamps_utc})
    df_target = anadir_calendario(df_target)
    df_target = anadir_festivos_puentes(df_target)
    df_target = mapear_tipo_efectivo(df_target)

    df_target["demanda_lag_24"] = serie_referencia.reindex(
        (df_target["datetime_utc"] - pd.Timedelta(hours=24)).to_numpy()
    ).to_numpy()

    n_nan = int(df_target["demanda_lag_24"].isna().sum())
    assert n_nan == 0, (
        f"{n_nan} fila(s) a predecir sin demanda_lag_24 (invariante de "
        "salida): no se emite ninguna predicción parcial."
    )

    # Una entrada absurda en demanda_lag_24 no da error: da una predicción
    # plausible y falsa (pliego Fase 2 §1, punto 2).
    assert_rango_fisico(
        df_target.set_index("datetime_utc")["demanda_lag_24"],
        "predecir.py: demanda_lag_24 (construir_target_df)",
    )

    return df_target, dia_target_madrid


def anadir_lag_168_prediccion(
    df_target: pd.DataFrame, serie_referencia: pd.Series
) -> pd.DataFrame:
    """demanda_lag_168 para el camino de predicción de v2 -- mismo mecanismo
    que `demanda_lag_24` en `construir_target_df`: búsqueda directa de
    timestamp (T - 168h) contra `serie_referencia`, no `shift` posicional.
    NO usa `src.features_v2.construir_features_6f`/`anadir_lag_168`: esas
    funciones asumen la serie contigua y completa del camino de
    entrenamiento (histórico horario sin huecos), y aquí no lo está más allá
    del propio día ancla -- exactamente la misma razón por la que
    `construir_target_df` ya evita `shift` para `demanda_lag_24` (ver su
    docstring). Solo se llama para v2; v1 no la necesita."""
    df_target = df_target.copy()
    df_target["demanda_lag_168"] = serie_referencia.reindex(
        (df_target["datetime_utc"] - pd.Timedelta(hours=168)).to_numpy()
    ).to_numpy()

    n_nan = int(df_target["demanda_lag_168"].isna().sum())
    assert n_nan == 0, (
        f"{n_nan} fila(s) a predecir sin demanda_lag_168 (invariante de "
        "salida): no se emite ninguna predicción parcial de v2. Revisa el "
        "margen de DIAS_VENTANA_DESCARGA frente a T-168h."
    )

    # Misma guarda que sobre demanda_lag_24: una entrada absurda no da
    # error, da una predicción plausible y falsa.
    assert_rango_fisico(
        df_target.set_index("datetime_utc")["demanda_lag_168"],
        "predecir.py: demanda_lag_168 (anadir_lag_168_prediccion)",
    )

    return df_target


def ancla_ya_registrada(dia_ancla_madrid: pd.Timestamp) -> bool:
    """True si `ancla_usada.csv` ya tiene una fila para este ancla
    (`ancla_ultimo_dia_real`), sin importar qué modelo la disparó. El dato
    de entrada del ancla (T-24h) no depende del modelo (decisión del 16/8):
    con dos modelos leyendo la misma ventana en la misma corrida, sigue
    siendo cierto -- pero ahora `main()` se invoca una vez por modelo
    (pliego Fase5bis PR B, §3.1), así que hace falta esta comprobación para
    que el ancla se escriba una sola vez por corrida, no una vez por modelo
    (si los dos modelos la escribieran, se duplicaría -- pliego, §3.2 punto 2)."""
    if not RUTA_ANCLA_USADA.exists():
        return False
    existentes = pd.read_csv(RUTA_ANCLA_USADA, dtype=str)
    return bool((existentes["ancla_ultimo_dia_real"] == dia_ancla_madrid.date().isoformat()).any())


def guardar_predicciones(filas_nuevas: pd.DataFrame) -> bool:
    """Append-only. Si `RUTA_PREDICCIONES` ya existe, no reescribe nada: si
    TODAS las filas nuevas ya estaban (misma clave horizonte+modelo), es un
    rerun del mismo día y no hace nada; si el solape es parcial, para --eso
    no debería pasar en un rerun normal.

    Devuelve `True` si escribió, `False` si fue un rerun idempotente. El
    llamador usa este valor para decidir si escribe también `ancla_usada.csv`
    (ver `main`): si se escribiera sin consultar esto, un rerun duplicaría
    las filas del ancla con `fecha_emision` distinta (es `now()`), y la
    deduplicación por clave no las cazaría."""
    if RUTA_PREDICCIONES.exists():
        existentes = pd.read_csv(RUTA_PREDICCIONES, dtype=str)
        clave_existente = set(zip(existentes["horizonte"], existentes["modelo"]))
        ya_escritas = filas_nuevas.apply(
            lambda f: (f["horizonte"], f["modelo"]) in clave_existente, axis=1
        )

        assert ya_escritas.all() or not ya_escritas.any(), (
            "Solape parcial con predicciones.csv: algunas horas del día "
            "objetivo ya estaban escritas y otras no. No debería pasar en "
            "un rerun normal -- revisa manualmente antes de continuar."
        )
        if ya_escritas.all():
            print("Ya estaba escrito (idempotencia) -- nada que añadir.")
            return False
        filas_nuevas.to_csv(RUTA_PREDICCIONES, mode="a", header=False, index=False)
    else:
        RUTA_PREDICCIONES.parent.mkdir(parents=True, exist_ok=True)
        filas_nuevas.to_csv(RUTA_PREDICCIONES, mode="w", header=True, index=False)
    return True


def construir_ancla_usada(
    df_target: pd.DataFrame,
    df_horario: pd.DataFrame,
    dia_ancla_madrid: pd.Timestamp,
    fecha_emision: str,
) -> pd.DataFrame:
    """Filas de `df_horario` que `demanda_lag_24` consumió realmente para
    construir `df_target` -- no la ventana de descarga completa de
    `DIAS_VENTANA_DESCARGA` días, solo `df_target["datetime_utc"] - 24h`
    (23/24/25 filas, una por hora del día objetivo). `n_lecturas` es 12 por
    construcción (viene de `serie_referencia`, que ya filtra por eso en
    `main`) -- se guarda igualmente para que el esquema sobreviva sin
    migración cuando llegue la Fase 5."""
    datetime_utc_ancla = df_target["datetime_utc"] - pd.Timedelta(hours=24)
    referencia = df_horario.set_index("datetime_utc").loc[datetime_utc_ancla]

    return pd.DataFrame(
        {
            "fecha_emision": fecha_emision,
            "datetime_utc_ancla": datetime_utc_ancla.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "demanda_real": referencia["demanda_real"].to_numpy(),
            "n_lecturas": referencia["n_lecturas"].to_numpy(),
            "ancla_ultimo_dia_real": dia_ancla_madrid.date().isoformat(),
        }
    )[COLUMNAS_ANCLA_USADA]


def guardar_ancla_usada(filas_nuevas: pd.DataFrame) -> None:
    """Append-only. Sin guarda de idempotencia propia: el llamador (`main`)
    solo invoca esto cuando `guardar_predicciones` devolvió `True` -- ver su
    docstring."""
    if RUTA_ANCLA_USADA.exists():
        filas_nuevas.to_csv(RUTA_ANCLA_USADA, mode="a", header=False, index=False)
    else:
        RUTA_ANCLA_USADA.parent.mkdir(parents=True, exist_ok=True)
        filas_nuevas.to_csv(RUTA_ANCLA_USADA, mode="w", header=True, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recuperar",
        action="store_true",
        help=(
            "Permite anclar en un día completo distinto de ayer (Madrid). "
            "Uso manual solamente -- NUNCA en el cron de Fase 3."
        ),
    )
    parser.add_argument(
        "--modelo",
        choices=sorted(MODELOS),
        required=True,
        help=(
            "Modelo con el que predecir (emisión paralela, Fase 5bis PR B). "
            "Sin default deliberado: el cron invoca este script una vez por "
            "modelo, en pasos y commits separados (pliego §3.1) -- un "
            "default silencioso escondería si algún paso se lanzó sin "
            "--modelo."
        ),
    )
    args = parser.parse_args()
    modelo = args.modelo
    config_modelo = MODELOS[modelo]

    ahora_utc = pd.Timestamp.now(tz="UTC")
    inicio_descarga = (ahora_utc - pd.Timedelta(days=DIAS_VENTANA_DESCARGA)).normalize()

    df_crudo = descargar_demanda_cruda(inicio_descarga, ahora_utc)
    df_horario = resample_horario_con_conteo(df_crudo)

    dia_ancla = encontrar_ultimo_dia_completo(df_horario)
    if dia_ancla is None:
        raise RuntimeError(
            f"Ningún día natural de Madrid completo en los últimos "
            f"{DIAS_VENTANA_DESCARGA} días. No se puede anclar la predicción."
        )

    ayer_madrid = ahora_utc.tz_convert(TZ_MADRID).normalize() - pd.DateOffset(days=1)
    if dia_ancla != ayer_madrid and not args.recuperar:
        raise RuntimeError(
            f"El último día completo en los datos es {dia_ancla.date()}, no "
            f"ayer ({ayer_madrid.date()}). Predecir el día siguiente a "
            f"{dia_ancla.date()} sería un día cuyo dato real ya existe -- "
            "contaminaría el MAE publicado. No se escribe nada. Si esto es "
            "una recuperación manual deliberada, relanza con --recuperar."
        )

    serie_referencia = (
        df_horario.loc[df_horario["n_lecturas"] == 12]
        .set_index("datetime_utc")["demanda_real"]
    )

    df_target, dia_target_madrid = construir_target_df(dia_ancla, serie_referencia)

    if modelo == "v2":
        # Única diferencia de v2 en el camino de predicción: una columna más
        # (demanda_lag_168), resuelta por el mismo mecanismo de búsqueda que
        # demanda_lag_24 -- ver docstring de anadir_lag_168_prediccion.
        df_target = anadir_lag_168_prediccion(df_target, serie_referencia)

    # Candado nº 2 (pliego Fase5bis PR B, §3.3): la lista de features con la
    # que se predice tiene que ser IDÉNTICA (mismo nombre, mismo orden) a la
    # declarada en el JSON de metadatos propio del modelo. El pipeline no
    # puede inventar, añadir ni reordenar features -- si el JSON y
    # `config_modelo["features"]` alguna vez divergen, esto para en rojo
    # antes de predecir nada, no después.
    metadatos_modelo = cargar_metadatos(config_modelo["ruta_metadatos"])
    features_modelo = config_modelo["features"]
    assert list(features_modelo) == metadatos_modelo["features"], (
        f"Candado nº 2: las features declaradas en "
        f"{config_modelo['ruta_metadatos'].name} "
        f"({metadatos_modelo['features']}) no coinciden con las que "
        f"predecir.py usa para {modelo} ({list(features_modelo)})."
    )

    modelo_cargado = cargar_modelo(ruta=config_modelo["ruta_modelo"])
    predicciones = predecir_con_modelo(modelo_cargado, df_target[features_modelo])

    # Hoy casi no puede saltar (un árbol solo devuelve medias de sus hojas):
    # su valor es de Fase 5, cuando el challenger se re-entrena con datos
    # frescos y este assert es lo único que hay entre unos datos de
    # entrenamiento corruptos y el CSV publicado (pliego Fase 2 §1, punto 3).
    assert_rango_fisico(
        pd.Series(predicciones, index=df_target["datetime_utc"].to_numpy()),
        "predecir.py: predicciones del modelo (main)",
    )

    fecha_emision = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    filas_nuevas = pd.DataFrame(
        {
            "fecha_emision": fecha_emision,
            "horizonte": df_target["datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valor_predicho": predicciones,
            "ancla_ultimo_dia_real": dia_ancla.date().isoformat(),
            "modelo": modelo,
        }
    )[COLUMNAS_PREDICCIONES]

    escrito = guardar_predicciones(filas_nuevas)
    # ancla_usada.csv no depende del modelo (§3.2 punto 2): se escribe una
    # sola vez por corrida aunque este script se invoque una vez por modelo
    # -- ver docstring de ancla_ya_registrada. Sin esta segunda comprobación,
    # la corrida del segundo modelo del día duplicaría las filas del ancla.
    if escrito and not ancla_ya_registrada(dia_ancla):
        ancla_usada = construir_ancla_usada(df_target, df_horario, dia_ancla, fecha_emision)
        guardar_ancla_usada(ancla_usada)

    print(
        f"Predicción emitida para {dia_target_madrid.date()} (Madrid, "
        f"{len(filas_nuevas)} horas, modelo {modelo}) -- ancla: "
        f"{dia_ancla.date()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
