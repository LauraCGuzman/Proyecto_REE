# src/features.py
"""Construcción de las 5 features del modelo publicado (`modelo_nivel_final`,
notebooks/modelo_demanda.ipynb, celda 28). Traslado literal del cálculo de
cada columna desde las celdas 6, 8, 9, 10 y 20 -- misma lógica, mismos
nombres de columna, mismo orden de operaciones.

Lo que SÍ cambia respecto al notebook, porque son artefactos de una pasada
de exploración sobre un dataset concreto y no forman parte de la
transformación en sí:
  - Los `assert len(df) == 30648` de las celdas 6 y 8 (atados al histórico
    completo 2023-2026H1): estas funciones también se llaman sobre lotes
    pequeños desde el pipeline diario, donde ese conteo no aplica.
  - Los `print` de verificación/inspección visual (conteos, head(), el caso
    de prueba del 2023-01-01).
Los asserts que SÍ son invariantes del dato en cualquier tamaño de lote se
conservan (continuidad horaria estricta de la celda 20, ausencia de nulos
tras el mapeo de `tipo_efectivo` de la celda 10) y se amplían en
`validar_invariantes_5f`, que `construir_features_5f` llama al final: 0 NaN
en hora/mes/tipo_efectivo/es_puente, categorías de `tipo_efectivo` ⊆ {0, 1},
y NaN de `demanda_lag_24` confinado al prefijo inicial del lote (nunca un
hueco interno). El notebook no necesitaba esto por escrito porque corre con
inspección visual de por medio; el pipeline diario corre solo.

Cualquier decisión no evidente en el propio código está documentada en la
celda origen del notebook, no aquí -- este módulo no reinterpreta el pliego,
lo traslada.
"""
from datetime import timedelta

import holidays
import numpy as np
import pandas as pd

# Lista blanca de features del modelo publicado (nivel, 5 features).
# Copia literal de `features_5f`, notebooks/modelo_demanda.ipynb celda 21.
FEATURES_5F = [
    "hora",
    "mes",
    "tipo_efectivo",
    "es_puente",
    "demanda_lag_24",
]

VARIABLE_OBJETIVO = "demanda_real"

# Copia literal del mapeo de la celda 10.
MAPEO_TIPO_EFECTIVO = {
    "laborable": 0,
    "no_laborable": 1,
}


def anadir_calendario(df: pd.DataFrame) -> pd.DataFrame:
    """hora, mes y tipo_dia en hora local de Madrid. Traslado literal de la
    celda 6 (sin el assert de conteo ni los prints de inspección, ver
    docstring del módulo)."""
    df_features = df.copy()

    datetime_madrid = df_features["datetime_utc"].dt.tz_convert("Europe/Madrid")

    df_features["hora"] = datetime_madrid.dt.hour
    df_features["mes"] = datetime_madrid.dt.month

    # tipo_dia: laborable (0-4 son L-V) o finde (5-6 son S-D)
    df_features["tipo_dia"] = np.where(
        datetime_madrid.dt.weekday < 5, "laborable", "finde"
    )

    return df_features


def anadir_festivos_puentes(df: pd.DataFrame) -> pd.DataFrame:
    """es_festivo y es_puente. Traslado literal de la celda 8 (requiere que
    `anadir_calendario` se haya aplicado antes: reutiliza el `datetime_utc`
    ya presente en `df`, no la columna `tipo_dia`).

    `holidays.ES(observed=False)`: con `observed=True` (el default de la
    librería) los festivos que caen en domingo desaparecen del calendario.
    Solo festivos nacionales (los autonómicos sobre-marcarían para demanda
    peninsular agregada)."""
    df_features = df.copy()

    datetime_madrid = df_features["datetime_utc"].dt.tz_convert("Europe/Madrid")

    fechas_locales = datetime_madrid.dt.date.unique()
    años_presentes = datetime_madrid.dt.year.unique()

    es_holidays = holidays.ES(years=años_presentes, observed=False)

    festivos_set = {d for d in fechas_locales if d in es_holidays}
    findes_set = {
        d for d in fechas_locales if d.weekday() >= 5
    }  # 5=Sábado, 6=Domingo

    puentes_set = set()
    for d in fechas_locales:
        # Un puente debe ser un día laborable y no festivo
        if d not in festivos_set and d not in findes_set:
            mañana = d + timedelta(days=1)
            ayer = d - timedelta(days=1)

            # Caso A: Lunes puente (Martes festivo y Domingo fin de semana)
            # Caso B: Viernes puente (Jueves festivo y Sábado fin de semana)
            if (mañana in festivos_set and ayer in findes_set) or (
                ayer in festivos_set and mañana in findes_set
            ):
                puentes_set.add(d)

    df_features["fecha_local"] = datetime_madrid.dt.date

    df_features["es_festivo"] = df_features["fecha_local"].isin(festivos_set)
    df_features["es_puente"] = df_features["fecha_local"].isin(puentes_set)

    df_features.drop(columns=["fecha_local"], inplace=True)

    return df_features


def mapear_tipo_efectivo(df: pd.DataFrame) -> pd.DataFrame:
    """tipo_efectivo binario (0 laborable / 1 no laborable). Traslado literal
    de las celdas 9 y 10 (requiere `tipo_dia` y `es_festivo`, ver
    `anadir_calendario` y `anadir_festivos_puentes`)."""
    df_features = df.copy()

    df_features["tipo_efectivo"] = np.where(
        (df_features["tipo_dia"] == "finde") | (df_features["es_festivo"] == True),
        "no_laborable",
        "laborable",
    )

    df_features["tipo_efectivo"] = df_features["tipo_efectivo"].map(
        MAPEO_TIPO_EFECTIVO
    )

    nulos_post_mapeo = df_features["tipo_efectivo"].isna().sum()
    assert nulos_post_mapeo == 0, (
        f"¡Alerta! Hay {nulos_post_mapeo} valores en 'tipo_efectivo' que no se "
        "mapearon. Revisa si el texto es exactamente 'laborable' o 'no_laborable'."
    )

    return df_features


def anadir_lag_24(df: pd.DataFrame) -> pd.DataFrame:
    """demanda_lag_24 = demanda_real de la misma hora, un día antes. Traslado
    literal de la celda 20 (pasos 1-3: orden cronológico, verificación de
    continuidad horaria y creación del lag; el resto de la celda 20 --
    partición train/test y dropna -- pertenece a la orquestación del
    notebook, no a la construcción de features)."""
    df_features = df.sort_values("datetime_utc").reset_index(drop=True)

    diferencias_tiempo = df_features["datetime_utc"].diff().dropna()
    horas_entre_filas = diferencias_tiempo.dt.total_seconds() / 3600.0

    assert (
        horas_entre_filas.unique() == [1.0]
    ).all(), "¡Alerta! La serie temporal tiene huecos o duplicados. El shift por posición fallará."

    df_features["demanda_lag_24"] = df_features["demanda_real"].shift(24)

    return df_features


def validar_invariantes_5f(df: pd.DataFrame) -> None:
    """Invariantes de las 5 features, en versión independiente del tamaño
    del lote -- el notebook los comprobaba contra un conteo fijo (30648) o a
    ojo sobre un `head()`; este módulo corre desatendido en el pipeline
    diario y no tiene esa inspección visual de por medio."""
    # hora, mes, tipo_efectivo, es_puente: nunca deben venir con NaN, sea
    # cual sea el tamaño del lote (a diferencia de demanda_lag_24, ver abajo).
    for col in ("hora", "mes", "tipo_efectivo", "es_puente"):
        n_nulos = int(df[col].isna().sum())
        assert n_nulos == 0, f"{n_nulos} NaN en '{col}' -- no debería poder pasar."

    # tipo_efectivo: solo las dos categorías del mapeo (celda 10), nunca un
    # tercer valor colado.
    categorias = set(df["tipo_efectivo"].unique())
    assert categorias <= {0, 1}, (
        f"tipo_efectivo tiene categorías fuera de {{0, 1}}: {categorias - {0, 1}}"
    )

    # demanda_lag_24: shift(24) sobre una serie contigua (ya garantizado por
    # el assert de continuidad de anadir_lag_24) solo puede dejar NaN en el
    # prefijo inicial del lote. Un NaN fuera de ese prefijo es un hueco
    # interno o un error de alineación -- la misma "trampa de montaje" que
    # la celda 20 comprueba sobre test, aquí generalizada a cualquier lote.
    n_prefijo = min(24, len(df))
    assert not df["demanda_lag_24"].iloc[n_prefijo:].isna().any(), (
        "demanda_lag_24 tiene NaN fuera del prefijo inicial esperado "
        f"(primeras {n_prefijo} filas): hueco interno o error de alineación."
    )


def construir_features_5f(df: pd.DataFrame) -> pd.DataFrame:
    """Encadena las cuatro transformaciones anteriores, en el mismo orden que
    el notebook (calendario -> festivos/puentes -> tipo_efectivo -> lag_24),
    valida los invariantes del resultado y deja el DataFrame con las
    columnas de `FEATURES_5F` listas para `src/modelo.py`."""
    df_features = anadir_calendario(df)
    df_features = anadir_festivos_puentes(df_features)
    df_features = mapear_tipo_efectivo(df_features)
    df_features = anadir_lag_24(df_features)
    validar_invariantes_5f(df_features)
    return df_features
