# src/features_v2.py
"""Construcción de las 6 features de v2 (`modelo_lag168_v2`, sandbox ya
revisado). Traslado literal desde `sandbox/modelo_lag168_v2.ipynb`: celda 30
(`demanda_lag_168 = demanda_real.shift(168)`) y celda 31 (`features_6f`,
partición train/test, dropna del prefijo). Pliego: `PLIEGO_Fase5bis_v2_lag168.md`,
Parte 2.

v2 = v1 + `demanda_lag_168`. Nada más (pliego, 1.1): por eso las tres
transformaciones de calendario (`anadir_calendario`, `anadir_festivos_puentes`,
`mapear_tipo_efectivo`) y el propio `anadir_lag_24` se REUTILIZAN de
`src/features.py` sin copiarlos ni modificarlos -- ninguna de las cuatro
depende de qué lag adicional se use, y duplicarlas sería el segundo cambio
que el pliego prohíbe. `FEATURES_5F` no se toca ni se renombra: v2 declara su
propia lista, `FEATURES_6F`.

`es_puente` se queda en la lista aunque v1 nunca decida sobre ella (ver
README, "Los puentes: no hay datos suficientes"): retirarla sería otro
cambio a la vez, exactamente lo que el pliego prohíbe.
"""
from src.features import (
    anadir_calendario,
    anadir_festivos_puentes,
    anadir_lag_24,
    mapear_tipo_efectivo,
)

# Lista blanca de features de v2 (6 features). Copia literal de `features_6f`,
# sandbox/modelo_lag168_v2.ipynb celda 31.
FEATURES_6F = [
    "hora",
    "mes",
    "tipo_efectivo",
    "es_puente",
    "demanda_lag_24",
    "demanda_lag_168",
]

VARIABLE_OBJETIVO = "demanda_real"


def anadir_lag_168(df):
    """demanda_lag_168 = demanda_real de la misma hora, 7 días (168h) antes.
    Traslado literal de la celda 30 del sandbox. Se llama después de
    `anadir_lag_24`, que ya deja `df` ordenado por `datetime_utc` y con la
    continuidad horaria verificada -- no hace falta repetir esa comprobación
    aquí, `shift(168)` es posicional sobre la misma serie contigua."""
    df_features = df.copy()
    df_features["demanda_lag_168"] = df_features["demanda_real"].shift(168)
    return df_features


def validar_invariantes_6f(df) -> None:
    """Invariantes de las 6 features de v2, mismo criterio que
    `src.features.validar_invariantes_5f` (independiente del tamaño del
    lote), ampliado con `demanda_lag_168`. El horizonte de arranque cambia
    respecto a v1: `demanda_lag_168` deja NaN en las primeras 168 horas de la
    serie, no en las primeras 24 (pliego, 1.2)."""
    for col in ("hora", "mes", "tipo_efectivo", "es_puente"):
        n_nulos = int(df[col].isna().sum())
        assert n_nulos == 0, f"{n_nulos} NaN en '{col}' -- no debería poder pasar."

    categorias = set(df["tipo_efectivo"].unique())
    assert categorias <= {0, 1}, (
        f"tipo_efectivo tiene categorías fuera de {{0, 1}}: {categorias - {0, 1}}"
    )

    # demanda_lag_24: mismo criterio que v1 (prefijo de 24h).
    n_prefijo_24 = min(24, len(df))
    assert not df["demanda_lag_24"].iloc[n_prefijo_24:].isna().any(), (
        "demanda_lag_24 tiene NaN fuera del prefijo inicial esperado "
        f"(primeras {n_prefijo_24} filas): hueco interno o error de alineación."
    )

    # demanda_lag_168: prefijo de 168h, escrito a mano igual que el de 24h en
    # src/features.py -- no se deriva de ningún parámetro de configuración.
    n_prefijo_168 = min(168, len(df))
    assert not df["demanda_lag_168"].iloc[n_prefijo_168:].isna().any(), (
        "demanda_lag_168 tiene NaN fuera del prefijo inicial esperado "
        f"(primeras {n_prefijo_168} filas): hueco interno o error de alineación."
    )


def construir_features_6f(df):
    """Encadena las cinco transformaciones (calendario -> festivos/puentes ->
    tipo_efectivo -> lag_24 -> lag_168), en el mismo orden que
    `sandbox/modelo_lag168_v2.ipynb`, valida los invariantes del resultado y
    deja el DataFrame con las columnas de `FEATURES_6F` listas."""
    df_features = anadir_calendario(df)
    df_features = anadir_festivos_puentes(df_features)
    df_features = mapear_tipo_efectivo(df_features)
    df_features = anadir_lag_24(df_features)
    df_features = anadir_lag_168(df_features)
    validar_invariantes_6f(df_features)
    return df_features
