# src/datos.py
"""Descarga y ensamblado horario de demanda_real (indicador e·sios 1293).

Capa fina sobre `src/esios_client.py`: no reimplementa la sesión HTTP, los
reintentos ni el guardado en Parquet -- eso ya vive en `crear_sesion` y
`descargar_indicador`. Lo único nuevo aquí es:
  1. la resolución del token vía variable de entorno (hoy inline y repetida
     en las celdas de descarga_historico.ipynb y analisis_historico_2_demandas.ipynb),
  2. el resample a horario, que hoy vive suelto en
     analisis_historico_2_demandas.ipynb (celda 24).

Variable de entorno: `API_ESIOS`, no `ESIOS_TOKEN` -- decisión del 12/8: se
mantiene el nombre que ya usa el `.env` y `descarga_historico.ipynb` en vez
del nombre del pliego, para no tener dos convenciones distintas en el mismo
repo apuntando al mismo secreto.

Alcance deliberadamente reducido frente al resample de la celda 24: aquí solo
se agrega `demanda_real`. `demanda_prevista` (544) solo se usaba en el
notebook para el benchmark de REE, y `es_evento` es una anotación manual de
dos ventanas históricas conocidas (28-A y 11-J, ver `notebooks/
analisis_historico_2_demandas.ipynb` celda 13) que no tiene sentido
recalcular sobre una descarga diaria en vivo.
"""
import os

import pandas as pd
from dotenv import load_dotenv

from src.esios_client import crear_sesion, descargar_indicador

INDICADOR_DEMANDA_REAL = 1293


def obtener_token() -> str:
    """Carga `.env` si existe y devuelve el token de e·sios desde la variable
    de entorno `API_ESIOS`. Lanza si no está definida -- sin fallback
    silencioso a un token vacío."""
    load_dotenv()
    token = os.getenv("API_ESIOS")
    if not token:
        raise RuntimeError(
            "No se encuentra la variable de entorno API_ESIOS. Defínela en "
            ".env (local) o como GitHub Secret (pipeline)."
        )
    return token


def resample_horario_con_conteo(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega demanda_real de 5 minutos a horario (media) y añade
    `n_lecturas` (cuántas lecturas de 5 min entraron en cada hora), sin
    exigir que las 12 estén completas. Traslado literal del paso de resample
    de `analisis_historico_2_demandas.ipynb` (celda 24), reducido a
    `demanda_real` -- ver docstring del módulo.

    Deliberadamente sin assert de completitud: a diferencia de una descarga
    histórica cerrada, una descarga en vivo (`pipeline/predecir.py`) siempre
    tiene la última hora en curso incompleta -- no es un error, es la
    definición de "ahora mismo". `resample_horario` (más abajo) añade ese
    assert para cuando sí hace falta; `predecir.py` usa esta versión sin
    assert para decidir por sí mismo qué horas son fiables."""
    return (
        df.set_index("datetime_utc")
        .resample("1h")
        .agg(demanda_real=("demanda_real", "mean"), n_lecturas=("demanda_real", "count"))
        .reset_index()
    )


def resample_horario(df: pd.DataFrame) -> pd.DataFrame:
    """Como `resample_horario_con_conteo`, pero exige que cada hora tenga
    exactamente 12 lecturas de 5 min y devuelve solo `demanda_real` (sin
    `n_lecturas`).

    Dos invariantes que el notebook no necesitaba (allí el histórico ya
    estaba validado por `validar_rejilla` antes de llegar aquí) pero que este
    módulo sí, porque puede correr desatendido sobre una descarga fresca ya
    cerrada (p.ej. la Fase 1, o cualquier ventana que el llamador garantiza
    completa): "let it scream", no se promedia sobre un hueco.
      - `datetime_utc` debe seguir en UTC tras el resample.
    """
    df_horario = resample_horario_con_conteo(df)

    incompletas = df_horario.loc[df_horario["n_lecturas"] != 12, "datetime_utc"]
    assert incompletas.empty, (
        "Horas con un número de lecturas de 5 min distinto de 12 (descarga "
        f"incompleta): {incompletas.tolist()}"
    )
    df_horario = df_horario.drop(columns="n_lecturas")

    assert str(df_horario["datetime_utc"].dt.tz) == "UTC", (
        f"datetime_utc debe quedar en UTC tras el resample, no {df_horario['datetime_utc'].dt.tz}."
    )

    return df_horario


def descargar_demanda_cruda(inicio: pd.Timestamp, fin: pd.Timestamp) -> pd.DataFrame:
    """Descarga demanda_real (indicador 1293) entre `inicio` y `fin` (ventana
    semiabierta [inicio, fin), ambos tz-aware), sin resamplear. Es la
    primitiva que usa `pipeline/predecir.py` (necesita decidir hora a hora
    qué está completo antes de agregar) y sobre la que se apoya
    `descargar_demanda_horaria`."""
    token = obtener_token()
    sesion = crear_sesion(token)
    return descargar_indicador(sesion, INDICADOR_DEMANDA_REAL, inicio, fin, "demanda_real")


def descargar_demanda_horaria(inicio: pd.Timestamp, fin: pd.Timestamp) -> pd.DataFrame:
    """Descarga demanda_real y la agrega a horario, exigiendo que la ventana
    completa esté cerrada (ver `resample_horario`). No persiste nada a
    disco: eso es responsabilidad de quien llame."""
    return resample_horario(descargar_demanda_cruda(inicio, fin))
