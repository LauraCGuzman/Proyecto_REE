# src/quality.py
"""Gate estructural de ingesta + triaje semántico. Son dos preguntas distintas:
   1) ¿el mes tiene las filas que debería?  -> validar_rejilla (estructura)
   2) ¿los valores son plausibles?          -> triaje_deltas (semántica)
El 28-A pasa (1) si la telemetría siguió emitiendo, y en (2) es EVENTO, no CORRUPCION."""
import pandas as pd


def validar_rejilla(df: pd.DataFrame, inicio: pd.Timestamp, fin: pd.Timestamp):
    """
    Gate de validación estricta para el DataFrame de ESIOS.
    Comprueba anclaje de inicio, regularidad de 5 min y longitud exacta.
    Lanza AssertionError si alguna condición no se cumple.
    """
    # 0. El DataFrame no puede estar vacío si se esperan datos
    assert not df.empty, "El DataFrame está completamente vacío."
    
    # Aseguramos que las entradas de control estén en UTC para comparar peras con peras
    inicio_utc = inicio.tz_convert('UTC')
    fin_utc = fin.tz_convert('UTC')
    
    # 1. CHECK ANCLAJE: Evita que la rejilla esté desplazada (el hueco que detectaste)
    assert df['datetime_utc'].iloc[0] == inicio_utc, \
        f"Error de anclaje: El primer registro ({df['datetime_utc'].iloc[0]}) no coincide con el inicio pedido ({inicio_utc})."
        
    # 2. CHECK DELTA: CAZA duplicados (delta 0) y huecos internos (delta > 5min)
    # Usamos .dropna() para ignorar el NaT de la primera fila que genera .diff()
    deltas = df['datetime_utc'].diff().dropna()
    assert deltas.eq(pd.Timedelta(minutes=5)).all(), \
        "Error de regularidad: Se detectaron saltos de tiempo incorrectos o filas duplicadas."
        
    # 3. CHECK LONGITUD: Ancla el span y asegura que no falten filas al inicio/final
    #Ventana semi abierto [inicio, final)
    filas_esperadas = int((fin_utc - inicio_utc) / pd.Timedelta(minutes=5)) 
    
    assert len(df) == filas_esperadas, \
        f"Error de longitud: Se esperaban {filas_esperadas} filas, pero se obtuvieron {len(df)}."
    
def triaje_deltas(df, col_valor="demanda_real", umbral_mw=2000):
    """
    Identifica anomalías en los deltas de una serie temporal basándose en un umbral
    y en valores de demanda imposibles (menores o iguales a cero).
    
    Exige que los datos vengan pre-ordenados cronológicamente.
    """
    # 1. Evitar efectos secundarios mutando el DataFrame original
    df = df.copy()

    # 2. Contrato: Exigir que el DataFrame ya venga ordenado
    assert df["datetime_utc"].is_monotonic_increasing, "El DataFrame no viene ordenado por datetime_utc."

    # 3. Calcular el delta con la columna parametrizada
    df['delta'] = df[col_valor].diff()

    # 4. Obtener el valor absoluto de los deltas
    deltas_absolutos = df['delta'].abs()

    # 5. Filtrar por umbral o por valores imposibles (<= 0)
    filtro = (deltas_absolutos > umbral_mw) | (df[col_valor] <= 0)

    return df[filtro]


def contar_nans_por_estacion(
    df: pd.DataFrame, columnas: list[str]
) -> dict[str, pd.Series]:
    """Suma los NaNs de cada columna agrupados por la estación ('indicativo')."""
    return {
        col: df[col].isna().groupby(df["indicativo"]).sum()
        for col in columnas
        if col in df.columns
    }


def obtener_longitud_maxima_racha(serie_bool: pd.Series) -> int:
    """Devuelve la racha más larga de True consecutivos en una serie booleana."""
    if not serie_bool.any():
        return 0
    # Creamos bloques de grupos consecutivos cada vez que cambia el valor booleano
    bloques = (serie_bool != serie_bool.shift()).cumsum()
    # Filtramos solo los bloques correspondientes a True (NaNs) y medimos su longitud
    return serie_bool[serie_bool].groupby(bloques).size().max()