# src/esios_client.py
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
import os
import time
from src.quality import validar_rejilla
from pathlib import Path


BASE_URL = "https://api.esios.ree.es/indicators"

def crear_sesion(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "x-api-key": token,
        "Accept": "application/json; application/vnd.esios-api-v2+json",
        "Content-Type": "application/json",
    })
    # FALLO C: reintentos con backoff en transitorios; sin esto el batch cae en el primer 429
    retry = Retry(total=5, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s



def descargar_indicador(sesion: requests.Session, indicador_id: int,

                       inicio: str, fin: str, nombre_col: str) -> pd.DataFrame:

    """Fetch + normaliza. LANZA (raise_for_status). No valida conteo, no guarda."""


    url = f"https://api.esios.ree.es/indicators/{indicador_id}"

    params = {"start_date": inicio, "end_date": fin}


    # SOLUCIÓN ROBUSTA: Timeout como tupla (connect, read)
    # 5s para establecer conexión (falla rápido si la API está caída)
    # 60s para leer datos (da margen si el volumen de filas ralentiza la respuesta)

    response = sesion.get(url, params=params, timeout=(5, 60))

    response.raise_for_status()

   
    # Nota de diseño: Si cambia el esquema del JSON, el KeyError resultante
    # se captura y enriquece con contexto en la capa del runner superior.

    datos_json = response.json()['indicator']['values']
    df = pd.DataFrame(datos_json)

    # Gestión del caso vacío requerida por el gate

    if df.empty:

        return pd.DataFrame(columns=['datetime_utc', nombre_col])


    # Normalización estricta a UTC

    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
    df = df.rename(columns={'value': nombre_col})
    df = df.sort_values(by='datetime_utc').reset_index(drop=True)
    df = df[df['datetime_utc'] < fin.tz_convert('UTC')]

    return df[['datetime_utc', nombre_col]]


def descargar_rango(sesion: requests.Session, indicador_id: int, 
                 fecha_inicio: str, fecha_fin: str, 
                 nombre_col: str) -> list:
    """
    Modula el rango de fechas en bloques mensuales completos, descarga desde la API,
    valida la estructura de 5 minutos mediante el gate de calidad y persiste en Parquet.
    
    Exige estrictamente que fecha_inicio y fecha_fin coincidan con el inicio de un mes.
    La ventana de descarga por tramo es semiabierta [inicio, fin).
    """
    RAIZ = Path(__file__).resolve().parents[1]
    ruta_carpeta = RAIZ / "data" / "raw"
    os.makedirs(ruta_carpeta, exist_ok=True)
    
    # Asegurar Timestamps localizados en UTC
    ts_inicio = pd.Timestamp(fecha_inicio, tz='UTC')
    ts_fin = pd.Timestamp(fecha_fin, tz='UTC')
    
    # "Let it scream": Evitar ventanas parciales silenciosas que corrompan el histórico
    assert ts_inicio.is_month_start and ts_fin.is_month_start, \
        "Los bordes de fecha_inicio y fecha_fin deben ser primeros de mes (meses completos)."
    
    # Generar los cortes de control (todos serán primeros de mes)
    cortes_meses = pd.date_range(start=ts_inicio, end=ts_fin, freq='MS', tz='UTC')
    
    resumen_descargas = []

    # Emparejamiento limpio de ventanas mensuales [inicio, fin) sin lógica de guardia compleja
    for mes_actual_inicio, mes_actual_fin in zip(cortes_meses[:-1], cortes_meses[1:]):
        anio = mes_actual_inicio.year
        mes = mes_actual_inicio.month
        
        ruta_fichero = f"{ruta_carpeta}/{indicador_id}_{anio}-{mes:02d}.parquet"
        filas_esperadas = int((mes_actual_fin - mes_actual_inicio) / pd.Timedelta(minutes=5))
        
        resultado = {
            "indicador": indicador_id,
            "mes": f"{anio}-{mes:02d}",
            "filas_obtenidas": 0,
            "filas_esperadas": filas_esperadas,
            "estado": "PENDIENTE",
            "detalle": ""
        }
        
        # 1. Idempotencia defensiva
        if os.path.exists(ruta_fichero):
            try:
                df_existente = pd.read_parquet(ruta_fichero)
                # Validar si el fichero existente cumple con el gate estructural
                validar_rejilla(df_existente, mes_actual_inicio, mes_actual_fin)
                
                resultado["filas_obtenidas"] = len(df_existente)
                resultado["estado"] = "EXISTENTE (SKIPPED)"
                resumen_descargas.append(resultado)
                continue
            except Exception as e:
                resultado["estado"] = "EXISTENTE_CORRUPTO"
                resultado["detalle"] = f"Archivo local inválido: {e}. Se reintenta descarga."
        
        # 2. Descarga y Validación
        try:
            # Consistencia de tipos: Contrato cerrado usando Timestamps UTC-aware
            df = descargar_indicador(
                sesion=sesion, 
                indicador_id=indicador_id, 
                inicio=mes_actual_inicio, 
                fin=mes_actual_fin, 
                nombre_col=nombre_col
            )
            
            # 3. Validar los datos con la función estricta de quality.py
            validar_rejilla(df, mes_actual_inicio, mes_actual_fin)
            
            # 4. Guardar si pasó el Gate
            df.to_parquet(ruta_fichero, index=False)
            resultado["filas_obtenidas"] = len(df)
            resultado["estado"] = "OK"
            resultado["detalle"] = "Descargado y verificado correctamente."
            
        except requests.exceptions.HTTPError as e:
            resultado["estado"] = f"ERROR_API_{e.response.status_code if e.response else 'REQ'}"
            resultado["detalle"] = f"Error al descargar de ESIOS: {e}"
        except AssertionError as e:
            resultado["estado"] = "DESCUADRE_CONTEO_REJILLA"
            resultado["detalle"] = f"Fallo en el Gate de calidad estructural: {e}"
            if 'df' in locals() and df is not None:
                resultado["filas_obtenidas"] = len(df)
        except Exception as e:
            resultado["estado"] = "ERROR_SISTEMA_DISCO"
            resultado["detalle"] = f"Error inesperado: {str(e)}"
            
        resumen_descargas.append(resultado)
        
        # Control de cortesía para no provocar 429 de forma masiva en bucles largos
        time.sleep(1)
        
    return resumen_descargas