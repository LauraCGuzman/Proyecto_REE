# src/clima.py
"""Interpolación de huecos y agregación por zona climática de las
temperaturas AEMET. interpolar_temperaturas y media_ponderada son traslado
literal de API_aemet_datos_faltantes.ipynb (celdas 10 y 14) -- esa es la
versión canónica, la que produce el df_final_zonas_temperaturas real.

Nota: API_aemet.ipynb tiene sus propias versiones locales de ambas (celdas 12
y 22), sin tocar -- no estaba pedido y forzar su convergencia con las de aquí
sería más que un traslado literal:
- interpolar_temperaturas: implementación realmente distinta (coacciona a NaN
  en vez de exigir float estricto, y no lleva el conteo antes/después).
- media_ponderada: mismo cálculo, solo difiere en el nombre del primer
  parámetro (df_grupo) y en que no tiene tipos ni default para col_peso.
Ambas quedan anotadas para decidir aparte.
"""
import numpy as np
import pandas as pd

from src.quality import contar_nans_por_estacion


# Estación AEMET -> zona climática usada para ponderar la demanda eléctrica.
ZONAS = {
    "3129": "continental",
    "9434": "continental",
    "2539": "continental",
    "0076": "mediterraneo",
    "8414A": "mediterraneo",
    "6155A": "mediterraneo",
    "5783": "guadalquivir",
    "1082": "cantabrico",
}

# Pesos poblacionales derivados del Padrón Municipal del INE (Datos 2023).
# Representan la representatividad relativa de cada estación sobre la demanda
# eléctrica total.
PESOS = {
    "3129": 0.401,  # Madrid
    "9434": 0.081,  # Zaragoza
    "2539": 0.035,  # Valladolid
    "0076": 0.198,  # Barcelona
    "8414A": 0.097,  # Valencia
    "6155A": 0.069,  # Málaga
    "5783": 0.079,  # Sevilla
    "1082": 0.040,  # Bilbao
}


def interpolar_temperaturas(
    df: pd.DataFrame, limite_hueco: int = 3
) -> pd.DataFrame:
    """Ordena por estación y fecha, valida que tmin y tmax sean floats estrictos,

    e interpola con límite de días en huecos internos hacia adelante.
    """
    df_temp = df.copy()

    # 1. Orden cronológico estricto por estación
    df_temp["fecha"] = pd.to_datetime(df_temp["fecha"])
    df_temp = df_temp.sort_values(by=["indicativo", "fecha"]).reset_index(
        drop=True
    )

    columnas_temp = ["tmin", "tmax"]

    # 2. Validación de tipos explícita ("Let it scream")
    for col in columnas_temp:
        if col in df_temp.columns:
            if not pd.api.types.is_float_dtype(df_temp[col]):
                raise TypeError(
                    f"La columna '{col}' debe ser de tipo flotante. Tipo actual: {df_temp[col].dtype}"
                )

    # 3. Conteo de NaNs ANTES
    nans_antes = contar_nans_por_estacion(df_temp, columnas_temp)

    # 4. Interpolación lineal agrupada con dirección explícita
    for col in columnas_temp:
        if col in df_temp.columns:
            df_temp[col] = df_temp.groupby("indicativo")[col].transform(
                lambda x: x.interpolate(
                    method="linear",
                    limit=limite_hueco,
                    limit_direction="forward",
                    limit_area="inside",
                )
            )

    # 5. Conteo de NaNs DESPUÉS
    nans_despues = contar_nans_por_estacion(df_temp, columnas_temp)

    # 6. Auditoría y verificación de invariantes
    print("=== CONTROL DE IMPUTACIÓN DE NaNs ===")
    for col in columnas_temp:
        if col in df_temp.columns:
            total_antes = nans_antes[col].sum()
            total_despues = nans_despues[col].sum()
            print(f"\n[{col.upper()}] NaNs totales: {total_antes} → {total_despues}")

            estaciones_afectadas = (nans_antes[col] > 0) | (
                nans_despues[col] > 0
            )
            df_comparativa = pd.DataFrame(
                {
                    "Antes": nans_antes[col][estaciones_afectadas],
                    "Después": nans_despues[col][estaciones_afectadas],
                }
            )
            print(df_comparativa.to_string())

            if total_despues > total_antes:
                raise ValueError(
                    f"Error de consistencia: Los NaNs en {col} aumentaron tras la interpolación ({total_antes} → {total_despues})."
                )

    return df_temp


def media_ponderada(
    df_sub: pd.DataFrame, col_valor: str, col_peso: str = "peso"
) -> float:
    """Calcula la media ponderada sobre subgrupos filtrando NaNs tanto en valor como en peso."""
    mask = df_sub[col_valor].notna() & df_sub[col_peso].notna()
    if not mask.any():
        return np.nan
    return (df_sub.loc[mask, col_valor] * df_sub.loc[mask, col_peso]).sum() / (
        df_sub.loc[mask, col_peso].sum()
    )
