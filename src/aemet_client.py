# src/aemet_client.py
"""Descarga de series climatológicas diarias de AEMET OpenData para varias
estaciones, con reintentos, rate-limit y puerta de calidad sobre tmin/tmax.
Traslado literal desde API_aemet.ipynb (celda 8, la versión multi-estación).

Nota: la celda 7 original de ese notebook tiene una función de un solo paso
(extraer_historico_aemet, estación única) que quedó superada por la de aquí
y no se ha movido -- no estaba pedida esta sesión, y su notebook la sigue
usando de forma autocontenida como demo de la estación de Málaga.
"""
import time

import pandas as pd
import requests


def limpiar_y_castear_flotante(serie_columna):
    serie_limpia = serie_columna.str.replace(',', '.', regex=False)
    serie_float = pd.to_numeric(serie_limpia, errors='coerce')
    # NaN nuevos = corrupción (no eran NaN antes, se volvieron NaN al parsear)
    corruptos = serie_float.isna() & serie_limpia.notna()
    if corruptos.any():
        raise ValueError(f"Strings no parseables (corrupción): {corruptos.sum()} valores.")
    return serie_float


def extraer_historico_aemet_estacion(indicativo_estacion, fecha_inicio, fecha_fin, token):
    """
    Extrae los datos climatológicos mes a mes para una única estación
    dentro del rango de fechas especificado.
    """
    # Generamos los intervalos mensuales dinámicamente según las fechas de entrada
    inicios_mes = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="MS")
    finales_mes = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="ME")

    # Si la fecha de inicio no cae en día 1, la forzamos para el primer intervalo
    if inicios_mes.empty or inicios_mes[0] > pd.Timestamp(fecha_inicio):
        inicios_mes = inicios_mes.insert(0, pd.Timestamp(fecha_inicio).to_period('M').to_timestamp())
    if finales_mes.empty or finales_mes[-1] < pd.Timestamp(fecha_fin):
        finales_mes = finales_mes.append(pd.DatetimeIndex([pd.Timestamp(fecha_fin)]))

    intervalos = list(zip(inicios_mes, finales_mes))

    dataframes_validos = []
    meses_en_cuarentena = []

    headers = {'cache-control': "no-cache"}
    querystring = {"api_key": token}

    for inicio, fin in intervalos:
        mes_actual_str = inicio.strftime('%Y-%m')

        # Evitar pedir fechas futuras en base al momento actual de ejecución
        if inicio > pd.Timestamp.now():
            continue

        fecha_ini_str = inicio.strftime("%Y-%m-%dT00:00:00UTC")
        fecha_fin_str = fin.strftime("%Y-%m-%dT23:59:59UTC")

        url_paso1 = f"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{fecha_ini_str}/fechafin/{fecha_fin_str}/estacion/{indicativo_estacion}/"

        exito_mes = False
        intentos_totales = 0
        MAX_INTENTOS = 4

        while not exito_mes and intentos_totales < MAX_INTENTOS:
            intentos_totales += 1
            try:
                res_paso1 = requests.request("GET", url_paso1, headers=headers, params=querystring)
                json_paso1 = res_paso1.json()
            except Exception as e:
                print(f" -> [🚨 ERROR RED] Fallo de conexión en {mes_actual_str} (Intento {intentos_totales})")
                time.sleep(10)
                continue

            if json_paso1.get("estado") == 200:
                url_fresca = json_paso1.get("datos")
                try:
                    res_paso2 = requests.get(url_fresca)
                    if res_paso2.status_code != 200:
                        time.sleep(10)
                        continue
                    datos_mes = res_paso2.json()
                except Exception:
                    time.sleep(10)
                    continue

                try:
                    df_mes = pd.DataFrame(datos_mes)

                    if df_mes.empty:
                        raise ValueError("El servidor devolvió una estructura vacía.")

                    # Puerta de calidad y casteo
                    df_mes['tmed'] = limpiar_y_castear_flotante(df_mes['tmed'])
                    df_mes['tmax'] = limpiar_y_castear_flotante(df_mes['tmax'])
                    df_mes['tmin'] = limpiar_y_castear_flotante(df_mes['tmin'])
                    df_mes['fecha'] = pd.to_datetime(df_mes['fecha'], errors='raise')

                    dataframes_validos.append(df_mes)
                    print(f" -> [OK] Estación {indicativo_estacion} | {mes_actual_str} extraído ({len(df_mes)} días).")
                    exito_mes = True

                except Exception as e:
                    motivo_error = f"Error en capa de datos/parseo: {str(e)}"
                    print(f" -> [🚨 CUARENTENA] Estación {indicativo_estacion} | {mes_actual_str} aislado. Motivo: {e}")
                    meses_en_cuarentena.append({"estacion": indicativo_estacion, "mes": mes_actual_str, "error": motivo_error})
                    exito_mes = True

            elif json_paso1.get("estado") == 429:
                print(f" -> [💥 RATE LIMIT] 429 en {mes_actual_str}. Durmiendo 60 segundos...")
                time.sleep(60)
            else:
                motivo_error = f"Error API Paso 1 ({json_paso1.get('estado')}): {json_paso1.get('descripcion')}"
                print(f" -> [🚨 CUARENTENA] Estación {indicativo_estacion} | {mes_actual_str} aislado. Motivo: {json_paso1.get('descripcion')}")
                meses_en_cuarentena.append({"estacion": indicativo_estacion, "mes": mes_actual_str, "error": motivo_error})
                exito_mes = True

        if not exito_mes:
            meses_en_cuarentena.append({"estacion": indicativo_estacion, "mes": mes_actual_str, "error": "Agotados intentos de red."})

        time.sleep(2)

    return dataframes_validos, meses_en_cuarentena


def extraer_historico_multi_estacion(indicativos, fecha_inicio, fecha_fin, token):
    """
    Función principal que procesa una o varias estaciones climatológicas.

    :param indicativos: String único (ej. "6172X") o lista de strings (ej. ["6172X", "3195"])
    :param fecha_inicio: String de fecha inicial compatible con pandas (ej. "2023-01-01")
    :param fecha_fin: String de fecha final (ej. "2026-07-31")
    :param token: Tu clave de API personal de AEMET OpenData
    :return: (df_historico_completo, cuarentena_global)
    """
    # Si nos pasan un solo indicativo como string, lo convertimos a lista
    if isinstance(indicativos, str):
        indicativos = [indicativos]

    todos_los_dataframes = []
    cuarentena_global = []

    print(f"🚀 INICIANDO EXTRACCIÓN HISTÓRICA MULTI-ESTACIÓN ({fecha_inicio} a {fecha_fin})")
    print(f"Estaciones a procesar: {indicativos}")
    print("=" * 70)

    for estacion in indicativos:
        print(f"\n📡 Procesando estación: {estacion}")
        print("-" * 50)
        dfs_estacion, cuarentena_estacion = extraer_historico_aemet_estacion(
            indicativo_estacion=estacion,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            token=token
        )

        if dfs_estacion:
            todos_los_dataframes.extend(dfs_estacion)
        if cuarentena_estacion:
            cuarentena_global.extend(cuarentena_estacion)

    # Ensamblado de resultados
    df_historico_completo = pd.DataFrame()

    print("\n" + "=" * 70)
    print("🏁 FIN DEL PROCESAMIENTO GLOBAL")
    print("=" * 70)

    if todos_los_dataframes:
        df_historico_completo = pd.concat(todos_los_dataframes, ignore_index=True)
        print(f"\n✅ Extracción terminada con éxito.")
        print(f"   • Registros totales salvados: {len(df_historico_completo)} días.")
        print(f"   • Estaciones rescatadas: {df_historico_completo['indicativo'].unique()}")
        print(f"❌ Bloques mensuales enviados a cuarentena: {len(cuarentena_global)}")

        print("\n📊 RESUMEN ESTADÍSTICO POR ESTACIÓN (DE LAS TRES TEMPERATURAS):")
        # Agrupamos por indicativo para que el análisis estadístico diferencie cada estación
        print(df_historico_completo.groupby('indicativo')[['tmed', 'tmax', 'tmin']].describe().T)
    else:
        print("\n🚨 Error crítico: No se ha podido salvar ningún dato del periodo para ninguna estación.")

    if cuarentena_global:
        print("\n📋 REGISTRO DETALLADO DE LA CUARENTENA:")
        for item in cuarentena_global:
            print(f"  • Estación {item['estacion']} | {item['mes']}: {item['error']}")
    else:
        print("\n🎉 ¡Milagro! Ningún mes de ninguna estación requirió cuarentena.")

    return df_historico_completo, cuarentena_global
