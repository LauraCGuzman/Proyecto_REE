"""Panel de monitorización del pipeline REE en producción (Fase 6, pliego
`PLIEGO_Fase6_streamlit.md`, PR 2; gráficos interactivos, PR
`fase6-graficos-interactivos`).

Lector objetivo: no conoce el proyecto. En treinta segundos debe entender
qué se predice, cómo va y qué falla. No es un cuadro de mando de negocio:
es la prueba de que el modelo está vivo, medido y con sus límites
conocidos. Por eso el texto es corto y sin jerga interna: sin nombres de
notebook, de celda ni de ruta de fichero -- eso es detalle de
implementación, no algo que el lector objetivo necesite.

Panel de SOLO LECTURA (pliego §0, regla 2): no toca `pipeline/`, `data/`,
`reports/`, `modelos/` ni el YAML. Lee `data/errores.csv`,
`data/predicciones.csv`, `data/metricas.json` y `reports/estado_pipeline.md`
-- nada más.

Regla 1 del pliego (la ventana abierta, 22/8-2/10): de v2 se muestra que
existe y que emite -- nunca sus métricas (MAE, sesgo, serie de error, tabla
de ventanas), y nunca junto a v1 en la misma vista. La única excepción,
decidida por Laura el 22/8, es la curva real vs predicha: accesible para
los dos modelos por separado, con un selector que nunca los enfrenta --
excluyente, sin opción "ambos", sin dos curvas en pantalla a la vez.

Gráficos en Plotly (hover + zoom), no matplotlib: `scripts/grafico_deriva.py`
NO se toca en este PR -- su PNG (commiteado a mano en `reports/`, Fase 4)
sigue sirviendo tal cual para el repo. El panel construye su propia versión
interactiva de la deriva a partir de los MISMOS datos ya calculados por ese
módulo (`serie_diaria_v1`, `_serie_con_huecos_explicitos`,
`dias_no_laborable_a_laborable`, `MAE_TEST_NOTEBOOK_MW`) -- lo que cambia es
solo cómo se dibuja, no cómo se calcula.

Vive en `panel/`, no en `app/`: `app/` está en `.gitignore` como carpeta de
trabajo local (Laura, 9/8) -- no se toca ni se le quitan excepciones. Este
panel necesita estar versionado para poder desplegarse desde el repo, así
que va en una carpeta propia.

Uso local:
    streamlit run panel/panel.py

Despliegue: Streamlit Community Cloud, fichero principal `panel/panel.py`,
dependencias en `panel/requirements.txt` (deliberadamente separado del
`requirements.txt` de la raíz -- ver ese fichero, "sin scikit-learn").
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.paths import DIR_DATA, DIR_REPORTS  # noqa: E402

# Reutilizado de scripts/grafico_deriva.py (pliego §2.1 punto 4) -- ver la
# nota larga más abajo, junto a cada función, sobre qué se reutiliza tal
# cual y qué NO se reutiliza y por qué. `construir_grafico` (la versión
# matplotlib de dos paneles) YA NO se importa: el panel dibuja su propia
# versión Plotly con los mismos datos (ver `construir_grafico_deriva`).
from scripts.grafico_deriva import (  # noqa: E402
    COBERTURA_MINIMA_DIAS,
    MAE_TEST_NOTEBOOK_MW,
    NOMBRE_MODELO as V1,
    NOMBRE_MODELO_V2 as V2,
    TZ_MADRID,
    _leer_csv,
    _serie_con_huecos_explicitos,
    _texto_cobertura_v2,
    cobertura_v2,
    dias_no_laborable_a_laborable,
    serie_diaria_v1,
)

RUTA_ERRORES = DIR_DATA / "errores.csv"
RUTA_METRICAS = DIR_DATA / "metricas.json"
RUTA_PREDICCIONES = DIR_DATA / "predicciones.csv"
RUTA_ESTADO = DIR_REPORTS / "estado_pipeline.md"

MODELOS = (V1, V2)

# Fin de la ventana de prueba de v2 (pliego §0, regla 1): antes de esta
# fecha, ninguna métrica de v2 se muestra, sin excepción.
FECHA_FIN_PRUEBA_V2 = date(2026, 10, 2)

# TTL de la caché: "los datos cambian una vez al día" (pliego §2.7) --
# una hora es sobrado para no releer los ficheros en cada interacción del
# usuario dentro de una misma sesión, y corto de sobra para no quedarse con
# datos de ayer si el contenedor de Streamlit Cloud lleva horas despierto.
TTL_CACHE_SEGUNDOS = 3600

COLOR_REAL = "#2ca02c"
COLOR_PREDICHO = "#ff7f0e"
COLOR_MAE = "#1f77b4"
COLOR_SESGO = "#d62728"
COLOR_TRANSICION = "rgba(218, 165, 32, 0.20)"


# --------------------------------------------------------------------- #
# Carga de datos -- cada fuente comprobada por separado (pliego §2.6:
# "sin try/except amplios; comprobar antes lo que se pueda comprobar").
# --------------------------------------------------------------------- #


@st.cache_data(ttl=TTL_CACHE_SEGUNDOS, show_spinner=False)
def cargar_errores() -> pd.DataFrame | None:
    if not RUTA_ERRORES.exists():
        return None
    df = _leer_csv(RUTA_ERRORES)
    return df if not df.empty else None


@st.cache_data(ttl=TTL_CACHE_SEGUNDOS, show_spinner=False)
def cargar_predicciones() -> pd.DataFrame | None:
    if not RUTA_PREDICCIONES.exists():
        return None
    df = _leer_csv(RUTA_PREDICCIONES)
    return df if not df.empty else None


@st.cache_data(ttl=TTL_CACHE_SEGUNDOS, show_spinner=False)
def cargar_metricas() -> tuple[dict | None, str | None]:
    """A diferencia de `scripts.grafico_deriva._leer_metricas` (que exige
    los dos bloques v1 Y v2 y para en rojo si falta uno -- correcto para un
    script de un solo modelo), el panel exige solo v1: la ausencia del
    bloque de v2 es "el estado real hasta hoy" (pliego §2.6, primera bala),
    no un error. Por eso el panel NO reutiliza `_leer_metricas` y valida a
    mano."""
    if not RUTA_METRICAS.exists():
        return None, "no hay datos de estado todavía."
    bruto = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))
    if V1 not in bruto:
        return None, "no hay datos de v1 todavía."
    return bruto, None


@st.cache_data(ttl=TTL_CACHE_SEGUNDOS, show_spinner=False)
def cargar_estado_md() -> str | None:
    if not RUTA_ESTADO.exists():
        return None
    texto = RUTA_ESTADO.read_text(encoding="utf-8")
    return texto if texto.strip() else None


def extraer_fechas_presentes(estado_md: str, modelo: str) -> int | None:
    """El mismo conteo que ya calcula y escribe el pipeline (fechas de
    calendario distintas con al menos una hora, publicadas + diagnóstico)
    -- leído del texto generado, no recalculado aquí (pliego §2.9: "NO
    recalcular métricas por tu cuenta"). Solo se usa el número: el texto
    que lo rodea (rutas de fichero incluidas) no llega a la pantalla."""
    marcador_modelo = f"## Modelo: {modelo}"
    if marcador_modelo not in estado_md:
        return None
    resto = estado_md.split(marcador_modelo, 1)[1].split("\n---", 1)[0]
    m = re.search(r"Fechas presentes en `data/errores\.csv`[^:]*: (\d+)", resto)
    return int(m.group(1)) if m else None


def texto_v2_seguro(metricas: dict) -> str:
    """Envoltorio de `cobertura_v2`/`_texto_cobertura_v2` (reutilizadas tal
    cual): esas dos funciones asumen que la clave "v2" existe en
    `metricas`; el panel puede recibir un `metricas.json` sin ese bloque
    (pliego §2.6) y no debe reventar por ello."""
    if V2 not in metricas:
        return "v2: sin datos todavía."
    return _texto_cobertura_v2(cobertura_v2(metricas))


# --------------------------------------------------------------------- #
# Curva real vs predicha (pliego §2.3) -- granularidad horaria y
# multi-modelo, deliberadamente NO construida reutilizando o generalizando
# `serie_diaria_v1` (agregación diaria, fija a v1, pensada para el gráfico
# de deriva). Generalizarla a un módulo común es justo el tipo de refactor
# que el pliego pide anotar antes de hacer ("si hay que refactorizar algo a
# un módulo común, decirlo antes de hacerlo") -- decisión: NO se hace en
# este PR. `scripts/grafico_deriva.py` se queda intacto; lo único que
# comparten esta función y `serie_diaria_v1` es un filtro de una línea
# (`modelo == X` y `h_adelanto_h > 0`), que no es la lógica que el pliego
# pide no duplicar (agregación diaria, huecos, media de moda, referencia
# histórica) -- eso sigue viviendo solo en `grafico_deriva.py`.
# --------------------------------------------------------------------- #


def horas_publicadas(errores: pd.DataFrame, modelo: str) -> pd.DataFrame:
    """Horas publicadas (`h_adelanto_h > 0`) de un modelo, con la fecha de
    calendario en Madrid añadida -- real y predicho, sin agregar. Fuente
    única: `data/errores.csv` (pliego §2.3: "las de diagnóstico no entran:
    no eran previsión")."""
    publicado = errores[
        (errores["modelo"] == modelo) & (errores["h_adelanto_h"] > 0)
    ].copy()
    if publicado.empty:
        return publicado
    publicado["horizonte_madrid"] = pd.to_datetime(
        publicado["horizonte"], utc=True
    ).dt.tz_convert(TZ_MADRID)
    publicado["fecha"] = publicado["horizonte_madrid"].dt.date
    return publicado.sort_values("horizonte_madrid")


def dia_con_mayor_mae(publicado: pd.DataFrame) -> date | None:
    """Día por defecto del selector (pliego §2.3: "el día con mayor MAE de
    los disponibles"). Un solo `groupby` sobre la misma tabla horaria que ya
    se pinta -- no una segunda implementación de la agregación diaria de
    `serie_diaria_v1` (esa vive en un solo sitio, ver nota de arriba)."""
    if publicado.empty:
        return None
    mae_por_dia = publicado.groupby("fecha")["error"].apply(lambda s: s.abs().mean())
    return mae_por_dia.idxmax()


def techo_modelo(predicciones: pd.DataFrame | None, modelo: str) -> float | None:
    """Techo del modelo seleccionado: `max(valor_predicho)` sobre
    `predicciones.csv` filtrado a ese modelo, calculado en vivo, nunca
    escrito a mano (pliego §2.3). Cada modelo tiene el suyo -- árboles
    distintos, hojas distintas."""
    if predicciones is None:
        return None
    sub = predicciones[predicciones["modelo"] == modelo]
    return float(sub["valor_predicho"].max()) if not sub.empty else None


def construir_grafico_curva(dia_df: pd.DataFrame, modelo: str, techo: float | None) -> go.Figure:
    """Curva real vs predicha de un solo día, un solo modelo (pliego §2.3:
    "un modelo en pantalla; ninguna superposición"). Plotly: hover con el
    valor exacto de cada hora y zoom, que un PNG estático no da."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dia_df["horizonte_madrid"],
            y=dia_df["valor_real"],
            mode="lines+markers",
            name="Real",
            line=dict(color=COLOR_REAL, width=2),
            hovertemplate="%{x|%H:%M} -- Real: %{y:,.0f} MW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dia_df["horizonte_madrid"],
            y=dia_df["valor_predicho"],
            mode="lines+markers",
            name="Predicho",
            line=dict(color=COLOR_PREDICHO, width=2),
            hovertemplate="%{x|%H:%M} -- Predicho: %{y:,.0f} MW<extra></extra>",
        )
    )
    if techo is not None:
        fig.add_hline(
            y=techo,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"{techo:,.1f} MW -- máximo predicho observado de {modelo} (mínimo del techo real)",
            annotation_position="top left",
            annotation_font_size=10,
        )
    fig.update_layout(
        title=f"Curva real vs predicha -- {modelo}, {dia_df['fecha'].iloc[0]}",
        xaxis_title="Hora (Madrid)",
        yaxis_title="Demanda (MW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80),
    )
    return fig


# --------------------------------------------------------------------- #
# Deriva (pliego §2.4) -- misma serie que `scripts/grafico_deriva.py`
# (reutilizada tal cual: `serie_diaria_v1`, `_serie_con_huecos_explicitos`,
# `dias_no_laborable_a_laborable`, `MAE_TEST_NOTEBOOK_MW`), dibujada en
# Plotly en vez de matplotlib porque `construir_grafico` de ese módulo
# escribe el PNG que sigue viviendo en `reports/` -- no se toca, no se
# reutiliza su función de dibujo, solo sus datos.
# --------------------------------------------------------------------- #


def construir_grafico_deriva(serie: pd.DataFrame, dias_marcados: set) -> go.Figure:
    """Dos paneles apilados, eje de tiempo compartido: MAE arriba (con la
    referencia histórica como línea horizontal etiquetada -- nunca como
    umbral), sesgo abajo (con el cero marcado, el signo es la información).
    Días de transición no_laborable -> laborable resaltados en los dos
    paneles. Días con `n` por debajo de la moda, anotados con su `n`."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("MAE diario -- v1 (solo horas publicadas)", "Sesgo diario -- v1"),
    )

    x = pd.to_datetime(serie["fecha"])

    for dia in sorted(dias_marcados):
        dia_ts = pd.Timestamp(dia)
        if x.min() <= dia_ts <= x.max():
            fig.add_vrect(
                x0=dia_ts - pd.Timedelta(hours=12),
                x1=dia_ts + pd.Timedelta(hours=12),
                fillcolor=COLOR_TRANSICION,
                line_width=0,
                row="all",
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=serie["mae"],
            mode="lines+markers",
            name="MAE diario",
            line=dict(color=COLOR_MAE, width=2),
            hovertemplate="%{x|%d/%m} -- MAE: %{y:,.0f} MW<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=MAE_TEST_NOTEBOOK_MW,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"{MAE_TEST_NOTEBOOK_MW:,.2f} MW -- referencia histórica, no es meta de producción",
        annotation_position="top left",
        annotation_font_size=10,
        row=1,
        col=1,
    )

    n_moda = serie["n"].mode()
    n_moda = n_moda.iloc[0] if not n_moda.empty else None
    for _, fila in serie.iterrows():
        if pd.isna(fila["n"]):
            continue
        if n_moda is not None and fila["n"] != n_moda:
            fig.add_annotation(
                x=pd.Timestamp(fila["fecha"]),
                y=fila["mae"],
                text=f"n={int(fila['n'])}",
                showarrow=True,
                arrowhead=0,
                arrowcolor="dimgray",
                ax=25,
                ay=25,
                font=dict(size=10, color="dimgray"),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=serie["sesgo"],
            mode="lines+markers",
            name="Sesgo diario",
            line=dict(color=COLOR_SESGO, width=2),
            hovertemplate="%{x|%d/%m} -- Sesgo: %{y:,.0f} MW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)

    fig.update_yaxes(title_text="MAE diario (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Sesgo diario (MW)<br>+ infrapredicción / − sobrepredicción", row=2, col=1)
    fig.update_xaxes(title_text="Fecha del horizonte (Madrid)", row=2, col=1)
    fig.update_layout(showlegend=False, hovermode="x unified", height=650)
    return fig


# --------------------------------------------------------------------- #
# Página -- envuelta en main() y bajo `if __name__ == "__main__"` (patrón
# estándar de Streamlit, ver docs de apps multipágina) para poder importar
# las funciones puras de arriba desde tests/test_panel.py sin ejecutar la
# UI: `streamlit run panel/panel.py` sí ejecuta el módulo como `__main__`,
# un `import panel.panel` desde un test no.
# --------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title="REE -- panel de producción", layout="wide")
    st.title("Panel de producción -- demanda eléctrica REE")

    _pagina()


def _pagina() -> None:
    errores = cargar_errores()
    predicciones = cargar_predicciones()
    metricas, error_metricas = cargar_metricas()
    estado_md = cargar_estado_md()

    # ----- Bloque de estado (pliego §2.2) ------------------------------ #
    st.header("Estado del pipeline")

    if metricas is None:
        st.error(f"No se puede mostrar el estado: {error_metricas}")
    else:
        bloque_v1 = metricas[V1]
        fechas_v1 = extraer_fechas_presentes(estado_md, V1) if estado_md else None

        col1, col2, col3 = st.columns(3)
        col1.metric("Última corrida (UTC)", bloque_v1["actualizado_utc"])
        col2.metric(
            "Fechas con datos (v1)",
            fechas_v1 if fechas_v1 is not None else "—",
            help="Fechas de calendario distintas con al menos una hora "
            "evaluada (publicadas + diagnóstico).",
        )
        col3.metric(
            "Muestra suficiente (v1)",
            "no" if bloque_v1["muestra_insuficiente"] else "sí",
        )

        st.subheader("MAE y sesgo de producción -- v1, solo horas publicadas")
        filas = []
        for ventana in ("7d", "30d"):
            v = bloque_v1["publicado"]["ventanas"][ventana]
            mae = f"{v['mae']:.2f}" if v["mae"] is not None else "—"
            sesgo = f"{v['sesgo_medio']:+.2f}" if v["sesgo_medio"] is not None else "—"
            filas.append(
                {
                    "Ventana": ventana,
                    "MAE (MW)": mae,
                    "Sesgo medio (MW)": sesgo,
                    "n horas": v["n_horas"],
                    "Fechas cubiertas": v["dias_cubiertos"],
                }
            )
        st.table(pd.DataFrame(filas).set_index("Ventana"))

        # Criterio 6bis: el encuadre, en pantalla y no en un desplegable --
        # recortado a una sola línea (pliego, PR de gráficos interactivos):
        # el concepto ("no comparable"), no el origen del número.
        referencia = bloque_v1.get("referencia", {})
        mae_referencia = referencia.get("mae_test_notebook")
        if mae_referencia is not None:
            st.caption(
                f"El MAE de producción no es comparable con los "
                f"{mae_referencia:,.2f} MW de la medición de referencia: "
                "limitaciones conocidas del modelo, no fallos del pipeline."
            )
        else:
            st.caption("Sin cifra de referencia todavía.")

    st.divider()

    # ----- Curva real vs predicha (pliego §2.3) ------------------------ #
    st.header("Curva real vs predicha")

    modelo_elegido = st.selectbox("Modelo", MODELOS, index=0)

    if modelo_elegido == V2:
        st.warning(
            "**v2 está en periodo de prueba hasta el 2 de octubre.** Sus "
            "métricas no se muestran hasta entonces: los criterios de "
            "comparación se fijaron por escrito antes de empezar y se evalúan "
            "una sola vez, al cierre."
        )

    if errores is None:
        st.info("Sin datos de errores todavía: no hay curva que mostrar.")
    else:
        publicado = horas_publicadas(errores, modelo_elegido)
        if publicado.empty:
            st.info(
                f"{modelo_elegido} no tiene todavía ninguna hora publicada "
                "evaluada. Su primera curva aparecerá en cuanto la tenga."
            )
        else:
            dias_disponibles = sorted(publicado["fecha"].unique())
            dia_defecto = dia_con_mayor_mae(publicado)
            indice_defecto = (
                dias_disponibles.index(dia_defecto) if dia_defecto in dias_disponibles else 0
            )
            dia_elegido = st.selectbox(
                "Día",
                dias_disponibles,
                index=indice_defecto,
                help="Por defecto, el día con mayor MAE de los disponibles.",
            )

            dia_df = publicado[publicado["fecha"] == dia_elegido]
            techo = techo_modelo(predicciones, modelo_elegido)
            fig_curva = construir_grafico_curva(dia_df, modelo_elegido, techo)
            st.plotly_chart(fig_curva, use_container_width=True)

            transicion = dias_no_laborable_a_laborable(dia_elegido, dia_elegido)
            if dia_elegido in transicion:
                st.caption(
                    f"{dia_elegido} es un día de transición no laborable → "
                    "laborable: el modelo no condiciona en el tipo de día "
                    "anterior, y suele notarse aquí."
                )

    st.divider()

    # ----- Deriva (pliego §2.4) ----------------------------------------- #
    st.header("Deriva del error diario -- v1")

    if errores is None:
        st.info("Sin datos de errores todavía: no hay deriva que mostrar.")
    elif metricas is None:
        st.info(f"No se puede mostrar la deriva: {error_metricas}")
    else:
        serie = serie_diaria_v1(errores)
        if serie.empty:
            st.info("Ninguna hora publicada de v1 todavía.")
        else:
            serie_con_huecos = _serie_con_huecos_explicitos(serie)
            dias_marcados = dias_no_laborable_a_laborable(
                serie["fecha"].min(), serie["fecha"].max()
            )
            fig_deriva = construir_grafico_deriva(serie_con_huecos, dias_marcados)
            st.plotly_chart(fig_deriva, use_container_width=True)
            st.caption(texto_v2_seguro(metricas))
            st.caption(
                "El signo del sesgo importa más que su tamaño: dos días pueden "
                "tener MAE alto por motivos opuestos -- uno infrapredice, el "
                "otro sobrepredice."
            )

    st.divider()

    # ----- v2 (pliego §2.5) --------------------------------------------- #
    st.header("v2 -- en periodo de prueba")

    if metricas is None or V2 not in metricas:
        st.info("Sin datos de v2 todavía.")
    else:
        bloque_v2 = metricas[V2]
        fechas_v2 = extraer_fechas_presentes(estado_md, V2) if estado_md else None
        en_verde = bloque_v2["actualizado_utc"] == metricas[V1]["actualizado_utc"]

        if errores is not None:
            emisiones_v2 = errores[errores["modelo"] == V2]
            desde = (
                pd.to_datetime(emisiones_v2["fecha_emision"], utc=True).min()
                if not emisiones_v2.empty
                else None
            )
        else:
            desde = None

        st.markdown(
            f"- **Emite desde:** {desde.date().isoformat() if desde is not None else '—'}\n"
            f"- **Fechas con datos:** {fechas_v2 if fechas_v2 is not None else '—'}\n"
            f"- **Última corrida:** "
            f"{'en verde' if en_verde else 'no coincide con la de v1 -- revisar'} "
            f"({bloque_v2['actualizado_utc']})"
        )
        st.markdown(
            "Sus métricas (MAE, sesgo, serie de error) **no se muestran a "
            "propósito** hasta el 2 de octubre de 2026: están pre-registradas "
            "y se evalúan una sola vez, al cierre. No es una disculpa por lo "
            "que falta: es la disciplina del proyecto, escrita.\n\n"
            "Su curva real vs predicha sí es accesible, desde el selector de "
            "arriba."
        )


if __name__ == "__main__":
    main()
