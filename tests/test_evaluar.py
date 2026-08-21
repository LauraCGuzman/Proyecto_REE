# tests/test_evaluar.py
"""Tests de `pipeline/evaluar.py`. Solo funciones puras, sin red -- datos
sintéticos en memoria, nunca e·sios ni ficheros del repo (pliego Fase 2 §5).

Se puede ejecutar como script (`python tests/test_evaluar.py`) o recolectar
con pytest, mismo patrón que `tests/test_paridad.py`: las funciones `test_*`
usan solo `assert` plano."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from pipeline.evaluar import (
    COLUMNAS_ERRORES,
    COBERTURA_MINIMA_DIAS,
    MODELOS_ACTIVOS,
    NOMBRE_MODELO,
    NOMBRE_MODELO_V2,
    calcular_metricas,
    construir_estado_pipeline_md,
    construir_filas_error,
    filas_pendientes,
)


def _horizontes(n: int, inicio="2026-08-14T00:00:00Z") -> list[str]:
    base = pd.Timestamp(inicio, tz="UTC")
    return [(base + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n)]


def _predicciones_sinteticas(n: int) -> pd.DataFrame:
    horizontes = _horizontes(n)
    return pd.DataFrame(
        {
            "fecha_emision": ["2026-08-13T06:45:48Z"] * n,
            "horizonte": horizontes,
            "valor_predicho": [30_000.0 + i for i in range(n)],
            "ancla_ultimo_dia_real": ["2026-08-13"] * n,
            "modelo": [NOMBRE_MODELO] * n,
        }
    )


def _errores_sinteticos_desde(predicciones: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    """Fabrica filas de errores.csv ya evaluadas para un subconjunto de
    `predicciones` (por posición), con valores dummy -- solo la clave
    (horizonte, modelo) importa para estos tests."""
    filas = predicciones.iloc[indices]
    return pd.DataFrame(
        {
            "fecha_evaluacion": ["2026-08-15T05:31:00Z"] * len(filas),
            "horizonte": filas["horizonte"].to_numpy(),
            "modelo": filas["modelo"].to_numpy(),
            "fecha_emision": filas["fecha_emision"].to_numpy(),
            "valor_predicho": filas["valor_predicho"].to_numpy(),
            "valor_real": filas["valor_predicho"].to_numpy() + 100.0,
            "error": [100.0] * len(filas),
            "h_adelanto_h": [12.0] * len(filas),
        }
    )[COLUMNAS_ERRORES]


def test_deteccion_de_pendientes():
    """24 filas en predicciones.csv, 10 ya evaluadas en errores.csv ->
    exactamente las 14 restantes están pendientes."""
    predicciones = _predicciones_sinteticas(24)
    errores = _errores_sinteticos_desde(predicciones, list(range(10)))

    pendientes = filas_pendientes(predicciones, errores)

    assert len(pendientes) == 14, f"Se esperaban 14 pendientes, hay {len(pendientes)}"
    horizontes_pendientes = set(pendientes["horizonte"])
    horizontes_evaluados = set(errores["horizonte"])
    assert horizontes_pendientes.isdisjoint(horizontes_evaluados)
    assert horizontes_pendientes == set(predicciones["horizonte"]) - horizontes_evaluados


def test_hora_incompleta_sigue_pendiente():
    """Un horizonte cuyo `n_lecturas` es 11 nunca entra en `reales` (que solo
    trae horas con n_lecturas == 12): sigue pendiente, no se escribe fila
    para ella, y no se lanza ninguna excepción."""
    predicciones = _predicciones_sinteticas(2)
    horizonte_completo = pd.Timestamp(predicciones.iloc[0]["horizonte"])
    horizonte_incompleto = pd.Timestamp(predicciones.iloc[1]["horizonte"])

    # Solo el primer horizonte tiene dato real disponible -- el segundo
    # simula la hora con n_lecturas=11 (filtrada aguas arriba, nunca llega
    # a `reales`).
    reales = pd.Series([30_500.0], index=[horizonte_completo])

    filas_error = construir_filas_error(predicciones, reales, "2026-08-16T05:31:00Z")

    assert len(filas_error) == 1, f"Se esperaba 1 fila de error, hay {len(filas_error)}"
    assert filas_error.iloc[0]["horizonte"] == horizonte_completo.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert horizonte_incompleto.strftime("%Y-%m-%dT%H:%M:%SZ") not in filas_error["horizonte"].to_numpy()


def test_signo_h_adelanto_h():
    """Una fila emitida-en-futuro (horizonte tras fecha_emision) y una
    emitida-en-pasado (el caso real de las 22:00Z-06:00Z, horizonte antes de
    fecha_emision): signos opuestos, y el filtro de publicado
    (h_adelanto_h > 0) se queda con una sola."""
    predicciones = pd.DataFrame(
        {
            "fecha_emision": [
                "2026-08-16T05:31:00Z",  # emitida a las 05:31...
                "2026-08-16T05:31:00Z",
            ],
            "horizonte": [
                "2026-08-16T22:00:00Z",  # ...horizonte 22:00 el mismo día: futuro -> h_adelanto_h > 0
                "2026-08-16T02:00:00Z",  # ...horizonte 02:00 el mismo día: pasado -> h_adelanto_h < 0
            ],
            "valor_predicho": [30_000.0, 29_000.0],
            "ancla_ultimo_dia_real": ["2026-08-15", "2026-08-15"],
            "modelo": [NOMBRE_MODELO, NOMBRE_MODELO],
        }
    )
    reales = pd.Series(
        [30_100.0, 28_900.0],
        index=[pd.Timestamp("2026-08-16T22:00:00Z"), pd.Timestamp("2026-08-16T02:00:00Z")],
    )

    filas_error = construir_filas_error(predicciones, reales, "2026-08-17T05:31:00Z")

    assert len(filas_error) == 2
    signos = set(filas_error["h_adelanto_h"] > 0)
    assert signos == {True, False}, "Los dos signos de h_adelanto_h deben coexistir"

    publicado = filas_error[filas_error["h_adelanto_h"] > 0]
    assert len(publicado) == 1, "El filtro de publicado debe quedarse con una sola fila"


def test_idempotencia_dos_corridas():
    """Dos corridas seguidas sobre el mismo estado: `errores.csv` no crece
    en la segunda (todo lo pendiente en la primera corrida ya quedó
    evaluado)."""
    predicciones = _predicciones_sinteticas(5)
    horizonte_ts = pd.to_datetime(predicciones["horizonte"], utc=True)
    reales = pd.Series(
        (predicciones["valor_predicho"] + 50.0).to_numpy(), index=horizonte_ts.to_numpy()
    )

    errores = pd.DataFrame(columns=COLUMNAS_ERRORES)

    # --- Corrida 1 ---
    pendientes_1 = filas_pendientes(predicciones, errores)
    assert len(pendientes_1) == 5
    filas_nuevas_1 = construir_filas_error(pendientes_1, reales, "2026-08-16T05:31:00Z")
    errores = pd.concat([errores, filas_nuevas_1], ignore_index=True)
    n_tras_corrida_1 = len(errores)

    # --- Corrida 2, mismo estado ---
    pendientes_2 = filas_pendientes(predicciones, errores)
    assert pendientes_2.empty, "No debería quedar nada pendiente tras la corrida 1"
    filas_nuevas_2 = construir_filas_error(pendientes_2, reales, "2026-08-17T05:31:00Z")
    errores = pd.concat([errores, filas_nuevas_2], ignore_index=True)

    assert len(errores) == n_tras_corrida_1, "errores.csv no debe crecer en la segunda corrida"
    assert filas_nuevas_2.empty


def test_muestra_insuficiente():
    """3 días de errores -> `mae` es `null` en las tres ventanas (7d/30d/90d)
    y `muestra_insuficiente` es `True`: ninguna ventana llega a los
    `COBERTURA_MINIMA_DIAS` fechas de calendario distintas que gobiernan la
    guarda (PR fix-guarda-cobertura: antes era un span en días, `cobertura_dias`
    < `COBERTURA_MINIMA_DIAS`; ahora es `dias_cubiertos`)."""
    ahora_utc = pd.Timestamp("2026-08-16T05:31:00Z")
    horizontes = _horizontes(3 * 24, inicio="2026-08-14T00:00:00Z")  # 3 días completos

    errores = pd.DataFrame(
        {
            "fecha_evaluacion": ["2026-08-16T05:31:00Z"] * len(horizontes),
            "horizonte": horizontes,
            "modelo": [NOMBRE_MODELO] * len(horizontes),
            "fecha_emision": ["2026-08-13T06:45:48Z"] * len(horizontes),
            "valor_predicho": [30_000.0] * len(horizontes),
            "valor_real": [30_500.0] * len(horizontes),
            "error": [500.0] * len(horizontes),
            "h_adelanto_h": [12.0] * len(horizontes),  # publicado: h_adelanto_h > 0
        }
    )[COLUMNAS_ERRORES]

    metricas = calcular_metricas(errores, NOMBRE_MODELO, ahora_utc)

    assert metricas["muestra_insuficiente"] is True
    for nombre_ventana, valores in metricas["publicado"]["ventanas"].items():
        assert valores["mae"] is None, f"mae de {nombre_ventana} debería ser null"
        assert valores["sesgo_medio"] is None, f"sesgo_medio de {nombre_ventana} debería ser null"
        assert valores["dias_cubiertos"] < COBERTURA_MINIMA_DIAS
        assert valores["cobertura_dias"] < COBERTURA_MINIMA_DIAS  # se conserva, ya no gobierna


def test_siete_dias_consecutivos_escribe_mae():
    """El caso que fallaba antes de este PR (pliego §1.0): siete días
    calendario consecutivos, cada uno con horas publicadas 06:00Z-21:00Z (16
    horas/día, el patrón real del cron). El span entre el primer y el último
    horizonte es 6,6 días (6 días y 15 horas) -- con la guarda vieja
    (`cobertura_dias >= COBERTURA_MINIMA_DIAS`) `mae` se quedaba en `null`
    para siempre con la ventana llena. Con `dias_cubiertos` (7 fechas
    distintas) `mae` se escribe."""
    dias = ["2026-08-13", "2026-08-14", "2026-08-15", "2026-08-16", "2026-08-17",
            "2026-08-18", "2026-08-19"]
    horizontes = [f"{dia}T{hora:02d}:00:00Z" for dia in dias for hora in range(6, 22)]
    n = len(horizontes)
    assert n == 7 * 16

    errores = pd.DataFrame(
        {
            "fecha_evaluacion": ["2026-08-19T05:49:00Z"] * n,
            "horizonte": horizontes,
            "modelo": [NOMBRE_MODELO] * n,
            "fecha_emision": ["2026-08-12T06:45:48Z"] * n,
            "valor_predicho": [30_000.0] * n,
            "valor_real": [30_500.0] * n,
            "error": [500.0] * n,
            "h_adelanto_h": [12.0] * n,  # publicado: h_adelanto_h > 0
        }
    )[COLUMNAS_ERRORES]

    ahora_utc = pd.Timestamp("2026-08-19T05:49:00Z")
    metricas = calcular_metricas(errores, NOMBRE_MODELO, ahora_utc)

    ventana_7d = metricas["publicado"]["ventanas"]["7d"]
    assert ventana_7d["cobertura_dias"] < COBERTURA_MINIMA_DIAS, (
        "el span (6,6) debe seguir siendo menor que 7 -- es el hallazgo que "
        "motiva este PR, no algo que este test deba corregir"
    )
    assert ventana_7d["dias_cubiertos"] == COBERTURA_MINIMA_DIAS
    assert ventana_7d["mae"] is not None, "mae no debería ser None con 7 fechas distintas"
    assert ventana_7d["mae"] == 500.0
    assert ventana_7d["sesgo_medio"] == 500.0


def test_referencia_v1_y_v2_no_comparables():
    """Fase 5bis PR B (emisión paralela): el bloque `referencia` cambia de
    forma según el modelo -- v1 sigue con `mae_test_notebook` (sin tocar);
    v2 usa `mae_val_notebook` y lleva una `nota` explícita de que no es de
    test. Los dos modelos activos deben tener referencia declarada."""
    ahora_utc = pd.Timestamp("2026-08-20T05:31:00Z")
    errores_vacio = pd.DataFrame(columns=COLUMNAS_ERRORES)

    assert MODELOS_ACTIVOS == [NOMBRE_MODELO, NOMBRE_MODELO_V2]

    metricas_v1 = calcular_metricas(errores_vacio, NOMBRE_MODELO, ahora_utc)
    ref_v1 = metricas_v1["referencia"]
    assert "mae_test_notebook" in ref_v1
    assert "mae_val_notebook" not in ref_v1
    assert "nota" not in ref_v1

    metricas_v2 = calcular_metricas(errores_vacio, NOMBRE_MODELO_V2, ahora_utc)
    ref_v2 = metricas_v2["referencia"]
    assert "mae_val_notebook" in ref_v2
    assert "mae_test_notebook" not in ref_v2
    assert "nota" in ref_v2 and "test" in ref_v2["nota"].lower()

    # La referencia de persistencia sí es compartida (no depende del modelo).
    assert ref_v1["mae_baseline_persistencia"] == ref_v2["mae_baseline_persistencia"]


def test_v2_aparece_con_guion_el_dia_uno():
    """Pliego §3.5, último punto: "v2 arranca de cero y estará en `—` sus
    primeros seis días. Es correcto." -- con errores.csv sin ninguna fila
    `modelo == "v2"` todavía, calcular_metricas("v2", ...) debe devolver un
    bloque con mae/sesgo en null (no lanzar, no devolver un bloque vacío)."""
    ahora_utc = pd.Timestamp("2026-08-20T05:31:00Z")
    # errores.csv con SOLO filas de v1 -- v2 nunca ha evaluado nada todavía.
    errores_solo_v1 = pd.DataFrame(
        {
            "fecha_evaluacion": ["2026-08-20T05:31:00Z"] * 24,
            "horizonte": _horizontes(24, inicio="2026-08-19T00:00:00Z"),
            "modelo": [NOMBRE_MODELO] * 24,
            "fecha_emision": ["2026-08-19T05:45:00Z"] * 24,
            "valor_predicho": [30_000.0] * 24,
            "valor_real": [30_500.0] * 24,
            "error": [500.0] * 24,
            "h_adelanto_h": [12.0] * 24,
        }
    )[COLUMNAS_ERRORES]

    metricas_v2 = calcular_metricas(errores_solo_v1, NOMBRE_MODELO_V2, ahora_utc)
    assert metricas_v2["modelo"] == NOMBRE_MODELO_V2
    assert metricas_v2["muestra_insuficiente"] is True
    for valores in metricas_v2["publicado"]["ventanas"].values():
        assert valores["mae"] is None
        assert valores["n_horas"] == 0


def test_construir_estado_pipeline_md_dos_bloques():
    """`estado_pipeline.md` con los dos modelos, cada uno en su propio
    bloque (pliego §3.6): sin mezclar sus números, y sin que el bloque de
    v2 arrastre la prosa de "techo de extrapolación" verificada solo para
    v1 (pliego, comprobación de esta sesión)."""
    ahora_utc = pd.Timestamp("2026-08-20T05:31:00Z")
    errores_vacio = pd.DataFrame(columns=COLUMNAS_ERRORES)

    metricas_por_modelo = {
        NOMBRE_MODELO: calcular_metricas(errores_vacio, NOMBRE_MODELO, ahora_utc),
        NOMBRE_MODELO_V2: calcular_metricas(errores_vacio, NOMBRE_MODELO_V2, ahora_utc),
    }
    n_dias_por_modelo = {NOMBRE_MODELO: 0, NOMBRE_MODELO_V2: 0}

    md = construir_estado_pipeline_md(metricas_por_modelo, n_dias_por_modelo)

    assert "## Modelo: v1" in md
    assert "## Modelo: v2" in md
    assert md.index("## Modelo: v1") < md.index("## Modelo: v2"), (
        "v1 va primero, mismo orden que los pasos del pliego"
    )
    assert "38.861,1 MW" in md, "la prosa de v1 (techo de extrapolación) debe seguir presente"
    # La prosa de "techo de extrapolación"/"arranque de la semana" es
    # observación medida de v1: no debe aparecer también bajo el bloque de
    # v2 sin haberse medido para v2.
    bloque_v2 = md[md.index("## Modelo: v2"):]
    assert "38.861,1 MW" not in bloque_v2
    assert "no comparable" in bloque_v2.lower() or "no es comparable" in bloque_v2.lower()


def main() -> int:
    test_deteccion_de_pendientes()
    print("✔ test_deteccion_de_pendientes")
    test_hora_incompleta_sigue_pendiente()
    print("✔ test_hora_incompleta_sigue_pendiente")
    test_signo_h_adelanto_h()
    print("✔ test_signo_h_adelanto_h")
    test_idempotencia_dos_corridas()
    print("✔ test_idempotencia_dos_corridas")
    test_muestra_insuficiente()
    print("✔ test_muestra_insuficiente")
    test_siete_dias_consecutivos_escribe_mae()
    print("✔ test_siete_dias_consecutivos_escribe_mae")
    test_referencia_v1_y_v2_no_comparables()
    print("✔ test_referencia_v1_y_v2_no_comparables")
    test_v2_aparece_con_guion_el_dia_uno()
    print("✔ test_v2_aparece_con_guion_el_dia_uno")
    test_construir_estado_pipeline_md_dos_bloques()
    print("✔ test_construir_estado_pipeline_md_dos_bloques")
    print("\nTodos los tests de evaluar.py en verde.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
