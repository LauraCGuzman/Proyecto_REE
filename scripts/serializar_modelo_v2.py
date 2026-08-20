"""Serializa `modelo_v2` (`demanda_lag_168`) y su fixture dorada de paridad.

Mismo patrón que `scripts/serializar_modelo.py` para v1: ejecuta
`notebooks/modelo_v2_lag168.ipynb` en un kernel limpio (nbclient), solo hasta
la celda que entrena `modelo_lag168_v2` (celda 14 en el índice actual,
localizada por ancla de texto, nunca por número -- mismo mecanismo que v1).

Diferencia deliberada con v1: el notebook de v2 no tiene ninguna celda que
lea el conjunto de test (2026). Ejecutar "hasta la celda ancla" en v2 nunca
toca 2026, porque esa partición no se materializa en ningún punto del
notebook (pliego, PR A, "NO leer el conjunto de test. Ni una vez").

Salidas (todas derivadas de los objetos en memoria del propio kernel, mismo
motivo que v1 -- la paridad es garantía de construcción):
  - modelos/modelo_v2.pkl       El objeto `modelo_lag168_v2` tal cual.
  - modelos/modelo_v2.json      Metadatos: fecha, rango de datos, MAE/sesgo
        de VALIDACIÓN (no de test -- v2 no lee test en este PR), features.
  - tests/fixtures/paridad_v2_golden.parquet
        Fila por fila de un subconjunto fijo de `train_clean_6f` (no de test:
        v2 no lo lee). `tests/test_paridad_v2.py` la usa como oráculo de
        tolerancia 0.

No se ejecuta si `modelos/modelo_v2.pkl` ya existe -- usar --forzar para
regenerarlo a sabiendas (mismo criterio que v1).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

warnings.filterwarnings("ignore", category=RuntimeWarning, module="zmq")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

RAIZ = Path(__file__).resolve().parent.parent
NOTEBOOK = RAIZ / "notebooks" / "modelo_v2_lag168.ipynb"
DIR_MODELOS = RAIZ / "modelos"
DIR_FIXTURES = RAIZ / "tests" / "fixtures"
RUTA_MODELO = DIR_MODELOS / "modelo_v2.pkl"
RUTA_METADATOS = DIR_MODELOS / "modelo_v2.json"
RUTA_FIXTURE = DIR_FIXTURES / "paridad_v2_golden.parquet"

BACKEND_HEADLESS_SRC = "import matplotlib; matplotlib.use('Agg')"

# Ancla de la celda que entrena el modelo final de v2 (celda 14 hoy).
ANCLA_CELDA_MODELO = (
    "modelo_lag168_v2 = DecisionTreeRegressor(random_state=42, **params_v2)"
)

# Mismo criterio de muestreo del fixture que v1 (scripts/serializar_modelo.py):
# bordes (arranque/final) + es_evento + un es_puente=True + muestreo regular.
# Fuente distinta a propósito: train_clean_6f (2023-2025), nunca test (2026).
N_BORDE = 5
N_PUENTE = 1
PASO_MUESTREO = 400

EXTRACCION_SRC = rf"""
import joblib as _gate_joblib
import json as _gate_json
import pandas as _gate_pd

import holidays as _gate_holidays
import sklearn as _gate_sklearn

_gate_dir_modelos = Path(r"{DIR_MODELOS}")
_gate_dir_fixtures = Path(r"{DIR_FIXTURES}")
_gate_dir_modelos.mkdir(parents=True, exist_ok=True)
_gate_dir_fixtures.mkdir(parents=True, exist_ok=True)

_gate_joblib.dump(modelo_lag168_v2, Path(r"{RUTA_MODELO}"))

_gate_metadatos = {{
    "fecha_serializacion": _gate_pd.Timestamp.now("UTC").isoformat(),
    "notebook_origen": "notebooks/modelo_v2_lag168.ipynb",
    "celda_ancla": {ANCLA_CELDA_MODELO!r},
    "features": list(FEATURES_6F),
    "variable_objetivo": VARIABLE_OBJETIVO,
    "hiperparametros": dict(params_v2),
    "version_sklearn": _gate_sklearn.__version__,
    "version_holidays": _gate_holidays.__version__,
    "rango_datos_entrenamiento": {{
        "inicio": train_clean_6f["datetime_utc"].min().isoformat(),
        "fin": train_clean_6f["datetime_utc"].max().isoformat(),
        "n_filas": int(len(train_clean_6f)),
    }},
    "mae_val": _r(MAE_VAL_V2),
    "sesgo_val": _r(SESGO_VAL_V2),
    "conjunto_val": (
        "Validacion 2025 (8760 filas); modelo de comparacion entrenado con "
        "train_fit_6f (2023-2024) y medido en val_6f (2025), NO el modelo "
        "serializado (que entrena train_clean_6f completo, 2023-2025). "
        "Referencia documental (pliego Fase5bis, 1.3), no criterio de "
        "promocion. No lleva campos de test: PR A no lee el conjunto de "
        "test (2026) -- desviacion de esquema respecto a modelo_v1.json "
        "anotada en el log."
    ),
}}
Path(r"{RUTA_METADATOS}").write_text(
    _gate_json.dumps(_gate_metadatos, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

# --- Fixture dorada -------------------------------------------------------
# Fuente: train_clean_6f (2023-2025), nunca test. No es una reimplementación
# aparte de las predicciones: son las del propio modelo_lag168_v2 en memoria.
_gate_muestra = train_clean_6f.reset_index(drop=True).copy()
_gate_muestra["_pred_modelo_v2"] = modelo_lag168_v2.predict(_gate_muestra[FEATURES_6F])

_gate_n = len(_gate_muestra)
_gate_idx_borde = set(range({N_BORDE!r})) | set(range(_gate_n - {N_BORDE!r}, _gate_n))
_gate_idx_evento = set(_gate_muestra.index[_gate_muestra["es_evento"]])
_gate_idx_puente = set(_gate_muestra.index[_gate_muestra["es_puente"]][: {N_PUENTE!r}])
_gate_idx_muestreo = set(range(0, _gate_n, {PASO_MUESTREO!r}))
_gate_idx_fixture = sorted(
    _gate_idx_borde | _gate_idx_evento | _gate_idx_puente | _gate_idx_muestreo
)

_gate_columnas_fixture = (
    ["datetime_utc"] + list(FEATURES_6F) + ["es_evento", "demanda_real", "_pred_modelo_v2"]
)
_gate_fixture = _gate_muestra.loc[_gate_idx_fixture, _gate_columnas_fixture].reset_index(drop=True)
_gate_fixture = _gate_fixture.rename(columns={{"_pred_modelo_v2": "prediccion_modelo_v2"}})
_gate_fixture.to_parquet(Path(r"{RUTA_FIXTURE}"), index=False, engine="pyarrow")

print("###SERIALIZAR_V2_OK###")
print(f"modelo_v2.pkl: {{Path(r'{RUTA_MODELO}')}}")
print(f"modelo_v2.json: {{Path(r'{RUTA_METADATOS}')}}")
print(f"fixture dorada: {{len(_gate_fixture)}} filas -> {{Path(r'{RUTA_FIXTURE}')}}")
"""


def encontrar_celda(nb, ancla: str) -> int:
    coincidencias = [
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and ancla in "".join(c.source)
    ]
    if not coincidencias:
        raise RuntimeError(
            f"No se encuentra ninguna celda con el ancla: {ancla!r}. "
            "El notebook ha cambiado; el script no puede localizar dónde parar."
        )
    if len(coincidencias) > 1:
        raise RuntimeError(
            f"El ancla {ancla!r} aparece en más de una celda ({coincidencias})."
        )
    return coincidencias[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Regenera modelo_v2.pkl aunque ya exista.",
    )
    args = parser.parse_args()

    if RUTA_MODELO.exists() and not args.forzar:
        print(
            f"{RUTA_MODELO.relative_to(RAIZ)} ya existe. Usa --forzar para "
            "regenerarlo a sabiendas."
        )
        return 0

    nb = nbformat.read(NOTEBOOK, as_version=4)
    idx_celda_modelo = encontrar_celda(nb, ANCLA_CELDA_MODELO)

    # `_r` (redondeo a 4 decimales), igual que en scripts/serializar_modelo.py:
    # no vive en el notebook, se inyecta aquí para reutilizar el mismo
    # criterio de redondeo en los metadatos.
    helper_cell = nbformat.v4.new_code_cell(
        "def _r(x):\n    return round(float(x), 4)"
    )

    print(f"Ejecutando {NOTEBOOK.relative_to(RAIZ)} hasta la celda {idx_celda_modelo} "
          "(kernel limpio, nbclient)...")

    client = NotebookClient(nb, kernel_name="python3", timeout=1800, allow_errors=False)
    payload_texto = None

    with client.setup_kernel():
        backend_cell = nbformat.v4.new_code_cell(BACKEND_HEADLESS_SRC)
        nb.cells.append(backend_cell)
        client.execute_cell(backend_cell, len(nb.cells) - 1, store_history=False)

        nb.cells.append(helper_cell)
        client.execute_cell(helper_cell, len(nb.cells) - 1, store_history=False)

        for i in range(idx_celda_modelo + 1):
            cell = nb.cells[i]
            if cell.cell_type != "code":
                continue
            try:
                client.execute_cell(cell, i)
            except CellExecutionError:
                print(f"\n!! Fallo ejecutando la celda {i} del notebook.", file=sys.stderr)
                raise

        extraccion_cell = nbformat.v4.new_code_cell(EXTRACCION_SRC)
        nb.cells.append(extraccion_cell)
        out_cell = client.execute_cell(
            extraccion_cell, len(nb.cells) - 1, store_history=False
        )
        textos = [o.get("text", "") for o in out_cell.get("outputs", []) if o.get("text")]
        payload_texto = "".join(textos) if textos else None

    if payload_texto is None or "###SERIALIZAR_V2_OK###" not in payload_texto:
        raise RuntimeError(
            "La celda de extracción no confirmó la serialización. "
            f"Salida cruda:\n{payload_texto}"
        )

    print(payload_texto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
