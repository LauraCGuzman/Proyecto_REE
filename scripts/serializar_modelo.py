"""Serializa el modelo champion (`modelo_v1`) y su fixture dorada de paridad.

Ejecuta `notebooks/modelo_demanda.ipynb` en un kernel limpio (nbclient), solo
hasta la celda que entrena `modelo_nivel_final` (celda 28 en el índice
actual, localizada por ancla de texto, nunca por número -- mismo mecanismo
que `scripts/gate_numeros.py`). No hace falta ejecutar el resto del
notebook (13f, delta, 14f, lag_168): ninguna de esas celdas participa en el
modelo publicado.

Salidas (todas se derivan directamente de los objetos en memoria del propio
kernel, nunca de una reimplementación aparte -- así la paridad es garantía
de construcción, no de que dos implementaciones coincidan por casualidad):
  - modelos/modelo_v1.pkl       El objeto `modelo_nivel_final` tal cual.
  - modelos/modelo_v1.json      Metadatos: fecha, rango de datos, MAE test, features.
  - tests/fixtures/paridad_golden.parquet
        Fila por fila de un subconjunto fijo de `test_model`: datetime_utc,
        las 5 features, es_evento, demanda_real y la predicción real de
        `modelo_nivel_final` sobre esas filas. `tests/test_paridad.py` la usa
        como oráculo de tolerancia 0.

No se ejecuta si `modelos/modelo_v1.pkl` ya existe -- el champion está
congelado (pliego Fase 1.5, "Qué NO hacer"): usar --forzar para regenerarlo
a sabiendas.
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
NOTEBOOK = RAIZ / "notebooks" / "modelo_demanda.ipynb"
DIR_MODELOS = RAIZ / "modelos"
DIR_FIXTURES = RAIZ / "tests" / "fixtures"
RUTA_MODELO = DIR_MODELOS / "modelo_v1.pkl"
RUTA_METADATOS = DIR_MODELOS / "modelo_v1.json"
RUTA_FIXTURE = DIR_FIXTURES / "paridad_golden.parquet"

BACKEND_HEADLESS_SRC = "import matplotlib; matplotlib.use('Agg')"

# Ancla de la celda que entrena y mide el modelo publicado (celda 28 hoy).
ANCLA_CELDA_MODELO = (
    "modelo_nivel_final = DecisionTreeRegressor(random_state=42, **params_nivel)"
)

# Número de filas del fixture dorado por cada franja: bordes de test_model
# (para cubrir el arranque y el final del rango) + todas las filas es_evento
# (para no dejar sin cubrir la máscara del 11-J, que es la más frágil) +
# muestreo regular para variar hora/mes/tipo_efectivo.
N_BORDE = 5
PASO_MUESTREO = 400

EXTRACCION_SRC = rf"""
import joblib as _gate_joblib
import json as _gate_json
import pandas as _gate_pd

_gate_dir_modelos = Path(r"{DIR_MODELOS}")
_gate_dir_fixtures = Path(r"{DIR_FIXTURES}")
_gate_dir_modelos.mkdir(parents=True, exist_ok=True)
_gate_dir_fixtures.mkdir(parents=True, exist_ok=True)

_gate_joblib.dump(modelo_nivel_final, Path(r"{RUTA_MODELO}"))

_gate_metadatos = {{
    "fecha_serializacion": _gate_pd.Timestamp.now("UTC").isoformat(),
    "notebook_origen": "notebooks/modelo_demanda.ipynb",
    "celda_ancla": {ANCLA_CELDA_MODELO!r},
    "features": list(features_5f),
    "variable_objetivo": variable_objetivo,
    "hiperparametros": dict(params_nivel),
    "rango_datos_entrenamiento": {{
        "inicio": train_model_clean["datetime_utc"].min().isoformat(),
        "fin": train_model_clean["datetime_utc"].max().isoformat(),
        "n_filas": int(len(train_model_clean)),
    }},
    "mae_test_4344": _r(MAE_NIVEL),
    "mae_test_4336_censo_ree": _r(MAE_NIVEL_4336),
    "sesgo_test": _r(SESGO_NIVEL),
}}
Path(r"{RUTA_METADATOS}").write_text(
    _gate_json.dumps(_gate_metadatos, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

# --- Fixture dorada -----------------------------------------------------
_gate_test = test_model.reset_index(drop=True).copy()
_gate_test["_pred_modelo_v1"] = pred_test_nivel

_gate_n = len(_gate_test)
_gate_idx_borde = set(range({N_BORDE!r})) | set(range(_gate_n - {N_BORDE!r}, _gate_n))
_gate_idx_evento = set(_gate_test.index[_gate_test["es_evento"]])
_gate_idx_muestreo = set(range(0, _gate_n, {PASO_MUESTREO!r}))
_gate_idx_fixture = sorted(_gate_idx_borde | _gate_idx_evento | _gate_idx_muestreo)

_gate_columnas_fixture = (
    ["datetime_utc"] + list(features_5f) + ["es_evento", "demanda_real", "_pred_modelo_v1"]
)
_gate_fixture = _gate_test.loc[_gate_idx_fixture, _gate_columnas_fixture].reset_index(drop=True)
_gate_fixture = _gate_fixture.rename(columns={{"_pred_modelo_v1": "prediccion_modelo_v1"}})
_gate_fixture.to_parquet(Path(r"{RUTA_FIXTURE}"), index=False, engine="pyarrow")

print("###SERIALIZAR_OK###")
print(f"modelo_v1.pkl: {{Path(r'{RUTA_MODELO}')}}")
print(f"modelo_v1.json: {{Path(r'{RUTA_METADATOS}')}}")
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
        help="Regenera modelo_v1.pkl aunque ya exista (el champion está congelado).",
    )
    args = parser.parse_args()

    if RUTA_MODELO.exists() and not args.forzar:
        print(
            f"{RUTA_MODELO.relative_to(RAIZ)} ya existe. El champion está "
            "congelado (pliego Fase 1.5). Usa --forzar para regenerarlo a sabiendas."
        )
        return 0

    nb = nbformat.read(NOTEBOOK, as_version=4)
    idx_celda_modelo = encontrar_celda(nb, ANCLA_CELDA_MODELO)

    # `_r` (redondeo a 4 decimales) vive en la celda de extracción de
    # gate_numeros.py, no en el notebook: se inyecta aquí también para poder
    # reutilizar el mismo criterio de redondeo en los metadatos.
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

    if payload_texto is None or "###SERIALIZAR_OK###" not in payload_texto:
        raise RuntimeError(
            "La celda de extracción no confirmó la serialización. "
            f"Salida cruda:\n{payload_texto}"
        )

    print(payload_texto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
