"""Gate de regresión numérica — §0 de PLAN_REFACTOR_REE_FASE1.md.

Ejecuta `notebooks/modelo_demanda.ipynb` de arriba abajo en un kernel limpio
(nbclient), extrae las métricas, los conteos de fila y los hiperparámetros
ganadores que fija el §0, y los vuelca a `reports/baseline_numeros.json`.

Uso:
    .venv\\Scripts\\python.exe scripts\\gate_numeros.py            # genera/actualiza el baseline
    .venv\\Scripts\\python.exe scripts\\gate_numeros.py --check    # compara contra el baseline existente

`--check` NUNCA reescribe el baseline. Si algo difiere, sale con código != 0 y
lista las diferencias. Debe ejecutarse con el intérprete del venv del proyecto
(para que nbclient encuentre el kernel "python3" instalado ahí).

Notas de implementación:
- No se modifica `notebooks/modelo_demanda.ipynb` en disco: el notebook se
  ejecuta celda a celda con nbclient sobre un objeto en memoria; las celdas de
  extracción que este script inyecta (backend headless de matplotlib, capturas
  intermedias, volcado final a JSON) se añaden solo a esa copia en memoria y
  nunca se escriben de vuelta al .ipynb.
- El MAE de persistencia (B2 del pliego, antes un `print` suelto sin
  variable) ya vive en el notebook como `MAE_PERSISTENCIA`; la extracción
  final lo lee directo, sin celda de captura inyectada — ya no hace falta.
- `train`, `test`, `train_clean`, `train_fit` y `val` se reasignan varias veces
  a lo largo del notebook (una vez por rama de features / rejilla de
  hiperparámetros). El JSON final solo ve el último valor de cada una, así que
  `CAPTURAS_POST_CELDA` engancha una captura después de cada celda que fija
  una de esas particiones, localizada por un fragmento de texto único dentro
  de esa celda, nunca por índice. Ver `conteos_por_etapa` en el JSON de
  salida — las claves están agrupadas en dos familias:
    - `rama_*`: el recorte de filas tras el dropna de esa rama (5f, 13f,
      lag168). `rama_5f` usa `train_model_clean`/`test_model`; `rama_13f` y
      `rama_lag168` usan `train_clean`/`test`, pero en dos estados distintos de
      `train_clean` (dropna de un lag vs. de dos).
    - `rejilla_*`: el `train_fit`/`val` con el que corrió cada barrido de
      hiperparámetros (5f, 13f, delta, 14f, 5f_lags).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Silencia el RuntimeWarning cosmético de pyzmq con el proactor loop de Windows
# (no afecta a la ejecución: nbclient añade su propio selector thread).
warnings.filterwarnings("ignore", category=RuntimeWarning, module="zmq")

# La consola de Windows usa cp1252 por defecto, que no representa "Δ", "✔",
# "✘" ni las tildes de los mensajes de diff. Sin esto, un --check con
# diferencias podría reventar con UnicodeEncodeError justo al reportarlas.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

RAIZ = Path(__file__).resolve().parent.parent
NOTEBOOK = RAIZ / "notebooks" / "modelo_demanda.ipynb"
BASELINE = RAIZ / "reports" / "baseline_numeros.json"

BACKEND_HEADLESS_SRC = "import matplotlib; matplotlib.use('Agg')"

# Capturas inyectadas justo después de que termine de ejecutar la celda
# ancla. El ancla es un fragmento de texto verificado único en todo el
# notebook (ver tests de este script) — nunca un índice de celda, porque los
# índices rotan si se borran celdas. Si el refactor mueve o reescribe una de
# estas celdas de forma que el ancla deje de aparecer (o aparezca más de una
# vez), `encontrar_celda` para la ejecución en vez de capturar en el sitio
# equivocado en silencio.
CAPTURAS_POST_CELDA: list[tuple[str, str, str]] = [
    (
        # celda 21 — fija train_model_clean/test_model (rama 5f, único punto
        # de asignación: no hace falta capturar nada más para esta rama).
        'train_model_clean = train_model.dropna(subset=["demanda_lag_24"]).copy()',
        '_gate_rama_5f = {"train_model_clean": len(train_model_clean), '
        '"test_model": len(test_model)}',
        "rama_5f",
    ),
    (
        # celda 29 — rejilla nivel/5f, train_fit/val sobre train_model_clean.
        "train_fit = train_model_clean[años_sel <= 2024].copy()",
        '_gate_rejilla_5f = {"train_fit": len(train_fit), "val": len(val)}',
        "rejilla_5f",
    ),
    (
        # celda 39 — rama 13f: train_clean tras el dropna de un solo lag.
        'train_clean = train.dropna(subset=["demanda_lag_24"]).copy()',
        '_gate_rama_13f = {"train_clean": len(train_clean), "test": len(test)}',
        "rama_13f",
    ),
    (
        # celda 43 — rejilla 13f. El texto de train_fit/val es idéntico al de
        # las celdas 47/57/59 (todas cortan train_clean igual); se ancla en la
        # cabecera markdown de la celda, que sí es única.
        "REJILLA SOBRE VALIDACIÓN — 13 features (calendario + 8 temperaturas), NIVEL",
        '_gate_rejilla_13f = {"train_fit": len(train_fit), "val": len(val)}',
        "rejilla_13f",
    ),
    (
        # celda 47 — rejilla delta. Reutiliza el train_clean de la rama 13f
        # ("misma partición que la celda 43" dice el propio comentario), pero
        # es su propio barrido con su propio train_fit/val: se captura aparte.
        "MODELO DELTA — objetivo `demanda_real - demanda_lag_24`, 12 features",
        '_gate_rejilla_delta = {"train_fit": len(train_fit), "val": len(val)}',
        "rejilla_delta",
    ),
    (
        # celda 55 — rama lag168: train_clean tras el dropna de los dos lags.
        'train_clean = train.dropna(subset=["demanda_lag_24", "demanda_lag_168"]).copy()',
        '_gate_rama_lag168 = {"train_clean": len(train_clean), "test": len(test)}',
        "rama_lag168",
    ),
    (
        # celda 57 — rejilla 14f.
        "REJILLA SOBRE VALIDACIÓN — 14 features (calendario + 8 temperaturas + 2 lags), NIVEL",
        '_gate_rejilla_14f = {"train_fit": len(train_fit), "val": len(val)}',
        "rejilla_14f",
    ),
    (
        # celda 59 — rejilla 5f_lags.
        "REJILLA SOBRE VALIDACIÓN — 5 features con lag_168 (sin temperatura, sin es_puente), NIVEL",
        '_gate_rejilla_5f_lags = {"train_fit": len(train_fit), "val": len(val)}',
        "rejilla_5f_lags",
    ),
]

EXTRACCION_SRC = r"""
import json as _gate_json


def _gate_default(o):
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


def _r(x):
    return round(float(x), 4)


_gate_metricas = {
    "mae_ree": _r(mae_ree),
    "mae_modelo_tonto": _r(mae_modelo_tonto),
    "MAE_VAL_5F": _r(MAE_VAL_5F),
    "MAE_NIVEL": _r(MAE_NIVEL),
    "MAE_NIVEL_4336": _r(MAE_NIVEL_4336),
    "SESGO_NIVEL": _r(SESGO_NIVEL),
    "MAE_VAL_13F": _r(MAE_VAL_13F),
    "MAE_VAL_DELTA": _r(MAE_VAL_DELTA),
    "MAE_DELTA": _r(MAE_DELTA),
    "MAE_DELTA_4336": _r(MAE_DELTA_4336),
    "SESGO_DELTA": _r(SESGO_DELTA),
    "MAE_VAL_14F": _r(MAE_VAL_14F),
    "MAE_VAL_5f_lags": _r(MAE_VAL_5f_lags),
    "MAE_LAG168": _r(MAE_LAG168),
    "MAE_LAG168_4336": _r(MAE_LAG168_4336),
    "SESGO_LAG168_4336": _r(SESGO_LAG168_4336),
    "MAE_PERSISTENCIA": _r(MAE_PERSISTENCIA),
}

_gate_conteos = {
    "len_train_model_clean": len(train_model_clean),
    "len_test_model": len(test_model),
    "len_train_clean": len(train_clean),
    "len_test": len(test),
    "len_train_fit": len(train_fit),
    "len_val": len(val),
    "N_VAL_5F": int(N_VAL_5F),
}

_gate_params = {
    "params_nivel": dict(params_nivel),
    "params_13f": dict(params_13f),
    "params_delta": dict(params_delta),
    "params_14f": dict(params_14f),
    "params_5f_lags": dict(params_5f_lags),
}

# Dispersión del top-5 de cada rejilla: el criterio que decide si una mejora
# de MAE es real o ruido de desempate (ver dispersion_top en la celda 29).
_gate_dispersiones = {
    "dispersion_nivel": _r(dispersion_top(ranking_nivel)),
    "dispersion_13f": _r(dispersion_top(ranking_13f)),
    "dispersion_delta": _r(dispersion_top(ranking_delta)),
    "dispersion_14f": _r(dispersion_top(ranking_14f)),
    "dispersion_5f_lags": _r(dispersion_top(ranking_5f_lag)),
}

# Conteos de fila por etapa (no solo el estado final): train/test/train_clean/
# train_fit/val se reasignan varias veces a lo largo del notebook, una por
# rama de features y por rejilla de hiperparámetros. Ver docstring del módulo.
_gate_conteos_por_etapa = {
    "rama_5f": _gate_rama_5f,
    "rejilla_5f": _gate_rejilla_5f,
    "rama_13f": _gate_rama_13f,
    "rejilla_13f": _gate_rejilla_13f,
    "rejilla_delta": _gate_rejilla_delta,
    "rama_lag168": _gate_rama_lag168,
    "rejilla_14f": _gate_rejilla_14f,
    "rejilla_5f_lags": _gate_rejilla_5f_lags,
}

_gate_df_ventanas = df_ventanas.astype(object).where(
    df_ventanas.notna(), None
).to_dict(orient="records")

_gate_payload = {
    "metricas": _gate_metricas,
    "conteos_fila": _gate_conteos,
    "conteos_por_etapa": _gate_conteos_por_etapa,
    "params": _gate_params,
    "dispersiones": _gate_dispersiones,
    "df_ventanas": _gate_df_ventanas,
}

print("###GATE_JSON_START###")
print(_gate_json.dumps(_gate_payload, ensure_ascii=False, default=_gate_default))
print("###GATE_JSON_END###")
"""


def encontrar_celda(nb, ancla: str) -> int:
    """Localiza el índice de la única celda de código cuyo source contiene `ancla`."""
    coincidencias = [
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and ancla in "".join(c.source)
    ]
    if not coincidencias:
        raise RuntimeError(
            f"No se encuentra ninguna celda con el ancla: {ancla!r}. "
            "El notebook ha cambiado; el gate no puede localizar el punto de captura."
        )
    if len(coincidencias) > 1:
        raise RuntimeError(
            f"El ancla {ancla!r} aparece en más de una celda ({coincidencias}). "
            "Ambigüedad: hay que precisar el texto de búsqueda."
        )
    return coincidencias[0]


def ejecutar_notebook(notebook_path: Path) -> dict:
    nb = nbformat.read(notebook_path, as_version=4)
    total_original = len(nb.cells)

    # Resuelve cada ancla a un índice de celda ANTES de ejecutar nada: si una
    # ha dejado de ser única o ha desaparecido, el gate para aquí, no a medio
    # "Run All" con parte del kernel ya poblado.
    capturas_por_indice: dict[int, list[tuple[str, str]]] = {}
    for ancla, codigo, nombre in CAPTURAS_POST_CELDA:
        idx = encontrar_celda(nb, ancla)
        capturas_por_indice.setdefault(idx, []).append((nombre, codigo))

    client = NotebookClient(nb, kernel_name="python3", timeout=1800, allow_errors=False)

    payload_texto = None

    with client.setup_kernel():
        # Solo afecta al backend de renderizado de esta ejecución headless;
        # no cambia ningún cálculo del notebook.
        backend_cell = nbformat.v4.new_code_cell(BACKEND_HEADLESS_SRC)
        nb.cells.append(backend_cell)
        client.execute_cell(backend_cell, len(nb.cells) - 1, store_history=False)

        # Un único paso por las celdas originales, en su índice real (0..total_original-1).
        # Las celdas inyectadas se añaden siempre al final de nb.cells, así que nunca
        # desplazan los índices de las celdas del notebook todavía por ejecutar.
        for i in range(total_original):
            cell = nb.cells[i]
            if cell.cell_type != "code":
                continue
            try:
                client.execute_cell(cell, i)
            except CellExecutionError:
                print(f"\n!! Fallo ejecutando la celda {i} del notebook.", file=sys.stderr)
                raise
            for nombre, codigo in capturas_por_indice.get(i, []):
                captura_cell = nbformat.v4.new_code_cell(codigo)
                nb.cells.append(captura_cell)
                try:
                    client.execute_cell(captura_cell, len(nb.cells) - 1, store_history=False)
                except CellExecutionError:
                    print(
                        f"\n!! Fallo capturando '{nombre}' justo después de la celda {i}.",
                        file=sys.stderr,
                    )
                    raise

        extraccion_cell = nbformat.v4.new_code_cell(EXTRACCION_SRC)
        nb.cells.append(extraccion_cell)
        out_cell = client.execute_cell(
            extraccion_cell, len(nb.cells) - 1, store_history=False
        )

        # Concatenar TODOS los chunks de stdout de la celda, en orden: un
        # print() grande puede llegar troceado en varios outputs separados
        # (buffering de ipykernel), y quedarnos solo con el primer chunk que
        # contenga el marcador de inicio puede cortar el JSON a la mitad.
        textos = [o.get("text", "") for o in out_cell.get("outputs", []) if o.get("text")]
        payload_texto = "".join(textos) if textos else None

    if payload_texto is None:
        raise RuntimeError(
            "La celda de extracción no produjo el JSON esperado. "
            "Revisa la salida del kernel."
        )

    inicio = payload_texto.index("###GATE_JSON_START###") + len("###GATE_JSON_START###")
    fin = payload_texto.index("###GATE_JSON_END###")
    return json.loads(payload_texto[inicio:fin].strip())


def comparar(actual: dict, baseline: dict, ruta: str = "") -> list[str]:
    """Comparación recursiva; cualquier desviación distinta de cero se reporta."""
    diffs: list[str] = []
    if isinstance(baseline, dict):
        if not isinstance(actual, dict):
            return [f"{ruta}: tipo distinto (baseline=dict, actual={type(actual).__name__})"]
        claves = set(baseline) | set(actual)
        for k in sorted(claves):
            sub_ruta = f"{ruta}.{k}" if ruta else k
            if k not in actual:
                diffs.append(f"{sub_ruta}: presente en baseline, ausente en actual")
            elif k not in baseline:
                diffs.append(f"{sub_ruta}: presente en actual, ausente en baseline")
            else:
                diffs.extend(comparar(actual[k], baseline[k], sub_ruta))
    elif isinstance(baseline, list):
        if not isinstance(actual, list):
            return [f"{ruta}: tipo distinto (baseline=list, actual={type(actual).__name__})"]
        if len(baseline) != len(actual):
            diffs.append(f"{ruta}: longitud distinta (baseline={len(baseline)}, actual={len(actual)})")
        for i, (a, b) in enumerate(zip(actual, baseline)):
            diffs.extend(comparar(a, b, f"{ruta}[{i}]"))
    elif isinstance(baseline, float) or isinstance(actual, float):
        if float(actual) != float(baseline):
            diffs.append(f"{ruta}: baseline={baseline} actual={actual} (Δ={float(actual) - float(baseline):+.4f})")
    else:
        if actual != baseline:
            diffs.append(f"{ruta}: baseline={baseline!r} actual={actual!r}")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compara contra reports/baseline_numeros.json en vez de (re)generarlo.",
    )
    args = parser.parse_args()

    print(f"Ejecutando {NOTEBOOK.relative_to(RAIZ)} en kernel limpio (nbclient)...")
    resultado = ejecutar_notebook(NOTEBOOK)

    if args.check:
        if not BASELINE.exists():
            print(f"No existe {BASELINE.relative_to(RAIZ)}. Ejecuta sin --check primero.", file=sys.stderr)
            return 2
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        diffs = comparar(resultado, baseline)
        if diffs:
            print(f"\n✘ GATE FALLIDO — {len(diffs)} diferencia(s) contra el baseline:\n")
            for d in diffs:
                print(f"  - {d}")
            print("\nNo se ha tocado el baseline. Si el refactor es correcto, esto no debería pasar nunca.")
            return 1
        print("\n✔ GATE OK — cero diferencias contra el baseline.")
        return 0

    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nBaseline escrito en {BASELINE.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
