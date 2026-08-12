# src/paths.py
"""Resolución de rutas del proyecto. Antes copiado y pegado (raiz_proyecto +
RAIZ + DIR_RAW + DIR_PROCESSED + el assert) de forma idéntica en cada
notebook (analisis_historico_demanda, analisis_historico_demanda_prevista,
analisis_historico_2_demandas, merge_temperatura, modelo_demanda, API_aemet,
API_aemet_datos_faltantes). Un solo sitio.

Los notebooks siguen necesitando un bootstrap propio y mínimo (encontrar la
raíz y meterla en sys.path) porque hasta que RAIZ esté en sys.path, Python no
puede resolver `from src.paths import ...` — es la única parte que no se
puede centralizar del todo. Bootstrap estándar para cada notebook:

    import sys
    from pathlib import Path

    for candidato in (Path.cwd(), *Path.cwd().resolve().parents):
        if (candidato / "src").is_dir():
            if str(candidato) not in sys.path:
                sys.path.insert(0, str(candidato))
            break

    from src.paths import RAIZ, DIR_RAW, DIR_PROCESSED, DIR_REPORTS, DIR_MODELOS
"""
import sys
from pathlib import Path


def raiz_proyecto(marcador="src") -> Path:
    p = Path.cwd().resolve()
    for candidato in (p, *p.parents):
        if (candidato / marcador).is_dir():
            return candidato
    raise RuntimeError(f"No encuentro '{marcador}' desde {p}")


# RAIZ se deriva de dónde vive este archivo, no del cwd: src/paths.py está
# siempre en <raiz>/src/paths.py, así que parents[1] es determinista pase lo
# que pase desde dónde se importe (notebooks/, sandbox/, la raíz del repo...).
# raiz_proyecto() se queda disponible por si algo la necesita explícitamente,
# pero ya no decide RAIZ.
RAIZ = Path(__file__).resolve().parents[1]
DIR_DATA = RAIZ / "data"
DIR_RAW = DIR_DATA / "raw"
DIR_PROCESSED = DIR_DATA / "processed"
DIR_REPORTS = RAIZ / "reports"
DIR_MODELOS = RAIZ / "modelos"

# data/ está en .gitignore: en un git clone limpio estas carpetas no existen
# todavía, y eso es normal, no un fallo. Crearlas es idempotente. Lo que sí
# indicaría una raíz mal resuelta es que falte el propio marcador 'src' —
# pero eso ya no puede pasar aquí, porque RAIZ se deriva de este archivo.
DIR_RAW.mkdir(parents=True, exist_ok=True)
DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
DIR_REPORTS.mkdir(parents=True, exist_ok=True)
DIR_MODELOS.mkdir(parents=True, exist_ok=True)

if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
