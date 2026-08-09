# Decisiones de calidad de datos

Justificaciones de decisiones de calidad que viven (o vivían) solo como
traza de exploración en un notebook de `sandbox/`, y que documentan cierres
ya tomados — se extraen aquí para que sobrevivan a la salida del notebook
del repo.

## Umbral de plausibilidad `UMBRAL_DELTA = 2000` (indicador 1293, demanda real a 5 min)

**Origen:** `sandbox/analisis_datos.ipynb`, sección *"Filtro de plausibilidad
delta_5min sobre el crudo de 1293, antes del resample horario"* (celdas 36-47).

### Contexto

Sobre la serie cruda a 5 minutos del indicador 1293 (demanda real peninsular,
ventana de muestra 2026-05-28 a 2026-06-28, 9.215 filas) se calcula
`delta_5min = df_raw["indicador_1293"].diff()` y se examina su distribución
para fijar un umbral que separe saltos de demanda físicamente plausibles de
artefactos de la serie (huecos, duplicados, errores de telemetría).

```
count    9215.000000
mean        0.155182
std       222.357724
min      -767.000000
25%      -134.000000
50%        -8.000000
75%       126.000000
max      8556.000000
```

### Por qué no un umbral por percentil

La idea inicial (anotada en el notebook como candidata) era usar el
percentil 99.9 de `|delta_5min|` como umbral. En la ventana de muestra ese
percentil vale **763 MW** (`quantile([0.001, 0.999])` → -617.5 / 763.3).
Se descartó: un umbral de 763 MW habría marcado como implausibles saltos que,
verificados fila a fila, resultan ser rampas reales de demanda matutina en
día laborable (ver abajo) — falsos positivos sobre comportamiento físico
normal, no sobre corrupción de datos.

### Verificación fila a fila de los saltos más grandes

Inspeccionando los 15 mayores `delta_5min` positivos:

| fila | delta_5min | fecha/hora (UTC) | interpretación |
|---|---|---|---|
| 4284 | **8.556 MW** | — | artefacto: un orden de magnitud por encima de cualquier otro salto de la muestra; no hay rampa de demanda real que salte 8,5 GW en 5 minutos |
| 3580 | **1.420 MW** | 2026-06-09 08:20 UTC (martes, laborable) | **salto físico verificado**: rampa de demanda de mañana en día laborable (indicador 1293 pasa a 34.215 MW) |
| 2123 | 1.028 MW | 2026-06-04 06:55 UTC (jueves, laborable) | mismo patrón: rampa matutina laborable (indicador 1293 en 33.113 MW) |

El salto de 8.556 MW no tiene una fila vecina con una magnitud comparable
(el siguiente en la lista ya es 1.420 MW) — es un valor aislado, consistente
con un artefacto de telemetría y no con una dinámica de demanda real.

### Umbral confirmado

```python
UMBRAL_DELTA = 2000  # simétrico: se aplica a |delta_5min|
```

Justificación cerrada: **por encima del máximo salto físico verificado
(1.420 MW, rampa de mañana laborable)**, y **un orden de magnitud por debajo
del artefacto (8.556 MW)**. Sobre la ventana de muestra, este umbral marca
exactamente **1 fila** como implausible (`salto_implausible.sum() == 1`) — la
del artefacto de 8.556 MW — sin falsos positivos sobre las rampas matutinas
reales.

## Pendiente — verificación de reproducibilidad del pipeline AEMET

`API_aemet.ipynb`, `API_aemet_datos_faltantes.ipynb` y `merge_temperatura.ipynb`
quedan publicados **sin salidas**: requieren credencial de AEMET, y ejecutarlos
sobrescribiría `data/processed/df_final_zonas_temperaturas_2023_2026.parquet`.

Queda sin verificar si ese parquet se regenera idéntico desde el pipeline
actual (a diferencia de `demanda_horaria.parquet`, que sí se comprobó byte a
byte). Comprobación pendiente para la próxima vez que haga falta reejecutar
el pipeline AEMET: copiar el parquet actual fuera del repo antes de ejecutar,
y comparar el nuevo contra el copiado — shape, columnas, dtypes y md5, igual
que se hizo con `demanda_horaria.parquet`.
