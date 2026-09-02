# Spanish Electricity Demand — Forecasting Model & Daily Pipeline

Red Eléctrica de España (REE) publishes two things every hour: the electricity actually consumed and the volume it had forecast. This project builds an independent forecasting model on that public data, measures it against baselines, and runs it daily in production.

> **Status: evaluation phase.** Two models emit a forecast every day. The comparison between them is open and is governed by criteria registered in writing before any production data was seen.

---

## Results on historical data

Mean absolute error, in megawatts. All figures measured on the **same 4,336 hours of H1 2026**, never seen during training.

| | MAE | What it is |
|---|---|---|
| Calendar means | 2,793 MW | The starting point: "a Tuesday in March at 08:00 consumes, on average, this" |
| **Persistence** | **1,840 MW** | The actual benchmark: "today at this hour consumes the same as yesterday at this hour" |
| **Model v1** | **1,263 MW** | Calendar + demand at the same hour yesterday (`lag_24`) — **31 % over persistence** |
| **Model v2** | **1,005 MW** | Adds demand at the same hour one week earlier (`lag_168`) — 45 % over persistence |
| REE official forecast | 266 MW | Context, not a target — see *Limits* |

The system operator's forecast does not constitute a valid comparison reference, as it operates under a different scope and different information sources from those available to this project. The model is therefore evaluated using persistence as its baseline.

**Split:** 2023–2024 train · 2025 validation · H1 2026 test. Always past → future; no model is trained on data later than what it predicts.

**Rationale for the dual model (v1 and v2):** Both versions were evaluated by temporal validation with five sliding partitions (rolling validation), producing a technical tie (differences of 10–14 MW against a method noise floor of ~20 MW). On the 2026 test set, however, v2 outperforms v1 by 258 MW and shows a significantly less skewed error distribution. Since the standard validation protocol failed to discriminate this behaviour, the decision was to keep and deploy both models in parallel.

---

## Production

Automated since **14 August 2026** (GitHub Actions at ~05:45 UTC): emits a daily prediction, evaluates it against REE's closing data in separate commits, and freezes artefacts with no retraining.

* **Published vs. diagnostic:** Hours that have already elapsed when the forecast is issued are recorded for diagnosis only and excluded from the metric. Active window: **16 hours per day**.
* **Current status:** Live metric in [`reports/estado_pipeline.md`](reports/estado_pipeline.md).

| Model | MAE (30d) | Mean bias | Published hours | Dates |
|---|---|---|---|---|
| v1 | 1,995 MW | +1,359 MW | 274 | 19 |
| v2 | 1,677 MW | +1,204 MW | 147 | 11 |

*Note: different windows and counts (not a like-for-like comparison until October).*

### Why production MAE exceeds the notebook's
1. **Week start:** With no day type for *D−1* as an anchor, the `non-working → working` transition (Mondays) raises the MAE to 5,651 MW (vs. 1,153 MW for `working → working`).
2. **Extrapolation ceiling:** Decision trees do not extrapolate beyond their training range (e.g. v1 capped at 38,861 MW against real demand above 40,000 MW in August).

---

## Tested and discarded

* **Temperature (AEMET):** Improved validation (986 to 968 MW), but using the observed value instead of the weather forecast implied information leakage.
* **Gradient boosting:** Better fit in training (795 MW) but worse generalisation (1,686 MW), showing that the non-modellable residual error is pure noise.
* **Bridge days:** 7 cases in 3 years (~168 hours out of 26,000); the model ignored the variable for lack of sample volume.

---

## Limits

* **Structural bias:** Generalised underestimation (+755 MW for v1, +413 MW for v2 on test) against a 2026 with higher average demand.
* **Statistical discrepancy:** The year-on-year growth computed here differs from REE's official press releases in certain months. The aggregation method reproduces the published figures for the training years, so the divergence is confined to 2026. Cause undetermined; documented as an open limit.

---

## Data & stack

* **Sources:** e·sios API (REE demand and official forecast, 2023–2026), AEMET OpenData and working calendars.
* **Stack:** Python, pandas, scikit-learn, Plotly. Decision tree chosen for explainability.
* **Quality control:** Ingestion schema validation and a numeric regression *gate* that blocks changes if any of 17 key reference metrics move.

---

## Repository

```text
data/raw/           raw downloads (not versioned)
data/processed/     clean series
data/               daily pipeline history
src/                clients, validation and utilities
notebooks/          analysis and modelling
pipeline/           daily prediction/evaluation scripts
reports/            dynamic pipeline status
scripts/            numeric regression gate
modelos/            serialised artefacts and reference snapshot
tests/              unit and parity tests
.github/            CI/CD automation
```

Design decisions not covered here are documented in the notebook. For anything further, get in touch.

**Licence:** MIT. The code is free; source data keeps the conditions of its providers, which require attribution.

---
---

# Demanda eléctrica española — modelo de previsión y pipeline diario

Red Eléctrica de España (REE) publica cada hora dos cosas: la electricidad realmente consumida y la que había previsto consumir. Este proyecto construye un modelo propio de previsión sobre esos datos públicos, lo mide contra baselines y lo ejecuta a diario en producción.

> **Estado: fase de evaluación.** Dos modelos emiten previsión cada día. La comparación entre ambos está abierta y la gobiernan criterios registrados por escrito antes de ver ningún dato de producción.

---

## Resultados sobre datos históricos

Error absoluto medio, en megavatios. Todas las cifras medidas sobre las **mismas 4.336 horas del primer semestre de 2026**, nunca vistas en entrenamiento.

| | MAE | Qué es |
|---|---|---|
| Medias por calendario | 2.793 MW | El punto de partida: «un martes de marzo a las 8:00 se consume, de media, esto» |
| **Persistencia** | **1.840 MW** | La vara de medir real: «hoy a esta hora se consume lo mismo que ayer a esta hora» |
| **Modelo v1** | **1.263 MW** | Calendario + demanda de ayer a la misma hora (`lag_24`) — **31 % sobre persistencia** |
| **Modelo v2** | **1.005 MW** | Añade la demanda de hace una semana a la misma hora (`lag_168`) — 45 % sobre persistencia |
| Previsión oficial de REE | 266 MW | Contexto, no objetivo — ver *Límites* |

La previsión del operador del sistema no constituye una referencia válida de comparación, ya que opera bajo un alcance y unas fuentes de información distintas a las de este proyecto. Por este motivo, el modelo se evalúa utilizando la persistencia como baseline.

**Partición:** 2023–2024 train · 2025 validación · 1S 2026 test. Siempre pasado → futuro; ningún modelo se entrena con datos posteriores a los que predice.

**Justificación de la doble modelización (v1 y v2):** Ambas versiones se evaluaron mediante validación temporal con cinco particiones deslizantes (rolling validation), obteniendo un empate técnico (diferencias de 10–14 MW frente a un ruido de método de ~20 MW). Sin embargo, en el conjunto de test de 2026, v2 supera a v1 en 258 MW y presenta una distribución de errores significativamente menos sesgada. Dado que el protocolo de validación tradicional no logró discriminar este comportamiento, se ha optado por mantener y desplegar ambos modelos en paralelo.

---

## Producción

Automatizado desde el **14 de agosto de 2026** (GitHub Actions a ~05:45 UTC): emite predicción diaria, evalúa con el cierre real de REE en commits separados y congela artefactos sin reentrenamiento.

* **Publicado vs. Diagnóstico:** Las horas transcurridas antes de emitir la previsión se registran solo para diagnóstico y se excluyen del cálculo. Ventana activa: **16 horas al día**.
* **Estado actual:** Métrica en [`reports/estado_pipeline.md`](reports/estado_pipeline.md).

| Modelo | MAE (30d) | Sesgo medio | Horas publicadas | Fechas |
|---|---|---|---|---|
| v1 | 1.995 MW | +1.359 MW | 274 | 19 |
| v2 | 1.677 MW | +1.204 MW | 147 | 11 |

*Nota: Ventanas y recuentos distintos (comparación no homogénea hasta octubre).*

### Por qué el MAE de producción supera al del notebook
1. **Arranque de la semana:** Sin el tipo de día de *D−1* como ancla, la transición `no laborable → laborable` (lunes) eleva el MAE a 5.651 MW (vs. 1.153 MW en `laborable → laborable`).
2. **Techo de extrapolación:** Los árboles de decisión no extrapolan fuera de rango (ej. v1 limitado a 38.861 MW con demandas reales >40.000 MW en agosto).

---

## Probado y descartado

* **Temperatura (AEMET):** Mejoraba la validación (986 a 968 MW), pero usar la observación real en lugar de la previsión meteorológica implicaba una fuga de información.
* **Gradient boosting:** Mejor ajuste en entrenamiento (795 MW) pero peor generalización (1.686 MW), evidenciando que el error residual no modelable es ruido puro.
* **Días puente:** 7 casos en 3 años (~168 horas de 26.000); el modelo ignoró la variable por falta de volumen muestral.

---

## Límites

* **Sesgo estructural:** Subestimación generalizada (+755 MW en v1, +413 MW en v2 en test) ante un 2026 con mayor demanda media.
* **Discrepancia estadística:** El crecimiento interanual calculado difiere de las notas oficiales de REE en ciertos meses. El método de agregación reproduce las cifras publicadas en los años de entrenamiento, luego la divergencia se circunscribe a 2026. Causa sin determinar; documentado como límite abierto.

---

## Datos y stack

* **Fuentes:** API e·sios (demanda y previsión oficial de REE, 2023–2026), AEMET OpenData y calendarios laborales.
* **Stack:** Python, pandas, scikit-learn, Plotly. Árbol de decisión por explicabilidad.
* **Control de calidad:** Validación de esquemas de ingesta y *gate* de regresión numérica que bloquea cambios si varían 17 métricas clave de referencia.

---

## Repositorio

```text
data/raw/           descargas crudas (no versionado)
data/processed/     series limpias
data/               histórico del pipeline diario
src/                clientes, validación y utilidades
notebooks/          análisis y modelado
pipeline/           scripts diarios de predicción/evaluación
reports/            estado dinámico del pipeline
scripts/            gate de regresión numérica
modelos/            artefactos serializados y foto de referencia
tests/              tests unitarios y de paridad
.github/            automatización CI/CD
```

Las decisiones de diseño que no aparecen aquí están documentadas en el notebook. Para cualquier cosa más allá de eso, contacto directo.

**Licencia:** MIT. El código es libre; los datos de origen conservan las condiciones de sus proveedores, que exigen reconocimiento de la fuente.
