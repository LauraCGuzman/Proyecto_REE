# Previsión de demanda eléctrica española y detección de anomalías

## Qué hace este proyecto

Red Eléctrica de España publica cada hora dos cosas: cuánta electricidad se ha consumido de verdad, y cuánta habían previsto que se iba a consumir. Este proyecto construye un modelo propio de previsión, y lo compara con el oficial.

---

## El resultado, de un vistazo

Error medio de previsión, en megavatios. Menos es mejor.

| | Error medio | Qué es |
|---|---|---|
| Tabla de medias por calendario | 2.793 MW | El punto de partida: "un martes de marzo a las 8 de la mañana, de media, se consume esto" |
| Persistencia | 1.840 MW | La referencia dura: "hoy a esta hora se consumirá lo mismo que ayer a esta hora" |
| **Modelo A** | **1.263 MW** | Calendario más el consumo de ayer a la misma hora |
| **Modelo B** | **1.005 MW** | Añade además el consumo de hace una semana a la misma hora |
| Previsión oficial de REE | 266 MW | No es el objetivo. Ver la sección de límites |

Todos los números están medidos sobre el mismo conjunto de horas del primer semestre de 2026: **4.336 horas** que el modelo no había visto nunca.

---

## Cómo se ha medido

Los datos se parten en tres trozos, y cada trozo tiene un trabajo distinto:

- **2023–2024 — train.** 
- **2025 — evaluate** 
- **P 2026 — test**

El orden es siempre pasado → futuro. Nunca se entrena con datos posteriores a los que se predicen, porque eso daría un resultado buenísimo y completamente falso.


---

## Por qué se publican dos modelos y no uno

Lo normal sería quedarse con el que da mejor número.

**Modelo A** usa el consumo de ayer a la misma hora.
**Modelo B** añade el consumo de hace exactamente una semana a la misma hora.

La razón para probar el B: el consumo de un lunes se parece más al del lunes anterior que al del domingo. Sonaba prometedor.

Y aquí está el problema. Los dos modelos se compararon **cinco veces**, siempre sin tocar 2026: una vez con el corte principal (entrenar con 2023–2024, decidir con 2025) y cuatro veces más moviendo ese corte hacia atrás, de forma que cada comparación entrena con menos historia y evalúa en un semestre distinto de 2024 o 2025. En las cinco salieron empatados: diferencias de 10 a 14 MW cuando el propio método tiene un margen de error de unos 20. Es decir, indistinguibles.

Pero al medirlos sobre 2026, el modelo B es **258 MW mejor** y además su error está mucho menos desviado hacia un lado.

---

## Lo que se quedó fuera

Tres cosas se probaron a fondo y no entraron en el modelo publicado. Están aquí porque el proyecto no va de acertar, va de averiguar. La primera es la que más me enseñó, y no por lo que parece.

### La temperatura ayudaba, y la quité

Construí una tubería de datos completa con **8 estaciones meteorológicas** de AEMET repartidas por la península, agrupadas en cuatro zonas climáticas y ponderadas por población.

Y funciona. En validación baja el error de 986 a 968 MW: una mejora de 18 MW cuando el ruido del método es de 12. Pasa el criterio.

La columna que construí es temperatura **observada**, la que se conoce *después*. Un modelo que predice el día siguiente a las 00:00 tiene la previsión meteorológica, que trae su propio error encima. Medir con la observada y presentar el resultado como rendimiento en producción es fuga de información.

Para reconstruirlo bien haría falta el archivo de previsiones meteorológicas, no el de observaciones. La API pública de AEMET sirve lo segundo, no lo primero.


### Los modelos más sofisticados tampoco

Probé *gradient boosting*, una técnica bastante más potente que el árbol de decisión que uso. Aprende los datos de entrenamiento mucho mejor (error de 795 MW) y falla igual o peor con datos nuevos (1.686 MW).

Ese desajuste tiene un nombre: memorizar en vez de aprender. Y lo que dice es que la parte del consumo que no explican el calendario y los consumos anteriores es en gran medida **ruido**. No hay más señal que sacar por ahí. Un modelo más complejo no la encuentra porque no está.

### Los puentes: no hay datos suficientes

Tenía una variable que marca los días puente. El modelo **nunca la usó**, literalmente cero decisiones basadas en ella, en cinco mediciones independientes.

Es que en tres años solo hay **7 puentes**, unas 168 horas sobre 26.000. No hay casos suficientes para que el modelo aprenda nada de ellos.

---

## Límites, y cosas que no se ven en los números


### La subida de 2026 queda fuera de lo que el histórico permite aprender

Los dos modelos predicen por debajo de la realidad: 755 MW de media el modelo A, 413 el modelo B.

La razón es que 2026 consume unos 2.237 MW más de media que los años de entrenamiento, y un árbol de decisión no extrapola: no puede predecir un valor por encima de todo lo que ha visto. Es comprobable por otra vía: al forzar el árbol más simple, el sesgo no baja, sube. No es un problema de ajuste, es el techo de la herramienta.

Y hay un fleco abierto sobre el tamaño de esa subida. Al agregar mi serie por meses y compararla con las cifras que REE publica en sus notas de prensa, los años de entrenamiento coinciden y 2026 se separa: mi serie crece entre el 3 y el 12 % mes a mes sobre 2025, mientras las cifras publicadas dan −1,1 % en abril y +0,5 % en junio. Junio de 2025 coincide al 0,03 %, así que el método de agregación es correcto y lo que cambia está en 2026.

---

## El modelo en producción

Desde el **14 de agosto de 2026** el Modelo A emite una previsión al día, sin intervención manual, y compara cada predicción con el dato real cuando la red lo publica. Todo lo anterior de este README es un ejercicio medido sobre datos históricos; esta sección es lo que ocurre cuando el mismo modelo tiene que funcionar todos los días.

**Métrica viva, actualizada en cada corrida:** [`reports/estado_pipeline.md`](reports/estado_pipeline.md).

### Cómo funciona

Una acción programada de GitHub se dispara cada mañana en torno a las 05:45 UTC y hace dos cosas, en este orden y en dos commits separados:

1. **Predice** el día natural siguiente al último dato disponible — 23, 24 o 25 horas según el cambio de hora, nunca 24 fijas.
2. **Evalúa** las predicciones anteriores contra el consumo real, para las horas que ya tienen dato cerrado.

Si la evaluación falla, la predicción del día ya está guardada.

El modelo **está congelado**. No se re-entrena, no se ajusta y no se sustituye: el pipeline es infraestructura alrededor de un modelo fijo, y su valor está en que los números que produce no los ha elegido nadie.


### Dos cortes que no se mezclan nunca

Cuando la previsión se emite a las 05:45, unas ocho horas del día que predice ya han pasado. Esas horas se guardan, porque sirven para diagnosticar, pero **no cuentan como previsión** y no se agregan nunca con las demás en un mismo número. Cada fila lleva anotado cuántas horas de antelación tenía, así que el corte se recalcula en cada lectura en vez de quedar congelado en el código.

Las horas realmente publicadas —las que se predijeron antes de que ocurrieran— son 16 al día.

---

---

## Datos y stack

**Fuentes**

- **Red Eléctrica de España**, API e·sios: consumo real y previsión oficial, resolución de 5 minutos agregada a horaria, enero 2023 – junio 2026. Datos de e·sios, elaboración propia.
- **AEMET OpenData**: temperaturas diarias de 8 estaciones de aeropuerto. Información elaborada a partir de datos de la Agencia Estatal de Meteorología (AEMET).
- Calendario laboral nacional y autonómico, con festivos y puentes.

**Herramientas**

Python, pandas, scikit-learn, plotly. Sin dependencias exóticas: el modelo es un árbol de decisión, y la elección es deliberada — se puede explicar entero.

**Control de calidad**

La ingesta valida el número de filas esperadas antes de guardar nada, distingue huecos reales de datos corruptos, y se detiene si algo no cuadra en vez de continuar en silencio. Varios de los hallazgos de este proyecto salieron de comprobaciones que fallaron cuando debían.

Encima de eso hay un **gate de regresión numérica**: un script que reejecuta el notebook en un intérprete limpio, extrae 17 métricas, los hiperparámetros ganadores, las dispersiones y los conteos de filas por etapa, y los compara contra una foto de referencia guardada en el repositorio. Si un número se mueve, falla. Se escribió para poder reorganizar el código sin romper los resultados en silencio, y hace que "los números siguen siendo los mismos" sea una comprobación y no una impresión.

---

## Estructura del repositorio

```
data/raw/        descargas crudas de e·sios y AEMET (no versionado)
data/processed/  series limpias y consolidadas (versionado: git hace de control de cambios)
data/            históricos del pipeline diario: predicciones, errores y anclas usadas
src/             clientes de API, rutas, validación de calidad, utilidades de modelado
notebooks/       modelo de demanda y detección de anomalías
pipeline/        scripts de predicción y evaluación diarias
reports/         estado del pipeline, regenerado en cada corrida
.github/         acción programada que ejecuta el pipeline
scripts/         gate de regresión numérica
modelos/         modelo serializado y foto de referencia de los números
tests/           test de paridad notebook ↔ src, y tests del pipeline
```

---

## Licencia

MIT. El código es libre; los datos de origen conservan las condiciones de sus proveedores, que exigen reconocimiento de la fuente.
