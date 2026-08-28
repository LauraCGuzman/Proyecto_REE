<!-- GENERADO POR pipeline/evaluar.py -- NO EDITAR A MANO -->

# Estado del pipeline REE

## Modelo: v1

Última corrida: 2026-08-28T17:28:50Z
Fechas presentes en `data/errores.csv` para este modelo (publicadas + diagnóstico): 16

### Métrica publicada (`h_adelanto_h > 0`)

`Fechas cubiertas` cuenta fechas de calendario distintas con al menos una hora publicada en la ventana: es la magnitud que decide si el MAE se escribe o se muestra `—`. `Span (días)` es la distancia en días entre el primer y el último horizonte publicado de la ventana; se conserva como dato informativo, pero no gobierna nada. Es menor que `Fechas cubiertas` cuando las fechas son contiguas (el span mide días transcurridos entre extremos, no fechas contadas), y mayor cuando hay huecos entre ellas (fechas dispersas en el tiempo estiran el span sin sumar fechas cubiertas).

| Ventana | MAE (MW) | Sesgo medio (MW) | n horas | Fechas cubiertas (gobierna) | Span (días) | MAE test notebook (MW) |
|---|---|---|---|---|---|---|
| 7d | 1553.48 | +1276.59 | 89 | 7 | 6.1 | 1263.02 |
| 30d | 1836.56 | +1081.53 | 212 | 14 | 13.6 | 1263.02 |
| 90d | 1836.56 | +1081.53 | 212 | 14 | 13.6 | 1263.02 |

### Por qué el MAE de producción no coincide con el del notebook

El MAE de producción de la tabla de arriba no es directamente comparable al 1263.02 MW medido en el conjunto de test del notebook (`notebooks/modelo_demanda.ipynb`, celda 28). Dos limitaciones conocidas y diagnosticadas del modelo -- no fallos del pipeline -- explican buena parte de la diferencia:

- **Arranque de la semana.** El modelo condiciona el nivel de la predicción en `tipo_efectivo(D)` y en `demanda_lag_24`, pero nunca en `tipo_efectivo(D-1)`: no sabe de qué tipo de día viene su punto de partida. El régimen de transición `no laborable → laborable` (los lunes) es donde más se nota, y toda ventana de 7 días contiene exactamente un lunes -- esta limitación es estado estacionario de cualquier ventana de producción, no un transitorio que vaya a desaparecer con más datos.
- **Techo de extrapolación.** El modelo (`DecisionTreeRegressor`) no extrapola por encima del rango de entrenamiento: su predicción máxima estaba clavada en 38.861,1 MW en agosto de 2026, y la demanda real superó los 40.000 MW dos veces esa misma semana del 14/8/2026 -- en esas horas la desviación entre predicción y demanda real es grande por construcción del modelo, no por un fallo del pipeline.

---

## Modelo: v2

Última corrida: 2026-08-28T17:28:50Z
Fechas presentes en `data/errores.csv` para este modelo (publicadas + diagnóstico): 8

### Métrica publicada (`h_adelanto_h > 0`)

`Fechas cubiertas` cuenta fechas de calendario distintas con al menos una hora publicada en la ventana: es la magnitud que decide si el MAE se escribe o se muestra `—`. `Span (días)` es la distancia en días entre el primer y el último horizonte publicado de la ventana; se conserva como dato informativo, pero no gobierna nada. Es menor que `Fechas cubiertas` cuando las fechas son contiguas (el span mide días transcurridos entre extremos, no fechas contadas), y mayor cuando hay huecos entre ellas (fechas dispersas en el tiempo estiran el span sin sumar fechas cubiertas).

| Ventana | MAE (MW) | Sesgo medio (MW) | n horas | Fechas cubiertas (gobierna) | Span (días) |
|---|---|---|---|---|---|
| 7d | — | — | 85 | 6 | 5.6 |
| 30d | — | — | 85 | 6 | 5.6 |
| 90d | — | — | 85 | 6 | 5.6 |

> **Muestra insuficiente todavía.** Ninguna ventana alcanza las 7 fechas de calendario distintas con al menos una hora publicada que exige la cobertura mínima -- el MAE se muestra como `—` (`null` en `data/metricas.json`) porque lo que no se puede afirmar todavía no se escribe.

### Referencia de notebook

Este modelo no tiene todavía una referencia de notebook para el propio artefacto serializado -- sin columna en la tabla de arriba a propósito, para no invitar a compararla con la de v1. v2 no tiene una referencia de notebook para el modelo SERIALIZADO (entrena con validación incluida, 2023-2025 completo): no hay una medida en validación posible para ese modelo. El mae_val/sesgo_val de modelos/modelo_v2.json es de un árbol de comparación distinto, entrenado solo con 2023-2024 -- se queda documentado ahí, con su propia nota, y no se repite aquí para no sugerir que es del serializado.

No hay tampoco una sección de "limitaciones diagnosticadas" propia de este modelo todavía: las de v1 (arranque de la semana, techo de extrapolación) son observaciones medidas sobre su árbol, no se asumen iguales aquí sin medirlas -- queda para cuando la ventana de seis semanas (pliego Fase5bis, 0.1) dé señal suficiente.

