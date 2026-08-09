# Previsión de demanda eléctrica española y detección de anomalías

## Qué hace este proyecto

Red Eléctrica de España publica cada hora dos cosas: cuánta electricidad se ha consumido de verdad, y cuánta habían previsto que se iba a consumir. Este proyecto construye un modelo propio de previsión, lo compara con el oficial, y usa las diferencias entre lo previsto y lo real para detectar comportamientos raros en la red.

La idea de fondo: si tienes un modelo que sabe cuánta demanda *debería* haber a una hora determinada, entonces las horas en las que la realidad se aparta mucho de esa expectativa son horas que merece la pena mirar. Un apagón, un fallo de telemetría, un festivo mal etiquetado, un episodio industrial.

---

## El resultado, de un vistazo

Error medio de previsión, en megavatios. Menos es mejor.

| | Error medio | Qué es |
|---|---|---|
| Tabla de medias por calendario | 2.793 MW | El punto de partida: "un martes de marzo a las 8 de la mañana, de media, se consume esto" |
| **Modelo A** | **1.263 MW** | Añade el consumo de ayer a la misma hora |
| **Modelo B** | **1.005 MW** | Añade además el consumo de hace una semana a la misma hora |
| Previsión oficial de REE | 266 MW | El techo. Ver la nota más abajo |

Todos los números están medidos sobre el mismo conjunto de horas del primer semestre de 2026: **4.336 horas** que el modelo no había visto nunca.

Pasar de 2.793 a 1.005 significa reducir el error a poco más de un tercio. La previsión oficial de REE sigue siendo cuatro veces mejor, y eso no es un fracaso: es un dato sobre lo que hace falta para llegar ahí.

---

## Cómo se ha medido, y por qué importa

Esta es la parte que decide si los números anteriores valen algo.

Los datos se parten en tres trozos, y cada trozo tiene un trabajo distinto:

- **2023–2024 — entrenar.** El modelo aprende de aquí.
- **2025 — elegir.** Aquí se prueban las distintas configuraciones posibles y se escoge la mejor. El modelo no aprende de estas horas, solo se usan para decidir.
- **Primer semestre de 2026 — medir.** Se toca **una sola vez por modelo**, al final. Es la única medida honesta de cómo funcionaría el modelo con datos que no existían cuando se construyó.

El orden es siempre pasado → futuro. Nunca se entrena con datos posteriores a los que se predicen, porque eso daría un resultado buenísimo y completamente falso.

**Este proyecto tuvo ese error y lo arregló.** En una versión anterior, la elección de configuración se hacía mirando el trozo de 2026. Eso hacía que los números salieran mejor de lo que eran. Al separar "elegir" de "medir", el error del modelo A subió de 1.244 a 1.263 MW. Esos 19 megavatios son la diferencia entre un número bonito y un número cierto.

Hay un candado en el código que impide leer el trozo de 2026 dos veces para el mismo modelo. Si se intenta, el programa se detiene.

---

## Por qué se publican dos modelos y no uno

Lo normal sería quedarse con el que da mejor número. Aquí no se puede, y explicar por qué es probablemente lo más interesante del proyecto.

**Modelo A** usa el consumo de ayer a la misma hora.
**Modelo B** añade el consumo de hace exactamente una semana a la misma hora.

La razón para probar el B: el consumo de un lunes se parece más al del lunes anterior que al del domingo. Sonaba prometedor.

Y aquí está el problema. Los dos modelos se compararon **cinco veces distintas** usando el trozo de 2025, el que sirve para elegir. En las cinco salieron empatados. Diferencias de 10 a 14 MW cuando el propio método tiene un margen de error de unos 20. Es decir: indistinguibles.

Pero al medirlos sobre 2026, el modelo B es **258 MW mejor** y además su error está mucho menos desviado hacia un lado.

**No sé por qué.** Y lo que hay que hacer con eso es decirlo, no esconderlo.

Si publico solo el A, estoy tirando una mejora real porque mi método de selección no la vio venir. Si publico solo el B, lo estoy eligiendo porque gana en el trozo que no debía usarse para elegir — exactamente el error que este proyecto arregló hace dos semanas.

Así que van los dos, con la contradicción encima de la mesa.

### Lo que se intentó para explicarla

Mi primera hipótesis: 2026 consume bastante más que los años de entrenamiento (unos 2.237 MW más de media). Quizá el modelo B es mejor precisamente cuando el nivel general de consumo sube, y el trozo de 2025 no podía detectarlo porque su nivel es parecido al de 2023–2024.

Para probarlo, repetí la comparación en cuatro periodos distintos, cada uno con su propio salto de nivel:

| Periodo evaluado | El consumo sube | ¿Gana el modelo B? |
|---|---|---|
| Primer semestre 2024 | Baja 196 MW | **Sí, por 28 MW** |
| Segundo semestre 2025 | Sube 1.101 MW | No, empate |
| Segundo semestre 2024 | Sube 739 MW | No, empate |
| Primer semestre 2025 | Sube 590 MW | No, empate |

**La hipótesis es falsa.** El modelo B gana justo en el único periodo donde el consumo *baja*, y en los tres donde sube da igual, sin importar cuánto suba. Si mi explicación fuera correcta, el periodo con el salto de 1.101 MW tendría que destacar. No destaca.

Así que la contradicción entre 2025 y 2026 sigue sin explicación. Queda escrita como pregunta abierta.

---

## Lo que no funcionó

Tres cosas se probaron a fondo y no sirvieron. Están aquí porque el proyecto no va de acertar, va de averiguar.

### La temperatura no ayuda

Construí una tubería de datos completa con **8 estaciones meteorológicas** de AEMET repartidas por la península, agrupadas en cuatro zonas climáticas y ponderadas por población. Funciona bien y fue bastante trabajo.

Y no sirve de nada. El error empeora ligeramente al añadirla (1.374 frente a 1.263 MW).

La explicación es que la temperatura sí afecta al consumo — se ve clarísimo en los datos, con más gasto en frío y en calor — pero **esa información ya está dentro del consumo de ayer**. Si ayer hizo frío y se consumió mucho, hoy probablemente también hará frío. El dato de temperatura no añade nada que el consumo de ayer no dijera ya.

Está medido tres veces con tres métodos distintos. En la última, las ocho columnas de temperatura juntas explican un 2 % de las decisiones del modelo.

### Los modelos más sofisticados tampoco

Probé *gradient boosting*, una técnica bastante más potente que el árbol de decisión que uso. Aprende los datos de entrenamiento mucho mejor (error de 795 MW) y falla igual o peor con datos nuevos (1.686 MW).

Ese desajuste tiene un nombre: memorizar en vez de aprender. Y lo que dice es que la parte del consumo que no explican el calendario y los consumos anteriores es en gran medida **ruido**. No hay más señal que sacar por ahí. Un modelo más complejo no la encuentra porque no está.

### Los puentes: no hay datos suficientes

Tenía una variable que marca los días puente. El modelo **nunca la usó** — literalmente cero decisiones basadas en ella, en cinco mediciones independientes.

Ojo con la lectura, porque es fácil equivocarse aquí: no es que los puentes no afecten al consumo. Es que en tres años solo hay **7 puentes**, unas 168 horas sobre 26.000. No hay casos suficientes para que el modelo aprenda nada de ellos.

Es una distinción importante. "No hay efecto" y "no hay datos para verlo" se parecen en el resultado y son cosas muy distintas.

---

## Lo que aprendí sobre medir

Este es el hallazgo que más me va a servir en el futuro, y no tiene nada que ver con la electricidad.

Al principio comparaba los modelos de una sola forma: entrenar con 2023–2024, evaluar con 2025. Un número, una conclusión.

Cuando repetí la misma comparación en cuatro periodos distintos, el resultado del modelo B iba **de 28 MW mejor a 14 MW peor** según qué periodo eligiera.

Es decir: mi método de comparación tenía un margen de error de unos 20 MW, y yo estaba tomando decisiones basadas en diferencias de 15. **Estaba midiendo ruido y llamándolo resultado.** Y no había forma de saberlo con un solo corte, porque un solo número no viene con su propio margen de error.

De esto salió una regla que ahora aplico siempre: **si la mejora es más pequeña que la dispersión del método, no es una mejora.** Fue lo que descartó la temperatura y lo que impidió dar por bueno el modelo B en validación.

---

## Detección de anomalías

La segunda mitad del proyecto. Con un modelo de previsión funcionando, las horas donde la realidad se aparta mucho de lo esperado son candidatas a investigar.

El método actual es deliberadamente simple: se calcula, para cada combinación de tipo de día y hora, cuánto se desvía normalmente la realidad de la previsión, y se marcan las horas que se salen de ese patrón habitual.

Sobre los casos encontrados hasta ahora:

- **El apagón del 28 de abril de 2025** aparece, como era de esperar.
- **Un episodio del 11 de junio** que parecía una anomalía de consumo resultó ser un **fallo de telemetría**: datos corruptos, no comportamiento raro de la red. Esas 8 horas están excluidas de todas las mediciones de este README.
- El primer grupo de anomalías que investigué a fondo resultó ser también un artefacto de los datos, no un evento real. Es lo normal: la mayor parte de lo que un detector marca al principio son problemas de calidad de datos.

---

## Límites, y cosas que no se ven en los números

Un proyecto sin esta sección no es honesto.

**La comparación con REE no es del todo justa, y no en mi favor.** La previsión oficial se actualiza cada cinco minutos con información fresca. La mía se calcula una vez, a las 00:00 del día que predice. Comparar 266 con 1.005 MW es comparar dos cosas distintas: el número de REE no es un objetivo alcanzable para una previsión a un día.

**Mi modelo asume que el día anterior se conoce completo.** Usa el consumo de ayer a la misma hora, lo que implica que a las 00:00 ya tengo todas las horas de ayer. Es defendible, pero REE emite su previsión antes de eso. Con un margen de emisión más realista, el modelo necesitaría una arquitectura distinta.

**El trozo de 2026 se leyó cuatro veces durante la exploración del modelo B**, tres de ellas con una configuración fijada a mano en vez de elegida en validación. Eso significa que el 1.005 MW tiene algo menos de valor como "dato nunca visto" que el 1.263 del modelo A. La cifra es correcta; su estatus es ligeramente peor. Se dice.

**Durante el apagón de abril de 2025, la previsión oficial de REE se fue a cero durante unas 35 horas.** No se puede comparar un modelo contra una previsión que no existe, así que ese tramo queda fuera del benchmark.

**Los dos modelos arrastran el mismo problema de fondo:** su error está desviado hacia un lado. Predicen de media 755 MW (modelo A) y 413 MW (modelo B) por debajo de la realidad de 2026. La causa es que 2026 consume más que cualquier año de entrenamiento, y este tipo de modelo no sabe extrapolar: no puede predecir un valor que nunca ha visto. No es un problema de ajuste, es una limitación de la herramienta.

---

## Datos y stack

**Fuentes**
- Red Eléctrica de España, API e·sios: consumo real y previsión oficial, resolución de 5 minutos agregada a horaria, enero 2023 – junio 2026.
- AEMET OpenData: temperaturas diarias de 8 estaciones de aeropuerto.
- Calendario laboral nacional y autonómico, con festivos y puentes.

**Herramientas**
Python, pandas, scikit-learn, plotly. Sin dependencias exóticas: el modelo es un árbol de decisión, y la elección es deliberada — se puede explicar entero.

**Control de calidad**
La ingesta valida el número de filas esperadas antes de guardar nada, distingue huecos reales de datos corruptos, y se detiene si algo no cuadra en vez de continuar en silencio. Varios de los hallazgos de este proyecto salieron de comprobaciones que fallaron cuando debían.

---

## Estructura del repositorio

```
data/          crudo, intermedio y procesado (no versionado)
src/           clientes de API y validación de calidad
notebooks/     modelo de demanda y detección de anomalías
```

---

*Fase 1 de tres. La Fase 2 usará este modelo para proyectar tensiones en la red bajo escenarios de shock geopolítico y climático.*