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
| Persistencia | 1.840 MW | La referencia dura: "hoy a esta hora se consumirá lo mismo que ayer a esta hora" |
| **Modelo A** | **1.263 MW** | Calendario más el consumo de ayer a la misma hora |
| **Modelo B** | **1.005 MW** | Añade además el consumo de hace una semana a la misma hora |
| Previsión oficial de REE | 266 MW | No es el objetivo. Ver la sección de límites |

Todos los números están medidos sobre el mismo conjunto de horas del primer semestre de 2026: **4.336 horas** que el modelo no había visto nunca.

La comparación que importa es contra persistencia, porque es la que un modelo tiene que batir para justificar su existencia: **1.840 → 1.263 MW es un 31 % menos de error**. La tabla de calendario sola no llega ni a eso, y la previsión oficial de REE juega en otra liga por razones que explico más abajo.

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

Y aquí está el problema. Los dos modelos se compararon **cinco veces**, siempre sin tocar 2026: una vez con el corte principal (entrenar con 2023–2024, decidir con 2025) y cuatro veces más moviendo ese corte hacia atrás, de forma que cada comparación entrena con menos historia y evalúa en un semestre distinto de 2024 o 2025. En las cinco salieron empatados: diferencias de 10 a 14 MW cuando el propio método tiene un margen de error de unos 20. Es decir, indistinguibles.

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
| Segundo semestre 2024 | Sube 739 MW | No, pierde por 14 MW |
| Primer semestre 2025 | Sube 590 MW | No, pierde por 12 MW |
| Segundo semestre 2025 | Sube 1.101 MW | No, pierde por 10 MW |

**La hipótesis es falsa.** El modelo B gana justo en el único periodo donde el consumo *baja*, y en los tres donde sube pierde de forma plana, sin importar si el salto es de 590 o de 1.101 MW. Si mi explicación fuera correcta, el periodo con el salto de 1.101 MW tendría que destacar. No destaca.

Hay que añadir que la comparación no aísla una sola variable: la versión con el consumo de hace una semana retira además la marca de días puente, trabaja con un censo de horas ligeramente distinto y sale de otra rejilla de hiperparámetros. Para atribuir el efecto a una causa concreta haría falta una comparación controlada que todavía no he hecho.

Así que la contradicción entre 2025 y 2026 sigue sin explicación. Queda escrita como pregunta abierta.

---

## Lo que se quedó fuera

Tres cosas se probaron a fondo y no entraron en el modelo publicado. Están aquí porque el proyecto no va de acertar, va de averiguar. La primera es la que más me enseñó, y no por lo que parece.

### La temperatura ayudaba, y la quité

Construí una tubería de datos completa con **8 estaciones meteorológicas** de AEMET repartidas por la península, agrupadas en cuatro zonas climáticas y ponderadas por población.

Y funciona. En validación baja el error de 986 a 968 MW: una mejora de 18 MW cuando el ruido del método es de 12. Pasa el criterio.

La quité igualmente. La columna que construí es temperatura **observada**, la que se conoce *después*. Un modelo que predice el día siguiente a las 00:00 no tiene eso: tiene la previsión meteorológica, que trae su propio error encima. Medir con la observada y presentar el resultado como rendimiento en producción es fuga de información — el modelo estaría usando algo que en el momento de predecir no existe. Lo que dice esa mejora de 18 MW es cuánto ayudaría *en el mejor caso imaginable*, y ese caso no se da.

Para reconstruirlo bien haría falta el archivo de previsiones meteorológicas, no el de observaciones. La API pública de AEMET sirve lo segundo, no lo primero.

Como referencia del tamaño real del efecto: las ocho columnas juntas pesan un 5,9 % de las decisiones del árbol; con el consumo de hace una semana también dentro, bajan al 2,0 %. Es un efecto pequeño, y buena parte de él ya viene dentro del consumo de ayer: si ayer hizo frío y se consumió mucho, hoy probablemente también hará frío.

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

De esto salió una regla que ahora aplico siempre: **si la mejora es más pequeña que la dispersión del método, no es una mejora.** Fue lo que impidió dar por bueno el modelo B en validación, y lo que hizo visible que la mejora de la temperatura sí superaba el ruido — que es justo por lo que hubo que descartarla por otro motivo y no por el número.

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

### La subida de 2026 queda fuera de lo que el histórico permite aprender

Los dos modelos predicen por debajo de la realidad: 755 MW de media el modelo A, 413 el modelo B.

La razón es que 2026 consume unos 2.237 MW más de media que los años de entrenamiento, y un árbol de decisión no extrapola: no puede predecir un valor por encima de todo lo que ha visto. Es comprobable por otra vía — al forzar el árbol más simple, el sesgo no baja, sube. No es un problema de ajuste, es el techo de la herramienta.

Y hay un fleco abierto sobre el tamaño de esa subida. Al agregar mi serie por meses y compararla con las cifras que REE publica en sus notas de prensa, los años de entrenamiento coinciden y 2026 se separa: mi serie crece entre el 3 y el 12 % mes a mes sobre 2025, mientras las cifras publicadas dan −1,1 % en abril y +0,5 % en junio. Junio de 2025 coincide al 0,03 %, así que el método de agregación es correcto y lo que cambia está en 2026.

No sé si la diferencia está en qué agrega exactamente el indicador que descargo o en la demanda misma, y he escrito al operador del sistema para preguntarlo. Mientras no haya respuesta se queda como límite declarado y no corrijo nada: los errores medios comparan real y previsión de la **misma fuente y el mismo alcance** en cada periodo, así que son internamente consistentes. Lo que está en el aire es cuánta parte del sesgo es del modelo y cuánta de los datos.

### La comparación con REE no mide lo mismo

La previsión oficial acierta a 266 MW. La mía, a 1.005 en el mejor caso. No son el mismo problema y presentarlo como una derrota sería un error de encuadre.

Mi modelo predice el día completo desde las 00:00 usando el consumo del día anterior. REE, además de emitir su previsión antes de eso, dispone de previsión meteorológica operativa (la que aquí no puedo usar sin fuga) y recalibra continuamente contra el nivel real de consumo, que es exactamente lo que un árbol no sabe hacer.

Queda una pregunta abierta sobre la serie oficial: no he verificado si el indicador que descargo es la previsión del día anterior congelada o una serie que se sobrescribe con reajustes posteriores. La resolución de la serie es de cinco minutos, pero eso es su granularidad temporal, no prueba de que se reemita cada cinco minutos. Está en la misma consulta cursada al operador. **El 31 % de mejora sobre persistencia no depende de esto en absoluto.**

### Dos cosas más

**El modelo asume que el día anterior se conoce completo.** Usa el consumo de ayer a la misma hora, lo que implica que a las 00:00 ya tengo las 24 horas de ayer cerradas. Es defendible para un ejercicio, pero un margen de emisión realista obligaría a otra arquitectura.

**El trozo de 2026 se leyó cuatro veces durante la exploración del modelo B**, tres de ellas con una configuración fijada a mano en vez de elegida en validación. El 1.005 MW es correcto, pero tiene algo menos de estatus como "dato nunca visto" que el 1.263 del modelo A.

**Durante el apagón de abril de 2025, la previsión oficial se fue a cero unas 35 horas.** No se puede comparar un modelo contra una previsión que no existe, así que ese tramo queda fuera del benchmark.

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
src/             clientes de API, rutas, validación de calidad, utilidades de modelado
notebooks/       modelo de demanda y detección de anomalías
scripts/         gate de regresión numérica
modelos/         modelo serializado y foto de referencia de los números
tests/           test de paridad notebook ↔ src
```

---

## Licencia

MIT. El código es libre; los datos de origen conservan las condiciones de sus proveedores, que exigen reconocimiento de la fuente.

---

*Fase 1 de tres. La Fase 2 usará este modelo para proyectar tensiones en la red bajo escenarios de shock geopolítico y climático.*
