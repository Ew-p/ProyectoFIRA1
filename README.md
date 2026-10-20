# Pruebas de predicción del rendimiento de cebada por parcela

> Qué se hizo, qué se probó y qué resultó, a partir del registro de trabajo (197 parcelas, 2022–2025).

## Contenido

- [Resumen](#resumen)
- [1. Introducción](#1-introducción)
- [2. Marco conceptual](#2-marco-conceptual)
- [3. Descripción de los datos](#3-descripción-de-los-datos)
- [4. Metodología de extracción de rasters](#4-metodología-de-extracción-de-rasters)
- [5. Verificación de integridad de los datos](#5-verificación-de-integridad-de-los-datos)
- [6. ¿Mejora el clima la predicción?](#6-mejora-el-clima-la-predicción)
- [7. Dimensionalidad: de ~100 a 1,104 características](#7-dimensionalidad-de-100-a-1104-características)
- [8. Discusión](#8-discusión)
- [9. Conclusiones y trabajo futuro](#9-conclusiones-y-trabajo-futuro)
- [Glosario](#glosario)
- [Apéndice A. Resumen de comandos](#apéndice-a-resumen-de-comandos)

## Resumen

Este documento describe, de forma ordenada, el proceso seguido para estimar el rendimiento de cebada (toneladas por hectárea, t/ha) en un conjunto de 197 parcelas agrícolas, de las cuales 138 se usan para entrenar modelos y 59 quedan reservadas para generar predicciones finales. El trabajo integra tres fuentes de información: (i) índices espectrales derivados de los sensores satelitales Landsat y Sentinel-2, (ii) series mensuales de temperatura mínima, temperatura máxima y precipitación para los años 2022 a 2025, y (iii) variables topográficas y de ubicación.

El documento cubre cuatro bloques. Primero, el desarrollo de un programa en Python que extrae, para cada parcela, el valor de rasters mensuales (archivos GeoTIFF) y produce un archivo de Excel por año. Segundo, la verificación de integridad de los datos obtenidos: los cuatro años de precipitación y los cuatro de temperatura resultaron completos, con las 197 parcelas y los 12 meses en cada archivo. Tercero, la evaluación, mediante validación cruzada de 5 pliegues con 5 repeticiones, de si las variables climáticas mejoran la predicción. Cuarto, la explicación de por qué el conjunto de características crece de alrededor de un centenar a 1,104 columnas al conservar la dimensión mensual.

El hallazgo central es que las variables climáticas **no mejoran** el modelo una vez que se conocen el municipio y las coordenadas de la parcela: el error de validación cruzada se mantiene en RMSE ≈ 0.487 t/ha con R² ≈ 0.653, y agregar seis variables climáticas lo empeora ligeramente (RMSE 0.508; R² 0.623). Se argumenta que la señal climática, a una resolución de 5 km, funciona principalmente como un indicador indirecto de la ubicación geográfica. En paralelo, se explica que la expansión a 1,104 características es una decisión de diseño deliberada, compensada con selección de variables (LASSO, con k = 15) y con 1,000 corridas de validación, y que su alternativa —condensar el tiempo en estadísticos— rindió peor (R² 0.44 frente a 0.565).

## 1. Introducción

### 1.1 Contexto y problema

Conocer el rendimiento esperado de un cultivo antes de la cosecha tiene valor práctico para el productor, para la planeación de insumos y para la comercialización. En el caso de la cebada, un cultivo sensible al agua disponible, a las temperaturas durante el desarrollo y a las condiciones del suelo, el rendimiento varía de una parcela a otra incluso dentro de una misma región. El problema consiste en predecir el rendimiento por parcela a partir de información que puede obtenerse de forma remota, sin visitar cada terreno.

Los datos de partida son 197 parcelas con su identificador (`ID_POLIGONO`), su superficie (`AREA_HA`), el municipio al que pertenecen, un identificador de conjunto (`CONJUNTO`) y, solo para el subconjunto de entrenamiento, el rendimiento observado (`RENDIMIENTO_T_HA`). La partición es 70/30: 138 parcelas con rendimiento conocido sirven para entrenar y validar, y 59 parcelas sin rendimiento conocido son las que se deben predecir.

### 1.2 Alcance del documento

Este texto se redactó a partir del registro de trabajo del proyecto (un intercambio de trabajo en el que se desarrollaron los programas, se revisaron archivos y se corrieron experimentos). Por ello, las cifras que se reportan son las que aparecen en ese registro. No se introducen resultados nuevos ni se reejecutan los modelos. Cuando el registro deja algo impreciso, el documento lo señala de manera explícita en lugar de suponer una respuesta; estos puntos se reúnen en la sección 7.6.

## 2. Marco conceptual

### 2.1 Rasters y archivos GeoTIFF

Un **raster** es una malla regular de celdas (píxeles), cada una con un valor numérico. En este proyecto, cada raster mensual representa una variable continua sobre el territorio —por ejemplo, la precipitación acumulada de un mes— y cada píxel cubre un área de aproximadamente 5 km por 5 km. Un **GeoTIFF** es un archivo de imagen TIFF que lleva incrustada, en sus etiquetas, la información de georreferenciación: dónde está el origen de la malla, qué tamaño tiene cada píxel y en qué sistema de coordenadas se expresa.

Leer la georreferencia directamente de las etiquetas del archivo, en lugar de escribir coordenadas a mano en el programa, tiene una ventaja importante: el programa se vuelve genérico. Si algún raster llegara con otra resolución o con otra extensión, el cálculo del píxel correspondiente a cada parcela se ajusta solo.

### 2.2 Centroide de un polígono

Cada parcela está representada en el shapefile como un polígono. Para asociarle un único valor de un raster se utiliza su **centroide**, el punto central del polígono. Con las coordenadas del centroide (longitud y latitud) se localiza el píxel que lo contiene y se toma su valor. Este procedimiento supone que la variable es aproximadamente homogénea dentro de la parcela, algo razonable cuando el píxel (5 km) es mucho mayor que la parcela.

### 2.3 Consecuencia de la resolución de 5 km

Como el píxel es grande en comparación con las parcelas, varias parcelas cercanas caen en el mismo píxel y reciben **exactamente el mismo valor** en cada mes. Esto no es un error del programa; es una propiedad de los datos. Tiene, sin embargo, una consecuencia analítica de fondo: una variable que toma el mismo valor para todas las parcelas de una zona distingue entre zonas, pero no entre parcelas de la misma zona. Esa propiedad es la clave para entender por qué el clima, en este proyecto, termina comportándose como un sustituto de la ubicación (capítulo 6).

### 2.4 Validación cruzada k-fold

La **validación cruzada de k pliegues** (k-fold) es un método para estimar qué tan bien generalizará un modelo a datos que no ha visto. Los datos de entrenamiento se dividen en k grupos (pliegues) de tamaño similar; se entrena el modelo con k − 1 pliegues y se mide el error en el pliegue restante, y se repite hasta que cada pliegue haya sido el de prueba una vez. El error final es el promedio de los k resultados.

En este trabajo se usó un esquema de **5 pliegues repetido 5 veces** (5×5), con particiones distintas en cada repetición. Repetir reduce la dependencia del resultado respecto de una partición particular, algo especialmente relevante cuando solo hay 138 observaciones. Todos los modelos se comparan con el mismo esquema, lo que vuelve las comparaciones justas.

### 2.5 Métricas de error
- **RMSE** (raíz del error cuadrático medio): error típico de la predicción, expresado en las mismas unidades que el objetivo, aquí toneladas por hectárea. Menor es mejor. Penaliza más los errores grandes.
- **R²** (coeficiente de determinación): proporción de la variabilidad del rendimiento que el modelo explica. Un valor de 0.65 indica que explica alrededor del 65 %. Mayor es mejor.

### 2.6 Modelos utilizados

**ExtraTrees** (árboles extremadamente aleatorizados) es un método de conjunto que promedia muchos árboles de decisión construidos con cortes elegidos al azar. Maneja bien relaciones no lineales y no exige escalar las variables. Es el modelo con el que se comparan los conjuntos de variables en el capítulo 6.

**LASSO** es una regresión lineal con penalización de tipo L1 que empuja a cero los coeficientes de las variables poco útiles. Funciona, por tanto, como un método de **selección de características**: de cientos o miles de columnas conserva solo un puñado. Resulta apropiado cuando hay muchas más columnas que observaciones, como ocurre con el conjunto de 1,104 características frente a 138 parcelas.

### 2.7 Sobreajuste y selección de características

Cuando el número de columnas supera con holgura al número de observaciones, un modelo puede encontrar patrones que existen solo por azar en la muestra y que no se repiten en datos nuevos. A esto se le llama **sobreajuste**. Las defensas empleadas en el proyecto son dos: la selección de pocas variables (por ejemplo, k = 15) y la repetición masiva de la validación (1,000 corridas) para comprobar que lo seleccionado no es ruido.

## 3. Descripción de los datos

### 3.1 Parcelas y particiones

| **Elemento**     | **Descripción**                                        | **Cantidad** |
|------------------|--------------------------------------------------------|--------------|
| Parcelas totales | Polígonos del shapefile `Parcelas_Reto_AGC_CONJUNTO.shp` | 197          |
| Entrenamiento    | Parcelas con rendimiento observado                     | 138          |
| Predicción       | Parcelas cuyo rendimiento se debe estimar              | 59           |
| Partición        | Proporción entrenamiento / predicción                  | 70 / 30      |

*Tabla 1. Resumen de las parcelas del reto.*

El shapefile ya contiene atributos útiles —municipio, área y conjunto—, que el programa de extracción conserva en cada archivo de salida. Un archivo adicional (`ID_area_rendimiento_70_30_Reto_AgroCebada.csv`) aporta `AREA_HA`, `RENDIMIENTO_T_HA` y `CONJUNTO`, y se integra a los conjuntos agregados.

### 3.2 Datos satelitales

Los datos satelitales provienen de dos sensores, Landsat y Sentinel-2, y se entregan en cuatro archivos (cada sensor por separado para entrenamiento y para predicción). Cada fila de esos archivos corresponde a una fecha y una parcela, y contiene un conjunto de características espectrales: índices de vegetación y de suelo como NDVI, EVI, SAVI, LAI, NDTI, CRC y STI, entre otros. Para evitar mediciones contaminadas por nubes, la agregación aplica un **filtro de nubosidad del 20 %**.

Estos archivos se combinan mediante el programa `agregar_por_parcela.py`, que produce dos libros de Excel: `agregado_train.xlsx` (138 filas) y `agregado_prediccion.xlsx` (59 filas). El primero contiene 1,105 características más `AREA_HA` y `RENDIMIENTO_T_HA`; el segundo tiene las mismas columnas, sin el rendimiento.

### 3.3 Datos climáticos

Los datos climáticos son rasters mensuales de tres variables: precipitación (PREC), temperatura mínima (Tmin) y temperatura máxima (Tmax), para los años 2022, 2023, 2024 y 2025. Los archivos siguen la convención de nombre terminada en \_AAAA_MM.tif (por ejemplo, ...\_2022_01.tif para enero de 2022). Un ejemplo de control usado durante el desarrollo fue la precipitación de enero de 2022, cuyo valor para el municipio de Emiliano Zapata resultó cercano a 8 mm.

### 3.4 Variables topográficas y de ubicación

Además de la información anterior, el proyecto incorporó variables topográficas y la ubicación de cada parcela (municipio y coordenadas lon/lat). Según el registro de trabajo, la temperatura y la topografía ya se habían evaluado por separado con anterioridad, y en ambos casos se concluyó que actuaban como un “municipio disfrazado”, es decir, que explicaban rendimiento principalmente por dónde se ubican las parcelas.

## 4. Metodología de extracción de rasters

### 4.1 Necesidad del programa

Extraer a mano el valor de cada raster mensual para 197 parcelas, en cuatro años y tres variables, equivale a miles de consultas (197 parcelas × 12 meses × 4 años × 3 variables = 28,368 valores). El programa `extraer_tif_por_anio.py` automatiza ese proceso y entrega una tabla ordenada por año.

### 4.2 Funcionamiento

El programa realiza los siguientes pasos:
1.  Recorre la carpeta indicada y localiza cualquier archivo cuyo nombre termine en \_AAAA_MM.tif. No importa cómo se llame la variable (PREC\_, Tmin\_, Tmax\_ u otra), ni si los archivos están sueltos o en subcarpetas por año.
2.  Lee la georreferencia (origen, tamaño de píxel) de las etiquetas del propio GeoTIFF.
3.  Obtiene el centroide de cada parcela a partir del shapefile (que debe tener su archivo .dbf al lado). Como alternativa, acepta un CSV con las columnas `ID_POLIGONO`, lon y lat para quien no disponga del shapefile.
4.  Convierte las coordenadas del centroide a la posición de fila y columna del raster y toma el valor del píxel correspondiente, mes por mes.
5.  Escribe un archivo de Excel por año, con una fila por parcela (incluyendo Municipio, Área y Conjunto) y una columna por cada mes disponible (PREC_01, PREC_02, …).

Si falta algún mes (por ejemplo, un diciembre que aún no se ha descargado), simplemente no aparece la columna correspondiente; el programa no falla.

### 4.3 Modos de uso: interfaz y línea de comandos

El proyecto sigue una convención de “pequeñas interfaces” en cada programa, de modo que quien lo use no tenga que recordar todos los argumentos. Por ello `extraer_tif_por_anio.py` ofrece dos modos equivalentes.

**Modo interfaz.** Se ejecuta sin argumentos y pregunta, paso a paso: la carpeta raíz con los .tif; si se dispone de shapefile o de CSV de centroides; la ruta del archivo; y la carpeta donde guardar los Excel. Al final pregunta si se desea procesar otra carpeta, de manera que se pueda correr primero precipitación y luego temperatura sin cerrar el programa.

```bash
python3 extraer_tif_por_anio.py
```

**Modo directo.** Se pasan todos los parámetros en una sola línea, útil para repetir el proceso sin interacción.

```bash
python3 extraer_tif_por_anio.py --tifs "ruta\Precipitacion_mensual" --shp "ruta\Parcelas_Reto_AGC_CONJUNTO.shp" --salida "ruta\salidas"
```

Ambos modos se probaron y producen resultados idénticos (mismos archivos de Excel y mismos valores).

### 4.4 Validación del programa

Antes de aplicarlo a los datos completos, el programa se probó simulando la estructura real de carpetas por año con dos rasters de ejemplo (temperatura mínima de enero de 2022 y precipitación de enero de 2022) y con el shapefile real. Los valores obtenidos coincidieron con los que se habían calculado manualmente con anterioridad, lo que respalda la correcta lectura de la georreferencia y la correcta asignación de píxeles.

### 4.5 Nombres de las salidas

Los archivos de precipitación se nombraron `PREC_2022.xlsx` a `PREC_2025.xlsx`. En el caso de la temperatura, Tmin y Tmax se colocaron en la misma carpeta, por lo que el programa detectó dos variables distintas y usó el nombre genérico `multivariable_2022.xlsx` a `multivariable_2025.xlsx`, con columnas del tipo Tmin_01 y Tmax_06. El número de la columna indica el mes: 01 es enero y 12 es diciembre.

## 5. Verificación de integridad de los datos

### 5.1 Criterios de revisión

Dado que se sospechaba que podían faltar parcelas o meses, se revisaron los ocho archivos de salida con tres criterios: (a) que cada archivo contuviera las 197 parcelas; (b) que no hubiera identificadores duplicados; y (c) que no hubiera valores nulos. Además, se verificó qué meses estaban presentes en cada archivo.

### 5.2 Precipitación

| **Archivo**    | **Parcelas** | **Duplicados** | **Nulos** | **Meses presentes** |
|----------------|--------------|----------------|-----------|---------------------|
| `PREC_2022.xlsx` | 197          | 0              | 0         | 01 a 12             |
| `PREC_2023.xlsx` | 197          | 0              | 0         | 01 a 12             |
| `PREC_2024.xlsx` | 197          | 0              | 0         | 01 a 12             |
| `PREC_2025.xlsx` | 197          | 0              | 0         | 01 a 12             |

*Tabla 2. Verificación de los archivos de precipitación.*

### 5.3 Temperatura

Los cuatro archivos multivariable contienen, además de los metadatos, 24 columnas de datos (12 de Tmin y 12 de Tmax), lo que da 32 columnas en total. Todos tienen las 197 parcelas, sin duplicados ni nulos.

| **Archivo**             | **Parcelas** | **Tmin**          | **Tmax**          |
|-------------------------|--------------|-------------------|-------------------|
| `multivariable_2022.xlsx` | 197          | 01 a 12 completos | 01 a 12 completos |
| `multivariable_2023.xlsx` | 197          | 01 a 12 completos | 01 a 12 completos |
| `multivariable_2024.xlsx` | 197          | 01 a 12 completos | 01 a 12 completos |
| `multivariable_2025.xlsx` | 197          | 01 a 12 completos | 01 a 12 completos |

*Tabla 3. Verificación de los archivos de temperatura.*

### 5.4 Conclusión de la verificación

La sospecha inicial de faltantes no se confirmó: la extracción está completa para las tres variables y los cuatro años. Con ello queda cerrada la etapa de obtención de datos externos y es posible pasar a la construcción de variables resumen.

## 6. ¿Mejora el clima la predicción?

### 6.1 Variables resumen con sentido agronómico

Incorporar 36 columnas mensuales por año directamente a un modelo con solo 138 observaciones invitaría al sobreajuste. En su lugar, se construyeron pocas variables resumen con una interpretación agronómica. El registro reporta seis variables climáticas; las que se describen con su correlación son cinco:
- **amplitud_termica**: diferencia entre temperatura máxima y mínima (Tmax − Tmin).
- **tmin_helada_ene_feb**: temperatura mínima de enero y febrero, asociada al riesgo de heladas.
- **prec_arranque_abr_may**: precipitación acumulada de abril y mayo, relacionada con el arranque del cultivo.
- **prec_lluvia_jun_sep**: precipitación acumulada de junio a septiembre, la temporada principal de lluvias.
- **tmax_calor_may_ago**: temperatura máxima de mayo a agosto, indicadora de estrés por calor.

### 6.2 Diseño del experimento

Se compararon cinco conjuntos de variables con el modelo ExtraTrees y el esquema de validación cruzada 5×5 descrito en el capítulo 2. El conjunto de referencia (base) combina el municipio con las coordenadas lon/lat. Los demás conjuntos añaden, sucesivamente, tres índices satelitales compactos, las seis variables climáticas, ambas cosas, y finalmente se evalúa el clima por sí solo, sin municipio ni satélite.

### 6.3 Resultados

| **Conjunto de variables**                   | **RMSE (t/ha)** | **R²** |
|---------------------------------------------|-----------------|--------|
| Municipio + lon/lat (base)                  | 0.487           | 0.653  |
| Base + 3 satelitales compactas              | 0.487           | 0.653  |
| Base + 6 climáticas                         | 0.508           | 0.623  |
| Base + satélite + clima                     | 0.487           | 0.653  |
| Solo climáticas (sin municipio ni satélite) | 0.517           | 0.610  |

*Tabla 4. Resultados de validación cruzada (5 pliegues × 5 repeticiones) con ExtraTrees.*

Tres lecturas se desprenden de la tabla. Primera: agregar el clima al modelo base **no reduce el error**; en los casos en que el resultado es idéntico al de la base (0.487), el modelo prácticamente ignora las columnas nuevas. Segunda: cuando se añade el clima junto con la base en forma directa, el error **empeora** (0.508), consistente con la entrada de ruido. Tercera: el clima por sí solo explica una parte sustancial de la variabilidad (R² = 0.610), lo que indica que sí contiene información, pero esa información ya está contenida en el municipio y las coordenadas.

### 6.4 Correlaciones con el rendimiento

| **Variable climática**         | **Correlación con el rendimiento** |
|--------------------------------|------------------------------------|
| amplitud_termica (Tmax − Tmin) | −0.70                              |
| tmin_helada_ene_feb            | +0.67                              |
| prec_arranque_abr_may          | −0.64                              |
| tmax_calor_may_ago             | −0.40                              |
| prec_lluvia_jun_sep            | −0.36                              |

*Tabla 5. Correlación de cada variable climática con el rendimiento.*

Las correlaciones son elevadas, algo que a primera vista parecería prometedor. Sin embargo, una correlación alta no implica por sí misma que la variable aporte información adicional. Aquí, la explicación es geográfica: las parcelas de Chignahuapan y Almoloya, que tienen los mejores rendimientos, son más frías, con menor amplitud térmica y con menos lluvia de arranque que las de Calpulalpan y Singuilucan. Esa diferencia se debe a su posición en la malla de 5 km, no a un efecto agronómico aislado del clima dentro de cada zona.

### 6.5 Importancia de las variables

En el modelo que combina todo, el municipio y las coordenadas lon/lat concentran el **72 %** de la importancia, y el conjunto de variables climáticas aporta solo el **28 %**. Además, ese 28 % probablemente ya estaba capturado por lo que explica el municipio. Es una situación típica de **colinealidad estructural**: dos grupos de variables que miden, en esencia, lo mismo con distinto nombre.

### 6.6 Decisión

Con base en estos resultados, se decidió **no incluir** las seis variables climáticas en el modelo final. El modelo vigente permanece como: municipio + lon/lat + 3 índices satelitales compactos, con un RMSE ≈ 0.49 t/ha y un R² ≈ 0.65 en validación cruzada. Con esta evaluación se agotaron las fuentes externas consideradas (temperatura, precipitación y topografía) sin hallar una mejora real por encima de la geografía y el satélite.

> *Interpretación: el clima no es inútil; es redundante. Lo que explica de la variación del rendimiento ya lo explica la ubicación de la parcela.*

## 7. Dimensionalidad: de ~100 a 1,104 características

### 7.1 La pregunta

Una pregunta natural surgió durante el trabajo: ¿por qué se pasó de unas 96 características a más de mil? La respuesta es de naturaleza combinatoria y refleja una decisión de diseño, no un error.

### 7.2 La cuenta

Cada archivo de sensores contiene, por fecha y parcela, un conjunto de características espectrales (índices como NDVI, EVI, SAVI, LAI, NDTI, CRC y STI). Esas características describen un solo día. Pero cada parcela tiene muchas fechas repartidas en cuatro años, y el modelo necesita **una sola fila por parcela**. Es preciso, por tanto, “aplanar” el tiempo.

La solución adoptada en `agregar_por_parcela.py` fue construir una columna por cada índice y por cada mes. Así, en lugar de resumir un año en un único número, el modelo puede conocer, por ejemplo, el NDVI de junio. La relación resultante es:

```bash
número de columnas = índices espectrales × 12 meses   (1,104 / 12 = 92 por mes)
```

Nótese que 1,104 dividido entre 12 es exactamente 92, de modo que el resultado corresponde a 92 columnas espectrales por mes. El registro de trabajo se refiere a ese número de manera imprecisa (“23 a 96”, según cómo se cuenten los nombres con sufijos); la cifra de 92 es la que cuadra aritméticamente con 1,104, aunque conviene confirmarla contra el archivo de datos real.

### 7.3 ¿Por qué no condensar el tiempo?

Una alternativa para evitar la multiplicación es resumir cada índice con estadísticos del año (media, desviación estándar, asimetría y curtosis). Ese experimento se realizó, pero rindió peor: R² de 0.44 frente a 0.565 con la versión mensual. La razón es que el resumen anual pierde **cuándo** ocurre cada cosa, y en este problema el mes importa: junio resultó ser un mes clave. Conservar la dimensión mensual, aun a costa de tener muchas columnas, preserva esa señal.

### 7.4 Cómo se controla el exceso de columnas

Tener 1,104 columnas con 138 observaciones plantea un riesgo evidente de sobreajuste. El proyecto lo gestiona de dos maneras complementarias:
1.  **Selección de características.** Métodos como LASSO (y, en comparaciones, Random Forest) reducen las 1,104 columnas a solo 15 (parámetro k = 15).
2.  **Validación repetida.** Un programa realiza 1,000 corridas de validación cruzada para comprobar que lo elegido no es producto del azar.

La ejecución se hace con el programa de 1,000 corridas, indicando los archivos de entrenamiento y predicción, la variable objetivo, la columna identificadora y el número de características a conservar:

```bash
python3 tu_script_1000.py --train agregado_train.xlsx --prediccion agregado_prediccion.xlsx --target RENDIMIENTO_T_HA --id ID_POLIGONO --k 15
```

### 7.5 Resultado asociado

Este conjunto agregado es el que ha dado el mejor resultado confirmado dentro de esa línea de trabajo: LASSO con un R² de aproximadamente 0.55 a 0.565.

### 7.6 Puntos que conviene aclarar

Al integrar el registro aparecen dos puntos que no quedan del todo resueltos en el texto original y que se señalan aquí con honestidad:
- **Dos referencias de R² distintas.** El modelo ExtraTrees con municipio, coordenadas y tres índices compactos reporta R² ≈ 0.65 (capítulo 6), mientras que el LASSO sobre las 1,104 columnas reporta R² ≈ 0.55–0.565. Son pipelines distintos y, aparentemente, con esquemas de evaluación distintos (5×5 en uno; 1,000 corridas con k = 15 en el otro). Antes de elegir un modelo final conviene compararlos bajo exactamente el mismo esquema de validación.
- **Número exacto de características por mes.** Como se indicó, el registro habla de “23 a 96”, pero 1,104 / 12 = 92. Confirmar la cifra evita confusiones al describir el conjunto de datos.
- **Sexta variable climática.** El registro menciona seis variables climáticas, pero reporta la correlación de cinco; la restante no se identifica en el texto.
- **Valores idénticos entre parcelas.** La resolución de 5 km hace que varias parcelas compartan valor climático; los modelos que usen clima deben interpretarse sabiendo esto.

## 8. Discusión

### 8.1 Por qué el clima “no ayuda” aunque correlacione fuerte

El resultado más llamativo del proyecto es la aparente paradoja de variables con correlaciones de −0.70 o +0.67 que no mejoran el modelo. La resolución de la paradoja está en distinguir entre **información** e **información adicional**. Una variable es útil para un modelo solo si aporta algo que las demás variables no tienen. El municipio y las coordenadas determinan, de forma casi unívoca, a qué píxel de 5 km pertenece una parcela y, por tanto, su valor climático. El clima, en esta escala, es una función de la ubicación; no puede añadir lo que la ubicación ya dice.

### 8.2 Implicaciones prácticas
- Si el objetivo es **predecir** el rendimiento de parcelas dentro de la misma región, la ubicación y el satélite bastan; el clima de 5 km no mejora la predicción.
- Si el objetivo fuera **explicar** el rendimiento o **extrapolar** a una región nueva, el clima sí podría ser relevante, pues allí la ubicación aprendida ya no serviría. Este escenario no se evaluó.
- Para detectar un efecto climático real dentro de una zona haría falta una resolución espacial más fina que 5 km, o datos de estaciones locales.

### 8.3 Valor metodológico de la validación cruzada

Los resultados ilustran por qué es importante medir con validación cruzada en lugar de confiar en correlaciones o en el ajuste dentro de la muestra. Con las correlaciones solas, se habría concluido que el clima era un predictor excelente. La validación cruzada mostró que, una vez que se controla por la ubicación, no lo es. Del mismo modo, la validación repetida protege contra elegir, entre 1,104 columnas, aquellas que lucen bien por casualidad.

### 8.4 Limitaciones
- Tamaño muestral reducido (138 parcelas de entrenamiento), lo que vuelve ruidosas las comparaciones entre modelos con diferencias pequeñas.
- Resolución espacial gruesa del clima (5 km) frente al tamaño de las parcelas.
- Uso del centroide como único punto de muestreo por parcela, sin promediar dentro del polígono.
- Cuatro años de datos climáticos: poca variabilidad interanual para estimar efectos del tiempo atmosférico.
- Comparación pendiente entre los pipelines de ExtraTrees y de LASSO bajo un mismo esquema de validación.

## 9. Conclusiones y trabajo futuro

### 9.1 Conclusiones
1.  Se desarrolló un programa de extracción de rasters mensuales por parcela, con interfaz interactiva y modo de línea de comandos, validado contra cálculos manuales.
2.  Los datos extraídos de precipitación, temperatura mínima y temperatura máxima para 2022–2025 están completos: 197 parcelas, 12 meses, sin duplicados ni nulos.
3.  El clima, resumido en variables con sentido agronómico, no mejora el modelo en validación cruzada 5×5 (RMSE 0.487 con la base; 0.508 al añadir clima) y se comporta como sustituto de la ubicación geográfica.
4.  El modelo vigente es municipio + lon/lat + tres índices satelitales compactos, con RMSE ≈ 0.49 t/ha y R² ≈ 0.65.
5.  El crecimiento a 1,104 columnas es consecuencia de conservar la dimensión mensual (índices × 12 meses); el aplanamiento por estadísticos rindió peor (R² 0.44 frente a 0.565), y el exceso se controla con LASSO (k = 15) y 1,000 corridas de validación.

### 9.2 Trabajo futuro
- Generar las **predicciones finales de las 59 parcelas** con el modelo seleccionado, que era el siguiente paso planteado en el registro de trabajo.
- Comparar de manera directa ExtraTrees y LASSO con el mismo esquema de validación y elegir el modelo final con base en esa comparación.
- Explorar rasters climáticos de mayor resolución o datos de estaciones para intentar capturar variación dentro de las zonas.
- Evaluar la capacidad de generalización a municipios no vistos (validación dejando fuera un municipio completo), donde el clima podría sí aportar.
- Documentar la lista exacta de las 15 variables seleccionadas por LASSO y su interpretación agronómica (por ejemplo, el papel del mes de junio).

## Glosario

| **Término**                 | **Definición**                                                                                                      |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------|
| Raster                      | Malla de celdas (píxeles) con un valor numérico por celda.                                                          |
| GeoTIFF                     | Archivo TIFF con georreferenciación incrustada en sus etiquetas.                                                    |
| Shapefile                   | Formato de datos vectoriales; aquí contiene los polígonos de las parcelas (requiere el archivo .dbf junto al .shp). |
| Centroide                   | Punto central de un polígono.                                                                                       |
| Píxel                       | Celda individual de un raster; aquí cubre ~5 km × 5 km.                                                             |
| Validación cruzada (k-fold) | Método para estimar el error de generalización dividiendo los datos en k pliegues.                                  |
| RMSE                        | Raíz del error cuadrático medio; error típico en las unidades del objetivo (t/ha).                                  |
| R²                          | Proporción de la variabilidad explicada por el modelo.                                                              |
| ExtraTrees                  | Conjunto de árboles de decisión con cortes aleatorios.                                                              |
| LASSO                       | Regresión lineal con penalización L1 que selecciona variables.                                                      |
| Sobreajuste                 | Ajuste a patrones que existen solo en la muestra de entrenamiento.                                                  |
| NDVI, EVI, SAVI, LAI        | Índices espectrales de vegetación (verdor, biomasa, área foliar).                                                   |
| NDTI, CRC, STI              | Índices relacionados con residuos de cultivo y condición del suelo.                                                 |
| Colinealidad                | Situación en que dos o más variables contienen información casi idéntica.                                           |
| Prec / Tmin / Tmax          | Precipitación, temperatura mínima y temperatura máxima.                                                             |

*Tabla 6. Glosario de términos.*


```bash
python3 tu_script_1000.py --train agregado_train.xlsx --prediccion agregado_prediccion.xlsx --target RENDIMIENTO_T_HA --id ID_POLIGONO --k 15
```
