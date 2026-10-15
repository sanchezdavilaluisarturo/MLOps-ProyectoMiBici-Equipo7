# Diseño del pipeline de ML — Proyecto MiBici (Equipo 7)

> **Estado:** borrador para acordar en el equipo. Las cifras salen del EDA (notebooks 0.01 a 0.03 y `definicion-datos.md`). Los umbrales y parámetros marcados como *propuesta* se fijarán con evidencia en el EDA.

## 1. Vista general

![Pipeline de ML por pregunta](pipeline-ml.png)

*Una limpieza común se bifurca en tres carriles, uno por pregunta. Cada carril recorre agregación, features, partición, entrenamiento y evaluación, y registra en MLflow con el mismo `data_version`. En verde, los controles contra fuga de información del futuro.* El diagrama de componentes y herramientas está en la sección *Arquitectura* del README ([`arquitectura.png`](arquitectura.png)).

El flujo sigue la convención de datos de CCDS (`raw` → `interim` → `processed`): cada etapa lee de una carpeta y escribe en la siguiente, y nunca se modifica lo que ya está en `raw`.

| Etapa | Entrada | Código (CCDS) | Salida |
|---|---|---|---|
| **1. Ingesta** | CSV del portal | `dataset.py` | `data/raw/` con los CSV originales y `manifest.json` (registros y SHA-256) |
| **2. Limpieza** | `data/raw/` | `dataset.py` | `data/interim/` con Parquet limpio (columnas, fechas y tipos normalizados) |
| **3. Agregación y features** | `data/interim/` | `features.py` | `data/processed/` con tablas de features por estación-hora y por estación |
| **4. Partición** | `data/processed/` | `features.py` | Conjuntos de entrenamiento, validación y prueba (temporal) |
| **5. Entrenamiento** | `data/processed/` | `modeling/train.py` | `models/` con los modelos serializados, un run de MLflow por pregunta |
| **6. Evaluación** | Modelos y conjuntos | `modeling/train.py`, `plots.py` | Métricas en MLflow y figuras en `reports/figures/` |
| **7. Predicción** | Modelo registrado y datos nuevos | `modeling/predict.py` | Predicciones por lotes |

**Trazabilidad:** cada run de MLflow guarda `data_version`, el hash del manifiesto, de modo que cualquier métrica se puede ligar a la versión exacta de los datos.

**Estado de los notebooks frente a CCDS:** `0.01` y `0.03` leen los Parquet de `data/external/` y `0.02` guarda el Parquet en `data/interim/`. Al migrar la lógica a `dataset.py`, todos deben leer de `data/interim/`, que es donde el diagrama ubica el Parquet limpio. `external` queda para datos de terceros que no pasan por la ingesta.

**Principios comunes**
- **Reproducibilidad:** semilla 42, dependencias con versiones fijadas y manifiesto de datos.
- **Sin fuga:** partición temporal y cortes de información definidos en `definicion-datos.md` (sección 5).
- **Cada decisión de limpieza se justifica con una cifra del EDA.**
- **Cada corrida** registra en MLflow parámetros, métricas, etiquetas (equipo, pregunta, commit) y artefactos.

## 2. Limpieza común a las tres preguntas

| Regla | Cifra del EDA | Tratamiento propuesto |
|---|---|---|
| Viajes duplicados | 758 filas repetidas completas | Eliminar, conservando una |
| Estaciones sin nomenclatura | 3 ids (124, 405, 407), 3,961 viajes (0.015 %) | Excluir: sin capacidad ni ubicación no se pueden usar en P2 ni P3 |
| Duración cero | 53,581 viajes (0.2 %) | Excluir de la demanda (probable desbloqueo fallido); medir sensibilidad |
| Duración extrema | 157 viajes de más de 24 h (máximo 469 h) | Excluir de las features de duración; no afectan conteos |
| Estaciones con capacidad 0 | 4 estaciones | Excluir de P2; revisar su papel en P1 y P3 |
| Meses sin datos | 2023-04 y 2023-05 | No imputar; marcar el hueco y no generar rezagos que crucen el hueco |
| Datos personales | `usuario_id`, `genero`, `anio_de_nacimiento` | Descartar al pasar a `processed` (minimización) |

## 3. P1: demanda de viajes (regresión)

### Etapas

| Etapa | Descripción |
|---|---|
| **Agregación** | Contar viajes por `origen_id` y hora de `inicio_del_viaje`. Completar con 0 las horas sin viajes **desde la primera hora activa de cada estación**, para no inventar demanda antes de que existiera |
| **Features** | Ver tabla siguiente |
| **Partición** | Temporal (entrenamiento / validación / prueba) con una brecha de 24 h entre conjuntos |
| **Entrenamiento** | Baseline → regresión lineal con regularización → modelo de árboles con objetivo de conteo (Poisson) |
| **Evaluación** | MAE y RMSE por conjunto, error por estación y por hora del día, y comparación contra el baseline |

### Variables

| Variable | Tipo | Por qué es pertinente |
|---|---|---|
| `hora`, `dia_semana`, `mes`, `es_fin_de_semana` | Calendario | El uso tiene ciclos diarios y semanales fuertes (commuting) |
| `hora_sin`, `hora_cos` | Calendario cíclico | Evita que las 23:00 y las 00:00 parezcan lejanas |
| `es_festivo` | Calendario | La demanda cae o cambia de perfil en feriados |
| `demanda_lag_24h`, `lag_48h`, `lag_168h` | Rezago | La misma hora de ayer y de la semana pasada es el mejor predictor de la demanda |
| `demanda_media_7d_misma_hora` | Ventana (cerrada a la izquierda) | Suaviza el ruido sin usar el futuro |
| `capacidad`, `latitud`, `longitud` | Estación | Estaciones más grandes y céntricas concentran más viajes |
| `demanda_media_estacion` | Estación (solo entrenamiento) | Nivel base de cada estación |
| `periodo_pandemia` | Indicador | Separa el choque de 2020, cuando el volumen cayó a 129,276 viajes en abril |

**Restricción de horizonte:** todos los rezagos son de **24 horas o más**, porque se predice un día adelante.

### Modelos y métricas

| Rol | Modelo | Nota |
|---|---|---|
| Baseline | Media histórica por estación, hora y día de la semana | Es el piso que se debe superar |
| Candidato 1 | Regresión lineal con regularización (Ridge) | Interpretable |
| Candidato 2 | XGBoost o LightGBM con objetivo Poisson | Captura no linealidades e interacciones |

- **Métricas:** MAE (principal, en viajes), RMSE (penaliza picos). Se evita MAPE porque hay muchas horas con 0 viajes.
- **Ajuste de hiperparámetros:** Optuna con validación temporal expansiva.

## 4. P2: riesgo de quedarse sin bicis o sin anclajes (clasificación)

### Etapas

| Etapa | Descripción |
|---|---|
| **Agregación** | Por estación y hora: salidas (`origen_id`, `inicio_del_viaje`) y llegadas (`destino_id`, `fin_del_viaje`); `neto = llegadas − salidas`, acumulado desde el inicio del día |
| **Etiqueta** | En el instante `t`, riesgo en `(t, t+2h]` si el acumulado alcanza `±u·capacidad`. Tres clases: normal, sin bicis y sin anclajes |
| **Features** | Ver tabla siguiente; todas calculadas con información hasta `t` |
| **Partición** | Temporal con brecha de 2 h entre conjuntos |
| **Entrenamiento** | Regla simple → regresión logística → XGBoost con pesos de clase |
| **Evaluación** | Métricas para clase minoritaria, umbral de decisión elegido en validación y análisis por estación |

### Variables

| Variable | Por qué es pertinente |
|---|---|
| `neto_acumulado_dia` | Estado actual aproximado de la estación respecto al inicio del día |
| `neto_1h`, `neto_2h`, `neto_3h` | Tendencia reciente: una estación que se vacía a ritmo constante llegará a cero |
| `neto_sobre_capacidad` | Normaliza por tamaño: −5 bicis es grave en una estación de 15 y no en una de 39 |
| `salidas_1h`, `llegadas_1h` | Distingue una estación muy activa de una con movimiento equilibrado |
| `hora`, `dia_semana`, `es_festivo` | Los picos de desbalance siguen la hora y el tipo de día |
| `neto_misma_hora_semana_anterior` | Patrón repetitivo de desbalance por estación |
| `capacidad` | Define cuántas bicis o anclajes son críticos |
| `demanda_esperada_2h` (opcional) | Salida de P1: encadena ambos modelos |

### Modelos y métricas

| Rol | Modelo | Nota |
|---|---|---|
| Baseline | Regla: `neto_sobre_capacidad` supera un umbral | Lo que haría un operador sin ML |
| Candidato 1 | Regresión logística con pesos de clase | Interpretable |
| Candidato 2 | XGBoost con pesos de clase | Interacciones hora × estación |

- **Desbalance:** el riesgo es un evento raro. Se usan pesos de clase y se evita *accuracy*.
- **Métricas:** PR-AUC (principal), recall y precisión de la clase de riesgo, F1 y matriz de confusión.
- **Umbral:** se elige en validación según el costo operativo: perder una estación en riesgo cuesta más que una alerta falsa, así que se prioriza el recall con una precisión mínima acordada.

### Limitación declarada

No hay inventario de bicicletas, por lo que la etiqueta es un **proxy**: el acumulado parte de un nivel desconocido y no ve el rebalanceo realizado. El modelo detecta **presión de desbalance**, no el inventario real. Debe validarse con el equipo de operación antes de usarse.

## 5. P3: grupos de estaciones (clustering)

### Etapas

| Etapa | Descripción |
|---|---|
| **Agregación** | Una fila por estación, calculada **solo con el periodo de entrenamiento** |
| **Features** | Ver tabla siguiente |
| **Preparación** | Filtrar estaciones con pocos viajes, estandarizar y, si hace falta, reducir con PCA |
| **Entrenamiento** | K-Means (baseline) y HDBSCAN o clustering jerárquico como alternativa |
| **Evaluación** | Silueta, estabilidad en el tiempo e interpretación de cada grupo |

### Variables

| Variable | Por qué es pertinente |
|---|---|
| `perfil_horario_laboral` (24 valores, normalizado) | Distingue estaciones de commuting (picos de mañana y tarde) de las de ocio |
| `perfil_horario_fin_de_semana` (24 valores) | Revela uso recreativo |
| `viajes_diarios_medios` (en logaritmo) | Separa estaciones de alto y bajo volumen |
| `razon_salidas_llegadas` | Estaciones que son origen neto o destino neto |
| `desbalance_hora_pico` | Estaciones que se vacían o se llenan siempre a la misma hora |
| `proporcion_viajes_circulares` | Origen = destino, típico de uso recreativo |
| `duracion_mediana_salida` | Viajes cortos de conexión frente a recorridos largos |
| `latitud`, `longitud` | **No entran al clustering**; se usan después para ubicar y describir cada grupo |

### Evaluación

| Criterio | Cómo se mide |
|---|---|
| **Cohesión y separación** | Coeficiente de silueta e índice de Davies-Bouldin para varios valores de `k` |
| **Estabilidad** | Índice de Rand ajustado entre los grupos del entrenamiento y los recalculados con el año de validación |
| **Interpretabilidad** | Perfil horario medio por grupo y mapa de estaciones; cada grupo debe poder nombrarse (por ejemplo, "oficinas") |

El criterio decisivo es la **utilidad para planeación urbana**, no solo la métrica: un `k` con silueta ligeramente menor pero grupos interpretables es preferible.

## 6. Comparación entre preguntas

| | P1 | P2 | P3 |
|---|---|---|---|
| **Unidad de análisis** | Estación-hora | Estación-hora | Estación |
| **Objetivo** | Conteo de viajes | Clase de riesgo | Sin objetivo |
| **Baseline** | Media por estación-hora-día | Regla por umbral | K-Means |
| **Modelo candidato** | XGBoost (Poisson) | XGBoost con pesos | HDBSCAN |
| **Métrica principal** | MAE | PR-AUC | Silueta + estabilidad |
| **Riesgo principal** | Fuga por rezagos | Proxy sin inventario | Grupos no interpretables |

## 7. Pendientes del equipo

- Valor del umbral `u` del riesgo en P2 y la ventana de acumulación (inicio del día u otra).
- Tratamiento de los 53,581 viajes con duración cero (excluir o conservar) y su sensibilidad.
- Número de estaciones mínimo de viajes para entrar a P3.
- Qué modelos candidatos se implementan primero dado el tiempo disponible.
