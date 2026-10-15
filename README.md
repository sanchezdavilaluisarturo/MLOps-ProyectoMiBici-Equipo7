# MLOps-ProyectoMiBici-Equipo7

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto integrador de **Operaciones de Aprendizaje Automático (MLOps, TC5061)** del Equipo 7. Propone y construye una solución de Machine Learning sobre los datos abiertos de **MiBici**, el sistema de bicicleta pública del Área Metropolitana de Guadalajara.

## Instalación

Requiere Python 3.11 y [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/sanchezdavilaluisarturo/MLOps-ProyectoMiBici-Equipo7.git
cd MLOps-ProyectoMiBici-Equipo7
uv venv --python 3.11 .venv
source .venv/bin/activate
make requirements        # uv pip install -r requirements.txt
```

## Ejecución

```bash
# Ingesta: descarga, normaliza, guarda Parquet y genera el manifiesto
python -m tc5061_proyecto_equipo07.dataset          # pendiente de implementar

# Experimento base en MLflow (P1, P2, P3)
python -m tc5061_proyecto_equipo07.modeling.train   # pendiente de implementar
mlflow ui                                                # http://127.0.0.1:5000
```

### Estado actual: ejecución desde los notebooks

Mientras el módulo `tc5061_proyecto_equipo07` no implemente la ingesta y el entrenamiento, el flujo se ejecuta desde `notebooks/` (cada notebook se corre desde esa carpeta):

| Orden | Notebook | Qué deja listo |
|---|---|---|
| 1 | `0.02-dsh-exploracion-carga` | CSV en `data/raw/` y Parquet unido en `data/interim/` |
| 2 | `0.03-lasd-exploracion-carga` | Parquet mensuales, EDA, limpieza, partición temporal, tablas en `data/processed/` y baseline de P3 en MLflow |

La primera ejecución de `0.03` descarga los CSV faltantes y tarda varios minutos; las siguientes reutilizan los archivos existentes (el notebook completo corre en menos de un minuto).

**MLflow del equipo:** https://mlflow.equipo7mlops.com/ (experimento `p3`, run `baseline`).

## Propósito

Responder con ML tres preguntas operativas de MiBici, sobre una base reproducible (ingesta versionada, features reutilizables y experimentos registrados en MLflow) que las siguientes entregas del curso extenderán.

## Preguntas del proyecto

| # | Pregunta de negocio | Tarea | Usuario |
|---|---|---|---|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Operador del sistema (planeación de flota) |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes? | Clasificación | Equipo de rebalanceo |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Planeación urbana y expansión |

> Los datos no incluyen inventario de bicicletas por estación; para P2 el riesgo se aproxima con el balance neto de viajes (salidas − llegadas) acumulado frente a la capacidad de la estación.

Detalle de cada pregunta (qué se predice, relevancia y decisión que habilita): [ML Canvas](references/ml-canvas.md).

## Datos

### Fuente

[Datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/), sistema de bicicleta pública del Área Metropolitana de Guadalajara. El portal publica **solo CSV**:

- **Viajes:** `datos_abiertos_AAAA_MM.csv`, un archivo por mes desde dic-2014 (138 archivos publicados).
- **Estaciones:** `nomenclatura_AAAA_MM.csv`, una sola versión vigente (`nomenclatura_2026_09.csv`).
- Los datos **no se versionan en Git** (ver `.gitignore`); se reconstruyen desde el portal y se convierten a Parquet (ver los notebooks `0.02` y `0.03`).
- **Enlaces, registros y SHA-256 de cada archivo:** [`references/fuentes-datos.csv`](references/fuentes-datos.csv).

### Alcance del análisis

| Concepto | Valor |
|---|---|
| Rango acordado por el equipo | enero 2020 a septiembre 2026 |
| Viajes | **26,389,162** en 79 archivos mensuales |
| Primer y último inicio de viaje | 2020-01-01 06:02:20 a 2026-09-30 23:59:48 |
| Usuarios distintos | 81,115 |
| Estaciones en los viajes | 475 (de 484 en la nomenclatura) |
| Mes con menos viajes | 2020-04 (129,276), inicio de la pandemia de COVID-19 |
| Mes con más viajes | 2026-09 (490,056) |
| Tamaño | 2.1 GB en CSV, 0.6 GB en Parquet |

> **Meses faltantes:** 2023-04 y 2023-05 ya no están publicados en el portal (se publicaban el 13-oct-2026 y desaparecieron sin aviso). El análisis trabaja con 79 de los 81 meses del rango y el hueco se tratará en la limpieza.

Viajes por año (2026 llega solo hasta septiembre):

| 2020 | 2021 | 2022 | 2023* | 2024 | 2025 | 2026 |
|---:|---:|---:|---:|---:|---:|---:|
| 2,866,072 | 3,184,410 | 4,202,942 | 3,595,720 | 4,645,921 | 4,532,032 | 3,362,065 |

\* 2023 no incluye abril ni mayo.

### Esquema de viajes

| Columna (CSV original) | Columna en Parquet | Tipo | Nulos | Descripción |
|---|---|---|---:|---|
| `Viaje_Id` | `viaje_id` | entero | 0 | Identificador del viaje |
| `Usuario_Id` | `usuario_id` | entero | 0 | Identificador del usuario (anonimizado) |
| `Genero` | `genero` | categoría (`M`, `F`) | 135,232 (0.51 %) | Género del usuario |
| `Año_de_nacimiento` | `anio_de_nacimiento` | entero | 33,796 (0.13 %) | Año de nacimiento |
| `Inicio_del_viaje` | `inicio_del_viaje` | fecha y hora | 0 | Inicio del viaje |
| `Fin_del_viaje` | `fin_del_viaje` | fecha y hora | 0 | Fin del viaje |
| `Origen_Id` | `origen_id` | entero | 0 | Estación de origen |
| `Destino_Id` | `destino_id` | entero | 0 | Estación de destino |

### Esquema de estaciones

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | entero | Identificador de la estación (único, de 2 a 519) |
| `name` | texto | Código y cruce de calles, por ejemplo `(GDL-001) C. Epigmenio Glez./ Av. 16 de Sept.` |
| `latitude`, `longitude` | decimal | Ubicación (entre 20.636 y 20.740 N, y entre -103.433 y -103.297 O) |
| `dpcapacity` | entero | Capacidad en anclajes; promedio 16.9, de 0 a 39, **8,171 anclajes en total** |

La nomenclatura tiene **484 estaciones** sin nulos.

### Hallazgos de calidad

Detectados en el EDA. El tratamiento aplicado a cada uno se documenta en *Limpieza y partición*, más abajo.

| Hallazgo | Evidencia | Notebook |
|---|---|---|
| **Nombre de columna corrupto** | 2024-02 y 2024-05 traen `A}äe_nacimiento` en lugar de `Año_de_nacimiento` | 0.02 |
| **Codificación mezclada** | 43 archivos (54.4 %) en latin-1 y 36 (45.6 %) en UTF-8, sin regla por fecha | 0.02 |
| **Formatos de fecha mezclados** | 76 archivos `AAAA-MM-DD hh:mm:ss`; 2023-11, 2024-09 y 2024-10 usan `DD/MM/AAAA hh:mm` | 0.02 |
| **Viajes duplicados** | 758 filas completamente repetidas (1,516 filas comparten un `viaje_id` repetido) | 0.02, 0.03 |
| **Años de nacimiento imposibles** | 1,255 anteriores a 1900 (mínimo 1) y 2,093 posteriores a 2010 (máximo 2023) | 0.01 |
| **Duración cero** | 53,581 viajes con inicio igual a fin | 0.03 |
| **Duración extrema** | 157 viajes de más de 24 horas (máximo 469 horas, cerca de 20 días); la mediana es 9.4 min y el percentil 99, 32.8 min | 0.01, 0.03 |
| **Estaciones sin nomenclatura** | 3 ids (124, 405 y 407) aparecen en viajes y no en la nomenclatura: afecta a 3,961 viajes | 0.01 |
| **Estaciones sin capacidad** | 4 estaciones con `dpcapacity = 0`, que afecta el cálculo de riesgo de P2 | 0.03 |
| **Nomenclatura única** | Solo existe la versión 2026-09; estaciones retiradas o reubicadas no se pueden reconstruir para años anteriores | 0.02 |
| **Sin servicio de madrugada** | Solo 159 viajes entre la 1:00 y las 4:59 (0.0006 %); la hora 0 tiene 141,841 (0.5 %) | 0.03 |
| **Estaciones nuevas** | 109 estaciones no tienen viajes en entrenamiento y aparecen en prueba (3.8 % de sus viajes); 12 de la nomenclatura no registran ninguno | 0.03 |
| **Capacidad sin relación con la demanda** | Correlación de Spearman entre salidas y capacidad de −0.02 | 0.03 |
| **Demanda concentrada** | El 10 % de las estaciones (48) concentra el 37.8 % de los viajes; 74 tienen menos de 1,000 salidas | 0.03 |
| **Sin inventario de bicicletas** | Los datos no registran bicis por estación (ver nota en Preguntas) | — |

### Privacidad

Los viajes incluyen `usuario_id`, género y año de nacimiento. Son datos personales aunque estén anonimizados; no se publican valores individuales en los notebooks ni en este README, solo agregados.

Rango de fechas y su justificación, variables y construcción de cada objetivo, condiciones de uso publicadas por MiBici y estrategia de partición temporal: [Definición de los datos](references/definicion-datos.md).

### Patrones del EDA

| Tema | Hallazgo |
|---|---|
| **Tendencia** | El promedio mensual crece de 238,839 viajes (2020) a 387,160 (2024) y se estabiliza: 377,669 en 2025 y 373,563 en 2026 |
| **Pandemia** | Cae de 367,864 viajes (feb-2020) a 129,276 (abr-2020), −64.9 %; el nivel de feb-2020 se supera hasta mar-2022 |
| **Días laborales** | Dos picos: 8:00 (8.3 % de los viajes) y 18:00 (10.5 %) |
| **Fin de semana** | Un solo pico amplio entre 11:00 y 14:00; sábado y domingo concentran el 20.3 % de los viajes |
| **Desbalance** | Acumulan bicicletas estaciones del centro (GDL-049, +145,871) y pierden las de Av. Vallarta (GDL-127, −95,538) |
| **Pendiente** | 2026-09 (490,056) es el máximo de la serie y +30 % contra 2025-09; falta verificar que no sea una carga atípica |

### Limpieza y partición

Reglas aplicadas en `0.03` (sección 3), cada una con su justificación en el notebook:

| Hallazgo | Decisión | Registros |
|---|---|---:|
| Viajes con `viaje_id` repetido (filas idénticas) | Eliminar | 758 |
| Origen o destino sin nomenclatura | Excluir | 3,961 |
| Duración cero | Excluir (sensibilidad por probar en el modelado) | 53,580 |
| Duración mayor a 24 h | Excluir | 157 |
| `usuario_id`, `genero`, `anio_de_nacimiento` | Descartar (minimización de datos personales) | — |
| Hueco 2023-04 y 2023-05; horas 1:00 a 4:59 | No imputar; excluir de la tabla de demanda | — |

**Resultado:** 58,456 registros retirados (0.222 %); quedan **26,330,706 viajes** en `data/interim/viajes_limpios_2020-01_2026-09.parquet`.

**Partición temporal** con una brecha de 24 horas antes de cada frontera, para que ninguna etiqueta use datos del conjunto siguiente:

| Conjunto | Periodo | Viajes | % |
|---|---|---:|---:|
| train | 2020-01-01 a 2024-09-29 | 17,291,138 | 65.7 |
| valid | 2024-10-01 a 2025-09-29 | 4,520,744 | 17.2 |
| test | 2025-10-01 a 2026-09-30 | 4,489,362 | 17.0 |
| brecha | 24 h antes de cada frontera | 29,462 | 0.1 |

**Preprocesamiento:** `data/processed/estaciones_train.parquet` (484 estaciones, calculada solo con entrenamiento) y `data/processed/demanda_estacion_hora.parquet` (15,812,633 filas para 472 estaciones; 44.3 % de las estación-horas sin viajes).

## Arquitectura

![Arquitectura del sistema](references/arquitectura.png)

| Componente | Herramienta | Justificación |
|---|---|---|
| Ingesta (CLI) | Python + Typer + loguru | Comando único, sin pasos manuales |
| Almacenamiento | Parquet (pyarrow) | Columnar y comprimido para millones de viajes |
| Versionado de datos | Manifiesto JSON + `hashlib` en Git | Trazabilidad simple: nº de registros y SHA-256 por archivo |
| Procesamiento | pandas | Suficiente para agregados por estación/hora |
| Modelado | scikit-learn, XGBoost, Optuna | Baselines y candidatos para regresión, clasificación y clustering |
| Experimentos | MLflow Tracking | Parámetros, métricas, tags (equipo, pregunta, commit) y artefactos |

Etapas por pregunta (limpieza, agregaciones, features, entrenamiento y evaluación): [Diseño del pipeline de ML](references/pipeline-ml.md).

## Modelo base de P3 (clustering de estaciones)

Notebook `0.03`, Paso 4. K-Means sobre variables estandarizadas, ajustado solo con el conjunto de entrenamiento.

| Elemento | Valor |
|---|---|
| **Estaciones agrupadas** | 362 (con al menos 1,000 salidas en entrenamiento) |
| **Variables** | 44: perfil horario laboral y de fin de semana (20 horas de servicio cada uno), volumen diario, balance neto sobre salidas, proporción de viajes circulares y duración mediana |
| **`k`** | 2, elegido por silueta entre 2 y 10 |
| **Silueta** | 0.146 (estructura débil: los grupos se solapan) |
| **Davies-Bouldin / Calinski-Harabasz** | 2.13 / 69.6 |
| **Estabilidad (ARI con el año de validación)** | 0.68; 31 de 359 estaciones (8.6 %) cambian de grupo |

| Grupo | Estaciones | Perfil | Volumen mediano |
|---|---:|---|---|
| 0 | 166 | Pico vespertino a las 18:00 (12.7 % de las salidas laborales); centro y poniente; tiende a perder bicicletas (−9.7 %) | 29.3 viajes por día |
| 1 | 196 | Pico matutino a las 8:00 (10.9 %); periferia y franja centro-norte; balance +3.3 % | 19.7 viajes por día |

**Límites:** con `k = 2` la granularidad no basta para decidir ampliaciones o reubicaciones, y la separación se explica casi solo por horario y volumen. Las 109 estaciones nuevas quedan fuera del agrupamiento. Siguientes pasos: PCA o ponderación de variables, `k` de 4 a 5 y HDBSCAN. La lectura "residencial frente a zonas de trabajo" es una hipótesis por verificar.

### Registro en MLflow

- **Servidor:** https://mlflow.equipo7mlops.com/ · **Experimento:** `p3` · **Run:** `baseline`.
- **Registrado:** 14 parámetros, 11 métricas, etiquetas (equipo, pregunta, commit, autor), artefactos (asignación por estación, resumen de grupos, mapa y silueta por `k`) y el modelo (`StandardScaler` + `KMeans`).
- **Trazabilidad:** el parámetro `data_version` son los primeros 16 caracteres del SHA-256 de `references/fuentes-datos.csv`.
- **Copia local:** los mismos artefactos se guardan en `reports/figures/p3_*` y sí se versionan en Git.

> **Al crear un experimento nuevo en este servidor, usar `artifact_location="mlflow-artifacts:/<nombre>"`.** Sin esa opción el servidor asigna una ruta local suya (`/home/ec2-user/mlflow/artifacts/...`) y la subida de artefactos falla con `Errno 45`. El servidor es MLflow 3.1.4 y el `requirements.txt` fija el cliente en 3.16.0: conviene alinearlos.

## Estructura de carpetas

```
├── data/              <- (ignorado por Git)
│   ├── raw/           <- CSV originales del portal (viajes/ y estaciones/)
│   ├── external/      <- Parquet mensuales por viaje
│   ├── interim/       <- viajes_limpios_2020-01_2026-09.parquet
│   └── processed/     <- estaciones_train.parquet y demanda_estacion_hora.parquet
├── docs/              <- Documentación (mkdocs)
├── models/            <- Modelos serializados (ignorado por Git)
├── notebooks/         <- Convención FASE.NUMERO-INICIALES-descripcion.ipynb
│   ├── 0.01-fat-exploracion-inicial.ipynb    <- EDA sobre Parquet: nulos, años de nacimiento, duración y conteo de entradas/salidas por estación
│   ├── 0.02-dsh-exploracion-carga.ipynb      <- Descarga del portal, inventario por archivo (codificación, esquema, fechas), unión y Parquet
│   ├── 0.03-lasd-exploracion-carga.ipynb     <- Parquet mensuales, EDA, limpieza, partición temporal, preprocesamiento y baseline de P3 en MLflow
│   └── 0.1-equipo7-mibici.ipynb              <- Plantilla del entregable (pasos 1 a 4 y conclusiones)
├── references/        <- Documentos de diseño
│   ├── ml-canvas.md          <- ML Canvas y descripción de P1, P2 y P3
│   ├── definicion-datos.md   <- Rango, fuente, variables, objetivos, condiciones de uso y partición
│   ├── pipeline-ml.md        <- Pipeline por pregunta: limpieza, features, entrenamiento y evaluación
│   ├── pipeline-ml.png       <- Diagrama del pipeline por pregunta (fuente editable: pipeline-ml.svg)
│   ├── fuentes-datos.csv     <- URLs, registros y SHA-256 de los archivos del portal
│   └── arquitectura.png      <- Diagrama de arquitectura
├── reports/figures/   <- Gráficas y tablas generadas (p3_asignacion_grupos.csv, p3_resumen_grupos.csv, p3_variables.json, p3_mapa_grupos.png, p3_silueta_por_k.png)
├── tc5061_proyecto_equipo07/   <- Módulo: config.py, dataset.py, features.py, modeling/, plots.py
├── Makefile
├── pyproject.toml
└── requirements.txt   <- Dependencias con versiones fijadas
```

## Flujo de trabajo con Git (GitHub Flow)

- `main` está protegida: todo cambio entra por Pull Request con al menos una revisión.
- Ramas desde `main`: `feat/<tema>`, `fix/<tema>`, `docs/<tema>`, `chore/<tema>`.
- Commits con [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `config:`, `refactor:`.
- Cada integrante abre al menos un PR y revisa al menos uno de otro integrante.

## Equipo

| Integrante | Matrícula | Rol |
|---|---|---|
| Daniel Sánchez Hernández | A01840324 | *por definir* |
| Luis Arturo Sánchez Davila | A01840576 | *por definir* |
| Fernando Arreola Treviño | A01840331 | *por definir* |
