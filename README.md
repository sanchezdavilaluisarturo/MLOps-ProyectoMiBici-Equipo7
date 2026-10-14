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

## Propósito

Responder con ML tres preguntas operativas de MiBici, sobre una base reproducible (ingesta versionada, features reutilizables y experimentos registrados en MLflow) que las siguientes entregas del curso extenderán.

## Preguntas del proyecto

| # | Pregunta de negocio | Tarea | Usuario |
|---|---|---|---|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Operador del sistema (planeación de flota) |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes? | Clasificación | Equipo de rebalanceo |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Planeación urbana y expansión |

> Los datos no incluyen inventario de bicicletas por estación; para P2 el riesgo se aproxima con el balance neto de viajes (salidas − llegadas) acumulado frente a la capacidad de la estación.

## Datos

- **Viajes:** `datos_abiertos_AAAA_MM.csv` (mensuales, desde dic-2014), del portal de datos abiertos de MiBici.
- **Estaciones:** `nomenclatura_AAAA_MM.csv` (id, nombre, latitud, longitud, capacidad).
- Rango de fechas, número de viajes/estaciones, condiciones de uso y tratamiento de datos personales: *(completar tras ejecutar la ingesta)*.
- Los datos **no se versionan en Git** (ver `.gitignore`); se reconstruyen con el comando de ingesta y se verifican contra el manifiesto.

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

## Estructura de carpetas

```
├── data/              <- raw / interim / processed / external (ignorado por Git)
├── docs/              <- Documentación (mkdocs)
├── models/            <- Modelos serializados (ignorado por Git)
├── notebooks/         <- Convención FASE.NUMERO-INICIALES-descripcion.ipynb
├── references/        <- ML Canvas y diagrama de arquitectura
├── reports/figures/   <- Gráficas generadas
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
