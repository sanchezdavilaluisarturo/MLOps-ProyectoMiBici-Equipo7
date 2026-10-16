"""Configuración compartida del proyecto: rutas, fuente de datos, esquema de viajes y MLflow."""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Variables de entorno del archivo .env, si existe (por ejemplo, MLFLOW_TRACKING_URI)
load_dotenv()

# Rutas del proyecto (estructura CCDS)
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REFERENCES_DIR = PROJ_ROOT / "references"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Fuente de los datos
URL_PORTAL = "https://mibici.net/es/datos-abiertos/"
# El portal responde 406 al User-Agent por defecto de requests
ENCABEZADOS_HTTP = {
    "User-Agent": "Mozilla/5.0 (proyecto academico MLOps TC5061)",
    "Accept": "text/html,text/csv,*/*",
}

# Rango acordado por el equipo (meses inclusivos)
MES_INICIO = "2020-01"
MES_FIN = "2026-09"
# El portal retiró estos meses el 14-oct-2026 y los volvió a publicar el 15-oct-2026.
# Se toleran si vuelven a faltar; cualquier otro hueco detiene la ingesta.
MESES_SIN_PUBLICAR = ["2023-04", "2023-05"]

# Archivos de datos
RUTA_VIAJES_CRUDOS = RAW_DATA_DIR / "viajes"
RUTA_ESTACIONES_CRUDAS = RAW_DATA_DIR / "estaciones"
RUTA_MANIFIESTO = REFERENCES_DIR / "fuentes-datos.csv"
RUTA_VIAJES_INTERIM = INTERIM_DATA_DIR / f"viajes_{MES_INICIO}_{MES_FIN}.parquet"
RUTA_VIAJES_LIMPIOS = PROCESSED_DATA_DIR / f"viajes_limpios_{MES_INICIO}_{MES_FIN}.parquet"

# Esquema de los viajes: encabezado original -> nombre normalizado.
# 2024-02 y 2024-05 traen la "ñ" dañada en el año de nacimiento.
COLUMNAS_NORMALIZADAS = {
    "Viaje_Id": "viaje_id",
    "Usuario_Id": "usuario_id",
    "Genero": "genero",
    "Año_de_nacimiento": "anio_de_nacimiento",
    "A}äe_nacimiento": "anio_de_nacimiento",
    "Inicio_del_viaje": "inicio_del_viaje",
    "Fin_del_viaje": "fin_del_viaje",
    "Origen_Id": "origen_id",
    "Destino_Id": "destino_id",
}
COLUMNAS_VIAJES = [
    "viaje_id",
    "usuario_id",
    "genero",
    "anio_de_nacimiento",
    "inicio_del_viaje",
    "fin_del_viaje",
    "origen_id",
    "destino_id",
]
TIPOS_ENTEROS = {
    "viaje_id": "Int64",
    "usuario_id": "Int64",
    "anio_de_nacimiento": "Int16",
    "origen_id": "Int32",
    "destino_id": "Int32",
}

# Limpieza común (ver dataset.limpiar)
DURACION_MAXIMA_MIN = 24 * 60
# Un año de nacimiento que implica un usuario de 5 años o menos se considera inválido
EDAD_MINIMA = 6

# Entre la 1:00 y las 4:59 el servicio no opera (194 viajes de 27086557 antes de limpiar)
HORAS_SIN_SERVICIO = [1, 2, 3, 4]

SEMILLA = 42

# MLflow: el experimento incluye el número de equipo; la URI viene del .env
EXPERIMENTO_MLFLOW = "mibici-equipo07"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Paleta Okabe-Ito (apta para daltonismo) para las gráficas del proyecto
AZUL = "#0072B2"
NARANJA = "#E69F00"
VERDE = "#009E73"
AMARILLO = "#F0E442"
BERMELLON = "#D55E00"
GRIS = "#999999"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
