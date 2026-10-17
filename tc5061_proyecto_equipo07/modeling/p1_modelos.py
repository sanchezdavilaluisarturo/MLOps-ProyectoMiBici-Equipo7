"""Modelos de P1: baseline estacional y candidatos de boosting con pérdida Poisson.

Por qué estos candidatos (según el EDA de P1):
- El target es un conteo con 44% de ceros y varianza ~4 veces la media: la pérdida Poisson
  modela conteos y nunca predice viajes negativos.
- La tabla tiene ~16M de filas: los dos usan histogramas y entrenan en minutos.
- Los rezagos de estaciones nuevas vienen vacíos (NaN): los dos los manejan sin imputar.
- La demanda depende de combinaciones (estación × hora × día): los árboles las capturan
  sin crearlas a mano.
HGB viene en sklearn; XGBoost es el estándar de la industria y sirve para comparar.

Las clases viven en este módulo (y no en train.py) para que los modelos guardados en MLflow se
puedan volver a cargar: train.py se ejecuta como `__main__` y un objeto definido ahí quedaría
ligado a ese nombre.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.features import P1NivelEstacion

# Nombre del run, etapa y familia de cada modelo (etiquetas `stage` y `model_family` en MLflow)
P1_MODELOS = {
    "baseline": ("p1-baseline-promedio-4sem", "baseline", "promedio_historico"),
    "hgb": ("p1-hgb-poisson", "candidato", "hist_gradient_boosting"),
    "xgb": ("p1-xgb-poisson", "candidato", "xgboost"),
}

# Descripción de cada run en MLflow (caja "Description")
P1_DESCRIPCIONES = {
    "baseline": "Promedio de la misma hora en las 4 semanas anteriores; si la estación no tiene "
    "historia, usa su nivel de train o el del sistema. Referencia que los candidatos deben superar.",
    "hgb": "HistGradientBoosting con pérdida Poisson sobre el nivel histórico de cada estación "
    "aprendido en train; predice los viajes del día siguiente por estación-hora.",
    "xgb": "XGBoost con pérdida Poisson sobre el nivel histórico de cada estación aprendido en "
    "train; predice los viajes del día siguiente por estación-hora.",
}


class P1BaselinePromedio(BaseEstimator, RegressorMixin):
    """Predice el promedio de la misma hora en las semanas anteriores (baseline estacional).

    Si una estación-hora no tiene historia (estación nueva), usa su nivel de train y, si tampoco
    existe, el nivel del sistema para esa hora y día. Tiene la misma interfaz `fit`/`predict` que
    los modelos candidatos, así que se evalúa y registra con las mismas funciones.
    """

    def fit(self, X: pd.DataFrame, y) -> "P1BaselinePromedio":
        self.nivel_ = P1NivelEstacion().fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        niveles = self.nivel_.transform(X)
        prediccion = X[f"viajes_promedio_{cfg.P1_SEMANAS_PROMEDIO}sem"].to_numpy(
            "float64", copy=True
        )
        for respaldo in ["viajes_nivel_estacion", "viajes_nivel_sistema"]:
            faltante = np.isnan(prediccion)
            prediccion[faltante] = niveles[respaldo].to_numpy("float64")[faltante]
        return np.nan_to_num(prediccion, nan=0.0)


def p1_construir_modelo(clave: str, semilla: int = cfg.SEMILLA):
    """Regresa el modelo sin entrenar: baseline, HGB o XGBoost (estos dos con el nivel de train)."""
    if clave == "baseline":
        return P1BaselinePromedio()
    if clave == "hgb":
        regresor = HistGradientBoostingRegressor(
            **cfg.P1_PARAMETROS_HGB, categorical_features="from_dtype", random_state=semilla
        )
    elif clave == "xgb":
        regresor = XGBRegressor(
            **cfg.P1_PARAMETROS_XGB, enable_categorical=True, random_state=semilla
        )
    else:
        raise ValueError(f"Modelo de P1 desconocido: {clave} (opciones: {', '.join(P1_MODELOS)})")
    return Pipeline([("nivel", P1NivelEstacion()), ("modelo", regresor)])
