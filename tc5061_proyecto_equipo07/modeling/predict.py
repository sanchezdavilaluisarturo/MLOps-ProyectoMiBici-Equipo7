"""Inferencia de P3 (clustering de estaciones): asignar grupos con un modelo ya ajustado.

Se usa para medir la estabilidad de los grupos en otro periodo (el año de validación) y, más
adelante, para asignar grupo a las estaciones nuevas, que no tienen historia en entrenamiento.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def p3_reasignar_grupos(
    modelo, caracteristicas: pd.DataFrame, columnas_modelo: list[str]
) -> np.ndarray:
    """Regresa el grupo más cercano de cada estación según el modelo ajustado en entrenamiento.

    `caracteristicas` trae las variables de otro periodo (por ejemplo, validación); el modelo
    ya incluye su escalador, así que no hay que estandarizarlas a mano.
    """
    return modelo.predict(caracteristicas[columnas_modelo])


def p3_estabilidad_ari(grupo_original, grupo_reasignado) -> float:
    """Índice de Rand ajustado entre dos asignaciones (1 = idénticas, 0 = azar)."""
    return float(adjusted_rand_score(grupo_original, grupo_reasignado))
