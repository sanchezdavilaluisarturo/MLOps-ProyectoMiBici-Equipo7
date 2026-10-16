"""Pruebas de las funciones p3_* de features.py."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features


def p3_viajes_laborales() -> pd.DataFrame:
    """Estación 1: 3 salidas laborales (8:00, 8:30, 18:00) y 1 de sábado; estación 2: 1 salida."""
    inicio = pd.to_datetime(
        [
            "2024-01-08 08:00",  # lunes
            "2024-01-09 08:30",  # martes
            "2024-01-10 18:00",  # miércoles
            "2024-01-13 11:00",  # sábado
            "2024-01-08 09:00",  # lunes, estación 2
        ]
    )
    return pd.DataFrame(
        {
            "inicio_del_viaje": inicio,
            "fin_del_viaje": inicio + pd.Timedelta(minutes=10),
            "origen_id": [1, 1, 1, 1, 2],
            "destino_id": [2, 1, 2, 2, 1],
            "duracion_min": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )


def test_p3_variables_de_tiempo_agrega_columnas_en_su_lugar():
    viajes = p3_viajes_laborales().drop(columns="duracion_min")
    devuelto = features.p3_variables_de_tiempo(viajes)
    assert devuelto is viajes
    assert viajes["duracion_min"].tolist() == [10.0] * 5
    assert viajes["hora"].iloc[0] == 8 and viajes["dia_semana"].iloc[3] == 5  # 13-ene es sábado


def test_p3_perfil_de_estaciones_tiene_el_esquema_esperado():
    perfil = features.p3_perfil_de_estaciones(p3_viajes_laborales())
    n_horas = len(cfg.P3_HORAS_SERVICIO)
    laborales = [c for c in perfil.columns if c.startswith("lab_")]
    fines = [c for c in perfil.columns if c.startswith("fin_")]
    assert len(laborales) == n_horas and len(fines) == n_horas
    assert {"salidas", "log_viajes_por_dia", "neto_sobre_salidas"} <= set(perfil.columns)
    assert set(perfil.index) == {1, 2}

    est1 = perfil.loc[1]
    assert est1["salidas"] == 4
    # Las proporciones de cada bloque suman 1 (hay salidas laborales y de fin de semana)
    assert est1[laborales].sum() == pytest.approx(1.0)
    assert est1[fines].sum() == pytest.approx(1.0)
    assert est1["lab_h08"] == pytest.approx(2 / 3)
    assert est1["lab_h18"] == pytest.approx(1 / 3)
    # Estación 1: 4 salidas y 1 llegada propia (viaje circular) + 1 desde la estación 2
    assert est1["neto_sobre_salidas"] == pytest.approx((2 - 4) / 4)
    assert est1["prop_circulares"] == pytest.approx(1 / 4)


def p3_limpios_con_particion() -> pd.DataFrame:
    viajes = p3_viajes_laborales()
    viajes["particion"] = pd.Categorical(
        ["train"] * 5, categories=["train", "valid", "test", "brecha"]
    )
    viajes["periodo_archivo"] = "2024-01"
    return viajes


def test_p3_tabla_estaciones_cuenta_solo_entrenamiento():
    estaciones = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "latitude": [20.0, 20.1, 20.2],
            "longitude": [-103.0, -103.1, -103.2],
            "dpcapacity": [15, 0, 20],
        }
    )
    tabla = features.p3_tabla_estaciones(p3_limpios_con_particion(), estaciones)
    assert tabla.loc[1, "salidas_train"] == 4 and tabla.loc[2, "salidas_train"] == 1
    assert bool(tabla.loc[3, "activa_en_train"]) is False
    assert bool(tabla.loc[2, "capacidad_cero"]) is True
    assert tabla.loc[1, "neto_train"] == tabla.loc[1, "llegadas_train"] - 4


def test_p3_demanda_estacion_hora_excluye_horas_sin_servicio_y_no_inventa_ceros():
    limpios = pd.DataFrame(
        {
            "inicio_del_viaje": pd.to_datetime(
                [
                    "2024-01-10 08:30",
                    "2024-01-10 08:50",
                    "2024-01-10 09:10",
                    "2024-01-20 03:15",  # hora sin servicio: queda fuera de la rejilla
                    "2024-01-20 10:00",
                ]
            ),
            "origen_id": pd.array([7, 7, 7, 9, 9], dtype="Int32"),
            "periodo_archivo": pd.Categorical(["2024-01"] * 5),
        }
    )
    tabla, excluidos = features.p3_demanda_estacion_hora(limpios, "2024-01", "2024-01")
    assert excluidos == 1
    assert not tabla["hora_inicio"].dt.hour.isin(cfg.HORAS_SIN_SERVICIO).any()
    assert tabla["viajes"].sum() == 4
    # Los ceros empiezan en la primera hora activa de cada estación, no antes
    primera_7 = tabla.loc[tabla["origen_id"] == 7, "hora_inicio"].min()
    assert primera_7 == pd.Timestamp("2024-01-10 08:00")
    assert tabla.loc[(tabla["origen_id"] == 7) & (tabla["hora_inicio"] == primera_7), "viajes"].item() == 2
    assert set(tabla["particion"].unique()) == {"train"}


def test_p3_viajes_por_mes_y_conjunto():
    limpios = p3_limpios_con_particion()
    tabla = features.p3_viajes_por_mes_y_conjunto(limpios)
    assert list(tabla.columns) == ["train", "brecha", "valid", "test"]
    assert tabla.loc["2024-01", "train"] == 5
    assert np.all(tabla[["brecha", "valid", "test"]].to_numpy() == 0)
