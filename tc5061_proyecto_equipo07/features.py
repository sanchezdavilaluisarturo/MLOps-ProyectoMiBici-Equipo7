"""Features de P3 (clustering de estaciones): variables por estación y tablas de preprocesamiento.

Entrada: data/processed/viajes_modelado_AAAA-MM_AAAA-MM.parquet (`dataset p3-particionar`).
Salidas en data/processed/: estaciones_train.parquet y demanda_estacion_hora.parquet.

Uso: python -m tc5061_proyecto_equipo07.features p3-features
"""

from loguru import logger
import numpy as np
import pandas as pd
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset

app = typer.Typer()


def p3_variables_de_tiempo(viajes: pd.DataFrame) -> pd.DataFrame:
    """Agrega duracion_min, hora y dia_semana (0 = lunes); `viajes` se modifica en su lugar."""
    viajes["duracion_min"] = (
        viajes["fin_del_viaje"] - viajes["inicio_del_viaje"]
    ).dt.total_seconds() / 60
    viajes["hora"] = viajes["inicio_del_viaje"].dt.hour.astype("int8")
    viajes["dia_semana"] = viajes["inicio_del_viaje"].dt.dayofweek.astype("int8")
    return viajes


def p3_perfil_de_estaciones(viajes_periodo: pd.DataFrame) -> pd.DataFrame:
    """Regresa una fila por estación con su perfil horario y sus indicadores de uso.

    Variables: proporción de salidas por hora de servicio en días laborales (`lab_hHH`) y en fin
    de semana (`fin_hHH`), volumen diario (`log_viajes_por_dia`), balance neto sobre salidas,
    proporción de viajes circulares y duración mediana. Requiere `duracion_min`.
    """
    inicio = viajes_periodo["inicio_del_viaje"]
    base = pd.DataFrame(
        {
            "estacion": viajes_periodo["origen_id"].to_numpy(),
            "hora": inicio.dt.hour.to_numpy(),
            "laboral": (inicio.dt.dayofweek < 5).to_numpy(),
        }
    )
    conteo = (
        base.groupby(["estacion", "laboral", "hora"])
        .size()
        .unstack("hora", fill_value=0)
        .reindex(columns=cfg.P3_HORAS_SERVICIO, fill_value=0)
    )
    estaciones_ids = conteo.index.get_level_values("estacion").unique()

    def perfil(es_laboral, prefijo):
        bloque = conteo.xs(es_laboral, level="laboral").reindex(estaciones_ids, fill_value=0)
        proporcion = bloque.div(bloque.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        proporcion.columns = [f"{prefijo}_h{hora:02d}" for hora in proporcion.columns]
        return proporcion

    perfiles = pd.concat([perfil(True, "lab"), perfil(False, "fin")], axis=1)

    salidas = viajes_periodo["origen_id"].value_counts()
    llegadas = viajes_periodo["destino_id"].value_counts()
    dias_activos = (
        pd.DataFrame(
            {
                "estacion": viajes_periodo["origen_id"].to_numpy(),
                "dia": inicio.dt.floor("D").to_numpy(),
            }
        )
        .groupby("estacion")["dia"]
        .nunique()
    )
    circulares = (
        (viajes_periodo["origen_id"] == viajes_periodo["destino_id"])
        .groupby(viajes_periodo["origen_id"])
        .mean()
    )
    duracion = viajes_periodo.groupby("origen_id", observed=True)["duracion_min"].median()

    indicadores = pd.DataFrame(
        {
            "salidas": salidas,
            "log_viajes_por_dia": np.log1p(salidas / dias_activos),
            "neto_sobre_salidas": (llegadas.reindex(salidas.index).fillna(0) - salidas) / salidas,
            "prop_circulares": circulares,
            "duracion_mediana_min": duracion,
        }
    ).reindex(estaciones_ids)
    indicadores.index = indicadores.index.astype(int)
    perfiles.index = perfiles.index.astype(int)
    return perfiles.join(indicadores)


def p3_construir_caracteristicas(
    limpios: pd.DataFrame,
    min_salidas_train: int = cfg.P3_MIN_SALIDAS_TRAIN,
    min_salidas_valid: int = cfg.P3_MIN_SALIDAS_VALID,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Regresa las variables por estación de train y de valid, y los nombres de las columnas.

    Con pocos viajes el perfil horario es ruido, así que se filtran las estaciones con menos de
    `min_salidas_train` salidas en train (y de `min_salidas_valid` en valid, para la prueba de
    estabilidad). `limpios` debe traer `particion` y `duracion_min`.
    """
    train = p3_perfil_de_estaciones(limpios[limpios["particion"] == "train"])
    train = train[train["salidas"] >= min_salidas_train]
    columnas_modelo = [columna for columna in train.columns if columna != "salidas"]
    valid = p3_perfil_de_estaciones(limpios[limpios["particion"] == "valid"])
    valid = valid[valid["salidas"] >= min_salidas_valid]
    return train, valid, columnas_modelo


def p3_tabla_estaciones(limpios: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Regresa capacidad, ubicación y actividad por estación, calculada solo con train."""
    entrenamiento = limpios[limpios["particion"] == "train"]
    actividad = pd.DataFrame(
        {
            "salidas_train": entrenamiento["origen_id"].value_counts(),
            "llegadas_train": entrenamiento["destino_id"].value_counts(),
        }
    )
    tabla = (
        estaciones.set_index("id")[["name", "latitude", "longitude", "dpcapacity"]]
        .join(actividad)
        .fillna({"salidas_train": 0, "llegadas_train": 0})
    )
    columnas = ["salidas_train", "llegadas_train"]
    tabla[columnas] = tabla[columnas].astype(int)
    tabla["neto_train"] = tabla["llegadas_train"] - tabla["salidas_train"]
    tabla["capacidad_cero"] = tabla["dpcapacity"] == 0
    tabla["activa_en_train"] = tabla["salidas_train"] > 0
    return tabla


def p3_viajes_por_mes_y_conjunto(limpios: pd.DataFrame) -> pd.DataFrame:
    """Regresa los viajes de cada mes (filas) por conjunto de la partición (columnas)."""
    return (
        limpios.groupby(["periodo_archivo", "particion"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "brecha", "valid", "test"], fill_value=0)
    )


def p3_demanda_estacion_hora(
    limpios: pd.DataFrame, mes_inicio: str = cfg.MES_INICIO, mes_fin: str = cfg.MES_FIN
) -> tuple[pd.DataFrame, int]:
    """Regresa (demanda por estación-hora, estación-horas con viajes fuera de la rejilla).

    La rejilla evita inventar datos: los ceros comienzan en la primera hora activa de cada
    estación, y se excluyen las horas sin servicio (1:00 a 4:59) y las de meses sin datos.
    """
    hora_inicio = limpios["inicio_del_viaje"].dt.floor("h")
    demanda = (
        pd.DataFrame({"origen_id": limpios["origen_id"], "hora_inicio": hora_inicio})
        .groupby(["origen_id", "hora_inicio"], observed=True)
        .size()
        .rename("viajes")
        .reset_index()
    )

    # Meses sin datos: se calculan de los datos, no de una lista fija
    meses_sin_datos = dataset.p3_meses_sin_datos(
        limpios["periodo_archivo"].astype(str).unique(), mes_inicio, mes_fin
    )

    # Horas válidas: dentro del rango, con servicio y en meses con datos
    ultima_hora = pd.Timestamp(mes_fin + "-01") + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
    horas = pd.date_range(f"{mes_inicio}-01", ultima_hora, freq="h")
    meses_hora = horas.strftime("%Y-%m")
    horas_validas = horas[
        ~horas.hour.isin(cfg.HORAS_SIN_SERVICIO) & ~np.isin(meses_hora, meses_sin_datos)
    ]

    # Rejilla: para cada estación, todas las horas válidas desde su primera hora con viajes
    primera_hora = demanda.groupby("origen_id")["hora_inicio"].min()
    ids, instantes = [], []
    for estacion, primera in primera_hora.items():
        posicion = horas_validas.searchsorted(primera)
        ids.append(np.full(len(horas_validas) - posicion, estacion, dtype="int32"))
        instantes.append(horas_validas[posicion:].to_numpy())
    rejilla = pd.DataFrame(
        {"origen_id": np.concatenate(ids), "hora_inicio": np.concatenate(instantes)}
    )

    tabla = rejilla.merge(
        demanda.astype({"origen_id": "int32"}), on=["origen_id", "hora_inicio"], how="left"
    )
    tabla["viajes"] = tabla["viajes"].fillna(0).astype("int16")
    tabla["particion"] = dataset.p3_asignar_particion(tabla["hora_inicio"])

    excluidos = int(
        len(demanda) - demanda.merge(rejilla, on=["origen_id", "hora_inicio"]).shape[0]
    )
    return tabla, excluidos


@app.callback()
def p3_ayuda_features():
    """Features de P3: tablas de estaciones y de demanda por estación-hora."""


@app.command()
def p3_features(mes_inicio: str = cfg.MES_INICIO, mes_fin: str = cfg.MES_FIN):
    """Deja estaciones_train.parquet y demanda_estacion_hora.parquet en data/processed/."""
    ruta = dataset.p3_ruta_viajes_modelado(mes_inicio, mes_fin)
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `make data` y `p3-particionar`")
    estaciones = dataset.p3_cargar_estaciones()

    # Cada tabla lee solo las columnas que necesita (menos memoria en equipos pequeños y en CI)
    limpios = pd.read_parquet(ruta, columns=["particion", "origen_id", "destino_id"])
    tabla = p3_tabla_estaciones(limpios, estaciones)
    del limpios
    cfg.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(cfg.RUTA_ESTACIONES_TRAIN)
    logger.info(
        f"Estaciones: {len(tabla)} | activas en train: {int(tabla['activa_en_train'].sum())}"
    )

    limpios = pd.read_parquet(ruta, columns=["inicio_del_viaje", "origen_id", "periodo_archivo"])
    demanda, excluidos = p3_demanda_estacion_hora(limpios, mes_inicio, mes_fin)
    demanda.to_parquet(cfg.RUTA_DEMANDA_ESTACION_HORA, index=False)
    logger.info(f"Demanda estación-hora: {len(demanda)} filas | fuera de la rejilla: {excluidos}")
    logger.success("Features de P3 listas en data/processed/")


if __name__ == "__main__":
    app()
