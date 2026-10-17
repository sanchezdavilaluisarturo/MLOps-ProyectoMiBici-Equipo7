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


# ---------------------------------------------------------------------------------------------
# P2 (clasificación de riesgos): reglas de limpieza propias de la pregunta.
# Entrada: data/processed/viajes_limpios_AAAA-MM_AAAA-MM.parquet (`dataset limpiar`).
# ---------------------------------------------------------------------------------------------


def p2_estaciones_con_capacidad(estaciones: pd.DataFrame) -> pd.DataFrame:
    """Regresa solo las estaciones con capacidad mayor a 0 (P2 analiza la capacidad)."""
    return estaciones[estaciones["dpcapacity"] > 0]


def p2_descartar_estaciones_inexistentes(
    viajes: pd.DataFrame, estaciones: pd.DataFrame
) -> pd.DataFrame:
    """Regresa los viajes cuyo origen y destino están en `estaciones`."""
    ids = estaciones["id"].unique()
    return viajes[viajes["origen_id"].isin(ids) & viajes["destino_id"].isin(ids)]


def p2_descartar_fuera_de_rango(
    viajes: pd.DataFrame, fecha_limite: str = cfg.P2_FECHA_LIMITE_FIN_VIAJE
) -> pd.DataFrame:
    """Regresa los viajes que terminan antes de `fecha_limite`."""
    return viajes[viajes["fin_del_viaje"] < fecha_limite]


def p2_descartar_horas_sin_servicio(
    viajes: pd.DataFrame, horas_sin_servicio=cfg.P2_HORAS_SIN_SERVICIO
) -> pd.DataFrame:
    """Regresa los viajes que inician en horas con servicio (se consideran anómalos los demás)."""
    return viajes[~viajes["inicio_del_viaje"].dt.hour.isin(horas_sin_servicio)]


def p2_limpiar_viajes(viajes: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Aplica en orden las reglas de limpieza de P2."""
    viajes = p2_descartar_estaciones_inexistentes(viajes, p2_estaciones_con_capacidad(estaciones))
    viajes = p2_descartar_fuera_de_rango(viajes)
    return p2_descartar_horas_sin_servicio(viajes)


def p2_cargar_viajes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regresa (viajes limpios para P2, estaciones con capacidad) listos para el modelado."""
    if not cfg.RUTA_VIAJES_LIMPIOS.exists():
        raise FileNotFoundError(f"Falta {cfg.RUTA_VIAJES_LIMPIOS.name}; ejecutar `make data`")
    # Solo las columnas que usa P2: menos memoria y los datos personales no pasan
    viajes = pd.read_parquet(
        cfg.RUTA_VIAJES_LIMPIOS,
        columns=["inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"],
    )
    estaciones = p2_estaciones_con_capacidad(dataset.p3_cargar_estaciones())

    limpios = p2_limpiar_viajes(viajes, estaciones)
    logger.info(f"Viajes P2: {len(limpios)} registros ({len(viajes) - len(limpios)} retirados)")
    logger.success(f"Limpieza de P2 lista: {len(limpios)} viajes y {len(estaciones)} estaciones")
    return limpios, estaciones


def p2_intervalo(ts: pd.Series, bin_min: int) -> pd.Series:
    """Regresa el bin de cada instante; los de horas sin servicio van al último bin de las 0:00."""
    b = ts.dt.floor(f"{bin_min}min")
    ultimo = b.dt.floor("D") + pd.Timedelta(hours=1) - pd.Timedelta(minutes=bin_min)
    return b.where(~b.dt.hour.isin(cfg.P2_HORAS_SIN_SERVICIO), ultimo)


def p2_obtener_dataframe_de_eventos_con_bins(viajes: pd.DataFrame, bin_min: int) -> pd.DataFrame:
    """Regresa salidas y entradas por estación y bin (solo los bins con actividad)."""
    salidas = (
        viajes.groupby(["origen_id", p2_intervalo(viajes["inicio_del_viaje"], bin_min)])
        .size()
        .rename("salidas")
    )
    entradas = (
        viajes.groupby(["destino_id", p2_intervalo(viajes["fin_del_viaje"], bin_min)])
        .size()
        .rename("entradas")
    )
    salidas.index.names = entradas.index.names = ["estacion", "bin"]
    return pd.concat([salidas, entradas], axis=1).fillna(0).astype("int32")


def p2_generar_arreglo_de_bins(eventos: pd.DataFrame, bin_min: int) -> pd.DatetimeIndex:
    """Regresa los bins del primer día al último bin de `eventos`, sin horas sin servicio."""
    bins = pd.date_range(
        eventos.index.get_level_values("bin").min().floor("D"),
        eventos.index.get_level_values("bin").max(),
        freq=f"{bin_min}min",
    )
    return bins[~bins.hour.isin(cfg.P2_HORAS_SIN_SERVICIO)]


def p2_obtener_dataframe_de_cambios(viajes: pd.DataFrame, bin_min: int) -> pd.DataFrame:
    """Regresa salidas y entradas para cada estación y cada bin (los ceros también son filas)."""
    eventos = p2_obtener_dataframe_de_eventos_con_bins(viajes, bin_min)
    bins = p2_generar_arreglo_de_bins(eventos, bin_min)
    rejilla = pd.MultiIndex.from_product(
        [eventos.index.get_level_values("estacion").unique(), bins], names=["estacion", "bin"]
    )
    return eventos.reindex(rejilla, fill_value=0).reset_index()


def p2_agregar_suma_acumulada_a_cambios(cambios: pd.DataFrame) -> pd.DataFrame:
    """Agrega neto, dia (de servicio, 05:00 a 00:59) y cumsum_dia; quita estación-días sin viajes."""
    cambios["neto"] = cambios["entradas"] - cambios["salidas"]
    cambios["dia"] = (cambios["bin"] - pd.Timedelta(hours=5)).dt.floor("D")

    actividad = (
        cambios.groupby(["estacion", "dia"])[["salidas", "entradas"]].transform("sum").sum(axis=1)
    )
    cambios = cambios[actividad > 0]
    cambios = cambios.sort_values(["estacion", "bin"]).reset_index(drop=True)
    cambios["cumsum_dia"] = cambios.groupby(["estacion", "dia"], sort=False)["neto"].cumsum()
    return cambios


def p2_agregar_datos_de_estacion_a_cambios(
    cambios: pd.DataFrame, estaciones: pd.DataFrame
) -> pd.DataFrame:
    """Agrega latitud, longitud y capacidad de cada estación."""
    cambios = cambios.merge(
        estaciones[["id", "latitude", "longitude", "dpcapacity"]],
        left_on="estacion",
        right_on="id",
        how="left",
    )
    return cambios.drop("id", axis=1)


def p2_obtener_mayores_cambios_en_proximos_h_bins(
    cambios: pd.DataFrame, h: int
) -> tuple[pd.Series, pd.Series]:
    """Regresa (vacia, llena): peor salida y peor entrada netas en t+1..t+h, sobre la capacidad.

    Son NaN cuando la ventana futura se sale del día de servicio.
    """
    cumsums = cambios.groupby(["estacion", "dia"], sort=False)["cumsum_dia"]
    cambios_futuros = pd.concat(
        [cumsums.shift(-k) - cambios["cumsum_dia"] for k in range(1, h + 1)], axis=1
    )
    vacia = (-cambios_futuros).max(axis=1, skipna=False).clip(lower=0) / cambios["dpcapacity"]
    llena = cambios_futuros.max(axis=1, skipna=False).clip(lower=0) / cambios["dpcapacity"]
    return vacia, llena


def p2_agregar_clase_de_riesgo(cambios: pd.DataFrame, alfa: float = cfg.P2_ALFA) -> pd.DataFrame:
    """Agrega clase: 0 = sin riesgo, 1 = sin bicis, 2 = sin anclajes.

    Si una ventana tiene ambos eventos gana el de mayor estrés relativo. Se quitan las filas cuya
    ventana futura se sale del día.
    """
    datos = cambios.dropna(subset=["vacia", "llena"]).copy()
    vacia_positiva, llena_positiva = datos["vacia"] >= alfa, datos["llena"] >= alfa
    datos["clase"] = 0
    datos.loc[vacia_positiva, "clase"] = 1
    datos.loc[llena_positiva, "clase"] = 2
    ambas = vacia_positiva & llena_positiva
    datos.loc[ambas, "clase"] = np.where(
        datos.loc[ambas, "vacia"] >= datos.loc[ambas, "llena"], 1, 2
    )
    logger.info(
        f"Ventanas con ambos eventos: {int(ambas.sum())} de "
        f"{int((vacia_positiva | llena_positiva).sum())} positivas"
    )
    logger.info(
        "Distribución de clases: "
        f"{datos['clase'].value_counts(normalize=True).sort_index().round(4).to_dict()}"
    )
    return datos


def p2_ventana(cambios: pd.DataFrame, col: str, k: int) -> pd.Series:
    """Regresa la suma de `col` en los últimos k bins del mismo estación-día (incluye el actual)."""
    c = cambios.groupby(["estacion", "dia"], sort=False)[col].cumsum()
    return c - c.groupby([cambios["estacion"], cambios["dia"]], sort=False).shift(k, fill_value=0)


def p2_agregar_caracteristicas_retrospectivas(cambios: pd.DataFrame) -> pd.DataFrame:
    """Agrega neto relativo a la capacidad (1, 2, 4 y 8 bins) y salidas y entradas (2 y 4 bins)."""
    for k in (1, 2, 4, 8):
        cambios[f"neto_{k}_relativo"] = p2_ventana(cambios, "neto", k) / cambios["dpcapacity"]
    for k in (2, 4):
        cambios[f"salidas_{k}"] = p2_ventana(cambios, "salidas", k)
        cambios[f"entradas_{k}"] = p2_ventana(cambios, "entradas", k)
    return cambios


def p2_agregar_caracteristicas_de_hora(cambios: pd.DataFrame, bin_min: int) -> pd.DataFrame:
    """Agrega hora_idx y su codificación cíclica (hora_sin, hora_cos)."""
    cambios["hora_idx"] = (cambios["bin"].dt.hour * 60 + cambios["bin"].dt.minute) // bin_min
    bins_por_dia = 60 * 24 / bin_min
    cambios["hora_sin"] = np.sin(2 * np.pi * cambios["hora_idx"] / bins_por_dia)
    cambios["hora_cos"] = np.cos(2 * np.pi * cambios["hora_idx"] / bins_por_dia)
    return cambios


def p2_construir_caracteristicas(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    bin_min: int = cfg.P2_BIN_MIN,
    h_bins: int = cfg.P2_H_BINS,
    alfa: float = cfg.P2_ALFA,
) -> pd.DataFrame:
    """Regresa la tabla estación-bin con las columnas de P2_FEATURES y la clase a predecir.
    Limitación importante: por definición del planteamiento, los últimos h_bins registros por día
    no tendrán clase de riesgo a clasificar. Estos registros se filtran.
    """
    cambios = p2_obtener_dataframe_de_cambios(viajes, bin_min)
    # Cada viaje es exactamente una salida y una entrada
    if cambios["salidas"].sum() != len(viajes) or cambios["entradas"].sum() != len(viajes):
        raise ValueError("Las salidas o entradas por bin no coinciden con el número de viajes")

    cambios = p2_agregar_suma_acumulada_a_cambios(cambios)
    cambios = p2_agregar_datos_de_estacion_a_cambios(cambios, estaciones)
    cambios["vacia"], cambios["llena"] = p2_obtener_mayores_cambios_en_proximos_h_bins(
        cambios, h_bins
    )
    cambios = p2_agregar_clase_de_riesgo(cambios, alfa)

    cambios = p2_agregar_caracteristicas_retrospectivas(cambios)
    cambios["cumsum_dia_relativo"] = cambios["cumsum_dia"] / cambios["dpcapacity"]
    cambios = p2_agregar_caracteristicas_de_hora(cambios, bin_min)
    cambios["es_fin_semana"] = (cambios["bin"].dt.dayofweek >= 5).astype(int)
    
    # Optimización de memoria
    columnas_salida = cfg.P2_FEATURES + ["clase", "estacion", "dia", "hora_idx"]
    cambios = cambios[columnas_salida]
    for c in cambios.select_dtypes("integer").columns:
        cambios[c] = pd.to_numeric(cambios[c], downcast="integer")        # sin pérdida
    
    for c in cambios.select_dtypes("float64").columns:       # decimales a float32
        cambios[c] = cambios[c].astype("float32")
        
    logger.success(
        f"Features de P2 listas: {len(cambios)} filas estación-bin, "
        f"{cambios['estacion'].nunique()} estaciones, {len(cfg.P2_FEATURES)} features"
    )
    return cambios


if __name__ == "__main__":
    app()
