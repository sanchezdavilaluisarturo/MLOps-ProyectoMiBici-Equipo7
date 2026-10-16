"""Features de P1 (demanda por estación-hora) y de P3 (clustering de estaciones).

P1: lee data/processed/viajes_limpios_AAAA-MM_AAAA-MM.parquet (`make data`) y deja
data/processed/p1_demanda_estacion_hora.parquet con el target y sus variables.
P3: lee data/processed/viajes_modelado_AAAA-MM_AAAA-MM.parquet (`dataset p3-particionar`) y deja
estaciones_train.parquet y demanda_estacion_hora.parquet en data/processed/.

Uso: python -m tc5061_proyecto_equipo07.features p1-features | p3-features
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


# Horas de servicio por día: la malla de P1 tiene 20 columnas por día, así que un día = 20 posiciones
P1_HORAS_POR_DIA = 24 - len(cfg.HORAS_SIN_SERVICIO)


def p1_horas_de_servicio(mes_inicio: str, mes_fin: str) -> pd.DatetimeIndex:
    """Regresa todas las horas con servicio del rango (sin las de 1:00 a 4:59, en que no opera).

    Incluye las horas de meses sin datos para que cada día tenga el mismo número de horas y un
    rezago de N días sea siempre un desplazamiento de N * 20 posiciones.
    """
    ultima_hora = pd.Period(mes_fin, "M").end_time.floor("h")
    horas = pd.date_range(f"{mes_inicio}-01", ultima_hora, freq="h")
    return horas[~horas.hour.isin(cfg.HORAS_SIN_SERVICIO)]


def p1_matriz_demanda(
    limpios: pd.DataFrame, horas: pd.DatetimeIndex, meses_sin_datos: list[str]
) -> tuple[np.ndarray, pd.Index, pd.Series]:
    """Cuenta los viajes iniciados por estación y hora en una matriz estaciones x horas.

    Regresa (matriz, ids de estación, primera hora con viajes de cada estación). Una hora sin
    viajes dentro del periodo de operación de la estación vale 0; antes de su primer viaje,
    después del último y en meses sin datos vale NaN, porque ahí no hay demanda que medir.
    """
    hora_inicio = limpios["inicio_del_viaje"].dt.floor("h")
    en_servicio = ~hora_inicio.dt.hour.isin(cfg.HORAS_SIN_SERVICIO)
    conteo = (
        pd.DataFrame(
            {
                "origen_id": limpios.loc[en_servicio, "origen_id"].to_numpy(),
                "hora": hora_inicio[en_servicio].to_numpy(),
            }
        )
        .groupby(["origen_id", "hora"])
        .size()
    )
    ids = conteo.index.get_level_values("origen_id").unique().sort_values()
    columnas = horas.get_indexer(conteo.index.get_level_values("hora"))
    if (columnas < 0).any():
        raise ValueError("Hay viajes fuera del rango de horas; revisar mes_inicio y mes_fin")

    matriz = np.zeros((len(ids), len(horas)), dtype="float32")
    matriz[ids.get_indexer(conteo.index.get_level_values("origen_id")), columnas] = conteo
    operacion = conteo.index.to_frame(index=False).groupby("origen_id")["hora"].agg(["min", "max"])
    operacion = operacion.reindex(ids)
    posicion = np.arange(len(horas))
    inicio = horas.searchsorted(operacion["min"])
    fin = horas.searchsorted(operacion["max"], side="right")
    matriz[(posicion < inicio[:, None]) | (posicion >= fin[:, None])] = np.nan
    matriz[:, np.isin(horas.strftime("%Y-%m"), meses_sin_datos)] = np.nan
    return matriz, ids, operacion["min"]


def p1_rezago(matriz: np.ndarray, dias: int) -> np.ndarray:
    """Regresa la demanda de la misma hora `dias` días antes (NaN si ese momento no existe).

    Con `dias` >= 1 la variable solo usa información disponible al cierre del día anterior, que es
    cuando se hace la predicción del día siguiente: no hay fuga de información del futuro.
    """
    if dias < 1:
        raise ValueError("Los rezagos de P1 deben ser de al menos un día (horizonte de 24 h)")
    posiciones = dias * P1_HORAS_POR_DIA
    rezagada = np.full_like(matriz, np.nan)
    rezagada[:, posiciones:] = matriz[:, :-posiciones]
    return rezagada


def p1_promedio_misma_hora(matriz: np.ndarray, semanas: int) -> np.ndarray:
    """Promedia la demanda de la misma hora y día de la semana en las `semanas` anteriores.

    Una sola estación-hora es ruidosa (autocorrelación semanal mediana de 0.40 en el EDA); el
    promedio de varias semanas estima mejor el nivel esperado. Ignora las semanas sin dato.
    """
    suma = np.zeros_like(matriz)
    validas = np.zeros_like(matriz)
    for semana in range(1, semanas + 1):
        rezagada = p1_rezago(matriz, 7 * semana)
        hay_dato = ~np.isnan(rezagada)
        suma[hay_dato] += rezagada[hay_dato]
        validas += hay_dato
    with np.errstate(invalid="ignore", divide="ignore"):
        return suma / validas  # 0 / 0 = NaN cuando ninguna semana tiene dato


def p1_dias_festivos(anios: list[int]) -> pd.DatetimeIndex:
    """Regresa los días de descanso obligatorio en México (Ley Federal del Trabajo, art. 74).

    En el EDA los días con menos viajes son festivos, por lo que la demanda cambia en esos días.
    """

    def lunes(anio: int, mes: int, numero: int) -> pd.Timestamp:
        dias = pd.date_range(f"{anio}-{mes:02d}-01", periods=7)
        return dias[dias.dayofweek == 0][0] + pd.Timedelta(weeks=numero - 1)

    festivos = []
    for anio in anios:
        festivos += [
            pd.Timestamp(anio, 1, 1),
            lunes(anio, 2, 1),  # Día de la Constitución
            lunes(anio, 3, 3),  # Natalicio de Benito Juárez
            pd.Timestamp(anio, 5, 1),
            pd.Timestamp(anio, 9, 16),
            lunes(anio, 11, 3),  # Día de la Revolución
            pd.Timestamp(anio, 12, 25),
        ]
        if (anio - 2024) % 6 == 0:
            festivos.append(pd.Timestamp(anio, 10, 1))  # Transmisión del Poder Ejecutivo
    return pd.DatetimeIndex(sorted(festivos))


def p1_variables_calendario(hora_inicio: pd.Series) -> pd.DataFrame:
    """Regresa hora, día de la semana, fin de semana, mes y día festivo de cada hora a predecir.

    Son las variables con más señal en el EDA: dos picos entre semana (8:00 y 18:00), un solo pico
    el fin de semana y menos viajes en festivos. Se conocen de antemano, así que no hay fuga.
    """
    anios = range(hora_inicio.dt.year.min(), hora_inicio.dt.year.max() + 1)
    festivos = p1_dias_festivos(list(anios))
    dia_semana = hora_inicio.dt.dayofweek
    return pd.DataFrame(
        {
            "hora": hora_inicio.dt.hour.astype("int8"),
            "dia_semana": dia_semana.astype("int8"),
            "es_fin_de_semana": (dia_semana >= 5).astype("int8"),
            "mes": hora_inicio.dt.month.astype("int8"),
            "es_festivo": hora_inicio.dt.normalize().isin(festivos).astype("int8"),
        },
        index=hora_inicio.index,
    )


def p1_variables_estacion(estaciones: pd.DataFrame) -> pd.DataFrame:
    """Regresa capacidad, ubicación y zona (municipio del código del nombre) por id de estación.

    El nivel de demanda cambia mucho entre estaciones (de 1.51 a 229.91 viajes por día en el
    EDA) y se concentra en el centro de la red, por lo que la ubicación ayuda a distinguirlas.
    """
    zona = estaciones["name"].str.extract(r"([A-Z]{3})-\d{3}", expand=False).fillna("SIN_ZONA")
    return pd.DataFrame(
        {
            "capacidad": estaciones["dpcapacity"].astype("int16").to_numpy(),
            "latitud": estaciones["latitude"].astype("float32").to_numpy(),
            "longitud": estaciones["longitude"].astype("float32").to_numpy(),
            "zona": pd.Categorical(zona),
        },
        index=pd.Index(estaciones["id"].astype("int32"), name="origen_id"),
    )


def p1_tabla_modelado(
    limpios: pd.DataFrame,
    estaciones: pd.DataFrame,
    mes_inicio: str = cfg.MES_INICIO,
    mes_fin: str = cfg.MES_FIN,
) -> pd.DataFrame:
    """Regresa una fila por estación y hora de servicio con el target `viajes` y sus variables.

    Requiere `inicio_del_viaje`, `origen_id` y `periodo_archivo` de los viajes limpios. Cada fila
    trae la partición temporal del proyecto; las horas en `brecha` no se usan para entrenar ni
    evaluar.
    """
    meses_sin_datos = dataset.p3_meses_sin_datos(
        limpios["periodo_archivo"].astype(str).unique(), mes_inicio, mes_fin
    )
    horas = p1_horas_de_servicio(mes_inicio, mes_fin)
    matriz, ids, apertura = p1_matriz_demanda(limpios, horas, meses_sin_datos)

    variables = {f"viajes_rezago_{dias}d": p1_rezago(matriz, dias) for dias in cfg.P1_REZAGOS_DIAS}
    promedio = f"viajes_promedio_{cfg.P1_SEMANAS_PROMEDIO}sem"
    variables[promedio] = p1_promedio_misma_hora(matriz, cfg.P1_SEMANAS_PROMEDIO)

    # Solo las estación-horas dentro del periodo de operación (las celdas que no son NaN)
    filas, columnas = np.nonzero(~np.isnan(matriz))
    tabla = pd.DataFrame(
        {
            "origen_id": ids.to_numpy()[filas].astype("int32"),
            "hora_inicio": horas[columnas],
            "viajes": matriz[filas, columnas].astype("int16"),
        }
    )
    for nombre, valores in variables.items():
        tabla[nombre] = valores[filas, columnas]
    del matriz, variables

    tabla = pd.concat([tabla, p1_variables_calendario(tabla["hora_inicio"])], axis=1)
    tabla = tabla.join(p1_variables_estacion(estaciones), on="origen_id")
    dias_operando = tabla["hora_inicio"] - tabla["origen_id"].map(apertura).dt.normalize()
    tabla["dias_desde_apertura"] = dias_operando.dt.days.astype("int16")
    tabla["particion"] = dataset.p3_asignar_particion(tabla["hora_inicio"])
    return tabla


@app.callback()
def p3_ayuda_features():
    """Features de P1 (demanda por estación-hora) y de P3 (tablas de estaciones y de demanda)."""


@app.command()
def p1_features(mes_inicio: str = cfg.MES_INICIO, mes_fin: str = cfg.MES_FIN):
    """Deja p1_demanda_estacion_hora.parquet en data/processed/ (target y variables de P1)."""
    ruta = cfg.PROCESSED_DATA_DIR / f"viajes_limpios_{mes_inicio}_{mes_fin}.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `make data`")
    limpios = pd.read_parquet(ruta, columns=["inicio_del_viaje", "origen_id", "periodo_archivo"])
    tabla = p1_tabla_modelado(limpios, dataset.p3_cargar_estaciones(), mes_inicio, mes_fin)

    # Control: la tabla debe contener todos los viajes limpios que inician en horas de servicio
    en_servicio = ~limpios["inicio_del_viaje"].dt.hour.isin(cfg.HORAS_SIN_SERVICIO)
    esperados, en_tabla = int(en_servicio.sum()), int(tabla["viajes"].sum())
    if en_tabla != esperados:
        raise ValueError(f"La tabla suma {en_tabla} viajes y se esperaban {esperados}")

    cfg.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(cfg.RUTA_P1_DEMANDA, index=False)
    por_conjunto = tabla.groupby("particion", observed=True)["viajes"].agg(["size", "sum"])
    logger.info(f"Viajes en horas sin servicio (fuera de P1): {len(limpios) - esperados}")
    logger.info(f"Estación-horas y viajes por conjunto:\n{por_conjunto.to_string()}")
    logger.success(f"{cfg.RUTA_P1_DEMANDA.name}: {len(tabla)} filas y {en_tabla} viajes")


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
