"""Gráficas de P3 (clustering de estaciones).

Cada función recibe datos ya calculados y regresa la figura (matplotlib); guardarla es opcional
con `p3_guardar_figura`. Usa la paleta Okabe-Ito de config.py, apta para daltonismo.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tc5061_proyecto_equipo07.config import AZUL, GRIS, NARANJA, VERDE

# Colores por grupo (Okabe-Ito); sirven hasta 10 grupos
P3_COLORES = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#999999",
    "#000000",
    "#882255",
]
P3_DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def p3_guardar_figura(figura, ruta: Path, dpi: int = 150) -> Path:
    """Guarda la figura en `ruta` (crea la carpeta), la cierra y regresa la ruta."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=dpi)
    plt.close(figura)
    return ruta


# ---------------------------------------------------------------------------------------------
# EDA (0.05)
# ---------------------------------------------------------------------------------------------


def p3_grafica_duracion(duracion: pd.Series):
    """Distribución de la duración: hasta 60 min y escala logarítmica de todos los positivos."""
    fig, ejes = plt.subplots(1, 2, figsize=(14, 4))
    duracion[(duracion > 0) & (duracion <= 60)].plot.hist(bins=60, ax=ejes[0], color=AZUL)
    ejes[0].set_title("Duración de los viajes (hasta 60 min)")
    ejes[0].set_xlabel("Minutos")
    ejes[0].set_ylabel("Viajes")

    np.log10(duracion[duracion > 0]).plot.hist(bins=60, ax=ejes[1], color=NARANJA)
    ejes[1].set_title("Duración de los viajes (escala log10)")
    ejes[1].set_xlabel("log10(minutos)")
    ejes[1].set_ylabel("Viajes")
    fig.tight_layout()
    return fig


def p3_grafica_serie_mensual(mensual: pd.Series):
    """Viajes por mes sobre el calendario completo (los meses sin dato quedan como hueco)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(mensual.index.to_timestamp(), mensual.values, marker="o", markersize=3, color=AZUL)
    ax.set_title("Viajes por mes")
    ax.set_ylabel("Viajes")
    ax.grid(alpha=0.3)
    return fig


def p3_grafica_heatmap_hora_dia(proporcion: pd.DataFrame):
    """Mapa de calor de la proporción de viajes por día de la semana (filas) y hora (columnas)."""
    fig, ax = plt.subplots(figsize=(14, 3.5))
    sns.heatmap(
        proporcion, cmap="Blues", ax=ax, yticklabels=P3_DIAS, cbar_kws={"label": "% de los viajes"}
    )
    ax.set_title("Viajes por día de la semana y hora de inicio")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("")
    return fig


def p3_grafica_perfil_horario(laboral: pd.Series, fin_de_semana: pd.Series):
    """Perfil horario de días laborales frente a fin de semana (% dentro de cada grupo)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        laboral.index,
        laboral / laboral.sum() * 100,
        marker="o",
        color=AZUL,
        label="Lunes a viernes",
    )
    ax.plot(
        fin_de_semana.index,
        fin_de_semana / fin_de_semana.sum() * 100,
        marker="o",
        color=NARANJA,
        label="Sábado y domingo",
    )
    ax.set_title("Perfil horario de los viajes")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("% de los viajes del grupo")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def p3_grafica_concentracion_balance(orden: pd.Series, estaciones: pd.DataFrame):
    """Concentración de la demanda por estación y distribución del balance neto (% de salidas).

    `orden` son las salidas por estación de mayor a menor; `estaciones` trae `neto_%`.
    """
    fig, ejes = plt.subplots(1, 2, figsize=(14, 4))
    acumulado = orden.cumsum() / orden.sum() * 100
    ejes[0].plot(
        np.arange(1, len(acumulado) + 1) / len(acumulado) * 100, acumulado.to_numpy(), color=AZUL
    )
    ejes[0].set_title("Concentración de los viajes por estación")
    ejes[0].set_xlabel("% de estaciones (de mayor a menor demanda)")
    ejes[0].set_ylabel("% acumulado de los viajes")
    ejes[0].grid(alpha=0.3)

    estaciones["neto_%"].plot.hist(bins=40, ax=ejes[1], color=VERDE)
    ejes[1].set_title("Balance neto como % de las salidas")
    ejes[1].set_xlabel("(llegadas − salidas) / salidas, %")
    ejes[1].set_ylabel("Estaciones")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# Limpieza y partición (0.06)
# ---------------------------------------------------------------------------------------------


def p3_grafica_particion(mensual_particion: pd.DataFrame):
    """Viajes por mes apilados por conjunto (train, brecha, valid, test).

    El índice se pasa a texto AAAA-MM: un índice de fechas con frecuencia mensual rompe el
    gráfico de barras de pandas cuando la serie no tiene huecos.
    """
    tabla = mensual_particion.copy()
    tabla.index = tabla.index.astype(str)
    ax = tabla.plot.bar(
        stacked=True, figsize=(14, 4), width=0.9, color=[AZUL, GRIS, NARANJA, VERDE]
    )
    ax.set_xticks(range(0, len(tabla), 6))
    ax.set_xticklabels(tabla.index[::6], rotation=45)
    ax.set_title("Viajes por mes y conjunto")
    ax.set_xlabel("")
    ax.set_ylabel("Viajes")
    return ax.get_figure()


# ---------------------------------------------------------------------------------------------
# Modelo base y modelo alternativo (0.07 y 0.08)
# ---------------------------------------------------------------------------------------------


def p3_grafica_k(resultados_k: pd.DataFrame, k_elegido: int):
    """Silueta e inercia por número de grupos `k`."""
    fig, ejes = plt.subplots(1, 2, figsize=(12, 3.5))
    resultados_k["silueta"].plot(marker="o", ax=ejes[0], color=AZUL)
    ejes[0].axvline(k_elegido, color=NARANJA, linestyle="--", label=f"k elegido = {k_elegido}")
    ejes[0].set_title("Coeficiente de silueta por k")
    ejes[0].set_xlabel("k")
    ejes[0].legend()
    resultados_k["inercia"].plot(marker="o", ax=ejes[1], color=VERDE)
    ejes[1].set_title("Inercia por k")
    ejes[1].set_xlabel("k")
    fig.tight_layout()
    return fig


def p3_grafica_silueta_por_k(resultados_k: pd.DataFrame, k_elegido: int):
    """Silueta por `k` en un solo panel (versión que se guarda como artefacto)."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    resultados_k["silueta"].plot(marker="o", ax=ax, color=AZUL)
    ax.axvline(k_elegido, color=NARANJA, linestyle="--")
    ax.set_title("Silueta por k")
    ax.set_xlabel("k")
    fig.tight_layout()
    return fig


def p3_grafica_perfil_grupos(
    caracteristicas: pd.DataFrame,
    columnas_lab: list[str],
    columnas_fin: list[str],
    k: int,
    horas: list[int],
    columna_grupo: str = "grupo",
):
    """Perfil horario medio de cada grupo, en días laborales y en fin de semana."""
    fig, ejes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for grupo in range(k):
        miembros = caracteristicas[columna_grupo] == grupo
        ejes[0].plot(
            horas,
            caracteristicas.loc[miembros, columnas_lab].mean() * 100,
            marker="o",
            color=P3_COLORES[grupo],
            label=f"Grupo {grupo} (n={int(miembros.sum())})",
        )
        ejes[1].plot(
            horas,
            caracteristicas.loc[miembros, columnas_fin].mean() * 100,
            marker="o",
            color=P3_COLORES[grupo],
        )
    ejes[0].set_title("Perfil horario, lunes a viernes")
    ejes[1].set_title("Perfil horario, sábado y domingo")
    for eje in ejes:
        eje.set_xlabel("Hora del día")
        eje.set_xticks(range(0, 24, 2))
        eje.grid(alpha=0.3)
    ejes[0].set_ylabel("% de las salidas de la estación")
    ejes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def p3_grafica_mapa_grupos(
    mapa: pd.DataFrame,
    k: int,
    titulo: str = "Estaciones por grupo (baseline K-Means)",
    alpha: float = 0.85,
    columna_grupo: str = "grupo",
):
    """Ubicación de las estaciones coloreadas por grupo (`mapa` trae latitude y longitude)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for grupo in range(k):
        miembros = mapa[columna_grupo] == grupo
        ax.scatter(
            mapa.loc[miembros, "longitude"],
            mapa.loc[miembros, "latitude"],
            s=18,
            color=P3_COLORES[grupo],
            label=f"Grupo {grupo}",
            alpha=alpha,
        )
    ax.set_title(titulo)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def p3_grafica_perfiles_y_mapa(
    asignacion: pd.DataFrame,
    k: int,
    columnas_lab: list[str],
    columnas_fin: list[str],
    horas: list[int],
    columna_grupo: str = "grupo_alt",
):
    """Perfil horario (laboral y fin de semana) y ubicación de los grupos del modelo alternativo."""
    fig, ejes = plt.subplots(1, 3, figsize=(17, 4.5))
    for grupo in range(k):
        miembros = asignacion[columna_grupo] == grupo
        ejes[0].plot(
            horas,
            asignacion.loc[miembros, columnas_lab].mean() * 100,
            marker="o",
            color=P3_COLORES[grupo],
            label=f"Grupo {grupo} (n={int(miembros.sum())})",
        )
        ejes[1].plot(
            horas,
            asignacion.loc[miembros, columnas_fin].mean() * 100,
            marker="o",
            color=P3_COLORES[grupo],
        )
        ejes[2].scatter(
            asignacion.loc[miembros, "longitude"],
            asignacion.loc[miembros, "latitude"],
            s=18,
            color=P3_COLORES[grupo],
            alpha=0.85,
        )
    ejes[0].set_title("Perfil horario, lunes a viernes")
    ejes[1].set_title("Perfil horario, sábado y domingo")
    ejes[2].set_title("Ubicación de las estaciones")
    for eje in ejes[:2]:
        eje.set_xlabel("Hora del día")
        eje.set_xticks(range(0, 24, 2))
        eje.grid(alpha=0.3)
    ejes[0].set_ylabel("% de las salidas de la estación")
    ejes[0].legend(fontsize=8)
    ejes[2].set_xlabel("Longitud")
    ejes[2].set_ylabel("Latitud")
    ejes[2].grid(alpha=0.3)
    fig.tight_layout()
    return fig
