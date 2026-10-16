# Definición de los datos — Proyecto MiBici (Equipo 7)

> **Estado:** borrador para acordar en el equipo. Las cifras se midieron sobre los archivos descargados el 13-oct-2026. Lo marcado como *propuesta* es una decisión pendiente.

## 0. Rango de fechas propuesto

**Enero de 2020 a septiembre de 2026** (81 meses; 79 con datos).

| Razón | Detalle |
|---|---|
| **Años completos** | Cubre seis años completos (2020 a 2025) y 2026 hasta septiembre, de modo que la estacionalidad anual aparece varias veces en el entrenamiento |
| **Tres regímenes** | Prepandemia (ene-feb 2020), choque de la pandemia (mar 2020 en adelante, con el mínimo en abril de 2020: 129,276 viajes) y recuperación hasta 490,056 viajes en septiembre de 2026. Un modelo operativo debe funcionar en el régimen actual, pero necesita ver los cambios |
| **Volumen manejable** | 26.4 millones de viajes y 2.1 GB en CSV (0.6 GB en Parquet) caben en pandas sin infraestructura adicional |
| **Esquema consistente** | Desde 2020 el esquema tiene las mismas 8 columnas; la normalización de codificación y fechas está resuelta para estos 79 archivos |
| **Más reciente = más relevante** | La nomenclatura vigente es la de 2026-09; cuanto más lejos del presente, menos representa a la red actual |

**Riesgo conocido:** 2020 está distorsionado por la pandemia (2.87 M de viajes, 10.9 % del total). Alternativa a evaluar: iniciar en 2021-01. *Propuesta:* conservar 2020 y agregar una variable indicadora de periodo de pandemia; revisar el impacto en la validación.

## 1. Fuente

- **Portal:** [Datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/).
- **Viajes:** un CSV por mes, `https://mibici.net/site/assets/files/{id}/datos_abiertos_AAAA_MM.csv`. El `{id}` cambia por archivo, así que no se puede construir la URL; por eso se leen de la página.
- **Estaciones:** [`nomenclatura_2026_09.csv`](https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv). Es la única versión publicada.
- **Manifiesto:** [`fuentes-datos.csv`](fuentes-datos.csv) lista las 80 URLs, los registros por archivo y el SHA-256 de cada uno, para verificar que dos integrantes trabajan con los mismos datos.

## 2. Tamaño real obtenido

| Concepto | Valor |
|---|---|
| Meses en el rango | 81 |
| Meses con archivo | **79** (faltan 2023-04 y 2023-05, retirados del portal) |
| Viajes | **26,389,162** |
| Estaciones en la nomenclatura | **484** |
| Estaciones con viajes | **475** |
| Usuarios distintos | 81,115 |
| Tamaño | 2,097.6 MB en CSV y 602 MB en Parquet |

**Calidad que afecta el tamaño efectivo:** 758 viajes duplicados, 3 estaciones sin nomenclatura (3,961 viajes) y 53,581 viajes con duración cero. Detalle en la sección *Datos* del README.

## 3. Variables originales y variables objetivo

### Variables que se usarán

| Variable | Origen | Uso |
|---|---|---|
| `inicio_del_viaje` | Viajes | Hora, día, mes y rezagos (P1, P2, P3) |
| `fin_del_viaje` | Viajes | Hora de llegada (P2) |
| `origen_id` | Viajes | Estación de salida (P1, P2, P3) |
| `destino_id` | Viajes | Estación de llegada (P2, P3) |
| `viaje_id` | Viajes | Solo para eliminar duplicados; no es feature |
| `id`, `latitude`, `longitude`, `dpcapacity` | Nomenclatura | Capacidad (P2) y ubicación (P3) |

**No se usarán** `usuario_id`, `genero` ni `anio_de_nacimiento`: ninguna de las tres preguntas es sobre personas y son datos personales (ver sección 4). Además tienen problemas de calidad (años de nacimiento imposibles, nulos).

### Construcción de cada variable objetivo

| Pregunta | Objetivo | Construcción |
|---|---|---|
| **P1** (regresión) | `viajes_inicio(estación, hora)` | Contar viajes por `origen_id` y hora de `inicio_del_viaje`; completar con 0 las horas sin viajes para que la serie sea continua |
| **P2** (clasificación) | `riesgo(estación, ventana)` | `neto = llegadas − salidas` acumulado desde el inicio del día (llegadas por `destino_id` y `fin_del_viaje`; salidas por `origen_id` y `inicio_del_viaje`). Riesgo de *sin bicis* si `neto ≤ −u·dpcapacity` y de *sin anclajes* si `neto ≥ u·dpcapacity`; `u` por definir |
| **P3** (clustering) | Sin objetivo | Vector por estación: perfil horario (laboral y fin de semana), volumen, balance neto promedio y ubicación, calculado **solo con el periodo de entrenamiento** |

**Limitación de P2:** no hay inventario de bicicletas, así que el objetivo es un proxy. El acumulado parte de un nivel desconocido y no ve el rebalanceo que hizo la operación. Se excluyen las 4 estaciones con `dpcapacity = 0`.

## 4. Condiciones de uso y datos personales

### Condiciones publicadas (consultadas el 15-oct-2026)

| Documento | Qué dice sobre los datos abiertos |
|---|---|
| Página de datos abiertos | **No publica licencia ni condiciones de uso específicas.** Solo el pie "Todos los derechos reservados Mi Bici © 2026" |
| [Términos y condiciones](https://mibici.net/es/termino-y-condiciones/) | **Ninguna cláusula sobre datos abiertos**; tratan el uso del servicio (por ejemplo, la cuenta se suspende tras 8 horas de uso continuo) |
| [Aviso de privacidad](https://mibici.net/es/aviso-de-privacidad/) | **No menciona los datos abiertos** ni la anonimización. Describe datos del personal de la Agencia, de visitantes y de trámites, no de los usuarios del servicio |

**Consecuencia:** no hay una licencia que autorice ni prohíba la reutilización. Se asume el uso **académico, no comercial y con cita de la fuente**, y se evita redistribuir los datos. *Propuesta:* mencionarlo explícitamente en el entregable y, si se publica el repositorio, no subir los datos (ya está en `.gitignore`).

### Tratamiento de datos personales

`usuario_id`, `genero` y `anio_de_nacimiento` son datos personales o cuasi-identificadores: combinados con fecha, hora y estaciones podrían permitir reidentificar a una persona.

| Medida | Cómo |
|---|---|
| **Minimización** | No se cargan al conjunto de modelado; se descartan al pasar de `interim` a `processed` |
| **Agregación** | Todos los objetivos y features son por estación, nunca por persona |
| **Sin valores individuales** | Notebooks y reportes muestran solo agregados; no se imprimen filas con `usuario_id` |
| **Sin redistribución** | Los datos no se versionan en Git ni se comparten fuera del equipo |
| **Sin reidentificación** | No se cruzan los datos con otras fuentes |

## 5. Estrategia de partición

### Propuesta: partición temporal en tres bloques de 12 meses

| Conjunto | Periodo | Meses | Viajes | % |
|---|---|---:|---:|---:|
| **Entrenamiento** | 2020-01 a 2024-09 | 57 (55 con datos) | 17,345,237 | 65.7 % |
| **Validación** | 2024-10 a 2025-09 | 12 | 4,546,048 | 17.2 % |
| **Prueba** | 2025-10 a 2026-09 | 12 | 4,497,877 | 17.0 % |

**Por qué bloques de 12 meses:** cada conjunto de validación y prueba cubre las cuatro estaciones del año; con bloques de 6 meses, uno quedaría sin verano o sin temporada de lluvias y su error no sería comparable.

### Cómo se evita usar información del futuro

1. **Orden temporal estricto:** ningún registro de validación o prueba es anterior a uno de entrenamiento. No hay partición aleatoria.
2. **Features con corte:** las ventanas y rezagos usan solo datos con `t' ≤ t − horizonte` (por ejemplo, la demanda de hace 24 h para predecir mañana). Las medias móviles se calculan con ventana cerrada a la izquierda.
3. **Brecha entre conjuntos:** se descarta un periodo igual al horizonte (24 h para P1, 2 h para P2) en cada frontera, para que ninguna feature del conjunto siguiente use viajes del anterior.
4. **Ajuste solo con entrenamiento:** escaladores, codificadores, imputaciones y agrupamientos de P3 se calculan con el conjunto de entrenamiento y se aplican a los demás.
5. **Validación cruzada temporal:** para ajustar hiperparámetros, ventanas expansivas (`TimeSeriesSplit`) dentro de entrenamiento.
6. **Prueba una sola vez:** el conjunto de prueba se evalúa al final, con el modelo elegido, y no se usa para decidir nada.
7. **Contraste con un baseline:** todo modelo debe superar la media histórica por hora y estación calculada con el entrenamiento.

### Riesgos conocidos de fuga o sesgo

- **La nomenclatura es de 2026-09:** capacidad y ubicación de periodos anteriores son las de hoy. Es un atributo estático, pero una estación reubicada o retirada no se refleja en los años previos. Se documenta como limitación.
- **Huecos 2023-04 y 2023-05:** caen en entrenamiento y rompen los rezagos; no se imputan sin acordarlo.
- **Septiembre de 2026** (490,056 viajes) es el máximo del periodo y cae en prueba; verificar que sea un valor real y no una carga atípica antes de interpretar el error.
- **Cambio de régimen:** el entrenamiento incluye la pandemia y la prueba no; el error en prueba puede ser mejor o peor por esa diferencia, no por el modelo.
