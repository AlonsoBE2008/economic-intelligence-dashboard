"""Transformaciones econométricas de series de tiempo.

Este módulo procesa las series descargadas por ``src.data.fetcher``
(``BCRPFetcher``) para dejarlas listas para análisis y modelado:

- Variaciones porcentuales (interanual y de período a período).
- Test de estacionariedad Augmented Dickey-Fuller (ADF).
- Desestacionalización por descomposición clásica.
- Pipeline de preparación para modelos ARIMA/VAR.
- Resumen descriptivo multiserie.

Convención de entrada: los DataFrames llegan con las columnas "fecha"
(datetime), "valor" (float64) y "serie" (str), tal como los devuelve
``BCRPFetcher.fetch_series``.
"""

from typing import Any

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def variacion_interanual(serie: pd.Series, periodos: int = 12) -> pd.Series:
    """Calcula la variación porcentual interanual de una serie.

    Fórmula: (Xt - Xt-n) / Xt-n * 100, donde n = periodos. Para series
    mensuales usar periodos=12 y para trimestrales periodos=4.

    Args:
        serie: Serie de valores ordenada por fecha ascendente.
        periodos: Retardo, en períodos, contra el que se compara cada
            observación (12 para mensual, 4 para trimestral).

    Returns:
        Serie con el mismo índice que la entrada, con los primeros
        ``periodos`` valores como NaN y nombre
        ``"{serie.name}_var_interanual"``.
    """
    rezagada = serie.shift(periodos)
    resultado = (serie - rezagada) / rezagada * 100
    resultado.name = f"{serie.name}_var_interanual"
    return resultado


def variacion_mensual(serie: pd.Series) -> pd.Series:
    """Calcula la variación porcentual de un período al siguiente.

    Fórmula: (Xt - Xt-1) / Xt-1 * 100.

    Args:
        serie: Serie de valores ordenada por fecha ascendente.

    Returns:
        Serie con el mismo índice que la entrada, con el primer valor
        como NaN y nombre ``"{serie.name}_var_mensual"``.
    """
    rezagada = serie.shift(1)
    resultado = (serie - rezagada) / rezagada * 100
    resultado.name = f"{serie.name}_var_mensual"
    return resultado


def test_estacionariedad(
    serie: pd.Series, alpha: float = 0.05
) -> dict[str, Any]:
    """Ejecuta el test Augmented Dickey-Fuller (ADF) de estacionariedad.

    Args:
        serie: Serie a evaluar; los NaN se ignoran antes del test.
        alpha: Nivel de significancia (por defecto 0.05).

    Returns:
        Diccionario con las claves:
        - "es_estacionaria": True si el p-value es menor que alpha.
        - "p_value": p-value del test.
        - "estadistico_adf": estadístico ADF.
        - "valores_criticos": valores críticos con las claves "1%",
          "5%" y "10%".
        - "interpretacion": texto con la conclusión del test.

    Raises:
        ValueError: Si la serie no tiene observaciones válidas.
    """
    serie_limpia = serie.dropna()
    if serie_limpia.empty:
        raise ValueError("La serie no tiene observaciones válidas para el test ADF.")
    resultado_adf = adfuller(serie_limpia, autolag="AIC")
    p_value = float(resultado_adf[1])
    estadistico = float(resultado_adf[0])
    criticos = resultado_adf[4]
    es_estacionaria = bool(p_value < alpha)
    if es_estacionaria:
        interpretacion = (
            f"Serie estacionaria (p={p_value:.4f}): apta para modelado directo."
        )
    else:
        interpretacion = (
            f"Serie no estacionaria (p={p_value:.4f}): aplicar diferenciación "
            f"antes del modelo."
        )
    return {
        "es_estacionaria": es_estacionaria,
        "p_value": p_value,
        "estadistico_adf": estadistico,
        "valores_criticos": {
            "1%": float(criticos["1%"]),
            "5%": float(criticos["5%"]),
            "10%": float(criticos["10%"]),
        },
        "interpretacion": interpretacion,
    }


def desestacionalizar(
    df: pd.DataFrame, columna_valor: str = "valor", modelo: str = "additive"
) -> pd.DataFrame:
    """Desestacionaliza una serie mediante descomposición clásica.

    Detecta la frecuencia por la diferencia promedio en días entre
    fechas consecutivas: ~30 días (mensual) usa period=12 y ~90 días
    (trimestral) usa period=4. La descomposición se hace con
    ``seasonal_decompose`` usando extrapolación de tendencia.

    Args:
        df: DataFrame con columnas "fecha" y la columna de valores.
        columna_valor: Columna con los valores a descomponer.
        modelo: Modelo de descomposición ("additive" o "multiplicative").

    Returns:
        Copia del DataFrame original con las columnas nuevas "tendencia",
        "estacionalidad" y "residuo".

    Raises:
        ValueError: Si faltan columnas, la serie es muy corta (menos de
            2*period observaciones), contiene NaN o tiene una frecuencia
            no soportada.
    """
    if columna_valor not in df.columns or "fecha" not in df.columns:
        raise ValueError(
            f"El DataFrame debe contener las columnas 'fecha' y "
            f"'{columna_valor}'."
        )
    if len(df) < 2:
        raise ValueError(
            "Se necesitan al menos 2 observaciones para inferir la frecuencia."
        )
    fechas = pd.to_datetime(df["fecha"])
    diferencia_promedio = fechas.diff().dt.days.mean()
    if diferencia_promedio <= 45:
        periodo = 12
    elif diferencia_promedio <= 120:
        periodo = 4
    else:
        raise ValueError(
            f"Frecuencia no soportada para desestacionalizar: diferencia "
            f"promedio de {diferencia_promedio:.1f} días entre observaciones "
            f"(se soportan series mensuales y trimestrales)."
        )
    if len(df) < 2 * periodo:
        raise ValueError(
            f"Serie muy corta para desestacionalizar con period={periodo}: "
            f"se necesitan al menos {2 * periodo} observaciones y solo hay "
            f"{len(df)}."
        )
    if df[columna_valor].isna().any():
        raise ValueError(
            "La serie contiene valores NaN; elimínalos antes de "
            "desestacionalizar."
        )
    descomposicion = seasonal_decompose(
        df[columna_valor], model=modelo, period=periodo, extrapolate_trend="freq"
    )
    df_salida = df.copy()
    df_salida["tendencia"] = descomposicion.trend
    df_salida["estacionalidad"] = descomposicion.seasonal
    df_salida["residuo"] = descomposicion.resid
    return df_salida


def preparar_para_modelo(
    df: pd.DataFrame, columna_valor: str = "valor"
) -> pd.DataFrame:
    """Prepara una serie para modelado ARIMA/VAR.

    Pipeline: elimina NaN, ordena por fecha, establece la fecha como
    índice con la frecuencia inferida (rellenando huecos con el último
    valor disponible), evalúa la estacionariedad con el test ADF y, si
    la serie no es estacionaria, aplica la primera diferencia.

    Args:
        df: DataFrame con columnas "fecha" y la columna de valores.
        columna_valor: Columna con los valores a preparar.

    Returns:
        DataFrame con "fecha" como índice regular (DatetimeIndex), las
        columnas originales restantes, la columna de valores
        (potencialmente diferenciada) y la columna booleana
        "fue_diferenciada".

    Raises:
        ValueError: Si no quedan observaciones tras limpiar o si no se
            puede inferir la frecuencia de las fechas.
    """
    nombre_serie = (
        str(df["serie"].iloc[0])
        if "serie" in df.columns and len(df) > 0
        else columna_valor
    )
    df_limpio = df.dropna(subset=[columna_valor])
    if df_limpio.empty:
        raise ValueError(f"La columna '{columna_valor}' no tiene observaciones válidas.")
    df_limpio = df_limpio.sort_values("fecha").reset_index(drop=True)
    df_limpio = df_limpio.set_index("fecha")
    try:
        frecuencia = pd.infer_freq(df_limpio.index)
    except ValueError as error:
        raise ValueError(
            f"No se pudo inferir la frecuencia de las fechas de "
            f"'{nombre_serie}': {error}"
        ) from error
    if frecuencia is None:
        raise ValueError(
            f"No se pudo inferir la frecuencia de las fechas de '{nombre_serie}'."
        )
    df_limpio = df_limpio.asfreq(frecuencia, method="ffill")
    resultado_adf = test_estacionariedad(df_limpio[columna_valor])
    fue_diferenciada = not resultado_adf["es_estacionaria"]
    if fue_diferenciada:
        df_limpio[columna_valor] = df_limpio[columna_valor].diff()
        df_limpio = df_limpio.dropna(subset=[columna_valor])
    df_limpio["fue_diferenciada"] = fue_diferenciada
    print(f"[{nombre_serie}] ADF: {resultado_adf['interpretacion']}")
    if fue_diferenciada:
        print(f"[{nombre_serie}] Serie no estacionaria: se aplicó primera diferencia.")
    else:
        print(f"[{nombre_serie}] Serie estacionaria: no fue necesaria la diferenciación.")
    return df_limpio


def resumen_descriptivo(datos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Genera un resumen estadístico de todas las series del dashboard.

    Para cada serie calcula: media, mediana, desviación estándar,
    mínimo, máximo, último valor, fecha del último valor y cantidad de
    observaciones válidas. Las series sin observaciones válidas se
    omiten.

    Args:
        datos: Diccionario {nombre_serie: DataFrame} tal como lo devuelve
            ``cargar_datos_dashboard``.

    Returns:
        DataFrame con una fila por serie y las columnas "media",
        "mediana", "desviacion_estandar", "minimo", "maximo",
        "ultimo_valor", "fecha_ultimo_valor" y "observaciones". Los
        valores numéricos están redondeados a 4 decimales.
    """
    filas: dict[str, dict[str, Any]] = {}
    for nombre, df in datos.items():
        df_ordenado = df.sort_values("fecha")
        valores = df_ordenado["valor"].dropna()
        if valores.empty:
            continue
        filas[nombre] = {
            "media": float(valores.mean()),
            "mediana": float(valores.median()),
            "desviacion_estandar": float(valores.std()),
            "minimo": float(valores.min()),
            "maximo": float(valores.max()),
            "ultimo_valor": float(valores.iloc[-1]),
            "fecha_ultimo_valor": df_ordenado.loc[valores.index[-1], "fecha"],
            "observaciones": int(len(valores)),
        }
    resumen = pd.DataFrame.from_dict(filas, orient="index")
    resumen.index.name = "serie"
    return resumen.round(4)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.fetcher import cargar_datos_dashboard

    print("Cargando datos del BCRP...")
    datos = cargar_datos_dashboard("2018-1", "2024-6")

    print("\n--- Resumen Descriptivo ---")
    resumen = resumen_descriptivo(datos)
    print(resumen.to_string())

    for nombre, df in datos.items():
        print(f"\n--- {nombre} ---")
        serie = df.set_index("fecha")["valor"]
        serie.name = nombre

        var_ia = variacion_interanual(serie, periodos=4 if nombre == "pbi_real" else 12)
        print(f"Variación interanual (últimos 3): {var_ia.dropna().tail(3).to_dict()}")

        adf = test_estacionariedad(serie)
        print(f"ADF: {adf['interpretacion']}")

        df_desest = desestacionalizar(df)
        print(f"Desestacionalización OK — columnas: {df_desest.columns.tolist()}")

        df_modelo = preparar_para_modelo(df)
        print(f"Listo para modelo — shape: {df_modelo.shape}, diferenciada: {df_modelo['fue_diferenciada'].iloc[0]}")
