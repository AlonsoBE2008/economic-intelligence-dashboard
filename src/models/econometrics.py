"""Modelos econométricos: proyección ARIMA e índice de riesgo sectorial.

Este módulo ajusta modelos ARIMA sobre las series preparadas por
``src.processing.transformations`` y construye un índice de riesgo
sectorial a partir de volatilidad, tendencia y z-score del último valor.

La ruta del proyecto se añade a ``sys.path`` al inicio para que el módulo
sea ejecutable directamente como script y también importable como parte
del paquete ``src.models``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

from src.processing.transformations import preparar_para_modelo, variacion_mensual


def ajustar_arima(
    df: pd.DataFrame,
    columna_valor: str = "valor",
    orden: tuple[int, int, int] = (1, 1, 1),
) -> dict[str, Any]:
    """Ajusta un modelo ARIMA a una serie del dashboard.

    La serie se prepara primero con ``preparar_para_modelo`` (limpieza,
    índice de fechas regular y diferenciación si es necesario) y luego
    se ajusta el ARIMA con el orden indicado.

    Nota: el parámetro ``disp`` de la API antigua de ARIMA no existe en
    ``statsmodels.tsa.arima.model.ARIMA`` (>= 0.13); el ajuste moderno es
    silencioso por defecto, que es lo que lograba ``disp=False``.

    Args:
        df: DataFrame con columnas "fecha" y la columna de valores
            (salida del fetcher).
        columna_valor: Columna con los valores a modelar.
        orden: Orden (p, d, q) del modelo ARIMA.

    Returns:
        Diccionario con las claves "modelo" (objeto ARIMAResults
        ajustado), "aic" (float), "bic" (float), "orden" (tupla usada),
        "n_observaciones" (int) y "serie" (nombre de la serie).
    """
    df_preparado = preparar_para_modelo(df, columna_valor)
    modelo = ARIMA(df_preparado[columna_valor], order=orden)
    resultados = modelo.fit()
    nombre_serie = (
        str(df["serie"].iloc[0])
        if "serie" in df.columns and len(df) > 0
        else columna_valor
    )
    return {
        "modelo": resultados,
        "aic": float(resultados.aic),
        "bic": float(resultados.bic),
        "orden": orden,
        "n_observaciones": int(resultados.nobs),
        "serie": nombre_serie,
    }


def proyectar(
    resultado_arima: dict[str, Any], pasos: int = 6
) -> pd.DataFrame:
    """Proyecta una serie ajustada con ARIMA hacia adelante.

    Args:
        resultado_arima: Diccionario devuelto por ``ajustar_arima``.
        pasos: Número de períodos futuros a proyectar.

    Returns:
        DataFrame con columnas "fecha" (fechas futuras inferidas desde
        el último índice del modelo), "proyeccion" (valores proyectados),
        "ic_inferior" e "ic_superior" (límites del intervalo de
        confianza al 95%).
    """
    modelo = resultado_arima["modelo"]
    valores = modelo.forecast(steps=pasos)
    forecast_obj = modelo.get_forecast(steps=pasos)
    conf_int = forecast_obj.conf_int(alpha=0.05)
    return pd.DataFrame(
        {
            "fecha": pd.DatetimeIndex(valores.index),
            "proyeccion": valores.to_numpy(dtype="float64"),
            "ic_inferior": conf_int.iloc[:, 0].to_numpy(dtype="float64"),
            "ic_superior": conf_int.iloc[:, 1].to_numpy(dtype="float64"),
        }
    )


def calcular_indice_riesgo(
    datos: dict[str, pd.DataFrame], ventana: int = 12
) -> pd.DataFrame:
    """Calcula el Índice de Riesgo Sectorial de las series del dashboard.

    Para cada serie, sobre los últimos ``ventana`` períodos, mide:
    - volatilidad: desviación estándar de la variación de un período a
      otro,
    - tendencia: pendiente de una regresión lineal simple,
    - z_score_ultimo: desviaciones estándar del último valor respecto a
      la media de la ventana.

    Las tres métricas se normalizan a [0, 1] con MinMaxScaler y el
    índice es el promedio ponderado
    ``0.5 * volatilidad + 0.3 * |z_score| + 0.2 * |tendencia|``.
    Se clasifica en "Bajo" (< 0.33), "Medio" ([0.33, 0.66)) y
    "Alto" (>= 0.66).

    Args:
        datos: Diccionario {nombre_serie: DataFrame} tal como lo devuelve
            ``cargar_datos_dashboard``.
        ventana: Número de períodos recientes usados para las métricas.

    Returns:
        DataFrame con columnas "serie", "volatilidad", "tendencia",
        "z_score_ultimo", "indice_riesgo" y "clasificacion"; una fila por
        serie, ordenada por "indice_riesgo" descendente. Las series con
        menos de 3 observaciones válidas se omiten.
    """
    metricas: list[dict[str, Any]] = []
    for nombre, df in datos.items():
        valores = df.sort_values("fecha")["valor"].dropna()
        if len(valores) < 3:
            continue
        ventana_valores = valores.tail(ventana)
        volatilidad = float(variacion_mensual(ventana_valores).std())
        tendencia = float(
            np.polyfit(
                range(len(ventana_valores)), ventana_valores.to_numpy(), 1
            )[0]
        )
        media = float(ventana_valores.mean())
        desviacion = float(ventana_valores.std())
        z_score = (
            (float(ventana_valores.iloc[-1]) - media) / desviacion
            if desviacion > 0
            else 0.0
        )
        metricas.append(
            {
                "serie": nombre,
                "volatilidad": volatilidad,
                "tendencia": tendencia,
                "z_score_ultimo": z_score,
            }
        )
    df_metricas = pd.DataFrame(
        metricas,
        columns=["serie", "volatilidad", "tendencia", "z_score_ultimo"],
    )
    if df_metricas.empty:
        return df_metricas
    normalizadas = MinMaxScaler().fit_transform(
        df_metricas[["volatilidad", "tendencia", "z_score_ultimo"]]
    )
    df_metricas["indice_riesgo"] = (
        0.5 * normalizadas[:, 0]
        + 0.3 * np.abs(normalizadas[:, 1])
        + 0.2 * np.abs(normalizadas[:, 2])
    )
    df_metricas["clasificacion"] = np.where(
        df_metricas["indice_riesgo"] < 0.33,
        "Bajo",
        np.where(df_metricas["indice_riesgo"] < 0.66, "Medio", "Alto"),
    )
    return (
        df_metricas.sort_values("indice_riesgo", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.fetcher import cargar_datos_dashboard

    print("Cargando datos...")
    datos = cargar_datos_dashboard("2018-1", "2024-6")

    print("\n--- ARIMA tasa_referencia ---")
    resultado = ajustar_arima(datos["tasa_referencia"])
    print(f"AIC: {resultado['aic']:.2f} | BIC: {resultado['bic']:.2f}")
    proyeccion = proyectar(resultado, pasos=6)
    print(proyeccion.to_string())

    print("\n--- Índice de Riesgo Sectorial ---")
    riesgo = calcular_indice_riesgo(datos)
    print(riesgo.to_string())
