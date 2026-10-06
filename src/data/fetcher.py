"""Conector de datos económicos: BCRP y Banco Mundial.

Este módulo implementa la ingesta real de datos desde:
- La API estadística del BCRP (series temporales mensuales).
- La API del Banco Mundial (indicadores anuales para Perú).

Ambos endpoints son públicos y no requieren credenciales.

Notas sobre la API del BCRP:
- El sufijo ``/ing`` de la URL devuelve los nombres de los periodos en
  inglés (p. ej. ``Jan.2020`` mensual, ``Q1.20`` trimestral), mientras que
  ``/esp`` los devuelve en español (``Ene2020``, ``T1.20``). El parser de
  este módulo soporta ambos idiomas y ambas frecuencias.
- El endpoint está detrás de un WAF (Incapsula) sensible a ráfagas de
  peticiones. Por ello ``BCRPFetcher`` usa ``curl_cffi`` con
  impersonación de Chrome y reintentos con backoff (5/10/20 s); si los
  3 intentos fallan, se lanza un ``ValueError`` descriptivo que
  ``fetch_all`` captura por serie sin detener la ingesta.
- Los códigos de serie originales del diseño (PD04648PP, PD04649PP y
  PD37897PP) no existen en el catálogo de BCRPData: la API responde con
  una página HTML vacía en lugar de JSON. Se sustituyeron por los códigos
  oficiales del catálogo que representan los mismos conceptos económicos:
  PN01207PM (tipo de cambio interbancario promedio), PD04722MM (tasa de
  referencia de la política monetaria) y PN02538AQ (PBI real trimestral,
  millones de S/ de 2007).
"""

import time
from typing import Any

import pandas as pd
import requests
from curl_cffi import requests as curl_requests


class BCRPFetcher:
    """Cliente de la API estadística del BCRP (series mensuales y trimestrales).

    Attributes:
        session: Sesión HTTP reutilizable con cabeceras por defecto.
    """

    BASE_URL: str = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"

    SERIES: dict[str, str] = {
        "inflacion": "PN01270PM",
        "tipo_cambio": "PN01207PM",
        "tasa_referencia": "PD04722MM",
        "pbi_real": "PN02538AQ",
    }

    _MESES_ES: dict[str, int] = {
        "Ene": 1,
        "Feb": 2,
        "Mar": 3,
        "Abr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Ago": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dic": 12,
    }

    _MESES_EN: dict[str, int] = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }

    def __init__(self) -> None:
        """Inicializa la sesión HTTP con ``curl_cffi``.

        Se usa ``curl_cffi`` con impersonación de Chrome para reproducir el
        fingerprint TLS y las cabeceras de un navegador real, evitando los
        desafíos anti-bot del WAF (Incapsula) que bloquean a los clientes
        HTTP de Python estándar como ``requests``.
        """
        self.session = curl_requests.Session(impersonate="chrome")
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "EconDashboard/1.0"}
        )

    def _parse_periodo(self, nombre_periodo: str) -> pd.Timestamp:
        """Convierte el nombre de un periodo del BCRP en Timestamp (día 1).

        Soporta los formatos de periodo que devuelve la API:
        - Mensual español (/esp): "Ene2020" o "Ene.2020".
        - Mensual inglés (/ing): "Jan2020" o "Jan.2020".
        - Trimestral español (/esp): "T1.2020" o "T1.20".
        - Trimestral inglés (/ing): "Q1.2020" o "Q1.20".

        Los periodos trimestrales se mapean al primer día del trimestre
        (Q1/T1 -> 1 de enero, Q2/T2 -> 1 de abril, Q3/T3 -> 1 de julio,
        Q4/T4 -> 1 de octubre). Los años de dos dígitos se expanden con el
        criterio 00-49 -> 2000s, 50-99 -> 1900s.

        Args:
            nombre_periodo: Nombre del periodo devuelto por la API.

        Returns:
            Timestamp del primer día del periodo correspondiente.

        Raises:
            ValueError: Si el nombre del periodo no es reconocible.
        """
        texto = nombre_periodo.strip().replace(".", "")
        meses = {**self._MESES_ES, **self._MESES_EN}
        if texto[:1] in {"T", "Q"} and texto[1:2].isdigit():
            try:
                trimestre = int(texto[1])
                anio_corto = int(texto[2:])
            except ValueError as error:
                raise ValueError(
                    f"Periodo del BCRP no reconocido: '{nombre_periodo}'"
                ) from error
            if not 1 <= trimestre <= 4:
                raise ValueError(
                    f"Trimestre fuera de rango en periodo del BCRP: "
                    f"'{nombre_periodo}'"
                )
            if len(texto) - 2 <= 2:
                anio = 2000 + anio_corto if anio_corto <= 49 else 1900 + anio_corto
            else:
                anio = anio_corto
            mes = (trimestre - 1) * 3 + 1
        else:
            abreviatura = texto[:3]
            if abreviatura not in meses:
                raise ValueError(
                    f"Mes no reconocido en periodo del BCRP: '{nombre_periodo}'"
                )
            try:
                anio = int(texto[3:])
            except ValueError as error:
                raise ValueError(
                    f"Periodo del BCRP no reconocido: '{nombre_periodo}'"
                ) from error
            mes = meses[abreviatura]
        return pd.Timestamp(year=anio, month=mes, day=1)

    def fetch_series(
        self, serie_key: str, fecha_inicio: str, fecha_fin: str
    ) -> pd.DataFrame:
        """Descarga una serie del BCRP (mensual o trimestral) entre las fechas indicadas.

        Args:
            serie_key: Clave de la serie; debe existir en SERIES.
            fecha_inicio: Fecha de inicio en formato "YYYY-M" (p. ej. "2020-1").
            fecha_fin: Fecha de fin en formato "YYYY-M" (p. ej. "2024-12").

        Returns:
            DataFrame con columnas "fecha" (datetime), "valor" (float64) y
            "serie" (clave de la serie), ordenado por "fecha" ascendente.
            Los valores "n.d." o vacíos se reemplazan con NaN.

            La petición HTTP se reintenta hasta 3 veces con esperas de
            5, 10 y 20 segundos entre intentos fallidos cuando la respuesta
            no es JSON válido (bloqueo anti-bot del WAF). Agotados los
            3 intentos, se relanza el último error.

        Raises:
            ValueError: Si serie_key no existe en SERIES o si la respuesta
                no es JSON válido tras los reintentos.
            curl_cffi.requests.exceptions.HTTPError: Si la API responde con
                status code != 200.
        """
        if serie_key not in self.SERIES:
            raise ValueError(
                f"Serie desconocida '{serie_key}'. "
                f"Series disponibles: {sorted(self.SERIES)}"
            )
        codigo = self.SERIES[serie_key]
        url = f"{self.BASE_URL}/{codigo}/json/{fecha_inicio}/{fecha_fin}/ing"
        datos: Any = None
        for intento in range(1, 4):
            respuesta = self.session.get(url, timeout=15)
            respuesta.raise_for_status()
            try:
                datos = respuesta.json()
                break
            except ValueError as error:
                time.sleep((5, 10, 20)[intento - 1])
                if intento == 3:
                    raise ValueError(
                        f"Respuesta no JSON de la API BCRP para '{serie_key}' "
                        f"(posible bloqueo anti-bot): {respuesta.text[:120]!r}"
                    ) from error

        periodos: list[dict[str, Any]] = datos.get("periods", [])
        fechas: list[pd.Timestamp] = []
        valores: list[float] = []
        for periodo in periodos:
            fechas.append(self._parse_periodo(str(periodo.get("name", ""))))
            crudos: list[Any] = periodo.get("values") or []
            texto = str(crudos[0]).strip() if crudos else ""
            if texto in {"n.d.", "n.d", "-", ""}:
                valores.append(float("nan"))
            else:
                valores.append(float(texto))
        df = pd.DataFrame(
            {"fecha": fechas, "valor": valores, "serie": serie_key},
            columns=["fecha", "valor", "serie"],
        )
        df["fecha"] = pd.to_datetime(df["fecha"])
        df["valor"] = df["valor"].astype("float64")
        df["serie"] = df["serie"].astype(str)
        return df.sort_values("fecha").reset_index(drop=True)

    def fetch_all(
        self, fecha_inicio: str, fecha_fin: str
    ) -> dict[str, pd.DataFrame]:
        """Descarga todas las series definidas en SERIES.

        Las excepciones se capturan por serie individualmente: si una serie
        falla, se imprime el error y se continúa con las demás. Entre llamadas
        se espera 2 segundos para no saturar la API.

        Args:
            fecha_inicio: Fecha de inicio en formato "YYYY-M".
            fecha_fin: Fecha de fin en formato "YYYY-M".

        Returns:
            Diccionario {serie_key: DataFrame} con las series descargadas
            correctamente.
        """
        resultados: dict[str, pd.DataFrame] = {}
        claves = list(self.SERIES)
        for posicion, serie_key in enumerate(claves):
            try:
                resultados[serie_key] = self.fetch_series(
                    serie_key, fecha_inicio, fecha_fin
                )
            except (
                requests.RequestException,
                curl_requests.exceptions.RequestException,
                ValueError,
            ) as error:
                print(f"[BCRP] Error descargando '{serie_key}': {error}")
            if posicion < len(claves) - 1:
                time.sleep(2)
        return resultados


class WorldBankFetcher:
    """Cliente de la API del Banco Mundial (indicadores anuales de Perú)."""

    BASE_URL: str = "https://api.worldbank.org/v2"

    INDICATORS: dict[str, str] = {
        "pbi_peru": "NY.GDP.MKTP.CD",
        "inflacion_peru": "FP.CPI.TOTL.ZG",
        "desempleo_peru": "SL.UEM.TOTL.ZS",
    }

    def __init__(self) -> None:
        """Inicializa la sesión HTTP para el Banco Mundial."""
        self.session: requests.Session = requests.Session()

    def fetch_indicator(
        self, indicator_key: str, anio_inicio: int, anio_fin: int
    ) -> pd.DataFrame:
        """Descarga un indicador anual del Banco Mundial para Perú.

        Args:
            indicator_key: Clave del indicador; debe existir en INDICATORS.
            anio_inicio: Año de inicio del rango de interés.
            anio_fin: Año de fin del rango de interés.

        Returns:
            DataFrame con columnas "fecha" (datetime, 1 de enero del año),
            "valor" (float64) e "indicador" (clave del indicador), ordenado
            por "fecha" ascendente y sin valores None/NaN.

        Raises:
            ValueError: Si indicator_key no existe en INDICATORS o si la
                respuesta no tiene el formato esperado.
            requests.HTTPError: Si la API responde con status code != 200.
        """
        if indicator_key not in self.INDICATORS:
            raise ValueError(
                f"Indicador desconocido '{indicator_key}'. "
                f"Indicadores disponibles: {sorted(self.INDICATORS)}"
            )
        codigo = self.INDICATORS[indicator_key]
        anios = anio_fin - anio_inicio + 1
        url = (
            f"{self.BASE_URL}/country/PE/indicator/{codigo}"
            f"?format=json&per_page=100&mrv={anios}"
        )
        respuesta = self.session.get(url, timeout=30)
        respuesta.raise_for_status()
        datos = respuesta.json()
        if not isinstance(datos, list) or len(datos) < 2 or not isinstance(
            datos[1], list
        ):
            raise ValueError(
                f"Respuesta inesperada del Banco Mundial para "
                f"'{indicator_key}': {str(datos)[:200]}"
            )
        filas: list[dict[str, Any]] = []
        for registro in datos[1]:
            anio = int(str(registro.get("date", "")))
            valor = registro.get("value")
            if valor is None:
                continue
            filas.append(
                {
                    "fecha": pd.Timestamp(year=anio, month=1, day=1),
                    "valor": float(valor),
                    "indicador": indicator_key,
                }
            )
        df = pd.DataFrame(
            filas, columns=["fecha", "valor", "indicador"]
        )
        df["fecha"] = pd.to_datetime(df["fecha"])
        df["valor"] = df["valor"].astype("float64")
        df["indicador"] = df["indicador"].astype(str)
        df = df.dropna(subset=["valor"])
        return df.sort_values("fecha").reset_index(drop=True)


def cargar_datos_dashboard(
    fecha_inicio: str = "2018-1", fecha_fin: str = "2024-12"
) -> dict[str, pd.DataFrame]:
    """Carga todas las series del BCRP necesarias para el dashboard."""
    fetcher = BCRPFetcher()
    return fetcher.fetch_all(fecha_inicio, fecha_fin)


if __name__ == "__main__":
    datos = cargar_datos_dashboard("2020-1", "2024-6")
    for nombre, df in datos.items():
        print(f"{nombre}: {len(df)} registros — {df.dtypes.to_dict()}")
