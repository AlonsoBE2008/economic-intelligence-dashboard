"""Dashboard de Inteligencia Económica — UI Streamlit.

Aplicación principal con tres páginas: indicadores clave del BCRP,
proyecciones ARIMA con intervalos de confianza e Índice de Riesgo
Sectorial. Los datos se cargan desde la API del BCRP con caché de 1 hora.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Dashboard de Inteligencia Económica",
    page_icon="🇵🇪",
    layout="wide",
)

try:
    from src.data.fetcher import cargar_datos_dashboard
    from src.processing.transformations import (
        preparar_para_modelo,
        resumen_descriptivo,
        variacion_mensual,
    )
    from src.models.econometrics import (
        ajustar_arima,
        calcular_indice_riesgo,
        proyectar,
    )
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    st.stop()


@st.cache_data(ttl=3600)
def cargar_datos() -> dict[str, pd.DataFrame]:
    """Descarga las series del BCRP con caché de una hora."""
    from src.data.fetcher import cargar_datos_dashboard

    return cargar_datos_dashboard("2018-1", "2024-6")


METADATOS: dict[str, dict[str, str]] = {
    "inflacion": {
        "nombre": "Inflación (IPC Lima)",
        "unidad": "Índice (2009=100)",
        "color": "#E63946",
    },
    "tipo_cambio": {
        "nombre": "Tipo de Cambio Interbancario",
        "unidad": "S/ por US$",
        "color": "#457B9D",
    },
    "tasa_referencia": {
        "nombre": "Tasa de Referencia BCRP",
        "unidad": "%",
        "color": "#2A9D8F",
    },
    "pbi_real": {
        "nombre": "PBI Real",
        "unidad": "M S/ de 2007",
        "color": "#E9C46A",
    },
}

COLOR_PROYECCION = "#264653"
COLOR_BANDA_IC = "rgba(38, 70, 83, 0.15)"


def pagina_indicadores(datos: dict[str, pd.DataFrame]) -> None:
    """Página de indicadores: métricas clave, gráfico y resumen."""
    st.header("📊 Indicadores")

    columnas = st.columns(4)
    for columna, clave in zip(columnas, METADATOS):
        with columna:
            if clave not in datos:
                st.metric(label=METADATOS[clave]["nombre"], value="n.d.")
                continue
            serie = datos[clave].sort_values("fecha").set_index("fecha")["valor"]
            variaciones = variacion_mensual(serie)
            variacion_ultimo = variaciones.iloc[-1]
            st.metric(
                label=METADATOS[clave]["nombre"],
                value=f"{serie.iloc[-1]:,.2f}",
                delta=(
                    f"{variacion_ultimo:+.2f}%"
                    if pd.notna(variacion_ultimo)
                    else None
                ),
            )

    nombre_a_clave = {
        METADATOS[clave]["nombre"]: clave
        for clave in METADATOS
        if clave in datos
    }
    nombre_elegido = st.selectbox("Serie a graficar", list(nombre_a_clave))
    clave = nombre_a_clave[nombre_elegido]
    df_serie = datos[clave].sort_values("fecha")

    fig = go.Figure(
        go.Scatter(
            x=df_serie["fecha"],
            y=df_serie["valor"],
            mode="lines",
            name=METADATOS[clave]["nombre"],
            line=dict(color=METADATOS[clave]["color"]),
        )
    )
    fig.update_layout(
        title=METADATOS[clave]["nombre"],
        xaxis_title="Fecha",
        yaxis_title=METADATOS[clave]["unidad"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(resumen_descriptivo(datos), use_container_width=True)

    st.divider()
    st.subheader("📄 Exportar Reporte")
    if st.button("Generar Reporte PDF"):
        with st.spinner("Generando PDF..."):
            from src.reports.pdf_generator import generar_reporte_pdf
            from src.models.econometrics import calcular_indice_riesgo
            riesgo = calcular_indice_riesgo(datos)
            ruta = generar_reporte_pdf(datos, riesgo)
            with open(ruta, "rb") as f:
                pdf_bytes = f.read()
        st.download_button(
            label="⬇️ Descargar PDF",
            data=pdf_bytes,
            file_name="reporte_economico.pdf",
            mime="application/pdf"
        )


def pagina_proyecciones(datos: dict[str, pd.DataFrame]) -> None:
    """Página de proyecciones ARIMA con intervalo de confianza al 95%."""
    st.header("🔮 Proyecciones ARIMA")
    st.info(
        "Los modelos ARIMA proyectan la tendencia de corto plazo. "
        "Los intervalos sombreados representan el 95% de confianza."
    )

    nombre_a_clave = {
        METADATOS[clave]["nombre"]: clave
        for clave in METADATOS
        if clave in datos
    }
    nombre_elegido = st.selectbox("Serie a proyectar", list(nombre_a_clave))
    pasos = st.slider("Pasos a proyectar", 3, 12, 6)

    if st.button("Generar Proyección"):
        clave = nombre_a_clave[nombre_elegido]
        with st.spinner("Ajustando modelo ARIMA..."):
            df_preparado = preparar_para_modelo(datos[clave])
            resultado = ajustar_arima(datos[clave])
            proyeccion = proyectar(resultado, pasos=pasos)

        columna_aic, columna_bic = st.columns(2)
        columna_aic.metric("AIC", f"{resultado['aic']:.2f}")
        columna_bic.metric("BIC", f"{resultado['bic']:.2f}")

        historico = df_preparado["valor"].tail(24)
        fecha_puente = historico.index[-1]
        valor_puente = float(historico.iloc[-1])
        fechas_proyeccion = [fecha_puente] + proyeccion["fecha"].tolist()
        valores_proyeccion = [valor_puente] + proyeccion["proyeccion"].tolist()
        ic_superior = [valor_puente] + proyeccion["ic_superior"].tolist()
        ic_inferior = [valor_puente] + proyeccion["ic_inferior"].tolist()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=historico.index,
                y=historico,
                mode="lines",
                name="Histórico",
                line=dict(color=METADATOS[clave]["color"]),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fechas_proyeccion,
                y=valores_proyeccion,
                mode="lines",
                name="Proyección",
                line=dict(color=COLOR_PROYECCION, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fechas_proyeccion,
                y=ic_superior,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fechas_proyeccion,
                y=ic_inferior,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=COLOR_BANDA_IC,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        escala = (
            "Δ valor (serie diferenciada)"
            if bool(df_preparado["fue_diferenciada"].iloc[0])
            else METADATOS[clave]["unidad"]
        )
        fig.update_layout(
            title=f"Proyección ARIMA — {METADATOS[clave]['nombre']}",
            xaxis_title="Fecha",
            yaxis_title=escala,
        )
        st.plotly_chart(fig, use_container_width=True)


def pagina_riesgo(datos: dict[str, pd.DataFrame]) -> None:
    """Página del Índice de Riesgo Sectorial."""
    st.header("⚠️ Índice de Riesgo")
    riesgo = calcular_indice_riesgo(datos)
    st.dataframe(riesgo, use_container_width=True)

    colores = {"Bajo": "#2A9D8F", "Medio": "#E9C46A", "Alto": "#E63946"}
    fig = go.Figure(
        go.Bar(
            x=riesgo["indice_riesgo"],
            y=riesgo["serie"],
            orientation="h",
            marker_color=[colores[str(c)] for c in riesgo["clasificacion"]],
        )
    )
    fig.update_layout(
        title="Índice de Riesgo Sectorial",
        xaxis_title="Índice de riesgo (0 a 1)",
        yaxis_title="Serie",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    for _, fila in riesgo.iterrows():
        with st.expander(str(fila["serie"])):
            columna1, columna2, columna3 = st.columns(3)
            columna1.metric("Volatilidad", f"{fila['volatilidad']:.4f}")
            columna2.metric("Tendencia", f"{fila['tendencia']:.4f}")
            columna3.metric("Z-score último", f"{fila['z_score_ultimo']:.4f}")


try:
    with st.spinner("Cargando datos del BCRP..."):
        datos = cargar_datos()
except Exception as e:
    st.error(f"Error cargando datos del BCRP: {e}")
    st.stop()

if not datos:
    st.error("No se pudo descargar ninguna serie del BCRP.")
    st.stop()

st.title("Dashboard de Inteligencia Económica 🇵🇪")
st.caption("BCRP · INEI · Banco Mundial")

pagina = st.sidebar.radio(
    "Navegación",
    ["📊 Indicadores", "🔮 Proyecciones ARIMA", "⚠️ Índice de Riesgo"],
)

if pagina == "📊 Indicadores":
    pagina_indicadores(datos)
elif pagina == "🔮 Proyecciones ARIMA":
    pagina_proyecciones(datos)
else:
    pagina_riesgo(datos)
