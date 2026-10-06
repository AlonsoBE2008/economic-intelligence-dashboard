"""Generador de reportes PDF del Dashboard de Inteligencia Económica.

Construye un reporte ejecutivo con reportlab: portada de color, resumen
de indicadores clave, gráficos de las series generados con matplotlib en
memoria e Índice de Riesgo Sectorial, con pie de página en todas las
hojas.

METADATOS replica los nombres legibles, unidades y colores usados en la
UI para mantener consistencia visual entre la app y el reporte.
"""

import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

COLOR_PRINCIPAL = colors.HexColor("#1D3557")
COLOR_BORDE = colors.HexColor("#457B9D")
COLOR_FILA_ALTERNA = colors.HexColor("#F1FAEE")
COLOR_CLASIFICACION: dict[str, tuple[colors.Color, colors.Color]] = {
    "Bajo": (colors.HexColor("#2A9D8F"), colors.white),
    "Medio": (colors.HexColor("#E9C46A"), colors.black),
    "Alto": (colors.HexColor("#E63946"), colors.white),
}

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


def _pie_de_pagina(canvas: Canvas, documento: SimpleDocTemplate) -> None:
    """Dibuja el pie de página (número de página y nota de uso interno)."""
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(A4[0] / 2, 24, f"Página {canvas.getPageNumber()}")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawCentredString(
        A4[0] / 2, 13, "Dashboard de Inteligencia Económica — Uso interno"
    )
    canvas.restoreState()


def _tabla_indicadores(datos: dict[str, pd.DataFrame]) -> Table:
    """Construye la tabla resumen de indicadores clave."""
    filas: list[list[str]] = [
        ["Serie", "Último Valor", "Unidad", "Variación Mensual"]
    ]
    for clave, df in datos.items():
        valores = df.sort_values("fecha")["valor"]
        if len(valores) < 2:
            continue
        penultimo = float(valores.iloc[-2])
        ultimo = float(valores.iloc[-1])
        variacion = (ultimo - penultimo) / penultimo * 100 if penultimo else 0.0
        filas.append(
            [
                METADATOS[clave]["nombre"],
                f"{ultimo:,.2f}",
                METADATOS[clave]["unidad"],
                f"{variacion:+.2f}%",
            ]
        )
    tabla = Table(filas)
    tabla.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRINCIPAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [COLOR_FILA_ALTERNA, colors.white],
                ),
                ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tabla


def _grafico_serie(
    df: pd.DataFrame, nombre: str, color: str, ancho: float
) -> Image:
    """Genera el gráfico de línea de una serie, embebido en memoria."""
    df_ordenado = df.sort_values("fecha")
    figura, eje = plt.subplots(figsize=(16, 4))
    eje.plot(df_ordenado["fecha"], df_ordenado["valor"], color=color, linewidth=1.5)
    eje.set_title(nombre)
    eje.grid(alpha=0.3)
    eje.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    bufer = io.BytesIO()
    figura.savefig(bufer, format="png", dpi=100)
    plt.close(figura)
    bufer.seek(0)
    return Image(bufer, width=ancho, height=ancho / 4)


def _tabla_riesgo(riesgo: pd.DataFrame) -> Table:
    """Construye la tabla del Índice de Riesgo Sectorial."""
    filas: list[list[str]] = [
        ["Serie", "Volatilidad", "Tendencia", "Z-Score", "Índice", "Clasificación"]
    ]
    for _, fila in riesgo.iterrows():
        filas.append(
            [
                str(fila["serie"]),
                f"{fila['volatilidad']:.4f}",
                f"{fila['tendencia']:.4f}",
                f"{fila['z_score_ultimo']:.4f}",
                f"{fila['indice_riesgo']:.4f}",
                str(fila["clasificacion"]),
            ]
        )
    estilo: list[tuple[str, object, object, object]] = [
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRINCIPAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [COLOR_FILA_ALTERNA, colors.white],
        ),
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for posicion, clasificacion in enumerate(riesgo["clasificacion"], start=1):
        color_fondo, color_texto = COLOR_CLASIFICACION.get(
            str(clasificacion), (colors.white, colors.black)
        )
        estilo.append(("BACKGROUND", (5, posicion), (5, posicion), color_fondo))
        estilo.append(("TEXTCOLOR", (5, posicion), (5, posicion), color_texto))
    tabla = Table(filas)
    tabla.setStyle(TableStyle(estilo))
    return tabla


def generar_reporte_pdf(
    datos: dict[str, pd.DataFrame],
    riesgo: pd.DataFrame,
    ruta_salida: str = "reporte_economico.pdf",
) -> str:
    """Genera el reporte PDF completo del dashboard.

    Args:
        datos: Diccionario {serie: DataFrame} con las series del BCRP,
            tal como lo devuelve ``cargar_datos_dashboard``.
        riesgo: DataFrame del Índice de Riesgo Sectorial, salida de
            ``calcular_indice_riesgo``.
        ruta_salida: Ruta del PDF a generar (por defecto, la raíz del
            proyecto).

    Returns:
        La ruta del archivo PDF generado.
    """
    fecha_texto = f"Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}"

    def _portada(canvas: Canvas, documento: SimpleDocTemplate) -> None:
        """Dibuja la portada de la primera hoja."""
        canvas.saveState()
        canvas.setFillColor(COLOR_PRINCIPAL)
        canvas.rect(0, A4[1] - 120, A4[0], 120, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 20)
        canvas.drawCentredString(
            A4[0] / 2, A4[1] - 55, "REPORTE DE INTELIGENCIA ECONÓMICA"
        )
        canvas.setFont("Helvetica", 12)
        canvas.drawCentredString(
            A4[0] / 2, A4[1] - 75, "BCRP · Banco Central de Reserva del Perú"
        )
        canvas.setFont("Helvetica", 10)
        canvas.drawCentredString(A4[0] / 2, A4[1] - 93, fecha_texto)
        canvas.restoreState()

    def _portada_y_pie(canvas: Canvas, documento: SimpleDocTemplate) -> None:
        """Dibuja la portada y el pie de página de la primera hoja."""
        _portada(canvas, documento)
        _pie_de_pagina(canvas, documento)

    documento = SimpleDocTemplate(
        ruta_salida,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    estilos = getSampleStyleSheet()
    estilo_titulo = ParagraphStyle(
        "titulo_seccion",
        parent=estilos["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=COLOR_PRINCIPAL,
        spaceBefore=10,
        spaceAfter=4,
    )

    def separador() -> HRFlowable:
        """Crea una línea separadora horizontal nueva."""
        return HRFlowable(width="100%", thickness=1, color=COLOR_BORDE, spaceAfter=10)

    historia: list[Flowable] = []
    historia.append(Spacer(0, 95))
    historia.append(Paragraph("INDICADORES CLAVE", estilo_titulo))
    historia.append(separador())
    historia.append(_tabla_indicadores(datos))

    historia.append(Paragraph("GRÁFICOS DE SERIES", estilo_titulo))
    historia.append(separador())
    for clave, df_serie in datos.items():
        historia.append(
            _grafico_serie(
                df_serie,
                METADATOS[clave]["nombre"],
                METADATOS[clave]["color"],
                documento.width,
            )
        )
        historia.append(Spacer(0, 12))

    historia.append(Paragraph("ÍNDICE DE RIESGO SECTORIAL", estilo_titulo))
    historia.append(separador())
    historia.append(_tabla_riesgo(riesgo))

    documento.build(historia, onFirstPage=_portada_y_pie, onLaterPages=_pie_de_pagina)
    return ruta_salida
