
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from datetime import datetime

from simulador import ejecutar_simulacion, validar_portafolio


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="ChaosRisk",
    page_icon="🌪️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .info-card {
        padding: 1rem;
        border-radius: 0.8rem;
        border: 1px solid rgba(128,128,128,0.25);
        background-color: rgba(250,250,250,0.70);
        margin-bottom: 1rem;
    }
    .small-text {
        font-size: 0.9rem;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🌪️ ChaosRisk</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Simulación de riesgos catastróficos con Monte Carlo, teoría del caos y reaseguro.</div>',
    unsafe_allow_html=True
)


# =========================================================
# FUNCIONES AUXILIARES DE INTERFAZ
# =========================================================

def mostrar_requisitos_portafolio():
    st.markdown("### 📌 Requisitos del portafolio de entrada")

    st.info(
        "El archivo debe estar en formato Excel (.xlsx). Cada fila debe representar una póliza asegurada. "
        "Si falta alguna columna, si hay valores negativos, datos vacíos o categorías no reconocidas, "
        "la aplicación lo avisará antes de ejecutar la simulación."
    )

    requisitos = pd.DataFrame(
        {
            "Columna requerida": [
                "tipo_riesgo",
                "tipo_construccion",
                "anio_construccion",
                "ocupacion",
                "valor_expuesto",
                "deducible",
                "limite_cobertura",
            ],
            "Valores esperados": [
                "Terremoto, Inundacion/Inundación, Tormenta",
                "Concreto, Acero, Mamposteria/Mampostería, Mixta",
                "Año numérico entre 1900 y el año actual",
                "Residencial, Comercial, Industrial",
                "Número mayor que 0",
                "Número mayor o igual que 0",
                "Número mayor que 0",
            ],
            "Uso dentro del simulador": [
                "Filtra las pólizas que serán simuladas.",
                "Ajusta el factor de vulnerabilidad.",
                "Ajusta la vulnerabilidad por antigüedad.",
                "Ajusta la vulnerabilidad por uso del inmueble.",
                "Representa el monto económico expuesto.",
                "Reduce la pérdida cubierta por póliza.",
                "Acota la pérdida máxima cubierta por póliza.",
            ],
        }
    )

    st.dataframe(requisitos, use_container_width=True)

    with st.expander("Ver ejemplo de portafolio válido"):
        ejemplo = pd.DataFrame(
            {
                "tipo_riesgo": ["Terremoto", "Inundacion", "Tormenta"],
                "tipo_construccion": ["Concreto", "Mamposteria", "Acero"],
                "anio_construccion": [2010, 1985, 2020],
                "ocupacion": ["Residencial", "Comercial", "Industrial"],
                "valor_expuesto": [2_500_000, 8_000_000, 15_000_000],
                "deducible": [50_000, 100_000, 250_000],
                "limite_cobertura": [2_000_000, 6_000_000, 12_000_000],
            }
        )
        st.dataframe(ejemplo, use_container_width=True)


def explicar_bloque(titulo, texto):
    with st.expander(f"ℹ️ ¿Qué significa {titulo}?"):
        st.write(texto)


def crear_reporte_pdf(resultados, tipo_riesgo, escenario, periodo_anios, n_simulaciones, nivel_confianza):
    """
    Genera un PDF simple con las métricas principales y tablas del reporte.
    Requiere reportlab en requirements.txt:
    reportlab
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak,
        )
    except Exception as exc:
        raise RuntimeError(
            "No se pudo generar el PDF. Agrega 'reportlab' a requirements.txt."
        ) from exc

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.spaceAfter = 8

    small_style = ParagraphStyle(
        "small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
    )

    story = []

    story.append(Paragraph("Reporte de simulación - ChaosRisk", title_style))
    story.append(Paragraph(f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M')}", body_style))
    story.append(Spacer(1, 10))

    parametros = [
        ["Parámetro", "Valor"],
        ["Tipo de riesgo", str(tipo_riesgo)],
        ["Escenario", str(escenario)],
        ["Periodo de aseguramiento", f"{periodo_anios} año(s)"],
        ["Simulaciones Monte Carlo", f"{n_simulaciones:,}"],
        ["Nivel de confianza", f"{nivel_confianza:.0%}"],
    ]

    tabla_parametros = Table(parametros, colWidths=[180, 300])
    tabla_parametros.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(Paragraph("Parámetros de simulación", heading_style))
    story.append(tabla_parametros)
    story.append(Spacer(1, 12))

    resumen = [
        ["Métrica", "Valor"],
        ["Pérdida esperada retenida del periodo", f"${resultados['Perdida Esperada Periodo']:,.2f}"],
        ["AAL retenido anualizado", f"${resultados['AAL Anual']:,.2f}"],
        [f"VaR retenido {nivel_confianza:.0%}", f"${resultados['VaR']:,.2f}"],
        [f"TVaR retenido {nivel_confianza:.0%}", f"${resultados['TVaR']:,.2f}"],
        ["PML retenido", f"${resultados['PML']:,.2f}"],
        ["Prima sugerida del periodo", f"${resultados['Prima']:,.2f}"],
        ["Exponente de Lyapunov final", f"{resultados['Lyapunov']:.6f}"],
        ["Factor caótico final", f"{resultados['Factor Caos']:.6f}"],
        ["Cobertura promedio", f"{resultados['Cobertura Promedio']:.2%}"],
        ["Cobertura peor caso", f"{resultados['Cobertura Peor Caso']:.2%}"],
        ["Recalcular prima", str(resultados["Recalcular Prima"])],
    ]

    tabla_resumen = Table(resumen, colWidths=[250, 230])
    tabla_resumen.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(Paragraph("Resumen ejecutivo", heading_style))
    story.append(tabla_resumen)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Interpretación general", heading_style))
    story.append(
        Paragraph(
            "El reporte resume las pérdidas simuladas para el portafolio cargado. "
            "La pérdida esperada retenida corresponde al promedio de pérdidas que permanecerían en la aseguradora "
            "después de aplicar deducibles, límites de cobertura y reaseguro. El VaR representa un percentil extremo "
            "de la distribución de pérdidas, mientras que el TVaR estima el promedio de las pérdidas que superan dicho umbral. "
            "El exponente de Lyapunov se usa como indicador de sensibilidad del componente no lineal del modelo.",
            body_style,
        )
    )

    story.append(PageBreak())

    story.append(Paragraph("Métricas completas del modelo", heading_style))
    df_metricas = resultados["df_metricas"].copy()
    df_metricas["valor"] = df_metricas["valor"].astype(str)
    data_metricas = [["Métrica", "Valor"]] + df_metricas.values.tolist()
    tabla_metricas = Table(data_metricas[:45], colWidths=[230, 250], repeatRows=1)
    tabla_metricas.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(tabla_metricas)

    story.append(PageBreak())

    story.append(Paragraph("Capas de reaseguro XoL", heading_style))
    df_capas = resultados["df_capas"].copy()
    data_capas = [df_capas.columns.tolist()] + df_capas.astype(str).values.tolist()
    tabla_capas = Table(data_capas, repeatRows=1)
    tabla_capas.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(tabla_capas)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Historial de calibración caótica", heading_style))
    df_cal = resultados["df_historial_calibracion"].copy()
    data_cal = [df_cal.columns.tolist()] + df_cal.round(6).astype(str).values.tolist()
    tabla_cal = Table(data_cal, repeatRows=1)
    tabla_cal.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(tabla_cal)

    doc.build(story)
    buffer.seek(0)
    return buffer


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Configuración")

    archivo_portafolio = st.file_uploader(
        "Sube tu portafolio Excel",
        type=["xlsx"],
        help="El archivo debe contener las columnas requeridas: tipo_riesgo, tipo_construccion, anio_construccion, ocupacion, valor_expuesto, deducible y limite_cobertura."
    )

    tipo_riesgo = st.selectbox(
        "Tipo de desastre natural",
        ["Terremoto", "Inundacion", "Tormenta"],
        help="Selecciona el fenómeno que se simulará. El sistema filtrará únicamente las pólizas del portafolio que coincidan con este tipo de riesgo."
    )

    escenario = st.selectbox(
        "Nivel de severidad",
        ["Moderado", "Severo", "Extremo", "Catastrofico"],
        help="Controla el percentil histórico usado para representar la severidad central del evento."
    )

    periodo_anios = st.number_input(
        "Periodo de aseguramiento",
        min_value=1,
        max_value=30,
        value=1,
        help="Número de años cubiertos por la simulación. El modelo ajusta la frecuencia esperada multiplicando la frecuencia anual por este periodo."
    )

    n_simulaciones = st.slider(
        "Simulaciones Monte Carlo",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        help="Cantidad de escenarios aleatorios generados. Un número mayor suele dar resultados más estables, pero tarda más."
    )

    nivel_confianza = st.selectbox(
        "Nivel de confianza",
        [0.90, 0.95, 0.99],
        index=1,
        help="Percentil usado para calcular VaR y TVaR."
    )

    ejecutar = st.button("Ejecutar simulación", type="primary")


# =========================================================
# BLOQUE DE PORTAFOLIO
# =========================================================

mostrar_requisitos_portafolio()

if archivo_portafolio is None:
    st.warning("Sube un portafolio Excel para comenzar.")
    st.stop()

try:
    df_portafolio = pd.read_excel(archivo_portafolio)
except Exception as exc:
    st.error("No se pudo leer el archivo Excel. Revisa que el archivo no esté dañado y que tenga extensión .xlsx.")
    st.exception(exc)
    st.stop()

validacion = validar_portafolio(df_portafolio)

if not validacion["ok"]:
    st.error("El portafolio tiene errores que deben corregirse antes de ejecutar la simulación.")
    for error in validacion["errores"]:
        st.write(f"❌ {error}")
    st.stop()

if validacion["advertencias"]:
    st.warning("El portafolio fue cargado, pero se encontraron advertencias:")
    for advertencia in validacion["advertencias"]:
        st.write(f"⚠️ {advertencia}")

df_portafolio = validacion["df_limpio"]

st.success(f"Portafolio cargado correctamente: {len(df_portafolio)} pólizas.")

with st.expander("Vista previa del portafolio"):
    st.dataframe(df_portafolio.head(20), use_container_width=True)

conteo_riesgos = df_portafolio["tipo_riesgo"].value_counts().rename_axis("tipo_riesgo").reset_index(name="polizas")
st.markdown("#### Distribución de pólizas por tipo de riesgo")
st.dataframe(conteo_riesgos, use_container_width=True)

if tipo_riesgo not in set(df_portafolio["tipo_riesgo"].unique()):
    st.error(
        f"No hay pólizas con tipo_riesgo = '{tipo_riesgo}'. "
        "Selecciona otro tipo de riesgo o corrige el portafolio."
    )
    st.stop()


# =========================================================
# EJECUCIÓN
# =========================================================

if ejecutar:
    try:
        with st.spinner("Ejecutando simulación Monte Carlo y calibración del componente caótico..."):
            resultados = ejecutar_simulacion(
                df_portafolio=df_portafolio,
                tipo_riesgo=tipo_riesgo,
                escenario=escenario,
                periodo_anios=periodo_anios,
                n_simulaciones=n_simulaciones,
                nivel_confianza=nivel_confianza
            )
    except Exception as exc:
        st.error("No se pudo ejecutar la simulación. Revisa los datos de entrada y los archivos históricos.")
        st.exception(exc)
        st.stop()

    st.divider()

    st.subheader("📊 Resumen del portafolio")

    label_perdida = "AAL retenido" if periodo_anios == 1 else "Pérdida esperada retenida del periodo"

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        label_perdida,
        f"${resultados['Perdida Esperada Periodo']:,.0f}",
        help="Promedio de la pérdida retenida por la aseguradora después de deducibles, límites y reaseguro. Si el periodo es mayor a un año, representa la pérdida esperada acumulada del periodo."
    )

    c2.metric(
        f"VaR {nivel_confianza:.0%}",
        f"${resultados['VaR']:,.0f}",
        help="Pérdida máxima esperada hasta el percentil seleccionado. Por ejemplo, VaR 95% deja aproximadamente el 5% de escenarios con pérdidas mayores."
    )

    c3.metric(
        f"TVaR {nivel_confianza:.0%}",
        f"${resultados['TVaR']:,.0f}",
        help="Promedio de las pérdidas que superan el VaR. Es útil para medir la severidad de la cola de la distribución."
    )

    c4.metric(
        "Prima sugerida",
        f"${resultados['Prima']:,.0f}",
        help="Prima técnica/comercial sugerida para el periodo, considerando pérdida esperada, carga de seguridad y costo estimado del reaseguro."
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Lyapunov",
        f"{resultados['Lyapunov']:.4f}",
        help="Mide la sensibilidad del componente no lineal a condiciones iniciales. Valores positivos altos indican mayor comportamiento caótico. El simulador intenta reducir r si supera el límite definido."
    )

    c6.metric(
        "Factor caótico",
        f"{resultados['Factor Caos']:.4f}",
        help="Factor promedio generado por el mapa logístico y aplicado sobre la severidad simulada."
    )

    c7.metric(
        "Cobertura promedio",
        f"{resultados['Cobertura Promedio']:.2%}",
        help="Porcentaje promedio de pérdida bruta que no permanece retenida por la aseguradora, considerando límites y reaseguro."
    )

    c8.metric(
        "Cobertura peor caso",
        f"{resultados['Cobertura Peor Caso']:.2%}",
        help="Cobertura observada en el escenario de mayor pérdida bruta simulada."
    )

    if periodo_anios > 1:
        st.info(
            f"Como el periodo seleccionado es de {periodo_anios} años, el modelo aumenta la frecuencia esperada de eventos usando lambda_periodo = lambda_anual × periodo. "
            f"El AAL anualizado estimado es: ${resultados['AAL Anual']:,.0f}."
        )

    explicar_bloque(
        "las métricas principales",
        "AAL o pérdida esperada resume el promedio de pérdidas; VaR mide un percentil extremo; TVaR mide el promedio de los escenarios que superan ese percentil; la prima sugerida agrega carga de seguridad y costo de reaseguro."
    )

    st.divider()

    st.subheader("🛡️ Reaseguro y prima")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Prima actual estimada",
        f"${resultados['Prima Actual']:,.0f}",
        help="Estimación de la prima actual usando una regla comercial simple sobre el límite de cobertura o valor expuesto."
    )

    r2.metric(
        "Prima reaseguro",
        f"${resultados['Prima Reaseguro']:,.0f}",
        help="Costo estimado de las capas de reaseguro XoL incluidas en el modelo."
    )

    r3.metric(
        "Diferencia de prima",
        f"${resultados['Diferencia Prima']:,.0f}",
        help="Diferencia entre la prima sugerida por el modelo y la prima actual estimada."
    )

    r4.metric(
        "¿Recalcular prima?",
        resultados["Recalcular Prima"],
        help="Indica si la prima sugerida es mayor que la prima actual estimada."
    )

    explicar_bloque(
        "el bloque de reaseguro y prima",
        "Este bloque permite comparar la prima actual estimada contra la prima sugerida por el modelo. También muestra el costo estimado del reaseguro y si conviene recalcular la prima."
    )

    st.divider()

    df_sim = resultados["df_simulaciones"]

    st.subheader("📉 Distribución de pérdidas retenidas")
    explicar_bloque(
        "la distribución de pérdidas retenidas",
        "El histograma muestra cómo se distribuyen las pérdidas que permanecen en la aseguradora después de aplicar pólizas y reaseguro. La cola derecha representa escenarios extremos."
    )

    fig = px.histogram(
        df_sim,
        x="perdida_retenida_aseguradora",
        nbins=60,
        title="Distribución de pérdidas retenidas por la aseguradora"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Pérdidas brutas vs retenidas")
    explicar_bloque(
        "la gráfica de pérdidas brutas vs retenidas",
        "Cada punto representa una simulación. El eje X muestra la pérdida bruta antes de reaseguro y el eje Y la pérdida retenida por la aseguradora."
    )

    fig2 = px.scatter(
        df_sim,
        x="perdida_bruta",
        y="perdida_retenida_aseguradora",
        title="Relación entre pérdida bruta y pérdida retenida"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🌀 Trayectoria caótica")
    explicar_bloque(
        "la trayectoria caótica",
        "La trayectoria corresponde al mapa logístico usado para generar el factor caótico. Si el exponente de Lyapunov era demasiado alto, el simulador ajustó el parámetro r durante la calibración."
    )

    fig_caos = px.line(
        resultados["df_trayectoria_caos"],
        x="iteracion",
        y="x_t",
        title="Mapa logístico: trayectoria caótica"
    )
    st.plotly_chart(fig_caos, use_container_width=True)

    st.subheader("🔁 Historial de calibración caótica")
    st.dataframe(resultados["df_historial_calibracion"], use_container_width=True)
    explicar_bloque(
        "el historial de calibración",
        "Esta tabla muestra cómo el simulador modificó los parámetros del mapa logístico. Si Lyapunov superaba el límite, se reducía r para disminuir la sensibilidad a condiciones iniciales."
    )

    st.subheader("🧱 Capas de reaseguro XoL")
    st.dataframe(resultados["df_capas"], use_container_width=True)
    explicar_bloque(
        "las capas de reaseguro XoL",
        "Cada capa tiene una retención y un límite. La aseguradora absorbe pérdidas hasta la retención; el reasegurador cubre el exceso hasta el límite de la capa."
    )

    st.subheader("🏠 Clasificación de riesgo")
    st.dataframe(resultados["df_clasificacion"], use_container_width=True)
    explicar_bloque(
        "la clasificación de riesgo",
        "Clasifica pólizas según exposición ajustada por vulnerabilidad. Sirve para detectar pólizas con mayor concentración de riesgo."
    )

    st.subheader("📋 Métricas del modelo")
    st.dataframe(resultados["df_metricas"], use_container_width=True)
    explicar_bloque(
        "la tabla de métricas del modelo",
        "Resume parámetros usados, frecuencia histórica, severidad, resultados de reaseguro, métricas de riesgo, prima sugerida y calibración caótica."
    )

    st.divider()

    st.subheader("⬇️ Descargas")

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="Descargar reporte Excel",
            data=resultados["excel_bytes"].getvalue(),
            file_name=f"reporte_{tipo_riesgo}_{escenario}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with d2:
        try:
            pdf_bytes = crear_reporte_pdf(
                resultados=resultados,
                tipo_riesgo=tipo_riesgo,
                escenario=escenario,
                periodo_anios=periodo_anios,
                n_simulaciones=n_simulaciones,
                nivel_confianza=nivel_confianza,
            )
            st.download_button(
                label="Descargar reporte PDF",
                data=pdf_bytes.getvalue(),
                file_name=f"reporte_{tipo_riesgo}_{escenario}.pdf",
                mime="application/pdf"
            )
        except Exception as exc:
            st.warning("No se pudo habilitar la descarga PDF. Agrega reportlab a requirements.txt.")
            st.caption(str(exc))

else:
    st.warning("Configura los parámetros y presiona 'Ejecutar simulación'.")
