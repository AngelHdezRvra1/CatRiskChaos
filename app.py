
import streamlit as st
import pandas as pd
import numpy as np
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


def mostrar_diagnostico_histograma(df_sim):
    """
    Muestra diagnósticos para interpretar si el histograma tiene picos por:
    1) simulaciones sin evento,
    2) retención del reaseguro XoL,
    3) cola extrema.
    """
    total = len(df_sim)
    if total == 0:
        return

    perdidas_ret = df_sim["perdida_retenida_aseguradora"]
    perdidas_brutas = df_sim["perdida_bruta"]

    escenarios_cero = int((perdidas_ret == 0).sum())
    escenarios_retencion_100m = int(np.isclose(perdidas_ret, 100_000_000, atol=1).sum())
    escenarios_con_perdida = int((perdidas_ret > 0).sum())

    diagnostico = pd.DataFrame(
        {
            "Indicador": [
                "Simulaciones totales",
                "Escenarios con pérdida retenida igual a 0",
                "Escenarios con pérdida retenida positiva",
                "Escenarios retenidos cerca de 100 millones",
                "Porcentaje en cero",
                "Porcentaje cerca de 100 millones",
                "Pérdida retenida mínima",
                "Pérdida retenida promedio",
                "Pérdida retenida máxima",
                "Pérdida bruta promedio",
                "Pérdida bruta máxima",
            ],
            "Valor": [
                f"{total:,}",
                f"{escenarios_cero:,}",
                f"{escenarios_con_perdida:,}",
                f"{escenarios_retencion_100m:,}",
                f"{escenarios_cero / total:.2%}",
                f"{escenarios_retencion_100m / total:.2%}",
                f"${perdidas_ret.min():,.2f}",
                f"${perdidas_ret.mean():,.2f}",
                f"${perdidas_ret.max():,.2f}",
                f"${perdidas_brutas.mean():,.2f}",
                f"${perdidas_brutas.max():,.2f}",
            ],
        }
    )

    st.subheader("🔎 Diagnóstico del histograma")
    st.dataframe(diagnostico, use_container_width=True)

    if escenarios_cero / total > 0.30:
        st.info(
            "El pico cercano a cero tiene sentido: una parte importante de las simulaciones no tuvo eventos "
            "o generó pérdidas retenidas nulas después de aplicar condiciones de póliza y reaseguro."
        )

    if escenarios_retencion_100m / total > 0.10:
        st.warning(
            "El pico cercano a 100 millones probablemente es causado por la retención de la primera capa XoL. "
            "Cuando muchas pérdidas superan la retención, la pérdida retenida por la aseguradora se concentra alrededor de ese umbral."
        )


def crear_reporte_pdf(resultados, tipo_riesgo, escenario, periodo_anios, n_simulaciones, nivel_confianza):
    """
    Genera un PDF completo y paginado.
    Incluye:
    - Resumen ejecutivo.
    - Métricas.
    - Capas de reaseguro.
    - Historial completo de calibración caótica.
    - Diagnóstico del histograma.
    - Todas las pólizas del portafolio filtrado/clasificado.
    - Una muestra amplia de simulaciones.

    Requiere reportlab en requirements.txt:
    reportlab
    """
    try:
        from reportlab.lib.pagesizes import legal, landscape
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak,
            KeepTogether,
        )
    except Exception as exc:
        raise RuntimeError(
            "No se pudo generar el PDF. Agrega 'reportlab' a requirements.txt."
        ) from exc

    buffer = BytesIO()

    # Legal horizontal da más espacio para tablas anchas.
    page_size = landscape(legal)
    page_width, page_height = page_size
    left_margin = 22
    right_margin = 22
    top_margin = 25
    bottom_margin = 25
    usable_width = page_width - left_margin - right_margin

    doc = SimpleDocTemplate(
        buffer,
        pagesize=page_size,
        rightMargin=right_margin,
        leftMargin=left_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.spaceAfter = 8

    small_style = ParagraphStyle(
        "small_table",
        parent=styles["BodyText"],
        fontSize=5.6,
        leading=6.2,
        wordWrap="CJK",
    )

    header_style = ParagraphStyle(
        "header_table",
        parent=styles["BodyText"],
        fontSize=5.7,
        leading=6.3,
        alignment=1,
        wordWrap="CJK",
        fontName="Helvetica-Bold",
    )

    note_style = ParagraphStyle(
        "note",
        parent=styles["BodyText"],
        fontSize=7,
        leading=8,
    )

    def fmt_valor(x):
        """Formato legible para celdas del PDF."""
        try:
            if pd.isna(x):
                return ""
        except Exception:
            pass

        if isinstance(x, (float, int, np.integer, np.floating)):
            if abs(float(x)) >= 1000:
                return f"{float(x):,.2f}"
            return f"{float(x):.6g}"

        texto = str(x)
        # Evita celdas gigantes en PDF.
        if len(texto) > 80:
            texto = texto[:77] + "..."
        return texto

    def df_to_pdf_table(df, max_rows=None, font_size=5.6, header_font_size=5.7):
        """
        Convierte un DataFrame a tabla ReportLab que se ajusta al ancho disponible.
        La tabla se pagina verticalmente con repeatRows=1.
        """
        df_local = df.copy()

        if max_rows is not None:
            df_local = df_local.head(max_rows)

        # Convertir encabezados y celdas en Paragraph para permitir saltos de línea.
        header = [
            Paragraph(str(col).replace("_", " "), header_style)
            for col in df_local.columns
        ]

        data = [header]

        for _, row in df_local.iterrows():
            data.append([
                Paragraph(fmt_valor(value), small_style)
                for value in row.tolist()
            ])

        ncols = max(len(df_local.columns), 1)

        # Ancho mínimo razonable por columna.
        col_width = usable_width / ncols
        col_widths = [col_width] * ncols

        table = Table(
            data,
            colWidths=col_widths,
            repeatRows=1,
            splitByRow=True
        )

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), font_size),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )

        return table

    def agregar_tabla_df(story, titulo, df, descripcion=None, max_rows=None):
        story.append(Paragraph(titulo, heading_style))

        if descripcion:
            story.append(Paragraph(descripcion, note_style))
            story.append(Spacer(1, 5))

        if df is None or df.empty:
            story.append(Paragraph("No hay información disponible para esta sección.", body_style))
            story.append(Spacer(1, 10))
            return

        story.append(df_to_pdf_table(df, max_rows=max_rows))
        story.append(Spacer(1, 10))

    story = []

    story.append(Paragraph("Reporte de simulación - ChaosRisk", title_style))
    story.append(Paragraph(f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M')}", body_style))
    story.append(Spacer(1, 8))

    parametros = pd.DataFrame(
        {
            "Parámetro": [
                "Tipo de riesgo",
                "Escenario",
                "Periodo de aseguramiento",
                "Simulaciones Monte Carlo",
                "Nivel de confianza",
            ],
            "Valor": [
                str(tipo_riesgo),
                str(escenario),
                f"{periodo_anios} año(s)",
                f"{n_simulaciones:,}",
                f"{nivel_confianza:.0%}",
            ],
        }
    )

    agregar_tabla_df(
        story,
        "Parámetros de simulación",
        parametros,
        "Configuración seleccionada por el usuario para generar los escenarios de pérdida."
    )

    resumen = pd.DataFrame(
        {
            "Métrica": [
                "Pérdida esperada retenida del periodo",
                "AAL retenido anualizado",
                f"VaR retenido {nivel_confianza:.0%}",
                f"TVaR retenido {nivel_confianza:.0%}",
                "PML retenido",
                "Prima sugerida del periodo",
                "Exponente de Lyapunov final",
                "Factor caótico final",
                "Cobertura promedio",
                "Cobertura peor caso",
                "Recalcular prima",
            ],
            "Valor": [
                f"${resultados['Perdida Esperada Periodo']:,.2f}",
                f"${resultados['AAL Anual']:,.2f}",
                f"${resultados['VaR']:,.2f}",
                f"${resultados['TVaR']:,.2f}",
                f"${resultados['PML']:,.2f}",
                f"${resultados['Prima']:,.2f}",
                f"{resultados['Lyapunov']:.6f}",
                f"{resultados['Factor Caos']:.6f}",
                f"{resultados['Cobertura Promedio']:.2%}",
                f"{resultados['Cobertura Peor Caso']:.2%}",
                str(resultados["Recalcular Prima"]),
            ],
        }
    )

    agregar_tabla_df(
        story,
        "Resumen ejecutivo",
        resumen,
        "Principales resultados agregados del portafolio después de aplicar simulación, componente caótico y reaseguro."
    )

    story.append(Paragraph("Interpretación general", heading_style))
    story.append(
        Paragraph(
            "El reporte resume las pérdidas simuladas para el portafolio cargado. "
            "La pérdida esperada retenida corresponde al promedio de pérdidas que permanecerían en la aseguradora "
            "después de aplicar deducibles, límites de cobertura y reaseguro. El VaR representa un percentil extremo "
            "de la distribución de pérdidas, mientras que el TVaR estima el promedio de las pérdidas que superan dicho umbral. "
            "El exponente de Lyapunov se utiliza como indicador de sensibilidad del componente no lineal del modelo.",
            note_style,
        )
    )

    story.append(PageBreak())

    agregar_tabla_df(
        story,
        "Métricas completas del modelo",
        resultados["df_metricas"],
        "Tabla completa de parámetros, calibración y métricas finales del modelo."
    )

    story.append(PageBreak())

    agregar_tabla_df(
        story,
        "Capas de reaseguro XoL",
        resultados["df_capas"],
        "Capas utilizadas para transferir pérdidas de la aseguradora hacia el reasegurador."
    )

    story.append(PageBreak())

    agregar_tabla_df(
        story,
        "Historial completo de calibración caótica",
        resultados["df_historial_calibracion"],
        "Esta tabla muestra todas las iteraciones de calibración. Si el exponente de Lyapunov supera el límite definido, el modelo reduce el parámetro r para disminuir la sensibilidad a condiciones iniciales."
    )

    if "df_diagnostico_histograma" in resultados:
        story.append(PageBreak())
        agregar_tabla_df(
            story,
            "Diagnóstico del histograma",
            resultados["df_diagnostico_histograma"],
            "Ayuda a interpretar si los picos del histograma provienen de escenarios sin pérdida, de la retención XoL o de pérdidas extremas."
        )

    story.append(PageBreak())

    df_clasificacion_pdf = resultados["df_clasificacion"].copy()

    # Reordenar columnas para que primero salgan las variables más importantes.
    columnas_preferidas = [
        "tipo_riesgo",
        "tipo_construccion",
        "anio_construccion",
        "ocupacion",
        "valor_expuesto",
        "deducible",
        "limite_cobertura",
        "factor_vulnerabilidad",
        "exposicion_ajustada",
        "clasificacion_riesgo",
    ]

    columnas_existentes = [c for c in columnas_preferidas if c in df_clasificacion_pdf.columns]
    columnas_restantes = [c for c in df_clasificacion_pdf.columns if c not in columnas_existentes]
    df_clasificacion_pdf = df_clasificacion_pdf[columnas_existentes + columnas_restantes]

    agregar_tabla_df(
        story,
        f"Clasificación de riesgo por póliza ({len(df_clasificacion_pdf)} pólizas)",
        df_clasificacion_pdf,
        "Se muestran todas las pólizas disponibles para el tipo de riesgo seleccionado. Si el portafolio contiene 100 pólizas, se imprimen las 100; si contiene x pólizas, se imprimen x pólizas. La tabla se divide automáticamente en varias páginas cuando es necesario."
    )

    story.append(PageBreak())

    df_sim_pdf = resultados["df_simulaciones"].copy()

    # Las simulaciones pueden ser 10,000 o 50,000; incluir todas haría un PDF enorme.
    # Se incluye una muestra amplia y el Excel queda como respaldo completo.
    max_sim_pdf = min(len(df_sim_pdf), 300)

    agregar_tabla_df(
        story,
        f"Muestra de simulaciones Monte Carlo ({max_sim_pdf} de {len(df_sim_pdf)})",
        df_sim_pdf.head(max_sim_pdf),
        "Por tamaño del archivo, el PDF incluye una muestra de simulaciones. El archivo Excel descargable contiene la tabla completa de simulaciones."
    )

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
        "El histograma muestra cómo se distribuyen las pérdidas que permanecen en la aseguradora después de aplicar pólizas y reaseguro. "
        "Un pico en cero suele indicar simulaciones sin evento o sin pérdida retenida. Un pico cerca de la retención puede aparecer por el reaseguro XoL."
    )

    fig = px.histogram(
        df_sim,
        x="perdida_retenida_aseguradora",
        nbins=80,
        title="Distribución de pérdidas retenidas por la aseguradora"
    )
    fig.update_layout(
        xaxis_title="Pérdida retenida por la aseguradora",
        yaxis_title="Número de simulaciones",
        bargap=0.03
    )
    st.plotly_chart(fig, use_container_width=True)

    mostrar_diagnostico_histograma(df_sim)

    st.subheader("📊 Distribución de pérdidas brutas")
    explicar_bloque(
        "la distribución de pérdidas brutas",
        "Esta gráfica muestra las pérdidas antes de aplicar reaseguro. Sirve para comparar si los picos del histograma retenido vienen del modelo de eventos o del efecto de la retención XoL."
    )

    fig_bruta = px.histogram(
        df_sim,
        x="perdida_bruta",
        nbins=80,
        title="Distribución de pérdidas brutas antes de reaseguro"
    )
    fig_bruta.update_layout(
        xaxis_title="Pérdida bruta",
        yaxis_title="Número de simulaciones",
        bargap=0.03
    )
    st.plotly_chart(fig_bruta, use_container_width=True)

    df_sim_positivas = df_sim[df_sim["perdida_retenida_aseguradora"] > 0].copy()

    if not df_sim_positivas.empty:
        st.subheader("📈 Distribución de pérdidas retenidas positivas")
        explicar_bloque(
            "la distribución sin ceros",
            "Esta vista elimina las simulaciones con pérdida retenida igual a cero para observar mejor la forma de la cola y los escenarios con siniestros."
        )

        fig_pos = px.histogram(
            df_sim_positivas,
            x="perdida_retenida_aseguradora",
            nbins=80,
            title="Distribución de pérdidas retenidas positivas"
        )
        fig_pos.update_layout(
            xaxis_title="Pérdida retenida positiva",
            yaxis_title="Número de simulaciones",
            bargap=0.03
        )
        st.plotly_chart(fig_pos, use_container_width=True)

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
