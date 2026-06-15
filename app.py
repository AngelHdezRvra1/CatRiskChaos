import streamlit as st
import pandas as pd
import plotly.express as px

from simulador import ejecutar_simulacion

st.set_page_config(
    page_title="CatRisk Chaos",
    page_icon="🌪️",
    layout="wide"
)

st.title("CatRisk Chaos")
st.subheader(
    "Simulación de riesgos catastróficos con Monte Carlo, teoría del caos y reaseguro"
)

with st.sidebar:
    st.header("Configuración")

    archivo_portafolio = st.file_uploader(
        "Sube tu portafolio Excel",
        type=["xlsx"]
    )

    tipo_riesgo = st.selectbox(
        "Tipo de desastre natural",
        ["Terremoto", "Inundacion", "Tormenta"]
    )

    escenario = st.selectbox(
        "Nivel de severidad",
        ["Moderado", "Severo", "Extremo", "Catastrofico"]
    )

    periodo_anios = st.number_input(
        "Periodo de aseguramiento",
        min_value=1,
        max_value=30,
        value=1
    )

    n_simulaciones = st.slider(
        "Simulaciones Monte Carlo",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )

    nivel_confianza = st.selectbox(
        "Nivel de confianza",
        [0.90, 0.95, 0.99],
        index=1
    )

    ejecutar = st.button("Ejecutar simulación")

if archivo_portafolio is None:
    st.info("Sube un portafolio Excel para comenzar.")
    st.stop()

df_portafolio = pd.read_excel(archivo_portafolio)

st.success(
    f"Portafolio cargado correctamente: {len(df_portafolio)} pólizas"
)

with st.expander("Vista previa del portafolio"):
    st.dataframe(
        df_portafolio.head(20),
        use_container_width=True
    )

if ejecutar:
    with st.spinner("Ejecutando simulación Monte Carlo..."):

        resultados = ejecutar_simulacion(
            df_portafolio=df_portafolio,
            tipo_riesgo=tipo_riesgo,
            escenario=escenario,
            periodo_anios=periodo_anios,
            n_simulaciones=n_simulaciones,
            nivel_confianza=nivel_confianza
        )

    st.divider()

    st.subheader("Resumen del Portafolio")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "AAL retenido",
        f"${resultados['AAL']:,.0f}"
    )

    c2.metric(
        f"VaR {nivel_confianza:.0%}",
        f"${resultados['VaR']:,.0f}"
    )

    c3.metric(
        f"TVaR {nivel_confianza:.0%}",
        f"${resultados['TVaR']:,.0f}"
    )

    c4.metric(
        "Prima sugerida",
        f"${resultados['Prima']:,.0f}"
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Lyapunov",
        f"{resultados['Lyapunov']:.4f}"
    )

    c6.metric(
        "Factor caótico",
        f"{resultados['Factor Caos']:.4f}"
    )

    c7.metric(
        "Cobertura promedio",
        f"{resultados['Cobertura Promedio']:.2%}"
    )

    c8.metric(
        "Cobertura peor caso",
        f"{resultados['Cobertura Peor Caso']:.2%}"
    )

    st.divider()

    st.subheader("Reaseguro y prima")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Prima actual estimada",
        f"${resultados['Prima Actual']:,.0f}"
    )

    r2.metric(
        "Prima reaseguro",
        f"${resultados['Prima Reaseguro']:,.0f}"
    )

    r3.metric(
        "Diferencia de prima",
        f"${resultados['Diferencia Prima']:,.0f}"
    )

    r4.metric(
        "¿Recalcular prima?",
        resultados["Recalcular Prima"]
    )

    st.divider()

    df_sim = resultados["df_simulaciones"]

    st.subheader("Distribución de pérdidas retenidas")

    fig = px.histogram(
        df_sim,
        x="perdida_retenida_aseguradora",
        nbins=60,
        title="Distribución de pérdidas retenidas por la aseguradora"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Pérdidas brutas vs retenidas")

    fig2 = px.scatter(
        df_sim,
        x="perdida_bruta",
        y="perdida_retenida_aseguradora",
        title="Relación entre pérdida bruta y pérdida retenida"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Trayectoria caótica")

    fig_caos = px.line(
        resultados["df_trayectoria_caos"],
        x="iteracion",
        y="x_t",
        title="Mapa logístico: trayectoria caótica"
    )

    st.plotly_chart(
        fig_caos,
        use_container_width=True
    )

    st.subheader("Capas de reaseguro XoL")

    st.dataframe(
        resultados["df_capas"],
        use_container_width=True
    )

    st.subheader("Clasificación de riesgo")

    st.dataframe(
        resultados["df_clasificacion"],
        use_container_width=True
    )

    st.subheader("Métricas del modelo")

    st.dataframe(
        resultados["df_metricas"],
        use_container_width=True
    )

    st.download_button(
        label="Descargar reporte Excel",
        data=resultados["excel_bytes"],
        file_name=f"reporte_{tipo_riesgo}_{escenario}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.warning(
        "Configura los parámetros y presiona 'Ejecutar simulación'."
    )
