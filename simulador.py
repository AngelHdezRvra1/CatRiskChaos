# =====================================================
# FASE 2 + FASE 3 + REASEGURO XoL MULTICAPA AJUSTADO
# Monte Carlo histórico + Modelo Caótico + Lyapunov
# Trabajo Terminal:
# Teoría del Caos en Seguros y Riesgos Catastróficos
# Autor: Angel Hernandez Rivera
# =====================================================

import numpy as np
import pandas as pd

np.random.seed(42)

# =====================================================
# 1. ENTRADAS DEL USUARIO
# =====================================================

archivo_portafolio = "portafolio_catastrofico_100_polizas.xlsx"
archivo_historico = "Mexico_Datos.xlsx"

tipo_riesgo = "Terremoto"
escenario = "Severo"

periodo_anios = 1
n_simulaciones = 10000
nivel_confianza = 0.95

usar_frecuencia_historica = True
usar_severidad_historica = True
lambda_manual = 0.25

# =====================================================
# 2. MODELO CAÓTICO
# =====================================================

usar_modelo_caotico = True

r_inicial = 3.75
x0_inicial = 0.35
iteraciones_caos = 100

activar_loop_calibracion = True
max_iteraciones_calibracion = 10

limite_lyapunov = 0.20
limite_cobertura_peor = 0.55

# =====================================================
# 3. REASEGURO XoL MULTICAPA AJUSTADO
# =====================================================

usar_reaseguro_xol = True

capas_xol = [
    {
        "nombre": "Capa 1",
        "retencion": 100_000_000,
        "limite": 400_000_000,
        "costo": 0.07
    },
    {
        "nombre": "Capa 2",
        "retencion": 500_000_000,
        "limite": 300_000_000,
        "costo": 0.05
    }
]

# =====================================================
# 4. MAPEOS
# =====================================================

mapeo_eventos = {
    "Terremoto": "Earthquake",
    "Inundacion": "Flood",
    "Tormenta": "Storm"
}

percentiles_escenario = {
    "Moderado": 0.50,
    "Severo": 0.75,
    "Extremo": 0.90,
    "Catastrofico": 0.95
}

# =====================================================
# 5. CARGA DE DATOS
# =====================================================

df_portafolio = pd.read_excel(archivo_portafolio)

df_hist = pd.read_excel(
    archivo_historico,
    sheet_name="EM-DAT Data"
)

df_riesgo = df_portafolio[
    df_portafolio["tipo_riesgo"] == tipo_riesgo
].copy()

if df_riesgo.empty:
    raise ValueError("No hay pólizas para ese tipo de riesgo.")

tipo_emdat = mapeo_eventos[tipo_riesgo]

df_eventos = df_hist[
    df_hist["Disaster Type"] == tipo_emdat
].copy()

if df_eventos.empty:
    raise ValueError("No hay eventos históricos para ese riesgo.")

# =====================================================
# 6. FRECUENCIA HISTÓRICA
# =====================================================

anio_min = int(df_eventos["Start Year"].min())
anio_max = int(df_eventos["Start Year"].max())

anios_observados = anio_max - anio_min + 1
eventos_observados = len(df_eventos)

lambda_historica = eventos_observados / anios_observados

lambda_anual = lambda_historica if usar_frecuencia_historica else lambda_manual
lambda_periodo = lambda_anual * periodo_anios

# =====================================================
# 7. SEVERIDAD HISTÓRICA
# =====================================================

def obtener_variable_severidad(df_eventos):

    posibles_columnas = [
        "Insured Damage, Adjusted ('000 US$)",
        "Insured Damage ('000 US$)",
        "Total Damage, Adjusted ('000 US$)",
        "Total Damage ('000 US$)",
        "Total Affected",
        "Magnitude"
    ]

    for col in posibles_columnas:

        if col in df_eventos.columns:

            serie = df_eventos[col].dropna()
            serie = serie[serie > 0]

            if len(serie) >= 5:
                return col, serie

    raise ValueError("No hay suficientes datos históricos de severidad.")


col_severidad, severidad_historica = obtener_variable_severidad(df_eventos)

percentil = percentiles_escenario[escenario]

severidad_base = severidad_historica.quantile(percentil)
severidad_min = severidad_historica.min()
severidad_max = severidad_historica.max()

factor_severidad_historico = (
    (severidad_base - severidad_min)
    / (severidad_max - severidad_min)
)

factor_severidad_historico = np.clip(
    factor_severidad_historico,
    0.05,
    1.00
)

factores_manual = {
    "Moderado": 0.20,
    "Severo": 0.35,
    "Extremo": 0.55,
    "Catastrofico": 0.80
}

severidad_central = (
    factor_severidad_historico
    if usar_severidad_historica
    else factores_manual[escenario]
)

# =====================================================
# 8. MODELO CAÓTICO: MAPA LOGÍSTICO + LYAPUNOV
# =====================================================

def generar_trayectoria_caotica(r, x0, n):

    x = x0
    trayectoria = []

    for _ in range(n):
        x = r * x * (1 - x)
        trayectoria.append(x)

    return np.array(trayectoria)


def calcular_lyapunov_logistico(r, trayectoria):

    derivadas = np.abs(
        r * (1 - 2 * trayectoria)
    )

    derivadas = np.where(
        derivadas == 0,
        1e-10,
        derivadas
    )

    return np.mean(np.log(derivadas))


def obtener_factor_caos(r, x0):

    trayectoria = generar_trayectoria_caotica(
        r,
        x0,
        iteraciones_caos
    )

    lyapunov = calcular_lyapunov_logistico(
        r,
        trayectoria
    )

    factor_caos = 0.75 + 0.50 * np.mean(trayectoria)

    return factor_caos, lyapunov, trayectoria

# =====================================================
# 9. FACTORES DE VULNERABILIDAD
# =====================================================

def factor_construccion(tipo):

    return {
        "Concreto": 0.80,
        "Acero": 0.75,
        "Mamposteria": 1.20,
        "Mixta": 1.00
    }.get(tipo, 1.00)


def factor_antiguedad(anio):

    if anio < 1980:
        return 1.30

    elif anio < 2000:
        return 1.10

    elif anio < 2015:
        return 1.00

    else:
        return 0.90


def factor_ocupacion(ocupacion):

    return {
        "Residencial": 0.90,
        "Comercial": 1.00,
        "Industrial": 1.15
    }.get(ocupacion, 1.00)


def calcular_perdida_poliza(poliza, severidad_evento):

    vulnerabilidad = (
        factor_construccion(poliza["tipo_construccion"])
        * factor_antiguedad(poliza["anio_construccion"])
        * factor_ocupacion(poliza["ocupacion"])
    )

    perdida_bruta = (
        poliza["valor_expuesto"]
        * severidad_evento
        * vulnerabilidad
    )

    perdida_neta = max(
        perdida_bruta - poliza["deducible"],
        0
    )

    perdida_neta = min(
        perdida_neta,
        poliza["limite_cobertura"]
    )

    return perdida_bruta, perdida_neta

# =====================================================
# 10. REASEGURO XoL MULTICAPA
# =====================================================

def aplicar_reaseguro_xol_multicapa(perdida_neta_antes_reaseguro):

    if not usar_reaseguro_xol:

        detalle_vacio = {
            capa["nombre"]: 0
            for capa in capas_xol
        }

        return perdida_neta_antes_reaseguro, 0, detalle_vacio

    perdida_cedida_total = 0
    detalle_capas = {}

    for capa in capas_xol:

        nombre = capa["nombre"]
        retencion = capa["retencion"]
        limite = capa["limite"]

        perdida_cedida_capa = min(
            max(
                perdida_neta_antes_reaseguro - retencion,
                0
            ),
            limite
        )

        perdida_cedida_total += perdida_cedida_capa
        detalle_capas[nombre] = perdida_cedida_capa

    perdida_retenida = (
        perdida_neta_antes_reaseguro
        - perdida_cedida_total
    )

    return perdida_retenida, perdida_cedida_total, detalle_capas

# =====================================================
# 11. SIMULACIÓN MONTE CARLO
# =====================================================

def simular_montecarlo(r, x0):

    if usar_modelo_caotico:

        factor_caos, lyapunov, trayectoria_caos = obtener_factor_caos(
            r,
            x0
        )

    else:

        factor_caos = 1
        lyapunov = 0
        trayectoria_caos = np.array([])

    perdidas_brutas = []
    perdidas_netas_antes_reaseguro = []
    perdidas_retenidas = []
    perdidas_cedidas = []

    cedido_por_capa = {
        capa["nombre"]: []
        for capa in capas_xol
    }

    for _ in range(n_simulaciones):

        numero_eventos = np.random.poisson(
            lambda_periodo
        )

        perdida_bruta_total = 0
        perdida_neta_total = 0

        for _ in range(numero_eventos):

            severidad_evento = np.random.lognormal(
                mean=np.log(severidad_central),
                sigma=0.35
            )

            severidad_evento *= factor_caos

            severidad_evento = np.clip(
                severidad_evento,
                0.01,
                1.00
            )

            for _, poliza in df_riesgo.iterrows():

                perdida_bruta, perdida_neta = calcular_perdida_poliza(
                    poliza,
                    severidad_evento
                )

                perdida_bruta_total += perdida_bruta
                perdida_neta_total += perdida_neta

        perdida_retenida, perdida_cedida, detalle_capas = aplicar_reaseguro_xol_multicapa(
            perdida_neta_total
        )

        perdidas_brutas.append(perdida_bruta_total)
        perdidas_netas_antes_reaseguro.append(perdida_neta_total)
        perdidas_retenidas.append(perdida_retenida)
        perdidas_cedidas.append(perdida_cedida)

        for capa in capas_xol:
            nombre = capa["nombre"]
            cedido_por_capa[nombre].append(
                detalle_capas.get(nombre, 0)
            )

    return (
        np.array(perdidas_brutas),
        np.array(perdidas_netas_antes_reaseguro),
        np.array(perdidas_retenidas),
        np.array(perdidas_cedidas),
        cedido_por_capa,
        lyapunov,
        factor_caos,
        trayectoria_caos
    )

# =====================================================
# 12. MÉTRICAS
# =====================================================

def calcular_metricas(
    perdidas_brutas,
    perdidas_netas_antes_reaseguro,
    perdidas_retenidas,
    perdidas_cedidas
):

    AAL_antes = np.mean(perdidas_netas_antes_reaseguro)
    AAL_retenido = np.mean(perdidas_retenidas)
    AAL_cedido = np.mean(perdidas_cedidas)

    VaR_antes = np.quantile(
        perdidas_netas_antes_reaseguro,
        nivel_confianza
    )

    TVaR_antes = perdidas_netas_antes_reaseguro[
        perdidas_netas_antes_reaseguro >= VaR_antes
    ].mean()

    VaR_retenido = np.quantile(
        perdidas_retenidas,
        nivel_confianza
    )

    TVaR_retenido = perdidas_retenidas[
        perdidas_retenidas >= VaR_retenido
    ].mean()

    perdida_maxima_retenida = np.max(
        perdidas_retenidas
    )

    perdida_bruta_promedio = np.mean(perdidas_brutas)
    perdida_retenida_promedio = np.mean(perdidas_retenidas)

    cobertura_promedio = (
        1 - perdida_retenida_promedio / perdida_bruta_promedio
        if perdida_bruta_promedio > 0
        else 0
    )

    indice_peor = np.argmax(perdidas_brutas)

    cobertura_peor = (
        1 - perdidas_retenidas[indice_peor] / perdidas_brutas[indice_peor]
        if perdidas_brutas[indice_peor] > 0
        else 0
    )

    reduccion_AAL = (
        1 - AAL_retenido / AAL_antes
        if AAL_antes > 0
        else 0
    )

    reduccion_TVaR = (
        1 - TVaR_retenido / TVaR_antes
        if TVaR_antes > 0
        else 0
    )

    return {
        "AAL Antes Reaseguro": AAL_antes,
        "AAL Retenido": AAL_retenido,
        "AAL Cedido": AAL_cedido,
        "VaR Antes Reaseguro": VaR_antes,
        "TVaR Antes Reaseguro": TVaR_antes,
        "VaR Retenido": VaR_retenido,
        "TVaR Retenido": TVaR_retenido,
        "PML Retenido": VaR_retenido,
        "Maxima Perdida Retenida": perdida_maxima_retenida,
        "Cobertura Promedio": cobertura_promedio,
        "Cobertura Peor Caso": cobertura_peor,
        "Reduccion AAL Reaseguro": reduccion_AAL,
        "Reduccion TVaR Reaseguro": reduccion_TVaR
    }

# =====================================================
# 13. LOOP DE CALIBRACIÓN
# =====================================================

r_actual = r_inicial
x0_actual = x0_inicial

historial_calibracion = []

for iteracion in range(
    1,
    max_iteraciones_calibracion + 1
):

    (
        perdidas_brutas,
        perdidas_netas_antes_reaseguro,
        perdidas_retenidas,
        perdidas_cedidas,
        cedido_por_capa,
        lyapunov,
        factor_caos,
        trayectoria_caos
    ) = simular_montecarlo(
        r_actual,
        x0_actual
    )

    metricas = calcular_metricas(
        perdidas_brutas,
        perdidas_netas_antes_reaseguro,
        perdidas_retenidas,
        perdidas_cedidas
    )

    historial_calibracion.append({
        "iteracion": iteracion,
        "r_caos": r_actual,
        "x0": x0_actual,
        "lyapunov": lyapunov,
        "factor_caos": factor_caos,
        "AAL_antes_reaseguro": metricas["AAL Antes Reaseguro"],
        "AAL_retenido": metricas["AAL Retenido"],
        "VaR_retenido": metricas["VaR Retenido"],
        "TVaR_retenido": metricas["TVaR Retenido"],
        "Cobertura_peor_caso": metricas["Cobertura Peor Caso"],
        "Reduccion_TVaR_reaseguro": metricas["Reduccion TVaR Reaseguro"]
    })

    condicion_estable = lyapunov <= limite_lyapunov

    condicion_cobertura = (
        metricas["Cobertura Peor Caso"]
        >= limite_cobertura_peor
    )

    if not activar_loop_calibracion:
        break

    if condicion_estable and condicion_cobertura:
        break

    if lyapunov > limite_lyapunov:
        r_actual -= 0.05

    if metricas["Cobertura Peor Caso"] < limite_cobertura_peor:
        x0_actual = max(
            0.10,
            x0_actual - 0.03
        )

    r_actual = np.clip(
        r_actual,
        3.30,
        3.95
    )

    x0_actual = np.clip(
        x0_actual,
        0.10,
        0.90
    )

df_historial_calibracion = pd.DataFrame(
    historial_calibracion
)

# =====================================================
# 14. TARIFICACIÓN
# =====================================================

AAL = metricas["AAL Retenido"]
VaR = metricas["VaR Retenido"]
TVaR = metricas["TVaR Retenido"]
PML = metricas["PML Retenido"]
perdida_maxima = metricas["Maxima Perdida Retenida"]

factor_seguridad = 0.20

prima_reaseguro = (
    sum(
        capa["limite"] * capa["costo"]
        for capa in capas_xol
    )
    if usar_reaseguro_xol
    else 0
)

prima_pura = AAL

prima_tecnica = (
    AAL
    + factor_seguridad * (TVaR - AAL)
)

prima_sugerida = (
    prima_tecnica
    + prima_reaseguro
)

prima_actual_estimativa = (
    df_riesgo["suma_asegurada"].sum()
    * 0.015
)

prima_minima_comercial = AAL * 1.05
prima_maxima_comercial = AAL * 2.00 + prima_reaseguro

prima_sugerida_comercial = min(
    max(
        prima_sugerida,
        prima_minima_comercial
    ),
    prima_maxima_comercial
)

diferencia_prima = (
    prima_sugerida_comercial
    - prima_actual_estimativa
)

recalcular_prima = (
    "Sí"
    if prima_sugerida_comercial > prima_actual_estimativa
    else "No"
)

# =====================================================
# 15. CLASIFICACIÓN DE RIESGO
# =====================================================

df_clasificacion = df_riesgo.copy()

df_clasificacion["factor_vulnerabilidad"] = df_clasificacion.apply(
    lambda x:
    factor_construccion(x["tipo_construccion"])
    * factor_antiguedad(x["anio_construccion"])
    * factor_ocupacion(x["ocupacion"]),
    axis=1
)

df_clasificacion["exposicion_ajustada"] = (
    df_clasificacion["valor_expuesto"]
    * df_clasificacion["factor_vulnerabilidad"]
)

umbral = df_clasificacion[
    "exposicion_ajustada"
].quantile(0.70)

df_clasificacion["clasificacion_riesgo"] = np.where(
    df_clasificacion["exposicion_ajustada"] >= umbral,
    "Riesgo severo",
    "Riesgo moderado"
)

# =====================================================
# 16. RESULTADOS EN CONSOLA
# =====================================================

print("=" * 90)
print("FASE 2 + FASE 3 + REASEGURO XoL MULTICAPA AJUSTADO")
print("Monte Carlo histórico + Modelo Caótico + Lyapunov + Reaseguro Multicapa")
print("=" * 90)

print(f"Tipo de riesgo: {tipo_riesgo}")
print(f"Escenario: {escenario}")
print(f"Periodo: {periodo_anios} año(s)")
print(f"Simulaciones: {n_simulaciones:,}")

print("\nCALIBRACIÓN HISTÓRICA")
print(f"Años observados: {anio_min}-{anio_max}")
print(f"Eventos observados: {eventos_observados}")
print(f"Lambda histórica anual: {lambda_historica:.4f}")
print(f"Lambda usada en periodo: {lambda_periodo:.4f}")
print(f"Variable severidad: {col_severidad}")
print(f"Severidad central: {severidad_central:.4f}")

print("\nMODELO CAÓTICO")
print(f"r final: {r_actual:.4f}")
print(f"x0 final: {x0_actual:.4f}")
print(f"Exponente de Lyapunov: {lyapunov:.4f}")
print(f"Factor caótico aplicado: {factor_caos:.4f}")

print("\nREASEGURO XoL MULTICAPA AJUSTADO")
print(f"Usar reaseguro XoL: {usar_reaseguro_xol}")

for capa in capas_xol:
    print(
        f"{capa['nombre']}: "
        f"Retención ${capa['retencion']:,.2f} | "
        f"Límite ${capa['limite']:,.2f} | "
        f"Costo {capa['costo']:.2%}"
    )

print(f"Prima estimada de reaseguro: ${prima_reaseguro:,.2f} MXN")
print(f"AAL antes de reaseguro: ${metricas['AAL Antes Reaseguro']:,.2f} MXN")
print(f"AAL retenido: ${metricas['AAL Retenido']:,.2f} MXN")
print(f"AAL cedido: ${metricas['AAL Cedido']:,.2f} MXN")
print(f"Reducción AAL por reaseguro: {metricas['Reduccion AAL Reaseguro']:.2%}")
print(f"Reducción TVaR por reaseguro: {metricas['Reduccion TVaR Reaseguro']:.2%}")

print("\nMÉTRICAS DE RIESGO RETENIDO")
print(f"AAL retenido: ${AAL:,.2f} MXN")
print(f"VaR retenido {nivel_confianza:.0%}: ${VaR:,.2f} MXN")
print(f"TVaR retenido {nivel_confianza:.0%}: ${TVaR:,.2f} MXN")
print(f"PML retenido: ${PML:,.2f} MXN")
print(f"Máxima pérdida retenida: ${perdida_maxima:,.2f} MXN")

print("\nPRIMAS")
print(f"Prima actual estimada: ${prima_actual_estimativa:,.2f} MXN")
print(f"Prima pura retenida: ${prima_pura:,.2f} MXN")
print(f"Prima técnica retenida: ${prima_tecnica:,.2f} MXN")
print(f"Prima reaseguro: ${prima_reaseguro:,.2f} MXN")
print(f"Prima sugerida comercial: ${prima_sugerida_comercial:,.2f} MXN")
print(f"Diferencia de prima: ${diferencia_prima:,.2f} MXN")
print(f"¿Recalcular prima?: {recalcular_prima}")

print("\nCOBERTURA")
print(f"Cobertura promedio: {metricas['Cobertura Promedio']:.2%}")
print(f"Cobertura peor caso: {metricas['Cobertura Peor Caso']:.2%}")

# =====================================================
# 17. EXPORTAR RESULTADOS A EXCEL
# =====================================================

df_simulaciones = pd.DataFrame({
    "simulacion": range(1, n_simulaciones + 1),
    "perdida_bruta": perdidas_brutas,
    "perdida_neta_antes_reaseguro": perdidas_netas_antes_reaseguro,
    "perdida_retenida_aseguradora": perdidas_retenidas,
    "perdida_cedida_reasegurador": perdidas_cedidas
})

for capa in capas_xol:
    nombre = capa["nombre"]
    df_simulaciones[f"cedido_{nombre}"] = cedido_por_capa[nombre]

df_metricas = pd.DataFrame({
    "metrica": [
        "Tipo de riesgo",
        "Escenario",
        "Periodo",
        "Simulaciones",
        "Años observados",
        "Eventos observados",
        "Lambda histórica anual",
        "Lambda usada en periodo",
        "Variable severidad histórica",
        "Severidad central",
        "r final",
        "x0 final",
        "Exponente Lyapunov",
        "Factor caótico",
        "Usar reaseguro XoL",
        "Prima reaseguro",
        "AAL antes reaseguro",
        "AAL retenido",
        "AAL cedido",
        f"VaR retenido {nivel_confianza:.0%}",
        f"TVaR retenido {nivel_confianza:.0%}",
        "PML retenido",
        "Máxima pérdida retenida",
        "Reducción AAL reaseguro",
        "Reducción TVaR reaseguro",
        "Prima actual estimada",
        "Prima pura retenida",
        "Prima técnica retenida",
        "Prima sugerida comercial",
        "Diferencia de prima",
        "Recalcular prima",
        "Cobertura promedio",
        "Cobertura peor caso"
    ],
    "valor": [
        tipo_riesgo,
        escenario,
        periodo_anios,
        n_simulaciones,
        f"{anio_min}-{anio_max}",
        eventos_observados,
        lambda_historica,
        lambda_periodo,
        col_severidad,
        severidad_central,
        r_actual,
        x0_actual,
        lyapunov,
        factor_caos,
        usar_reaseguro_xol,
        prima_reaseguro,
        metricas["AAL Antes Reaseguro"],
        metricas["AAL Retenido"],
        metricas["AAL Cedido"],
        VaR,
        TVaR,
        PML,
        perdida_maxima,
        metricas["Reduccion AAL Reaseguro"],
        metricas["Reduccion TVaR Reaseguro"],
        prima_actual_estimativa,
        prima_pura,
        prima_tecnica,
        prima_sugerida_comercial,
        diferencia_prima,
        recalcular_prima,
        metricas["Cobertura Promedio"],
        metricas["Cobertura Peor Caso"]
    ]
})

df_capas = pd.DataFrame(capas_xol)

df_trayectoria_caos = pd.DataFrame({
    "iteracion": range(1, len(trayectoria_caos) + 1),
    "x_t": trayectoria_caos
})

archivo_salida = f"fase2_3_xol_multicapa_ajustado_{tipo_riesgo}_{escenario}.xlsx"

with pd.ExcelWriter(
    archivo_salida,
    engine="openpyxl"
) as writer:

    df_simulaciones.to_excel(
        writer,
        sheet_name="Simulaciones",
        index=False
    )

    df_metricas.to_excel(
        writer,
        sheet_name="Metricas",
        index=False
    )

    df_capas.to_excel(
        writer,
        sheet_name="Capas_XoL",
        index=False
    )

    df_clasificacion.to_excel(
        writer,
        sheet_name="Clasificacion",
        index=False
    )

    df_historial_calibracion.to_excel(
        writer,
        sheet_name="Loop_Calibracion",
        index=False
    )

    df_trayectoria_caos.to_excel(
        writer,
        sheet_name="Trayectoria_Caos",
        index=False
    )

    df_eventos.to_excel(
        writer,
        sheet_name="Eventos_Historicos",
        index=False
    )

print("\nArchivo generado:")
print(archivo_salida)
