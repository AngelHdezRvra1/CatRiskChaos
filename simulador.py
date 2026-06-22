
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime


# =========================================================
# VALIDACIÓN Y NORMALIZACIÓN DE PORTAFOLIO
# =========================================================

COLUMNAS_NECESARIAS = [
    "tipo_riesgo",
    "tipo_construccion",
    "anio_construccion",
    "ocupacion",
    "valor_expuesto",
    "deducible",
    "limite_cobertura",
]

MAPEO_TIPO_RIESGO = {
    "terremoto": "Terremoto",
    "earthquake": "Terremoto",
    "inundacion": "Inundacion",
    "inundación": "Inundacion",
    "flood": "Inundacion",
    "tormenta": "Tormenta",
    "storm": "Tormenta",
}

MAPEO_CONSTRUCCION = {
    "concreto": "Concreto",
    "acero": "Acero",
    "mamposteria": "Mamposteria",
    "mampostería": "Mamposteria",
    "mixta": "Mixta",
}

MAPEO_OCUPACION = {
    "residencial": "Residencial",
    "comercial": "Comercial",
    "industrial": "Industrial",
}


def _normalizar_texto(valor):
    if pd.isna(valor):
        return ""
    return str(valor).strip().lower()


def validar_portafolio(df_portafolio):
    """
    Valida y normaliza el portafolio antes de ejecutar el simulador.

    Retorna:
    {
        "ok": bool,
        "errores": list[str],
        "advertencias": list[str],
        "df_limpio": DataFrame
    }
    """
    errores = []
    advertencias = []

    if df_portafolio is None or df_portafolio.empty:
        return {
            "ok": False,
            "errores": ["El archivo está vacío o no pudo leerse."],
            "advertencias": [],
            "df_limpio": pd.DataFrame(),
        }

    df = df_portafolio.copy()
    df.columns = [str(c).strip() for c in df.columns]

    faltantes = [col for col in COLUMNAS_NECESARIAS if col not in df.columns]
    if faltantes:
        errores.append(f"Faltan columnas requeridas: {faltantes}.")
        return {
            "ok": False,
            "errores": errores,
            "advertencias": advertencias,
            "df_limpio": df,
        }

    # Normalización de categorías
    df["tipo_riesgo"] = df["tipo_riesgo"].apply(
        lambda x: MAPEO_TIPO_RIESGO.get(_normalizar_texto(x), str(x).strip())
    )
    df["tipo_construccion"] = df["tipo_construccion"].apply(
        lambda x: MAPEO_CONSTRUCCION.get(_normalizar_texto(x), str(x).strip())
    )
    df["ocupacion"] = df["ocupacion"].apply(
        lambda x: MAPEO_OCUPACION.get(_normalizar_texto(x), str(x).strip())
    )

    riesgos_validos = set(MAPEO_TIPO_RIESGO.values())
    construcciones_validas = set(MAPEO_CONSTRUCCION.values())
    ocupaciones_validas = set(MAPEO_OCUPACION.values())

    riesgos_invalidos = sorted(set(df["tipo_riesgo"].dropna()) - riesgos_validos)
    construcciones_invalidas = sorted(set(df["tipo_construccion"].dropna()) - construcciones_validas)
    ocupaciones_invalidas = sorted(set(df["ocupacion"].dropna()) - ocupaciones_validas)

    if riesgos_invalidos:
        errores.append(f"Valores no válidos en tipo_riesgo: {riesgos_invalidos}. Usa Terremoto, Inundacion o Tormenta.")

    if construcciones_invalidas:
        errores.append(f"Valores no válidos en tipo_construccion: {construcciones_invalidas}. Usa Concreto, Acero, Mamposteria o Mixta.")

    if ocupaciones_invalidas:
        errores.append(f"Valores no válidos en ocupacion: {ocupaciones_invalidas}. Usa Residencial, Comercial o Industrial.")

    # Conversión numérica
    columnas_numericas = [
        "anio_construccion",
        "valor_expuesto",
        "deducible",
        "limite_cobertura",
    ]

    for col in columnas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    nulos = df[COLUMNAS_NECESARIAS].isna().sum()
    nulos = nulos[nulos > 0]
    if len(nulos) > 0:
        errores.append(f"Hay valores vacíos o no numéricos en columnas requeridas: {nulos.to_dict()}.")

    anio_actual = datetime.now().year

    if (df["anio_construccion"] < 1900).any() or (df["anio_construccion"] > anio_actual).any():
        errores.append(f"anio_construccion debe estar entre 1900 y {anio_actual}.")

    if (df["valor_expuesto"] <= 0).any():
        errores.append("valor_expuesto debe ser mayor que 0 en todas las pólizas.")

    if (df["deducible"] < 0).any():
        errores.append("deducible debe ser mayor o igual que 0 en todas las pólizas.")

    if (df["limite_cobertura"] <= 0).any():
        errores.append("limite_cobertura debe ser mayor que 0 en todas las pólizas.")

    if (df["deducible"] > df["limite_cobertura"]).any():
        advertencias.append("Hay pólizas donde el deducible es mayor que el límite de cobertura. Revisa si esto es intencional.")

    if (df["limite_cobertura"] > df["valor_expuesto"]).any():
        advertencias.append("Hay pólizas donde el límite de cobertura es mayor que el valor expuesto. El simulador puede continuar, pero conviene revisarlo.")

    if len(df) < 10:
        advertencias.append("El portafolio tiene menos de 10 pólizas. Los resultados pueden no ser representativos.")

    ok = len(errores) == 0

    return {
        "ok": ok,
        "errores": errores,
        "advertencias": advertencias,
        "df_limpio": df,
    }


# =========================================================
# SIMULADOR PRINCIPAL
# =========================================================

def ejecutar_simulacion(
    df_portafolio,
    tipo_riesgo,
    escenario,
    periodo_anios,
    n_simulaciones,
    nivel_confianza,
    archivo_historico="data/Mexico_Datos.xlsx",
    seed=42
):
    validacion = validar_portafolio(df_portafolio)

    if not validacion["ok"]:
        raise ValueError("Portafolio inválido: " + " | ".join(validacion["errores"]))

    df_portafolio = validacion["df_limpio"].copy()

    # Normalizar entradas de usuario
    tipo_riesgo = MAPEO_TIPO_RIESGO.get(_normalizar_texto(tipo_riesgo), tipo_riesgo)
    escenario = str(escenario).strip()

    rng = np.random.default_rng(seed)

    usar_frecuencia_historica = True
    usar_severidad_historica = True
    lambda_manual = 0.25

    usar_modelo_caotico = True
    r_inicial = 3.75
    x0_inicial = 0.35
    iteraciones_caos = 100
    activar_loop_calibracion = True
    max_iteraciones_calibracion = 10
    limite_lyapunov = 0.20
    limite_cobertura_peor = 0.55

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

    mapeo_eventos = {
        "Terremoto": "Earthquake",
        "Inundacion": "Flood",
        "Tormenta": "Storm"
    }

    percentiles_escenario = {
        "Moderado": 0.50,
        "Severo": 0.75,
        "Extremo": 0.90,
        "Catastrofico": 0.95,
        "Catastrófico": 0.95,
    }

    if tipo_riesgo not in mapeo_eventos:
        raise ValueError("Tipo de riesgo no reconocido. Usa Terremoto, Inundacion o Tormenta.")

    if escenario not in percentiles_escenario:
        raise ValueError("Escenario no reconocido. Usa Moderado, Severo, Extremo o Catastrofico.")

    if periodo_anios < 1:
        raise ValueError("El periodo de aseguramiento debe ser mayor o igual que 1.")

    df_hist = pd.read_excel(
        archivo_historico,
        sheet_name="EM-DAT Data"
    )

    df_riesgo = df_portafolio[
        df_portafolio["tipo_riesgo"] == tipo_riesgo
    ].copy()

    if df_riesgo.empty:
        raise ValueError(
            f"No hay pólizas para el tipo de riesgo seleccionado: {tipo_riesgo}."
        )

    tipo_emdat = mapeo_eventos[tipo_riesgo]

    if "Disaster Type" not in df_hist.columns:
        raise ValueError("El archivo histórico no contiene la columna 'Disaster Type'.")

    df_eventos = df_hist[
        df_hist["Disaster Type"] == tipo_emdat
    ].copy()

    if df_eventos.empty:
        raise ValueError(
            f"No hay eventos históricos para el riesgo {tipo_riesgo} ({tipo_emdat})."
        )

    if "Start Year" not in df_eventos.columns:
        raise ValueError("El archivo histórico no contiene la columna 'Start Year'.")

    anio_min = int(df_eventos["Start Year"].min())
    anio_max = int(df_eventos["Start Year"].max())

    anios_observados = anio_max - anio_min + 1
    eventos_observados = len(df_eventos)

    lambda_historica = eventos_observados / anios_observados

    lambda_anual = (
        lambda_historica
        if usar_frecuencia_historica
        else lambda_manual
    )

    # Esta línea garantiza que el periodo de aseguramiento afecte la frecuencia.
    lambda_periodo = lambda_anual * periodo_anios

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

        raise ValueError(
            "No hay suficientes datos históricos de severidad para este riesgo."
        )

    col_severidad, severidad_historica = obtener_variable_severidad(
        df_eventos
    )

    percentil = percentiles_escenario[escenario]

    severidad_base = severidad_historica.quantile(percentil)
    severidad_min = severidad_historica.min()
    severidad_max = severidad_historica.max()

    if severidad_max == severidad_min:
        factor_severidad_historico = 0.20
    else:
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
        "Catastrofico": 0.80,
        "Catastrófico": 0.80,
    }

    severidad_central = (
        factor_severidad_historico
        if usar_severidad_historica
        else factores_manual[escenario]
    )

    def generar_trayectoria_caotica(r, x0, n):
        x = float(x0)
        trayectoria = []

        for _ in range(n):
            x = r * x * (1 - x)
            x = np.clip(x, 1e-10, 1 - 1e-10)
            trayectoria.append(x)

        return np.array(trayectoria)

    def calcular_lyapunov_logistico(r, trayectoria):
        derivadas = np.abs(
            r * (1 - 2 * trayectoria)
        )

        derivadas = np.where(
            derivadas <= 1e-12,
            1e-12,
            derivadas
        )

        return float(np.mean(np.log(derivadas)))

    def obtener_factor_caos(r, x0):
        trayectoria = generar_trayectoria_caotica(
            r,
            x0,
            max(iteraciones_caos, n_simulaciones)
        )

        lyapunov = calcular_lyapunov_logistico(
            r,
            trayectoria[:iteraciones_caos]
        )

        factor_caos_medio = 0.75 + 0.50 * np.mean(
            trayectoria[:iteraciones_caos]
        )

        return factor_caos_medio, lyapunov, trayectoria

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

    def simular_montecarlo(r, x0):
        if usar_modelo_caotico:
            factor_caos_medio, lyapunov, trayectoria_caos = obtener_factor_caos(
                r,
                x0
            )
        else:
            factor_caos_medio = 1.0
            lyapunov = 0.0
            trayectoria_caos = np.ones(n_simulaciones)

        perdidas_brutas = []
        perdidas_netas_antes_reaseguro = []
        perdidas_retenidas = []
        perdidas_cedidas = []

        cedido_por_capa = {
            capa["nombre"]: []
            for capa in capas_xol
        }

        polizas_dict = df_riesgo.to_dict(orient="records")

        for i in range(n_simulaciones):
            # La frecuencia ya incluye el periodo de aseguramiento.
            numero_eventos = rng.poisson(lambda_periodo)

            perdida_bruta_periodo = 0
            perdida_neta_periodo = 0

            idx_caos = i % len(trayectoria_caos)

            factor_caos_dinamico = (
                0.75 + 0.50 * trayectoria_caos[idx_caos]
                if usar_modelo_caotico
                else 1.0
            )

            for _ in range(numero_eventos):
                severidad_evento = rng.lognormal(
                    mean=np.log(max(severidad_central, 1e-6)),
                    sigma=0.35
                )

                severidad_evento *= factor_caos_dinamico

                severidad_evento = np.clip(
                    severidad_evento,
                    0.01,
                    1.00
                )

                for poliza in polizas_dict:
                    perdida_bruta, perdida_neta = calcular_perdida_poliza(
                        poliza,
                        severidad_evento
                    )

                    perdida_bruta_periodo += perdida_bruta
                    perdida_neta_periodo += perdida_neta

            perdida_retenida, perdida_cedida, detalle_capas = aplicar_reaseguro_xol_multicapa(
                perdida_neta_periodo
            )

            perdidas_brutas.append(perdida_bruta_periodo)
            perdidas_netas_antes_reaseguro.append(perdida_neta_periodo)
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
            factor_caos_medio,
            trayectoria_caos[:iteraciones_caos]
        )

    def calcular_metricas(
        perdidas_brutas,
        perdidas_netas_antes_reaseguro,
        perdidas_retenidas,
        perdidas_cedidas
    ):
        perdida_esperada_antes_periodo = np.mean(perdidas_netas_antes_reaseguro)
        perdida_esperada_retenida_periodo = np.mean(perdidas_retenidas)
        perdida_esperada_cedida_periodo = np.mean(perdidas_cedidas)

        aal_antes_anual = perdida_esperada_antes_periodo / periodo_anios
        aal_retenido_anual = perdida_esperada_retenida_periodo / periodo_anios
        aal_cedido_anual = perdida_esperada_cedida_periodo / periodo_anios

        VaR_antes = np.quantile(
            perdidas_netas_antes_reaseguro,
            nivel_confianza
        )

        elementos_tvar_antes = perdidas_netas_antes_reaseguro[
            perdidas_netas_antes_reaseguro >= VaR_antes
        ]

        TVaR_antes = (
            elementos_tvar_antes.mean()
            if len(elementos_tvar_antes) > 0
            else VaR_antes
        )

        VaR_retenido = np.quantile(
            perdidas_retenidas,
            nivel_confianza
        )

        elementos_tvar_ret = perdidas_retenidas[
            perdidas_retenidas >= VaR_retenido
        ]

        TVaR_retenido = (
            elementos_tvar_ret.mean()
            if len(elementos_tvar_ret) > 0
            else VaR_retenido
        )

        perdida_maxima_retenida = np.max(
            perdidas_retenidas
        )

        perdida_bruta_promedio = np.mean(
            perdidas_brutas
        )

        cobertura_promedio = (
            1 - perdida_esperada_retenida_periodo / perdida_bruta_promedio
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
            1 - perdida_esperada_retenida_periodo / perdida_esperada_antes_periodo
            if perdida_esperada_antes_periodo > 0
            else 0
        )

        reduccion_TVaR = (
            1 - TVaR_retenido / TVaR_antes
            if TVaR_antes > 0
            else 0
        )

        return {
            "Perdida Esperada Antes Reaseguro Periodo": perdida_esperada_antes_periodo,
            "Perdida Esperada Retenida Periodo": perdida_esperada_retenida_periodo,
            "Perdida Esperada Cedida Periodo": perdida_esperada_cedida_periodo,
            "AAL Antes Reaseguro Anual": aal_antes_anual,
            "AAL Retenido Anual": aal_retenido_anual,
            "AAL Cedido Anual": aal_cedido_anual,
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

        condicion_estable = (
            lyapunov <= limite_lyapunov
        )

        condicion_cobertura = (
            metricas["Cobertura Peor Caso"]
            >= limite_cobertura_peor
        )

        historial_calibracion.append({
            "iteracion": iteracion,
            "r_caos": r_actual,
            "x0": x0_actual,
            "lyapunov": lyapunov,
            "limite_lyapunov": limite_lyapunov,
            "modelo_estable": condicion_estable,
            "factor_caos": factor_caos,
            "perdida_esperada_retenida_periodo": metricas["Perdida Esperada Retenida Periodo"],
            "AAL_retenido_anual": metricas["AAL Retenido Anual"],
            "VaR_retenido": metricas["VaR Retenido"],
            "TVaR_retenido": metricas["TVaR Retenido"],
            "Cobertura_peor_caso": metricas["Cobertura Peor Caso"],
            "Reduccion_TVaR_reaseguro": metricas["Reduccion TVaR Reaseguro"]
        })

        if not activar_loop_calibracion:
            break

        if condicion_estable and condicion_cobertura:
            break

        # Ajuste no lineal:
        # Si Lyapunov es demasiado alto, se reduce r para disminuir sensibilidad a condiciones iniciales.
        if lyapunov > limite_lyapunov:
            r_actual -= 0.05

        # Si la cobertura del peor caso es baja, se modifica x0 para suavizar el factor dinámico.
        if metricas["Cobertura Peor Caso"] < limite_cobertura_peor:
            x0_actual = max(
                0.10,
                x0_actual - 0.03
            )

        r_actual = float(np.clip(
            r_actual,
            3.30,
            3.95
        ))

        x0_actual = float(np.clip(
            x0_actual,
            0.10,
            0.90
        ))

    df_historial_calibracion = pd.DataFrame(
        historial_calibracion
    )

    perdida_esperada_periodo = metricas["Perdida Esperada Retenida Periodo"]
    aal_anual = metricas["AAL Retenido Anual"]
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

    prima_pura_periodo = perdida_esperada_periodo

    prima_tecnica_periodo = (
        perdida_esperada_periodo
        + factor_seguridad * (TVaR - perdida_esperada_periodo)
    )

    prima_sugerida_periodo = (
        prima_tecnica_periodo
        + prima_reaseguro
    )

    col_suma = (
        "limite_cobertura"
        if "limite_cobertura" in df_riesgo.columns
        else "valor_expuesto"
    )

    prima_actual_estimativa = (
        df_riesgo[col_suma].sum()
        * 0.015
        * periodo_anios
    )

    prima_minima_comercial = perdida_esperada_periodo * 1.05
    prima_maxima_comercial = perdida_esperada_periodo * 2.00 + prima_reaseguro

    prima_sugerida_comercial = min(
        max(
            prima_sugerida_periodo,
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

    df_simulaciones["sin_perdida_retenida"] = (
        df_simulaciones["perdida_retenida_aseguradora"] == 0
    )

    df_simulaciones["cerca_retencion_100m"] = np.isclose(
        df_simulaciones["perdida_retenida_aseguradora"],
        100_000_000,
        atol=1
    )

    df_metricas = pd.DataFrame({
        "metrica": [
            "Tipo de riesgo",
            "Escenario",
            "Periodo de aseguramiento",
            "Simulaciones",
            "Nivel de confianza",
            "Años observados",
            "Eventos observados",
            "Lambda histórica anual",
            "Lambda usada en periodo",
            "Variable severidad histórica",
            "Severidad central",
            "r final",
            "x0 final",
            "Exponente Lyapunov",
            "Límite Lyapunov",
            "Modelo caótico estable",
            "Factor caótico",
            "Usar reaseguro XoL",
            "Prima reaseguro",
            "Pérdida esperada antes reaseguro periodo",
            "Pérdida esperada retenida periodo",
            "Pérdida esperada cedida periodo",
            "AAL antes reaseguro anualizado",
            "AAL retenido anualizado",
            "AAL cedido anualizado",
            f"VaR retenido {nivel_confianza:.0%}",
            f"TVaR retenido {nivel_confianza:.0%}",
            "PML retenido",
            "Máxima pérdida retenida",
            "Reducción AAL reaseguro",
            "Reducción TVaR reaseguro",
            "Prima actual estimada periodo",
            "Prima pura periodo",
            "Prima técnica periodo",
            "Prima sugerida comercial periodo",
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
            nivel_confianza,
            f"{anio_min}-{anio_max}",
            eventos_observados,
            lambda_historica,
            lambda_periodo,
            col_severidad,
            severidad_central,
            r_actual,
            x0_actual,
            lyapunov,
            limite_lyapunov,
            lyapunov <= limite_lyapunov,
            factor_caos,
            usar_reaseguro_xol,
            prima_reaseguro,
            metricas["Perdida Esperada Antes Reaseguro Periodo"],
            metricas["Perdida Esperada Retenida Periodo"],
            metricas["Perdida Esperada Cedida Periodo"],
            metricas["AAL Antes Reaseguro Anual"],
            metricas["AAL Retenido Anual"],
            metricas["AAL Cedido Anual"],
            VaR,
            TVaR,
            PML,
            perdida_maxima,
            metricas["Reduccion AAL Reaseguro"],
            metricas["Reduccion TVaR Reaseguro"],
            prima_actual_estimativa,
            prima_pura_periodo,
            prima_tecnica_periodo,
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

    total_sim = len(df_simulaciones)
    escenarios_cero = int((df_simulaciones["perdida_retenida_aseguradora"] == 0).sum())
    escenarios_retencion_100m = int(df_simulaciones["cerca_retencion_100m"].sum())
    escenarios_positivos = int((df_simulaciones["perdida_retenida_aseguradora"] > 0).sum())

    df_diagnostico_histograma = pd.DataFrame({
        "indicador": [
            "Simulaciones totales",
            "Escenarios con pérdida retenida igual a 0",
            "Escenarios con pérdida retenida positiva",
            "Escenarios cerca de retención 100M",
            "Porcentaje en cero",
            "Porcentaje cerca de retención 100M",
            "Pérdida retenida promedio",
            "Pérdida retenida máxima",
            "Pérdida bruta promedio",
            "Pérdida bruta máxima"
        ],
        "valor": [
            total_sim,
            escenarios_cero,
            escenarios_positivos,
            escenarios_retencion_100m,
            escenarios_cero / total_sim if total_sim > 0 else 0,
            escenarios_retencion_100m / total_sim if total_sim > 0 else 0,
            df_simulaciones["perdida_retenida_aseguradora"].mean(),
            df_simulaciones["perdida_retenida_aseguradora"].max(),
            df_simulaciones["perdida_bruta"].mean(),
            df_simulaciones["perdida_bruta"].max()
        ]
    })

    output = BytesIO()

    with pd.ExcelWriter(
        output,
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

        df_diagnostico_histograma.to_excel(
            writer,
            sheet_name="Diagnostico_Histograma",
            index=False
        )

        df_eventos.to_excel(
            writer,
            sheet_name="Eventos_Historicos",
            index=False
        )

    output.seek(0)

    return {
        "Perdida Esperada Periodo": perdida_esperada_periodo,
        "AAL": perdida_esperada_periodo,
        "AAL Anual": aal_anual,
        "VaR": VaR,
        "TVaR": TVaR,
        "PML": PML,
        "Maxima Perdida": perdida_maxima,
        "Prima": prima_sugerida_comercial,
        "Prima Actual": prima_actual_estimativa,
        "Prima Reaseguro": prima_reaseguro,
        "Diferencia Prima": diferencia_prima,
        "Recalcular Prima": recalcular_prima,
        "Lyapunov": lyapunov,
        "Factor Caos": factor_caos,
        "Cobertura Promedio": metricas["Cobertura Promedio"],
        "Cobertura Peor Caso": metricas["Cobertura Peor Caso"],
        "AAL Antes Reaseguro": metricas["Perdida Esperada Antes Reaseguro Periodo"],
        "AAL Cedido": metricas["Perdida Esperada Cedida Periodo"],
        "Reduccion AAL Reaseguro": metricas["Reduccion AAL Reaseguro"],
        "Reduccion TVaR Reaseguro": metricas["Reduccion TVaR Reaseguro"],
        "df_simulaciones": df_simulaciones,
        "df_metricas": df_metricas,
        "df_clasificacion": df_clasificacion,
        "df_trayectoria_caos": df_trayectoria_caos,
        "df_historial_calibracion": df_historial_calibracion,
        "df_capas": df_capas,
        "df_diagnostico_histograma": df_diagnostico_histograma,
        "excel_bytes": output
    }
