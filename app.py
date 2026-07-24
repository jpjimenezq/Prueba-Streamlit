"""
App de Streamlit: Cobranza de Usuarios
----------------------------------------
1. Genera datos sinteticos de cobranza.
2. Realiza EDA (analisis cuantitativo, cualitativo y visual).
3. Permite interaccion del usuario (filtros, parametros, descarga de datos).

Para ejecutar:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# Configuracion general de la pagina
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Cobranza de Usuarios - EDA",
    page_icon="💰",
    layout="wide",
)

# ------------------------------------------------------------------
# Generacion de datos sinteticos
# ------------------------------------------------------------------
REGIONES = ["Bogota", "Medellin", "Cali", "Barranquilla", "Bucaramanga", "Pereira"]
SEGMENTOS = ["Personal", "Pyme", "Empresarial"]
CANALES = ["Llamada", "WhatsApp", "Email", "SMS", "Visita"]
ESTADOS = ["Al dia", "Mora temprana", "Mora media", "Mora avanzada", "Castigada"]
PRODUCTOS = ["Tarjeta de credito", "Credito de libre inversion", "Credito hipotecario", "Credito vehicular"]


@st.cache_data(show_spinner=False)
def generar_datos_sinteticos(n_usuarios: int, semilla: int) -> pd.DataFrame:
    """Genera un dataset sintetico de cobranza de usuarios."""
    rng = np.random.default_rng(semilla)

    usuario_id = np.arange(1, n_usuarios + 1)
    edad = rng.integers(18, 75, n_usuarios)
    region = rng.choice(REGIONES, n_usuarios, p=[0.28, 0.20, 0.15, 0.13, 0.12, 0.12])
    segmento = rng.choice(SEGMENTOS, n_usuarios, p=[0.6, 0.3, 0.1])
    producto = rng.choice(PRODUCTOS, n_usuarios)
    canal_contacto = rng.choice(CANALES, n_usuarios)

    # Monto de deuda: distribucion log-normal para simular sesgo real
    monto_deuda = np.round(rng.lognormal(mean=13.5, sigma=0.9, size=n_usuarios), -3)
    monto_deuda = np.clip(monto_deuda, 50_000, 60_000_000)

    # Dias de mora correlacionados con estado de cobranza
    dias_mora = rng.integers(0, 400, n_usuarios)

    def clasificar_estado(d):
        if d == 0:
            return "Al dia"
        elif d <= 30:
            return "Mora temprana"
        elif d <= 90:
            return "Mora media"
        elif d <= 180:
            return "Mora avanzada"
        else:
            return "Castigada"

    estado_cobranza = np.array([clasificar_estado(d) for d in dias_mora])

    # Score de riesgo (0-1000), inversamente relacionado con dias de mora + ruido
    ruido = rng.normal(0, 60, n_usuarios)
    score_riesgo = np.clip(850 - dias_mora * 1.4 + ruido, 100, 950).round(0)

    # Porcentaje pagado de la deuda (mas mora -> menos pago)
    prob_pago_base = np.clip(1 - dias_mora / 400 + rng.normal(0, 0.1, n_usuarios), 0, 1)
    monto_pagado = np.round(monto_deuda * prob_pago_base, -2)

    # Numero de contactos realizados y efectividad
    num_contactos = rng.poisson(lam=3 + dias_mora / 100, size=n_usuarios)
    contacto_efectivo = rng.random(n_usuarios) < np.clip(0.6 - dias_mora / 600, 0.05, 0.6)

    # Fechas de vencimiento en el ultimo anio
    fecha_vencimiento = pd.to_datetime("2026-07-24") - pd.to_timedelta(
        rng.integers(0, 365, n_usuarios), unit="D"
    )

    promesa_pago = rng.choice([True, False], n_usuarios, p=[0.35, 0.65])

    df = pd.DataFrame(
        {
            "usuario_id": usuario_id,
            "edad": edad,
            "region": region,
            "segmento": segmento,
            "producto": producto,
            "canal_contacto": canal_contacto,
            "monto_deuda": monto_deuda,
            "monto_pagado": monto_pagado,
            "dias_mora": dias_mora,
            "estado_cobranza": estado_cobranza,
            "score_riesgo": score_riesgo,
            "num_contactos": num_contactos,
            "contacto_efectivo": contacto_efectivo,
            "promesa_pago": promesa_pago,
            "fecha_vencimiento": fecha_vencimiento,
        }
    )

    df["pct_pagado"] = np.round((df["monto_pagado"] / df["monto_deuda"]) * 100, 1)
    df["saldo_pendiente"] = df["monto_deuda"] - df["monto_pagado"]

    return df


# ------------------------------------------------------------------
# Sidebar: parametros de generacion + filtros interactivos
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Parametros de datos")
n_usuarios = st.sidebar.slider("Numero de usuarios a simular", 100, 20_000, 3_000, step=100)
semilla = st.sidebar.number_input("Semilla aleatoria", min_value=0, max_value=9999, value=42)

df = generar_datos_sinteticos(n_usuarios, semilla)

st.sidebar.header("🔎 Filtros interactivos")
regiones_sel = st.sidebar.multiselect("Region", options=REGIONES, default=REGIONES)
segmentos_sel = st.sidebar.multiselect("Segmento", options=SEGMENTOS, default=SEGMENTOS)
estados_sel = st.sidebar.multiselect("Estado de cobranza", options=ESTADOS, default=ESTADOS)
rango_mora = st.sidebar.slider(
    "Rango de dias de mora", int(df["dias_mora"].min()), int(df["dias_mora"].max()), (0, 400)
)

df_filtrado = df[
    df["region"].isin(regiones_sel)
    & df["segmento"].isin(segmentos_sel)
    & df["estado_cobranza"].isin(estados_sel)
    & df["dias_mora"].between(rango_mora[0], rango_mora[1])
]

st.sidebar.markdown(f"**Registros filtrados:** {len(df_filtrado):,} / {len(df):,}")

st.sidebar.download_button(
    label="⬇️ Descargar datos filtrados (CSV)",
    data=df_filtrado.to_csv(index=False).encode("utf-8"),
    file_name="cobranza_sintetica.csv",
    mime="text/csv",
)

# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.title("💰 Panel de Cobranza de Usuarios — Datos Sinteticos")
st.caption(
    "Datos generados de forma sintetica con fines demostrativos. "
    "Usa el panel lateral para ajustar la simulacion y los filtros."
)

# ------------------------------------------------------------------
# KPIs principales
# ------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Usuarios", f"{len(df_filtrado):,}")
col2.metric("Deuda total", f"${df_filtrado['monto_deuda'].sum():,.0f}")
col3.metric("Saldo pendiente", f"${df_filtrado['saldo_pendiente'].sum():,.0f}")
col4.metric("% Pagado promedio", f"{df_filtrado['pct_pagado'].mean():.1f}%")
col5.metric("Dias de mora promedio", f"{df_filtrado['dias_mora'].mean():.0f}")

st.divider()

# ------------------------------------------------------------------
# Tabs de navegacion
# ------------------------------------------------------------------
tab_datos, tab_cuanti, tab_cuali, tab_plots, tab_explorar = st.tabs(
    ["📄 Datos", "🔢 EDA Cuantitativo", "🔤 EDA Cualitativo", "📊 Visualizaciones", "🧭 Explorador"]
)

# --- Tab: Datos crudos ---
with tab_datos:
    st.subheader("Vista previa de los datos sinteticos")
    st.dataframe(df_filtrado.head(200), use_container_width=True)
    st.write("Tipos de datos:")
    st.dataframe(df_filtrado.dtypes.astype(str).rename("tipo"), use_container_width=True)

# --- Tab: EDA Cuantitativo ---
with tab_cuanti:
    st.subheader("Estadisticas descriptivas (variables numericas)")
    num_cols = df_filtrado.select_dtypes(include=np.number).columns.tolist()
    st.dataframe(df_filtrado[num_cols].describe().T, use_container_width=True)

    st.subheader("Matriz de correlacion")
    corr = df_filtrado[num_cols].corr(numeric_only=True)
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlacion entre variables numericas",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Distribucion de una variable numerica")
    var_num = st.selectbox("Selecciona variable numerica", num_cols, index=num_cols.index("monto_deuda"))
    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(df_filtrado, x=var_num, nbins=40, marginal="box", title=f"Histograma de {var_num}")
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_b:
        fig_box = px.box(df_filtrado, y=var_num, points="outliers", title=f"Boxplot de {var_num}")
        st.plotly_chart(fig_box, use_container_width=True)

# --- Tab: EDA Cualitativo ---
with tab_cuali:
    st.subheader("Frecuencias de variables categoricas")
    cat_cols = df_filtrado.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    var_cat = st.selectbox("Selecciona variable categorica", cat_cols)

    conteo = df_filtrado[var_cat].value_counts().reset_index()
    conteo.columns = [var_cat, "conteo"]
    conteo["porcentaje"] = (conteo["conteo"] / conteo["conteo"].sum() * 100).round(1)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.dataframe(conteo, use_container_width=True)
    with col_b:
        fig_bar = px.bar(conteo, x=var_cat, y="conteo", text="porcentaje", title=f"Distribucion de {var_cat}")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Tabla cruzada (crosstab)")
    col_c, col_d = st.columns(2)
    with col_c:
        var_cat_1 = st.selectbox("Variable A", cat_cols, index=0, key="cross_a")
    with col_d:
        var_cat_2 = st.selectbox("Variable B", cat_cols, index=min(1, len(cat_cols) - 1), key="cross_b")

    if var_cat_1 != var_cat_2:
        cross = pd.crosstab(df_filtrado[var_cat_1], df_filtrado[var_cat_2])
        st.dataframe(cross, use_container_width=True)
        fig_heat = px.imshow(cross, text_auto=True, aspect="auto", title=f"{var_cat_1} vs {var_cat_2}")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Selecciona dos variables categoricas diferentes para cruzarlas.")

# --- Tab: Visualizaciones adicionales ---
with tab_plots:
    st.subheader("Relacion entre variables")
    col_a, col_b = st.columns(2)
    with col_a:
        x_var = st.selectbox("Eje X", num_cols, index=num_cols.index("dias_mora"), key="scatter_x")
    with col_b:
        y_var = st.selectbox("Eje Y", num_cols, index=num_cols.index("pct_pagado"), key="scatter_y")

    color_var = st.selectbox("Color por", cat_cols, index=cat_cols.index("estado_cobranza"))
    fig_scatter = px.scatter(
        df_filtrado,
        x=x_var,
        y=y_var,
        color=color_var,
        opacity=0.6,
        hover_data=["usuario_id", "region", "segmento"],
        title=f"{y_var} vs {x_var} coloreado por {color_var}",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Deuda total por region y estado de cobranza")
    agg = df_filtrado.groupby(["region", "estado_cobranza"], as_index=False)["monto_deuda"].sum()
    fig_stack = px.bar(
        agg, x="region", y="monto_deuda", color="estado_cobranza", barmode="stack",
        title="Deuda total por region, segmentada por estado de cobranza",
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    st.subheader("Efectividad de contacto por canal")
    ef = (
        df_filtrado.groupby("canal_contacto", as_index=False)["contacto_efectivo"]
        .mean()
        .rename(columns={"contacto_efectivo": "tasa_efectividad"})
    )
    ef["tasa_efectividad"] = (ef["tasa_efectividad"] * 100).round(1)
    fig_ef = px.bar(ef, x="canal_contacto", y="tasa_efectividad", text="tasa_efectividad",
                     title="Tasa de efectividad de contacto por canal (%)")
    st.plotly_chart(fig_ef, use_container_width=True)

    st.subheader("Tendencia de vencimientos en el tiempo")
    serie = df_filtrado.copy()
    serie["mes_vencimiento"] = serie["fecha_vencimiento"].dt.to_period("M").astype(str)
    tendencia = serie.groupby("mes_vencimiento", as_index=False)["monto_deuda"].sum().sort_values("mes_vencimiento")
    fig_line = px.line(tendencia, x="mes_vencimiento", y="monto_deuda", markers=True,
                        title="Monto de deuda vencida por mes")
    st.plotly_chart(fig_line, use_container_width=True)

# --- Tab: Explorador interactivo por usuario ---
with tab_explorar:
    st.subheader("Buscar un usuario especifico")
    id_buscado = st.number_input(
        "ID de usuario", min_value=int(df["usuario_id"].min()), max_value=int(df["usuario_id"].max()), value=1
    )
    fila = df[df["usuario_id"] == id_buscado]
    if not fila.empty:
        st.dataframe(fila.T.rename(columns={fila.index[0]: "valor"}), use_container_width=True)
    else:
        st.warning("No se encontro ese usuario.")

    st.subheader("Top usuarios por saldo pendiente")
    top_n = st.slider("Cantidad a mostrar", 5, 50, 10)
    top_usuarios = df_filtrado.sort_values("saldo_pendiente", ascending=False).head(top_n)
    st.dataframe(
        top_usuarios[["usuario_id", "region", "segmento", "estado_cobranza", "monto_deuda", "saldo_pendiente"]],
        use_container_width=True,
    )

st.divider()
st.caption("App generada para practicar EDA de cobranza sobre datos 100% sinteticos.")
