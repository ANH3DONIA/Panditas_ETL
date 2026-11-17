import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes en Mexico", layout="wide")

st.title("Analisis de Diabetes en Mexico")
st.write("Dashboard interactivo con datos limpios de diabetes_limpio.csv")

df = pd.read_csv("diabetes_limpio.csv")

# Filtros laterales
st.sidebar.header("Filtros")
sexos = st.sidebar.multiselect("Sexo", ["Mujer", "Hombre"], default=["Mujer", "Hombre"])
edades = st.sidebar.multiselect(
    "Rango de edad",
    ["20-29", "30-44", "45-59", "60+"],
    default=["20-29", "30-44", "45-59", "60+"],
)
ciudades = st.sidebar.multiselect("Ciudad (opcional)", sorted(df["Ciudad"].unique()))

datos = df[df["sexo"].isin(sexos) & df["rango_edad"].isin(edades)]
if ciudades:
    datos = datos[datos["Ciudad"].isin(ciudades)]

if datos.empty:
    st.warning("No hay registros con los filtros seleccionados.")
    st.stop()

# Resumen principal (KPIs)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de pacientes", len(datos))
col2.metric("Glucosa promedio", f"{datos['glu_suero'].mean():.1f} mg/dL")
imc_prom = datos["imc"].mean()
col3.metric("IMC promedio", f"{imc_prom:.1f}" if pd.notna(imc_prom) else "Sin dato")
riesgo_alto = (datos["riesgo_texto"] == "Alto").mean() * 100
col4.metric("Pacientes en riesgo alto", f"{riesgo_alto:.1f} %")

st.divider()

# Graficas exploratorias
graf1, graf2 = st.columns(2)

with graf1:
    st.subheader("Pacientes por nivel de riesgo")
    conteo_riesgo = datos["riesgo_texto"].value_counts()
    st.bar_chart(conteo_riesgo)

with graf2:
    st.subheader("Glucosa promedio por rango de edad")
    glucosa_edad = datos.groupby("rango_edad")["glu_suero"].mean().round(2)
    st.bar_chart(glucosa_edad)

st.divider()

# Regresion lineal: glucosa -> hb1ac
st.subheader("Modelo 1: Regresion Lineal (Glucosa vs Hb1ac)")
datos_reg = df.dropna(subset=["glu_suero", "hb1ac"])
X_reg = datos_reg[["glu_suero"]]
y_reg = datos_reg["hb1ac"]
modelo_reg = LinearRegression()
modelo_reg.fit(X_reg, y_reg)

reg_col1, reg_col2 = st.columns([2, 1])

with reg_col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(datos_reg["glu_suero"], datos_reg["hb1ac"], color="steelblue", alpha=0.6, label="Datos reales")
    ordenados = datos_reg.sort_values("glu_suero")
    ax.plot(ordenados["glu_suero"], modelo_reg.predict(ordenados[["glu_suero"]]), color="red", label="Linea de regresion")
    ax.set_xlabel("Glucosa en suero (mg/dL)")
    ax.set_ylabel("Hb1ac (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with reg_col2:
    glucosa_input = st.number_input("Ingresa glucosa (mg/dL):", min_value=50, max_value=400, value=110)
    hb1ac_est = modelo_reg.predict(pd.DataFrame({"glu_suero": [glucosa_input]}))[0]
    st.metric("Hb1ac estimada", f"{hb1ac_est:.2f} %")

st.divider()

# Arbol de decision: prediccion de riesgo
st.subheader("Modelo 2: Arbol de Decision (Clasificacion de riesgo)")
datos_arbol = df[["edad", "imc", "glu_suero", "insulina", "trig", "riesgo_texto"]].dropna()
X_arb = datos_arbol[["edad", "imc", "glu_suero", "insulina", "trig"]]
y_arb = datos_arbol["riesgo_texto"]

X_ent, X_pru, y_ent, y_pru = train_test_split(X_arb, y_arb, test_size=0.30, random_state=42)
modelo_arb = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo_arb.fit(X_ent, y_ent)
exactitud = accuracy_score(y_pru, modelo_arb.predict(X_pru))
st.write(f"Exactitud del arbol de decision: **{exactitud * 100:.2f} %**")

c1, c2, c3, c4, c5 = st.columns(5)
in_edad = c1.number_input("Edad", min_value=18, max_value=100, value=35)
in_imc = c2.number_input("IMC", min_value=15.0, max_value=60.0, value=26.0)
in_glu = c3.number_input("Glucosa", min_value=40, max_value=400, value=105)
in_ins = c4.number_input("Insulina", min_value=1.0, max_value=100.0, value=10.0)
in_trig = c5.number_input("Trigliceridos", min_value=30, max_value=1000, value=150)

paciente = pd.DataFrame({
    "edad": [in_edad],
    "imc": [in_imc],
    "glu_suero": [in_glu],
    "insulina": [in_ins],
    "trig": [in_trig],
})
riesgo_pred = modelo_arb.predict(paciente)[0]
st.info(f"Riesgo predicho para el paciente: **{riesgo_pred}**")

st.subheader("Vista previa de los datos filtrados")
st.dataframe(datos.head(20))
