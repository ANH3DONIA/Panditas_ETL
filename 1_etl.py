import pandas as pd

print("ETAPA 1: EXTRACCIÓN")

print("\nImportar pandas y cargar")
df = pd.read_csv("Diabetes_Mexico.csv")

print("\nPrimeros 10 registros:")
print(df.head(10))

print("\nDimensiones, columnas y tipos de datos")
print(df.shape)
print(df.columns)
df.info()


print("\nETAPA 2: TRANSFORMACIÓN")

print("\nCantidad de valores nulos por columna")
print(df.isnull().sum())

print("\nCantidad de registros duplicados y eliminacion")
print("Duplicados:", df.duplicated().sum())
df = df.drop_duplicates()

print("\nEliminar columnas que no sirven para el analisis")
df = df.drop(columns=["muestra_suero", "ponde_venosa", "ponde_hemo", "estrato", "est_sel"])
print("Columnas restantes:", len(df.columns))

print("\nCorregir columnas con coma decimal y convertirlas a numero")
cols_coma = ["Peso", "Estatura", "ac_urico", "albu", "col_hdl", "creat", "insulina"]
for col in cols_coma:
    df[col] = df[col].astype(str).str.replace(",", ".")
    df[col] = pd.to_numeric(df[col], errors="coerce")
print(df[cols_coma].dtypes)

print("\nEliminar registros con edad vacia o negativa")
df = df.dropna(subset=["edad"])
df = df[df["edad"] > 0]
print("Registros validos:", len(df))

print("\nRecalcular IMC con peso / (estatura en metros)^2")
df["imc"] = (df["Peso"] / ((df["Estatura"] / 100) ** 2)).round(2)
print(df[["Peso", "Estatura", "imc"]].head())

print("\nUnificar formato de texto en sexo y Ciudad")
df["sexo"] = df["sexo"].str.strip().str.capitalize()
df["Ciudad"] = df["Ciudad"].str.strip().str.title()
print(df[["sexo", "Ciudad"]].head())

print("\nCrear columna rango_edad")
df["rango_edad"] = pd.cut(
    df["edad"],
    bins=[19, 29, 44, 59, 120],
    labels=["20-29", "30-44", "45-59", "60+"]
)
print(df[["edad", "rango_edad"]].head())

print("\nCrear columna categoria_glucosa")
df["categoria_glucosa"] = pd.cut(
    df["glu_suero"],
    bins=[0, 100, 126, 2000],
    labels=["Normal", "Prediabetes", "Diabetes"],
    right=False
)
print(df[["glu_suero", "categoria_glucosa"]].head())

print("\nCrear columna riesgo_texto con map()")
df["riesgo_texto"] = df["riesgo_diabetes_cat"].map({0: "Bajo", 1: "Moderado", 2: "Alto"})
print(df[["riesgo_diabetes_cat", "riesgo_texto"]].head())


print("\nETAPA 3: CARGA")

print("\nGuardar dataset limpio en diabetes_limpio.csv")
df.to_csv("diabetes_limpio.csv", index=False)

print("\nPrimeros 10 registros del dataset limpio")
print(df.head(10))


print("\nETAPA 4: ANÁLISIS")

print("\nCantidad de personas por nivel de riesgo")
print(df["riesgo_texto"].value_counts())

print("\nGlucosa promedio por rango de edad")
print(df.groupby("rango_edad")["glu_suero"].mean().round(2))

print("\nPromedio de IMC por sexo")
print(df.groupby("sexo")["imc"].mean().round(2))