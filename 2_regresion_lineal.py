import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

print("ETAPA 1: CARGA Y PREPARACIÓN")

print("\nCargar diabetes_limpio.csv:")
df = pd.read_csv("diabetes_limpio.csv")
print("Total de registros cargados:", len(df))

print("\nFiltrar registros que tienen dato de glu_suero y hb1ac:")
df = df.dropna(subset=["glu_suero", "hb1ac"])
print("Registros disponibles para el modelo:", len(df))


print("\nETAPA 2: ENTRENAMIENTO DEL MODELO")

print("\nDefinir X (glucosa en suero) y y (hemoglobina glucosilada hb1ac):")
X = df[["glu_suero"]]
y = df["hb1ac"]

print("\nDividir datos en 75% entrenamiento y 25% prueba:")
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print("Entrenamiento:", len(X_entrenamiento), "| Prueba:", len(X_prueba))

print("\nEntrenar el modelo de regresion lineal:")
modelo = LinearRegression()
modelo.fit(X_entrenamiento, y_entrenamiento)

print("\nObtener pendiente e intercepto:")
pendiente = modelo.coef_[0]
intercepto = modelo.intercept_
print("Pendiente:", round(pendiente, 4))
print("Intercepto:", round(intercepto, 2))


print("\nETAPA 3: EVALUACIÓN Y PREDICCIONES")

print("\nCalcular el error promedio absoluto (MAE) en el conjunto de prueba:")
predicciones = modelo.predict(X_prueba)
error = mean_absolute_error(y_prueba, predicciones)
print("Error promedio absoluto:", round(error, 2), "puntos de hb1ac")

print("\nPredecir hb1ac para 4 niveles de glucosa (90, 110, 140 y 200):")
glucosas_nuevas = pd.DataFrame({"glu_suero": [90, 110, 140, 200]})
predicciones_nuevas = modelo.predict(glucosas_nuevas)

resultados = pd.DataFrame({
    "Glucosa (mg/dL)": glucosas_nuevas["glu_suero"],
    "Hb1ac estimada (%)": predicciones_nuevas.round(2),
})
print(resultados)


print("\nETAPA 4: GRÁFICA")

print("\nGenerar y guardar la grafica de regresion lineal:")
plt.figure(figsize=(8, 5))
plt.scatter(df["glu_suero"], df["hb1ac"], color="steelblue", alpha=0.6, label="Datos reales")

ordenados = df.sort_values("glu_suero")
linea = modelo.predict(ordenados[["glu_suero"]])
plt.plot(ordenados["glu_suero"], linea, color="red", label="Linea de regresion")

plt.scatter(glucosas_nuevas["glu_suero"], predicciones_nuevas, color="green", s=80, label="Predicciones nuevas")
plt.xlabel("Glucosa en suero (mg/dL)")
plt.ylabel("Hemoglobina glucosilada (hb1ac %)")
plt.title("Regresion lineal: Glucosa vs Hb1ac")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("regresion_glucosa_hb1ac.png")
plt.show()
