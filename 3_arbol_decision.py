import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

print("ETAPA 1: CARGA Y PREPARACIÓN")

print("\nCargar diabetes_limpio.csv:")
df = pd.read_csv("diabetes_limpio.csv")

print("\nSeleccionar columnas necesarias y eliminar registros sin IMC:")
df = df[["edad", "imc", "glu_suero", "insulina", "trig", "riesgo_texto"]].dropna()
print("Registros disponibles:", len(df))


print("\nETAPA 2: ENTRENAMIENTO DEL ÁRBOL")

print("\nSeparar variables independientes (X) y variable objetivo (y):")
X = df[["edad", "imc", "glu_suero", "insulina", "trig"]]
y = df["riesgo_texto"]

print("\nDividir datos en 70% entrenamiento y 30% prueba:")
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.30, random_state=42
)
print("Entrenamiento:", len(X_entrenamiento), "| Prueba:", len(X_prueba))

print("\nEntrenar arbol de decision con profundidad maxima de 3:")
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_entrenamiento, y_entrenamiento)


print("\nETAPA 3: EVALUACIÓN Y PREDICCIONES")

print("\nComparar resultados reales vs predicciones (primeros 10):")
predicciones = modelo.predict(X_prueba)
resultados = pd.DataFrame({
    "Resultado real": y_prueba.values,
    "Prediccion": predicciones,
})
print(resultados.head(10))

print("\nCalcular aciertos, errores y porcentaje de exactitud:")
correctas = (resultados["Resultado real"] == resultados["Prediccion"]).sum()
incorrectas = len(resultados) - correctas
exactitud = accuracy_score(y_prueba, predicciones)
print("Predicciones correctas:", correctas)
print("Predicciones incorrectas:", incorrectas)
print("Porcentaje de exactitud:", round(exactitud * 100, 2), "%")

print("\nRealizar predicciones para 4 pacientes nuevos:")
nuevos_pacientes = pd.DataFrame({
    "edad": [25, 38, 52, 65],
    "imc": [22.5, 27.8, 31.0, 34.2],
    "glu_suero": [85, 108, 135, 180],
    "insulina": [6.0, 12.0, 18.0, 25.0],
    "trig": [95, 160, 220, 300],
})
nuevos_pacientes["Riesgo predicho"] = modelo.predict(nuevos_pacientes)
print(nuevos_pacientes)


print("\nETAPA 4: GRÁFICA")

print("\nDibujar y guardar el arbol de decision:")
plt.figure(figsize=(14, 7))
plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=modelo.classes_,
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title("Arbol de decision: Nivel de riesgo de diabetes")
plt.tight_layout()
plt.savefig("arbol_decision.png")
plt.show()
