# ============================================================
#  LAB 02 - Regresion Lineal Multiple
#  Dataset: Buzz in Social Media (TomsHardware)
#  Estudiante: Sebastian Arduz
#  Materia: SIS420 - Inteligencia Artificial I
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ 1. CARGAR DATOS CON PANDAS ============

# En Google Colab: subir el archivo TomsHardware.data
# from google.colab import files
# uploaded = files.upload()

nombres_columnas = []
categorias = ['NCD', 'BL', 'NAD', 'AI', 'NAC', 'ND', 'CS', 'AT', 'NA', 'ADL', 'AS_NA', 'AS_NAC']
for cat in categorias:
    for t in range(8):
        nombres_columnas.append(f'{cat}_{t}')
nombres_columnas.append('target_buzz')

datos = pd.read_csv('TomsHardware.data', header=None, names=nombres_columnas)

print("=" * 50)
print("INFORMACION DEL DATASET")
print("=" * 50)
print(f"Filas (m): {datos.shape[0]}")
print(f"Columnas (n): {datos.shape[1]}")
print(f"\nPrimeras 5 filas:")
print(datos.head())
print(f"\nEstadisticas del target (buzz):")
print(datos['target_buzz'].describe())

# ============ 2. PREPARAR DATOS ============

X = datos.drop('target_buzz', axis=1).values.astype(float)
y = datos['target_buzz'].values.astype(float)

m, n = X.shape
print(f"\nm = {m} muestras")
print(f"n = {n} features")

# ============ 3. NORMALIZAR FEATURES ============

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

X_norm, mu, sigma = featureNormalize(X)

X_norm = np.hstack([np.ones((m, 1)), X_norm])

print(f"\nNormalizacion aplicada (z-score)")
print(f"Matriz X con columna de 1s: {X_norm.shape}")

# ============ 4. FUNCION DE COSTO ============

def computeCostMulti(X, y, theta):
    m = len(y)
    prediccion = np.dot(X, theta)
    error = prediccion - y
    J = (1 / (2 * m)) * np.sum(error ** 2)
    return J

# ============ 5. GRADIENTE DESCENDENTE ============

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    J_historial = np.zeros(num_iters)

    for i in range(num_iters):
        prediccion = np.dot(X, theta)
        error = prediccion - y
        theta = theta - (alpha / m) * np.dot(error, X)
        J_historial[i] = computeCostMulti(X, y, theta)

    return theta, J_historial

# ============ 6. ENTRENAR MODELO ============

theta_inicial = np.zeros(n + 1)
alpha = 0.01
iteraciones = 10000

print(f"\n{'=' * 50}")
print("ENTRENAMIENTO")
print(f"{'=' * 50}")
print(f"Alpha (learning rate): {alpha}")
print(f"Iteraciones: {iteraciones}")

theta, J_historial = gradientDescentMulti(X_norm, y, theta_inicial, alpha, iteraciones)

print(f"Costo inicial: {J_historial[0]:,.2f}")
print(f"Costo final:   {J_historial[-1]:,.2f}")
print(f"Reduccion:     {((J_historial[0] - J_historial[-1]) / J_historial[0] * 100):.2f}%")

# ============ 7. GRAFICA: COSTO VS ITERACION ============

plt.figure(figsize=(10, 5))
plt.plot(range(iteraciones), J_historial, color='#e17055', linewidth=2)
plt.title('Funcion de Costo J(theta) vs Numero de Iteracion', fontsize=14)
plt.xlabel('Iteracion', fontsize=12)
plt.ylabel('Costo J(theta)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('costo_vs_iteraciones.png', dpi=100)
plt.show()

# ============ 8. EVALUACION ============

predicciones = np.dot(X_norm, theta)
errores = np.abs(predicciones - y)
error_promedio = np.mean(errores)

print(f"\n{'=' * 50}")
print("EVALUACION DEL MODELO")
print(f"{'=' * 50}")
print(f"Error absoluto promedio: {error_promedio:,.2f}")

# ============ 9. GRAFICA: REAL VS PREDICHO ============

muestra = min(500, len(y))
indices = np.random.choice(len(y), muestra, replace=False)

plt.figure(figsize=(10, 5))
plt.scatter(y[indices], predicciones[indices], alpha=0.4, color='#00cec9', s=15)
plt.plot([0, y.max() * 0.5], [0, y.max() * 0.5], 'r--', linewidth=2, label='Prediccion perfecta')
plt.title('Valor Real vs Valor Predicho', fontsize=14)
plt.xlabel('Buzz Real', fontsize=12)
plt.ylabel('Buzz Predicho', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('real_vs_predicho.png', dpi=100)
plt.show()

# ============ 10. ECUACION NORMAL ============

def normalEqn(X, y):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    return theta

X_normal = np.hstack([np.ones((m, 1)), X])
theta_normal = normalEqn(X_normal, y)

pred_normal = np.dot(X_normal, theta_normal)
error_normal = np.mean(np.abs(pred_normal - y))

print(f"\n{'=' * 50}")
print("COMPARACION DE METODOS")
print(f"{'=' * 50}")
print(f"Gradiente Descendente - Error: {error_promedio:,.2f}")
print(f"Ecuacion Normal       - Error: {error_normal:,.2f}")

# ============ 11. PROBAR DIFERENTES ALPHAS ============

plt.figure(figsize=(10, 5))
alphas = [0.001, 0.003, 0.01, 0.03]
colores_alpha = ['#6c5ce7', '#00cec9', '#e17055', '#fdcb6e']

for a, color in zip(alphas, colores_alpha):
    theta_tmp = np.zeros(n + 1)
    _, J_tmp = gradientDescentMulti(X_norm, y, theta_tmp, a, 3000)
    plt.plot(range(3000), J_tmp, color=color, linewidth=2, label=f'alpha = {a}')

plt.title('Convergencia con diferentes Learning Rates', fontsize=14)
plt.xlabel('Iteracion', fontsize=12)
plt.ylabel('Costo J(theta)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparacion_alphas.png', dpi=100)
plt.show()

print("\nGraficas guardadas: costo_vs_iteraciones.png, real_vs_predicho.png, comparacion_alphas.png")
