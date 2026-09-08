# Laboratorio 06 - Reconstruccion de los 5 laboratorios con PyTorch

**SIS 420 - Inteligencia Artificial I**
**Estudiante:** Arduz Espada Juan Sebastian

## Enunciado

Reconstruir con PyTorch todos los modelos de los primeros cinco laboratorios, usando
objetos para manejo de datos (`Dataset` y `DataLoader`), objetos para optimizacion y costo,
objetos para guardar los pesos, y generacion de metricas.

## Los modelos reconstruidos

| Lab | Dataset | Modelo | Funcion de costo | Optimizador |
|---|---|---|---|---|
| 01 | Auto MPG | `nn.Linear(1, 1)` | `nn.MSELoss` | `SGD` |
| 02 | Buzz in Social Media | `nn.Linear(96, 1)` | `nn.MSELoss` | `Adam` |
| 03a | Communities and Crime | `nn.Linear(n, 1)` | `nn.MSELoss` | `Adam` |
| 03b | Communities and Crime | `nn.Linear(40, 1)` sobre `[X, X²]` | `nn.MSELoss` | `Adam` |
| 03c | Communities and Crime | `torch.linalg.lstsq` | solucion cerrada | ninguno |
| 04 | Credit Card Default | `nn.Linear(26, 1)` | `nn.BCEWithLogitsLoss` | `Adam` |
| 05 | Kuzushiji-MNIST | `nn.Linear(784, 10)` | `nn.CrossEntropyLoss` | `Adam` |

Son **siete entrenamientos**, porque el laboratorio 03 comparaba tres modelos distintos.

## Cumplimiento de requisitos

| Objeto pedido | Implementacion |
|---|---|
| Manejo de datos | `DatasetTabular(Dataset)` + `DataLoader`, una sola clase para los 5 casos |
| Costo | `nn.MSELoss`, `nn.BCEWithLogitsLoss`, `nn.CrossEntropyLoss` |
| Optimizacion | `torch.optim.SGD`, `torch.optim.Adam` |
| Guardar pesos | `GestorCheckpoints` con mejor modelo y modelo final |
| Metricas | MAE, RMSE, R², exactitud, precision, sensibilidad, F1, AUC, matriz de confusion |
| Otros objetos | Clase `Entrenador` que encapsula el bucle completo |
| Video | **Pendiente de grabar** |

## Las tres clases propias

### `DatasetTabular(Dataset)`

Implementa `__init__`, `__len__` y `__getitem__`. El parametro `tipo_etiqueta` permite
usar la misma clase para los tres tipos de problema:

- `'float'` para regresion y clasificacion binaria, con forma `(m, 1)`
- `'long'` para clasificacion multiclase, que es lo que espera `CrossEntropyLoss`

### `GestorCheckpoints`

Guarda dos archivos por modelo: `_mejor.pt` (se sobrescribe cada vez que mejora la metrica
de validacion) y `_final.pt` (el estado al terminar). Cada checkpoint guarda el
`state_dict` del modelo, **el estado del optimizador**, la epoca y la metrica, de modo que
un entrenamiento interrumpido puede reanudarse exactamente donde quedo.

### `Entrenador`

Reemplaza el bucle `for i in range(iteraciones)` que se escribia a mano. Ejecuta los cinco
pasos de PyTorch en cada lote, evalua al final de cada epoca, registra el historial y
coordina el guardado de checkpoints.

## Archivos

| Archivo | Descripcion |
|---|---|
| `lab06_pytorch_reconstruccion.ipynb` | Cuadernillo completo. **Es el archivo a entregar.** |
| `pesos/` | Checkpoints, se generan al ejecutar (no versionados) |


Los datasets **no se incluyen**: el cuadernillo los descarga solo de sus fuentes
originales (UCI, codh.rois.ac.jp, y el propio repositorio de GitHub para los del lab 02 y
03).

## Resultados

| Modelo | Metrica | Manual (NumPy) | PyTorch |
|---|---|---|---|
| 01 - Lineal simple | `b` | 46.317364 | **46.336226** |
| 01 - Lineal simple | `w` | -0.00767661 | **-0.00767056** |
| 01 - Lineal simple | R² | — | 0.6843 |
| 02 - Lineal multiple | R² | — | **0.9579** |
| 03a - Lineal | R² | columna equivocada | 0.6753 |
| 03b - Polinomica | R² | columna equivocada | **0.6861** |
| 03c - Ecuacion normal | R² | columna equivocada | 0.6810 |
| 04 - Logistica binaria | AUC | 0.7263 | **0.7247** |
| 04 - Logistica binaria | F1 | 0.5152 | **0.5159** |
| 05 - Multiclase | Exactitud | 80.44 % | **81.19 %** |

Tiempo total de los siete entrenamientos: **~72 segundos** en CPU.

El laboratorio 05 sale **mejor en PyTorch** (81.19% contra 80.44%) porque
`CrossEntropyLoss` usa softmax, que trata las clases como mutuamente excluyentes, mientras
que el one vs. all original entrenaba 10 clasificadores independientes.

## Dos errores encontrados durante la reconstruccion

### 1. Columnas desalineadas en el dataset del laboratorio 03

El modelo 3a dio inicialmente **R² = 0.98** prediciendo criminalidad, un valor imposible.
La causa: `CommViolPredUnnormalizedData.txt` tiene **147 columnas** pero la lista de
nombres usada tenia **145**. Pandas no falla en ese caso, convierte las columnas sobrantes
del principio en indice y **desplaza todos los nombres**. La columna etiquetada
`ViolentCrimesPerPop` contenia otra variable.

Dos errores en la lista:

1. Faltaban `OwnOccQrange` (tras `OwnOccHiQuart`) y `RentQrange` (tras `RentHighQ`).
2. `ViolentCrimesPerPop` estaba **antes** de `murders`; en el dataset oficial va **al
   final**, tras `arsonsPerPop`.

Comprobacion de que la correccion es buena:

| | Antes (mal) | Despues (bien) |
|---|---|---|
| Mediana del target | 1.0 | 374.1 |
| Maximo del target | 1946 | 4877.1 |
| Mayor correlacion | `PctWorkMomYoungKids` 0.978 | `PctIlleg` 0.739 |
| R² del modelo lineal | 0.9835 | 0.6753 |
| Ejemplos | 2214 | 1994 |

> **Este mismo error esta en el cuadernillo del laboratorio 03 que ya esta en el
> repositorio.** Ahi el modelo se entreno contra una columna equivocada, asi que sus
> resultados no son validos.

### 2. Etiqueta sin normalizar en los modelos de regresion

El modelo 2 daba **R² = 0.156** pese a que la variable `ND_7` correlaciona **0.96** con la
etiqueta. La causa: **Adam limita cada paso al tamanio de la tasa de aprendizaje**. Con
`lr = 0.01` y una etiqueta que llega a 265.916, alcanzar el sesgo necesario (~3.500)
exigiria cientos de miles de pasos, y solo hay 3.560.

La solucion es **normalizar tambien la etiqueta** y deshacer la normalizacion antes de
calcular las metricas. Con eso el R² pasa de **0.156 a 0.9579**.

Las curvas de R² del historial no necesitan conversion porque **el R² es invariante a
transformaciones afines**; solo el MAE y el RMSE hay que devolverlos a la escala real.

## Detalles tecnicos a destacar en el video

1. **`nn.Linear(1, 1)` tiene exactamente los mismos dos parametros del laboratorio 01**:
   `weight` es `w` y `bias` es `b`. Como el peso del vehiculo va de 1613 a 5140, se
   normaliza `x` para entrenar y luego se deshace la normalizacion:

   `w = w' / σ` y `b = b' − w'·μ/σ`

   Eso permite comparar directamente con el resultado de la busqueda por rejilla.

2. **`BCEWithLogitsLoss` en lugar de `Sigmoid` + `BCELoss`.** En el laboratorio 04 hubo
   que escribir a mano una sigmoide estable con dos ramas segun el signo de `z`.
   `BCEWithLogitsLoss` combina sigmoide y entropia cruzada en una operacion y aplica ese
   cuidado internamente. Por eso el modelo devuelve **logits**, no probabilidades.

3. **`CrossEntropyLoss` no es one vs. all.** Aplica **softmax**, que fuerza a que las 10
   salidas sumen 1 y trata las clases como mutuamente excluyentes. El laboratorio 05 usaba
   10 clasificadores binarios independientes. Misma forma de matriz de pesos, funcion de
   costo distinta.

4. **`torch.linalg.lstsq` en lugar de `(XᵀX)⁻¹Xᵀy`.** La inversion directa falla o da
   resultados sin sentido cuando la matriz esta mal condicionada, que es el caso de
   Communities and Crime, donde muchas columnas correlacionan fuertemente entre si.

5. **`optimizador.zero_grad()` es obligatorio.** PyTorch **acumula** los gradientes en
   lugar de reemplazarlos. Omitir esa linea suma el gradiente del lote actual al del
   anterior y corrompe el entrenamiento de forma silenciosa.

6. **La verificacion de los checkpoints.** Se crea un modelo nuevo sin entrenar, se le
   cargan los pesos guardados y se comprueba con `torch.allclose` que produce predicciones
   identicas a las del modelo original.

## Lo que PyTorch aporta y lo que no

**Aporta:** el gradiente no se deriva a mano (`backward()`), la estabilidad numerica viene
resuelta, el entrenamiento por lotes es gratis con el `DataLoader`, y cambiar de
optimizador es una linea.

**No aporta:** los modelos siguen siendo **lineales** y tienen el mismo techo que tenian
antes. El laboratorio 05 sigue limitado a poco mas del 80% porque una capa lineal solo
traza fronteras lineales, sin importar la libreria. Para superarlo hay que cambiar el
modelo, no la herramienta.

## Pendiente

- [ ] Grabar el video explicativo
- [ ] Subir al repositorio https://github.com/Sebillasxd2/SIS-420---IA
- [ ] Entregar en eCampus con la direccion del repositorio
