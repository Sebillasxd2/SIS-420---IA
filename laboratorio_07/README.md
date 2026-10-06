# Laboratorio 07 - Aprendizaje no supervisado, semisupervisado y activo

| | |
|---|---|
| **Estudiante** | Arduz Espada Juan Sebastian |
| **Materia** | SIS 420 - Inteligencia Artificial I |
| **Dataset del punto 3** | Sign Language MNIST (registrado con el encargado) |

## Que hay en cada archivo

| Archivo | Punto | Que hace |
|---|---|---|
| `lab07_punto1_kmeans_2d.ipynb` | 1 | Generador de datos aleatorio modificado + KMeans en 2D |
| `lab07_punto2_kmeans_3d.ipynb` | 2 | Lo mismo pero con puntos de 3 dimensiones |
| `lab07_punto3_semisupervisado_activo.ipynb` | 3 | Dataset de imagenes sin etiquetas + semisupervisado + activo |
| `guion_video.txt` | - | Guion del video explicativo |

Los tres se ejecutan en Google Colab con "Entorno de ejecucion > Ejecutar todo". Los puntos 1 y 2 tardan menos de 1 minuto y no descargan nada; el punto 3 tarda menos de 2 minutos y descarga el dataset solo, con kagglehub.

---

## Punto 1 - Generador aleatorio + KMeans en 2D

### Como quedo el generador

Lo que pedia el enunciado y como lo hice:

| Pedido | Como lo resolvi |
|---|---|
| Entre 1 y 20 centroides | `rng.randint(1, 21)` |
| Distancia importante entre centroides | Muestreo por rechazo: sorteo una posicion y la descarto si cae a menos de 20 de alguna ya aceptada |
| Posiciones al azar | `rng.uniform` dentro de un area que crece con la cantidad de centroides |
| Distancia de los puntos a su centroide al azar | Cada centroide recibe su propia dispersion, sorteada entre 20/12 y 20/6 |
| Cantidad de puntos por centroide al azar | `rng.randint(60, 401)` por centroide |

La dispersion se limita a un sexto de la distancia minima para que los grupos se vean separados: como casi todos los puntos de una normal caen dentro de 3 desviaciones, cada grupo ocupa como mucho la mitad de la distancia hasta su vecino.

### Resultados de la corrida

- Se generaron **7 centroides y 1.719 puntos**, con distancia minima real de 24.86 entre centroides.
- KMeans propio (programado a mano) contra el de sklearn: **la misma inercia (20544.1675) y la misma posicion de los centroides** (diferencia 0.0). Use el propio para poder graficar cada iteracion.
- Error promedio entre cada centroide encontrado y el real: **0.20**.
- **Ningun punto quedo en el grupo equivocado** (0 de 1.719).
- El metodo del codo y el silhouette detectaron **k = 7**, que es el valor real.
- Validacion con 2, 5, 8, 12, 16 y 20 centroides: **el silhouette acerto el k en los 6 casos**.

### Lo que se grafica

1. El dataset generado, pintado por grupo real.
2. El proceso de KMeans iteracion por iteracion, con los centroides moviendose.
3. La curva de inercia por iteracion.
4. Los centroides finales con los 5 puntos mas cercanos a cada uno marcados en rojo.
5. Barras de puntos por centroide: generados contra asignados.
6. Metodo del codo y silhouette.
7. Diagrama de silueta por grupo.

Ademas hay dos tablas: una con la posicion de cada centroide (encontrada y real, con su error y su cantidad de puntos) y otra con los **5 puntos mas cercanos a cada centroide**, con sus coordenadas y su distancia.

### Un detalle que salio interesante

Para que se vea el proceso completo repeti el entrenamiento con un arranque malo a proposito: todos los centroides juntos cerca del promedio de los datos. Ahi KMeans tardo 10 iteraciones en vez de 2 y **termino en una solucion peor** (inercia 73.142 contra 20.544): dos centroides se quedaron en el mismo grupo y otro grupo se quedo sin ninguno. Muestra que KMeans depende de la inicializacion, y por eso se usa kmeans++ y se corre varias veces.

---

## Punto 2 - Lo mismo en 3D

Es el mismo codigo del punto 1 con `DIMENSION = 3`. El generador, KMeans, el codo y el silhouette funcionan igual, porque la distancia y el promedio se calculan de la misma forma con 2 o con 3 coordenadas. Lo unico que cambia son las graficas, que usan ejes 3D.

- Se generaron **7 centroides y 1.276 puntos** en 3D.
- KMeans propio contra sklearn: misma inercia (**24061.204**), diferencia 0.0.
- Error promedio de los centroides: **0.35**. Ningun punto mal agrupado (0 de 1.276).
- Silhouette detecto **k = 7**, el valor real, y acerto tambien en los 6 casos de la validacion (2, 5, 8, 12, 16 y 20).

---

## Punto 3 - Sign Language MNIST: semisupervisado y activo

### El dataset

| | |
|---|---|
| Nombre | Sign Language MNIST |
| Link | https://www.kaggle.com/datasets/datamunge/sign-language-mnist |
| Tipo | Imagenes de 28x28 en escala de grises |
| m | 27.455 imagenes de trabajo (+ 7.172 de prueba) |
| n | 784 caracteristicas (pixeles) |
| Clases | 24 letras del alfabeto de senias americano (sin J ni Z) |

Cumple lo pedido: m mayor a 10.000 y n mayor a 10. **Se trabaja como si no tuviera etiquetas**: el agrupamiento y la eleccion de que imagenes etiquetar se hacen sin mirarlas. Las etiquetas reales se usan solo para medir los resultados y para hacer de persona que etiqueta en el aprendizaje activo.

### Preprocesado: pixeles contra HOG

Probe dos formas de representar las imagenes y compare las dos:

| Caracteristicas | Exactitud con todas las etiquetas | Pureza de los clusters |
|---|---|---|
| pixeles + PCA(50) | 0.690 | 0.2017 |
| **HOG + PCA(50)** | **0.837** | **0.4865** |

HOG (histograma de gradientes orientados) mide hacia donde apuntan los bordes en cada pedacito de la imagen, o sea la forma de la mano, y no depende tanto del brillo de la foto. Por eso el resto del trabajo usa HOG.

### Aprendizaje no supervisado

KMeans con k = 24 sobre las 27.455 imagenes, sin etiquetas. La **pureza total es 0.49**: poniendole a cada cluster la letra que mas se repite adentro se acierta la mitad de las imagenes. Hay clusters casi perfectos (uno es 100% la letra Y, otro 99.8% la C, otro 99.5% la F) y otros mezclados.

### Aprendizaje semisupervisado (presupuesto: 50 etiquetas)

| Estrategia | Imagenes de entrenamiento | Exactitud |
|---|---|---|
| 50 al azar | 50 | 0.2752 |
| 50 representantes de KMeans | 50 | 0.3777 |
| **Propagacion total (50 -> 27.455)** | 27.455 | **0.4806** |
| Propagacion parcial (30% mas cercano) | 8.214 | 0.4559 |

Elegir las 50 con KMeans gana 10 puntos sobre elegirlas al azar, y propagar esas etiquetas al resto del cluster gana 21 puntos. Solo el 55% de las etiquetas propagadas son correctas, pero igual conviene porque el modelo pasa de ver 50 ejemplos a ver 27.455.

Con 50 etiquetas (el 0.18% del dataset) se llega al 57% de lo que se logra etiquetando las 27.455.

### Aprendizaje activo (10 rondas de 48 etiquetas)

| Estrategia | Exactitud final con 528 etiquetas |
|---|---|
| Pedir al azar | 0.6834 (+- 0.016) |
| **Pedir por incertidumbre (margen)** | **0.7334 (+- 0.015)** |

El modelo pide las imagenes donde la diferencia entre su primera y su segunda opcion es mas chica, o sea donde mas duda. Son las que estan en el borde entre dos letras. Todo se promedia sobre 3 semillas.

Con 528 etiquetas (el 1.9% del dataset) se llega a 0.7334, contra 0.837 que da etiquetar todo.

---

## Conceptos para la defensa

- **KMeans:** asigna cada punto al centroide mas cercano y despues mueve cada centroide al promedio de sus puntos; repite hasta que no se mueven.
- **Inercia:** suma de las distancias al cuadrado de cada punto a su centroide. Siempre baja al agregar grupos, por eso sola no sirve para elegir k.
- **Metodo del codo:** se elige el k donde la inercia deja de bajar fuerte.
- **Silhouette:** compara que tan cerca esta un punto de su grupo contra el grupo vecino mas cercano. Va de -1 a 1.
- **kmeans++:** forma de elegir los centroides iniciales dando mas probabilidad a los puntos lejanos, para no arrancar todos juntos.
- **PCA:** se queda con las direcciones donde los datos mas varian; sirve para bajar la cantidad de caracteristicas.
- **HOG:** describe la imagen por la direccion de sus bordes en vez de por el brillo de cada pixel.
- **Semisupervisado:** usar pocas etiquetas y aprovechar la estructura de los datos sin etiquetar (aca, propagar la etiqueta del representante a todo su cluster).
- **Activo:** el modelo elige que ejemplos quiere que le etiqueten, pidiendo aquellos donde esta mas dudoso.

## Como ejecutar

Los tres cuadernillos se abren en Google Colab y se ejecutan con "Ejecutar todo". El punto 3 necesita `kagglehub` y `scikit-image`, que ya vienen en Colab.
