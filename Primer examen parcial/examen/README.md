# Primer Examen Parcial - SIS 420 Inteligencia Artificial I

Estudiante: Arduz Espada Juan Sebastian

Modelo neuronal para clasificar quien habla en audios de 1 segundo, con el Speaker Recognition Dataset (Kaggle): https://www.kaggle.com/datasets/kongaevans/speaker-recognition-dataset

- 5 personas, 1500 audios por persona. Caracteristicas MFCC (80 por audio).
- Audio con ruido de fondo (los ruidos vienen en el mismo dataset).
- Division secuencial 70/15/15 en entrenamiento, validacion y prueba.
- Red neuronal en PyTorch: 1 capa oculta de 128 neuronas con ReLU, dropout 0.2, Adam.
- Resultado en prueba: 94.1% de exactitud (el modelo lineal saca 85.8%).

## Archivos

- Primer_Parcial_SIS420_Arduz_Speaker_Recognition.ipynb: cuadernillo de Google Colab. El dataset se descarga solo con kagglehub. Tarda unos 5 minutos.
- EXPLICACION.txt: explicacion de lo que hice.
