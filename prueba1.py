# Importamos las librerías que necesitaremos.
import argparse
from builtins import str

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.datasets import mnist  # En este módulo está MNIST en formato numpy
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Definimos los parámetros de entrada del programa. El único obligatorio es -o, el cual es la ruta
# donde guardaremos el gráfico con la curva de entrenamiento de la red.
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-o', '--output', required=True, help='Ruta al gráfico de salida.')
argument_parser.add_argument('-e', '--epochs', default=100, type=int,
                             help='Número de epochs que se entrenará la red (100 por defecto).')
argument_parser.add_argument('-b', '--batch-size', default=128, type=int,
                             help='Número de imágenes por lote (batch size).')
arguments = vars(argument_parser.parse_args())

# Aquí accedemos a MNIST mediante Keras. Si no lo ha hecho previamente, Keras descargará el conjunto de datos; si ya
# lo hizo, lo buscará en su cache local. El conjunto de datos ya viene dividido en subconjuntos de entrenamiento y pruebas.
print('[INFO] Descargando/accediendo a MNIST...')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Queremos normalizar los datos para evitar problemas de inestabilidad numérica. Aprovechamos la oportunidad
# para asegurarnos de que los datos estén representados con floats de 32-bits.
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Debido a que son imágenes en escala de grises, el único canal de color se halla implícito. Sin embargo,
# Keras espera tensores de 4 dimensiones (incluyendo el batch size), no de 3, por lo que tenemos que expandir las dimensiones
# de los datos.
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

# Ahora debemos convertir las etiquetas en vectores one-hot encoded.
# Esta será la transformación:
# - 0 --> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# - 1 --> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# - 2 --> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# - 3 --> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# - 4 --> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# - 5 --> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# - 6 --> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
# - 7 --> [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# - 8 --> [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
# - 9 --> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)


# Crearemos una red neuronal con una capa convolucional, y tres totalmente conectadas (FCN). Como función de activatión
# usamos ReLU.
# Como es un problema de clasificación múltiple, utilizamos softmax como función de activación de la última capa, ya que
# nos devuelve valores entre 0 y 1 para cada categoría, pudiendo *interpretarlos* como probabilidades.
model = Sequential()
model.add(Conv2D(input_shape=(28, 28, 1), filters=64, strides=(2, 2), padding='same', activation='relu', kernel_size=(3, 3)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Usamos Adam como optimizador porque sirve como buena selección por defecto. Como MNIST está balanceado, optamos
# por la exactitud (accuracy) como métrica.
print('[INFO] Entrenando red...')
adam = Adam()
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Usaremos 20% de la data de entrenamiento para validar el desempeño de la red en cada epoch.
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)
H = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=arguments['epochs'],
              batch_size=arguments['batch_size'])

# Nota cómo en las líneas 84 y 85 obtenemos el *índice* de la probabilidad más alta de cada vector, ya que esta
# será la categoría.
print('[INFO] Evaluando red...')
predictions = model.predict(X_test, batch_size=arguments['batch_size'])
print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in label_binarizer.classes_]))

# Finalmente, creamos un gráfico con las curvas de pérdida y exactitud a lo largo del proceso de entrenamiento. Esto
# nos permitirá analizar la evolución de la red.
print('[INFO] Generando gráfico...')
plt.style.use('ggplot')
plt.figure()
epoch_values = list(range(arguments['epochs']))
plt.plot(epoch_values, H.history['loss'], label='Pérdida de entrenamiento')
plt.plot(epoch_values, H.history['val_loss'], label='Pérdida de validación')
plt.plot(epoch_values, H.history['accuracy'], label='Exactitud de entrenamiento')
plt.plot(epoch_values, H.history['val_accuracy'], label='Exactitud de validación')

plt.title('Pérdida y Exactitud de Entrenamiento')
plt.xlabel('Epoch N°')
plt.ylabel('Pérdida/Exactitud')
plt.legend()
plt.savefig(arguments['output'])