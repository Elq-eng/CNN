import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.datasets import mnist  # En este módulo está MNIST en formato numpy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten,Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
#%% punto C
PATH = r'G:\trabajos\software2'
CKPATH = PATH + '\CHECKPOINTS'


(train_images, out_train), (test_images, out_test) = mnist.load_data()

#normalizamos los datos para evitar inestabilidad en ls datos 
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0


# Debido a que son imágenes en escala de grises de entrenamiento,por lo tanto keras espera tensores de 4 dimensiones, la idea es expandir las dimesiones

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)


#Normalizar la salida de los datos el cual son representados con un vector,teniando una variacion de 0 a 1 como max en la posicion que le corresponde
label_binarizer = LabelBinarizer()
out_train = label_binarizer.fit_transform(out_train)
out_test = label_binarizer.fit_transform(out_test)

#se crea la red neuronal  con una capa convolucional
model = Sequential()
model.add(Conv2D(input_shape=(28, 28, 1), filters=64, strides=(2, 2), padding='same', activation='relu', kernel_size=(3, 3)))

# se crean dos capas de agrupacion 
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(MaxPooling2D(pool_size = (2,2)))




#salida de la capa convolucional se entrega un volumen tridimensional, la idea es aplanarlo, para poder que sea accesible a la capa softmax

model.add(Flatten())

#agregamos dos capas dense 
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

#para mejorar se puede poner dos redes neuronales sencillas y una que quite el sobre ajuste del error dropout(0.5)
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# Usamos Adam como optimizador porque sirve como buena selección por defecto pero tambien esta rmsprop. optamos por la exactitud (accuracy) como métrica y una metrica de rendimiento categorical_crossentropy.
print('[INFO] Entrenando red...')
adam = Adam()
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


#Ahora que el modelo se ha definido completamente, con su función de pérdida y su optimizador, podemos entrenarlo. Lo hacemos con la función fit. la funcion puede tener en cuenta el numero de epoca los lotes y los datos de validacion por etapa si se desea.
X_train, X_valid, y_train, y_valid = train_test_split(train_images, out_train, train_size=0.7)

#%%
#si se desea guardar el modelo que tenga mejoress valores  checkpointer 
checkpointer = ModelCheckpoint(filepath = CKPATH + '\model_prueba2.hdf5',verbose = 1, save_best_only = True)

M = model.fit(train_images, out_train, epochs= 10,validation_data=(X_valid, y_valid),
              batch_size=128,callbacks=checkpointer)





#por ultimo se evalua la red con una prediccion 

print('[INFO] Evaluando red...')
result = model.evaluate(x = test_images, y = out_test )

#se imprime el valor de perdida y exactitud
for name, value in zip(model.metrics_names, result):
    print(name, value)

#%% matriz de confusion 

tf.keras.models.load_model(CKPATH + '\model_prueba2.hdf5')

test_predictions = model.predict_classes(test_images)
cm = confusion_matrix(y_true = np.argmax(out_test,axis =1), y_pred = test_predictions)
sns.heatmap(cm , annot = True,
            cmap = 'YlGnBu',fmt="d")
plt.xlabel("Predicted label")
plt.ylabel("True Label")
ax = plt.axes()
ax.set_title('Confusion Matrix \n Accuracy 97,6%' )
plt.show()
print(cm)

#%%
# Finalmente, se crea una grafica por cada uno de los items a tener en cuenta esto permite verificar como evolciona en el tiempo
print('Generando gráfico...')
plt.style.use('ggplot')
plt.figure()
epoch_values = list(range(10))
plt.plot(epoch_values, M.history['loss'], label='Pérdida de entrenamiento')
plt.plot(epoch_values, M.history['val_loss'], label='Pérdida de validación')
plt.plot(epoch_values, M.history['accuracy'], label='Exactitud de entrenamiento')
plt.plot(epoch_values, M.history['val_accuracy'], label='Exactitud de validación')

plt.title('Pérdida y Exactitud de Entrenamiento')
plt.xlabel('Epoch N°')
plt.ylabel('Pérdida/Exactitud')
plt.legend()

plt.savefig(PATH)


