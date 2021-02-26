#import sys
#import os ##sys y os son librerias para moverse en carpetas dentro del sistema operativo
#from tensorflow.python.keras.preprocessing.image import ImgeDataGenerator
#from tensorflow.python.keras import optimizers
#from tensorflow.python.keras.models import Sequentials #Para redes neuronales#
#from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation 
#from tensorflow.python.keras.layers import Covolution2D, MaxPooling2D
#from tensorflow.python.keras.layers import backend as K

#K.clear_session()
#data_training= '#/file_ruta_imagenes_entrenar#'
#data_validation='_ruta_imagenes_validar'
#Parametros 
#epoch 24 ##numero de iteraciones del dataset durante el training
#height, lenght = 100, 100
#batch_size=32 ##numero de imagenes que la computadora va a procesar en c/u de los pasos
#steps=930 ##numero de veces que se procesa la info en c/u de las epocas
#validation_steps=200
#filtersConv1=32
#filtersConv2=64
#size_filter1=(3,3)
#size_filter2=(2,2)
#size_pool=(2,2)
#classes=3
#lr=0.005 #learning rate

##IMAGE PREPROCESSING

#training_datagen=ImageDataGenerator(
   rescale=1./225, ##para mejorar el rango de pixeles y que el entrenamiento sea mas eficiente
   shear_range=0.3, ##inclinar la imagen para que la imagen no siempre tenga que estar centrada para reconocerla
   zoom_range=0.3,
   horizontal_flip=True ##la red neuronal aprenda a distinguir direccionalidad 
   )
   #validation_datagen=ImageDataGenerator(
    rescale=1./225
)
image_training=training_datagen.flow_from_directory(
   data_training,
   target_size= (height, lenght),
   batch_size= batch_size,
   class_mode='categorical'
)
image_validation=validation_data_gen.flow_from_directory(
   data_validation,
   target_size= (height, lenght),
   batch_size= batch_size,
   class_mode='categorical'
)

##CREATING CNN NETWORK
cnn= Sequential() ##para indicar que la red es secuencial
cnn.add(Convolution2d(filterConv1, size_filter1, padding='same', input_shape= (height, lenght), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size=size_pool))
cnn.add(Convolution2D(filterConv2, size_filter2, padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=size_pool))
cnn.add(Flatten))
cnn.add(Dense(256, activation='relu')) ##Dense, conecta las neuronas
cnn.add(Dropout(0.5)) ##orden para solo activar 50% de las 256 neuronas, asi aprende caminos alternos para clasificar los datos (mejora la adaptacion a nueva informacion)
cnn.add(Dense(classes, activation='softMax'))