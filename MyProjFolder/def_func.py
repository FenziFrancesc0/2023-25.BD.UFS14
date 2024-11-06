import azure.functions as func
import logging
import cv2
import os
import numpy as np
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import utils, models
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

app = func.FunctionApp()

@app.route(route="MyHttpTrigger", auth_level=func.AuthLevel.ANONYMOUS)
def MyHttpTrigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    # main
    # punto la cartella
    image_directory = './datasets'
    no_tumor_images = os.listdir(image_directory + 'no/')
    yes_tumor_images = os.listdir(image_directory + 'yes/')

    # quante immagini ci sono? No Tumor:  1500 | Tumor:  1500
    print('No Tumor: ', len(no_tumor_images))
    print('Tumor: ',len(yes_tumor_images))

    # preparazione del dataset
    dataset = []
    label = []
    input_size = 64

    for i, image_name in enumerate(no_tumor_images):
        if(image_name.split('.')[1]=='jpg'):
            # cv2.imread legge le immagini nel percorso
            image = cv2.imread(image_directory+'no/'+image_name)
            # converte l'immagine in un oggetto PIL in formato RGB
            image = Image.fromarray(image, 'RGB')
            image = image.resize((input_size,input_size))
            dataset.append(np.array(image))
            # aggiungo l'etichetta
            label.append(0)

    for i, image_name in enumerate(yes_tumor_images):
        if(image_name.split('.')[1]=='jpg'):
            # cv2.imread legge le immagini nel percorso
            image = cv2.imread(image_directory+'yes/'+image_name)
            # converte l'immagine in un oggetto PIL in formato RGB
            image = Image.fromarray(image, 'RGB')
            image = image.resize((input_size,input_size))
            dataset.append(np.array(image))
            # aggiungo l'etichetta
            label.append(1)
            
    dataset = np.array(dataset)
    label = np.array(label)

    # divisione tra train e test | 80% train 20% test
    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0, stratify=label)

    # normalizzazione
    x_train = x_train/255.0
    x_test = x_test/255.0

    # data augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2, # simula delle variazioni di posizione laterali
        height_shift_range=0.2, # simula delle variazioni di posizione verticali
        horizontal_flip=True,
        zoom_range=0.2
    )
    datagen.fit(x_train)

    print(x_train.shape) # risultato -> (2400, 64, 64, 3) -> (n_images, image_width, image_height, n_channel)
    print(x_test.shape) # (600, 64, 64, 3)
    print(y_train.shape) # (2400,)
    print(y_test.shape) # (600,) 

    # costruzione del modello
    model = Sequential()

    # livello 1
    model.add(Conv2D(32, (3,3), input_shape=(input_size, input_size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # livello 2
    model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform')) # he_uniform adatta la rete per relu
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # livello 3
    model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # livello 4
    model.add(Flatten())

    # livello 5
    model.add(Dense(64))
    model.add(Activation('relu'))

    # livello 6
    model.add(Dropout(0.5))

    # livello 7
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=True, min_lr=0.001)

    ### risultato accuracy SENZA STRATIFY: accuracy: 0.9886 - loss: 0.0324 - val_accuracy: 0.9800 - val_loss: 0.0694
    ## risultato accuracy CON STRATIFY: accuracy: 0.9950 - loss: 0.0194 - val_accuracy: 0.9817 - val_loss: 0.1176
    # risultato accuracy CON LE ULTIME MODIFICHE 0.9932 - loss: 0.0273 - val_accuracy: 0.9717 - val_loss: 0.1055
    history = model.fit(x_train, y_train, 
                        batch_size=16, 
                        epochs=10,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping, reduce_lr])

    model.save('/tmp/BEST-MODEL-BINARY-CE.h5')
    
    