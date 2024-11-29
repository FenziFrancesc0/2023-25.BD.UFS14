import azure.functions as func
import datetime
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

app = func.FunctionApp()

@app.route(route="MyHttpTrigger", auth_level=func.AuthLevel.ANONYMOUS)
def MyHttpTrigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    # main
    x_train, y_train = preprocess_data()

    cnn_model = CNN()
    
    cnn_model.build()
    cnn_model.compile()
    cnn_model.fit(x_train, y_train)

    cnn_model.save_weights("CNN.weights.h5")
    # fine main
    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "ciao",
             status_code=200
        )

class CNN:
    def __init__(self) -> None:
        self.class_array = np.load("data/class.npy")
        self.num_classes = len(self.class_array)

        self.model = models.Sequential([
            # livello 1
            layers.Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=(28, 28, 1)),
            # livello 2 - riduce la dimensionalità
            layers.MaxPooling2D(pool_size=(2, 2)),
            # livello 3
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            # livello 4: - riduce la dimensionalità
            layers.MaxPooling2D(pool_size=(2, 2)),
            # livello 5
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            # livello 6
            layers.Flatten(),
            # livello 7
            layers.Dense(128, activation='relu'),
            # livello 9
            layers.Dense(self.num_classes, activation='softmax')  # 345 classi
        ])
    
    def build(self):
        self.model.build()
    
    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(self, x_train, y_train):
        # Salva la history del training
        history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=2,  
            batch_size=32,
            validation_split=0.2
        )
        
        self.plot_history(history)

    def plot_history(self, history):
        # grafico dell'accuracy
        plt.figure(figsize=(12, 4))
        
        # grafico dell'accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # grafico loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
    
    def save_weights(self, path):
        self.model.save_weights(path)
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def predict(self, x):
        prediction = self.model.predict(x, verbose=0)[0]
        top = np.argsort(prediction)[-5:][::-1]
        top_encoded = self.class_array[top]
        return prediction, top_encoded
    
def preprocess_data():
    class_array = np.load("data/class.npy")
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")
    
    # mapping delle etichette
    class_dict = {word: idx for idx, word in enumerate(class_array)}
    y_train_mapped = np.array([class_dict[label] for label in y_train])

    x_train = x_train[:160000].astype(np.float32) / 255.0  # normalizzazione
    x_train = np.expand_dims(x_train, axis=-1)  # aggiungo del canale
    y_train_mapped = y_train_mapped[:160000]

    return x_train, y_train_mapped

# risultato con dropout 0.5: accuracy: 0.6989 - loss: 1.0834
# risultato senza dropout 0.5: accuracy: 0.7971 - loss: 0.7322 - val_accuracy: 0.7106 - val_loss: 1.0956
# risultato aggiornamento filtri: accuracy: 0.8258 - loss: 0.6312 - val_accuracy: 0.7191 - val_loss: 1.0796
