import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

model = load_model('BEST-MODEL-BINARY-CE.h5')

image = cv2.imread('/workspaces/2023-25.BD.UFS14/MyProjFolder/pred/pred8.jpg')

# converto l'immagine in formato PIL
img = Image.fromarray(image)

img = img.resize((64, 64))

img = np.array(img)

# rendo compatibile img
input_img = np.expand_dims(img, axis=0) # axis=0 aggiunge un nuova dimensione all'inizio dell'array

# normalizzo l'immagine
input_img = input_img / 255.0

# ottengo le probabilità delle classi
result = model.predict(input_img)

# soglia per la classificazione
predicted_class = (result > 0.8).astype("int32")

# 0 no tumore, 1 tumore
print(predicted_class)