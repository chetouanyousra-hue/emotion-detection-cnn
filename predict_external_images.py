import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ==========================
# Charger le modèle
# ==========================

model = load_model('fer2013_emotion_model.h5')

# ==========================
# Liste des émotions
# ==========================

emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]

# ==========================
# Charger une image
# ==========================

image_path = 'test.jpg'

image = cv2.imread(image_path)

# convertir en gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# redimensionner
face = cv2.resize(gray, (48, 48))

# normalisation
face = face / 255.0

# reshape
face = np.reshape(face, (1, 48, 48, 1))

# ==========================
# prédiction
# ==========================

prediction = model.predict(face)

emotion_index = np.argmax(prediction)

emotion = emotion_labels[emotion_index]

print("Emotion détectée : ", emotion)

# ==========================
# afficher image
# ==========================

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f'Emotion : {emotion}')
plt.axis('off')
plt.show()