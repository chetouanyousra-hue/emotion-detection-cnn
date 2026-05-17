
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

print("Chargement du dataset...")

# ==========================
# Chargement du dataset
# ==========================

emotion_data = pd.read_csv('fer2013.csv')

print(emotion_data.head())

# ==========================
# Préparation des données
# ==========================

pixels = emotion_data['pixels'].tolist()

X = []

for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(48, 48)
    X.append(face)

X = np.asarray(X)

# normalisation
X = X / 255.0

# reshape
X = X.reshape(X.shape[0], 48, 48, 1)

# labels
Y = emotion_data['emotion']

# one hot encoding
Y = to_categorical(Y, num_classes=7)

# séparation train / test
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("Train shape : ", X_train.shape)
print("Test shape : ", X_test.shape)

# ==========================
# Construction du modèle CNN
# ==========================

model = Sequential()

# couche convolutionnelle 1
model.add(Conv2D(
    32,
    (3, 3),
    activation='relu',
    input_shape=(48, 48, 1)
))

model.add(MaxPooling2D(pool_size=(2, 2)))

# couche convolutionnelle 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# couche convolutionnelle 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten
model.add(Flatten())

# dense
model.add(Dense(128, activation='relu'))

# dropout
model.add(Dropout(0.5))

# sortie
model.add(Dense(7, activation='softmax'))

# ==========================
# Compilation
# ==========================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================
# Entraînement
# ==========================

history = model.fit(
    X_train,
    Y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, Y_test)
)

# ==========================
# Sauvegarde du modèle
# ==========================

model.save('fer2013_emotion_model.h5')

print("Modèle sauvegardé avec succès")

# ==========================
# Évaluation
# ==========================

loss, accuracy = model.evaluate(X_test, Y_test)

print("Accuracy : ", accuracy)

# prédictions
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

print("Classification Report")
print(classification_report(Y_true, Y_pred_classes))

print("F1 Score : ", f1_score(Y_true, Y_pred_classes, average='weighted'))

print("Confusion Matrix")
print(confusion_matrix(Y_true, Y_pred_classes))

# ==========================
# Graphiques
# ==========================

plt.figure(figsize=(10, 5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()

# ==========================
# SHAP Visualisation
# ==========================

print("Calcul SHAP...")

background = X_train[:100]

test_images = X_test[:5]

explainer = shap.DeepExplainer(model, background)

shap_values = explainer.shap_values(test_images)

shap.image_plot(shap_values, test_images)
