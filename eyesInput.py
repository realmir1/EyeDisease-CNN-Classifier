import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


veriyolu = "/kaggle/input/eye-dataset/Eye dataset"

image_size = (150, 150)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
train_generator = train_datagen.flow_from_directory(
    veriyolu,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    veriyolu,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(train_generator, validation_data=validation_generator, epochs=10)


model.save("image_classifier.h5")


def gercek_deger(image_path, model, class_indices):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_label = class_labels[predicted_class]
    
    plt.imshow(img)
    plt.title(f"Tahmin: {predicted_label}")
    plt.axis("off")
    plt.show()




    #Bu yapay zeka modelinde Adam algoritması kullandım. Ali emir sertbaş e-mail: sertbasali08@gmail.com. 
    # © 2025 - Tüm Hakları Saklıdır. 