import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


data_dir = "/kaggle/input/eye-dataset/Eye dataset/"
img_height, img_width = 150, 150
batch_size = 32


datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.1)

train_data = datagen.flow_from_directory(
    data_dir,target_size=(img_height, img_width),batch_size=batch_size, class_mode='categorical',subset='training' )
test_data = datagen.flow_from_directory(data_dir,target_size=(img_height, img_width),batch_size=batch_size,class_mode='categorical', subset='validation')


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),Conv2D(64, (3,3), activation='relu'),MaxPooling2D(2,2),Conv2D(128, (3,3), activation='relu'), MaxPooling2D(2,2),Flatten(),Dense(128, activation='relu'),
    Dropout(0.5),Dense(4, activation='softmax')])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


history = model.fit(train_data, validation_data=test_data, epochs=10)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluk Oranı')
plt.plot(history.history['val_accuracy'], label='Test Doğruluk Oranı')
plt.title('Doğruluk Oranı Grafiği ')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı Oranı')
plt.plot(history.history['val_loss'], label='Test Kaybı Oranı')
plt.title('Kayıp Oranı Grafiği')
plt.legend()
plt.show()


batch = next(test_data)
images, labels = batch
predictions = model.predict(images[:5])

class_names = list(train_data.class_indices.keys())

plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i])
    plt.title(f"Gerçek: {class_names[np.argmax(labels[i])]}\nTahmin: {class_names[np.argmax(predictions[i])]}")
    plt.axis('off')
plt.show()