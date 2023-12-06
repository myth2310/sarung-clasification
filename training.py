import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_dir = '/content/drive/MyDrive/Dataset Motif Sarung/train'
test_data_dir = '/content/drive/MyDrive/Dataset Motif Sarung/test'

# Definisikan jumlah kelas
num_classes = 3  # Ubah sesuai dengan jumlah kelas dalam dataset Anda

# Konfigurasi hyperparameter
batch_size = 32
epochs = 10
img_height, img_width = 150, 150

# Mempersiapkan data training dan testing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


# Membangun model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Evaluasi model
loss, accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {accuracy}')

# Simpan model ke dalam file
model.save('/content/drive/MyDrive/Dataset Motif Sarung/trained_model.h5')