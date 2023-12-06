from google.colab import files
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Mengunggah gambar
uploaded = files.upload()

# Daftar nama kelas yang sesuai dengan indeks
class_names = train_generator.class_indices

for fn in uploaded.keys():

  # Menampilkan gambar yang diunggah
  path = '/content/' + fn
  img = image.load_img(path, target_size=(img_height, img_width))
  plt.imshow(img)
  plt.axis('off')
  plt.show()

  # Praproses gambar
  img = image.load_img(path, target_size=(img_height, img_width))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = img_array / 255.0

  # Melakukan prediksi
  predictions = model.predict(img_array)
  predicted_class = np.argmax(predictions, axis=1)

  # Mendapatkan nama kelas yang sesuai dengan indeks
  predicted_class_name = [k for k, v in class_names.items() if v == predicted_class][0]

  # Menampilkan hasil prediksi
  print(f'Predicted class: {predicted_class_name}')
