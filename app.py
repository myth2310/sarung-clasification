from PIL import Image
from flask import Flask, request, jsonify, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import pygame

app = Flask(__name__)

# Memuat model yang telah dilatih
model = load_model('Dataset/trained_model.h5')

# Define the upload folder
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pygame.mixer.init()

# List of classes and corresponding sound files


# Menentukan ukuran target sesuai dengan bentuk input model
img_height, img_width = 150, 150

# Mendefinisikan kelas-kelas untuk prediksi
classes = {0: 'Batik Megamendung', 1: 'Batik Pekalongan'}  # Perbarui dengan nama kelas yang sesuai

# Membuat mapping antara kelas dan file suara
classes = {0: 'Batik Megamendung', 1: 'Batik Pekalongan'}  # Perbarui dengan nama kelas yang sesuai

# Membuat mapping antara kelas dan file suara
class_sound_mapping = {
    "Batik Megamendung": "static/sound/batik-megamendung.mp3",
    "Batik Pekalongan": "static/sound/batik-pekalongan.mp3"
}


def play_sound(predicted_class):
    sound_file = class_sound_mapping.get(predicted_class)
    if sound_file:
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()


@app.route('/realtime')
def realtime():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    cap = cv2.VideoCapture(1)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Konversi frame ke skala abu-abu untuk deteksi tepi
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Lakukan deteksi tepi pada frame
        edges = cv2.Canny(gray_frame, 50, 150)

        # Temukan kontur di dalam frame
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Pilih kontur terbesar
        largest_contour = max(contours, key=cv2.contourArea, default=())

        # Hanya pertimbangkan kontur dengan luas tertentu
        if cv2.contourArea(largest_contour) > 500:
            # Dapatkan bounding box dari kontur
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Ekstrak area dengan pola sarung
            sarong_pattern = frame[y:y + h, x:x + w]

            # Lakukan prediksi pada area dengan pola sarung
            sarong_pattern = cv2.resize(sarong_pattern, (img_width, img_height))
            sarong_pattern = sarong_pattern / 255.0
            sarong_pattern = np.expand_dims(sarong_pattern, axis=0)

            predictions = model.predict(sarong_pattern)
            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index]

            # Mendapatkan nama kelas yang sesuai dengan indeks
            predicted_class_name = classes.get(class_index, 'Unknown')

            # Menampilkan hasil prediksi di layar
            label = f"Class: {predicted_class_name}, {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mengubah frame menjadi format yang dapat ditampilkan di browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Kirim frame ke browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize the image data

# Route for the home page
@app.route('/')
def home():
    return render_template('image.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded file to the upload folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the uploaded image
        processed_image = preprocess_image(file_path)

        # Make a prediction
        prediction = model.predict(processed_image)
        predicted_class = classes[np.argmax(prediction)]

        # Play sound based on predicted class
        play_sound(predicted_class)

        # Render the result page with the uploaded image and prediction
        return render_template('image.html', filename=filename, prediction=predicted_class)



if __name__ == '__main__':
    app.run(debug=True)
