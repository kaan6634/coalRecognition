from flask import Flask, request, render_template, send_from_directory, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = tf.keras.models.load_model('coal_recognition_model.h5')

def preprocess_image(image, target_size):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

class_names = {0: "antrasit komuru", 1: "linyit kömürü", 2: "taş kömürü"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    predictions = []
    files = request.files.getlist('file')
    for file in files:
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            image = Image.open(filename)
            processed_image = preprocess_image(image, target_size=(224, 224))
            prediction = model.predict(processed_image)
            predicted_class_indices = np.argmax(prediction, axis=1)
            for idx in predicted_class_indices:
                predicted_class_name = class_names.get(idx, "Unknown")
                confidence = prediction[0][idx]
                predictions.append({
                    'class_name': predicted_class_name,
                    'confidence': f"{confidence:.4f}",
                    'image_url': url_for('uploaded_file', filename=file.filename)
                })

    return render_template('index.html', predictions=predictions)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
