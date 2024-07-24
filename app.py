from flask import Flask, request, render_template_string, send_from_directory, url_for
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

html_template = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coal Recognition</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f9;
            color: #333;
            margin: 0;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #444;
        }
        form {
            max-width: 500px;
            margin: 20px auto;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        input[type="file"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        input[type="submit"] {
            display: block;
            width: 100%;
            padding: 10px;
            background-color: #5cb85c;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #4cae4c;
        }
        .image-container {
            max-width: 500px;
            margin: 20px auto;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 10px 0;
            border-radius: 4px;
        }
        .image-container h2 {
            color: #5cb85c;
        }
        .image-container p {
            color: #555;
        }
    </style>
</head>
<body>
    <h1>Upload Coal Images for Recognition</h1>
    <form method="post" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file" multiple accept="image/*">
        <input type="submit" value="Upload">
    </form>
    {% if predictions %}
        {% for result in predictions %}
            <div class="image-container">
                <h2>Predicted Class: {{ result.class_name }}</h2>
                <p>Confidence: {{ result.confidence }}</p>
                <img src="{{ result.image_url }}" alt="Uploaded Image">
            </div>
        {% endfor %}
    {% endif %}
</body>
</html>

'''


@app.route('/')
def index():
    return render_template_string(html_template)


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

    return render_template_string(html_template, predictions=predictions)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
