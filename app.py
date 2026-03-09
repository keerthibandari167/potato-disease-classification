from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model('potato-disease-model-transfer.h5')

class_names = ['Early_blight', 'Healthy', 'Late_blight']

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type! Upload JPG or PNG.")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds[0])
    confidence = np.max(preds[0])
    predicted_class = class_names[predicted_index]

    if confidence < 0.7:
        return render_template('index.html',
                           filename=file.filename,
                           error="This does not look like a potato leaf!")
    return render_template('index.html',
                           filename=file.filename,
                           predicted_class=predicted_class,
                           confidence=f"{confidence*100:.2f}%")
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)