from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import gdown

app = Flask(__name__)

MODEL_PATH = "ResNet152V2-AIvsHumanGenImages.keras"
MODEL_URL = "https://drive.google.com/uc?id=1MPuzeNe1xu4GTolnbRoMC7BAV9eEYF_k"

# Download model if missing
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)

def prepare_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((512, 512))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/staticuploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('staticuploads', filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_url = None
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)
        file = request.files['file']
        upload_folder = "staticuploads"
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        img_array = prepare_image(filepath)
        pred = model.predict(img_array)
        prediction = "AI Generated" if pred[0][0] > 0.5 else "Human Created"
        img_url = url_for('uploaded_file', filename=file.filename)

    return render_template('index.html', prediction=prediction, img_url=img_url)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
