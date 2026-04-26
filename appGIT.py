from flask import Flask, render_template, jsonify, request
import base64
import cv2
import joblib
import torch
import numpy as np
import random
import os
import google.generativeai as genai

from model import SiameseNet

# =====================================
# CONFIG
# =====================================
API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)
gemini = genai.GenerativeModel("gemini-3-flash-preview")

IMG_SIZE = 224
DEVICE = "cpu"

app = Flask(__name__)

# =====================================
# LOAD MODELS
# =====================================
model = SiameseNet().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

svm = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
classes = joblib.load("classes.pkl")

# =====================================
# SENSOR STATE
# =====================================
current_temp = 20.0
current_hum = 70.0

# =====================================
# LAST RESULT
# =====================================
last_prediction = None
last_confidence = None
last_temp = None
last_humidity = None

# =====================================
def fake_sensor():
    global current_temp, current_hum

    current_temp += random.uniform(-0.2, 0.2)
    current_hum += random.uniform(-0.7, 0.7)

    current_temp = max(19.0, min(21.5, current_temp))
    current_hum = max(66.0, min(74.0, current_hum))

    return round(current_temp,1), round(current_hum,1)

# =====================================
def preprocess_bytes(img_bytes):

    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))

    return torch.tensor(img).unsqueeze(0).to(DEVICE)

# =====================================
def predict_image(img_bytes):

    img = preprocess_bytes(img_bytes)

    with torch.no_grad():
        emb = model.forward_once(img).cpu().numpy()

    emb = scaler.transform(emb)

    pred = svm.predict(emb)[0]
    probs = svm.predict_proba(emb)[0]

    label = classes[pred]
    conf = float(np.max(probs) * 100)

    return label, conf

# =====================================
@app.route("/")
def home():
    return render_template("indexGIT.html")

# =====================================
@app.route("/capture", methods=["POST"])
def capture():

    global last_prediction, last_confidence
    global last_temp, last_humidity

    try:
        file = request.files["image"]
        img_bytes = file.read()

        label, conf = predict_image(img_bytes)

        temp, hum = fake_sensor()

        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        last_prediction = label
        last_confidence = round(conf,2)
        last_temp = temp
        last_humidity = hum

        return jsonify({
            "prediction": label,
            "confidence": round(conf,2),
            "temperature": temp,
            "humidity": hum,
            "image": img_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# =====================================
@app.route("/advice")
def advice():

    global last_prediction, last_confidence
    global last_temp, last_humidity

    if last_prediction is None:
        return jsonify({"error":"Upload image first"})

    if last_prediction == "NOT_LEAF":
        return jsonify({
            "advice":"This does not appear to be a plant leaf. Please upload a clear leaf image."
        })

    prompt = f"""
You are an agricultural assistant.

Disease: {last_prediction}
Confidence: {last_confidence}%
Temperature: {last_temp} C
Humidity: {last_humidity} %

Give:
1. Cause
2. Immediate remedy
3. Prevention

Use simple language.
Keep under 90 words.
"""

    try:
        response = gemini.generate_content(prompt)
        return jsonify({"advice": response.text})

    except Exception as e:
        return jsonify({"error": str(e)})