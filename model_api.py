from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import uuid
from geopy.geocoders import Nominatim

# ---------------------------------------------------
# Flask App Configuration
# ---------------------------------------------------
app = Flask(__name__)
CORS(app)

# Load YOLOv8 model (same folder)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)

# Upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

geolocator = Nominatim(user_agent="manhole_api")

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def reverse_geocode(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), timeout=10)
        return location.address if location else "Unknown Location"
    except Exception:
        return "Unknown Location"

def predict_image(image_path):
    results = model(image_path)
    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            conf = float(box.conf)
            detections.append({"label": label, "confidence": round(conf, 3)})

    if detections:
        top = max(detections, key=lambda x: x["confidence"])
        return top["label"], top["confidence"]
    return "Unknown", 0.0

# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Model API is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")

    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_name)
        image.save(file_path)

        label, conf = predict_image(file_path)
        os.remove(file_path)

        address = None
        if latitude and longitude:
            address = reverse_geocode(latitude, longitude)

        return jsonify({
            "success": True,
            "condition": label,
            "confidence": conf,
            "address": address or "Unknown"
        }), 200

    return jsonify({"error": "Invalid file type"}), 400

# ---------------------------------------------------
# Run Flask App
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
