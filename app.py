from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/mnist_model.keras")


def preprocess_image(image):
    """
    Preprocess image to match MNIST format.
    """

    # Convert to grayscale
    image = image.convert("L")

    # Invert colors (MNIST = white digit on black background)
    image = ImageOps.invert(image)

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array
    img = np.array(image)

    # Normalize
    img = img.astype("float32") / 255.0

    # Flatten
    img = img.reshape(1, 784)

    return img


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]

    image = Image.open(io.BytesIO(file.read()))

    processed = preprocess_image(image)

    prediction = model.predict(processed)

    digit = int(np.argmax(prediction))

    confidence = float(np.max(prediction) * 100)

    return jsonify({
        "digit": digit,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)