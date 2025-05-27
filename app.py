from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from flask_cors import CORS


app = Flask(__name__)

CORS(app)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="mobilenet_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example preprocessing: resize to 224x224 and normalize


# load class name
def load_labels(path='labels.txt'):
    with open(path, 'r') as f:
        return [line.strip() for line in f]


labels = load_labels()  # put this outside the route for performance


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))  # depends on your model's expected size
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image_data = image_file.read()
    input_data = preprocess_image(image_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class index and confidence
    predicted_index = int(np.argmax(output_data))
    confidence = float(output_data[0][predicted_index])

    # Load and get label (assumes you have labels.txt)
    predicted_label = labels[predicted_index]

    return jsonify({
        "label": predicted_label,
        "confidence": confidence
    })


if __name__ == '__main__':
    app.run(debug=True)
