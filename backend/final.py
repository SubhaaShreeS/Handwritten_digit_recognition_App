'''
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("mnist_plus_custom_model_final.keras")

# Create folder to save split digit images
SAVE_DIR = "split_digits"
os.makedirs(SAVE_DIR, exist_ok=True)


def ensure_black_digit_white_bg(pil_img):
    """Ensure black digit on white background (MNIST style)."""
    gray = np.array(pil_img.convert('L'))
    mean_val = np.mean(gray)
    if mean_val < 127:
        gray = 255 - gray
    return Image.fromarray(gray)


def is_single_digit(image_array):
    """Use contour count to check if image has a single digit."""
    bin_img = (image_array < 127).astype(np.uint8)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) == 1


def preprocess_single_digit(pil_img):
    """Preprocess a single digit image."""
    image = pil_img.convert("L")
    img = np.array(image)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(img)
    if coords is None:
        raise ValueError("No digit found")
    
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]

    # Make square
    side = max(w, h)
    square = np.zeros((side, side), dtype=np.uint8)
    x_offset = (side - w) // 2
    y_offset = (side - h) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

    # Resize to 28x28
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    img_array = resized / 255.0
    return img_array.reshape(1, 28, 28, 1)


def resize_to_mnist(digit_img):
    """Resize image to 28x28 with padding to maintain aspect ratio."""
    h, w = digit_img.shape
    size = max(h, w)
    square = np.ones((size, size), dtype=np.uint8) * 255

    if h > w:
        pad = (h - w) // 2
        square[:, pad:pad + w] = digit_img
    else:
        pad = (w - h) // 2
        square[pad:pad + h, :] = digit_img

    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    return resized


def preprocess_and_segment(pil_img):
    """Segment multiple digits and return list of 28x28 images."""
    img = np.array(pil_img.convert('L'))
    if np.mean(img) < 127:
        img = 255 - img

    # Threshold + denoise
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.medianBlur(img, 3)

    # Dilation to connect fragments
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # Vertical projection to find digit splits
    vertical_sum = np.sum(img == 0, axis=0)
    split_points = []
    in_digit = False
    start = 0

    for i, val in enumerate(vertical_sum):
        if val > 0 and not in_digit:
            start = i
            in_digit = True
        elif val == 0 and in_digit:
            end = i
            if end - start >= 5:  # Filter small blobs
                split_points.append((start, end))
            in_digit = False

    digit_images = []
    for idx, (x_start, x_end) in enumerate(split_points):
        digit = img[:, x_start:x_end]
        rows = np.any(digit == 0, axis=1)
        digit = digit[np.where(rows)[0], :]
        if digit.size == 0:
            continue

        digit_resized = resize_to_mnist(digit)
        digit_images.append(digit_resized)

        # Save for debugging
        save_path = os.path.join(SAVE_DIR, f'split_digit_{idx}.png')
        Image.fromarray(digit_resized).save(save_path)

    return digit_images


def predict_digits(images):
    digits = []
    confidences = []

    for img in images:
        norm_img = img.astype('float32') / 255.0
        input_img = norm_img.reshape(1, 28, 28, 1)
        pred = model.predict(input_img)
        digit = int(np.argmax(pred))
        confidence = round(float(np.max(pred)), 4)

        digits.append(str(digit))
        confidences.append(confidence)

    return ''.join(digits), confidences


@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files.get('image') or request.files.get('file')
        pil_image = Image.open(file.stream)
        cleaned_image = ensure_black_digit_white_bg(pil_image)
        img_arr = np.array(cleaned_image.convert("L"))

        if is_single_digit(img_arr):
            input_tensor = preprocess_single_digit(cleaned_image)
            pred = model.predict(input_tensor)
            digit = int(np.argmax(pred))
            confidence = round(float(np.max(pred)), 4)

            return jsonify({
                "prediction": str(digit),
                "confidence": confidence,
                "type": "single"
            })

        digit_images = preprocess_and_segment(cleaned_image)
        if not digit_images:
            return jsonify({'error': 'No digits found'}), 400

        prediction, confidences = predict_digits(digit_images)

        return jsonify({
            'prediction': prediction,
            'confidences': confidences,
            'type': 'multi',
            'message': f'{len(digit_images)} digits split and saved to "{SAVE_DIR}/"'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9009, debug=True)'''

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("mnist_plus_custom_model_final.keras")

# Create folder to save split digit images
SAVE_DIR = "split_digits"
os.makedirs(SAVE_DIR, exist_ok=True)


def ensure_black_digit_white_bg(pil_img):
    """Ensure black digit on white background (MNIST style)."""
    gray = np.array(pil_img.convert('L'))
    mean_val = np.mean(gray)
    if mean_val < 127:
        gray = 255 - gray
    return Image.fromarray(gray)


def is_single_digit(image_array):
    """Use contour count to check if image has a single digit."""
    bin_img = (image_array < 127).astype(np.uint8)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) == 1


def preprocess_single_digit(pil_img):
    """Preprocess a single digit image."""
    image = pil_img.convert("L")
    img = np.array(image)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(img)
    if coords is None:
        raise ValueError("No digit found")

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]

    # Make square
    side = max(w, h)
    square = np.zeros((side, side), dtype=np.uint8)
    x_offset = (side - w) // 2
    y_offset = (side - h) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

    # Resize to 28x28
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    img_array = resized / 255.0
    return img_array.reshape(1, 28, 28, 1)


def resize_to_mnist(digit_img):
    """Resize image to 28x28 with padding to maintain aspect ratio."""
    h, w = digit_img.shape
    size = max(h, w)
    square = np.ones((size, size), dtype=np.uint8) * 255

    if h > w:
        pad = (h - w) // 2
        square[:, pad:pad + w] = digit_img
    else:
        pad = (w - h) // 2
        square[pad:pad + h, :] = digit_img

    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    return resized


def preprocess_and_segment(pil_img):
    """Segment multiple digits and return list of 28x28 images."""
    img = np.array(pil_img.convert('L'))
    if np.mean(img) < 127:
        img = 255 - img

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.medianBlur(img, 3)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    vertical_sum = np.sum(img == 0, axis=0)
    split_points = []
    in_digit = False
    start = 0

    for i, val in enumerate(vertical_sum):
        if val > 0 and not in_digit:
            start = i
            in_digit = True
        elif val == 0 and in_digit:
            end = i
            if end - start >= 5:
                split_points.append((start, end))
            in_digit = False

    digit_images = []
    for idx, (x_start, x_end) in enumerate(split_points):
        digit = img[:, x_start:x_end]
        rows = np.any(digit == 0, axis=1)
        digit = digit[np.where(rows)[0], :]
        if digit.size == 0:
            continue

        digit_resized = resize_to_mnist(digit)
        digit_images.append(digit_resized)

        save_path = os.path.join(SAVE_DIR, f'split_digit_{idx}.png')
        Image.fromarray(digit_resized).save(save_path)

    return digit_images


def predict_single_digit(pil_img):
    """Predict a single digit from PIL image using model."""
    input_tensor = preprocess_single_digit(pil_img)
    pred = model.predict(input_tensor)
    digit = int(np.argmax(pred))
    confidence = round(float(np.max(pred)), 4)
    return digit, confidence


@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        pil_image = Image.open(file.stream)
        cleaned_image = ensure_black_digit_white_bg(pil_image)
        img_arr = np.array(cleaned_image.convert("L"))

        # SINGLE digit
        if is_single_digit(img_arr):
            digit, confidence = predict_single_digit(cleaned_image)
            return jsonify({
                "prediction": str(digit),
                "confidence": confidence,
                "type": "single"
            })

        # MULTI digit
        digit_images = preprocess_and_segment(cleaned_image)
        if not digit_images:
            return jsonify({'error': 'No digits found'}), 400

        predictions = []
        confidences = []

        for idx, img_arr in enumerate(digit_images):
            pil_digit = Image.fromarray(img_arr)
            try:
                digit, confidence = predict_single_digit(pil_digit)
                predictions.append(str(digit))
                confidences.append(confidence)
            except Exception:
                predictions.append("?")
                confidences.append(0.0)

        return jsonify({
            'prediction': ''.join(predictions),
            'confidences': confidences,
            'type': 'multi',
            'message': f'{len(predictions)} digits processed using single-digit logic'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9009, debug=True)
