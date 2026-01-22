import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
import os
import tensorflow as tf
import numpy as np


def extract_characters(equation_img):
    # Get the image
    # equation_img = cv2.imread(equation_img_path, 0)  # specifiy the flag to be color
    # equation_img_gray = cv2.cvtColor(equation_img, cv2.COLOR_BGR2GRAY) #We could have directly loaded it gray

    # Get the threshold image
    _, img_thresh = cv2.threshold(equation_img, 100, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    img_thresh_cleaned = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Get image contours and sort them from left to right
    img_thresh_contours, _ = cv2.findContours(img_thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_thresh_contours = sorted(img_thresh_contours, key=lambda c: cv2.boundingRect(c)[0])

    # Store segmented characters
    segmented_chars = []
    bounding_boxes = []

    for i, contour in enumerate(img_thresh_contours):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out very small contours (noise)
        if w < 10 or h < 10:
            continue

        # Add some padding
        padding = 5
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(img_thresh.shape[1] - x_pad, w + 2 * padding)
        h_pad = min(img_thresh.shape[0] - y_pad, h + 2 * padding)

        # Extract the character
        char_img = img_thresh[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        segmented_chars.append(char_img)
        bounding_boxes.append((x_pad, y_pad, w_pad, h_pad))

    return segmented_chars

def load_models():
    # Load the model
    digit_model = tf.keras.models.load_model('./model/digit_recognizer.h5')
    operator_model = tf.keras.models.load_model('./model/operator_recognizer.h5')
    return digit_model,operator_model


def predict_chars(char, digit_model, operator_model, log=False):
    operators = ['+', '-', '*', '/']
    # Format the image for the DNN: Resize to 28x28 and normalize
    char_resized = cv2.resize(char, (28, 28))
    img_array = np.array(char_resized).reshape(-1, 28, 28, 1)  # Reshape for CNN (add channel dimension)
    img_array = img_array / 255  # Normalize pixel values (0-255 -> 0-1)

    # Digit model
    digit_pred = digit_model.predict(img_array)
    digit_predicted_class = np.argmax(digit_pred, axis=1)
    digit_pred_confidence = digit_pred[0][digit_predicted_class[0]]

    # Operator model
    operator_pred = operator_model.predict(img_array)
    operator_predicted_class = np.argmax(operator_pred, axis=1)
    operator_pred_confidence = operator_pred[0][operator_predicted_class[0]]

    if log:
        print(
            f"Prediction of operators model: {operators[operator_predicted_class[0]]} [{operator_pred_confidence:.4f}]")
        print(f"Prediction of dig model: {digit_predicted_class[0]} [{digit_pred_confidence:.4f}]")

    if operator_pred_confidence >= digit_pred_confidence:
        return operators[operator_predicted_class[0]]
    else:
        return str(digit_predicted_class[0])

def solve_equation(equation_string):
    try:
        # Use eval (safe for simple math expressions)
        result = eval(equation_string)
        return f"{equation_string} = {result}"
    except:
        return f"Error: {equation_string}"


def ocr_pipeline(equation_img,digit_model,operator_model):
    segmented_chars = extract_characters(equation_img)
    characters = []
    for char in segmented_chars:
        characters.append(predict_chars(char, digit_model, operator_model, log=True))

    return solve_equation(" ".join(characters))