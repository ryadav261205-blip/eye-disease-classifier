"""
CNN Model Architecture for Eye Disease Classification.
Uses TensorFlow/Keras to build, compile, save, load, and run inference.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import INPUT_SHAPE, LEARNING_RATE, DROPOUT_RATE, MODEL_PATH, CLASS_NAMES
from preprocessing import preprocess_image_for_prediction


def build_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = LEARNING_RATE,
    dropout_rate: float = DROPOUT_RATE,
) -> keras.Model:
    """
    Build and compile the Convolutional Neural Network (CNN).

    Architecture:
    1. Input Layer (28 x 28 x 3)
    2. Conv2D (16 filters, 3x3 kernel, ReLU, same padding)
    3. MaxPooling2D (2x2 pool size)
    4. Conv2D (32 filters, 3x3 kernel, ReLU, same padding)
    5. MaxPooling2D (2x2 pool size)
    6. Flatten
    7. Dense (64 units, ReLU)
    8. Dropout (rate: dropout_rate)
    9. Dense (1 unit, Sigmoid activation)

    Loss: Binary Cross-Entropy
    Optimizer: Adam
    Metrics: Accuracy
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape, name="input_layer"),
            layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu", name="conv2d_1"),
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv2d_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),
            layers.Flatten(name="flatten"),
            layers.Dense(64, activation="relu", name="dense_64"),
            layers.Dropout(dropout_rate, name="dropout"),
            layers.Dense(1, activation="sigmoid", name="output_sigmoid"),
        ],
        name="EyeDiseaseCNN",
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.BinaryCrossentropy()
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def save_trained_model(model: keras.Model, filepath: Union[str, Path] = MODEL_PATH) -> Path:
    """
    Save the trained model to disk in native Keras format (.keras).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return path


def load_trained_model(filepath: Union[str, Path] = MODEL_PATH) -> keras.Model:
    """
    Load a pre-trained Keras model from disk.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. Please train the model first by running 'python train.py'."
        )
    return keras.models.load_model(str(path))


def predict_disease(model: keras.Model, image_input: Any, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Run single-image prediction through the model:
    1. Preprocesses the image into shape (1, 28, 28, 3).
    2. Runs inference.
    3. Interprets the sigmoid score.

    Returns:
        dict with:
            - 'probability': float in [0.0, 1.0]
            - 'class_id': int (0 for Normal, 1 for Diabetic Retinopathy)
            - 'class_name': str ("Normal" or "Diabetic Retinopathy")
            - 'confidence': float in [0.5, 1.0] (confidence in the predicted class)
            - 'explanation': str explaining the sigmoid output
    """
    input_batch = preprocess_image_for_prediction(image_input)
    sigmoid_score = float(model.predict(input_batch, verbose=0)[0][0])

    if sigmoid_score >= threshold:
        predicted_class_id = 1
        predicted_class_name = CLASS_NAMES[1]  # Diabetic Retinopathy
        confidence = sigmoid_score
        explanation = (
            f"Sigmoid score is {sigmoid_score:.4f} (≥ {threshold}), indicating visual markers "
            f"associated with Diabetic Retinopathy with {confidence * 100:.2f}% model confidence."
        )
    else:
        predicted_class_id = 0
        predicted_class_name = CLASS_NAMES[0]  # Normal
        confidence = 1.0 - sigmoid_score
        explanation = (
            f"Sigmoid score is {sigmoid_score:.4f} (< {threshold}), indicating fundus features "
            f"classified as Normal retinal appearance with {confidence * 100:.2f}% model confidence."
        )

    return {
        "probability": sigmoid_score,
        "class_id": predicted_class_id,
        "class_name": predicted_class_name,
        "confidence": confidence,
        "explanation": explanation,
    }

