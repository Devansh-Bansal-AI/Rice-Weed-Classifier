import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# --- STEP 1: Define Constants and Paths ---

# The target image size used for training the model
IMG_SIZE = (224, 224)

# The path to your saved model file
MODEL_PATH = 'rice_weed_classifier_model.h5'

# The path to the single image you want to predict
IMAGE_TO_PREDICT_PATH = r'K:\DS_model\leaf2.jpg'

# The directory containing the class folders (e.g., your test data folder)
# We will use this to automatically determine the class names
DATASET_SPLIT_DIR = r'K:\Data_science\rice_weed_dataset_split\test'


# --- STEP 2: Load the Trained Model and Get Class Names ---

try:
    print("Loading the trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure 'rice_weed_classifier_model.h5' is in the same directory.")
    exit()

# Get the class names from the subdirectory names in your dataset
if not os.path.exists(DATASET_SPLIT_DIR):
    print(f"Error: The directory '{DATASET_SPLIT_DIR}' was not found.")
    exit()

CLASS_NAMES = sorted(os.listdir(DATASET_SPLIT_DIR))
print(f"Detected Classes: {CLASS_NAMES}")

# --- STEP 3: Define a Preprocessing Function ---

def preprocess_image(img_path):
    """
    Loads and preprocesses a single image for model prediction.
    """
    print(f"\nProcessing image: {img_path}")
    # Load the image and resize it to the target size
    img = image.load_img(img_path, target_size=IMG_SIZE)
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale pixel values (as done during training)
    img_array /= 255.0
    return img_array

# --- STEP 4: Predict a Single Image ---

def predict_single_image(img_path):
    """
    Predicts the class of a single, specified image.
    """
    if not os.path.exists(img_path):
        print(f"Error: The image file was not found at {img_path}")
        return

    # Preprocess the selected image
    processed_image = preprocess_image(img_path)

    # Make the prediction
    print("Making a prediction...")
    predictions = model.predict(processed_image)
    
    # Get the class with the highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    if predicted_class_index >= len(CLASS_NAMES):
        print("Prediction index is out of range for the detected classes.")
        return

    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    print("\n--- Prediction Result ---")
    print(f"File: {os.path.basename(img_path)}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("-------------------------")

# Run the prediction function with your image path
if __name__ == "__main__":
    predict_single_image(IMAGE_TO_PREDICT_PATH)