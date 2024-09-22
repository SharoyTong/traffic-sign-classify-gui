# models/svm_model.py
import joblib
import numpy as np
import cv2
from skimage.feature import hog
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained SVM model
svc = joblib.load("trained_svc_model.pkl")
scaler = joblib.load("scaler.pkl")


# Assume that names of the classes are the same as in your YOLO model
classes = { 
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)', 
    2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 
    4: 'Speed limit (70km/h)', 
    5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 
    7: 'Speed limit (100km/h)', 
    8: 'Speed limit (120km/h)', 
    9: 'No passing', 
    10: 'No passing veh over 3.5 tons', 
    11: 'Right-of-way at intersection', 
    12: 'Priority road', 
    13: 'Yield', 
    14: 'Stop', 
    15: 'No vehicles', 
    16: 'Veh > 3.5 tons prohibited', 
    17: 'No entry', 
    18: 'General caution', 
    19: 'Dangerous curve left', 
    20: 'Dangerous curve right', 
    21: 'Double curve', 
    22: 'Bumpy road', 
    23: 'Slippery road', 
    24: 'Road narrows on the right', 
    25: 'Road work', 
    26: 'Traffic signals', 
    27: 'Pedestrians', 
    28: 'Children crossing', 
    29: 'Bicycles crossing', 
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 
    32: 'End speed + passing limits', 
    33: 'Turn right ahead', 
    34: 'Turn left ahead', 
    35: 'Ahead only', 
    36: 'Go straight or right', 
    37: 'Go straight or left', 
    38: 'Keep right', 
    39: 'Keep left', 
    40: 'Roundabout mandatory', 
    41: 'End of no passing', 
    42: 'End no passing veh > 3.5 tons' 
}

# Function to map class number to description
def get_class_description(class_number):
    return classes.get(class_number, "Unknown class")


def predict_with_svm(image_path):
    processed_image, prediction = preprocess_image(image_path)
    return processed_image, prediction


def preprocess_image(image_path):
    # Load the image
    img = np.asarray(Image.open(image_path))

    # Ensure the image is in RGB format and handle any single-channel images
    if img.ndim == 2:  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # Image with alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Resize the image to 32x32
    img_resized = cv2.resize(img, (32, 32))

    # Convert to grayscale
    processed_image = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # Extract HOG features
    hog_features = hog(processed_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    # Scale the HOG features using the pre-loaded scaler
    hog_features_scaled = scaler.transform([hog_features])

    # Predict using the trained SVM model
    prediction = svc.predict(hog_features_scaled)[0]

    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    
    return processed_image, prediction
    
