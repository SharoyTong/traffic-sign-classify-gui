import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained VGG model
vgg_model = load_model('vgg16_model.keras')  # Make sure the path is correct

# Get the class indices mapping from the training dataset
class_indices = {
    '0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, 
    '15': 7, '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, 
    '20': 13, '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, 
    '26': 19, '27': 20, '28': 21, '29': 22, '3': 23, '30': 24, 
    '31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30, 
    '37': 31, '38': 32, '39': 33, '4': 34, '40': 35, '41': 36, 
    '42': 37, '5': 38, '6': 39, '7': 40, '8': 41, '9': 42
}

classes = {str(k): v for k, v in { 
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
    9: 'No passing', 10: 'No passing vehicles over 3.5 tons', 11: 'Right-of-way at intersection', 
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Vehicles > 3.5 tons prohibited', 
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right', 
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing', 
    42: 'End no passing vehicles > 3.5 tons'
}.items()}


def predict_with_vgg(image_path):
    roi, predicted_class_name, confidence = predict_image_with_roi_and_padding(vgg_model, image_path, classes, class_indices, padding=20)
    return roi, predicted_class_name, confidence

# Function to find the largest contour and crop the ROI with padding
def extract_roi_with_padding(img_path, padding=20):
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or could not be opened: {img_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection (Canny)
    edged = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return the original image
    if len(contours) == 0:
        print("No contours found.")
        return img

    # Find the largest contour, assuming it's the traffic sign
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate new coordinates for the ROI with padding
    x_pad = max(0, x - padding)
    y_pad = max(0, y - padding)
    w_pad = min(img.shape[1], x + w + padding)
    h_pad = min(img.shape[0], y + h + padding)

    # Crop the image to the padded bounding box
    roi_with_padding = img[y_pad:h_pad, x_pad:w_pad]

    return roi_with_padding

# Function to preprocess ROI and predict class
def predict_image_with_roi_and_padding(model, img_path, class_names_mapping, class_indices, target_size=(32, 32), padding=20):
    # Extract the ROI with padding from the image
    roi_with_padding = extract_roi_with_padding(img_path, padding=padding)

    # Preprocess the cropped ROI for classification
    roi = cv2.cvtColor(roi_with_padding, cv2.COLOR_BGR2RGB)  # Convert to RGB
    roi = cv2.resize(roi, target_size)                       # Resize the ROI
    roi = roi.astype('float32') / 255.0                      # Normalize to [0, 1]
    roi = np.expand_dims(roi, axis=0)                        # Add batch dimension

    # Predict class from the ROI
    predictions = model.predict(roi)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]

    # Reverse class indices to original class
    index_to_class = {v: k for k, v in class_indices.items()}
    true_class_id = index_to_class[predicted_class_idx]
    predicted_class_name = class_names_mapping[true_class_id]

    return roi, predicted_class_name, confidence


