import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

cnn_model = load_model('cnn_trained_model.keras')

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


def predict_with_cnn(image_path):
    preprocessed_image = preprocess_image(image_path, padding_factor=0.2)
    
    if preprocessed_image is not None:
        predictions = cnn_model.predict(preprocessed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_label = classes[predicted_class_index]
        prediction_probability = np.max(predictions)
        
        return predicted_label, prediction_probability, preprocessed_image[0]
    else:
        #print("Prediction could not be made as no ROI was detected.")
        return None, None, None
    

def preprocess_image(image_path, padding_factor=0.2):
    image = cv2.imread(image_path)
    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        pad_w = int(w * padding_factor)
        pad_h = int(h * padding_factor)
        
        x_start = max(0, x - pad_w)
        y_start = max(0, y - pad_h)
        x_end = min(image.shape[1], x + w + pad_w)
        y_end = min(image.shape[0], y + h + pad_h)
        
        roi = image[y_start:y_end, x_start:x_end]
        
        desired_size = 32
        original_size = roi.shape[:2]
        ratio = min(desired_size / original_size[0], desired_size / original_size[1])
        new_size = (int(original_size[1] * ratio), int(original_size[0] * ratio))
        resized_image = cv2.resize(roi, new_size, interpolation=cv2.INTER_AREA)
        
        delta_w = desired_size - new_size[0]
        delta_h = desired_size - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        padded_image = padded_image / 255.0

        image = np.expand_dims(padded_image, axis=0)
        
        return image
    else:
        print("No contour found. Returning original resized image.")
        return None


