# models/yolo_model.py
from ultralytics import YOLO

model = YOLO("best_YOLOv8_batch16.pt")

names = ['Speed limit 20km/h', 'Speed limit 30km/h', 'Speed limit 50km/h', 'Speed limit 60km/h', 'Speed limit 70km/h',
         'Speed limit 80km/h', 'End of speed limit 80km/h', 'Speed limit 100km/h', 'Speed limit 120km/h', 'No passing',
         'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 'Priority road',
         'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution',
         'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road',
         'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
         'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
         'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
         'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
         'End of no passing by vehicles over 3.5 metric tons']

def predict_with_yolo(image_path):
    results = model(image_path)
    best_prediction = results[0].probs.top1
    best_confidence = results[0].probs.top1conf.item()
    class_names = results[0].names
    return class_names[best_prediction], best_confidence