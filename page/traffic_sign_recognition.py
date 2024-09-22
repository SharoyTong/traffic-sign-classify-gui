# pages/traffic_sign_recognition.py
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import os
import cv2
from models.yolo_model import predict_with_yolo, names
from models.svm_model import predict_with_svm, get_class_description
from models.cnn_model import predict_with_cnn
from models.vgg_model import predict_with_vgg

def show():


    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        st.image('static/logo.png', width=256)

    
    # Centering and italicizing the welcome message
    st.markdown(
        """
        <div style="text-align: center; font-size:24px; font-style: italic;">
            Welcome to <strong>Traffic Vision Sign Recognition System</strong> powered by AI !  
        </div>
        <div style="text-align: center; font-size:24px; font-style: italic;">
            This system helps in identifying and predicting traffic signs using multiple AI models.
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("""
       <style>
       
       div.stMarkdown p, div.stMarkdown li {
            font-size: 18px;
        }
       
       </style>
   """, unsafe_allow_html=True)

    # Regular instructions without changes
    st.markdown(
        """
        **Instructions:**
        1. Upload an image of a traffic sign.
        2. Preview the image.
        3. Choose the prediction model (or choose all).
        4. Click 'Predict' to see the results.
        5. Conclusion and Results Discussion
        """
    )
    
    
    st.header("Step 1: Upload Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        display_image_preview(uploaded_image)
        handle_prediction(uploaded_image)


def display_image_preview(uploaded_image):
    st.header("Step 2: Image Preview")
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Traffic Sign", use_column_width=True)

def handle_prediction(uploaded_image):
    st.header("Step 3: Prediction")
    model_options = ["You Only Look Once (YOLOv8)", "Convolutional Neural Network (CNN)", "Support Vector Machine (SVM)", "Visual Geometric Group (VGG)", "I WANT ALL"]
    selected_model = st.selectbox("Select the AI model for prediction", model_options)

    if st.button("Predict"):
        st.write(f"Using the **{selected_model}** model for prediction.")
        st.header("Step 4: Tadaaa..! Here is your result !")
        results_displayed = False

        if selected_model in ["You Only Look Once (YOLOv8)", "I WANT ALL"]:
            with st.spinner('Predicting...'):
                predict_with_yolo_model(uploaded_image)
                results_displayed = True

        if selected_model in ["Support Vector Machine (SVM)", "I WANT ALL"]:
            with st.spinner('Predicting with SVM...'):
                predict_with_svm_model(uploaded_image)
                results_displayed = True

        if selected_model in ["Convolutional Neural Network (CNN)", "I WANT ALL"]:
            with st.spinner('Predicting with CNN...'):
                predict_with_cnn_model(uploaded_image)
                results_displayed = True
                
        if selected_model in ["Visual Geometric Group (VGG)", "I WANT ALL"]:
            with st.spinner('Predicting with VGG...'):
                predict_with_vgg_model(uploaded_image)
                results_displayed = True
                
        if results_displayed:
            if st.button("Upload Another Image"):
                st.experimental_rerun()

def predict_with_yolo_model(uploaded_image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_image.getvalue())
        tmp_file_path = tmp_file.name

    try:
        prediction, confidence = predict_with_yolo(tmp_file_path)
        display_prediction_result(prediction, confidence, "YOLOv8")
    finally:
        os.unlink(tmp_file_path)
        

def predict_with_svm_model(uploaded_image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_image.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Pass the tmp_file_path to the SVM predict function
        processed_image, prediction = predict_with_svm(tmp_file_path)  
        class_name = get_class_description(prediction)
        display_svm_prediction_results(processed_image, prediction, class_name)
        return processed_image, prediction, class_name
    finally:
        os.unlink(tmp_file_path)  # Clean up the temporary file


def predict_with_cnn_model(uploaded_image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_image.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Pass the tmp_file_path to the predict function
        predicted_label, prediction_probability, processed_image = predict_with_cnn(tmp_file_path)
        display_cnn_prediction_result(predicted_label, prediction_probability, processed_image)
    finally:
        os.unlink(tmp_file_path)  # Clean up the temporary file
   
    
def predict_with_vgg_model(uploaded_image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_image.getvalue())
        tmp_file_path = tmp_file.name

    try:
        roi, predicted_class_name, confidence = predict_with_vgg(tmp_file_path)
        display_vgg_prediction_result(roi, predicted_class_name, confidence)
    finally:
        os.unlink(tmp_file_path)
        
        
def subheader(text):
    st.markdown(
        f"""
        <h5 style='background-color: #AEC8FF; padding: 10px; border-radius: 5px; color: black;'>
            {text}
        </h5>
        """, 
        unsafe_allow_html=True
    )
    

def display_prediction_result(prediction, confidence, model_name):
    subheader("You Only Look Once (YOLOv8) Model")
    st.markdown("<br>", unsafe_allow_html=True) 
    confidence_percent = confidence * 100
    index = names.index(prediction)
    class_image = f'traffic_signs/{index}.png'
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(class_image, caption="Predicted Sign")
    with col2:
        st.success(f"{model_name} Prediction: '{prediction}'")
        st.info(f"Confidence: {confidence_percent:.2f}%")
        

def display_cnn_prediction_result(predicted_label, prediction_probability, processed_image):
    subheader("Convolutional Neural Network (CNN) Model")
    st.markdown("<br>", unsafe_allow_html=True) 
    if predicted_label is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Display the processed image
            processed_image_uint8 = (processed_image * 255).astype(np.uint8)
            st.image(cv2.cvtColor(processed_image_uint8, cv2.COLOR_BGR2RGB), caption="Processed Image", width=128)
        
        with col2:
            # Display prediction results
            st.success(f"CNN Predicted class: {predicted_label}")
            st.info(f"Confidence: {prediction_probability * 100:.2f}%")
            
    else:
        st.write("Prediction could not be made as no ROI was detected.")
        
        
def display_vgg_prediction_result(processed_image, predicted_class_name, confidence):
    subheader("Visual Geometric Group (VGG) Model")
    st.markdown("<br>", unsafe_allow_html=True) 
    if predicted_class_name is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Display the processed image in Streamlit
            st.image(processed_image, caption="Processed Image", width=128)
        
        with col2:
            # Display the prediction results
            st.success(f"VGG Predicted class: {predicted_class_name}")
            st.info(f"Confidence: {confidence * 100:.2f}%")
            
    else:
        st.warning("Prediction could not be made as no Region of Interest (ROI) was detected.")


def display_svm_prediction_results(processed_image, prediction, class_name): 
    subheader("Support Vector Machine (SVM) Model")
    st.markdown("<br>", unsafe_allow_html=True) 
    col1, col2 = st.columns(2)
    
    with col1:
        # Display the processed image
        #processed_image_uint8 = (processed_image * 255).astype(np.uint8)  # Scale to 0-255
        #processed_image_rgb = cv2.cvtColor(processed_image_uint8, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        st.image(processed_image, caption="Processed Image", width=128)
    
    with col2:
        st.success(f"SVM Prediction: '{class_name}'")
        
        