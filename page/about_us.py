# pages/about_us.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def show():

    
    st.title("About Us")
    st.image('static/background.png')
    st.audio('static/audio.mp3')
    
    with st.expander("Watch our video"):
        st.video('static/video.mp4')
    
    st.markdown("""
       <style>
       
       div.stMarkdown p, div.stMarkdown li {
            font-size: 18px;
        }
       
       </style>
   """, unsafe_allow_html=True)
   
   
    # Purpose and Objectives Section
    st.markdown("""
        <h2 style='text-align: center; background-color: #AEC8FF; padding: 10px; border-radius: 5px; color: black;'>Purpose and Objectives</h2>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown("""
        The **Traffic Sign Recognition System** is designed to help identify traffic signs with high accuracy using AI technology.  
        Our system leverages multiple AI models to ensure accurate and efficient recognition of traffic signs in real-world scenarios.
    """)
    st.markdown("<br>", unsafe_allow_html=True) 

    # Mission Section
    st.markdown("""
        <h2 style='text-align: center; background-color: #AEC8FF; padding: 10px; border-radius: 5px; color: black;'>Mission</h2>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown("""
        Our mission is to provide an intelligent traffic sign recognition system that enhances road safety through automation and machine learning.
    """)
    st.markdown("<br>", unsafe_allow_html=True) 

    # Vision Section
    st.markdown("""
        <h2 style='text-align: center; background-color: #AEC8FF; padding: 10px; border-radius: 5px; color: black;'>Vision</h2>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown("""
        We envision a future where traffic sign detection is fully automated, reducing accidents and improving the driving experience globally.
    """)
    st.markdown("<br>", unsafe_allow_html=True) 

    # Objectives Section
    st.markdown("""
        <h2 style='text-align: center; background-color: #AEC8FF; padding: 10px; border-radius: 5px; color: black;'>Objectives</h2>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown("""
        - To develop a German traffic sign recognition system using Artificial Intelligence.
        - To compare the performance of machine learning algorithms (YOLOv8, CNN, and SVM) using a confusion matrix.
        - To test the algorithms in real-life detection and demonstrate their capabilities through integrated GUI.
    """)
    st.markdown("<br>", unsafe_allow_html=True) 

    display_performance_metrics()
    display_additional_features()

def display_performance_metrics():
    st.markdown("""
        <h2 style='text-align: center; background-color: #AEC8FF; padding: 10px; border-radius: 5px; color: black;'>Model Performance Analytics</h2>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) 
    
    models = ["YOLOv8", "CNN", "SVM", "VGG"]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    data = {
        'YOLOv8': [0.9994, 0.9907, 0.9886, 0.9895],
        'CNN': [0.93, 0.89, 0.88, 0.88],
        'SVM': [0.78, 0.78, 0.78, 0.78],
        'VGG': [0.96, 0.96, 0.96, 0.96]
    }
    
    x = np.arange(len(metrics))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, model in enumerate(models):
        values = data[model]
        rects = ax.bar(x + i * width, values, width, label=model)
        ax.bar_label(rects, padding=3, fmt='%.4f', fontsize=8)

    ax.set_ylabel('Scores')
    ax.set_title('Performance Metrics of AI Models')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)  # Set y-axis limit to accommodate labels

    plt.tight_layout()
    st.pyplot(fig)
    
    st.image('static/linegraph.png', use_column_width=True)


def display_additional_features():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <h2 style='text-align: center; background-color: #AEC8FF; padding: 10px; border-radius: 5px; color: black;'>Additional Features</h2>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: justify;'>
            <ul>
                <li><strong>Cross-model predictions</strong>: Our system allows users to select multiple models to get an ensemble prediction.</li>
                <li><strong>Real-time feedback</strong>: After prediction, we provide confidence levels for each prediction.</li>
                <li><strong>Mobile-friendly</strong>: Our web interface is optimised for mobile devices, allowing users to access the system on-the-go.</li>
                <li><strong>User-friendly integrated GUI</strong>: The system features an intuitive GUI interface, making it easy for users to upload traffic sign images, select prediction models, and view results without needing any programming skills.</li>
                <li><strong>Confusion Matrix for Performance Analysis</strong>: The system includes a confusion matrix that helps users analyze the performance of each model.</li>
                <li><strong>Customizable Model Selection</strong>: Users can analyse and fine-tune the performance by selecting specific models based on their preferences or testing needs.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)