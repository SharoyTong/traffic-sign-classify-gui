# app.py
import streamlit as st
from page import about_us, traffic_sign_recognition

st.set_page_config(page_title="Traffic Vision Sign Recognition System", page_icon="🚦")

def main():
    st.sidebar.title("Navigation")
    
    #st.audio('static/audio.mp3', autoplay=True, loop=True)

    pages = {
        "About Us": about_us.show,
        "Traffic Sign Recognition": traffic_sign_recognition.show
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection]()

if __name__ == "__main__":
    main()