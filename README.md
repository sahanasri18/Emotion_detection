# 😊 Emotion Detection from Images

A beginner-friendly deep learning project that detects human emotions from uploaded images using a trained CNN model. This app is built with **Streamlit**, **OpenCV**, and **TensorFlow/Keras**.

## 💡 Features

- Detects emotions such as **Happy**, **Sad**, **Angry**, **Neutral**, **Surprise**, and more.
- Supports **multiple face detection** in one image.
- Shows **confidence scores** for each prediction.
- Simple drag-and-drop interface.

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy
- PIL (Python Imaging Library)

## 📁 Project Structure

Emotion_detector/
│
├── app.py # Streamlit main app
├── emotion_model.hdf5 # Pre-trained emotion detection model
├── README.md # This file
└── requirements.txt # Python dependencies

pip install -r requirements.txt
#to run 
streamlit run app.py


