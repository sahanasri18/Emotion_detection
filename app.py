import streamlit as st
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image
import base64
from io import BytesIO

# Load model
model = load_model("emotion_model.hdf5", compile=False)

# Emotion labels and emojis
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_emojis = ['😠', '🤢', '😱', '😄', '😢', '😲', '😐']

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# UI
st.set_page_config(page_title="Emotion Detection App", layout="centered")
st.title("😊 Emotion Detection App")
st.markdown("Upload an image, and I’ll detect the emotion(s) with confidence and emoji 😄")

# Theme toggle
dark_mode = st.toggle("🌗 Dark Mode")
if dark_mode:
    st.markdown("<style>body{background-color: #1e1e1e; color: white;}</style>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def convert_image_to_download(img_array):
    img_pil = Image.fromarray(img_array)
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    b64 = base64.b64encode(byte_im).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="emotion_result.png">📥 Download Result Image</a>'
    return href

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No faces detected.")
    else:
        face_count = 0
        for (x, y, w, h) in faces:
            if w < 30 or h < 30:
                continue

            face_count += 1
            roi_gray = gray[y:y+h, x:x+w]

            try:
                roi_resized = cv2.resize(roi_gray, (64, 64))
                roi_norm = roi_resized.astype('float32') / 255.0
                roi_reshaped = roi_norm.reshape(1, 64, 64, 1)

                prediction = model.predict(roi_reshaped, verbose=0)[0]
                max_index = np.argmax(prediction)
                emotion = emotion_labels[max_index]
                emoji = emotion_emojis[max_index]
                confidence = prediction[max_index] * 100

                label = f"{emotion} {emoji} ({confidence:.1f}%)"
                cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label_y = y - 10 if y - 10 > 10 else y + 20
                cv2.putText(img_np, label, (x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Confidence chart
                st.subheader(f"Face {face_count}: {label}")
                data = pd.DataFrame({
                    'Emotion': emotion_labels,
                    'Confidence (%)': [round(p * 100, 2) for p in prediction]
                })
                st.bar_chart(data.set_index('Emotion'))

            except Exception as e:
                st.error(f"Error processing face: {e}")

        st.image(img_np, channels="RGB", caption="Detected Emotion(s)")

        # Download button
        st.markdown("---")
        st.markdown("✅ Processing complete!")
        st.markdown(convert_image_to_download(img_np), unsafe_allow_html=True)




