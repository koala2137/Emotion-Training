import warnings
warnings.filterwarnings('ignore')

# 데이터 확인
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset 만들기
from tensorflow.keras.utils import to_categorical

# Detect Face
import cv2
from scipy.ndimage import zoom

# Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from base64 import b64decode
import cv2
import numpy as np
from PIL import Image
import io

model = load_model('./model.keras')
emotion_labels = ['angry', 'disgust', 'happy', 'sad', 'neutral']  # Ensure this aligns with the model's output



# 전체 이미지에서 얼굴을 찾아내는 함수
def detect_face(frame):

    # cascade pre-trained 모델 불러오기
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # RGB를 gray scale로 바꾸기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cascade 멀티스케일 분류
    detected_faces = face_cascade.detectMultiScale(gray,
                                                   scaleFactor = 1.1,
                                                   minNeighbors = 6,
                                                   minSize = (shape_x, shape_y),
                                                   flags = cv2.CASCADE_SCALE_IMAGE
                                                  )
    print(f"detected_faces are {detected_faces}")
    coord = []
    for x, y, w, h in detected_faces:
        if w > 100:
            sub_img = frame[y:y+h, x:x+w]
            coord.append([x, y, w, h])

    return gray, detected_faces, coord

# 전체 이미지에서 찾아낸 얼굴을 추출하는 함수
def extract_face_features(gray, detected_faces, coord, offset_coefficients=(0.075, 0.05)):
    new_face = []
    for det in detected_faces:

        # 얼굴로 감지된 영역
        x, y, w, h = det

        # 이미지 경계값 받기
        horizontal_offset = int(np.floor(offset_coefficients[0] * w))
        vertical_offset = int(np.floor(offset_coefficients[1] * h))

        # gray scacle 에서 해당 위치 가져오기
        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]

        # 얼굴 이미지만 확대
        new_extracted_face = zoom(extracted_face, (shape_x/extracted_face.shape[0], shape_y/extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max()) # sacled
        new_face.append(new_extracted_face)

    return new_face


shape_x = 96
shape_y = 96


def take_photo(filename='photo.jpg', quality=0.8):
    picture = st.camera_input("Take a picture")
    if picture:
        # st.image(picture)
        image = Image.open(picture)
        image.save('photo.jpg')
        return filename # which is photo.jpg


def analyze_emotion(filename, model):
    # Load the original image
    face = cv2.imread(filename)
    
    gray, detected_faces, coord = detect_face(face)
    if len(detected_faces) == 0:
        print("No face detected in the image.")
        return
    face_zoom = extract_face_features(gray, detected_faces, coord)

    # Convert the extracted face into the format required by the model
    img_array = np.reshape(face_zoom[0], (1, 96, 96, 1))

    # Model prediction
    prediction = model.predict(img_array)[0]  # Get the probabilities for each emotion

    # Convert prediction to percentages
    prediction_percentages = {emotion_labels[i]: round(prob * 100, 10) for i, prob in enumerate(prediction)}

    fig = plt.figure(figsize=(10,5)) 
    # Display the original and extracted face
    plt.subplot(121)
    plt.title("Original Face")
    plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    plt.subplot(122)
    plt.title("Extracted Face")
    plt.imshow(face_zoom[0], cmap='gray')
    plt.axis('off')
    plt.show()
    
    st.pyplot(fig)
    return prediction
    return None

import streamlit as st
# Streamlit 앱 실행
st.title("Streamlit Webcam Photo Capture")
filename = take_photo()
prediction = analyze_emotion(filename, model)
if prediction[3] >= 0.25:
    if prediction[0] < 0.15:
        neutral = 'SAD'
        model = 1
        st.write("Finished analyzing! Now move to the next room and go to the model number", model)
    else:
        # SAD, angry 동시 충족할 시 기준점에서 떨어진 정도로 결정
        if prediction[0] - 0.15 > prediction[3] - 0.25:
            neutral = 'ANGRY'
            model = 2
            st.write("Finished analyzing! Now move to the next room and go to the model number", model)
        else:
            neutral = 'SAD'
            model = 1
            st.write("Finished analyzing! Now move to the next room and go to the model number", model)
elif prediction[0] >= 0.15:
    neutral = 'ANGRY'
    model = 2
    st.write("Finished analyzing! Now move to the next room and go to the model number", model)
else:
    neutral = 'HAPPY'
    model = 3
    st.write("Finished analyzing! Now move to the next room and go to the model number", model)









# while True:
#         try:
#             print("""Just show your natural face! Do not make any facial expressions.
# We are trying to analyze your neutral face type. Here it Goes!""")
#             filename = take_photo()
#             print("Photo taken! Analyzing your neutral face type...")

#             prediction = analyze_emotion(filename, model)

#             if prediction[3] >= 0.25:
#                 if prediction[0] < 0.15:
#                     neutral = 'SAD'
#                     model = 1
#                 else:
#                     # SAD, angry 동시 충족할 시 기준점에서 떨어진 정도로 결정
#                     if prediction[0] - 0.15 > prediction[3] - 0.25:
#                         neutral = 'ANGRY'
#                         model = 2
#                     else:
#                         neutral = 'SAD'
#                         model = 1
#             elif prediction[0] >= 0.15:
#                 neutral = 'ANGRY'
#                 model = 2
#             else:
#                 neutral = 'HAPPY'
#                 model = 3

#             print("Finished analyzing! Now move to the next room and go to the model number", model)
#             break  # 성공적으로 실행되었으므로 루프 종료

#         except Exception as e:
#             print("An error occurred:", str(e))
#             print("Retrying...")