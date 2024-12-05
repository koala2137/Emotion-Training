import warnings
warnings.filterwarnings('ignore')
import time

# 데이터 확인
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Detect Face
import cv2
from scipy.ndimage import zoom

# Model
from tensorflow.keras.models import load_model
from base64 import b64decode
import cv2
import numpy as np
from PIL import Image

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

# 모델 로드
# models_of_interest = {'model_happy': load_model('./happy.keras'), 'model_sad': load_model('./sad.keras'), 'model_angry': load_model('./angry.keras'),}
model = load_model('./sad.keras')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def take_photo(filename='photo.jpg', quality=0.8):
    picture = st.camera_input("당신의 무표정을 보여주세요!")
    if picture:
        # st.image(picture)
        image = Image.open(picture)
        image.save(filename)
        return filename


def analyze_emotion(filename, model):
    # Load the original image
    face = cv2.imread(filename)
    
    gray, detected_faces, coord = detect_face(face)
    if len(detected_faces) == 0:
        st.toast('Failed detecting your face', icon='😓')
        time.sleep(.5)
        st.toast('Please clear the photo and try again!', icon='🙇‍♂️')
        os.remove(filename)
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
    
    return prediction, fig

def calculate_variance(predictions):
    """
    Calculate the variance for the given predictions.
    Each prediction is expected to be a list or array of emotion probabilities.
    """
    if len(predictions) < 2:
        ## return later HGH
        print("Not enough data to calculate variance.")
        return None

    predictions_array = np.array(predictions)  # Convert list of predictions to NumPy array
    variance = np.var(predictions_array, axis=0)  # Variance along each emotion dimension
    return variance

    
# -----------------------------------------------------------------------------------------------------------------------------------------
# Webpage Setting
import streamlit as st

# def delete_file(button_emo):
#     if os.path.exists(f"photo_{button_emo}.jpg"):
#         os.remove(f"photo_{button_emo}.jpg")

apptitle = "Project E:MBTI"
st.set_page_config(page_title=apptitle, page_icon=":rainbow:")
st.title("Project E:MBTI")

st.markdown("""
            <style>
            body {
                background-color: beige;
            }
            h2 {
                font-size: 20px;
            }
            .model_num {
                color: navy;
                font-size: 25px;
            }
            </style>
            """, unsafe_allow_html=True)

st.write("🛎️ 분석을 마친 후 페이지 하단에 생성될 버튼을 잊지 말고 눌러주세요! 🛎️")
st.markdown("---")
# Streamlit 앱 실행
emotions_of_interest = ['happy', 'sad', 'angry', 'usual']
kor_emotions = ['행복한', '슬픈', '화난', '평상시의']
pictures_of_interest = {}

def delete_file():
    for del_emo in emotions_of_interest:
        if os.path.exists(f"photo_{del_emo}.jpg"):
            os.remove(f"photo_{del_emo}.jpg")
    
    # if os.path.exists("photo_happy.jpg"):
    #     os.remove("photo_happy.jpg")
    # if os.path.exists("photo_sad.jpg"):
    #     os.remove("photo_sad.jpg")
    # if os.path.exists("photo_angry.jpg"):
    #     os.remove("photo_angry.jpg")
    # if os.path.exists("photo_usual.jpg"):
    #     os.remove("photo_usual.jpg")
    # for del_emo in emotions_of_interest:
    #     if os.path.exists(f"photo_{del_emo}.jpg"):
    #         try:
    #             os.remove(f"photo_{del_emo}.jpg")
    #             st.write(f"Deleted: photo_{del_emo}.jpg")
    #         except Exception as e:
    #             st.write(f"Error deleting photo_{del_emo}.jpg: {e}")
    #     else:
    #         st.write(f"File not found: photo_{del_emo}.jpg")

# if any(os.path.exists(f"photo_{emo}.jpg") for emo in emotions_of_interest):
#     st.button("Clear all the photos", use_container_width=True, on_click=delete_file)

# if any(os.path.exists(f"photo_{emo}.jpg") for emo in emotions_of_interest):
#     count = 0
#     button_emotions = []
#     for emo in emotions_of_interest:
#         if os.path.exists(f"photo_{emo}.jpg"):
#             count = count + 1
#             button_emotions.append(emo)
#     columns = st.columns(count)
#     if button_emotions:
#         for i, column in enumerate(columns):
#             button_emo = button_emotions[i]
#             column.button(f"Clear the {button_emo} photo", use_container_width=True, on_click=lambda: delete_file(button_emo))
    
for i, emo  in enumerate(emotions_of_interest):
    if os.path.exists(f"photo_{emo}.jpg"):
        st.subheader(f"당신의 {kor_emotions[i]} 표정을 보여주세요! 과장 없는 자연스러운 표정을 지어주세요.")
        st.write(f"Show your {emo} face! Do not exaggerate and just show your natural {emo} face.")
        st.image(f"photo_{emo}.jpg")
        st.markdown("---")
    else:
        new_emotions_of_interest = emotions_of_interest[i:]
        for i, new_emo in enumerate(new_emotions_of_interest):
            st.subheader(f"당신의 {kor_emotions[i]} 표정을 보여주세요! 과장 없는 자연스러운 표정을 지어주세요.")
            pictures_of_interest[new_emo] = st.camera_input(f"Show your {new_emo} face! Do not exaggerate and just show your natural {new_emo} face.")
            st.markdown("---")
            if pictures_of_interest[new_emo]:
                image = Image.open(pictures_of_interest[new_emo])
                rgb_image = np.array(image)
                bgr_image = rgb_image[..., ::-1]
                gray, detected_faces, coord = detect_face(bgr_image)
                if len(detected_faces) == 0:
                    st.toast('Failed detecting your face', icon='😓')
                    time.sleep(.5)
                    st.toast('Please clear the photo and try again!', icon='🙇‍♂️')
                else:
                    image.save(f"photo_{new_emo}.jpg")
                    st.toast('Image saved! Scroll down for the next.', icon='🙆‍♂️')
        break

predictions = []
if all(os.path.exists(f"photo_{emo}.jpg") for emo in emotions_of_interest):
    for emo in emotions_of_interest: 
        prediction, fig = analyze_emotion(f"photo_{emo}.jpg", model)
        predictions.append(prediction)

if len(predictions) == 4:
    variance = calculate_variance(predictions)
    total_variance_sum = sum(variance)

    progress_text = "Photo taken! Analyzing your expression type..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

    if total_variance_sum > 3.3205783900314145e-02:
        final_type = 'SAOP(Sad+Opened)'
    else:
        final_type = 'SACL(Sad+Closed)'
    
    result_container = st.container(border=True)
    result_container.title(":worried:")
    result_container.subheader('Finished analyzing!')
    result_container.subheader(f'Your e:MBTI is... {final_type}')
    if st.button("Do you want to retry? or Do you want to leave?"):
        emotions_of_interest = ['happy', 'sad', 'angry', 'usual']
        for del_emo in emotions_of_interest:
            if os.path.exists(f"photo_{del_emo}.jpg"):
                os.remove(f"photo_{del_emo}.jpg")
        st.rerun()        


