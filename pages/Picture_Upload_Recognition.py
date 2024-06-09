from FaceRecognitionHome import st
from FaceRecognitionHome import face_rec
import pandas as pd
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
import cv2

st.subheader("Picture Upload Recogniton")

# Getting Data
with st.spinner("Getting Data... "):
    db = face_rec.get_data(name = 'academy:register')
    # st.dataframe(db)
st.success("Data Loaded")


# Prediction


def process_images():
    data_list = []
    uploaded_files = st.file_uploader("Choose multiple images...", type="jpg", accept_multiple_files=True)



    for uploaded_file in uploaded_files:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        names, rolls = face_rec.face_recognition_upload_image(image, db, 'Features', threshold=0.4, name_role=['Name', 'Roll'])

        for name, roll in zip(names, rolls):        
            if name not in [item[0] for item in data_list]:
                data = [name, roll]
                data_list.append(data)
        
            
    return data_list
faces_found = process_images()
print(faces_found)
if faces_found:
    df = pd.DataFrame(faces_found,columns = ['Name', 'Roll'])
    st.write(df)  # Assuming st is from Streamlit
else:
    st.write("No faces detected in the uploaded images.")