from FaceRecognitionHome import st
from FaceRecognitionHome import face_rec
import pandas as pd
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.subheader("Real Time Recogniton")

# Getting Data
with st.spinner("Getting Data... "):
    db = face_rec.get_data(name = 'academy:register')
    st.dataframe(db)
st.success("Data Loaded")


# Realtime Prediction



new_faces = []
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Perform face recognition
    pred_image,name,roll =  face_rec.face_recognition(img, db, 
                                            'Features', 
                                            threshold=0.4, 
                                            name_role=['Name', 'Roll'])
    
    

    return av.VideoFrame.from_ndarray(pred_image, format="bgr24")


webrtc_streamer(key="Real Time Prediction", video_frame_callback=video_frame_callback)

st.subheader("New Faces")
st.write(pd.DataFrame(new_faces, columns=["Name", "Roll"]))