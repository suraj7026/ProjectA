from FaceRecognitionHome import st

st.subheader("Register New Face")

import cv2

st.title("Student Input and Face Capture")

# Input Form
name = st.text_input("Enter your name")
roll_number = st.text_input("Enter your roll number")

# Camera Access
if st.button("Open Camera"):
    camera = cv2.VideoCapture(0)  # Open the default camera
    success, frame = camera.read()

    if success: 
        st.image(frame, channels="BGR")  # Display the captured image

        # ---- Potential Face Recognition Logic ----
        # 1. Perform face detection (using OpenCV or similar)
        # 2. Extract facial features
        # 3. Compare extracted features against your database of faces
        # 4. Display a result (match found or not)

    camera.release()  # Release the camera resource 
