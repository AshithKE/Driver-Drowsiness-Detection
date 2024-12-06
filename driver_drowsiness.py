import cv2
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp
import winsound  # For Windows audio alerts, optional

# Function to calculate EAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize the camera and take the instance
cap = cv2.VideoCapture(0)

# Status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# EAR thresholds
EAR_THRESH_SLEEPY = 0.22  # Adjust based on testing
EAR_THRESH_DROWSY = 0.25  # Adjust based on testing

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            h, w, _ = frame.shape
            
            # Extract coordinates of left and right eyes based on MediaPipe Face Mesh landmarks
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in [362, 385, 387, 263, 373, 380]]

            # Calculate EAR for left and right eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Determine state based on EAR thresholds
            if ear < EAR_THRESH_SLEEPY:
                
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 10:  # Increased threshold for sleep detection
                    status = "SLEEPING !!!"
                    color = (255, 0, 0)
                    # Optionally, play a beep sound
                    winsound.Beep(1000, 1000)  # Beep sound for 1 second at 1000 Hz

            elif ear < EAR_THRESH_DROWSY:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "Drowsy !"
                    color = (0, 0, 255)
                    # Optionally, play a beep sound
                    winsound.Beep(1000, 500)  # Beep sound for 0.5 second at 1000 Hz

            else:
                drowsy = 0
                sleep = 0
                active += 1
                if active > 6:
                    status = "Active :)"
                    color = (0, 255, 0)

            cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
