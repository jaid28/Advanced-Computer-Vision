import cv2
import mediapipe as mp
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,       
    max_num_hands=2,              
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7  
)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handlms, mpHands.Hand_CONNECTIONS)
