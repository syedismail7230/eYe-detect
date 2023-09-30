import cv2
import numpy as np

def detect_eyes(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    eyes = []
    for (x, y, w, h) in faces:
        eyes.extend(eye_cascade.detectMultiScale(image[y:y+h, x:x+w], 1.3, 5))

    return eyes

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    eyes = detect_eyes(frame)

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Eye Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
