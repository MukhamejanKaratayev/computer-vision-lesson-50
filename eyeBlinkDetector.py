import cv2
import dlib
from math import hypot

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = facial_landmarks.part(
        eye_points[0]).x, facial_landmarks.part(eye_points[0]).y
    right_point = facial_landmarks.part(
        eye_points[3]).x, facial_landmarks.part(eye_points[3]).y

    center_top = midpoint(facial_landmarks.part(
        eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(
        eye_points[4]), facial_landmarks.part(eye_points[5]))

    horizontal_line_length = hypot(
        (left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    vertical_line_length = hypot(
        (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = horizontal_line_length / vertical_line_length
    return ratio


MAX_TRESH = 10  # max treshold for blinking
count = 0

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    for face in faces:
        landmarks = predictor(imgGray, face)
        left_eye_ratio = get_blinking_ratio(
            [36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio(
            [42, 43, 44, 45, 46, 47], landmarks)

        overall_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if overall_ratio > 5:
            cv2.putText(img, "BLINKING", (50, 150),
                        cv2.FONT_HERSHEY_PLAIN, 7, (0, 255, 0), 5)
            count = count + 1
            if count > MAX_TRESH:
                print("Человек уснул! Тревога!")
                cv2.putText(img, "ALERT! WAKE UP!", (50, 250),
                            cv2.FONT_HERSHEY_PLAIN, 7, (0, 0, 255), 5)
        else:
            count = 0

        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow('Webcam video', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
