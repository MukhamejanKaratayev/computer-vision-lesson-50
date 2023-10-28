import cv2
import numpy as np
import face_recognition
import os

path = 'Face recognition/face_images'

files = os.listdir(path)
face_images = []
for file in files:
    face_images.append(path + '/' + file)

loaded_images = []
encoded_images = []
for img in face_images:
    tmp = face_recognition.load_image_file(img)
    encoding = face_recognition.face_encodings(tmp)
    if len(encoding) > 0:
        encoded_images.append(encoding[0])
        loaded_images.append(tmp)
    else:
        print("No face found in image " + img)

# print('Loaded imgs: ', loaded_images)
# print('Encoded imgs: ', encoded_images)

known_face_encoding = encoded_images
known_face_names = []
for file in files:
    name = file.split('.')[0]
    known_face_names.append(name)

print(known_face_names)

face_locations = []
face_encodings = []
face_names = []

cap = cv2.VideoCapture(0)
process_this_frame = True

while True:
    success, img = cap.read()

    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_img)
        face_encodings = face_recognition.face_encodings(
            rgb_small_img, known_face_locations=face_locations, num_jitters=1, model='large')

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encoding, face_encoding)

            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_face_encoding, face_encoding)
            # known_face_names = ["Bill Gates", "Elon Musk", "Jeff Bezos", "Mark Zuckerberg"]
            # known_face_encoding = [encoding1, encoding2, encoding3, encoding4]
            # matches = [False, False, True, True]
            # face_distances = [0.3, 0.5, 0.12, 0.1]
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(img, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
