import cv2
import numpy as np

model_path = "Advanced face detector/models/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "Advanced face detector/models/deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(config_path, model_path)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    h, w = img.shape[:2]
    resized_image = cv2.resize(img, (300, 300))
    # [100, 100, 100] * 1 -> [100 - 104, 100 - 117, 100 - 123] -> standardization
    blob = cv2.dnn.blobFromImage(
        resized_image, 1, (300, 300), (104.0, 117.0, 123.0))

    net.setInput(blob)
    faces = net.forward()
    # print(faces.shape)
    # faces.shape = [1, 1, 200, 7]

    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
