import cv2
import numpy as np
from random import randint

modelPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, modelPath)
net.setInputSize(320, 320)
# normalization: -1 < pixel_value < 1
# пример значения писеля (134, 240, 56), размер 320x320
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
# (134 - 127.5, 240 - 127.5, 56 - 127.5) / 127.5 -> normalized pixel
# BGR -> RGB
net.setInputSwapRB(True)

with open('coco.names', 'rt') as f:
    # person\nbicycle\ncar\nmotorcycle...
    class_names = f.read().rstrip('\n').split('\n')
    print(class_names)

class_color = []
for i in class_names:
    class_color.append((randint(0, 255), randint(0, 255), randint(0, 255)))

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()  # BGR format
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=class_color[classId-1], thickness=2)
            cv2.putText(img, class_names[classId-1].upper(), (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, class_color[classId-1], 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
