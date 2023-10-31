import cv2
import numpy as np
from random import randint
from config import names_to_ru

# Загрузка модели
modelPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

# Настраиваем модель
net = cv2.dnn_DetectionModel(weightPath, modelPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Загрузка классов
with open('coco.names', 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# Генерация цветов для классов
class_color = []
for i in class_names:
    class_color.append((randint(0, 255), randint(0, 255), randint(0, 255)))

# Запуск видеопотока
cap = cv2.VideoCapture(0)

# Обработка видеопотока
while True:
    success, img = cap.read()  # BGR format
    classIds, confs, bbox = net.detect(img, confThreshold=0.55)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=class_color[classId-1], thickness=2)
            if class_names[classId-1] in names_to_ru:
                item_name = names_to_ru[class_names[classId-1]]
            else:
                item_name = class_names[classId-1]
            cv2.putText(img, item_name.upper(), (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, class_color[classId-1], 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
