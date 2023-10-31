import cv2
from facetools import FaceDetection, IdentityVerification, LivenessDetection
from facetools.utils import visualize_results
import os

data_folder = 'data'
resnet_model_path = data_folder + '/checkpoints/InceptionResnetV1_vggface2.onnx'
liveness_model_path = data_folder + '/checkpoints/OULU_Protocol_2_model_0_0.onnx'
facebank_path = data_folder + '/mukhamejan.csv'

faceDetector = FaceDetection()
identityVerifier = IdentityVerification(resnet_model_path, facebank_path)
livenessDetector = LivenessDetection(liveness_model_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    canvas = frame.copy()
    faces, boxes = faceDetector(canvas)
    for face_arr, box in zip(faces, boxes):
        min_sim_score, mean_sim_score = identityVerifier(face_arr)
        lv_score = livenessDetector(face_arr)
        canvas = visualize_results(canvas, box, lv_score, mean_sim_score)

    cv2.imshow('frame', canvas)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
