import cv2
import numpy as np

def getCountours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            cv2.drawContours(imgContours, cnt, -1, (0, 255, 0), 4)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print('Approx: ', approx)
            # print('Edge length: ', len(approx))
            objCor = len(approx)
            object_type = ''
            if objCor == 3:
                print('Hey triangle!')
                object_type = 'Triangle'
            elif objCor == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w/float(h)
                if aspect_ratio < 1.03 and aspect_ratio > 0.98:
                    object_type = 'Square'
                else:
                    object_type = 'Rectangle'
            elif objCor > 4 :
                object_type = 'Circles'
            else:
                object_type = 'None'

            cv2.rectangle(imgContours, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(imgContours, object_type, (x+(w//2)-10,y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (255,255,255),2) 


img = cv2.imread('data/shapes_and_colors.jpg')
imgContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)

imgCanny = cv2.Canny(imgBlur, 100, 100)
getCountours(imgCanny)

# cv2.imshow('Edge canny', imgCanny)
cv2.imshow('Result', imgContours)

cv2.waitKey(0)