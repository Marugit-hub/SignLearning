import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap= cv2.VideoCapture(0)
detector= HandDetector(maxHands=1)
offset = 20
imgsize=300
folder = "Data/D"
counter=0

while True:
    success,img=cap.read()
    hands,img=detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        imgcrop =img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgcropshape= imgcrop.shape


        aspectRatio=h/w
        if aspectRatio>1:
            k = imgsize / h
            wcal = math.ceil(k * w)
            imgresize = cv2.resize(imgcrop, (wcal, imgsize))
            imgresizeshape = imgresize.shape
            wgap = math.ceil((imgsize - wcal) / 2)
            imgwhite[:, wgap:wcal + wgap] = imgresize
        else:
            k = imgsize / w
            hcal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize,hcal ))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize - hcal) / 2)
            imgwhite[hgap:hcal+hgap,:] = imgresize


        cv2.imshow("imagecrop",imgcrop)
        cv2.imshow("imagewhite",imgwhite)


    cv2.imshow("image",img)
    key=cv2.waitKey(1)
    if key == ord('s') and counter<=100:
        counter+=1
        cv2.imwrite(f'{folder}/Img_{time.time()}.jpg',imgwhite)
        print(counter)
