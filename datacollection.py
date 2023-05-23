import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

offset = 20
imgsize = 300

folder = 'D:\handcv\SUPERB'
counter = 0

while True:
    success,img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgWhite = np.ones((imgsize,imgsize,3),np.uint8)*255
        imgCrop = img[ y-offset:y+h+offset, x-offset:x+w+offset]
                
        aspectRatio = h/w
        
        if aspectRatio > 1:
            k = imgsize/h
            wcal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wcal,imgsize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgsize-wcal)/2)
            imgWhite[:,wGap:wcal+wGap] = imgResize
            
        else:
            k = imgsize/w
            hcal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgsize,hcal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize-hcal)/2)
            imgWhite[hGap:hcal+hGap,:] = imgResize

        cv2.imshow("Cropped",imgCrop)
        cv2.imshow("Whited Image",imgWhite)
    cv2.imshow("Camera",img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)