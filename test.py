import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5","model/labels.txt")

offset = 20
imgsize = 300

folder = "D:\handcv\C"
counter = 0

labels = ["OK","Not Ok","Hi","Superb"]

while True:
    success,img = cap.read()
    imgOutput = img.copy()
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
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)
            
        else:
            k = imgsize/w
            hcal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgsize,hcal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize-hcal)/2)
            imgWhite[hGap:hcal+hGap,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)

        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+170,y-offset-50+50),(255,255,0),cv2.FILLED)
        cv2.putText(imgOutput,labels[index],(x,y-25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,255,0),4)

        #cv2.imshow("Cropped",imgCrop)
        #cv2.imshow("Whited Image",imgWhite)
    cv2.imshow("Camera",imgOutput)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
            break

# close screens
cap.release()
cv2.destroyAllWindows()