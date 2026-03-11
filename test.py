import cv2
from cvzone.HandTrackingModule import HandDetector  # Nhập Module theo dõi chấm tay
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)  # máy dò
classifier = Classifier("Model_4/hand_gesture_cnn.h5","Model_4/labels2.txt")

dolech = 20
imgSize = 28 #224

folder = "Data/C"
counter = 0

labels = [
    "A","B","C","D","E",
    "F","G","H","I","J",
    "K","L","M","N","O",
    "P","Q","R","S","T",
    "U","V","W","X"
]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img= detector.findHands(img)  # máy dò chấm, tìm bàn tay trả về hình ảnh
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize, 3), np.uint8) * 255#tạo ma trận số 1
        imgCrop = img[y-dolech:y + h+dolech, x-dolech:x + w+dolech]  # chiều cao bắt đầu và chiều cao kết thúc(tương tự chìu rộng)

        imgCropShape = imgCrop.shape

        aspectRatio = h/w # tỷ lệ khung hình = cao/r

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize- wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index =  classifier.getPrediction(imgWhite)
            print(prediction,index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap,:] = imgResize
            prediction, index =  classifier.getPrediction(imgWhite)


        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0, 255), 2)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)


    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)  # dộ trễ 1 milis

