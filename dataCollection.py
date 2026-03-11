import cv2
from cvzone.HandTrackingModule import HandDetector  # Nhập Module theo dõi chấm tay
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)  # máy dò

dolech = 20
imgSize = 224

folder = "Data/Train/Z"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # máy dò chấm, tìm bàn tay trả về hình ảnh
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # tạo ma trận số 1
        imgCrop = img[y - dolech:y + h + dolech, x - dolech:x + w + dolech]  # chiều cao bắt đầu và chiều cao kết thúc (tương tự chiều rộng)

        imgCropShape = imgCrop.shape

        aspectRatio = h / w  # tỷ lệ khung hình = cao / r

        if aspectRatio > 1:
            # ảnh cao hơn rộng
            k = imgSize / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            # ảnh rộng hơn cao
            k = imgSize / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)  # độ trễ 1 milis
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
