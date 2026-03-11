import cv2
import numpy as np
import math
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector


MODEL_PATH = "Model_4/asl_model.keras"
LABEL_PATH = "Model_4/labels2.txt"

IMG_SIZE = 224
offset   = 20


HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),
    (0, 5),  (5, 6),  (6, 7),  (7, 8),
    (5, 9),  (9, 10), (10,11), (11,12),
    (9, 13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0, 17)
]


model = load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f]

print(f"Model loaded | {len(labels)} classes: {labels}")


cap      = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)


def draw_landmarks_on_white(img_crop, lm_list, crop_x1, crop_y1, size=IMG_SIZE):
    h_crop, w_crop = img_crop.shape[:2]
    if h_crop == 0 or w_crop == 0:
        return np.ones((size, size, 3), np.uint8) * 255

    img_white = np.ones((size, size, 3), np.uint8) * 255
    aspect = h_crop / w_crop

    if aspect > 1:
        k      = size / h_crop
        w_cal  = math.ceil(k * w_crop)
        w_gap  = math.ceil((size - w_cal) / 2)
        img_resized = cv2.resize(img_crop, (w_cal, size))
        img_white[:, w_gap:w_gap + w_cal] = img_resized
        scale_x, scale_y = k, k
        off_x, off_y = w_gap, 0
    else:
        k      = size / w_crop
        h_cal  = math.ceil(k * h_crop)
        h_gap  = math.ceil((size - h_cal) / 2)
        img_resized = cv2.resize(img_crop, (size, h_cal))
        img_white[h_gap:h_gap + h_cal, :] = img_resized
        scale_x, scale_y = k, k
        off_x, off_y = 0, h_gap

    pts = {}
    for lm in lm_list:
        lm_id = lm[0]
        lx = int((lm[1] - crop_x1) * scale_x + off_x)
        ly = int((lm[2] - crop_y1) * scale_y + off_y)
        pts[lm_id] = (lx, ly)

    for (a, b) in HAND_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img_white, pts[a], pts[b],
                     color=(220, 220, 220), thickness=3, lineType=cv2.LINE_AA)

    for lm_id, (lx, ly) in pts.items():
        cv2.circle(img_white, (lx, ly), radius=8,
                   color=(0, 0, 220), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img_white, (lx, ly), radius=8,
                   color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return img_white


while True:
    success, img = cap.read()
    if not success:
        break

    img_output = img.copy()
    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand   = hands[0]
        x, y, w, h = hand['bbox']
        lm_list = hand['lmList']

        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        img_crop = img[y1:y2, x1:x2]
        if img_crop.size == 0:
            cv2.imshow("Image", img_output)
            if cv2.waitKey(1) in [27, ord('q')]:
                break
            continue

        img_white = draw_landmarks_on_white(img_crop, lm_list, x1, y1, size=IMG_SIZE)


        img_input  = img_white.astype(np.float32)
        img_input  = np.expand_dims(img_input, axis=0)

        prediction = model.predict(img_input, verbose=0)
        idx        = np.argmax(prediction)
        confidence = prediction[0][idx]

        print(f"Predict: {labels[idx]:<4}  Confidence: {confidence:.2%}")

        cv2.rectangle(img_output, (x1, y1), (x2, y2), (255, 0, 255), 2)

        label_text = f"{labels[idx]}  {confidence:.0%}"
        text_y     = max(y1 - 12, 30)
        cv2.putText(img_output, label_text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 0, 255), 3)

        cv2.imshow("Landmark Input (model sees this)", img_white)
        cv2.imshow("Crop", img_crop)

    cv2.imshow("Image", img_output)

    key = cv2.waitKey(1)
    if key in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
