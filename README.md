# ASLDetection
Dự án ASLDetection sử dụng mô hình học sâu (deep learning) để nhận diện các cử chỉ tay trong American Sign Language (ASL) từ webcam. 
Mô hình được huấn luyện trên dữ liệu hình ảnh bàn tay

Kiến trúc mô hình:
MediaPipe / cvzone HandDetector để phát hiện bàn tay và các keypoints.
OpenCV để xử lý ảnh và hiển thị.
CNN (Convolutional Neural Network) để huấn luyện mô hình nhận diện ký tự ASL.

# Chạy thư viện trước nha 
pip install -r requirements.txt

# Thu thập dataset 
(nếu muốn tự thu dataset của mình hoặc thêm dataset để train)
(nhớ khi thu thập data đổi từng đường dẫn bên trong)
python dataCollection.py

# Train model 
(dữ liệu ảnh đầu vào là 224x224 giống với ảnh thu thập từ dataset)
python train_model.py

# Test
python test.py

```text
ASLDataCollection/
├── dataCollection.py    # Thu thập dữ liệu
├── requirements.txt     # Các thư viện cần thiết
├── test.py              # Kiểm tra mô hình (Demo)
├── train_model.py       # Huấn luyện mô hình
├── README.md            # Tài liệu hướng dẫn
└── Data/
    ├── Train/
    │   ├── A/
    │   ├── B/
    │   └── C/
    └── Val/
        ├── A/
        ├── B/
        └── C/
