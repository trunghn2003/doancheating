from ultralytics import YOLO
import cv2

# Tải mô hình
model = YOLO('best.pt')  # Đường dẫn file best.pt

# Mở camera (0 là camera mặc định trên MacBook)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận frame từ camera!")
        break

    # Dự đoán trên frame
    results = model(frame)

    # Hiển thị kết quả
    annotated_frame = results[0].plot()  # Vẽ bounding box
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()