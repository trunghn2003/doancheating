from ultralytics import YOLO
model = YOLO('best.pt')  # Đường dẫn file best.pt
results = model('images.jpeg')    # Đặt ảnh test cùng thư mục
results[0].show()