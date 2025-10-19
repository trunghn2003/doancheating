import os
from collections import deque
import cv2
import numpy as np

# ============== 1) PHONE DETECTOR: YOLOv8 COCO ==============
from ultralytics import YOLO
yolo = YOLO("yolov8n.pt")  # pretrained COCO

# Tên lớp 'cell phone' trong COCO (ultralytics đã map sẵn)
PHONE_CLASS_NAME = "cell phone"

# ============== 2) FACE DETECTOR (để suy ra ROI tai) =========
import mediapipe as mp
mp_face = mp.solutions.face_detection
face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# ============== 3) (TÙY CHỌN) EARPHONE DETECTOR: ROBOFLOW =====
USE_ROBOFLOW = False  # bật True nếu bạn có API Key + model
ROBOFLOW_MODEL_ID = "cellphone-0aodn-pyq38/1"   # thay bằng model id thực tế trên Roboflow
ROBOFLOW_API_KEY = "Kw3rtMLVahV7lmO48WV1"  # export ROBOFLOW_API_KEY=...
rf_model = None

if USE_ROBOFLOW and ROBOFLOW_API_KEY:
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        # ví dụ workspace = 'your-workspace'
        # project = rf.workspace(workspace).project(project_slug)
        # nhưng tiện nhất: rf.model(MODEL_ID) nếu SDK hỗ trợ thẳng.
        rf_model = rf.model(ROBOFLOW_MODEL_ID)
        print("✅ Roboflow model loaded.")
    except Exception as e:
        print("⚠️ Roboflow init failed:", e)
        rf_model = None

# ============== 4) THAM SỐ & HÀM PHỤ TRỢ =====================

# Smoothing: cần phát hiện liên tiếp N khung mới báo
N_CONSEC_FRAMES_PHONE = 5
N_CONSEC_FRAMES_EARBUD = 8

phone_queue = deque(maxlen=N_CONSEC_FRAMES_PHONE)
earbud_queue = deque(maxlen=N_CONSEC_FRAMES_EARBUD)

def expand_box(xyxy, img_w, img_h, scale=1.15):
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    nx1 = max(0, int(cx - w/2)); ny1 = max(0, int(cy - h/2))
    nx2 = min(img_w - 1, int(cx + w/2)); ny2 = min(img_h - 1, int(cy + h/2))
    return nx1, ny1, nx2, ny2

def infer_phone_boxes(frame, conf_thres=0.3):
    """Trả về list bbox (x1,y1,x2,y2,conf) cho class 'cell phone'."""
    res = yolo(frame, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        cls_id = int(b.cls[0])
        cls_name = res.names[cls_id]
        conf = float(b.conf[0])
        if cls_name == PHONE_CLASS_NAME and conf >= conf_thres:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2, conf))
    return boxes

def face_to_ear_rois(frame, detections, ear_ratio=0.25, ear_height_ratio=0.45):
    """
    Suy ra ROI tai trái/phải tương đối theo bbox mặt.
    - ear_ratio: bề rộng vùng tai so với bề rộng mặt
    - ear_height_ratio: bề cao vùng tai so với bề cao mặt (ở giữa theo trục dọc)
    """
    H, W = frame.shape[:2]
    rois = []
    for det in detections:
        # mediapipe trả relative bbox (x,y,w,h) theo tỷ lệ
        rel = det.location_data.relative_bounding_box
        fx, fy, fw, fh = rel.xmin, rel.ymin, rel.width, rel.height
        x1 = int(max(0, fx * W)); y1 = int(max(0, fy * H))
        x2 = int(min(W - 1, (fx + fw) * W)); y2 = int(min(H - 1, (fy + fh) * H))
        bw, bh = (x2 - x1), (y2 - y1)

        ear_w = int(bw * ear_ratio)
        ear_h = int(bh * ear_height_ratio)
        ear_y1 = y1 + (bh - ear_h)//2
        ear_y2 = ear_y1 + ear_h

        # Tai trái (bên trái bbox mặt)
        left_roi = (max(0, x1 - ear_w), max(0, ear_y1), max(0, x1), min(H - 1, ear_y2))
        # Tai phải (bên phải bbox mặt)
        right_roi = (min(W - 1, x2), max(0, ear_y1), min(W - 1, x2 + ear_w), min(H - 1, ear_y2))

        rois.append(("left", left_roi))
        rois.append(("right", right_roi))
    return rois

def roboflow_detect_earbud(crop_bgr, conf_thres=0.5):
    """Gọi Roboflow model trên ROI tai (nếu có). Trả True nếu thấy earbud/earphone."""
    if rf_model is None:
        return False, 0.0
    # Roboflow nhận file path/np array tùy SDK version; dùng tạm qua buffer.
    try:
        # convert BGR->RGB
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        # một số SDK cần cv2.imwrite ra tệp tạm; ở đây dùng array:
        pred = rf_model.predict(crop_rgb, confidence=conf_thres).json()
        # Tùy model, class name có thể là 'earbud', 'earphone', 'earpiece'...
        found = False
        best = 0.0
        for p in pred.get("predictions", []):
            cls = p.get("class", "").lower()
            conf = float(p.get("confidence", 0.0))
            if any(k in cls for k in ["earbud", "earphone", "earpiece", "headset"]):
                found = True
                best = max(best, conf)
        return found, best
    except Exception as e:
        # Nếu lỗi API, coi như không phát hiện
        # print("RF error:", e)
        return False, 0.0

def draw_box(img, box, color, label=None):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# ============== 5) MAIN LOOP =================================
cap = cv2.VideoCapture(0)  # đổi sang đường dẫn video nếu cần
if not cap.isOpened():
    raise SystemExit("Cannot open camera")

print("Press 'q' to quit.")
def main():
    cap = cv2.VideoCapture(0)  # đổi sang đường dẫn video nếu cần
while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]

    # 1) Phone
    phone_boxes = infer_phone_boxes(frame, conf_thres=0.35)
    phone_present = len(phone_boxes) > 0
    phone_queue.append(1 if phone_present else 0)

    for (x1,y1,x2,y2,conf) in phone_boxes:
        draw_box(frame, (x1,y1,x2,y2), (0,255,255), f"phone {conf:.2f}")

    # 2) Face -> ear ROIs
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_det.process(rgb)
    ear_present = False
    best_ear_conf = 0.0

    if res.detections:
        ear_rois = face_to_ear_rois(frame, res.detections)
        for side, (ex1,ey1,ex2,ey2) in ear_rois:
            if ex2 <= ex1 or ey2 <= ey1:
                continue
            crop = frame[ey1:ey2, ex1:ex2]
            # 3) (Optional) detect earbud via Roboflow on ear-ROI
            ear_found, ear_conf = roboflow_detect_earbud(crop, conf_thres=0.5)
            if ear_found:
                ear_present = True
                best_ear_conf = max(best_ear_conf, ear_conf)
                draw_box(frame, (ex1,ey1,ex2,ey2), (0,0,255), f"{side} earbud {ear_conf:.2f}")
            else:
                draw_box(frame, (ex1,ey1,ex2,ey2), (200,200,200), f"{side} ear ROI")

    earbud_queue.append(1 if ear_present else 0)

    # 4) Smoothing & Alerts
    phone_alert = (sum(phone_queue) == phone_queue.maxlen)
    ear_alert = (sum(earbud_queue) == earbud_queue.maxlen)

    if phone_alert:
        cv2.putText(frame, "🚨 PHONE DETECTED (stable)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)

    if ear_alert:
        cv2.putText(frame, f"🚨 EARPHONE DETECTED (stable {best_ear_conf:.2f})", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Exam Monitor POC", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()

# ============== 6) ADD FACE TO DATABASE (DUMMY) =================
def add_face_to_database(image_path, name):
    """
    Hàm này sẽ tải ảnh từ đường dẫn `image_path`,
    trích xuất đặc trưng khuôn mặt (sử dụng thư viện như FaceNet),
    và lưu trữ vào cơ sở dữ liệu cùng với tên người dùng.
    (Đây chỉ là một ví dụ đơn giản, bạn cần thay thế bằng logic thực tế)
    """
    print(f"Đã thêm khuôn mặt của {name} từ ảnh {image_path} vào cơ sở dữ liệu.")

# ============== 7) MAIN ENTRY POINT ============================
if __name__ == "__main__":
    print("Press 'q' to quit.")
    # Ví dụ sử dụng hàm add_face_to_database (cần thay đổi logic bên trong)
    # add_face_to_database("path/to/your/image.jpg", "Your Name")
    main()
