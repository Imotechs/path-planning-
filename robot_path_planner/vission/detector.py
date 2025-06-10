import torch
import pathlib
import sys
import cv2
#from pyzbar.pyzbar import decode
import numpy as np
# Fix for loading PosixPath pickled data on Windows
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath
img_path = r"C:\Users\Admin\Desktop\Robotics2\Motion Planning\project\robot_path_planner\test_images\image.png"
# Load your model
qrcode_model_path =r"C:\Users\Admin\Desktop\Robotics2\Motion Planning\project\robot_path_planner\model\content\best_qrcode_model.pt"

qrcode_barcode_model_path = r'C:\Users\Admin\Desktop\Robotics2\Motion Planning\project\robot_path_planner\model\content\qrcode_model.pt'

# Set YOLOv5 directory path
yolov5_dir = r"C:\Users\Admin\Desktop\Robotics2\Motion Planning\project\robot_path_planner\model\yolov5"



# Load the model once at the top (avoid reloading every frame)
model = torch.hub.load(
    yolov5_dir,  # your local yolov5 clone
    'custom',
    path=qrcode_barcode_model_path,
    source='local'
)
model.conf = 0.25  # confidence threshold
model.iou = 0.45   # IoU threshold for NMS

def detect_objects_in_frame(frame):
    """
    Takes a single video frame (as a numpy array), returns bounding boxes and classes.
    Each box is in format (x1, y1, x2, y2, conf, class_id)
    """
    # Convert frame (OpenCV is BGR, PyTorch expects RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inference
    results = model(img, size=640)

    # Extract bounding boxes
    boxes = []
    for det in results.xyxy[0]:  # tensor shape: (N, 6)
        x1, y1, x2, y2, conf, cls = det
        boxes.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": float(conf),
            "class_id": int(cls)
        })

    return boxes





def get_barcode():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    best_center = None
    best_confidence = 0.0
    best_label = None

    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.flip(img, 1)  
        boxes = detect_objects_in_frame(img)
        detected_centers = []

        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            conf = box["confidence"]
            class_id = box["class_id"]

            if conf > 0.6:
                label = "qrcode" if class_id == 1 else "barcode"
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center = (center_x, center_y)
                detected_centers.append(center)

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 0, 255), 2)
                cv2.circle(img, center, 5, (0, 255, 0), -1)

                # Update best center by highest confidence
                if conf > best_confidence and label == "qrcode":
                    best_confidence = conf
                    best_center = center
                    best_label = label

        # Optionally break if we already have a good detection
        if best_center:
            print(f"Best {best_label} center: {best_center} (Conf: {best_confidence:.2f})")
            cap.release()
            cv2.destroyAllWindows()
            return best_center, best_label

        cv2.imshow("YOLO Detection", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None, None
