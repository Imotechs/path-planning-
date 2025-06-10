import cv2
from pyzbar.pyzbar import decode
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()
detected_centers=[]
while True:
    ret, img = cap.read()
    if not ret:
         break
    img = cv2.flip(img, 1)  
    barcode = decode(img)
    for code in barcode:
        data = code.data.decode('utf-8')
        print(data)
        points = np.array([code.polygon],np.int32)
        pts =points.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(255,0,255),3)
        pt2 = code.rect
        cv2.putText(img,data,(pt2[0],pt2[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,0,255),1)
        # Get bounding rectangle
        rect = code.rect  # rect = (x, y, width, height)
        x, y, w, h = rect
        center_x = x + w // 2
        center_y = y + h // 2
        center = (center_x, center_y)
        detected_centers.append(center)
        # Draw center
        cv2.circle(img, center, 5, (0, 255, 0), -1)
    cv2.imshow("Barcode/QR Detection", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
