import cv2
from pyzbar.pyzbar import decode
import numpy as np

def get_barcode_center():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    center = None

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        barcode = decode(img)
        
        for code in barcode:
            rect = code.rect
            x, y, w, h = rect
            center_x = x + w // 2
            center_y = y + h // 2
            center = (center_x, center_y)
            data = code.data.decode('utf-8')
            points = np.array([code.polygon],np.int32)
            pts =points.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(255,0,255),3)
            pt2 = code.rect
            cv2.putText(img,data,(pt2[0],pt2[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,0,255),1)
            cv2.circle(img, center, 5, (0, 255, 0), -1)
            #cv2.putText(img, code.data.decode(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            # Only return one detection for now
            
            #return center

        cv2.imshow("Barcode Detection", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None
