from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract

img = cv2.imread("captcha_test.jpg")
pytesseract.pytesseract.tesseract_cmd = r{your path here} #add tesseract path here
model = YOLO('yolov8m.pt')

result = model(img)
result = result[0]
bounds = np.array(result.boxes.xyxy.cpu(), dtype="int")
cls = np.array(result.boxes.cls.cpu(), dtype="int")
cls_name = result.names
extracted_text = pytesseract.image_to_string(img)
for box , class_text in zip(bounds, cls):
    (x,y,x2,y2) = box
    text = cls_name[class_text]
    if text in extracted_text:
        cv2.rectangle(img,(x,y),(x2,y2),(80,255,15),2)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
