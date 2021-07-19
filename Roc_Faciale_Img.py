import cv2
import pickle
from LBP import LocalBinaryPatterns as lbp

face_detect = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

with open("modelRN", "rb") as fi:
    model = pickle.load(fi)

lb = lbp(100,8)
image = cv2.imread("C:/Users/elmeh/Desktop/combine_images.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_detect.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
for(x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    hist = lb.describe(roi_gray)
    conf = model.predict(hist.reshape(1,-1))
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = str(conf[0])
    color = (51,153,255)
    stroke = 2
    cv2.putText(image, name, (x,y), font, 1,color, stroke, cv2.LINE_AA)
    color = (0,51,102)
    stroke = 2
    width = x+w
    height = y+h
    cv2.rectangle(image, (x,y),(width,height),color,stroke)
    cv2.imwrite('image.png',image)