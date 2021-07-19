import cv2
import pickle
from LBP import LocalBinaryPatterns as lbp

face_detect = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

with open("modelRN", "rb") as fi:
    model = pickle.load(fi)
    


cap = cv2.VideoCapture(0)
lb = lbp(100, 8)
i = 1
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        hist = lb.describe(roi_gray)
        conf = model.predict(hist.reshape(1,-1))
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = str(conf[0])
        color = (51,153,255)
        stroke = 2
        cv2.putText(frame, name, (x,y), font, 1,color, stroke, cv2.LINE_AA)
        color = (0,51,102)
        stroke = 2
        width = x+w
        height = y+h
        cv2.rectangle(frame, (x,y),(width,height),color,stroke)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
