import os
from typing import Counter
from PIL import Image
import numpy as np
import cv2
import pickle
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from LBP import LocalBinaryPatterns as lbp



root = os.path.dirname(os.path.abspath(__file__))
imgs = os.path.join(root,"images")

face_detect = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

x_train = []
x_ = []
y_train = []
lb = lbp(100, 8)
label_id = {}
cur_id = 0
for racin, dirs, files in os.walk(imgs):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(racin,file)
            label = os.path.basename(racin).replace(" ","-").lower()
            pil_img = Image.open(path)
            img_mat = np.array(pil_img)
            x_.append(label)
            hist = lb.describe(img_mat)
            x_train.append(hist)
            y_train.append(label)

x , xt , y , yt = train_test_split(x_train,y_train,test_size=0.2)
#model = MLPClassifier(hidden_layer_sizes=(13,13,3), max_iter=6000)
model = KNeighborsClassifier(n_neighbors=2)
#model = svm.SVC()

model.fit(x_train,y_train)

print(model.predict(xt))
print(model.score(xt,yt))

with open("modelKNN", "wb") as fi:
    pickle.dump(model,fi)
 


