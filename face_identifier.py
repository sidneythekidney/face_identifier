import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('face_test2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(len(faces))
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
res = cv2.resize(img, (1100, 600), interpolation = cv2.INTER_CUBIC)
cv2.imshow('img',res)
cv2.waitKey(0)
cv2.destroyAllWindows()