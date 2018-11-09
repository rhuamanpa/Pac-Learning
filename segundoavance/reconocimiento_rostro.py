import numpy as np
import cv2
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
#Importando el entrenamiento con el opencv
face_cascade = cv2.CascadeClassifier("cascade/data/haarcascade_frontalface_alt2.xml")

labels = {"person-name":1}
with open ("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while (True):

    #Capturar frame por frame
    ret,frame = cap.read()
    #Poniendo el frame a escala a grises
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #detectando rostro
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)

    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        
        roi_gray = gray[y:y+h, x:x+w]
        img_item = "imagen.png"
        cv2.imwrite(img_item,roi_gray)
        
        id_, conf = recognizer.predict(roi_gray)

        if conf>= 50: # and conf <= 85:
            print(id_)
            print(labels[id_])

        #Dibujando el rectangulo 
        color = (255,0,0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)

    #Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()