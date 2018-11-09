import cv2

face_cascade = cv2.CascadeClassifier("cascade/data/haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture(0)

while (True):

    #Capturar frame por frame
    ret,frame = cap.read()
    #Poniendo el frame a escala a grises
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #detectando rostro
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)

    for (x,y,w,h) in faces:
        #Dibujando el rectangulo 
        color = (0,0,255) # Estructura del color BGR
        stroke = 2 #Grosor del rectangulo
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)

    #Mostrando los frames resultantes
    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()