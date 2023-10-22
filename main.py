import cv2

haarcascade = "C:/Users/acer/Desktop/RalaTime_Face_Detection/model/haarcascade_frontalface_default.xml"

#to open the camera
cap = cv2.VideoCapture(0)

#set the width capture from camera
cap.set(3, 640)
#set the height capture from camera
cap.set(4, 480)

while True:
    #read method returns status and RGB image
    success, img = cap.read()

    #define the model
    facecascade = cv2.CascadeClassifier(haarcascade)
    #cascade support only gray scale(B&W), here does convertion
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #identify the 68 landmarkd and if it there return 4 cordinates called x,y,w,h
    face = facecascade.detectMultiScale(img_gry, 1.1,4)

    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 2) #finally added color of rectangle and thikness of rectangle

    cv2.imshow('Face', img)

    #opencv defualt camera shutdown function
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break