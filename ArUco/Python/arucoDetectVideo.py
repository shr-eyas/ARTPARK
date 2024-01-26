# pip install opencv-contrib-python==4.6.0.66   
# detecting aruco markers in live video feed

import cv2

cap = cv2.VideoCapture(0)

def findAruco(image, draw=True):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    print(ids)

    if draw:
        cv2.aruco.drawDetectedMarkers(image,corners,ids)

    return corners, ids

while cap.isOpened():
 
    _, image = cap.read()

    corners, ids = findAruco(image)

    cv2.imshow('image',image)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
