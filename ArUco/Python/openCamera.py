import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, image = cap.read()
 
    cv2.imshow('image',image)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
