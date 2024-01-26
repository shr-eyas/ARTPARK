import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cam_matrix = np.array([[936.0652516098202, 0, 633.7619367680327],
                       [0, 937.3543919408091, 344.3671310217896],
                       [0, 0, 1]])

dist_coeffs = np.array([0.3029196190859768, -1.441564743239442, -0.002203069687765735, -0.006587528866314235, -3.530766821016102])

detected_markers = []

while cap.isOpened():
    _, image = cap.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is not None:
  
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, cam_matrix, dist_coeffs)

        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]

            angle = np.linalg.norm(rvec)
            axis = rvec / angle if angle != 0 else np.array([0, 0, 1])
            half_angle = angle / 2.0
            sin_half = np.sin(half_angle)
            quaternion = np.array([axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, np.cos(half_angle)])

            detected_markers.append((marker_id, quaternion))
            print(f"Marker ID: {marker_id}, Quaternion: {quaternion}")

    cv2.imshow('image', image)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
