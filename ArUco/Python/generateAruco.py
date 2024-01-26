import cv2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_id = 1
marker_size = 200
border_bits = 1

markerImage = cv2.aruco.drawMarker(dictionary, marker_id, marker_size, 
                                   borderBits=border_bits)

cv2.imwrite('marker2.png', markerImage)
