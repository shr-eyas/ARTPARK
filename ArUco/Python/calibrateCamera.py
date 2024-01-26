import cv2
import numpy as np
import glob

CHECKERBOARD = (9, 6)  # Adjust dimensions based on your checkerboard
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []
imgpoints = []
objp = None
camera_matrix = None
distortion_coefficients = None

folder_path = "path/to/your/checkerboard/images"
images = glob.glob(folder_path + '/*.png')

for file in images:
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

h, w = image.shape[:2]

ret, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints, (w, h), None, None)

if ret:
    print("Calibration successful!")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", distortion_coefficients)
else:
    print("Calibration failed.")
