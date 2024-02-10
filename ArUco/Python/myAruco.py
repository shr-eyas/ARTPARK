import cv2
import numpy as np
import glob

class myAruco:

    def __init__(self, dictionary_type,  marker_length_m):
        """
        Initialize Aruco marker detector.

        Args:
            dictionary_type: Aruco dictionary type (e.g., cv2.aruco.DICT_6X6_250).
            marker_length_m: Length of the Aruco marker in meters.

        Note:
            User can input the intrinsic camera matrix and the distortion coeffiecent if available
            or else use the calibrate service if it is not available.
        """
        self.image = None
        self.distortion_coefficients = None
        self.camera_matrix = None
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.marker_length_m = marker_length_m
        self.detected_markers = []



    def detect(self, image, cam_matrix, dist_coeff):
        """
        Detect Aruco markers in the input image.

        Args:
            image: Input image.
            camera_matrix: Intrinsic camera matrix.
            distortion_coefficients: Distortion coefficients.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imshow('output', image)
        cv2.waitKey(10)
        self.detected_markers = []  

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length_m, cam_matrix, dist_coeff)

            for i in range(len(ids)):
   
                marker_id = int(ids[i][0])
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
             
                angle = np.linalg.norm(rvec)
                axis = rvec / angle if angle != 0 else np.array([0, 0, 1])
                half_angle = angle / 2.0
                sin_half = np.sin(half_angle)
                quaternion = np.array([axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, np.cos(half_angle)])

                self.detected_markers.append((marker_id, tvec, quaternion))



    def get(self):
        """Get the list of detected Aruco markers."""
        return self.detected_markers



    def calibrate(self, folder_path, checkerboard_dimensions):
        """
        Calibrate camera using images of a checkerboard.

        Args:
            folder_path: Path to the folder containing checkerboard images.
            checkerboard_dimensions: Tuple (rows, cols) specifying checkerboard dimensions.

        Returns:
            ret: Calibration success status.
            camera_matrix: Intrinsic camera matrix.
            distortion_coefficients: Distortion coefficients.
        """
        self.CHECKERBOARD = checkerboard_dimensions
        images = glob.glob(folder_path + '/*.png')
        
        for file in images:
            image = cv2.imread(file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            self.objpoints = []
            self.imgpoints = []
            self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
            self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)
            ret, corners = cv2.findChessboardCorners(
                gray, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)

        h, w = image.shape[:2]

        ret, self.camera_matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, (w, h), None, None)

        return ret, self.camera_matrix, self.distortion_coefficients



    def generate(self, marker_ID, border_bits, output_path):
        """
        Generate Aruco marker with the specified ID.

        Args:
            marker_ID: Aruco marker ID.
            border_bits: Number of border bits.
            output_path: Path to save the generated marker image.

        Returns:
            success: True if marker generation is successful, False otherwise.
        """
        try:
            marker_size = self.marker_length_m * 1000
            marker_image = cv2.aruco.drawMarker(self.dictionary, marker_ID, int(marker_size), borderBits=border_bits)
            cv2.imwrite(output_path, marker_image)
            self.get_logger().info(f"Marker with ID {marker_ID} saved to {output_path}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to generate ArUco marker: {str(e)}")
            return False

