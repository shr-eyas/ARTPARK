#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <vector>

using namespace cv;
using namespace std;

void rotationVectorToQuaternion(const double* rotationVector, double* quaternion) {
    double norm = std::sqrt(rotationVector[0] * rotationVector[0] +
                            rotationVector[1] * rotationVector[1] +
                            rotationVector[2] * rotationVector[2]);

    double angle = norm;
    double sin_half_angle = std::sin(angle / 2.0);

    if (std::abs(norm) > 1e-6) {
        quaternion[0] = rotationVector[0] / norm * sin_half_angle;
        quaternion[1] = rotationVector[1] / norm * sin_half_angle;
        quaternion[2] = rotationVector[2] / norm * sin_half_angle;
        quaternion[3] = std::cos(angle / 2.0);
    } else {
        quaternion[0] = 0.0;
        quaternion[1] = 0.0;
        quaternion[2] = 0.0;
        quaternion[3] = 1.0;
    }
}

class MyAruco {

private:
    cv::VideoCapture cap;
    cv::Ptr<aruco::Dictionary> dictionary;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    vector<vector<double>> quaternions;
    float markerSize;
    
public:
    MyAruco(float markerSize, int dictionaryID, const cv::Mat& userCameraMatrix, const cv::Mat& userDistCoeffs) : cap(0) {
        dictionary = cv::aruco::getPredefinedDictionary(dictionaryID);
        userCameraMatrix.copyTo(cameraMatrix);
        userDistCoeffs.copyTo(distCoeffs);
    }

    void detect() {
        while (cap.isOpened()) {
           
            if (!cap.isOpened()) {
                cerr << "Error: Unable to open webcam." << endl;
                break;
            }

            cv::Mat image;
            cap >> image;

            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

            vector<vector<cv::Point2f>> corners;
            vector<int> ids;
            aruco::DetectorParameters detectorParams;
            // Thresholding: increasing the window size as low values can 'break' the marker border 
            detectorParams.adaptiveThreshConstant = 7; 
            detectorParams.adaptiveThreshWinSizeStep = 4; // default 10 
            detectorParams.adaptiveThreshWinSizeMax = 53; // default 23 
            detectorParams.adaptiveThreshWinSizeMin = 3; 
            // Contour Filtering: reducing the minimum marker parameter so as to accomodate more contours 
            detectorParams.maxMarkerPerimeterRate = 4.0;
            detectorParams.minMarkerPerimeterRate = 0.1; // default 0.3 
            detectorParams.polygonalApproxAccuracyRate = 0.01;  // default 0.05 
            detectorParams.minCornerDistanceRate = 0.05;
            detectorParams.minMarkerDistanceRate = 0.05;
            detectorParams.minDistanceToBorder = 3;
            // Bits Extraction 
            detectorParams.markerBorderBits = 1;
            detectorParams.minOtsuStdDev = 5.0;
            detectorParams.perspectiveRemovePixelPerCell = 8; // default 4 
            detectorParams.perspectiveRemoveIgnoredMarginPerCell = 0.13;
            // Marker Detection 
            detectorParams.maxErroneousBitsInBorderRate = 0.04; // default 0.35 
            detectorParams.errorCorrectionRate = 0.6;
            // Corner Refinement 
            detectorParams.cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
            detectorParams.cornerRefinementMaxIterations = 30;
            detectorParams.cornerRefinementMinAccuracy = 0.01; // default 0.1 
            detectorParams.cornerRefinementWinSize = 5;

            cv::aruco::detectMarkers(gray, dictionary, corners, ids, cv::makePtr<aruco::DetectorParameters>(detectorParams));

            if (!ids.empty()) {
                aruco::drawDetectedMarkers(image, corners, ids);
                vector<cv::Vec3d> rvecs, tvecs;
                aruco::estimatePoseSingleMarkers(corners, markerSize, cameraMatrix, distCoeffs, rvecs, tvecs);
                quaternions.clear(); 
                for (int i = 0; i < ids.size(); i++) {
                    double quaternion[4];
                    rotationVectorToQuaternion(rvecs[i].val, quaternion);
                    quaternions.push_back(vector<double>(quaternion, quaternion + 4)); 
                }
            }
            cv::imshow("image", image);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows(); 
    }

    vector<vector<double>> get() {
        return quaternions;
    }

    void generate(int marker_id, int marker_size, int border_bits) {
        cv::Mat markerImage;
        aruco::drawMarker(dictionary, marker_id, marker_size, markerImage, border_bits);
        cv::imwrite("marker.png", markerImage);
    }

    void calibrate(const string& folderPath, const Size& checkerboardDimensions) {    
        vector<vector<Point3f>> objpoints;
        vector<vector<Point2f>> imgpoints;
        vector<Point3f> objp;

        for (int i = 0; i < checkerboardDimensions.height; i++) {
            for (int j = 0; j < checkerboardDimensions.width; j++) {
                objp.push_back(Point3f(j, i, 0));
            }
        }

        vector<String> images;
        glob(folderPath, images);

        Mat gray;
        for (size_t i = 0; i < images.size(); i++) {
            Mat img = imread(images[i]);
            cvtColor(img, gray, COLOR_BGR2GRAY);
            vector<Point2f> corners;
            bool patternFound = findChessboardCorners(gray, checkerboardDimensions, corners,
                CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_FAST_CHECK + CALIB_CB_NORMALIZE_IMAGE);

            if (patternFound) {
                objpoints.push_back(objp);
                cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
                imgpoints.push_back(corners);
            }
        }

        double rms = calibrateCamera(objpoints, imgpoints, gray.size(), cameraMatrix, distCoeffs, cv::noArray(), cv::noArray());
        cout << "Camera matrix : \n" << cameraMatrix << endl;
        cout << "Distortion coefficients : \n" << distCoeffs << endl;
    }
};

int main() {

    cv::Mat userCameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 500, 0, 1000, 300, 0, 0, 1);
    cv::Mat userDistCoeffs = (Mat_<double>(1, 5) << 0.1, 0.2, 0.0, 0.0, -0.1);

    MyAruco myAruco(0.05, cv::aruco::DICT_6X6_250, userCameraMatrix, userDistCoeffs);

    myAruco.detect();
    vector<vector<double>> quaternions = myAruco.get(); 


    // myAruco.generate(2, 200, 1);
    // myAruco.calibrate("/home/sophia/Documents/Calibration/*.png", Size(8, 6));

    // Use quaternions as needed
    for (int i = 0; i < quaternions.size(); ++i) {
        cout << "Quaternion " << i << ": ";
        for (int j = 0; j < quaternions[i].size(); ++j) {
            cout << quaternions[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
