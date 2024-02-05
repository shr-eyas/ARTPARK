#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <cmath>

using namespace std;
using namespace cv;

void rotationVectorToQuaternion(const double* rotationVector, double* quaternion) {

    double norm = std::sqrt(rotationVector[0]*rotationVector[0] + 
                            rotationVector[1]*rotationVector[1] + 
                            rotationVector[2]*rotationVector[2]);

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

int main(){

    cv::VideoCapture inputVideo(0);
    float markerSize = 0.05;
    cv::Mat image, imageCopy, cameraMatrix, distCoeffs;

    cameraMatrix = (Mat_<double>(3, 3) << 936.0652516098202, 0, 633.7619367680327,
                                         0, 937.3543919408091, 344.3671310217896,
                                         0, 0, 1);

    distCoeffs = (Mat_<double>(1, 5) << 0.3029196190859768, -1.441564743239442, -0.002203069687765735, -0.006587528866314235, -3.530766821016102);

    cv::Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::DetectorParameters detectorParams;
    
    /* Thresholding */ 
    detectorParams.adaptiveThreshConstant = 7; 

    /* increasing the window size as low values can 'break' the marker border */
    detectorParams.adaptiveThreshWinSizeMin = 3; 
    detectorParams.adaptiveThreshWinSizeMax = 53; /* default 23 */
    detectorParams.adaptiveThreshWinSizeStep = 4; /* default 10 */

    /* Contour Filtering
    reducing the minimum marker parameter so as to accomodate more contours */
    detectorParams.minMarkerPerimeterRate = 0.1; /* default 0.3 */
    detectorParams.maxMarkerPerimeterRate = 4.0;

    /* reducing maximum allowable error in square shape detection
    must be increased if images as distorted */
    detectorParams.polygonalApproxAccuracyRate = 0.01;  /* default 0.05 */

    detectorParams.minCornerDistanceRate = 0.05;
    detectorParams.minMarkerDistanceRate = 0.05;
    detectorParams.minDistanceToBorder = 3;

    /* Bits Extraction */
    detectorParams.markerBorderBits = 1;
    detectorParams.minOtsuStdDev = 5.0;
    detectorParams.perspectiveRemovePixelPerCell = 8; /* default 4 */
    detectorParams.perspectiveRemoveIgnoredMarginPerCell = 0.13;

    /* Marker Detection */
    detectorParams.maxErroneousBitsInBorderRate = 0.04; /* default 0.35 */
    detectorParams.errorCorrectionRate = 0.6;

    /* Corner Refinement */
    detectorParams.cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
    detectorParams.cornerRefinementMaxIterations = 30;
    detectorParams.cornerRefinementMinAccuracy = 0.01; /* default 0.1 */
    detectorParams.cornerRefinementWinSize = 5;

    while(inputVideo.isOpened()){
        inputVideo >> image;
        image.copyTo(imageCopy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        aruco::detectMarkers(image, dictionary, corners, ids, cv::makePtr<aruco::DetectorParameters>(detectorParams));

        if (!ids.empty()) {
            aruco::drawDetectedMarkers(imageCopy, corners, ids);
            std::vector<cv::Vec3d> rvecs, tvecs;
            aruco::estimatePoseSingleMarkers(corners, markerSize, cameraMatrix, distCoeffs, rvecs, tvecs);
            for (int i = 0; i < ids.size(); i++) {
                double quaternion[4];
                rotationVectorToQuaternion(rvecs[i].val, quaternion);
                cout << "Marker ID: " << ids[i] << ", Quaternion: (" << quaternion[0] << ", " << quaternion[1]
                     << ", " << quaternion[2] << ", " << quaternion[3] << ")" << endl;
            }
        }

        cv::imshow("Out", imageCopy);

        int key = cv::waitKey(1);
        if(key == 27){
            break;
        }
    }

    inputVideo.release();
    cv::destroyAllWindows;

    return 0;
}
