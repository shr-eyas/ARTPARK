#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main() {
    // Defining the dimensions of the checkerboard
    Size CHECKERBOARD(8, 6);
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);

    // Creating vectors to store vectors of 3D and 2D points
    vector<vector<Point3f>> objpoints;
    vector<vector<Point2f>> imgpoints;

    // Defining the world coordinates for 3D points
    vector<Point3f> objp;
    for (int i = 0; i < CHECKERBOARD.height; i++) {
        for (int j = 0; j < CHECKERBOARD.width; j++) {
            objp.push_back(Point3f(j, i, 0));
        }
    }

    // Extracting path of individual images stored in a given directory
    vector<String> images;
    glob("/home/sophia/Documents/Calibration/*.png", images);

    Mat gray;
    for (size_t i = 0; i < images.size(); i++) {
        Mat img = imread(images[i]);
        cvtColor(img, gray, COLOR_BGR2GRAY);

        // Find the chessboard corners
        vector<Point2f> corners;
        bool patternFound = findChessboardCorners(gray, CHECKERBOARD, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_FAST_CHECK + CALIB_CB_NORMALIZE_IMAGE);

        // If desired number of corners are found in the image, then patternFound = true
        if (patternFound) {
            objpoints.push_back(objp);

            // Refining pixel coordinates for given 2D points
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), criteria);
            imgpoints.push_back(corners);

            // Draw and display the corners
            drawChessboardCorners(img, CHECKERBOARD, corners, patternFound);

            imshow("img", img);
            waitKey(0);
        }
    }

    destroyAllWindows();

    // Get image size
    Size imageSize(gray.cols, gray.rows);

    // Performing camera calibration
    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    double rms = calibrateCamera(objpoints, imgpoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

    cout << "Camera matrix : \n" << cameraMatrix << endl;
    cout << "Distortion coefficients : \n" << distCoeffs << endl;

    return 0;
}

// Camera matrix : 
// [936.0652516098202, 0, 633.7619367680327;
//  0, 937.3543919408091, 344.3671310217896;
//  0, 0, 1]
// Distortion coefficients : 
// [0.3029196190859768, -1.441564743239442, -0.002203069687765735, -0.006587528866314235, -3.530766821016102]
