#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

using namespace std;
using namespace cv;

int main(){

    cv::VideoCapture inputVideo(0);
    float markerSize = 0.075; /* 75 mm */ 
    cv::Mat image, imageCopy, cameraMatrix, distCoeffs;

    cameraMatrix = (Mat_<double>(3, 3) << 936.0652516098202, 0, 633.7619367680327,
                                         0, 937.3543919408091, 344.3671310217896,
                                         0, 0, 1);

    distCoeffs = (Mat_<double>(1, 5) << 0.3029196190859768, -1.441564743239442, -0.002203069687765735, -0.006587528866314235, -3.530766821016102);

    cv::Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    while(inputVideo.isOpened()){
        inputVideo >> image;
        image.copyTo(imageCopy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        aruco::detectMarkers(image, dictionary, corners, ids);


        if (!ids.empty()) {
            aruco::drawDetectedMarkers(imageCopy, corners, ids);
            std::vector<cv::Vec3d> rvecs, tvecs;
            aruco::estimatePoseSingleMarkers(corners, markerSize, cameraMatrix, distCoeffs, rvecs, tvecs);
            for (int i = 0; i < ids.size(); i++) {
                aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);

                // Print the marker ID and quaternion
                cout << "Marker ID: " << ids[i] << ", Quaternion: " << rvecs[i] << endl;
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
