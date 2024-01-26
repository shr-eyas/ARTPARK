#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main() {
    int marker_id = 2;
    int marker_size = 200;
    int border_bits = 5;
    cv::Mat markerImage;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(
        cv::aruco::DICT_6X6_250);
    cv::aruco::drawMarker(dictionary, marker_id, marker_size, markerImage, border_bits);
    cv::imwrite("marker_2.png", markerImage);
    return 0;
}
