#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main() {

    cv::VideoCapture cap(0);

    cv::Mat image;

    while (cap.isOpened()){
        cap >> image;
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        cv::aruco::detectMarkers(gray, dictionary, corners, ids);

        if (!ids.empty()) {
            cv::aruco::drawDetectedMarkers(image, corners, ids);
        }
        cv::imshow("image", image);

        int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
