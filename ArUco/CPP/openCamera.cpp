#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main() {

    cv::VideoCapture cap(0);

    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        cv::imshow("Camera", frame);

        if (cv::waitKey(1) == 27) {
            break;
        }

    }
    
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
