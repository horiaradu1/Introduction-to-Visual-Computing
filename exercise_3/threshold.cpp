#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

char *image_path;
cv::Mat src;
cv::Mat src_grey;
cv::Mat dst_otsu;
cv::Mat dst_binary;
int threshold_value = 0;
int maxval = 255;
int sliderValue = 0;

void on_trackbar(int change, void*){
    cv::threshold(src_grey, dst_binary, change, maxval, cv::THRESH_BINARY);
    cv::imshow("Image threshold BINARY with Trackbar", dst_binary);
}

int main(int argc, char *argv[]){
    // Print the OpenCV version
    printf("OpenCV version: %d.%d\n", CV_MAJOR_VERSION, CV_MINOR_VERSION);

    if(argc == 2){
        image_path = argv[1];
        printf("%s\n", image_path);
    } 
    else{
        printf("Arguments are wrong\n");
        return 1;
    }

    src = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if(src.empty()){
        printf("Image is empty\n");
        return 1;
    }
    cv::cvtColor(src, src_grey, cv::COLOR_BGR2GRAY, 0);

    cv::namedWindow("Source Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Source Image", src);
    cv::waitKey(0);
    
    cv::namedWindow("Source Image - Greyed", cv::WINDOW_AUTOSIZE);
    cv::imshow("Source Image - Greyed", src_grey);
    cv::waitKey(0);

    cv::namedWindow("Image threshold OTSU", cv::WINDOW_AUTOSIZE);
    cv::threshold(src_grey, dst_otsu, threshold_value, maxval, cv::THRESH_OTSU);
    cv::imshow("Image threshold OTSU", dst_otsu);
    cv::waitKey(0);
    
    cv::namedWindow("Image threshold BINARY with Trackbar", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Threshold Value", "Image threshold BINARY with Trackbar", &sliderValue, 255, on_trackbar);
    on_trackbar(sliderValue, NULL);
    cv::waitKey(0);

    cv::imwrite("dst_otsu.png", dst_otsu);
    cv::imwrite("dst_binary.png", dst_binary);

    return 0;
}