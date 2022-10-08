#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
//#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

char *image_path;
int step;
cv::Mat src;
cv::Mat src_grey;
cv::Mat src_grey_scale;
cv::Mat dst;
cv::Mat dst_otsu;
cv::Mat dst_binary;
cv::Mat dst_canny;
cv::Mat dst_hough;
cv::Mat dst_gaussian;
cv::Mat dst_regression;
int threshold_value = 52;
int maxval = 255;
int sliderValue = 0;
int angleTh = 5;
int lengthTh = 10;

const int ratioThreshold = 2;
const int houghThreshold = 100;
const int minLineLength = 100;
const int maxLineGap = 5;

//Binary threshold variable
int threshold = 70;

//Polynomial regression function
std::vector<double> fitPoly(std::vector<cv::Point> points, int n)
{
  //Number of points
  int nPoints = points.size();

  //Vectors for all the points' xs and ys
  std::vector<float> xValues = std::vector<float>();
  std::vector<float> yValues = std::vector<float>();

  //Split the points into two vectors for x and y values
  for(int i = 0; i < nPoints; i++)
  {
    xValues.push_back(points[i].x);
    yValues.push_back(points[i].y);
  }

  //Augmented matrix
  double matrixSystem[n+1][n+2];
  for(int row = 0; row < n+1; row++)
  {
    for(int col = 0; col < n+1; col++)
    {
      matrixSystem[row][col] = 0;
      for(int i = 0; i < nPoints; i++)
        matrixSystem[row][col] += pow(xValues[i], row + col);
    }

    matrixSystem[row][n+1] = 0;
    for(int i = 0; i < nPoints; i++)
      matrixSystem[row][n+1] += pow(xValues[i], row) * yValues[i];

  }

  //Array that holds all the coefficients
  double coeffVec[n+2] = {};  // the "= {}" is needed in visual studio, but not in Linux

  //Gauss reduction
  for(int i = 0; i <= n-1; i++)
    for (int k=i+1; k <= n; k++)
    {
      double t=matrixSystem[k][i]/matrixSystem[i][i];

      for (int j=0;j<=n+1;j++)
        matrixSystem[k][j]=matrixSystem[k][j]-t*matrixSystem[i][j];

    }

  //Back-substitution
  for (int i=n;i>=0;i--)
  {
    coeffVec[i]=matrixSystem[i][n+1];
    for (int j=0;j<=n+1;j++)
      if (j!=i)
        coeffVec[i]=coeffVec[i]-matrixSystem[i][j]*coeffVec[j];

    coeffVec[i]=coeffVec[i]/matrixSystem[i][i];
  }

  //Construct the cv vector and return it
  std::vector<double> result = std::vector<double>();
  for(int i = 0; i < n+1; i++)
    result.push_back(coeffVec[i]);
  return result;
}

//Returns the point for the equation determined
//by a vector of coefficents, at a certain x location
cv::Point pointAtX(std::vector<double> coeff, double x)
{
  double y = 0;
  for(int i = 0; i < coeff.size(); i++)
  y += pow(x, i) * coeff[i];
  return cv::Point(x, y);
}

static float getLineLength(Vec4i line){
    // Get a lines length
    float x = line[2] - line[0];
    float y = line[3] - line[1];
    return sqrt(pow(x, 2) + pow(y, 2));
}

static float getLineAngle(Vec4i line){
    // Get a lines angle on the horizontal plane
    return atan2(line[3] - line[1], line[2] - line[0]) * 180 / CV_PI;
}

void horizonDetection(int, void*){
    // First, blur the gray scale image for a better Canny filter
    cv::GaussianBlur(src_grey_scale, dst_gaussian, Size(3, 3), 0);

    // !! Canny filter on image - experiment with values
    cv::Canny(dst_gaussian, dst_gaussian, threshold_value, threshold_value* ratioThreshold, 3);

    // Just display a canny filtered image for 1st photo
    dst_canny = Scalar::all(0);
    src_grey_scale.copyTo(dst_canny, dst_gaussian);
    if(step == 0){
        cv::imwrite("1_Canny.png", dst_canny);
        imshow("Image Canny Edge", dst_canny);
        return;
    }

    // Copy image and change colours + declare points and filtered points vectors
    cv::cvtColor(dst_canny, dst_hough, cv::COLOR_GRAY2BGR);
    vector<Vec4i> points;
    vector<Vec4i> filtered;

    // !! Hough Lines transformation filter on image - experiment with values
    cv::HoughLinesP(dst_canny, points, 1, CV_PI/180, houghThreshold, minLineLength, maxLineGap);

    // Filter out lines based on their length and angle
    for(std::size_t  i = 0; i < points.size(); i++){
        float lineLength = getLineLength(points[i]);
        float lineAngle = getLineAngle(points[i]);
        bool condition = false;
        // See which step, depending on what image to show
        if(step == 1){
            filtered = points;
            break;
        }else if(step == 2){
            condition = (lineLength > lengthTh);
        }else if(step == 3){
            condition = (lineAngle <= angleTh && lineAngle >= -angleTh);
        }else if(step == 4 || step == 5){
            condition = (lineLength > lengthTh && lineAngle <= angleTh && lineAngle >= -angleTh);
        }
        // Save points that passed the condition based on the step, into a vector
        if (condition == true){
            filtered.push_back(points[i]);
        }
    }
    // Create the lines based on the filtered points
    std::vector<cv::Point> filteredPoints;
    for(std::size_t  i = 0; i < filtered.size(); i++){
        Vec4i p = filtered[i];
        cv::line(dst_hough, Point(p[0], p[1]), Point(p[2], p[3]), Scalar( 0, 255, 0, 128 ), 2, cv::LINE_AA);
        filteredPoints.push_back(Point(p[0], p[1]));
        filteredPoints.push_back(Point(p[2], p[3]));
    }
    if(step == 1){
        cv::imwrite("2_Image_Hough_Lines.png", dst_hough);
        imshow("Image Hough Lines", dst_hough);
        return;
    }else if(step == 2){
        cv::imwrite("3_Image_Hough_Short_Lines_Removed.png", dst_hough);
        imshow("Image Hough Short Lines Removed", dst_hough);
        return;
    }else if(step == 3){
        cv::imwrite("4_Image_Hough_Horizontal_Lines.png", dst_hough);
        imshow("Image Hough Horizontal Lines", dst_hough);
        return;
    }else if(step == 4){
        cv::imwrite("5_Image_Canny_Hough_Both_Filters.png", dst_hough);
        imshow("Image Canny + Hough (With lines and angles filters)", dst_hough);
        return;
    }

    cv::cvtColor(dst_canny, dst_regression, cv::COLOR_GRAY2BGR);

    // Use the filtered points in regression to determine the horizon on images
    if(step == 5){
        std::vector<double> coeff = fitPoly(filteredPoints, 3);
        for(int i = 0; i < dst_hough.size().width; i++){
            cv::circle(dst_regression, pointAtX(coeff, i), 2, Scalar( 0, 255, 0, 128 ), 2);
        }
        cv::imwrite("6_Image_with_Regression.png", dst_regression);
        imshow("Image with Regression", dst_regression);
        return;
    }

    return;
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
    src_grey_scale = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if(src.empty() || src_grey_scale.empty()){
        printf("Image is empty\n");
        return 1;
    }
    cv::cvtColor(src, src_grey, cv::COLOR_BGR2GRAY, 0);

    // Display image gray scale

    // cv::namedWindow("Image Gray Scale", cv::WINDOW_AUTOSIZE);
    // imshow("Image Gray Scale", src_grey_scale);
    // cv::waitKey(0);
    // cv::destroyWindow("Image Gray Scale");
    // cv::destroyAllWindows;

    // !! TO CORRECTLY DETECT THE HORIZON FOR ALL IMAGES
    // !! YOU HAVE TO PLAY WITH THE TRACKBARS (CANNY FILTER, LINES LENGTH FILTER AND LINES ANGLE FILTER)
    // !! VALUES FOR THE 3 IMAGES IN THE PDF + SAVED SCREENSHOTS IN FOLDERS (horizon1 images, horizon2 images, horizon3 images)

    // Declare initial values for the threshold and filters
    threshold_value = 52;
    angleTh = 5;
    lengthTh = 10;
    step = 0;
    cv::namedWindow("Image Canny Edge", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Canny Edge Value", "Image Canny Edge", &threshold_value, maxval, horizonDetection);
    horizonDetection(0, 0);
    cv::waitKey(0);
    cv::destroyWindow("Image Canny Edge");

    step = 1;
    cv::namedWindow("Image Hough Lines", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Canny Edge Value", "Image Hough Lines", &threshold_value, maxval, horizonDetection);
    horizonDetection(0, 0);
    cv::waitKey(0);
    cv::destroyWindow("Image Hough Lines");
    cv::destroyAllWindows;

    step = 2;
    cv::namedWindow("Image Hough Short Lines Removed", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Canny Edge Value", "Image Hough Short Lines Removed", &threshold_value, maxval, horizonDetection);
    cv::createTrackbar("Line Filter", "Image Hough Short Lines Removed", &lengthTh, maxval, horizonDetection);
    horizonDetection(0, 0);
    cv::waitKey(0);
    cv::destroyWindow("Image Hough Short Lines Removed");
    cv::destroyAllWindows;
    
    step = 3;
    cv::namedWindow("Image Hough Horizontal Lines", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Canny Edge Value", "Image Hough Horizontal Lines", &threshold_value, maxval, horizonDetection);
    cv::createTrackbar("Angle Filter", "Image Hough Horizontal Lines", &angleTh, maxval, horizonDetection);
    horizonDetection(0, 0);
    cv::waitKey(0);
    cv::destroyWindow("Image Hough Horizontal Lines");
    cv::destroyAllWindows;

    step = 4;
    cv::namedWindow("Image Canny + Hough (With lines and angles filters)", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Canny Edge Value", "Image Canny + Hough (With lines and angles filters)", &threshold_value, maxval, horizonDetection);
    cv::createTrackbar("Line Filter", "Image Canny + Hough (With lines and angles filters)", &lengthTh, maxval, horizonDetection);
    cv::createTrackbar("Angle Filter", "Image Canny + Hough (With lines and angles filters)", &angleTh, maxval, horizonDetection);
    horizonDetection(0, 0);
    cv::waitKey(0);
    cv::destroyWindow("Image Canny + Hough (With lines and angles filters)");

    step = 5;
    cv::namedWindow("Image with Regression", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Canny Edge Value", "Image with Regression", &threshold_value, maxval, horizonDetection);
    cv::createTrackbar("Line Filter", "Image with Regression", &lengthTh, maxval, horizonDetection);
    cv::createTrackbar("Angle Filter", "Image with Regression", &angleTh, maxval, horizonDetection);
    horizonDetection(0, 0);
    cv::waitKey(0);
    cv::destroyWindow("Image with Regression");
    cv::destroyAllWindows;

    return 0;
}

