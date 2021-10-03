#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>

void FillImage(cv::Mat img)
{
  for (std::ptrdiff_t row = 0; row < img.rows; ++row)
    for (std::ptrdiff_t col = 0; col < img.cols; ++col)
      img.at<uchar>img(row, col) = row * img.rows + col * img.cols;
}

void PrintImage(cv::Mat img)
{
    for (std::ptrdiff_t row = 0; row < img.rows; ++row)
    {
        for (std::ptrdiff_t col = 0; col < img.cols; ++col)
          std::cout << img.at<uchar>img(row, col) << " ";
        std::cout << "\n";
    }
}

int main()
{
  cv::Mat img(5, 5);
  FillImage(img);
  PrintImage(img);
}