//===- correlation2D.cpp - Example of buddy-opt tool ----------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements a 2D correlation example with dip.corr_2d operation.
// The dip.corr_2d operation will be compiled into an object file with the
// buddy-opt tool.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//

#include <opencv2/opencv.hpp>

#include "../ConvOpt/kernels.h"
#include "Interface/buddy/core/ImageContainer.h"
#include <Interface/buddy/dip/dip.h>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

bool testImages(cv::Mat img1, cv::Mat img2) {
  if (img1.rows != img2.rows || img1.cols != img2.cols) {
    std::cout << "Dimensions not equal\n";
    return 0;
  }

  for (std::ptrdiff_t i = 0; i < img1.cols; ++i) {
    for (std::ptrdiff_t j = 0; j < img1.rows; ++j) {
      if (img1.at<uchar>(j, i) != img2.at<uchar>(j, i)) {
        // std::cout << "Pixels not equal at : (" << i << "," << j << ")\n";
        std::cout << (int)img1.at<uchar>(j, i) << "\n";
        std::cout << (int)img2.at<uchar>(j, i) << "\n\n";

        std::cout << img1 << "\n\n";
        std::cout << img2 << "\n\n";
        return 0;
      }
    }
  }

  // std::cout << img1 << "\n\n";
  // std::cout << img2 << "\n";

  return 1;
}

bool testImplementation(int argc, char *argv[], std::ptrdiff_t x,
                        std::ptrdiff_t y, std::ptrdiff_t boundaryOption) {
  // Read as grayscale image.
  Mat imageOrig = imread(argv[1], IMREAD_GRAYSCALE);
  Mat image;
  if (imageOrig.empty()) {
    cout << "Could not read the image: " << argv[1] << endl;
  }

  cv::resize(imageOrig, image, Size(100, 200), cv::INTER_LINEAR);

  // for (int i = 0; i < 7; ++i)
  // {
  //   for (int j = 0; j < 4; ++j)
  //     image.at<uchar>(j, i) = (uchar)13;
  // }
  // image.at<uchar>(2, 2) = (uchar)12;

  // for (int i = 0; i < 4; ++i)
  // {
  //   for (int j = 0; j < 4; ++j)
  //     std::cout << (int)image.at<uchar>(i, j) << " ";
  //   std::cout << "\n";
  // }

  // Define the kernel.
  float *kernelAlign = sobel3x3KernelAlign;
  int kernelRows = sobel3x3KernelRows;
  int kernelCols = sobel3x3KernelCols;

  // Define sizes and strides.
  intptr_t sizesKernel[2] = {kernelRows, kernelCols};
  intptr_t sizesOutput[2] = {image.rows, image.cols};

  // Define memref containers.
  Img<float, 2> input(image);
  MemRef<float, 2> kernel(kernelAlign, sizesKernel);
  MemRef<float, 2> output1(sizesOutput);
  MemRef<float, 2> output2(sizesOutput);

  Mat kernel1 = Mat(3, 3, CV_32FC1, sobel3x3KernelAlign);

  // Call the MLIR Corr2D function.
  // dip::Corr2D(&input, &kernel, &output1, x, y,
  //             dip::BOUNDARY_OPTION::REPLICATE_PADDING);
  dip::CorrFFT2D(&input, &kernel, &output1, 2 - x, 2 - y,
              dip::BOUNDARY_OPTION::REPLICATE_PADDING);

  // Define a cv::Mat with the output of Corr2D.
  Mat outputImageReplicatePadding(sizesOutput[0], sizesOutput[1], CV_32FC1,
                                  output1.getData());
  imwrite(argv[2], outputImageReplicatePadding);

  Mat o1 = imread(argv[2], IMREAD_GRAYSCALE);
  Mat opencvConstantPadding, opencvReplicatePadding;
  filter2D(image, opencvReplicatePadding, CV_8UC1, kernel1, cv::Point(x, y),
           0.0, cv::BORDER_REPLICATE);

  // if (!testImages(o1, opencvReplicatePadding)) {
  //   std::cout << "x, y = " << x << ", " << y << "\n";
  //   return 0;
  // }

  // // Call the MLIR Corr2D function.
  // dip::Corr2D(&input, &kernel, &output2, x, y,
  //             dip::BOUNDARY_OPTION::CONSTANT_PADDING, 0);

  // Define a cv::Mat with the output of Corr2D.
  Mat outputImageConstantPadding(sizesOutput[0], sizesOutput[1], CV_32FC1,
                                 output2.getData());
  imwrite(argv[3], outputImageConstantPadding);

  Mat o2 = imread(argv[3], IMREAD_GRAYSCALE);
  filter2D(image, opencvConstantPadding, CV_8UC1, kernel1, cv::Point(x, y), 0.0,
           cv::BORDER_CONSTANT);

  // std::cout << "OpenCV here\n";
  // for (int i = 0; i < 4; ++i)
  // {
  //   for (int j = 0; j < 4; ++j)
  //     std::cout << (int)opencvConstantPadding.at<uchar>(i, j) << " ";
  //   std::cout << "\n";
  // }

  if (!testImages(o1, opencvConstantPadding)) {
    std::cout << "x, y = " << x << ", " << y << "\n";
    return 0;
  }

  return 1;
}

int main(int argc, char *argv[]) {
  bool flag = 1;
  for (std::ptrdiff_t x = 0; x < 3; ++x) {
    for (std::ptrdiff_t y = 0; y < 3; ++y) {
      if (!testImplementation(argc, argv, x, y, 0)) {
        flag = 0;
        break;
      }
      if (!flag) {
        break;
      }
    }
  }

  return 0;
}
