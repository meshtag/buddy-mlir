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

#include "../kernels.h"
#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImageContainer.h>

using namespace cv;
using namespace std;

void check_filter_2d_n_channels() {
  Mat image = imread("../../examples/images/YuTu.png", IMREAD_COLOR), output1,
      output2;

  intptr_t kernelSize[2] = {3, 3};
  MemRef<float, 2> kernel(laplacianKernelAlign, kernelSize);
  Mat kernel1 = Mat(3, 3, CV_32FC1, laplacianKernelAlign);

  dip::Corr2DNChannels(image, &kernel, output1, 0, 0,
                       dip::BOUNDARY_OPTION::CONSTANT_PADDING);
  filter2D(image, output2, CV_32FC3, kernel1, cv::Point(0, 0), 0.0,
           cv::BORDER_CONSTANT);

  imwrite("dip-check-output1.png", output1);
  imwrite("opencv-check-output.png", output2);
}

int main(int argc, char *argv[]) {
  check_filter_2d_n_channels();

  return 0;
}
