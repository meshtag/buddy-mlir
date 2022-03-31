//===- dip.h --------------------------------------------------------===//
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
// Header file for DIP dialect specific operations and other entities.
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_DIP
#define INCLUDE_DIP

#include <Interface/buddy/dip/memref.h>

namespace dip {
namespace detail {
// Functions present inside dip::detail are not meant to be called by users
// directly.
// Declare the Corr2D C interface.
extern "C" {
void _mlir_ciface_corr_2d_constant_padding(
    MemRef_descriptor input, MemRef_descriptor kernel, MemRef_descriptor output,
    unsigned int centerX, unsigned int centerY, float constantValue);

void _mlir_ciface_corr_2d_replicate_padding(
    MemRef_descriptor input, MemRef_descriptor kernel, MemRef_descriptor output,
    unsigned int centerX, unsigned int centerY, float constantValue);
}
} // namespace detail

enum class BOUNDARY_OPTION { CONSTANT_PADDING, REPLICATE_PADDING };

void Corr2D(MemRef_descriptor input, MemRef_descriptor kernel,
            MemRef_descriptor output, unsigned int centerX,
            unsigned int centerY, BOUNDARY_OPTION option,
            float constantValue = 0) {
  if (option == BOUNDARY_OPTION::CONSTANT_PADDING) {
    detail::_mlir_ciface_corr_2d_constant_padding(
        input, kernel, output, centerX, centerY, constantValue);
  } else if (option == BOUNDARY_OPTION::REPLICATE_PADDING) {
    detail::_mlir_ciface_corr_2d_replicate_padding(input, kernel, output,
                                                   centerX, centerY, 0);
  }
}

MemRef_descriptor matToMemRef(cv::Mat container, bool b1 = 0)
{
  std::size_t containerSize = container.rows * container.cols;
  float *containerAlign = (float *)malloc(containerSize * sizeof(float));

  for (int i = 0; i < container.rows; i++) {
    for (int j = 0; j < container.cols; j++) {
      if (!b1)
        containerAlign[container.rows * i + j] = (float)container.at<float>(i, j);
      else 
        containerAlign[container.rows * i + j] = (float)container.at<uchar>(i, j);
    }
  }

  float *allocated = (float *)malloc(1 * sizeof(float));
  intptr_t sizesContainer[2] = {container.rows, container.cols};
  intptr_t stridesContainer[2] = {container.rows, container.cols};

  MemRef_descriptor containerMemRef =
      MemRef_Descriptor(allocated, containerAlign, 0, sizesContainer, stridesContainer);

  return containerMemRef;
}

void Corr2D_nchannels(cv::Mat &inputImage, cv::Mat &kernel, cv::Mat &outputImage, 
                         unsigned int centerX, unsigned int centerY, BOUNDARY_OPTION option, 
                         float constantValue = 0)
{
  std::vector<cv::Mat> inputChannels, outputChannels;
  std::vector<MemRef_descriptor> inputChannelMemRefs, outputChannelMemRefs;

  cv::split(inputImage, inputChannels);
  cv::split(outputImage, outputChannels);
  MemRef_descriptor kernelMemRef = matToMemRef(kernel);
  
  for (auto cI : inputChannels)
    inputChannelMemRefs.push_back(matToMemRef(cI, 1));

  for (auto cO : outputChannels)
    outputChannelMemRefs.push_back(matToMemRef(cO, 1));

  for (int i1 = 0; i1 < inputImage.channels(); ++i1)
  {
    dip::Corr2D(inputChannelMemRefs[i1], kernelMemRef, outputChannelMemRefs[i1], 
                centerX, centerY, option, constantValue);
  }

  outputChannels.clear();
  for (int i = 0; i < inputImage.channels(); ++i)
    outputChannels.push_back(cv::Mat(inputImage.rows, inputImage.cols, CV_32FC1, 
                      outputChannelMemRefs[i]->aligned));
  
  cv::merge(outputChannels, outputImage);
}
} // namespace dip
#endif
