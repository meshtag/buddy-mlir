//===- dip.h --------------------------------------------------------------===//
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

#include "Interface/buddy/core/ImageContainer.h"

namespace dip {
// Availale types of boundary extrapolation techniques provided in DIP dialect.
enum class BOUNDARY_OPTION { CONSTANT_PADDING, REPLICATE_PADDING };

// Available ways of specifying angles in image processing operations provided
// by the DIP dialect.
enum class ANGLE_TYPE { DEGREE, RADIAN };

// Available ways of interpolation techniques in image processing operations
// provided by the DIP dialect.
enum class INTERPOLATION_TYPE {
  NEAREST_NEIGHBOUR_INTERPOLATION,
  BILINEAR_INTERPOLATION
};

namespace detail {
// Functions present inside dip::detail are not meant to be called by users
// directly.

// Declare the Corr2D C interface.
extern "C" {
void _mlir_ciface_corr_2d_constant_padding(
    Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    unsigned int centerX, unsigned int centerY, float constantValue);

void _mlir_ciface_corr_2d_replicate_padding(
    Img<float, 2> *input, MemRef<float, 2> *kernel, MemRef<float, 2> *output,
    unsigned int centerX, unsigned int centerY, float constantValue);

// Declare the Rotate2D C interface.
void _mlir_ciface_rotate_2d(Img<float, 2> *input, float angleValue,
                            MemRef<float, 2> *output);

// Declare the Resize2D C interface.
void _mlir_ciface_resize_2d_nearest_neighbour_interpolation(
    Img<float, 2> *input, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 2> *output);

void _mlir_ciface_resize_2d_bilinear_interpolation(
    Img<float, 2> *input, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 2> *output);

void _mlir_ciface_corrfft_2d(
    Img<float, 2> *inputReal, MemRef<float, 2> *inputImag,
    MemRef<float, 2> *kernelReal, MemRef<float, 2> *kernelImag,
    MemRef<float, 2> *outputReal, MemRef<float, 2> *outputImag,
    MemRef<float, 2> *intermediateReal,
    MemRef<float, 2> *intermediateImag,
    unsigned int centerX, unsigned int centerY, float constantValue);
}

// Helper function for applying 2D resize operation on images.
MemRef<float, 2> Resize2D_Impl(Img<float, 2> *input, INTERPOLATION_TYPE type,
                               std::vector<float> scalingRatios,
                               intptr_t outputSize[2]) {
  MemRef<float, 2> output(outputSize);

  if (type == INTERPOLATION_TYPE::NEAREST_NEIGHBOUR_INTERPOLATION) {
    detail::_mlir_ciface_resize_2d_nearest_neighbour_interpolation(
        input, scalingRatios[0], scalingRatios[1], &output);
  } else if (type == INTERPOLATION_TYPE::BILINEAR_INTERPOLATION) {
    detail::_mlir_ciface_resize_2d_bilinear_interpolation(
        input, scalingRatios[0], scalingRatios[1], &output);
  } else {
    throw std::invalid_argument(
        "Please chose a supported type of interpolation "
        "(Nearest neighbour interpolation or Bilinear interpolation)\n");
  }

  return output;
}

// Pad kernel as per the requirements for using FFT in convolution.
void padKernel(MemRef<float, 2> *kernel, unsigned int centerX, unsigned int centerY, 
               intptr_t *paddedSizes, float *kernelPaddedData)
{
  // Apply padding so that the center of kernel is at top left of 2D padded container.
  for (long i = -static_cast<long>(centerY); i < static_cast<long>(kernel->getSizes()[0]) - centerY;
       ++i) {
    uint32_t r = (i < 0) ? (i + paddedSizes[0]) : i;
    for (long j = -static_cast<long>(centerX); j < static_cast<long>(kernel->getSizes()[1]) - 
         centerX; ++j) {
      uint32_t c = (j < 0) ? (j + paddedSizes[1]) : j;
      kernelPaddedData[r * paddedSizes[1] + c] = 
          kernel->getData()[(i + centerY) * kernel->getSizes()[1] + j + centerX];
    }
  }
}
} // namespace detail

// User interface for 2D Correlation.
void Corr2D(Img<float, 2> *input, MemRef<float, 2> *kernel,
            MemRef<float, 2> *output, unsigned int centerX,
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

void CorrFFT2D(Img<float, 2> *input, MemRef<float, 2> *kernel,
            MemRef<float, 2> *output, unsigned int centerX,
            unsigned int centerY, BOUNDARY_OPTION option,
            float constantValue = 0) {
  // Calculate padding sizes.
  intptr_t paddedSizes[2] = {
    1<<((uint8_t)ceil(log2(input->getSizes()[0] + kernel->getSizes()[0] - 1))),
    1<<((uint8_t)ceil(log2(input->getSizes()[1] + kernel->getSizes()[1] - 1)))
  };
  intptr_t paddedTSizes[2] = {paddedSizes[1], paddedSizes[0]};
	intptr_t paddedSize = paddedSizes[0] * paddedSizes[1];

  float flippedKernelData[kernel->getSizes()[0] * kernel->getSizes()[1]];
  for (uint32_t i = 0; i < kernel->getSizes()[0]; ++i)
    for (uint32_t j = 0; j < kernel->getSizes()[1]; ++j)
    {
      flippedKernelData[i * kernel->getSizes()[1] + j] = 
        kernel->getData()[(kernel->getSizes()[0] - 1 -i)*kernel->getSizes()[1] + 
        kernel->getSizes()[1] - 1 - j];
    }

  // std::cout << "Flipped kernel here\n";
  // for (uint32_t i = 0; i < kernel->getSizes()[0]; ++i)
  // {
  //   for (uint32_t j = 0; j < kernel->getSizes()[1]; ++j)
  //     std::cout << flippedKernelData[i * kernel->getSizes()[1] + j] << " ";
  //   std::cout << "\n";
  // }
  // std::cout << "\n\n";

  // Obtain padded input image.
  float inputPaddedDataReal[paddedSize] = {0};
  for (uint32_t i = 0; i < input->getSizes()[0]; ++i) {
    for (uint32_t j = 0; j < input->getSizes()[1]; ++j) {
      inputPaddedDataReal[i * paddedSizes[1] + j] = input->getData()[i * input->getSizes()[1] + j];
    }
  }

  // Do padding related modifications in above step. Constant padding with zero as constant for now.
  // flip kernel for correlation instead of convolution.

  intptr_t kernelSize[2] = {kernel->getSizes()[0], kernel->getSizes()[1]};
  MemRef<float, 2> flippedKernel(flippedKernelData, kernelSize);

  // Obtain padded kernel.
  // float *kernelPaddedDataReal = new float[paddedSize];
  float kernelPaddedDataReal[paddedSize] = {0};
  // detail::padKernel(kernel, centerX, centerY, paddedSizes, kernelPaddedDataReal);
  detail::padKernel(&flippedKernel, centerX, centerY, paddedSizes, kernelPaddedDataReal);

  // Declare padded containers for input image and kernel.
  // Also declare an intermediate container for calculation convenience.
  Img<float, 2> inputPaddedReal(inputPaddedDataReal, paddedSizes);
  MemRef<float, 2> inputPaddedImag(paddedSizes);

  MemRef<float, 2> kernelPaddedReal(kernelPaddedDataReal, paddedSizes);
  MemRef<float, 2> kernelPaddedImag(paddedSizes);

  MemRef<float, 2> intermediateReal(paddedTSizes);
  MemRef<float, 2> intermediateImag(paddedTSizes);

  MemRef<float, 2> outputReal(paddedSizes);
  MemRef<float, 2> outputImag(paddedSizes);

  // std::cout << "Here Here\n";
  // std::cout << intermediateReal.getSizes()[0] << "  " << intermediateReal.getSizes()[1] << "\n";
  // std::cout << inputPaddedReal.getSizes()[0] << "  " << inputPaddedReal.getSizes()[1] << "\n";
  // std::cout << input->getSizes()[0] << "  " << input->getSizes()[1] << "\n";

  detail::_mlir_ciface_corrfft_2d(
        &inputPaddedReal, &inputPaddedImag, &kernelPaddedReal, &kernelPaddedImag,
        &outputReal, &outputImag, &intermediateReal, &intermediateImag,
        centerX, centerY, constantValue);

  for (uint32_t i = 0; i < output->getSizes()[0]; ++i)
    for (uint32_t j = 0; j < output->getSizes()[1]; ++j)
      output->getData()[i * output->getSizes()[1] + j] = inputPaddedReal.getData()[i * paddedSizes[1] + j];

  // std::cout << paddedSizes[0] << " " << paddedSizes[1] << "\n\n";

  // for (int i = 0; i < 8; ++i)
  // {
  //   for (int j = 0; j < 8; ++j)
  //     std::cout << inputPaddedReal.getData()[8*i + j] << " ";
  //   std::cout << "\n";
  // }

  // std::cout << "\n";

  // for (int i = 0; i < 8; ++i)
  // {
  //   for (int j = 0; j < 8; ++j)
  //     std::cout << inputPaddedImag.getData()[8*i + j] << " ";
  //   std::cout << "\n";
  // }

  // std::cout << "\n";

  // for (uint32_t i = 0; i < output->getSizes()[0]; ++i)
  // {
  //   for (uint32_t j = 0; j < output->getSizes()[1]; ++j)
  //     std::cout << output->getData()[i * output->getSizes()[1] + j] << " ";
  //   std::cout << "\n";
  // }

  // for (int i = 0; i < 8; ++i)
  // {
  //   for (int j = 0; j < 8; ++j)
  //     std::cout << kernelPaddedReal.getData()[8*i + j] << " ";
  //   std::cout << "\n";
  // }

  // std::cout << "\n";

  // for (int i = 0; i < 8; ++i)
  // {
  //   for (int j = 0; j < 8; ++j)
  //     std::cout << kernelPaddedImag.getData()[8*i + j] << " ";
  //   std::cout << "\n";
  // }

  // delete[] inputPaddedDataReal;
  // delete[] kernelPaddedDataReal;
  // delete[] generalPaddedContainer;
}

MemRef<float, 2> Rotate2D(Img<float, 2> *input, float angle,
                          ANGLE_TYPE angleType) {
  float angleRad;

  if (angleType == ANGLE_TYPE::DEGREE)
    angleRad = M_PI * angle / 180;
  else
    angleRad = angle;

  float sinAngle = std::sin(angleRad);
  float cosAngle = std::cos(angleRad);

  int outputRows = std::round(std::abs(input->getSizes()[0] * cosAngle) +
                              std::abs(input->getSizes()[1] * sinAngle)) +
                   1;
  int outputCols = std::round(std::abs(input->getSizes()[1] * cosAngle) +
                              std::abs(input->getSizes()[0] * sinAngle)) +
                   1;

  intptr_t sizesOutput[2] = {outputRows, outputCols};
  MemRef<float, 2> output(sizesOutput);

  detail::_mlir_ciface_rotate_2d(input, angleRad, &output);

  return output;
}

// User interface for 2D Resize.
MemRef<float, 2> Resize2D(Img<float, 2> *input, INTERPOLATION_TYPE type,
                          std::vector<float> scalingRatios) {
  if (!scalingRatios[0] || !scalingRatios[1]) {
    throw std::invalid_argument(
        "Please enter non-zero values of scaling ratios.\n"
        "Note : scaling ratio = "
        "input_image_dimension / output_image_dimension\n");
  }

  intptr_t outputSize[2] = {
      static_cast<unsigned int>(input->getSizes()[0] / scalingRatios[1]),
      static_cast<unsigned int>(input->getSizes()[1] / scalingRatios[0])};

  return detail::Resize2D_Impl(input, type, scalingRatios, outputSize);
}

// User interface for 2D Resize.
MemRef<float, 2> Resize2D(Img<float, 2> *input, INTERPOLATION_TYPE type,
                          intptr_t outputSize[2]) {
  if (!outputSize[0] || !outputSize[1]) {
    throw std::invalid_argument(
        "Please enter non-zero values of output dimensions.\n");
  }

  std::vector<float> scalingRatios(2);
  scalingRatios[1] = input->getSizes()[0] * 1.0f / outputSize[0];
  scalingRatios[0] = input->getSizes()[1] * 1.0f / outputSize[1];

  return detail::Resize2D_Impl(input, type, scalingRatios, outputSize);
}
} // namespace dip
#endif
