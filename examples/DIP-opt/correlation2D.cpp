//====- edge-detection.cpp - Example of conv-opt tool ========================//
//
//
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <time.h>

#include "/home/prathamesh/buddy-mlir/examples/conv-opt/kernels.h"

using namespace cv;
using namespace std;

// Define Memref Descriptor.
typedef struct MemRef_descriptor_ *MemRef_descriptor;
typedef struct MemRef_descriptor_ {
  float *allocated;
  float *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
} Memref;

// Constructor
MemRef_descriptor MemRef_Descriptor(float *allocated, float *aligned,
                                    intptr_t offset, intptr_t sizes[2],
                                    intptr_t strides[2]) {
  MemRef_descriptor n = (MemRef_descriptor)malloc(sizeof(*n));
  n->allocated = allocated;
  n->aligned = aligned;
  n->offset = offset;
  for (int i = 0; i < 2; i++)
    n->sizes[i] = sizes[i];
  for (int j = 0; j < 2; j++)
    n->strides[j] = strides[j];

  return n;
}

// Declare the corr2d C interface.
extern "C" {
void _mlir_ciface_DIPCorr2D(MemRef_descriptor input, MemRef_descriptor kernel,
                            MemRef_descriptor output, int centerX, int centerY, int boundaryOption);
}

void printImage(cv::Mat img)
{
  std::cout << "\n";
  for (std::ptrdiff_t row = 0; row < img.rows; ++row)
  {
    for (std::ptrdiff_t col = 0; col < img.cols; ++col)
      std::cout << static_cast<unsigned int>(img.at<uchar>(col, row)) << " ";
    std::cout << "\n";
  }
  std::cout << "\n";
}

void testEquality(cv::Mat img1, cv::Mat img2)
{
  if (img1.rows != img2.rows || img1.cols != img2.cols)
  {
    std::cout << "Image dimensions are not equal\n";
    std::cout << "Img1 dimension : (" << img1.rows << "," << img1.cols << ")\n";
    std::cout << "Img2 dimension : (" << img2.rows << "," << img2.cols << ")\n";
    return;
  }
  bool flag = 1;
  for (std::ptrdiff_t row = 0; row < img1.rows; ++row)
  {
    for (std::ptrdiff_t col = 0; col < img1.cols; ++col)
    {
      if (img1.at<uchar>(col, row) != img2.at<uchar>(col, row))
      {
        std::cout << "Images are not same\n";
        std::cout << "They are different at : (" << col << "," << row << ")\n";
        std::cout << static_cast<unsigned int>(img1.at<uchar>(col, row)) << " "
                  << static_cast<unsigned int>(img2.at<uchar>(col, row)) << "\n";
        flag = 0;
        break;
      }
    }
    if (!flag)
      break;
  }
}

int main(int argc, char *argv[]) {
  printf("Start processing...\n");

  // Read as grayscale image.
  Mat image = imread(argv[1], IMREAD_GRAYSCALE);
  if (image.empty()) {
    cout << "Could not read the image: " << argv[1] << endl;
    return 1;
  }

  int inputSize = image.rows * image.cols;

  // Define the input with the image.
  float *inputAlign = (float *)malloc(inputSize * sizeof(float));
  int k = 0;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float pixelValue = (float)image.at<uchar>(i, j);
      inputAlign[k] = pixelValue;
      k++;
    }
  }

  // Define the kernel.
  float kernelAlign[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  // float *kernelAlign = laplacianKernelAlign;
  int kernelRows = laplacianKernelRows;
  int kernelCols = laplacianKernelCols;

  // Define the output.
  int outputRows = image.rows - kernelRows + 1;
  int outputCols = image.cols - kernelCols + 1;
  float *outputAlign = (float *)malloc(outputRows * outputCols * sizeof(float));

  // Define the allocated, sizes, and strides.
  float *allocated = (float *)malloc(1 * sizeof(float));
  intptr_t sizesInput[2] = {image.rows, image.cols};
  intptr_t sizesKernel[2] = {kernelRows, kernelCols};
  intptr_t sizesOutput[2] = {outputRows, outputCols};
  intptr_t stridesInput[2] = {image.rows, image.cols};
  intptr_t stridesKernel[2] = {kernelRows, kernelCols};
  intptr_t stridesOutput[2] = {outputRows, outputCols};

  // Define memref descriptors.
  MemRef_descriptor input =
      MemRef_Descriptor(allocated, inputAlign, 0, sizesInput, stridesInput);
  MemRef_descriptor kernel =
      MemRef_Descriptor(allocated, kernelAlign, 0, sizesKernel, stridesKernel);
  MemRef_descriptor output =
      MemRef_Descriptor(allocated, outputAlign, 0, sizesOutput, stridesOutput);

  // Choose a PNG compression level
  vector<int> compression_params;
  compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  for (std::ptrdiff_t row = 0; row < kernelRows; ++row)
  {
    for (std::ptrdiff_t col = 0; col < kernelCols; ++col)
    {
      _mlir_ciface_DIPCorr2D(input, kernel, output, col, row, 0);

       // Define a cv::Mat with the output of the conv2d.
      Mat outputImage(outputRows, outputCols, CV_32FC1, output->aligned);
      imwrite(argv[2], outputImage, compression_params);
      Mat imageOut = imread(argv[2], IMREAD_GRAYSCALE);


    }
  }

  printImage(image);

  std::cout << "Here\n";

  // testEquality(image, imageOut);

  free(inputAlign);
  free(outputAlign);
  free(input);
  free(kernel);
  free(output);

  return 0;
}
