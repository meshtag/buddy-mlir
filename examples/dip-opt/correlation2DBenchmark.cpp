#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include "../conv-opt/kernels.h"

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

// Declare the conv2d C interface.
extern "C" {
void _mlir_ciface_conv_2d(MemRef_descriptor input, MemRef_descriptor kernel,
                          MemRef_descriptor output);
}

// Declare the Corr2D C interface.
extern "C" {
void _mlir_ciface_dipCorr2D(MemRef_descriptor input, MemRef_descriptor kernel,
                            MemRef_descriptor output, unsigned int centerX,
                            unsigned int centerY, int boundaryOption);
}

// Read input image and specify kernel
Mat inputImage = imread("../../examples/conv-opt/images/YuTu.png", IMREAD_GRAYSCALE);
Mat kernel_opencv = Mat(3, 3, CV_32FC1, laplacianKernelAlign);
Mat output_opencv;

// Define the kernel.
float *kernelAlign = laplacianKernelAlign;
int kernelRows = laplacianKernelRows;
int kernelCols = laplacianKernelCols;

// Define output for buddy mlir implementation.
int outputRows = inputImage.rows;
int outputCols = inputImage.cols;
float *outputAlign = (float *)malloc(outputRows * outputCols * sizeof(float));

// Define allocated, sizes, and strides.
float *allocated = (float *)malloc(1 * sizeof(float));
intptr_t sizesInput[2] = {inputImage.rows, inputImage.cols};
intptr_t sizesKernel[2] = {kernelRows, kernelCols};
intptr_t sizesOutput[2] = {outputRows, outputCols};
intptr_t stridesInput[2] = {inputImage.rows, inputImage.cols};
intptr_t stridesKernel[2] = {kernelRows, kernelCols};
intptr_t stridesOutput[2] = {outputRows, outputCols};

float* fill_align(Mat image)
{
  int k = 0;
  float *inputAlign = (float *)malloc(image.rows * image.cols * sizeof(float));
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float pixelValue = (float)image.at<uchar>(i, j);
      inputAlign[k] = pixelValue;
      k++;
    }
  }
  return inputAlign;
}

float *inputAlign = fill_align(inputImage);

// Define memref descriptors.
MemRef_descriptor input =
    MemRef_Descriptor(allocated, inputAlign, 0, sizesInput, stridesInput);
MemRef_descriptor kernel =
    MemRef_Descriptor(allocated, kernelAlign, 0, sizesKernel, stridesKernel);
MemRef_descriptor output =
    MemRef_Descriptor(allocated, outputAlign, 0, sizesOutput, stridesOutput);

// Benchmarking function
static void BM_OpenCV(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // filter2D(inputImage, output_opencv, CV_32FC1, kernel_opencv);
      filter2D(inputImage, output_opencv, CV_8UC1, kernel_opencv, cv::Point(1, 1), 0.0,
           cv::BORDER_REPLICATE);
    }
  }
}

// Benchmarking function
static void BM_Buddy(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_dipCorr2D(input, kernel, output, 1, 1, 1);
    }
  }
}

// Register above functions as benchmarks with different arguments
BENCHMARK(BM_OpenCV)->Arg(1);
BENCHMARK(BM_OpenCV)->Arg(2);
BENCHMARK(BM_OpenCV)->Arg(4);
BENCHMARK(BM_OpenCV)->Arg(8);
BENCHMARK(BM_OpenCV)->Arg(16);

BENCHMARK(BM_Buddy)->Arg(1);
BENCHMARK(BM_Buddy)->Arg(2);
BENCHMARK(BM_Buddy)->Arg(4);
BENCHMARK(BM_Buddy)->Arg(8);
BENCHMARK(BM_Buddy)->Arg(16);

// Run benchmarks
int main(int argc, char** argv)
{
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  free(inputAlign);
  free(outputAlign);
  free(input);
  free(kernel);
  free(output);
  free(allocated);
}
