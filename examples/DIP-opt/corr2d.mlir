func @DIPCorr2D(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : i32, %centerY : i32, %boundaryOption : i32)
{
  DIP.Corr2D %inputImage, %kernel, %outputImage, %centerX, %centerY, %boundaryOption : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, i32, i32, i32
  return
}