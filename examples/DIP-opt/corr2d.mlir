func @DIPCorr2D(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : i64, %centerY : i64, %boundaryOption : i64)
{
  DIP.Corr2D %inputImage, %kernel, %outputImage, %centerX, %centerY, %boundaryOption : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, i64, i64, i64
  return
}