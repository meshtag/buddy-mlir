func @DIPCorr2D(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %boundaryOption : i64)
{
  DIP.Corr2D %inputImage, %kernel, %outputImage, %centerX, %centerY, %boundaryOption : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, i64
  return
}
