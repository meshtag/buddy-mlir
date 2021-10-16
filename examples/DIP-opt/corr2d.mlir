func @DIPCorr2D(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>)
{
  DIP.Corr2D %inputImage, %kernel, %outputImage : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  return
}