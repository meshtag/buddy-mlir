func @DIPCorr2D(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>) -> memref<?x?xf32>
{
  // %outputImage = DIP.Corr2D %inputImage, %kernel : memref<?x?xf32>, memref<?x?xf32> to memref<?x?xf32>
  // return %outputImage : memref<?x?xf32>
  return %inputImage : memref<?x?xf32>
}