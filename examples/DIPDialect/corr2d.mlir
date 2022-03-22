func @corr_2d(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %boundaryOption : index)
{
  dip.corr_2d %inputImage, %kernel, %outputImage, %centerX, %centerY, %boundaryOption : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index
  return
}

func @rotate_2d(%inputImage : memref<?x?xf32>, %angle : index, %outputImage : memref<?x?xf32>)
{
  dip.rotate_2d %inputImage, %angle, %outputImage : memref<?x?xf32>, index, memref<?x?xf32>
  return
}






  func private @print_memref_f32(memref<*xf32>)
  func @alloc_2d_filled_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.for %arg3 = %c0 to %arg0 step %c1 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        memref.store %arg2, %0[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    return %0 : memref<?x?xf32>
  }

  func @changeImage(%0 : memref<?x?xf32>, %arg0 : index, %arg1 : index) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %1 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>

    scf.for %arg3 = %c1 to %arg0 step %c2 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        %h11 = memref.load %0[%arg3, %arg4] : memref<?x?xf32>
        memref.store %h11, %1[%arg3, %arg4] : memref<?x?xf32>
      }
    }

    scf.for %arg3 = %c0 to %arg0 step %c2 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        %c11_f32 = arith.constant 1.0 : f32
        %h1 = memref.load %0[%arg3, %arg4] : memref<?x?xf32>
        %arg2 = arith.addf %h1, %c11_f32 : f32
        memref.store %arg2, %1[%arg3, %arg4] : memref<?x?xf32>
      }
    }

    %argLast = arith.subi %arg0, %c1 : index
    %77 = arith.constant 7.0 : f32

    scf.for %arg4 = %c0 to %arg1 step %c1 {
        %h11 = memref.load %0[%arg4, %argLast] : memref<?x?xf32>
        memref.store %77, %1[%arg4, %argLast] : memref<?x?xf32>
    }

    return %1 : memref<?x?xf32>
  }

  func @DIPCorr2D(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>) -> memref<?x?xf32>
  {

    %outputSize = arith.constant 19 : index
    %outputVal = arith.constant 0.0 : f32
    %output = call @alloc_2d_filled_f32(%outputSize, %outputSize, %outputVal) : (index, index, f32) -> memref<?x?xf32>

    %centerX = arith.constant 2 : index
    %centerY = arith.constant 0 : index
    %boundaryOption = arith.constant 0 : index

    %angle = arith.constant 90 : index

    // %printOutputImage1 = memref.cast %output : memref<?x?xf32> to memref<*xf32>
    // call @print_memref_f32(%printOutputImage1) : (memref<*xf32>) -> ()

    // dip.corr_2d %inputImage, %kernel, %output, %centerX, %centerY, %boundaryOption : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index
    dip.rotate_2d %inputImage, %angle, %output : memref<?x?xf32>, index, memref<?x?xf32>


    %printOutputImage = memref.cast %output : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%printOutputImage) : (memref<*xf32>) -> ()

    return %output : memref<?x?xf32>
  }

  func @main()
  {
    %cI = arith.constant 2.0 : f32
    %cK = arith.constant 1.0 : f32

    %inputSize = arith.constant 18 : index
    %kernelSize = arith.constant 3 : index

    %inputImage = call @alloc_2d_filled_f32(%inputSize, %inputSize, %cI) : (index, index, f32) -> memref<?x?xf32>
    %kernel = call @alloc_2d_filled_f32(%kernelSize, %kernelSize, %cK) : (index, index, f32) -> memref<?x?xf32>

    %i11 = call @changeImage(%inputImage, %inputSize, %inputSize) : (memref<?x?xf32>, index, index) -> memref<?x?xf32>
    %outputImage = call @DIPCorr2D(%i11, %kernel) : (memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>

    %printInputImage = memref.cast %i11 : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%printInputImage) : (memref<*xf32>) -> ()

    %printKernel = memref.cast %kernel : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%printKernel) : (memref<*xf32>) -> ()

    return
  }

