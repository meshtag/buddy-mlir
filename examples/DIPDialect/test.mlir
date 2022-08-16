#map0 = affine_map<(d0, d1) -> (d0 + d1 - 1)>
module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @alloc_2d_filled_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
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

  func.func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %cx = arith.constant 1 : index
    %const = arith.constant 0.000000e+00 : f32

    dip.corrfft_2d %arg0, %arg1, %arg2, %arg0, %cx, %cx, %const : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32

    return
  }

  func.func @main() {
    // Image and Output value.
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32

    %current_filter = arith.constant 3 : index
    %current_output = arith.constant 5 : index
    %current_image = arith.constant 5 : index

    // Filter.
    %filter = call @alloc_2d_filled_f32(%current_filter, %current_filter, %cst) : (index, index, f32) -> memref<?x?xf32>
    // Image.
    %image = call @alloc_2d_filled_f32(%current_image, %current_image, %cst) : (index, index, f32) -> memref<?x?xf32>
    // Output.
    %output = call @alloc_2d_filled_f32(%current_output, %current_output, %cst_0) : (index, index, f32) -> memref<?x?xf32>

    // Execution times.
    %reps = arith.constant 1 : index

    // Execute convolution for specific times.
    affine.for %arg0 = 0 to %reps {
      func.call @conv_2d(%image, %filter, %output) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    }

    // Print input.
    %print_input = memref.cast %image : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_input) : (memref<*xf32>) -> ()

    // Print kernel.
    %print_kernel = memref.cast %filter : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_kernel) : (memref<*xf32>) -> ()

    // Print output.
    %print_output = memref.cast %output : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_output) : (memref<*xf32>) -> ()

    memref.dealloc %image : memref<?x?xf32>
    memref.dealloc %filter : memref<?x?xf32>
    memref.dealloc %output : memref<?x?xf32>
    return
  }
}
