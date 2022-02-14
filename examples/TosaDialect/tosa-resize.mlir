module {
  func private @print_memref_f32(memref<*xf32>)

  func @alloc_2d_filled_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%c1, %c1, %arg0, %arg1) : memref<?x?x?x?xf32>
    scf.for %arg3 = %c0 to %arg0 step %c1 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        // memref.store %arg2, %0[%c0, %arg3, %arg4, %c0] : memref<?x?x?x?xf32>
        memref.store %arg2, %0[%c0, %c0, %arg3, %arg4] : memref<?x?x?x?xf32>
      }
    }

    return %0 : memref<?x?x?x?xf32>
  }

  func @main() {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %cst = arith.constant 2.000000e+00 : f32

    %current_filter = arith.constant 4 : index

    // Filter.
    %filter = call @alloc_2d_filled_f32(%current_filter, %current_filter, %cst) : (index, index, f32) -> memref<?x?x?x?xf32>
    %real_out = bufferization.to_tensor %filter : memref<?x?x?x?xf32>

    %resize_out = "tosa.resize"(%real_out) { output_size = [2, 2], stride = [0, 0], offset = [0, 0], stride_fp = [0.0 : f32, 0.0 : f32], offset_fp = [0.0 : f32, 0.0 : f32], shift = 0 : i32, mode = "NEAREST_NEIGHBOR" } : (tensor<?x?x?x?xf32>)  -> (tensor<?x?x?x?xf32>)
    %resize_out_memref = bufferization.to_memref %resize_out : memref<?x?x?x?xf32>

    // Print output.
    %print_output_resize = memref.cast %resize_out_memref : memref<?x?x?x?xf32> to memref<*xf32>
    // call @print_memref_f32(%print_output_resize) : (memref<*xf32>) -> ()

    %real_out_memref = bufferization.to_memref %real_out : memref<?x?x?x?xf32>

    // Print output.
    %print_output = memref.cast %real_out_memref : memref<?x?x?x?xf32> to memref<*xf32>
    call @print_memref_f32(%print_output) : (memref<*xf32>) -> ()

    memref.dealloc %filter : memref<?x?x?x?xf32>
    return
  }
}
