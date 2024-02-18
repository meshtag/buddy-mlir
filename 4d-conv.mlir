#map = affine_map<(d0) -> (d0)>
#map0 = affine_map<(d0, d1) -> (d0 + d1 - 1)>
#map1 = affine_map<(d0) -> (d0 ceildiv 256)>

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @print_flops(f64)
  func.func private @rtclock() -> f64

  func.func @conv(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.splat %cst : vector<256xf32>
    %dim = memref.dim %arg1, %c2 : memref<?x?x?x?xf32>
    %dim_0 = memref.dim %arg1, %c3 : memref<?x?x?x?xf32>

    %dim_0_1 = memref.dim %arg1, %c1 : memref<?x?x?x?xf32>
    %dim_0_0 = memref.dim %arg1, %c0 : memref<?x?x?x?xf32>

    %dim_1 = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
    %dim_2 = memref.dim %arg2, %c3 : memref<?x?x?x?xf32>

    %dim_2_1 = memref.dim %arg2, %c1 : memref<?x?x?x?xf32>
    %dim_2_0 = memref.dim %arg2, %c0 : memref<?x?x?x?xf32>


    
   affine.for %batchiv = #map(%c0) to #map(%dim_2_0) {
    affine.for %argf = #map(%c0) to #map(%dim_0_0) {
      affine.for %argc = #map(%c0) to #map(%dim_0_1) {
          affine.for %arg3 = #map(%c0) to #map(%dim_1) {
          affine.for %arg4 = #map(%c0) to #map(%dim) {
          affine.for %arg5 = #map(%c0) to #map(%dim_0) {
          affine.for %arg6 = #map(%c0) to #map1(%dim_2) {
            %1 = memref.load %arg1[%argf, %argc, %arg4, %arg5] : memref<?x?x?x?xf32>
            %2 = arith.index_cast %c0 : index to i32
            %3 = arith.sitofp %2 : i32 to f32
            %4 = arith.cmpf one, %1, %3 : f32
            scf.if %4 {
              %5 = vector.broadcast %1 : f32 to vector<256xf32>
              %6 = arith.muli %arg6, %c256 : index
              %7 = arith.subi %dim_2, %6 : index
              %8 = arith.cmpi sge, %7, %c256 : index
              scf.if %8 {
                %9 = affine.vector_load %arg0[%batchiv, %argc, %arg3 + %arg4, %arg5 + %arg6 * 256] : memref<?x?x?x?xf32>, vector<256xf32>
                %10 = affine.vector_load %arg2[%batchiv, %argf, %arg3, %arg6 * 256] : memref<?x?x?x?xf32>, vector<256xf32>
                %11 = vector.fma %9, %5, %10 : vector<256xf32>
                affine.vector_store %11, %arg2[%batchiv, %argf, %arg3, %arg6 * 256] : memref<?x?x?x?xf32>, vector<256xf32>
              } else {
                %9 = vector.create_mask %7 : vector<256xi1>
                %10 = arith.addi %arg3, %arg4 : index
                %11 = arith.muli %arg6, %c256 : index
                %12 = arith.addi %arg5, %11 : index
                %13 = vector.maskedload %arg0[%batchiv, %argc, %10, %12], %9, %0 : memref<?x?x?x?xf32>, vector<256xi1>, vector<256xf32> into vector<256xf32>
                %14 = vector.maskedload %arg2[%batchiv, %argf, %arg3, %11], %9, %0 : memref<?x?x?x?xf32>, vector<256xi1>, vector<256xf32> into vector<256xf32>
                %15 = vector.fma %13, %5, %14 : vector<256xf32>
                vector.maskedstore %arg2[%batchiv, %argf, %arg3, %11], %9, %15 : memref<?x?x?x?xf32>, vector<256xi1>, vector<256xf32>
              }
            }
          }
        }
      }
      }
      }
    }
    }
    return
  }

  func.func @alloc_4d_filled_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
    scf.for %arg5 = %c0 to %arg0 step %c1 {
      scf.for %arg6 = %c0 to %arg1 step %c1 {
        scf.for %arg7 = %c0 to %arg2 step %c1 {
          scf.for %arg8 = %c0 to %arg3 step %c1 {
            memref.store %arg4, %0[%arg5, %arg6, %arg7, %arg8] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return %0 : memref<?x?x?x?xf32>
  }

  func.func @main() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // Image and Output value.
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32

    %current_filter = arith.constant 256 : index
    %current_output = arith.constant 512 : index
    %current_image = affine.apply #map0(%current_output, %current_filter)

    // Filter.
    %filter = call @alloc_4d_filled_f32(%c1, %c1, %current_filter, %current_filter, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    // Image.
    %image = call @alloc_4d_filled_f32(%c1, %c1, %current_image, %current_image, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    // Output.
    %output = call @alloc_4d_filled_f32(%c1, %c1, %current_output, %current_output, %cst_0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

    // Execution times.
    %reps = arith.constant 1 : index

    // Record start time.
    %t_start = call @rtclock() : () -> f64

    // Execute convolution for specific times.
    affine.for %arg0 = 0 to %reps {
      func.call @conv(%image, %filter, %output) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()
    }

    // Record end time.
    %t_end = call @rtclock() : () -> f64
    // Get the total running time.
    %t = arith.subf %t_end, %t_start : f64

    vector.print %t : f64

    // Print output.
    %print_output = memref.cast %output : memref<?x?x?x?xf32> to memref<*xf32>
    // call @printMemrefF32(%print_output) : (memref<*xf32>) -> ()

    // 2 * [filter size]^2 * [output size]^2.
    %flops1 = arith.muli %current_output, %current_output : index
    %flops2 = arith.muli %current_filter, %current_filter : index
    %flops3 = arith.muli %c2, %flops2 : index
    %flops4 = arith.muli %flops1, %flops3 : index
    // Calculate FLOPS.
    %num_flops = arith.muli %reps, %flops4 : index
    %num_flops_i = arith.index_cast %num_flops : index to i64
    %num_flops_f = arith.sitofp %num_flops_i : i64 to f64
    %flops = arith.divf %num_flops_f, %t : f64
    // Print the FLOPS.
    // vector.print %flops : f64

    memref.dealloc %image : memref<?x?x?x?xf32>
    memref.dealloc %filter : memref<?x?x?x?xf32>
    memref.dealloc %output : memref<?x?x?x?xf32>
    return
  }
}
