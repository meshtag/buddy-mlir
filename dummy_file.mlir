module {
  func private @print_memref_f32(memref<*xf32>)
  func private @rtclock() -> f64

  func @alloc_2d_filled_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.for %arg3 = %c0 to %arg0 step %c1 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        memref.store %arg2, %0[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    return %0 : memref<?x?xf32>
  }

  func @main() {
    %cst = constant 2.000000e+00 : f32
    %cst_k = constant 1.000000e+00 : f32
    %cst_o = constant 0.000000e+00 : f32

    %kernel_size = constant 3 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index

    %image_size = constant 6 : index
    %center_y = constant 1 : index
    %center_x = constant 1 : index
    %constant_padding = constant 0.0 : f32
    %stride = constant 3 : index

    %image = call @alloc_2d_filled_f32(%image_size, %image_size, %cst)
               : (index, index, f32) -> memref<?x?xf32>
    %kernel = call @alloc_2d_filled_f32(%kernel_size, %kernel_size, %cst_k)
               : (index, index, f32) -> memref<?x?xf32>
    %output = call @alloc_2d_filled_f32(%image_size, %image_size, %cst_o)
               : (index, index, f32) -> memref<?x?xf32>

    %print_output = memref.cast %image : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%print_output) : (memref<*xf32>) -> ()

    %print_kernel = memref.cast %kernel : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%print_kernel) : (memref<*xf32>) -> ()

    %pseudo_kernel_size_helper = subi %kernel_size, %c1 : index
    %pseudo_image_size = addi %image_size, %pseudo_kernel_size_helper : index

    // Execution times.
    %reps = constant 200 : index

    // Record start time.
    %t_start = call @rtclock() : () -> f64

    // affine.for %rep = 0 to %reps {
    affine.for %ivs0 = %c0 to %image_size {
      affine.for %ivs1 = %c0 to %kernel_size {
        affine.for %iv = %c0 to %image_size step 3 {
          affine.for %iv_v = %c0 to %kernel_size {

            // Handle new kernel column using %iv_v
            %curr_row = addi %ivs0, %ivs1 : index
            %curr_col = addi %iv, %iv_v : index

            %kernel_val = memref.load %kernel[%ivs1, %iv_v] : memref<?x?xf32>
            %kernel_vec = vector.broadcast %kernel_val : f32 to vector<3xf32>

            %im_row = subi %curr_row, %center_y : index
            %im_col = subi %curr_col, %center_x : index

            %pos_row_up = cmpi slt, %curr_row, %center_y : index
            %pos_row_mid_flag = subi %pseudo_kernel_size_helper, %center_y : index
            %pos_row_mid_helper = subi %pseudo_image_size, %pos_row_mid_flag : index
            %pos_row_mid = cmpi slt, %curr_row, %pos_row_mid_helper : index

            %pos_col_left = cmpi slt, %curr_col, %center_x : index
            %pos_col_mid_flag = subi %pseudo_kernel_size_helper, %center_x : index
            // %stride_helper = subi %stride, %c1 : index
            %pos_col_mid_helper = subi %pseudo_image_size, %stride : index
            %pos_col_mid = cmpi slt, %curr_col, %pos_col_mid_helper : index

            // No Comparison required for lower region and right region.

            scf.if %pos_row_up {

              %input_vec = vector.broadcast %constant_padding : f32 to vector<3xf32>

              %res_vec = vector.load %output[%ivs0, %iv] : memref<?x?xf32>, vector<3xf32>
              %out_vec = vector.fma %kernel_vec, %input_vec, %res_vec : vector<3xf32>
              vector.store %out_vec, %output[%ivs0, %iv] : memref<?x?xf32>, vector<3xf32>

            } else {
              scf.if %pos_row_mid {
                scf.if %pos_col_left {

                  %im_mask_initial = vector.create_mask %center_x : vector<3xi1>
                  %padding = vector.broadcast %constant_padding : f32 to vector<3xf32>

                  %c11 = constant 1 : i1
                  %mask_inverter_helper = vector.broadcast %c11 : i1 to vector<3xi1>
                  %im_mask = subi %mask_inverter_helper, %im_mask_initial : vector<3xi1>

                  %input_vec = vector.maskedload %image[%im_row, %c0], %im_mask, %padding
                     : memref<?x?xf32>, vector<3xi1>, vector<3xf32> into vector<3xf32>

                  %res_vec = vector.load %output[%ivs0, %iv] : memref<?x?xf32>, vector<3xf32>
                  %out_vec = vector.fma %kernel_vec, %input_vec, %res_vec : vector<3xf32>
                  vector.store %out_vec, %output[%ivs0, %iv] : memref<?x?xf32>, vector<3xf32>

                } else {
                  scf.if %pos_col_mid {

                    %input_vec = vector.load %image[%im_row, %im_col] : memref<?x?xf32>, vector<3xf32>

                    %res_vec = vector.load %output[%ivs0, %iv] : memref<?x?xf32>, vector<3xf32>
                    %out_vec = vector.fma %kernel_vec, %input_vec, %res_vec : vector<3xf32>
                    vector.store %out_vec, %output[%ivs0, %iv] : memref<?x?xf32>, vector<3xf32>

                  } else {

                    // mid right region
                    %mask_helper = subi %kernel_size, %center_x : index
                    %im_mask = vector.create_mask %mask_helper : vector<3xi1>
                    %padding = vector.broadcast %constant_padding : f32 to vector<3xf32>

                    %input_vec = vector.maskedload %image[%im_row, %im_col], %im_mask, %padding 
                      : memref<?x?xf32>, vector<3xi1>, vector<3xf32> into vector<3xf32>

                    %res_vec = vector.load %output[%ivs0, %iv] : memref<?x?xf32>, vector<3xf32>
                    %out_vec = vector.fma %kernel_vec, %input_vec, %res_vec : vector<3xf32>
                    vector.store %out_vec, %output[%ivs0, %iv] : memref<?x?xf32>, vector<3xf32>

                    %dummy = constant 100 : index
                    %print_output_image = memref.cast %output : memref<?x?xf32> to memref<*xf32>
                    call @print_memref_f32(%print_output_image) : (memref<*xf32>) -> ()
                    vector.print %dummy : index
                    vector.print %curr_col : index
                    vector.print %input_vec : vector<3xf32>
                    vector.print %res_vec : vector<3xf32>
                    vector.print %kernel_vec : vector<3xf32>
                    vector.print %out_vec : vector<3xf32>
                    vector.print %dummy : index

                  }
                }
              } else {

                %input_vec = vector.broadcast %constant_padding : f32 to vector<3xf32>

                %res_vec = vector.load %output[%ivs0, %iv] : memref<?x?xf32>, vector<3xf32>
                %out_vec = vector.fma %kernel_vec, %input_vec, %res_vec : vector<3xf32>
                vector.store %out_vec, %output[%ivs0, %iv] : memref<?x?xf32>, vector<3xf32>

              }
            }
          }
        }
      }
    }
    // }

    // Record end time.
    %t_end = call @rtclock() : () -> f64

    // Get the total running time.
    %t = subf %t_end, %t_start : f64

    // vector.print %t : f64

    %print_output_image = memref.cast %output : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%print_output_image) : (memref<*xf32>) -> ()

    memref.dealloc %image : memref<?x?xf32>
    memref.dealloc %kernel : memref<?x?xf32>
    memref.dealloc %output : memref<?x?xf32>
    return
  }
}

// 0.000180006

// %dummy = constant 100 : index
   //                %print_output_image = memref.cast %output : memref<?x?xf32> to memref<*xf32>
      //             call @print_memref_f32(%print_output_image) : (memref<*xf32>) -> ()
         //          vector.print %dummy : index
            //       vector.print %curr_col : index
               //    vector.print %input_vec : vector<3xf32>
                  // vector.print %res_vec : vector<3xf32>
              //     vector.print %kernel_vec : vector<3xf32>
                 //  vector.print %out_vec : vector<3xf32>
                  // vector.print %curr_col : index
                  // vector.print %ivs1 : index
                  // vector.print %dummy : index
