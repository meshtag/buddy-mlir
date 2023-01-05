#map = affine_map<(d0) -> (d0)>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @alloc_2d_filled_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.for %arg3 = %c0 to %c3 step %c1 {
      scf.for %arg4 = %c0 to %c3 step %c1 {
        memref.store %cst_0, %0[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    scf.for %arg3 = %c0 to %c3 step %c1 {
      scf.for %arg4 = %c3 to %arg1 step %c1 {
        memref.store %cst, %0[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    scf.for %arg3 = %c3 to %arg0 step %c1 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        memref.store %cst, %0[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    return %0 : memref<?x?xf32>
  }
  func.func @alloc_2d_filled_f32_imag(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.for %arg3 = %c0 to %arg0 step %c1 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        memref.store %cst, %0[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    return %0 : memref<?x?xf32>
  }
  func.func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    return
  }
  func.func @main() {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c4_1 = arith.constant 4 : index
    %c4_2 = arith.constant 4 : index
    %c4_3 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = call @alloc_2d_filled_f32(%c4, %c4, %cst) : (index, index, f32) -> memref<?x?xf32>
    %1 = call @alloc_2d_filled_f32_imag(%c4, %c4, %cst) : (index, index, f32) -> memref<?x?xf32>
    %2 = call @alloc_2d_filled_f32(%c4_2, %c4_3, %cst) : (index, index, f32) -> memref<?x?xf32>
    %3 = call @alloc_2d_filled_f32_imag(%c4_2, %c4_3, %cst) : (index, index, f32) -> memref<?x?xf32>
    %4 = call @alloc_2d_filled_f32(%c4_1, %c4_1, %cst_0) : (index, index, f32) -> memref<?x?xf32>
    %5 = call @alloc_2d_filled_f32_imag(%c4_1, %c4_1, %cst_0) : (index, index, f32) -> memref<?x?xf32>
    %6 = call @alloc_2d_filled_f32(%c4_3, %c4_2, %cst_0) : (index, index, f32) -> memref<?x?xf32>
    %7 = call @alloc_2d_filled_f32_imag(%c4_3, %c4_2, %cst_0) : (index, index, f32) -> memref<?x?xf32>
    %c1 = arith.constant 1 : index
    %c1_4 = arith.constant 1 : index
    %cst_5 = arith.constant 0.000000e+00 : f32
    %8 = memref.cast %2 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%8) : (memref<*xf32>) -> ()
    %9 = memref.cast %0 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%9) : (memref<*xf32>) -> ()
    affine.for %arg0 = 0 to %c1 {
      %c0_6 = arith.constant 0 : index
      %c1_7 = arith.constant 1 : index
      %c1_8 = arith.constant 1 : index
      %11 = memref.dim %2, %c0_6 : memref<?x?xf32>
      %12 = memref.dim %2, %c1_7 : memref<?x?xf32>
      %13 = memref.dim %0, %c0_6 : memref<?x?xf32>
      %14 = memref.dim %0, %c1_7 : memref<?x?xf32>
      affine.for %arg1 = #map(%c0_6) to #map(%11) {
        %18 = arith.index_cast %12 : index to i32
        %19 = arith.sitofp %18 : i32 to f32
        %20 = math.log2 %19 : f32
        %21 = arith.fptoui %20 : f32 to i32
        %22 = arith.index_cast %21 : i32 to index
        %cst_9 = arith.constant -6.28318548 : f32
        %c3 = arith.constant 3 : index
        %c55 = arith.constant 55 : index
        %23 = scf.for %arg2 = %c0_6 to %c3 step %c1_7 iter_args(%arg3 = %c0_6) -> (index) {
          // vector.print %arg3 : index
          %25 = arith.addi %arg3, %c1_7 : index
          scf.yield %25 : index
        }
        vector.print %23 : index
        %24:2 = scf.for %arg2 = %c0_6 to %22 step %c1_7 iter_args(%arg3 = %c1_7, %arg4 = %12) -> (index, index) {
          %25 = arith.shrsi %arg4, %c1_7 : index
          %26 = arith.index_cast %arg4 : index to i32
          %27 = arith.sitofp %26 : i32 to f32
          %28 = arith.divf %cst_9, %27 : f32
          %29 = math.cos %28 : f32
          %30 = vector.broadcast %29 : f32 to vector<1xf32>
          %31 = math.sin %28 : f32
          %32 = vector.broadcast %31 : f32 to vector<1xf32>
          scf.for %arg5 = %c0_6 to %arg3 step %c1_7 {
            %34 = arith.muli %arg5, %arg4 : index
            %35 = arith.addi %34, %25 : index
            %36 = arith.divui %35, %c1_8 : index
            %37 = arith.muli %36, %c1_8 : index
            %38 = arith.subi %35, %37 : index
            %cst_10 = arith.constant 1.000000e+00 : f32
            %cst_11 = arith.constant 0.000000e+00 : f32
            %39 = vector.broadcast %cst_10 : f32 to vector<1xf32>
            %40 = vector.broadcast %cst_11 : f32 to vector<1xf32>
            %41:2 = scf.for %arg6 = %34 to %37 step %c1_8 iter_args(%arg7 = %39, %arg8 = %40) -> (vector<1xf32>, vector<1xf32>) {
              %42 = vector.load %2[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %43 = vector.load %3[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %44 = arith.addi %arg6, %25 : index
              %45 = vector.load %2[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %46 = vector.load %3[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %47 = arith.addf %42, %45 : vector<1xf32>
              %48 = arith.addf %43, %46 : vector<1xf32>
              vector.store %47, %2[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              vector.store %48, %3[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %49 = arith.subf %42, %45 : vector<1xf32>
              %50 = arith.subf %43, %46 : vector<1xf32>
              %51 = arith.mulf %49, %arg7 : vector<1xf32>
              %52 = arith.mulf %50, %arg8 : vector<1xf32>
              %53 = arith.mulf %49, %arg8 : vector<1xf32>
              %54 = arith.mulf %50, %arg7 : vector<1xf32>
              %55 = arith.subf %51, %52 : vector<1xf32>
              %56 = arith.addf %53, %54 : vector<1xf32>
              vector.store %55, %2[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              vector.store %56, %3[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %57 = arith.mulf %arg7, %30 : vector<1xf32>
              %58 = arith.mulf %arg8, %32 : vector<1xf32>
              %59 = arith.mulf %arg7, %32 : vector<1xf32>
              %60 = arith.mulf %arg8, %30 : vector<1xf32>
              %61 = arith.subf %57, %58 : vector<1xf32>
              %62 = arith.addf %59, %60 : vector<1xf32>
              scf.yield %61, %62 : vector<1xf32>, vector<1xf32>
            }
          }
          %33 = arith.shli %arg3, %c1_7 : index
          scf.yield %33, %25 : index, index
        }
      }
      affine.for %arg1 = #map(%c0_6) to #map(%11) {
        affine.for %arg2 = #map(%c0_6) to #map(%12) {
          %18 = memref.load %2[%arg1, %arg2] : memref<?x?xf32>
          memref.store %18, %4[%arg2, %arg1] : memref<?x?xf32>
        }
      }
      affine.for %arg1 = #map(%c0_6) to #map(%11) {
        affine.for %arg2 = #map(%c0_6) to #map(%12) {
          %18 = memref.load %3[%arg1, %arg2] : memref<?x?xf32>
          memref.store %18, %5[%arg2, %arg1] : memref<?x?xf32>
        }
      }
      affine.for %arg1 = #map(%c0_6) to #map(%12) {
        %18 = arith.index_cast %11 : index to i32
        %19 = arith.sitofp %18 : i32 to f32
        %20 = math.log2 %19 : f32
        %21 = arith.fptoui %20 : f32 to i32
        %22 = arith.index_cast %21 : i32 to index
        %cst_9 = arith.constant -6.28318548 : f32
        %c3 = arith.constant 3 : index
        %c55 = arith.constant 55 : index
        %23 = scf.for %arg2 = %c0_6 to %c3 step %c1_7 iter_args(%arg3 = %c0_6) -> (index) {
          // vector.print %arg3 : index
          %25 = arith.addi %arg3, %c1_7 : index
          scf.yield %25 : index
        }
        %24:2 = scf.for %arg2 = %c0_6 to %22 step %c1_7 iter_args(%arg3 = %c1_7, %arg4 = %11) -> (index, index) {
          %25 = arith.shrsi %arg4, %c1_7 : index
          %26 = arith.index_cast %arg4 : index to i32
          %27 = arith.sitofp %26 : i32 to f32
          %28 = arith.divf %cst_9, %27 : f32
          %29 = math.cos %28 : f32
          %30 = vector.broadcast %29 : f32 to vector<1xf32>
          %31 = math.sin %28 : f32
          %32 = vector.broadcast %31 : f32 to vector<1xf32>
          scf.for %arg5 = %c0_6 to %arg3 step %c1_7 {
            %34 = arith.muli %arg5, %arg4 : index
            %35 = arith.addi %34, %25 : index
            %36 = arith.divui %35, %c1_8 : index
            %37 = arith.muli %36, %c1_8 : index
            %38 = arith.subi %35, %37 : index
            %cst_10 = arith.constant 1.000000e+00 : f32
            %cst_11 = arith.constant 0.000000e+00 : f32
            %39 = vector.broadcast %cst_10 : f32 to vector<1xf32>
            %40 = vector.broadcast %cst_11 : f32 to vector<1xf32>
            %41:2 = scf.for %arg6 = %34 to %37 step %c1_8 iter_args(%arg7 = %39, %arg8 = %40) -> (vector<1xf32>, vector<1xf32>) {
              %42 = vector.load %4[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %43 = vector.load %5[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %44 = arith.addi %arg6, %25 : index
              %45 = vector.load %4[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %46 = vector.load %5[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %47 = arith.addf %42, %45 : vector<1xf32>
              %48 = arith.addf %43, %46 : vector<1xf32>
              vector.store %47, %4[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              vector.store %48, %5[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %49 = arith.subf %42, %45 : vector<1xf32>
              %50 = arith.subf %43, %46 : vector<1xf32>
              %51 = arith.mulf %49, %arg7 : vector<1xf32>
              %52 = arith.mulf %50, %arg8 : vector<1xf32>
              %53 = arith.mulf %49, %arg8 : vector<1xf32>
              %54 = arith.mulf %50, %arg7 : vector<1xf32>
              %55 = arith.subf %51, %52 : vector<1xf32>
              %56 = arith.addf %53, %54 : vector<1xf32>
              vector.store %55, %4[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              vector.store %56, %5[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %57 = arith.mulf %arg7, %30 : vector<1xf32>
              %58 = arith.mulf %arg8, %32 : vector<1xf32>
              %59 = arith.mulf %arg7, %32 : vector<1xf32>
              %60 = arith.mulf %arg8, %30 : vector<1xf32>
              %61 = arith.subf %57, %58 : vector<1xf32>
              %62 = arith.addf %59, %60 : vector<1xf32>
              scf.yield %61, %62 : vector<1xf32>, vector<1xf32>
            }
          }
          %33 = arith.shli %arg3, %c1_7 : index
          scf.yield %33, %25 : index, index
        }
      }
      %15 = arith.cmpi ne, %11, %12 : index
      scf.if %15 {
        affine.for %arg1 = #map(%c0_6) to #map(%12) {
          affine.for %arg2 = #map(%c0_6) to #map(%11) {
            %18 = memref.load %4[%arg1, %arg2] : memref<?x?xf32>
            memref.store %18, %2[%arg2, %arg1] : memref<?x?xf32>
          }
        }
        affine.for %arg1 = #map(%c0_6) to #map(%12) {
          affine.for %arg2 = #map(%c0_6) to #map(%11) {
            %18 = memref.load %5[%arg1, %arg2] : memref<?x?xf32>
            memref.store %18, %3[%arg2, %arg1] : memref<?x?xf32>
          }
        }
      } else {
        memref.copy %4, %2 : memref<?x?xf32> to memref<?x?xf32>
        memref.copy %5, %3 : memref<?x?xf32> to memref<?x?xf32>
      }
      affine.for %arg1 = #map(%c0_6) to #map(%13) {
        %18 = arith.index_cast %14 : index to i32
        %19 = arith.sitofp %18 : i32 to f32
        %20 = math.log2 %19 : f32
        %21 = arith.fptoui %20 : f32 to i32
        %22 = arith.index_cast %21 : i32 to index
        %cst_9 = arith.constant -6.28318548 : f32
        %c3 = arith.constant 3 : index
        %c55 = arith.constant 55 : index
        %23 = scf.for %arg2 = %c0_6 to %c3 step %c1_7 iter_args(%arg3 = %c0_6) -> (index) {
          // vector.print %arg3 : index
          %25 = arith.addi %arg3, %c1_7 : index
          scf.yield %25 : index
        }
        %24:2 = scf.for %arg2 = %c0_6 to %22 step %c1_7 iter_args(%arg3 = %c1_7, %arg4 = %14) -> (index, index) {
          %25 = arith.shrsi %arg4, %c1_7 : index
          %26 = arith.index_cast %arg4 : index to i32
          %27 = arith.sitofp %26 : i32 to f32
          %28 = arith.divf %cst_9, %27 : f32
          %29 = math.cos %28 : f32
          %30 = vector.broadcast %29 : f32 to vector<1xf32>
          %31 = math.sin %28 : f32
          %32 = vector.broadcast %31 : f32 to vector<1xf32>
          scf.for %arg5 = %c0_6 to %arg3 step %c1_7 {
            %34 = arith.muli %arg5, %arg4 : index
            %35 = arith.addi %34, %25 : index
            %36 = arith.divui %35, %c1_8 : index
            %37 = arith.muli %36, %c1_8 : index
            %38 = arith.subi %35, %37 : index
            %cst_10 = arith.constant 1.000000e+00 : f32
            %cst_11 = arith.constant 0.000000e+00 : f32
            %39 = vector.broadcast %cst_10 : f32 to vector<1xf32>
            %40 = vector.broadcast %cst_11 : f32 to vector<1xf32>
            %41:2 = scf.for %arg6 = %34 to %37 step %c1_8 iter_args(%arg7 = %39, %arg8 = %40) -> (vector<1xf32>, vector<1xf32>) {
              %42 = vector.load %0[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %43 = vector.load %1[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %44 = arith.addi %arg6, %25 : index
              %45 = vector.load %0[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %46 = vector.load %1[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %47 = arith.addf %42, %45 : vector<1xf32>
              %48 = arith.addf %43, %46 : vector<1xf32>
              vector.store %47, %0[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              vector.store %48, %1[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %49 = arith.subf %42, %45 : vector<1xf32>
              %50 = arith.subf %43, %46 : vector<1xf32>
              %51 = arith.mulf %49, %arg7 : vector<1xf32>
              %52 = arith.mulf %50, %arg8 : vector<1xf32>
              %53 = arith.mulf %49, %arg8 : vector<1xf32>
              %54 = arith.mulf %50, %arg7 : vector<1xf32>
              %55 = arith.subf %51, %52 : vector<1xf32>
              %56 = arith.addf %53, %54 : vector<1xf32>
              vector.store %55, %0[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              vector.store %56, %1[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %57 = arith.mulf %arg7, %30 : vector<1xf32>
              %58 = arith.mulf %arg8, %32 : vector<1xf32>
              %59 = arith.mulf %arg7, %32 : vector<1xf32>
              %60 = arith.mulf %arg8, %30 : vector<1xf32>
              %61 = arith.subf %57, %58 : vector<1xf32>
              %62 = arith.addf %59, %60 : vector<1xf32>
              scf.yield %61, %62 : vector<1xf32>, vector<1xf32>
            }
          }
          %33 = arith.shli %arg3, %c1_7 : index
          scf.yield %33, %25 : index, index
        }
      }
      affine.for %arg1 = #map(%c0_6) to #map(%13) {
        affine.for %arg2 = #map(%c0_6) to #map(%14) {
          %18 = memref.load %0[%arg1, %arg2] : memref<?x?xf32>
          memref.store %18, %4[%arg2, %arg1] : memref<?x?xf32>
        }
      }
      affine.for %arg1 = #map(%c0_6) to #map(%13) {
        affine.for %arg2 = #map(%c0_6) to #map(%14) {
          %18 = memref.load %1[%arg1, %arg2] : memref<?x?xf32>
          memref.store %18, %5[%arg2, %arg1] : memref<?x?xf32>
        }
      }
      affine.for %arg1 = #map(%c0_6) to #map(%14) {
        %18 = arith.index_cast %13 : index to i32
        %19 = arith.sitofp %18 : i32 to f32
        %20 = math.log2 %19 : f32
        %21 = arith.fptoui %20 : f32 to i32
        %22 = arith.index_cast %21 : i32 to index
        %cst_9 = arith.constant -6.28318548 : f32
        %c3 = arith.constant 3 : index
        %c55 = arith.constant 55 : index
        %23 = scf.for %arg2 = %c0_6 to %c3 step %c1_7 iter_args(%arg3 = %c0_6) -> (index) {
          // vector.print %arg3 : index
          %25 = arith.addi %arg3, %c1_7 : index
          scf.yield %25 : index
        }
        %24:2 = scf.for %arg2 = %c0_6 to %22 step %c1_7 iter_args(%arg3 = %c1_7, %arg4 = %13) -> (index, index) {
          %25 = arith.shrsi %arg4, %c1_7 : index
          %26 = arith.index_cast %arg4 : index to i32
          %27 = arith.sitofp %26 : i32 to f32
          %28 = arith.divf %cst_9, %27 : f32
          %29 = math.cos %28 : f32
          %30 = vector.broadcast %29 : f32 to vector<1xf32>
          %31 = math.sin %28 : f32
          %32 = vector.broadcast %31 : f32 to vector<1xf32>
          scf.for %arg5 = %c0_6 to %arg3 step %c1_7 {
            %34 = arith.muli %arg5, %arg4 : index
            %35 = arith.addi %34, %25 : index
            %36 = arith.divui %35, %c1_8 : index
            %37 = arith.muli %36, %c1_8 : index
            %38 = arith.subi %35, %37 : index
            %cst_10 = arith.constant 1.000000e+00 : f32
            %cst_11 = arith.constant 0.000000e+00 : f32
            %39 = vector.broadcast %cst_10 : f32 to vector<1xf32>
            %40 = vector.broadcast %cst_11 : f32 to vector<1xf32>
            %41:2 = scf.for %arg6 = %34 to %37 step %c1_8 iter_args(%arg7 = %39, %arg8 = %40) -> (vector<1xf32>, vector<1xf32>) {
              %42 = vector.load %4[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %43 = vector.load %5[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %44 = arith.addi %arg6, %25 : index
              %45 = vector.load %4[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %46 = vector.load %5[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %47 = arith.addf %42, %45 : vector<1xf32>
              %48 = arith.addf %43, %46 : vector<1xf32>
              vector.store %47, %4[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              vector.store %48, %5[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %49 = arith.subf %42, %45 : vector<1xf32>
              %50 = arith.subf %43, %46 : vector<1xf32>
              %51 = arith.mulf %49, %arg7 : vector<1xf32>
              %52 = arith.mulf %50, %arg8 : vector<1xf32>
              %53 = arith.mulf %49, %arg8 : vector<1xf32>
              %54 = arith.mulf %50, %arg7 : vector<1xf32>
              %55 = arith.subf %51, %52 : vector<1xf32>
              %56 = arith.addf %53, %54 : vector<1xf32>
              vector.store %55, %4[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              vector.store %56, %5[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %57 = arith.mulf %arg7, %30 : vector<1xf32>
              %58 = arith.mulf %arg8, %32 : vector<1xf32>
              %59 = arith.mulf %arg7, %32 : vector<1xf32>
              %60 = arith.mulf %arg8, %30 : vector<1xf32>
              %61 = arith.subf %57, %58 : vector<1xf32>
              %62 = arith.addf %59, %60 : vector<1xf32>
              scf.yield %61, %62 : vector<1xf32>, vector<1xf32>
            }
          }
          %33 = arith.shli %arg3, %c1_7 : index
          scf.yield %33, %25 : index, index
        }
      }
      %16 = arith.cmpi ne, %13, %14 : index
      scf.if %16 {
        affine.for %arg1 = #map(%c0_6) to #map(%14) {
          affine.for %arg2 = #map(%c0_6) to #map(%13) {
            %18 = memref.load %4[%arg1, %arg2] : memref<?x?xf32>
            memref.store %18, %0[%arg2, %arg1] : memref<?x?xf32>
          }
        }
        affine.for %arg1 = #map(%c0_6) to #map(%14) {
          affine.for %arg2 = #map(%c0_6) to #map(%13) {
            %18 = memref.load %5[%arg1, %arg2] : memref<?x?xf32>
            memref.store %18, %1[%arg2, %arg1] : memref<?x?xf32>
          }
        }
      } else {
        memref.copy %4, %0 : memref<?x?xf32> to memref<?x?xf32>
        memref.copy %5, %1 : memref<?x?xf32> to memref<?x?xf32>
      }
      affine.for %arg1 = #map(%c0_6) to #map(%11) {
        affine.for %arg2 = #map(%c0_6) to #map(%12) {
          %18 = vector.load %2[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
          %19 = vector.load %3[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
          %20 = vector.load %0[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
          %21 = vector.load %1[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
          %22 = arith.mulf %18, %20 : vector<1xf32>
          %23 = arith.mulf %19, %21 : vector<1xf32>
          %24 = arith.mulf %18, %21 : vector<1xf32>
          %25 = arith.mulf %19, %20 : vector<1xf32>
          %26 = arith.subf %22, %23 : vector<1xf32>
          %27 = arith.addf %24, %25 : vector<1xf32>
          vector.store %26, %2[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
          vector.store %27, %3[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
        }
      }
      affine.for %arg1 = #map(%c0_6) to #map(%11) {
        %18 = arith.shrsi %12, %c1_7 : index
        %19 = arith.index_cast %12 : index to i32
        %20 = arith.sitofp %19 : i32 to f32
        %21 = math.log2 %20 : f32
        %22 = arith.fptoui %21 : f32 to i32
        %23 = arith.index_cast %22 : i32 to index
        %cst_9 = arith.constant 6.28318548 : f32
        %24:2 = scf.for %arg2 = %c0_6 to %23 step %c1_7 iter_args(%arg3 = %18, %arg4 = %c1_7) -> (index, index) {
          %28 = arith.shli %arg4, %c1_7 : index
          %29 = arith.index_cast %28 : index to i32
          %30 = arith.sitofp %29 : i32 to f32
          %31 = arith.divf %cst_9, %30 : f32
          %32 = math.cos %31 : f32
          %33 = vector.broadcast %32 : f32 to vector<1xf32>
          %34 = math.sin %31 : f32
          %35 = vector.broadcast %34 : f32 to vector<1xf32>
          scf.for %arg5 = %c0_6 to %arg3 step %c1_7 {
            %37 = arith.muli %arg5, %28 : index
            %38 = arith.addi %37, %arg4 : index
            %cst_10 = arith.constant 1.000000e+00 : f32
            %cst_11 = arith.constant 0.000000e+00 : f32
            %39 = vector.broadcast %cst_10 : f32 to vector<1xf32>
            %40 = vector.broadcast %cst_11 : f32 to vector<1xf32>
            %41:2 = scf.for %arg6 = %37 to %38 step %c1_8 iter_args(%arg7 = %39, %arg8 = %40) -> (vector<1xf32>, vector<1xf32>) {
              %42 = vector.load %2[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %43 = vector.load %3[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %44 = arith.addi %arg6, %arg4 : index
              %45 = vector.load %2[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %46 = vector.load %3[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %47 = arith.mulf %45, %arg7 : vector<1xf32>
              %48 = arith.mulf %46, %arg8 : vector<1xf32>
              %49 = arith.mulf %45, %arg8 : vector<1xf32>
              %50 = arith.mulf %46, %arg7 : vector<1xf32>
              %51 = arith.subf %47, %48 : vector<1xf32>
              %52 = arith.addf %49, %50 : vector<1xf32>
              %53 = arith.addf %42, %51 : vector<1xf32>
              %54 = arith.addf %43, %52 : vector<1xf32>
              vector.store %53, %2[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              vector.store %54, %3[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %55 = arith.subf %42, %51 : vector<1xf32>
              %56 = arith.subf %43, %52 : vector<1xf32>
              vector.store %55, %2[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              vector.store %56, %3[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %57 = arith.mulf %arg7, %33 : vector<1xf32>
              %58 = arith.mulf %arg8, %35 : vector<1xf32>
              %59 = arith.mulf %arg7, %35 : vector<1xf32>
              %60 = arith.mulf %arg8, %33 : vector<1xf32>
              %61 = arith.subf %57, %58 : vector<1xf32>
              %62 = arith.addf %59, %60 : vector<1xf32>
              scf.yield %61, %62 : vector<1xf32>, vector<1xf32>
            }
          }
          %36 = arith.shrsi %arg3, %c1_7 : index
          scf.yield %36, %28 : index, index
        }
        %25 = arith.index_cast %12 : index to i32
        %26 = arith.sitofp %25 : i32 to f32
        %27 = vector.broadcast %26 : f32 to vector<1xf32>
        scf.for %arg2 = %c0_6 to %12 step %c1_8 {
          %28 = vector.load %2[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
          %29 = arith.divf %28, %27 : vector<1xf32>
          vector.store %29, %2[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
          %30 = vector.load %3[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
          %31 = arith.divf %30, %27 : vector<1xf32>
          vector.store %31, %3[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
        }
      }
      affine.for %arg1 = #map(%c0_6) to #map(%11) {
        affine.for %arg2 = #map(%c0_6) to #map(%12) {
          %18 = memref.load %2[%arg1, %arg2] : memref<?x?xf32>
          memref.store %18, %4[%arg2, %arg1] : memref<?x?xf32>
        }
      }
      affine.for %arg1 = #map(%c0_6) to #map(%11) {
        affine.for %arg2 = #map(%c0_6) to #map(%12) {
          %18 = memref.load %3[%arg1, %arg2] : memref<?x?xf32>
          memref.store %18, %5[%arg2, %arg1] : memref<?x?xf32>
        }
      }
      affine.for %arg1 = #map(%c0_6) to #map(%12) {
        %18 = arith.shrsi %11, %c1_7 : index
        %19 = arith.index_cast %11 : index to i32
        %20 = arith.sitofp %19 : i32 to f32
        %21 = math.log2 %20 : f32
        %22 = arith.fptoui %21 : f32 to i32
        %23 = arith.index_cast %22 : i32 to index
        %cst_9 = arith.constant 6.28318548 : f32
        %24:2 = scf.for %arg2 = %c0_6 to %23 step %c1_7 iter_args(%arg3 = %18, %arg4 = %c1_7) -> (index, index) {
          %28 = arith.shli %arg4, %c1_7 : index
          %29 = arith.index_cast %28 : index to i32
          %30 = arith.sitofp %29 : i32 to f32
          %31 = arith.divf %cst_9, %30 : f32
          %32 = math.cos %31 : f32
          %33 = vector.broadcast %32 : f32 to vector<1xf32>
          %34 = math.sin %31 : f32
          %35 = vector.broadcast %34 : f32 to vector<1xf32>
          scf.for %arg5 = %c0_6 to %arg3 step %c1_7 {
            %37 = arith.muli %arg5, %28 : index
            %38 = arith.addi %37, %arg4 : index
            %cst_10 = arith.constant 1.000000e+00 : f32
            %cst_11 = arith.constant 0.000000e+00 : f32
            %39 = vector.broadcast %cst_10 : f32 to vector<1xf32>
            %40 = vector.broadcast %cst_11 : f32 to vector<1xf32>
            %41:2 = scf.for %arg6 = %37 to %38 step %c1_8 iter_args(%arg7 = %39, %arg8 = %40) -> (vector<1xf32>, vector<1xf32>) {
              %42 = vector.load %4[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %43 = vector.load %5[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %44 = arith.addi %arg6, %arg4 : index
              %45 = vector.load %4[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %46 = vector.load %5[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %47 = arith.mulf %45, %arg7 : vector<1xf32>
              %48 = arith.mulf %46, %arg8 : vector<1xf32>
              %49 = arith.mulf %45, %arg8 : vector<1xf32>
              %50 = arith.mulf %46, %arg7 : vector<1xf32>
              %51 = arith.subf %47, %48 : vector<1xf32>
              %52 = arith.addf %49, %50 : vector<1xf32>
              %53 = arith.addf %42, %51 : vector<1xf32>
              %54 = arith.addf %43, %52 : vector<1xf32>
              vector.store %53, %4[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              vector.store %54, %5[%arg1, %arg6] : memref<?x?xf32>, vector<1xf32>
              %55 = arith.subf %42, %51 : vector<1xf32>
              %56 = arith.subf %43, %52 : vector<1xf32>
              vector.store %55, %4[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              vector.store %56, %5[%arg1, %44] : memref<?x?xf32>, vector<1xf32>
              %57 = arith.mulf %arg7, %33 : vector<1xf32>
              %58 = arith.mulf %arg8, %35 : vector<1xf32>
              %59 = arith.mulf %arg7, %35 : vector<1xf32>
              %60 = arith.mulf %arg8, %33 : vector<1xf32>
              %61 = arith.subf %57, %58 : vector<1xf32>
              %62 = arith.addf %59, %60 : vector<1xf32>
              scf.yield %61, %62 : vector<1xf32>, vector<1xf32>
            }
          }
          %36 = arith.shrsi %arg3, %c1_7 : index
          scf.yield %36, %28 : index, index
        }
        %25 = arith.index_cast %11 : index to i32
        %26 = arith.sitofp %25 : i32 to f32
        %27 = vector.broadcast %26 : f32 to vector<1xf32>
        scf.for %arg2 = %c0_6 to %11 step %c1_8 {
          %28 = vector.load %4[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
          %29 = arith.divf %28, %27 : vector<1xf32>
          vector.store %29, %4[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
          %30 = vector.load %5[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
          %31 = arith.divf %30, %27 : vector<1xf32>
          vector.store %31, %5[%arg1, %arg2] : memref<?x?xf32>, vector<1xf32>
        }
      }
      %17 = arith.cmpi ne, %11, %12 : index
      scf.if %17 {
        affine.for %arg1 = #map(%c0_6) to #map(%12) {
          affine.for %arg2 = #map(%c0_6) to #map(%11) {
            %18 = memref.load %4[%arg1, %arg2] : memref<?x?xf32>
            memref.store %18, %2[%arg2, %arg1] : memref<?x?xf32>
          }
        }
        affine.for %arg1 = #map(%c0_6) to #map(%12) {
          affine.for %arg2 = #map(%c0_6) to #map(%11) {
            %18 = memref.load %5[%arg1, %arg2] : memref<?x?xf32>
            memref.store %18, %3[%arg2, %arg1] : memref<?x?xf32>
          }
        }
      } else {
        memref.copy %4, %2 : memref<?x?xf32> to memref<?x?xf32>
        memref.copy %5, %3 : memref<?x?xf32> to memref<?x?xf32>
      }
    }
    %10 = memref.cast %2 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%10) : (memref<*xf32>) -> ()
    memref.dealloc %2 : memref<?x?xf32>
    memref.dealloc %3 : memref<?x?xf32>
    memref.dealloc %0 : memref<?x?xf32>
    memref.dealloc %1 : memref<?x?xf32>
    memref.dealloc %4 : memref<?x?xf32>
    memref.dealloc %5 : memref<?x?xf32>
    memref.dealloc %6 : memref<?x?xf32>
    memref.dealloc %7 : memref<?x?xf32>
    return
  }
}

