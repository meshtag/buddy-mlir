#map = affine_map<(d0, d1, d2) -> (d0 + d1 - d2)>
module {
  func.func @check_corr_2d(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>, %arg3: index, %arg4: index, %arg5: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    %dim_1 = memref.dim %arg1, %c0 : memref<?x?x?x?xf32>
    %dim_2 = memref.dim %arg1, %c1 : memref<?x?x?x?xf32>
    %0 = arith.addi %dim, %arg4 : index
    %1 = arith.addi %dim_0, %arg3 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_0, %dim_2, %c1)
    affine.parallel (%arg6) = (0) to (symbol(%dim)) {
      affine.parallel (%arg7) = (0) to (symbol(%dim_1)) {
        affine.parallel (%arg8) = (0) to (symbol(%dim_0)) step (64) {
          affine.parallel (%arg9) = (0) to (symbol(%dim_2)) {
            %3 = arith.addi %arg6, %arg7 : index
            %4 = arith.addi %arg8, %arg9 : index
            %5 = arith.subi %3, %arg4 : index
            %6 = arith.subi %4, %arg3 : index
            %7 = arith.addi %4, %c64 : index
            %8 = arith.cmpi slt, %3, %arg4 : index
          }
        }
      }
    }
    return
  }
  func.func @corr_2d_constant_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg1, %c0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg4 : index
    %1 = arith.addi %dim_0, %arg3 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_0, %dim_2, %c1)
    affine.parallel (%arg6) = (0) to (symbol(%dim)) {
      affine.parallel (%arg7) = (0) to (symbol(%dim_1)) {
        affine.parallel (%arg8) = (0) to (symbol(%dim_0)) step (64) {
          affine.parallel (%arg9) = (0) to (symbol(%dim_2)) {
            %3 = arith.addi %arg6, %arg7 : index
            %4 = arith.addi %arg8, %arg9 : index
            %5 = arith.subi %3, %arg4 : index
            %6 = arith.subi %4, %arg3 : index
            %7 = arith.addi %4, %c64 : index
            %8 = arith.cmpi slt, %3, %arg4 : index
          }
        }
      }
    }
    return
  }
  func.func @corr_2d_replicate_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg1, %c0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg4 : index
    %1 = arith.addi %dim_0, %arg3 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_0, %dim_2, %c1)
    affine.parallel (%arg6) = (0) to (symbol(%dim)) {
      affine.parallel (%arg7) = (0) to (symbol(%dim_1)) {
        affine.parallel (%arg8) = (0) to (symbol(%dim_0)) step (64) {
          affine.parallel (%arg9) = (0) to (symbol(%dim_2)) {
            %3 = arith.addi %arg6, %arg7 : index
            %4 = arith.addi %arg8, %arg9 : index
            %5 = arith.subi %3, %arg4 : index
            %6 = arith.subi %4, %arg3 : index
            %7 = arith.addi %4, %c64 : index
            %8 = arith.cmpi slt, %3, %arg4 : index
          }
        }
      }
    }
    return
  }
  func.func @corrfft_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %dim_2 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %dim_3 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %0 = arith.index_cast %dim_1 : index to i32
    %1 = arith.sitofp %0 : i32 to f32
    %2 = math.log2 %1 : f32
    %3 = arith.fptoui %2 : f32 to i32
    %4 = arith.index_cast %3 : i32 to index
    %cst = arith.constant -6.28318548 : f32
    affine.for %arg6 = 0 to %dim {
      %41:2 = scf.for %arg7 = %c0 to %4 step %c1 iter_args(%arg8 = %c1, %arg9 = %dim_1) -> (index, index) {
        %42 = arith.shrsi %arg9, %c1 : index
        %43 = arith.index_cast %arg9 : index to i32
        %44 = arith.sitofp %43 : i32 to f32
        %45 = arith.divf %cst, %44 : f32
        %46 = math.cos %45 : f32
        %47 = vector.broadcast %46 : f32 to vector<1xf32>
        %48 = math.sin %45 : f32
        %49 = vector.broadcast %48 : f32 to vector<1xf32>
        scf.for %arg10 = %c0 to %arg8 step %c1 {
          %51 = arith.muli %arg10, %arg9 : index
          %52 = arith.addi %51, %42 : index
          %cst_9 = arith.constant 1.000000e+00 : f32
          %cst_10 = arith.constant 0.000000e+00 : f32
          %cst_11 = arith.constant dense<1.000000e+00> : vector<1xf32>
          %cst_12 = arith.constant dense<0.000000e+00> : vector<1xf32>
          %53:2 = scf.for %arg11 = %51 to %52 step %c1_0 iter_args(%arg12 = %cst_11, %arg13 = %cst_12) -> (vector<1xf32>, vector<1xf32>) {
            %54 = vector.load %arg0[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %55 = vector.load %arg1[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %56 = arith.addi %arg11, %42 : index
            %57 = vector.load %arg0[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %58 = vector.load %arg1[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %59 = arith.addf %54, %57 : vector<1xf32>
            %60 = arith.addf %55, %58 : vector<1xf32>
            vector.store %59, %arg0[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            vector.store %60, %arg1[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %61 = arith.subf %54, %57 : vector<1xf32>
            %62 = arith.subf %55, %58 : vector<1xf32>
            %63 = arith.mulf %61, %arg12 : vector<1xf32>
            %64 = arith.mulf %62, %arg13 : vector<1xf32>
            %65 = arith.mulf %61, %arg13 : vector<1xf32>
            %66 = arith.mulf %62, %arg12 : vector<1xf32>
            %67 = arith.subf %63, %64 : vector<1xf32>
            %68 = arith.addf %65, %66 : vector<1xf32>
            vector.store %67, %arg0[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            vector.store %68, %arg1[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %69 = arith.mulf %arg12, %47 : vector<1xf32>
            %70 = arith.mulf %arg13, %49 : vector<1xf32>
            %71 = arith.mulf %arg12, %49 : vector<1xf32>
            %72 = arith.mulf %arg13, %47 : vector<1xf32>
            %73 = arith.subf %69, %70 : vector<1xf32>
            %74 = arith.addf %71, %72 : vector<1xf32>
            scf.yield %73, %74 : vector<1xf32>, vector<1xf32>
          }
        }
        %50 = arith.shli %arg8, %c1 : index
        scf.yield %50, %42 : index, index
      }
    }
    affine.for %arg6 = 0 to %dim {
      affine.for %arg7 = 0 to %dim_1 {
        %41 = memref.load %arg0[%arg6, %arg7] : memref<?x?xf32>
        memref.store %41, %arg4[%arg7, %arg6] : memref<?x?xf32>
      }
    }
    affine.for %arg6 = 0 to %dim {
      affine.for %arg7 = 0 to %dim_1 {
        %41 = memref.load %arg1[%arg6, %arg7] : memref<?x?xf32>
        memref.store %41, %arg5[%arg7, %arg6] : memref<?x?xf32>
      }
    }
    %5 = arith.index_cast %dim : index to i32
    %6 = arith.sitofp %5 : i32 to f32
    %7 = math.log2 %6 : f32
    %8 = arith.fptoui %7 : f32 to i32
    %9 = arith.index_cast %8 : i32 to index
    %cst_4 = arith.constant -6.28318548 : f32
    affine.for %arg6 = 0 to %dim_1 {
      %41:2 = scf.for %arg7 = %c0 to %9 step %c1 iter_args(%arg8 = %c1, %arg9 = %dim) -> (index, index) {
        %42 = arith.shrsi %arg9, %c1 : index
        %43 = arith.index_cast %arg9 : index to i32
        %44 = arith.sitofp %43 : i32 to f32
        %45 = arith.divf %cst_4, %44 : f32
        %46 = math.cos %45 : f32
        %47 = vector.broadcast %46 : f32 to vector<1xf32>
        %48 = math.sin %45 : f32
        %49 = vector.broadcast %48 : f32 to vector<1xf32>
        scf.for %arg10 = %c0 to %arg8 step %c1 {
          %51 = arith.muli %arg10, %arg9 : index
          %52 = arith.addi %51, %42 : index
          %cst_9 = arith.constant 1.000000e+00 : f32
          %cst_10 = arith.constant 0.000000e+00 : f32
          %cst_11 = arith.constant dense<1.000000e+00> : vector<1xf32>
          %cst_12 = arith.constant dense<0.000000e+00> : vector<1xf32>
          %53:2 = scf.for %arg11 = %51 to %52 step %c1_0 iter_args(%arg12 = %cst_11, %arg13 = %cst_12) -> (vector<1xf32>, vector<1xf32>) {
            %54 = vector.load %arg4[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %55 = vector.load %arg5[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %56 = arith.addi %arg11, %42 : index
            %57 = vector.load %arg4[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %58 = vector.load %arg5[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %59 = arith.addf %54, %57 : vector<1xf32>
            %60 = arith.addf %55, %58 : vector<1xf32>
            vector.store %59, %arg4[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            vector.store %60, %arg5[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %61 = arith.subf %54, %57 : vector<1xf32>
            %62 = arith.subf %55, %58 : vector<1xf32>
            %63 = arith.mulf %61, %arg12 : vector<1xf32>
            %64 = arith.mulf %62, %arg13 : vector<1xf32>
            %65 = arith.mulf %61, %arg13 : vector<1xf32>
            %66 = arith.mulf %62, %arg12 : vector<1xf32>
            %67 = arith.subf %63, %64 : vector<1xf32>
            %68 = arith.addf %65, %66 : vector<1xf32>
            vector.store %67, %arg4[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            vector.store %68, %arg5[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %69 = arith.mulf %arg12, %47 : vector<1xf32>
            %70 = arith.mulf %arg13, %49 : vector<1xf32>
            %71 = arith.mulf %arg12, %49 : vector<1xf32>
            %72 = arith.mulf %arg13, %47 : vector<1xf32>
            %73 = arith.subf %69, %70 : vector<1xf32>
            %74 = arith.addf %71, %72 : vector<1xf32>
            scf.yield %73, %74 : vector<1xf32>, vector<1xf32>
          }
        }
        %50 = arith.shli %arg8, %c1 : index
        scf.yield %50, %42 : index, index
      }
    }
    %10 = arith.cmpi ne, %dim, %dim_1 : index
    scf.if %10 {
      affine.for %arg6 = 0 to %dim_1 {
        affine.for %arg7 = 0 to %dim {
          %41 = memref.load %arg4[%arg6, %arg7] : memref<?x?xf32>
          memref.store %41, %arg0[%arg7, %arg6] : memref<?x?xf32>
        }
      }
      affine.for %arg6 = 0 to %dim_1 {
        affine.for %arg7 = 0 to %dim {
          %41 = memref.load %arg5[%arg6, %arg7] : memref<?x?xf32>
          memref.store %41, %arg1[%arg7, %arg6] : memref<?x?xf32>
        }
      }
    } else {
      memref.copy %arg4, %arg0 : memref<?x?xf32> to memref<?x?xf32>
      memref.copy %arg5, %arg1 : memref<?x?xf32> to memref<?x?xf32>
    }
    %11 = arith.index_cast %dim_3 : index to i32
    %12 = arith.sitofp %11 : i32 to f32
    %13 = math.log2 %12 : f32
    %14 = arith.fptoui %13 : f32 to i32
    %15 = arith.index_cast %14 : i32 to index
    %cst_5 = arith.constant -6.28318548 : f32
    affine.for %arg6 = 0 to %dim_2 {
      %41:2 = scf.for %arg7 = %c0 to %15 step %c1 iter_args(%arg8 = %c1, %arg9 = %dim_3) -> (index, index) {
        %42 = arith.shrsi %arg9, %c1 : index
        %43 = arith.index_cast %arg9 : index to i32
        %44 = arith.sitofp %43 : i32 to f32
        %45 = arith.divf %cst_5, %44 : f32
        %46 = math.cos %45 : f32
        %47 = vector.broadcast %46 : f32 to vector<1xf32>
        %48 = math.sin %45 : f32
        %49 = vector.broadcast %48 : f32 to vector<1xf32>
        scf.for %arg10 = %c0 to %arg8 step %c1 {
          %51 = arith.muli %arg10, %arg9 : index
          %52 = arith.addi %51, %42 : index
          %cst_9 = arith.constant 1.000000e+00 : f32
          %cst_10 = arith.constant 0.000000e+00 : f32
          %cst_11 = arith.constant dense<1.000000e+00> : vector<1xf32>
          %cst_12 = arith.constant dense<0.000000e+00> : vector<1xf32>
          %53:2 = scf.for %arg11 = %51 to %52 step %c1_0 iter_args(%arg12 = %cst_11, %arg13 = %cst_12) -> (vector<1xf32>, vector<1xf32>) {
            %54 = vector.load %arg2[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %55 = vector.load %arg3[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %56 = arith.addi %arg11, %42 : index
            %57 = vector.load %arg2[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %58 = vector.load %arg3[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %59 = arith.addf %54, %57 : vector<1xf32>
            %60 = arith.addf %55, %58 : vector<1xf32>
            vector.store %59, %arg2[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            vector.store %60, %arg3[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %61 = arith.subf %54, %57 : vector<1xf32>
            %62 = arith.subf %55, %58 : vector<1xf32>
            %63 = arith.mulf %61, %arg12 : vector<1xf32>
            %64 = arith.mulf %62, %arg13 : vector<1xf32>
            %65 = arith.mulf %61, %arg13 : vector<1xf32>
            %66 = arith.mulf %62, %arg12 : vector<1xf32>
            %67 = arith.subf %63, %64 : vector<1xf32>
            %68 = arith.addf %65, %66 : vector<1xf32>
            vector.store %67, %arg2[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            vector.store %68, %arg3[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %69 = arith.mulf %arg12, %47 : vector<1xf32>
            %70 = arith.mulf %arg13, %49 : vector<1xf32>
            %71 = arith.mulf %arg12, %49 : vector<1xf32>
            %72 = arith.mulf %arg13, %47 : vector<1xf32>
            %73 = arith.subf %69, %70 : vector<1xf32>
            %74 = arith.addf %71, %72 : vector<1xf32>
            scf.yield %73, %74 : vector<1xf32>, vector<1xf32>
          }
        }
        %50 = arith.shli %arg8, %c1 : index
        scf.yield %50, %42 : index, index
      }
    }
    affine.for %arg6 = 0 to %dim_2 {
      affine.for %arg7 = 0 to %dim_3 {
        %41 = memref.load %arg2[%arg6, %arg7] : memref<?x?xf32>
        memref.store %41, %arg4[%arg7, %arg6] : memref<?x?xf32>
      }
    }
    affine.for %arg6 = 0 to %dim_2 {
      affine.for %arg7 = 0 to %dim_3 {
        %41 = memref.load %arg3[%arg6, %arg7] : memref<?x?xf32>
        memref.store %41, %arg5[%arg7, %arg6] : memref<?x?xf32>
      }
    }
    %16 = arith.index_cast %dim_2 : index to i32
    %17 = arith.sitofp %16 : i32 to f32
    %18 = math.log2 %17 : f32
    %19 = arith.fptoui %18 : f32 to i32
    %20 = arith.index_cast %19 : i32 to index
    %cst_6 = arith.constant -6.28318548 : f32
    affine.for %arg6 = 0 to %dim_3 {
      %41:2 = scf.for %arg7 = %c0 to %20 step %c1 iter_args(%arg8 = %c1, %arg9 = %dim_2) -> (index, index) {
        %42 = arith.shrsi %arg9, %c1 : index
        %43 = arith.index_cast %arg9 : index to i32
        %44 = arith.sitofp %43 : i32 to f32
        %45 = arith.divf %cst_6, %44 : f32
        %46 = math.cos %45 : f32
        %47 = vector.broadcast %46 : f32 to vector<1xf32>
        %48 = math.sin %45 : f32
        %49 = vector.broadcast %48 : f32 to vector<1xf32>
        scf.for %arg10 = %c0 to %arg8 step %c1 {
          %51 = arith.muli %arg10, %arg9 : index
          %52 = arith.addi %51, %42 : index
          %cst_9 = arith.constant 1.000000e+00 : f32
          %cst_10 = arith.constant 0.000000e+00 : f32
          %cst_11 = arith.constant dense<1.000000e+00> : vector<1xf32>
          %cst_12 = arith.constant dense<0.000000e+00> : vector<1xf32>
          %53:2 = scf.for %arg11 = %51 to %52 step %c1_0 iter_args(%arg12 = %cst_11, %arg13 = %cst_12) -> (vector<1xf32>, vector<1xf32>) {
            %54 = vector.load %arg4[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %55 = vector.load %arg5[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %56 = arith.addi %arg11, %42 : index
            %57 = vector.load %arg4[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %58 = vector.load %arg5[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %59 = arith.addf %54, %57 : vector<1xf32>
            %60 = arith.addf %55, %58 : vector<1xf32>
            vector.store %59, %arg4[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            vector.store %60, %arg5[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %61 = arith.subf %54, %57 : vector<1xf32>
            %62 = arith.subf %55, %58 : vector<1xf32>
            %63 = arith.mulf %61, %arg12 : vector<1xf32>
            %64 = arith.mulf %62, %arg13 : vector<1xf32>
            %65 = arith.mulf %61, %arg13 : vector<1xf32>
            %66 = arith.mulf %62, %arg12 : vector<1xf32>
            %67 = arith.subf %63, %64 : vector<1xf32>
            %68 = arith.addf %65, %66 : vector<1xf32>
            vector.store %67, %arg4[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            vector.store %68, %arg5[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %69 = arith.mulf %arg12, %47 : vector<1xf32>
            %70 = arith.mulf %arg13, %49 : vector<1xf32>
            %71 = arith.mulf %arg12, %49 : vector<1xf32>
            %72 = arith.mulf %arg13, %47 : vector<1xf32>
            %73 = arith.subf %69, %70 : vector<1xf32>
            %74 = arith.addf %71, %72 : vector<1xf32>
            scf.yield %73, %74 : vector<1xf32>, vector<1xf32>
          }
        }
        %50 = arith.shli %arg8, %c1 : index
        scf.yield %50, %42 : index, index
      }
    }
    %21 = arith.cmpi ne, %dim_2, %dim_3 : index
    scf.if %21 {
      affine.for %arg6 = 0 to %dim_3 {
        affine.for %arg7 = 0 to %dim_2 {
          %41 = memref.load %arg4[%arg6, %arg7] : memref<?x?xf32>
          memref.store %41, %arg2[%arg7, %arg6] : memref<?x?xf32>
        }
      }
      affine.for %arg6 = 0 to %dim_3 {
        affine.for %arg7 = 0 to %dim_2 {
          %41 = memref.load %arg5[%arg6, %arg7] : memref<?x?xf32>
          memref.store %41, %arg3[%arg7, %arg6] : memref<?x?xf32>
        }
      }
    } else {
      memref.copy %arg4, %arg2 : memref<?x?xf32> to memref<?x?xf32>
      memref.copy %arg5, %arg3 : memref<?x?xf32> to memref<?x?xf32>
    }
    affine.for %arg6 = 0 to %dim {
      affine.for %arg7 = 0 to %dim_1 {
        %41 = vector.load %arg0[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
        %42 = vector.load %arg1[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
        %43 = vector.load %arg2[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
        %44 = vector.load %arg3[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
        %45 = arith.mulf %41, %43 : vector<1xf32>
        %46 = arith.mulf %42, %44 : vector<1xf32>
        %47 = arith.mulf %41, %44 : vector<1xf32>
        %48 = arith.mulf %42, %43 : vector<1xf32>
        %49 = arith.subf %45, %46 : vector<1xf32>
        %50 = arith.addf %47, %48 : vector<1xf32>
        vector.store %49, %arg0[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
        vector.store %50, %arg1[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
      }
    }
    %22 = arith.shrsi %dim_1, %c1 : index
    %23 = arith.index_cast %dim_1 : index to i32
    %24 = arith.sitofp %23 : i32 to f32
    %25 = math.log2 %24 : f32
    %26 = arith.fptoui %25 : f32 to i32
    %27 = arith.index_cast %26 : i32 to index
    %cst_7 = arith.constant 6.28318548 : f32
    %28 = arith.index_cast %dim_1 : index to i32
    %29 = arith.sitofp %28 : i32 to f32
    %30 = vector.broadcast %29 : f32 to vector<1xf32>
    affine.for %arg6 = 0 to %dim {
      %41:2 = scf.for %arg7 = %c0 to %27 step %c1 iter_args(%arg8 = %22, %arg9 = %c1) -> (index, index) {
        %42 = arith.shli %arg9, %c1 : index
        %43 = arith.index_cast %42 : index to i32
        %44 = arith.sitofp %43 : i32 to f32
        %45 = arith.divf %cst_7, %44 : f32
        %46 = math.cos %45 : f32
        %47 = vector.broadcast %46 : f32 to vector<1xf32>
        %48 = math.sin %45 : f32
        %49 = vector.broadcast %48 : f32 to vector<1xf32>
        scf.for %arg10 = %c0 to %arg8 step %c1 {
          %51 = arith.muli %arg10, %42 : index
          %52 = arith.addi %51, %arg9 : index
          %cst_9 = arith.constant 1.000000e+00 : f32
          %cst_10 = arith.constant 0.000000e+00 : f32
          %cst_11 = arith.constant dense<1.000000e+00> : vector<1xf32>
          %cst_12 = arith.constant dense<0.000000e+00> : vector<1xf32>
          %53:2 = scf.for %arg11 = %51 to %52 step %c1_0 iter_args(%arg12 = %cst_11, %arg13 = %cst_12) -> (vector<1xf32>, vector<1xf32>) {
            %54 = vector.load %arg0[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %55 = vector.load %arg1[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %56 = arith.addi %arg11, %arg9 : index
            %57 = vector.load %arg0[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %58 = vector.load %arg1[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %59 = arith.mulf %57, %arg12 : vector<1xf32>
            %60 = arith.mulf %58, %arg13 : vector<1xf32>
            %61 = arith.mulf %57, %arg13 : vector<1xf32>
            %62 = arith.mulf %58, %arg12 : vector<1xf32>
            %63 = arith.subf %59, %60 : vector<1xf32>
            %64 = arith.addf %61, %62 : vector<1xf32>
            %65 = arith.addf %54, %63 : vector<1xf32>
            %66 = arith.addf %55, %64 : vector<1xf32>
            vector.store %65, %arg0[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            vector.store %66, %arg1[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %67 = arith.subf %54, %63 : vector<1xf32>
            %68 = arith.subf %55, %64 : vector<1xf32>
            vector.store %67, %arg0[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            vector.store %68, %arg1[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %69 = arith.mulf %arg12, %47 : vector<1xf32>
            %70 = arith.mulf %arg13, %49 : vector<1xf32>
            %71 = arith.mulf %arg12, %49 : vector<1xf32>
            %72 = arith.mulf %arg13, %47 : vector<1xf32>
            %73 = arith.subf %69, %70 : vector<1xf32>
            %74 = arith.addf %71, %72 : vector<1xf32>
            scf.yield %73, %74 : vector<1xf32>, vector<1xf32>
          }
        }
        %50 = arith.shrsi %arg8, %c1 : index
        scf.yield %50, %42 : index, index
      }
      scf.for %arg7 = %c0 to %dim_1 step %c1_0 {
        %42 = vector.load %arg0[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
        %43 = arith.divf %42, %30 : vector<1xf32>
        vector.store %43, %arg0[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
        %44 = vector.load %arg1[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
        %45 = arith.divf %44, %30 : vector<1xf32>
        vector.store %45, %arg1[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
      }
    }
    affine.for %arg6 = 0 to %dim {
      affine.for %arg7 = 0 to %dim_1 {
        %41 = memref.load %arg0[%arg6, %arg7] : memref<?x?xf32>
        memref.store %41, %arg4[%arg7, %arg6] : memref<?x?xf32>
      }
    }
    affine.for %arg6 = 0 to %dim {
      affine.for %arg7 = 0 to %dim_1 {
        %41 = memref.load %arg1[%arg6, %arg7] : memref<?x?xf32>
        memref.store %41, %arg5[%arg7, %arg6] : memref<?x?xf32>
      }
    }
    %31 = arith.shrsi %dim, %c1 : index
    %32 = arith.index_cast %dim : index to i32
    %33 = arith.sitofp %32 : i32 to f32
    %34 = math.log2 %33 : f32
    %35 = arith.fptoui %34 : f32 to i32
    %36 = arith.index_cast %35 : i32 to index
    %cst_8 = arith.constant 6.28318548 : f32
    %37 = arith.index_cast %dim : index to i32
    %38 = arith.sitofp %37 : i32 to f32
    %39 = vector.broadcast %38 : f32 to vector<1xf32>
    affine.for %arg6 = 0 to %dim_1 {
      %41:2 = scf.for %arg7 = %c0 to %36 step %c1 iter_args(%arg8 = %31, %arg9 = %c1) -> (index, index) {
        %42 = arith.shli %arg9, %c1 : index
        %43 = arith.index_cast %42 : index to i32
        %44 = arith.sitofp %43 : i32 to f32
        %45 = arith.divf %cst_8, %44 : f32
        %46 = math.cos %45 : f32
        %47 = vector.broadcast %46 : f32 to vector<1xf32>
        %48 = math.sin %45 : f32
        %49 = vector.broadcast %48 : f32 to vector<1xf32>
        scf.for %arg10 = %c0 to %arg8 step %c1 {
          %51 = arith.muli %arg10, %42 : index
          %52 = arith.addi %51, %arg9 : index
          %cst_9 = arith.constant 1.000000e+00 : f32
          %cst_10 = arith.constant 0.000000e+00 : f32
          %cst_11 = arith.constant dense<1.000000e+00> : vector<1xf32>
          %cst_12 = arith.constant dense<0.000000e+00> : vector<1xf32>
          %53:2 = scf.for %arg11 = %51 to %52 step %c1_0 iter_args(%arg12 = %cst_11, %arg13 = %cst_12) -> (vector<1xf32>, vector<1xf32>) {
            %54 = vector.load %arg4[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %55 = vector.load %arg5[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %56 = arith.addi %arg11, %arg9 : index
            %57 = vector.load %arg4[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %58 = vector.load %arg5[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %59 = arith.mulf %57, %arg12 : vector<1xf32>
            %60 = arith.mulf %58, %arg13 : vector<1xf32>
            %61 = arith.mulf %57, %arg13 : vector<1xf32>
            %62 = arith.mulf %58, %arg12 : vector<1xf32>
            %63 = arith.subf %59, %60 : vector<1xf32>
            %64 = arith.addf %61, %62 : vector<1xf32>
            %65 = arith.addf %54, %63 : vector<1xf32>
            %66 = arith.addf %55, %64 : vector<1xf32>
            vector.store %65, %arg4[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            vector.store %66, %arg5[%arg6, %arg11] : memref<?x?xf32>, vector<1xf32>
            %67 = arith.subf %54, %63 : vector<1xf32>
            %68 = arith.subf %55, %64 : vector<1xf32>
            vector.store %67, %arg4[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            vector.store %68, %arg5[%arg6, %56] : memref<?x?xf32>, vector<1xf32>
            %69 = arith.mulf %arg12, %47 : vector<1xf32>
            %70 = arith.mulf %arg13, %49 : vector<1xf32>
            %71 = arith.mulf %arg12, %49 : vector<1xf32>
            %72 = arith.mulf %arg13, %47 : vector<1xf32>
            %73 = arith.subf %69, %70 : vector<1xf32>
            %74 = arith.addf %71, %72 : vector<1xf32>
            scf.yield %73, %74 : vector<1xf32>, vector<1xf32>
          }
        }
        %50 = arith.shrsi %arg8, %c1 : index
        scf.yield %50, %42 : index, index
      }
      scf.for %arg7 = %c0 to %dim step %c1_0 {
        %42 = vector.load %arg4[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
        %43 = arith.divf %42, %39 : vector<1xf32>
        vector.store %43, %arg4[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
        %44 = vector.load %arg5[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
        %45 = arith.divf %44, %39 : vector<1xf32>
        vector.store %45, %arg5[%arg6, %arg7] : memref<?x?xf32>, vector<1xf32>
      }
    }
    %40 = arith.cmpi ne, %dim, %dim_1 : index
    scf.if %40 {
      affine.for %arg6 = 0 to %dim_1 {
        affine.for %arg7 = 0 to %dim {
          %41 = memref.load %arg4[%arg6, %arg7] : memref<?x?xf32>
          memref.store %41, %arg0[%arg7, %arg6] : memref<?x?xf32>
        }
      }
      affine.for %arg6 = 0 to %dim_1 {
        affine.for %arg7 = 0 to %dim {
          %41 = memref.load %arg5[%arg6, %arg7] : memref<?x?xf32>
          memref.store %41, %arg1[%arg7, %arg6] : memref<?x?xf32>
        }
      }
    } else {
      memref.copy %arg4, %arg0 : memref<?x?xf32> to memref<?x?xf32>
      memref.copy %arg5, %arg1 : memref<?x?xf32> to memref<?x?xf32>
    }
    return
  }
  func.func @rotate_2d(%arg0: memref<?x?xf32>, %arg1: f32, %arg2: memref<?x?xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %c1 : index to i32
    %1 = arith.sitofp %0 : i32 to f32
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %2 = arith.shrsi %dim_0, %c1 : index
    %3 = arith.shrsi %dim, %c1 : index
    %4 = arith.index_cast %2 : index to i32
    %5 = arith.sitofp %4 : i32 to f32
    %6 = arith.index_cast %3 : index to i32
    %7 = arith.sitofp %6 : i32 to f32
    %8 = math.cos %arg1 : f32
    %9 = arith.mulf %8, %1 : f32
    %10 = math.sin %arg1 : f32
    %11 = arith.mulf %10, %1 : f32
    %cst = arith.constant 1.000000e+00 : f32
    %12 = arith.subf %cst, %9 : f32
    %13 = arith.mulf %12, %5 : f32
    %14 = arith.mulf %11, %7 : f32
    %15 = arith.mulf %11, %5 : f32
    %16 = arith.mulf %12, %7 : f32
    %17 = arith.subf %13, %14 : f32
    %18 = arith.negf %11 : f32
    %19 = arith.addf %15, %16 : f32
    %20 = arith.subi %dim_2, %dim_0 : index
    %21 = arith.subi %dim_1, %dim : index
    %22 = arith.shrsi %20, %c1 : index
    %23 = arith.shrsi %21, %c1 : index
    %24 = arith.index_cast %22 : index to i32
    %25 = arith.sitofp %24 : i32 to f32
    %26 = arith.index_cast %23 : index to i32
    %27 = arith.sitofp %26 : i32 to f32
    %28 = arith.addf %17, %25 : f32
    %29 = arith.addf %19, %27 : f32
    %c0_3 = arith.constant 0 : index
    %c1_4 = arith.constant 1 : index
    %cst_5 = arith.constant 0.000000e+00 : f32
    %30 = arith.mulf %9, %9 : f32
    %31 = arith.mulf %11, %18 : f32
    %32 = arith.subf %30, %31 : f32
    %33 = arith.cmpf oeq, %32, %cst_5 : f32
    %34 = scf.if %33 -> (f32) {
      scf.yield %cst_5 : f32
    } else {
      %cst_21 = arith.constant 1.000000e+00 : f32
      %55 = arith.divf %cst_21, %32 : f32
      scf.yield %55 : f32
    }
    %35 = arith.negf %34 : f32
    %36 = arith.mulf %9, %34 : f32
    %37 = arith.mulf %9, %34 : f32
    %38 = arith.mulf %11, %35 : f32
    %39 = arith.mulf %18, %35 : f32
    %40 = arith.mulf %36, %28 : f32
    %41 = arith.mulf %38, %29 : f32
    %42 = arith.addf %40, %41 : f32
    %43 = arith.mulf %28, %39 : f32
    %44 = arith.mulf %37, %29 : f32
    %45 = arith.addf %43, %44 : f32
    %46 = arith.negf %42 : f32
    %47 = arith.negf %45 : f32
    %48 = vector.splat %36 : vector<64xf32>
    %49 = vector.splat %46 : vector<64xf32>
    %50 = vector.splat %39 : vector<64xf32>
    %51 = vector.splat %47 : vector<64xf32>
    %dim_6 = memref.dim %arg2, %c0_3 : memref<?x?xf32>
    %dim_7 = memref.dim %arg2, %c1_4 : memref<?x?xf32>
    %c64 = arith.constant 64 : index
    %52 = arith.divui %dim_7, %c64 : index
    %53 = arith.addi %52, %c1_4 : index
    %54 = arith.muli %53, %c64 : index
    %cst_8 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01, 4.000000e+01, 4.100000e+01, 4.200000e+01, 4.300000e+01, 4.400000e+01, 4.500000e+01, 4.600000e+01, 4.700000e+01, 4.800000e+01, 4.900000e+01, 5.000000e+01, 5.100000e+01, 5.200000e+01, 5.300000e+01, 5.400000e+01, 5.500000e+01, 5.600000e+01, 5.700000e+01, 5.800000e+01, 5.900000e+01, 6.000000e+01, 6.100000e+01, 6.200000e+01, 6.300000e+01]> : vector<64xf32>
    %alloc = memref.alloc(%54) : memref<?xi32>
    %alloc_9 = memref.alloc(%54) : memref<?xi32>
    %cst_10 = arith.constant 3.200000e+01 : f32
    %c16_i32 = arith.constant 16 : i32
    %cst_11 = arith.constant dense<3.200000e+01> : vector<64xf32>
    %cst_12 = arith.constant dense<16> : vector<64xi32>
    affine.for %arg3 = 0 to %54 step 64 {
      %55 = arith.index_cast %arg3 : index to i32
      %56 = arith.sitofp %55 : i32 to f32
      %57 = vector.splat %56 : vector<64xf32>
      %58 = arith.addf %cst_8, %57 : vector<64xf32>
      %59 = arith.mulf %58, %48 : vector<64xf32>
      %60 = arith.mulf %58, %50 : vector<64xf32>
      %61 = arith.addf %59, %49 : vector<64xf32>
      %62 = arith.addf %60, %51 : vector<64xf32>
      %63 = arith.mulf %61, %cst_11 : vector<64xf32>
      %64 = arith.mulf %62, %cst_11 : vector<64xf32>
      %65 = arith.fptosi %63 : vector<64xf32> to vector<64xi32>
      %66 = arith.fptosi %64 : vector<64xf32> to vector<64xi32>
      %67 = arith.addi %65, %cst_12 : vector<64xi32>
      %68 = arith.addi %66, %cst_12 : vector<64xi32>
      vector.store %67, %alloc[%arg3] : memref<?xi32>, vector<64xi32>
      vector.store %68, %alloc_9[%arg3] : memref<?xi32>, vector<64xi32>
    }
    %c0_13 = arith.constant 0 : index
    %c1_14 = arith.constant 1 : index
    %cst_15 = arith.constant 3.200000e+01 : f32
    %c5_i32 = arith.constant 5 : i32
    %c64_16 = arith.constant 64 : index
    %cst_17 = arith.constant dense<5> : vector<64xi32>
    %alloc_18 = memref.alloc() : memref<2x16x64xi16>
    %alloc_19 = memref.alloc() : memref<2x16x64xi8>
    %c16 = arith.constant 16 : index
    %c64_20 = arith.constant 64 : index
    scf.for %arg3 = %c0_3 to %dim_6 step %c16 {
      %55 = arith.addi %arg3, %c16 : index
      %56 = arith.minui %dim_6, %55 : index
      %57 = arith.subi %56, %arg3 : index
      scf.for %arg4 = %c0_3 to %dim_7 step %c64_20 {
        %58 = arith.addi %arg4, %c64_20 : index
        %59 = arith.minui %dim_7, %58 : index
        %60 = arith.subi %59, %arg4 : index
        scf.for %arg5 = %arg3 to %56 step %c1_14 {
          %61 = arith.subi %arg5, %arg3 : index
          %62 = arith.index_cast %arg5 : index to i32
          %63 = arith.sitofp %62 : i32 to f32
          %64 = arith.mulf %63, %38 : f32
          %65 = arith.mulf %63, %37 : f32
          %66 = arith.mulf %64, %cst_15 : f32
          %67 = arith.mulf %65, %cst_15 : f32
          %68 = arith.fptosi %66 : f32 to i32
          %69 = arith.fptosi %67 : f32 to i32
          %70 = vector.splat %68 : vector<64xi32>
          %71 = vector.splat %69 : vector<64xi32>
          scf.for %arg6 = %arg4 to %59 step %c64_16 {
            %72 = arith.subi %arg6, %arg4 : index
            %73 = vector.load %alloc[%arg6] : memref<?xi32>, vector<64xi32>
            %74 = vector.load %alloc_9[%arg6] : memref<?xi32>, vector<64xi32>
            %75 = arith.addi %73, %70 : vector<64xi32>
            %76 = arith.addi %74, %71 : vector<64xi32>
            %77 = arith.trunci %75 : vector<64xi32> to vector<64xi8>
            %78 = arith.trunci %76 : vector<64xi32> to vector<64xi8>
            vector.store %77, %alloc_19[%c0_13, %61, %72] : memref<2x16x64xi8>, vector<64xi8>
            vector.store %78, %alloc_19[%c1_14, %61, %72] : memref<2x16x64xi8>, vector<64xi8>
            %79 = arith.shrsi %75, %cst_17 : vector<64xi32>
            %80 = arith.shrsi %76, %cst_17 : vector<64xi32>
            %81 = arith.trunci %79 : vector<64xi32> to vector<64xi16>
            %82 = arith.trunci %80 : vector<64xi32> to vector<64xi16>
            vector.store %81, %alloc_18[%c0_13, %61, %72] : memref<2x16x64xi16>, vector<64xi16>
            vector.store %82, %alloc_18[%c1_14, %61, %72] : memref<2x16x64xi16>, vector<64xi16>
          }
        }
        %c0_21 = arith.constant 0 : index
        %c1_22 = arith.constant 1 : index
        %dim_23 = memref.dim %arg0, %c0_21 : memref<?x?xf32>
        %dim_24 = memref.dim %arg0, %c1_22 : memref<?x?xf32>
        scf.for %arg5 = %c0_21 to %57 step %c1_22 {
          %61 = arith.addi %arg5, %arg3 : index
          scf.for %arg6 = %c0_21 to %60 step %c1_22 {
            %62 = arith.addi %arg6, %arg4 : index
            %63 = memref.load %alloc_18[%c0_21, %arg5, %arg6] : memref<2x16x64xi16>
            %64 = memref.load %alloc_18[%c1_22, %arg5, %arg6] : memref<2x16x64xi16>
            %65 = arith.index_cast %63 : i16 to index
            %66 = arith.index_cast %64 : i16 to index
            %67 = arith.cmpi sle, %c0_21, %65 : index
            %68 = arith.cmpi slt, %65, %dim_24 : index
            %69 = arith.andi %67, %68 : i1
            %70 = arith.cmpi sle, %c0_21, %66 : index
            %71 = arith.cmpi slt, %66, %dim_23 : index
            %72 = arith.andi %70, %71 : i1
            %73 = arith.andi %69, %72 : i1
            scf.if %73 {
              %74 = memref.load %arg0[%66, %65] : memref<?x?xf32>
              memref.store %74, %arg2[%61, %62] : memref<?x?xf32>
            }
          }
        }
      }
    }
    memref.dealloc %alloc_18 : memref<2x16x64xi16>
    memref.dealloc %alloc_19 : memref<2x16x64xi8>
    memref.dealloc %alloc : memref<?xi32>
    memref.dealloc %alloc_9 : memref<?xi32>
    return
  }
  func.func @resize_2d_nearest_neighbour_interpolation(%arg0: memref<?x?xf32>, %arg1: f32, %arg2: f32, %arg3: memref<?x?xf32>) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %c0 : index to i32
    %1 = arith.sitofp %0 : i32 to f32
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg3, %c0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg3, %c1 : memref<?x?xf32>
    %2 = arith.divui %dim_2, %c64 : index
    %3 = arith.muli %c64, %2 : index
    %4 = arith.subi %dim_2, %3 : index
    %5 = vector.splat %arg1 : vector<64xf32>
    %6 = vector.splat %arg2 : vector<64xf32>
    %7 = arith.subi %dim, %c1 : index
    %8 = arith.index_cast %7 : index to i32
    %9 = arith.sitofp %8 : i32 to f32
    %10 = arith.subi %dim_0, %c1 : index
    %11 = arith.index_cast %10 : index to i32
    %12 = arith.sitofp %11 : i32 to f32
    %13 = arith.subi %dim_1, %c1 : index
    %14 = arith.index_cast %13 : index to i32
    %15 = arith.sitofp %14 : i32 to f32
    %16 = arith.subi %dim_2, %c1 : index
    %17 = arith.index_cast %16 : index to i32
    %18 = arith.sitofp %17 : i32 to f32
    %19 = arith.cmpf uno, %1, %1 : f32
    %20 = arith.cmpf uno, %12, %12 : f32
    %21 = arith.cmpf uno, %1, %1 : f32
    %22 = arith.cmpf uno, %9, %9 : f32
    %23 = arith.cmpf uno, %1, %1 : f32
    %24 = arith.cmpf uno, %18, %18 : f32
    %25 = arith.cmpf uno, %1, %1 : f32
    %26 = arith.cmpf uno, %15, %15 : f32
    affine.for %arg4 = 0 to %dim_1 {
      %35 = arith.index_cast %arg4 : index to i32
      %36 = arith.sitofp %35 : i32 to f32
      %37 = vector.splat %36 : vector<64xf32>
      %38 = arith.mulf %37, %6 : vector<64xf32>
      %39 = math.ceil %38 : vector<64xf32>
      %40 = math.floor %38 : vector<64xf32>
      %41 = arith.subf %39, %38 : vector<64xf32>
      %42 = arith.subf %38, %40 : vector<64xf32>
      %43 = arith.cmpf ogt, %41, %42 : vector<64xf32>
      %44 = arith.select %43, %40, %39 : vector<64xi1>, vector<64xf32>
      affine.for %arg5 = 0 to %3 step 64 {
        %alloc = memref.alloc() : memref<64xf32>
        affine.for %arg6 = 0 to 64 {
          %53 = arith.addi %arg6, %arg5 : index
          %54 = arith.index_cast %53 : index to i32
          %55 = arith.sitofp %54 : i32 to f32
          memref.store %55, %alloc[%arg6] : memref<64xf32>
        }
        %45 = vector.load %alloc[%c0] : memref<64xf32>, vector<64xf32>
        %46 = arith.mulf %45, %5 : vector<64xf32>
        %47 = math.ceil %46 : vector<64xf32>
        %48 = math.floor %46 : vector<64xf32>
        %49 = arith.subf %47, %46 : vector<64xf32>
        %50 = arith.subf %46, %48 : vector<64xf32>
        %51 = arith.cmpf ogt, %49, %50 : vector<64xf32>
        %52 = arith.select %51, %48, %47 : vector<64xi1>, vector<64xf32>
        affine.for %arg6 = 0 to 64 {
          %53 = vector.extractelement %52[%arg6 : index] : vector<64xf32>
          %54 = vector.extractelement %44[%arg6 : index] : vector<64xf32>
          %55 = arith.cmpf ugt, %53, %1 : f32
          %56 = arith.select %55, %53, %1 : f32
          %57 = arith.select %19, %1, %56 : f32
          %58 = arith.cmpf ult, %57, %12 : f32
          %59 = arith.select %58, %57, %12 : f32
          %60 = arith.select %20, %12, %59 : f32
          %61 = arith.cmpf ugt, %54, %1 : f32
          %62 = arith.select %61, %54, %1 : f32
          %63 = arith.select %21, %1, %62 : f32
          %64 = arith.cmpf ult, %63, %9 : f32
          %65 = arith.select %64, %63, %9 : f32
          %66 = arith.select %22, %9, %65 : f32
          %67 = arith.fptoui %60 : f32 to i32
          %68 = arith.index_cast %67 : i32 to index
          %69 = arith.fptoui %66 : f32 to i32
          %70 = arith.index_cast %69 : i32 to index
          %71 = vector.extractelement %45[%arg6 : index] : vector<64xf32>
          %72 = vector.extractelement %37[%arg6 : index] : vector<64xf32>
          %73 = arith.cmpf ugt, %71, %1 : f32
          %74 = arith.select %73, %71, %1 : f32
          %75 = arith.select %23, %1, %74 : f32
          %76 = arith.cmpf ult, %75, %18 : f32
          %77 = arith.select %76, %75, %18 : f32
          %78 = arith.select %24, %18, %77 : f32
          %79 = arith.cmpf ugt, %72, %1 : f32
          %80 = arith.select %79, %72, %1 : f32
          %81 = arith.select %25, %1, %80 : f32
          %82 = arith.cmpf ult, %81, %15 : f32
          %83 = arith.select %82, %81, %15 : f32
          %84 = arith.select %26, %15, %83 : f32
          %85 = arith.fptoui %78 : f32 to i32
          %86 = arith.index_cast %85 : i32 to index
          %87 = arith.fptoui %84 : f32 to i32
          %88 = arith.index_cast %87 : i32 to index
          %89 = memref.load %arg0[%70, %68] : memref<?x?xf32>
          memref.store %89, %arg3[%88, %86] : memref<?x?xf32>
        }
      }
    }
    %27 = arith.cmpf uno, %1, %1 : f32
    %28 = arith.cmpf uno, %12, %12 : f32
    %29 = arith.cmpf uno, %1, %1 : f32
    %30 = arith.cmpf uno, %9, %9 : f32
    %31 = arith.cmpf uno, %1, %1 : f32
    %32 = arith.cmpf uno, %18, %18 : f32
    %33 = arith.cmpf uno, %1, %1 : f32
    %34 = arith.cmpf uno, %15, %15 : f32
    affine.for %arg4 = 0 to %dim_1 {
      %35 = arith.index_cast %arg4 : index to i32
      %36 = arith.sitofp %35 : i32 to f32
      %37 = vector.splat %36 : vector<64xf32>
      %38 = arith.mulf %37, %6 : vector<64xf32>
      %39 = math.ceil %38 : vector<64xf32>
      %40 = math.floor %38 : vector<64xf32>
      %41 = arith.subf %39, %38 : vector<64xf32>
      %42 = arith.subf %38, %40 : vector<64xf32>
      %43 = arith.cmpf ogt, %41, %42 : vector<64xf32>
      %44 = arith.select %43, %40, %39 : vector<64xi1>, vector<64xf32>
      affine.for %arg5 = %3 to %dim_2 step 64 {
        %alloc = memref.alloc() : memref<64xf32>
        affine.for %arg6 = 0 to %4 {
          %53 = arith.addi %arg6, %arg5 : index
          %54 = arith.index_cast %53 : index to i32
          %55 = arith.sitofp %54 : i32 to f32
          memref.store %55, %alloc[%arg6] : memref<64xf32>
        }
        %45 = vector.load %alloc[%c0] : memref<64xf32>, vector<64xf32>
        %46 = arith.mulf %45, %5 : vector<64xf32>
        %47 = math.ceil %46 : vector<64xf32>
        %48 = math.floor %46 : vector<64xf32>
        %49 = arith.subf %47, %46 : vector<64xf32>
        %50 = arith.subf %46, %48 : vector<64xf32>
        %51 = arith.cmpf ogt, %49, %50 : vector<64xf32>
        %52 = arith.select %51, %48, %47 : vector<64xi1>, vector<64xf32>
        affine.for %arg6 = 0 to %4 {
          %53 = vector.extractelement %52[%arg6 : index] : vector<64xf32>
          %54 = vector.extractelement %44[%arg6 : index] : vector<64xf32>
          %55 = arith.cmpf ugt, %53, %1 : f32
          %56 = arith.select %55, %53, %1 : f32
          %57 = arith.select %27, %1, %56 : f32
          %58 = arith.cmpf ult, %57, %12 : f32
          %59 = arith.select %58, %57, %12 : f32
          %60 = arith.select %28, %12, %59 : f32
          %61 = arith.cmpf ugt, %54, %1 : f32
          %62 = arith.select %61, %54, %1 : f32
          %63 = arith.select %29, %1, %62 : f32
          %64 = arith.cmpf ult, %63, %9 : f32
          %65 = arith.select %64, %63, %9 : f32
          %66 = arith.select %30, %9, %65 : f32
          %67 = arith.fptoui %60 : f32 to i32
          %68 = arith.index_cast %67 : i32 to index
          %69 = arith.fptoui %66 : f32 to i32
          %70 = arith.index_cast %69 : i32 to index
          %71 = vector.extractelement %45[%arg6 : index] : vector<64xf32>
          %72 = vector.extractelement %37[%arg6 : index] : vector<64xf32>
          %73 = arith.cmpf ugt, %71, %1 : f32
          %74 = arith.select %73, %71, %1 : f32
          %75 = arith.select %31, %1, %74 : f32
          %76 = arith.cmpf ult, %75, %18 : f32
          %77 = arith.select %76, %75, %18 : f32
          %78 = arith.select %32, %18, %77 : f32
          %79 = arith.cmpf ugt, %72, %1 : f32
          %80 = arith.select %79, %72, %1 : f32
          %81 = arith.select %33, %1, %80 : f32
          %82 = arith.cmpf ult, %81, %15 : f32
          %83 = arith.select %82, %81, %15 : f32
          %84 = arith.select %34, %15, %83 : f32
          %85 = arith.fptoui %78 : f32 to i32
          %86 = arith.index_cast %85 : i32 to index
          %87 = arith.fptoui %84 : f32 to i32
          %88 = arith.index_cast %87 : i32 to index
          %89 = memref.load %arg0[%70, %68] : memref<?x?xf32>
          memref.store %89, %arg3[%88, %86] : memref<?x?xf32>
        }
      }
    }
    return
  }
  func.func @resize_2d_bilinear_interpolation(%arg0: memref<?x?xf32>, %arg1: f32, %arg2: f32, %arg3: memref<?x?xf32>) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %c0 : index to i32
    %1 = arith.sitofp %0 : i32 to f32
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg3, %c0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg3, %c1 : memref<?x?xf32>
    %2 = arith.divui %dim_2, %c64 : index
    %3 = arith.muli %c64, %2 : index
    %4 = arith.subi %dim_2, %3 : index
    %5 = vector.splat %arg1 : vector<64xf32>
    %6 = vector.splat %arg2 : vector<64xf32>
    %7 = arith.subi %dim, %c1 : index
    %8 = arith.index_cast %7 : index to i32
    %9 = arith.sitofp %8 : i32 to f32
    %10 = arith.subi %dim_0, %c1 : index
    %11 = arith.index_cast %10 : index to i32
    %12 = arith.sitofp %11 : i32 to f32
    %13 = arith.subi %dim_1, %c1 : index
    %14 = arith.index_cast %13 : index to i32
    %15 = arith.sitofp %14 : i32 to f32
    %16 = arith.subi %dim_2, %c1 : index
    %17 = arith.index_cast %16 : index to i32
    %18 = arith.sitofp %17 : i32 to f32
    %19 = arith.index_cast %c1 : index to i32
    %20 = arith.sitofp %19 : i32 to f32
    %21 = arith.cmpf uno, %1, %1 : f32
    %22 = arith.cmpf uno, %18, %18 : f32
    %23 = arith.cmpf uno, %1, %1 : f32
    %24 = arith.cmpf uno, %15, %15 : f32
    %25 = arith.cmpf uno, %1, %1 : f32
    %26 = arith.cmpf uno, %12, %12 : f32
    %27 = arith.cmpf uno, %1, %1 : f32
    %28 = arith.cmpf uno, %9, %9 : f32
    %29 = arith.cmpf uno, %1, %1 : f32
    %30 = arith.cmpf uno, %12, %12 : f32
    %31 = arith.cmpf uno, %1, %1 : f32
    %32 = arith.cmpf uno, %9, %9 : f32
    %33 = arith.cmpf uno, %1, %1 : f32
    %34 = arith.cmpf uno, %12, %12 : f32
    %35 = arith.cmpf uno, %1, %1 : f32
    %36 = arith.cmpf uno, %9, %9 : f32
    affine.for %arg4 = 0 to %dim_1 {
      %53 = arith.index_cast %arg4 : index to i32
      %54 = arith.sitofp %53 : i32 to f32
      %55 = vector.splat %54 : vector<64xf32>
      %56 = arith.mulf %55, %6 : vector<64xf32>
      %57 = math.floor %56 : vector<64xf32>
      %58 = math.ceil %56 : vector<64xf32>
      %59 = arith.subf %56, %57 : vector<64xf32>
      affine.for %arg5 = 0 to %3 step 64 {
        %alloc = memref.alloc() : memref<64xf32>
        affine.for %arg6 = 0 to 64 {
          %65 = arith.addi %arg6, %arg5 : index
          %66 = arith.index_cast %65 : index to i32
          %67 = arith.sitofp %66 : i32 to f32
          memref.store %67, %alloc[%arg6] : memref<64xf32>
        }
        %60 = vector.load %alloc[%c0] : memref<64xf32>, vector<64xf32>
        %61 = arith.mulf %60, %5 : vector<64xf32>
        %62 = math.floor %61 : vector<64xf32>
        %63 = math.ceil %61 : vector<64xf32>
        %64 = arith.subf %61, %62 : vector<64xf32>
        affine.for %arg6 = 0 to 64 {
          %65 = vector.extractelement %60[%arg6 : index] : vector<64xf32>
          %66 = vector.extractelement %55[%arg6 : index] : vector<64xf32>
          %67 = arith.cmpf ugt, %65, %1 : f32
          %68 = arith.select %67, %65, %1 : f32
          %69 = arith.select %21, %1, %68 : f32
          %70 = arith.cmpf ult, %69, %18 : f32
          %71 = arith.select %70, %69, %18 : f32
          %72 = arith.select %22, %18, %71 : f32
          %73 = arith.cmpf ugt, %66, %1 : f32
          %74 = arith.select %73, %66, %1 : f32
          %75 = arith.select %23, %1, %74 : f32
          %76 = arith.cmpf ult, %75, %15 : f32
          %77 = arith.select %76, %75, %15 : f32
          %78 = arith.select %24, %15, %77 : f32
          %79 = arith.fptoui %72 : f32 to i32
          %80 = arith.index_cast %79 : i32 to index
          %81 = arith.fptoui %78 : f32 to i32
          %82 = arith.index_cast %81 : i32 to index
          %83 = vector.extractelement %62[%arg6 : index] : vector<64xf32>
          %84 = vector.extractelement %57[%arg6 : index] : vector<64xf32>
          %85 = arith.cmpf ugt, %83, %1 : f32
          %86 = arith.select %85, %83, %1 : f32
          %87 = arith.select %25, %1, %86 : f32
          %88 = arith.cmpf ult, %87, %12 : f32
          %89 = arith.select %88, %87, %12 : f32
          %90 = arith.select %26, %12, %89 : f32
          %91 = arith.cmpf ugt, %84, %1 : f32
          %92 = arith.select %91, %84, %1 : f32
          %93 = arith.select %27, %1, %92 : f32
          %94 = arith.cmpf ult, %93, %9 : f32
          %95 = arith.select %94, %93, %9 : f32
          %96 = arith.select %28, %9, %95 : f32
          %97 = arith.fptoui %90 : f32 to i32
          %98 = arith.index_cast %97 : i32 to index
          %99 = arith.fptoui %96 : f32 to i32
          %100 = arith.index_cast %99 : i32 to index
          %101 = vector.extractelement %63[%arg6 : index] : vector<64xf32>
          %102 = vector.extractelement %58[%arg6 : index] : vector<64xf32>
          %103 = arith.cmpf ugt, %101, %1 : f32
          %104 = arith.select %103, %101, %1 : f32
          %105 = arith.select %29, %1, %104 : f32
          %106 = arith.cmpf ult, %105, %12 : f32
          %107 = arith.select %106, %105, %12 : f32
          %108 = arith.select %30, %12, %107 : f32
          %109 = arith.cmpf ugt, %102, %1 : f32
          %110 = arith.select %109, %102, %1 : f32
          %111 = arith.select %31, %1, %110 : f32
          %112 = arith.cmpf ult, %111, %9 : f32
          %113 = arith.select %112, %111, %9 : f32
          %114 = arith.select %32, %9, %113 : f32
          %115 = arith.fptoui %108 : f32 to i32
          %116 = arith.index_cast %115 : i32 to index
          %117 = arith.fptoui %114 : f32 to i32
          %118 = arith.index_cast %117 : i32 to index
          %119 = vector.extractelement %64[%arg6 : index] : vector<64xf32>
          %120 = vector.extractelement %59[%arg6 : index] : vector<64xf32>
          %121 = arith.cmpf ugt, %119, %1 : f32
          %122 = arith.select %121, %119, %1 : f32
          %123 = arith.select %33, %1, %122 : f32
          %124 = arith.cmpf ult, %123, %12 : f32
          %125 = arith.select %124, %123, %12 : f32
          %126 = arith.select %34, %12, %125 : f32
          %127 = arith.cmpf ugt, %120, %1 : f32
          %128 = arith.select %127, %120, %1 : f32
          %129 = arith.select %35, %1, %128 : f32
          %130 = arith.cmpf ult, %129, %9 : f32
          %131 = arith.select %130, %129, %9 : f32
          %132 = arith.select %36, %9, %131 : f32
          %133 = arith.subf %20, %126 : f32
          %134 = arith.subf %20, %132 : f32
          %135 = memref.load %arg0[%100, %98] : memref<?x?xf32>
          %136 = memref.load %arg0[%118, %98] : memref<?x?xf32>
          %137 = memref.load %arg0[%100, %116] : memref<?x?xf32>
          %138 = memref.load %arg0[%118, %116] : memref<?x?xf32>
          %139 = arith.mulf %133, %134 : f32
          %140 = arith.mulf %126, %134 : f32
          %141 = arith.mulf %132, %133 : f32
          %142 = arith.mulf %126, %132 : f32
          %143 = arith.mulf %135, %139 : f32
          %144 = arith.mulf %136, %140 : f32
          %145 = arith.mulf %137, %141 : f32
          %146 = arith.mulf %138, %142 : f32
          %147 = arith.addf %143, %144 : f32
          %148 = arith.addf %145, %146 : f32
          %149 = arith.addf %147, %148 : f32
          %150 = math.ceil %149 : f32
          %151 = math.floor %149 : f32
          %152 = arith.subf %150, %149 : f32
          %153 = arith.subf %149, %151 : f32
          %154 = arith.cmpf ogt, %152, %153 : f32
          %155 = arith.select %154, %151, %150 : f32
          memref.store %155, %arg3[%82, %80] : memref<?x?xf32>
        }
      }
    }
    %37 = arith.cmpf uno, %1, %1 : f32
    %38 = arith.cmpf uno, %18, %18 : f32
    %39 = arith.cmpf uno, %1, %1 : f32
    %40 = arith.cmpf uno, %15, %15 : f32
    %41 = arith.cmpf uno, %1, %1 : f32
    %42 = arith.cmpf uno, %12, %12 : f32
    %43 = arith.cmpf uno, %1, %1 : f32
    %44 = arith.cmpf uno, %9, %9 : f32
    %45 = arith.cmpf uno, %1, %1 : f32
    %46 = arith.cmpf uno, %12, %12 : f32
    %47 = arith.cmpf uno, %1, %1 : f32
    %48 = arith.cmpf uno, %9, %9 : f32
    %49 = arith.cmpf uno, %1, %1 : f32
    %50 = arith.cmpf uno, %12, %12 : f32
    %51 = arith.cmpf uno, %1, %1 : f32
    %52 = arith.cmpf uno, %9, %9 : f32
    affine.for %arg4 = 0 to %dim_1 {
      %53 = arith.index_cast %arg4 : index to i32
      %54 = arith.sitofp %53 : i32 to f32
      %55 = vector.splat %54 : vector<64xf32>
      %56 = arith.mulf %55, %6 : vector<64xf32>
      %57 = math.floor %56 : vector<64xf32>
      %58 = math.ceil %56 : vector<64xf32>
      %59 = arith.subf %56, %57 : vector<64xf32>
      affine.for %arg5 = %3 to %dim_2 step 64 {
        %alloc = memref.alloc() : memref<64xf32>
        affine.for %arg6 = 0 to %4 {
          %65 = arith.addi %arg6, %arg5 : index
          %66 = arith.index_cast %65 : index to i32
          %67 = arith.sitofp %66 : i32 to f32
          memref.store %67, %alloc[%arg6] : memref<64xf32>
        }
        %60 = vector.load %alloc[%c0] : memref<64xf32>, vector<64xf32>
        %61 = arith.mulf %60, %5 : vector<64xf32>
        %62 = math.floor %61 : vector<64xf32>
        %63 = math.ceil %61 : vector<64xf32>
        %64 = arith.subf %61, %62 : vector<64xf32>
        affine.for %arg6 = 0 to %4 {
          %65 = vector.extractelement %60[%arg6 : index] : vector<64xf32>
          %66 = vector.extractelement %55[%arg6 : index] : vector<64xf32>
          %67 = arith.cmpf ugt, %65, %1 : f32
          %68 = arith.select %67, %65, %1 : f32
          %69 = arith.select %37, %1, %68 : f32
          %70 = arith.cmpf ult, %69, %18 : f32
          %71 = arith.select %70, %69, %18 : f32
          %72 = arith.select %38, %18, %71 : f32
          %73 = arith.cmpf ugt, %66, %1 : f32
          %74 = arith.select %73, %66, %1 : f32
          %75 = arith.select %39, %1, %74 : f32
          %76 = arith.cmpf ult, %75, %15 : f32
          %77 = arith.select %76, %75, %15 : f32
          %78 = arith.select %40, %15, %77 : f32
          %79 = arith.fptoui %72 : f32 to i32
          %80 = arith.index_cast %79 : i32 to index
          %81 = arith.fptoui %78 : f32 to i32
          %82 = arith.index_cast %81 : i32 to index
          %83 = vector.extractelement %62[%arg6 : index] : vector<64xf32>
          %84 = vector.extractelement %57[%arg6 : index] : vector<64xf32>
          %85 = arith.cmpf ugt, %83, %1 : f32
          %86 = arith.select %85, %83, %1 : f32
          %87 = arith.select %41, %1, %86 : f32
          %88 = arith.cmpf ult, %87, %12 : f32
          %89 = arith.select %88, %87, %12 : f32
          %90 = arith.select %42, %12, %89 : f32
          %91 = arith.cmpf ugt, %84, %1 : f32
          %92 = arith.select %91, %84, %1 : f32
          %93 = arith.select %43, %1, %92 : f32
          %94 = arith.cmpf ult, %93, %9 : f32
          %95 = arith.select %94, %93, %9 : f32
          %96 = arith.select %44, %9, %95 : f32
          %97 = arith.fptoui %90 : f32 to i32
          %98 = arith.index_cast %97 : i32 to index
          %99 = arith.fptoui %96 : f32 to i32
          %100 = arith.index_cast %99 : i32 to index
          %101 = vector.extractelement %63[%arg6 : index] : vector<64xf32>
          %102 = vector.extractelement %58[%arg6 : index] : vector<64xf32>
          %103 = arith.cmpf ugt, %101, %1 : f32
          %104 = arith.select %103, %101, %1 : f32
          %105 = arith.select %45, %1, %104 : f32
          %106 = arith.cmpf ult, %105, %12 : f32
          %107 = arith.select %106, %105, %12 : f32
          %108 = arith.select %46, %12, %107 : f32
          %109 = arith.cmpf ugt, %102, %1 : f32
          %110 = arith.select %109, %102, %1 : f32
          %111 = arith.select %47, %1, %110 : f32
          %112 = arith.cmpf ult, %111, %9 : f32
          %113 = arith.select %112, %111, %9 : f32
          %114 = arith.select %48, %9, %113 : f32
          %115 = arith.fptoui %108 : f32 to i32
          %116 = arith.index_cast %115 : i32 to index
          %117 = arith.fptoui %114 : f32 to i32
          %118 = arith.index_cast %117 : i32 to index
          %119 = vector.extractelement %64[%arg6 : index] : vector<64xf32>
          %120 = vector.extractelement %59[%arg6 : index] : vector<64xf32>
          %121 = arith.cmpf ugt, %119, %1 : f32
          %122 = arith.select %121, %119, %1 : f32
          %123 = arith.select %49, %1, %122 : f32
          %124 = arith.cmpf ult, %123, %12 : f32
          %125 = arith.select %124, %123, %12 : f32
          %126 = arith.select %50, %12, %125 : f32
          %127 = arith.cmpf ugt, %120, %1 : f32
          %128 = arith.select %127, %120, %1 : f32
          %129 = arith.select %51, %1, %128 : f32
          %130 = arith.cmpf ult, %129, %9 : f32
          %131 = arith.select %130, %129, %9 : f32
          %132 = arith.select %52, %9, %131 : f32
          %133 = arith.subf %20, %126 : f32
          %134 = arith.subf %20, %132 : f32
          %135 = memref.load %arg0[%100, %98] : memref<?x?xf32>
          %136 = memref.load %arg0[%118, %98] : memref<?x?xf32>
          %137 = memref.load %arg0[%100, %116] : memref<?x?xf32>
          %138 = memref.load %arg0[%118, %116] : memref<?x?xf32>
          %139 = arith.mulf %133, %134 : f32
          %140 = arith.mulf %126, %134 : f32
          %141 = arith.mulf %132, %133 : f32
          %142 = arith.mulf %126, %132 : f32
          %143 = arith.mulf %135, %139 : f32
          %144 = arith.mulf %136, %140 : f32
          %145 = arith.mulf %137, %141 : f32
          %146 = arith.mulf %138, %142 : f32
          %147 = arith.addf %143, %144 : f32
          %148 = arith.addf %145, %146 : f32
          %149 = arith.addf %147, %148 : f32
          %150 = math.ceil %149 : f32
          %151 = math.floor %149 : f32
          %152 = arith.subf %150, %149 : f32
          %153 = arith.subf %149, %151 : f32
          %154 = arith.cmpf ogt, %152, %153 : f32
          %155 = arith.select %154, %151, %150 : f32
          memref.store %155, %arg3[%82, %80] : memref<?x?xf32>
        }
      }
    }
    return
  }
  func.func @erosion_2d_constant_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: index, %arg5: index, %arg6: index, %arg7: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg0, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg5 : index
    %1 = arith.addi %dim_2, %arg4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg8) = (0) to (symbol(%dim)) {
      affine.parallel (%arg9) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg10) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg11) = (0) to (symbol(%dim_4)) {
            %3 = arith.addi %arg8, %arg9 : index
            %4 = arith.addi %arg10, %arg11 : index
            %5 = arith.subi %3, %arg5 : index
            %6 = arith.subi %4, %arg4 : index
            %7 = arith.addi %4, %c64 : index
            %8 = arith.cmpi slt, %3, %arg5 : index
          }
        }
      }
    }
    affine.for %arg8 = 0 to %arg6 {
      %3 = arith.cmpi sge, %arg8, %c1 : index
      scf.if %3 {
        memref.copy %arg2, %arg0 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg3, %arg2 : memref<?x?xf32> to memref<?x?xf32>
    }
    return
  }
  func.func @erosion_2d_replicate_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: index, %arg5: index, %arg6: index, %arg7: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg0, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg5 : index
    %1 = arith.addi %dim_2, %arg4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg8) = (0) to (symbol(%dim)) {
      affine.parallel (%arg9) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg10) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg11) = (0) to (symbol(%dim_4)) {
            %3 = arith.addi %arg8, %arg9 : index
            %4 = arith.addi %arg10, %arg11 : index
            %5 = arith.subi %3, %arg5 : index
            %6 = arith.subi %4, %arg4 : index
            %7 = arith.addi %4, %c64 : index
            %8 = arith.cmpi slt, %3, %arg5 : index
          }
        }
      }
    }
    affine.for %arg8 = 0 to %arg6 {
      %3 = arith.cmpi sge, %arg8, %c1 : index
      scf.if %3 {
        memref.copy %arg2, %arg0 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg3, %arg2 : memref<?x?xf32> to memref<?x?xf32>
    }
    return
  }
  func.func @dilation_2d_constant_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: index, %arg5: index, %arg6: index, %arg7: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg0, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg5 : index
    %1 = arith.addi %dim_2, %arg4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg8) = (0) to (symbol(%dim)) {
      affine.parallel (%arg9) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg10) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg11) = (0) to (symbol(%dim_4)) {
            %3 = arith.addi %arg8, %arg9 : index
            %4 = arith.addi %arg10, %arg11 : index
            %5 = arith.subi %3, %arg5 : index
            %6 = arith.subi %4, %arg4 : index
            %7 = arith.addi %4, %c64 : index
            %8 = arith.cmpi slt, %3, %arg5 : index
          }
        }
      }
    }
    affine.for %arg8 = 0 to %arg6 {
      %3 = arith.cmpi sge, %arg8, %c1 : index
      scf.if %3 {
        memref.copy %arg2, %arg0 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg3, %arg2 : memref<?x?xf32> to memref<?x?xf32>
    }
    return
  }
  func.func @dilation_2d_replicate_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: index, %arg5: index, %arg6: index, %arg7: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg0, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg5 : index
    %1 = arith.addi %dim_2, %arg4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg8) = (0) to (symbol(%dim)) {
      affine.parallel (%arg9) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg10) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg11) = (0) to (symbol(%dim_4)) {
            %3 = arith.addi %arg8, %arg9 : index
            %4 = arith.addi %arg10, %arg11 : index
            %5 = arith.subi %3, %arg5 : index
            %6 = arith.subi %4, %arg4 : index
            %7 = arith.addi %4, %c64 : index
            %8 = arith.cmpi slt, %3, %arg5 : index
          }
        }
      }
    }
    affine.for %arg8 = 0 to %arg6 {
      %3 = arith.cmpi sge, %arg8, %c1 : index
      scf.if %3 {
        memref.copy %arg2, %arg0 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg3, %arg2 : memref<?x?xf32> to memref<?x?xf32>
    }
    return
  }
  func.func @opening_2d_constant_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>, %arg6: index, %arg7: index, %arg8: index, %arg9: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg0, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg7 : index
    %1 = arith.addi %dim_2, %arg6 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg10) = (0) to (symbol(%dim)) {
      affine.parallel (%arg11) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg12) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg13) = (0) to (symbol(%dim_4)) {
            %6 = arith.addi %arg10, %arg11 : index
            %7 = arith.addi %arg12, %arg13 : index
            %8 = arith.subi %6, %arg7 : index
            %9 = arith.subi %7, %arg6 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg7 : index
          }
        }
      }
    }
    affine.for %arg10 = 0 to %arg8 {
      %6 = arith.cmpi sge, %arg10, %c1 : index
      scf.if %6 {
        memref.copy %arg3, %arg0 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg4, %arg3 : memref<?x?xf32> to memref<?x?xf32>
    }
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg3, %c0_6 : memref<?x?xf32>
    %dim_9 = memref.dim %arg3, %c1_7 : memref<?x?xf32>
    %dim_10 = memref.dim %arg1, %c0_6 : memref<?x?xf32>
    %dim_11 = memref.dim %arg1, %c1_7 : memref<?x?xf32>
    %3 = arith.addi %dim_8, %arg7 : index
    %4 = arith.addi %dim_9, %arg6 : index
    %cst_12 = arith.constant 0.000000e+00 : f32
    %cst_13 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %5 = affine.apply #map(%dim_9, %dim_11, %c1_7)
    affine.parallel (%arg10) = (0) to (symbol(%dim_8)) {
      affine.parallel (%arg11) = (0) to (symbol(%dim_10)) {
        affine.parallel (%arg12) = (0) to (symbol(%dim_9)) step (64) {
          affine.parallel (%arg13) = (0) to (symbol(%dim_11)) {
            %6 = arith.addi %arg10, %arg11 : index
            %7 = arith.addi %arg12, %arg13 : index
            %8 = arith.subi %6, %arg7 : index
            %9 = arith.subi %7, %arg6 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg7 : index
          }
        }
      }
    }
    affine.for %arg10 = 0 to %arg8 {
      %6 = arith.cmpi sge, %arg10, %c1 : index
      scf.if %6 {
        memref.copy %arg2, %arg3 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg5, %arg2 : memref<?x?xf32> to memref<?x?xf32>
    }
    return
  }
  func.func @opening_2d_replicate_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>, %arg6: index, %arg7: index, %arg8: index, %arg9: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg0, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg7 : index
    %1 = arith.addi %dim_2, %arg6 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg10) = (0) to (symbol(%dim)) {
      affine.parallel (%arg11) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg12) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg13) = (0) to (symbol(%dim_4)) {
            %6 = arith.addi %arg10, %arg11 : index
            %7 = arith.addi %arg12, %arg13 : index
            %8 = arith.subi %6, %arg7 : index
            %9 = arith.subi %7, %arg6 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg7 : index
          }
        }
      }
    }
    affine.for %arg10 = 0 to %arg8 {
      %6 = arith.cmpi sge, %arg10, %c1 : index
      scf.if %6 {
        memref.copy %arg3, %arg0 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg4, %arg3 : memref<?x?xf32> to memref<?x?xf32>
    }
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg3, %c0_6 : memref<?x?xf32>
    %dim_9 = memref.dim %arg3, %c1_7 : memref<?x?xf32>
    %dim_10 = memref.dim %arg1, %c0_6 : memref<?x?xf32>
    %dim_11 = memref.dim %arg1, %c1_7 : memref<?x?xf32>
    %3 = arith.addi %dim_8, %arg7 : index
    %4 = arith.addi %dim_9, %arg6 : index
    %cst_12 = arith.constant 0.000000e+00 : f32
    %cst_13 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %5 = affine.apply #map(%dim_9, %dim_11, %c1_7)
    affine.parallel (%arg10) = (0) to (symbol(%dim_8)) {
      affine.parallel (%arg11) = (0) to (symbol(%dim_10)) {
        affine.parallel (%arg12) = (0) to (symbol(%dim_9)) step (64) {
          affine.parallel (%arg13) = (0) to (symbol(%dim_11)) {
            %6 = arith.addi %arg10, %arg11 : index
            %7 = arith.addi %arg12, %arg13 : index
            %8 = arith.subi %6, %arg7 : index
            %9 = arith.subi %7, %arg6 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg7 : index
          }
        }
      }
    }
    affine.for %arg10 = 0 to %arg8 {
      %6 = arith.cmpi sge, %arg10, %c1 : index
      scf.if %6 {
        memref.copy %arg2, %arg3 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg5, %arg2 : memref<?x?xf32> to memref<?x?xf32>
    }
    return
  }
  func.func @closing_2d_constant_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>, %arg6: index, %arg7: index, %arg8: index, %arg9: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg0, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg7 : index
    %1 = arith.addi %dim_2, %arg6 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg10) = (0) to (symbol(%dim)) {
      affine.parallel (%arg11) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg12) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg13) = (0) to (symbol(%dim_4)) {
            %6 = arith.addi %arg10, %arg11 : index
            %7 = arith.addi %arg12, %arg13 : index
            %8 = arith.subi %6, %arg7 : index
            %9 = arith.subi %7, %arg6 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg7 : index
          }
        }
      }
    }
    affine.for %arg10 = 0 to %arg8 {
      %6 = arith.cmpi sge, %arg10, %c1 : index
      scf.if %6 {
        memref.copy %arg3, %arg0 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg4, %arg3 : memref<?x?xf32> to memref<?x?xf32>
    }
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg3, %c0_6 : memref<?x?xf32>
    %dim_9 = memref.dim %arg3, %c1_7 : memref<?x?xf32>
    %dim_10 = memref.dim %arg1, %c0_6 : memref<?x?xf32>
    %dim_11 = memref.dim %arg1, %c1_7 : memref<?x?xf32>
    %3 = arith.addi %dim_8, %arg7 : index
    %4 = arith.addi %dim_9, %arg6 : index
    %cst_12 = arith.constant 0.000000e+00 : f32
    %cst_13 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %5 = affine.apply #map(%dim_9, %dim_11, %c1_7)
    affine.parallel (%arg10) = (0) to (symbol(%dim_8)) {
      affine.parallel (%arg11) = (0) to (symbol(%dim_10)) {
        affine.parallel (%arg12) = (0) to (symbol(%dim_9)) step (64) {
          affine.parallel (%arg13) = (0) to (symbol(%dim_11)) {
            %6 = arith.addi %arg10, %arg11 : index
            %7 = arith.addi %arg12, %arg13 : index
            %8 = arith.subi %6, %arg7 : index
            %9 = arith.subi %7, %arg6 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg7 : index
          }
        }
      }
    }
    affine.for %arg10 = 0 to %arg8 {
      %6 = arith.cmpi sge, %arg10, %c1 : index
      scf.if %6 {
        memref.copy %arg2, %arg3 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg5, %arg2 : memref<?x?xf32> to memref<?x?xf32>
    }
    return
  }
  func.func @closing_2d_replicate_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>, %arg6: index, %arg7: index, %arg8: index, %arg9: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg0, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg7 : index
    %1 = arith.addi %dim_2, %arg6 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg10) = (0) to (symbol(%dim)) {
      affine.parallel (%arg11) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg12) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg13) = (0) to (symbol(%dim_4)) {
            %6 = arith.addi %arg10, %arg11 : index
            %7 = arith.addi %arg12, %arg13 : index
            %8 = arith.subi %6, %arg7 : index
            %9 = arith.subi %7, %arg6 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg7 : index
          }
        }
      }
    }
    affine.for %arg10 = 0 to %arg8 {
      %6 = arith.cmpi sge, %arg10, %c1 : index
      scf.if %6 {
        memref.copy %arg3, %arg0 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg4, %arg3 : memref<?x?xf32> to memref<?x?xf32>
    }
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg3, %c0_6 : memref<?x?xf32>
    %dim_9 = memref.dim %arg3, %c1_7 : memref<?x?xf32>
    %dim_10 = memref.dim %arg1, %c0_6 : memref<?x?xf32>
    %dim_11 = memref.dim %arg1, %c1_7 : memref<?x?xf32>
    %3 = arith.addi %dim_8, %arg7 : index
    %4 = arith.addi %dim_9, %arg6 : index
    %cst_12 = arith.constant 0.000000e+00 : f32
    %cst_13 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %5 = affine.apply #map(%dim_9, %dim_11, %c1_7)
    affine.parallel (%arg10) = (0) to (symbol(%dim_8)) {
      affine.parallel (%arg11) = (0) to (symbol(%dim_10)) {
        affine.parallel (%arg12) = (0) to (symbol(%dim_9)) step (64) {
          affine.parallel (%arg13) = (0) to (symbol(%dim_11)) {
            %6 = arith.addi %arg10, %arg11 : index
            %7 = arith.addi %arg12, %arg13 : index
            %8 = arith.subi %6, %arg7 : index
            %9 = arith.subi %7, %arg6 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg7 : index
          }
        }
      }
    }
    affine.for %arg10 = 0 to %arg8 {
      %6 = arith.cmpi sge, %arg10, %c1 : index
      scf.if %6 {
        memref.copy %arg2, %arg3 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg5, %arg2 : memref<?x?xf32> to memref<?x?xf32>
    }
    return
  }
  func.func @tophat_2d_constant_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>, %arg6: memref<?x?xf32>, %arg7: memref<?x?xf32>, %arg8: index, %arg9: index, %arg10: index, %arg11: f32) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    memref.copy %arg0, %arg5 : memref<?x?xf32> to memref<?x?xf32>
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg5, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg5, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg9 : index
    %1 = arith.addi %dim_2, %arg8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg12) = (0) to (symbol(%dim)) {
      affine.parallel (%arg13) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg14) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg15) = (0) to (symbol(%dim_4)) {
            %6 = arith.addi %arg12, %arg13 : index
            %7 = arith.addi %arg14, %arg15 : index
            %8 = arith.subi %6, %arg9 : index
            %9 = arith.subi %7, %arg8 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg9 : index
          }
        }
      }
    }
    affine.for %arg12 = 0 to %arg10 {
      %6 = arith.cmpi sge, %arg12, %c1 : index
      scf.if %6 {
        memref.copy %arg3, %arg5 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg6, %arg3 : memref<?x?xf32> to memref<?x?xf32>
    }
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg3, %c0_6 : memref<?x?xf32>
    %dim_9 = memref.dim %arg3, %c1_7 : memref<?x?xf32>
    %dim_10 = memref.dim %arg1, %c0_6 : memref<?x?xf32>
    %dim_11 = memref.dim %arg1, %c1_7 : memref<?x?xf32>
    %3 = arith.addi %dim_8, %arg9 : index
    %4 = arith.addi %dim_9, %arg8 : index
    %cst_12 = arith.constant 0.000000e+00 : f32
    %cst_13 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %5 = affine.apply #map(%dim_9, %dim_11, %c1_7)
    affine.parallel (%arg12) = (0) to (symbol(%dim_8)) {
      affine.parallel (%arg13) = (0) to (symbol(%dim_10)) {
        affine.parallel (%arg14) = (0) to (symbol(%dim_9)) step (64) {
          affine.parallel (%arg15) = (0) to (symbol(%dim_11)) {
            %6 = arith.addi %arg12, %arg13 : index
            %7 = arith.addi %arg14, %arg15 : index
            %8 = arith.subi %6, %arg9 : index
            %9 = arith.subi %7, %arg8 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg9 : index
          }
        }
      }
    }
    affine.for %arg12 = 0 to %arg10 {
      %6 = arith.cmpi sge, %arg12, %c1 : index
      scf.if %6 {
        memref.copy %arg4, %arg3 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg7, %arg4 : memref<?x?xf32> to memref<?x?xf32>
    }
    %dim_14 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %dim_15 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %cst_16 = arith.constant 0.000000e+00 : f32
    %cst_17 = arith.constant dense<0.000000e+00> : vector<64xf32>
    affine.for %arg12 = 0 to %dim_14 {
      affine.for %arg13 = 0 to %dim_15 step 64 {
        %6 = arith.addi %arg13, %c64 : index
        %7 = arith.subi %6, %c1 : index
        %8 = arith.cmpi sgt, %7, %dim_15 : index
        scf.if %8 {
          %9 = arith.subi %7, %dim_15 : index
          %10 = arith.subi %c64, %9 : index
          %11 = vector.create_mask %10 : vector<64xi1>
          %12 = vector.maskedload %arg4[%arg12, %arg13], %11, %cst_17 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %13 = vector.maskedload %arg0[%arg12, %arg13], %11, %cst_17 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %14 = arith.subf %13, %12 : vector<64xf32>
          vector.maskedstore %arg2[%arg12, %arg13], %11, %14 : memref<?x?xf32>, vector<64xi1>, vector<64xf32>
        } else {
          %9 = vector.load %arg4[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
          %10 = vector.load %arg0[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
          %11 = arith.subf %10, %9 : vector<64xf32>
          vector.store %11, %arg2[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
        }
      }
    }
    return
  }
  func.func @tophat_2d_replicate_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>, %arg6: memref<?x?xf32>, %arg7: memref<?x?xf32>, %arg8: index, %arg9: index, %arg10: index, %arg11: f32) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    memref.copy %arg0, %arg5 : memref<?x?xf32> to memref<?x?xf32>
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg5, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg5, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg9 : index
    %1 = arith.addi %dim_2, %arg8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg12) = (0) to (symbol(%dim)) {
      affine.parallel (%arg13) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg14) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg15) = (0) to (symbol(%dim_4)) {
            %6 = arith.addi %arg12, %arg13 : index
            %7 = arith.addi %arg14, %arg15 : index
            %8 = arith.subi %6, %arg9 : index
            %9 = arith.subi %7, %arg8 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg9 : index
          }
        }
      }
    }
    affine.for %arg12 = 0 to %arg10 {
      %6 = arith.cmpi sge, %arg12, %c1 : index
      scf.if %6 {
        memref.copy %arg3, %arg5 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg6, %arg3 : memref<?x?xf32> to memref<?x?xf32>
    }
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg3, %c0_6 : memref<?x?xf32>
    %dim_9 = memref.dim %arg3, %c1_7 : memref<?x?xf32>
    %dim_10 = memref.dim %arg1, %c0_6 : memref<?x?xf32>
    %dim_11 = memref.dim %arg1, %c1_7 : memref<?x?xf32>
    %3 = arith.addi %dim_8, %arg9 : index
    %4 = arith.addi %dim_9, %arg8 : index
    %cst_12 = arith.constant 0.000000e+00 : f32
    %cst_13 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %5 = affine.apply #map(%dim_9, %dim_11, %c1_7)
    affine.parallel (%arg12) = (0) to (symbol(%dim_8)) {
      affine.parallel (%arg13) = (0) to (symbol(%dim_10)) {
        affine.parallel (%arg14) = (0) to (symbol(%dim_9)) step (64) {
          affine.parallel (%arg15) = (0) to (symbol(%dim_11)) {
            %6 = arith.addi %arg12, %arg13 : index
            %7 = arith.addi %arg14, %arg15 : index
            %8 = arith.subi %6, %arg9 : index
            %9 = arith.subi %7, %arg8 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg9 : index
          }
        }
      }
    }
    affine.for %arg12 = 0 to %arg10 {
      %6 = arith.cmpi sge, %arg12, %c1 : index
      scf.if %6 {
        memref.copy %arg4, %arg3 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg7, %arg4 : memref<?x?xf32> to memref<?x?xf32>
    }
    %dim_14 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %dim_15 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %cst_16 = arith.constant 0.000000e+00 : f32
    %cst_17 = arith.constant dense<0.000000e+00> : vector<64xf32>
    affine.for %arg12 = 0 to %dim_14 {
      affine.for %arg13 = 0 to %dim_15 step 64 {
        %6 = arith.addi %arg13, %c64 : index
        %7 = arith.subi %6, %c1 : index
        %8 = arith.cmpi sgt, %7, %dim_15 : index
        scf.if %8 {
          %9 = arith.subi %7, %dim_15 : index
          %10 = arith.subi %c64, %9 : index
          %11 = vector.create_mask %10 : vector<64xi1>
          %12 = vector.maskedload %arg4[%arg12, %arg13], %11, %cst_17 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %13 = vector.maskedload %arg0[%arg12, %arg13], %11, %cst_17 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %14 = arith.subf %13, %12 : vector<64xf32>
          vector.maskedstore %arg2[%arg12, %arg13], %11, %14 : memref<?x?xf32>, vector<64xi1>, vector<64xf32>
        } else {
          %9 = vector.load %arg4[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
          %10 = vector.load %arg0[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
          %11 = arith.subf %10, %9 : vector<64xf32>
          vector.store %11, %arg2[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
        }
      }
    }
    return
  }
  func.func @bottomhat_2d_constant_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>, %arg6: memref<?x?xf32>, %arg7: memref<?x?xf32>, %arg8: index, %arg9: index, %arg10: index, %arg11: f32) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    memref.copy %arg0, %arg5 : memref<?x?xf32> to memref<?x?xf32>
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg5, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg5, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg9 : index
    %1 = arith.addi %dim_2, %arg8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg12) = (0) to (symbol(%dim)) {
      affine.parallel (%arg13) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg14) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg15) = (0) to (symbol(%dim_4)) {
            %6 = arith.addi %arg12, %arg13 : index
            %7 = arith.addi %arg14, %arg15 : index
            %8 = arith.subi %6, %arg9 : index
            %9 = arith.subi %7, %arg8 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg9 : index
          }
        }
      }
    }
    affine.for %arg12 = 0 to %arg10 {
      %6 = arith.cmpi sge, %arg12, %c1 : index
      scf.if %6 {
        memref.copy %arg3, %arg5 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg6, %arg3 : memref<?x?xf32> to memref<?x?xf32>
    }
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg3, %c0_6 : memref<?x?xf32>
    %dim_9 = memref.dim %arg3, %c1_7 : memref<?x?xf32>
    %dim_10 = memref.dim %arg1, %c0_6 : memref<?x?xf32>
    %dim_11 = memref.dim %arg1, %c1_7 : memref<?x?xf32>
    %3 = arith.addi %dim_8, %arg9 : index
    %4 = arith.addi %dim_9, %arg8 : index
    %cst_12 = arith.constant 0.000000e+00 : f32
    %cst_13 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %5 = affine.apply #map(%dim_9, %dim_11, %c1_7)
    affine.parallel (%arg12) = (0) to (symbol(%dim_8)) {
      affine.parallel (%arg13) = (0) to (symbol(%dim_10)) {
        affine.parallel (%arg14) = (0) to (symbol(%dim_9)) step (64) {
          affine.parallel (%arg15) = (0) to (symbol(%dim_11)) {
            %6 = arith.addi %arg12, %arg13 : index
            %7 = arith.addi %arg14, %arg15 : index
            %8 = arith.subi %6, %arg9 : index
            %9 = arith.subi %7, %arg8 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg9 : index
          }
        }
      }
    }
    affine.for %arg12 = 0 to %arg10 {
      %6 = arith.cmpi sge, %arg12, %c1 : index
      scf.if %6 {
        memref.copy %arg4, %arg3 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg7, %arg4 : memref<?x?xf32> to memref<?x?xf32>
    }
    %dim_14 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %dim_15 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %cst_16 = arith.constant 0.000000e+00 : f32
    %cst_17 = arith.constant dense<0.000000e+00> : vector<64xf32>
    affine.for %arg12 = 0 to %dim_14 {
      affine.for %arg13 = 0 to %dim_15 step 64 {
        %6 = arith.addi %arg13, %c64 : index
        %7 = arith.subi %6, %c1 : index
        %8 = arith.cmpi sgt, %7, %dim_15 : index
        scf.if %8 {
          %9 = arith.subi %7, %dim_15 : index
          %10 = arith.subi %c64, %9 : index
          %11 = vector.create_mask %10 : vector<64xi1>
          %12 = vector.maskedload %arg4[%arg12, %arg13], %11, %cst_17 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %13 = vector.maskedload %arg0[%arg12, %arg13], %11, %cst_17 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %14 = arith.subf %12, %13 : vector<64xf32>
          vector.maskedstore %arg2[%arg12, %arg13], %11, %14 : memref<?x?xf32>, vector<64xi1>, vector<64xf32>
        } else {
          %9 = vector.load %arg4[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
          %10 = vector.load %arg0[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
          %11 = arith.subf %9, %10 : vector<64xf32>
          vector.store %11, %arg2[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
        }
      }
    }
    return
  }
  func.func @bottomhat_2d_replicate_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>, %arg6: memref<?x?xf32>, %arg7: memref<?x?xf32>, %arg8: index, %arg9: index, %arg10: index, %arg11: f32) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    memref.copy %arg0, %arg5 : memref<?x?xf32> to memref<?x?xf32>
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg5, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg5, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg9 : index
    %1 = arith.addi %dim_2, %arg8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg12) = (0) to (symbol(%dim)) {
      affine.parallel (%arg13) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg14) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg15) = (0) to (symbol(%dim_4)) {
            %6 = arith.addi %arg12, %arg13 : index
            %7 = arith.addi %arg14, %arg15 : index
            %8 = arith.subi %6, %arg9 : index
            %9 = arith.subi %7, %arg8 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg9 : index
          }
        }
      }
    }
    affine.for %arg12 = 0 to %arg10 {
      %6 = arith.cmpi sge, %arg12, %c1 : index
      scf.if %6 {
        memref.copy %arg3, %arg5 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg6, %arg3 : memref<?x?xf32> to memref<?x?xf32>
    }
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg3, %c0_6 : memref<?x?xf32>
    %dim_9 = memref.dim %arg3, %c1_7 : memref<?x?xf32>
    %dim_10 = memref.dim %arg1, %c0_6 : memref<?x?xf32>
    %dim_11 = memref.dim %arg1, %c1_7 : memref<?x?xf32>
    %3 = arith.addi %dim_8, %arg9 : index
    %4 = arith.addi %dim_9, %arg8 : index
    %cst_12 = arith.constant 0.000000e+00 : f32
    %cst_13 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %5 = affine.apply #map(%dim_9, %dim_11, %c1_7)
    affine.parallel (%arg12) = (0) to (symbol(%dim_8)) {
      affine.parallel (%arg13) = (0) to (symbol(%dim_10)) {
        affine.parallel (%arg14) = (0) to (symbol(%dim_9)) step (64) {
          affine.parallel (%arg15) = (0) to (symbol(%dim_11)) {
            %6 = arith.addi %arg12, %arg13 : index
            %7 = arith.addi %arg14, %arg15 : index
            %8 = arith.subi %6, %arg9 : index
            %9 = arith.subi %7, %arg8 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg9 : index
          }
        }
      }
    }
    affine.for %arg12 = 0 to %arg10 {
      %6 = arith.cmpi sge, %arg12, %c1 : index
      scf.if %6 {
        memref.copy %arg4, %arg3 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg7, %arg4 : memref<?x?xf32> to memref<?x?xf32>
    }
    %dim_14 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %dim_15 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %cst_16 = arith.constant 0.000000e+00 : f32
    %cst_17 = arith.constant dense<0.000000e+00> : vector<64xf32>
    affine.for %arg12 = 0 to %dim_14 {
      affine.for %arg13 = 0 to %dim_15 step 64 {
        %6 = arith.addi %arg13, %c64 : index
        %7 = arith.subi %6, %c1 : index
        %8 = arith.cmpi sgt, %7, %dim_15 : index
        scf.if %8 {
          %9 = arith.subi %7, %dim_15 : index
          %10 = arith.subi %c64, %9 : index
          %11 = vector.create_mask %10 : vector<64xi1>
          %12 = vector.maskedload %arg4[%arg12, %arg13], %11, %cst_17 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %13 = vector.maskedload %arg0[%arg12, %arg13], %11, %cst_17 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %14 = arith.subf %12, %13 : vector<64xf32>
          vector.maskedstore %arg2[%arg12, %arg13], %11, %14 : memref<?x?xf32>, vector<64xi1>, vector<64xf32>
        } else {
          %9 = vector.load %arg4[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
          %10 = vector.load %arg0[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
          %11 = arith.subf %9, %10 : vector<64xf32>
          vector.store %11, %arg2[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
        }
      }
    }
    return
  }
  func.func @morphgrad_2d_constant_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>, %arg6: memref<?x?xf32>, %arg7: memref<?x?xf32>, %arg8: index, %arg9: index, %arg10: index, %arg11: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    memref.copy %arg0, %arg5 : memref<?x?xf32> to memref<?x?xf32>
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg0, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg9 : index
    %1 = arith.addi %dim_2, %arg8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg12) = (0) to (symbol(%dim)) {
      affine.parallel (%arg13) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg14) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg15) = (0) to (symbol(%dim_4)) {
            %6 = arith.addi %arg12, %arg13 : index
            %7 = arith.addi %arg14, %arg15 : index
            %8 = arith.subi %6, %arg9 : index
            %9 = arith.subi %7, %arg8 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg9 : index
          }
        }
      }
    }
    affine.for %arg12 = 0 to %arg10 {
      %6 = arith.cmpi sge, %arg12, %c1 : index
      scf.if %6 {
        memref.copy %arg3, %arg0 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg6, %arg3 : memref<?x?xf32> to memref<?x?xf32>
    }
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg5, %c0_6 : memref<?x?xf32>
    %dim_9 = memref.dim %arg5, %c1_7 : memref<?x?xf32>
    %dim_10 = memref.dim %arg1, %c0_6 : memref<?x?xf32>
    %dim_11 = memref.dim %arg1, %c1_7 : memref<?x?xf32>
    %3 = arith.addi %dim_8, %arg9 : index
    %4 = arith.addi %dim_9, %arg8 : index
    %cst_12 = arith.constant 0.000000e+00 : f32
    %cst_13 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %5 = affine.apply #map(%dim_9, %dim_11, %c1_7)
    affine.parallel (%arg12) = (0) to (symbol(%dim_8)) {
      affine.parallel (%arg13) = (0) to (symbol(%dim_10)) {
        affine.parallel (%arg14) = (0) to (symbol(%dim_9)) step (64) {
          affine.parallel (%arg15) = (0) to (symbol(%dim_11)) {
            %6 = arith.addi %arg12, %arg13 : index
            %7 = arith.addi %arg14, %arg15 : index
            %8 = arith.subi %6, %arg9 : index
            %9 = arith.subi %7, %arg8 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg9 : index
          }
        }
      }
    }
    affine.for %arg12 = 0 to %arg10 {
      %6 = arith.cmpi sge, %arg12, %c1 : index
      scf.if %6 {
        memref.copy %arg4, %arg5 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg7, %arg4 : memref<?x?xf32> to memref<?x?xf32>
    }
    %dim_14 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %dim_15 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %cst_16 = arith.constant 0.000000e+00 : f32
    %cst_17 = arith.constant dense<0.000000e+00> : vector<64xf32>
    affine.for %arg12 = 0 to %dim_14 {
      affine.for %arg13 = 0 to %dim_15 step 64 {
        %6 = arith.addi %arg13, %c64 : index
        %7 = arith.subi %6, %c1 : index
        %8 = arith.cmpi sgt, %7, %dim_15 : index
        scf.if %8 {
          %9 = arith.subi %7, %dim_15 : index
          %10 = arith.subi %c64, %9 : index
          %11 = vector.create_mask %10 : vector<64xi1>
          %12 = vector.maskedload %arg3[%arg12, %arg13], %11, %cst_17 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %13 = vector.maskedload %arg4[%arg12, %arg13], %11, %cst_17 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %14 = arith.subf %12, %13 : vector<64xf32>
          vector.maskedstore %arg2[%arg12, %arg13], %11, %14 : memref<?x?xf32>, vector<64xi1>, vector<64xf32>
        } else {
          %9 = vector.load %arg3[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
          %10 = vector.load %arg4[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
          %11 = arith.subf %9, %10 : vector<64xf32>
          vector.store %11, %arg2[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
        }
      }
    }
    return
  }
  func.func @morphgrad_2d_replicate_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>, %arg6: memref<?x?xf32>, %arg7: memref<?x?xf32>, %arg8: index, %arg9: index, %arg10: index, %arg11: f32) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    memref.copy %arg0, %arg5 : memref<?x?xf32> to memref<?x?xf32>
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg0, %c1_1 : memref<?x?xf32>
    %dim_3 = memref.dim %arg1, %c0_0 : memref<?x?xf32>
    %dim_4 = memref.dim %arg1, %c1_1 : memref<?x?xf32>
    %0 = arith.addi %dim, %arg9 : index
    %1 = arith.addi %dim_2, %arg8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %2 = affine.apply #map(%dim_2, %dim_4, %c1_1)
    affine.parallel (%arg12) = (0) to (symbol(%dim)) {
      affine.parallel (%arg13) = (0) to (symbol(%dim_3)) {
        affine.parallel (%arg14) = (0) to (symbol(%dim_2)) step (64) {
          affine.parallel (%arg15) = (0) to (symbol(%dim_4)) {
            %6 = arith.addi %arg12, %arg13 : index
            %7 = arith.addi %arg14, %arg15 : index
            %8 = arith.subi %6, %arg9 : index
            %9 = arith.subi %7, %arg8 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg9 : index
          }
        }
      }
    }
    affine.for %arg12 = 0 to %arg10 {
      %6 = arith.cmpi sge, %arg12, %c1 : index
      scf.if %6 {
        memref.copy %arg3, %arg0 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg6, %arg3 : memref<?x?xf32> to memref<?x?xf32>
    }
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %dim_8 = memref.dim %arg5, %c0_6 : memref<?x?xf32>
    %dim_9 = memref.dim %arg5, %c1_7 : memref<?x?xf32>
    %dim_10 = memref.dim %arg1, %c0_6 : memref<?x?xf32>
    %dim_11 = memref.dim %arg1, %c1_7 : memref<?x?xf32>
    %3 = arith.addi %dim_8, %arg9 : index
    %4 = arith.addi %dim_9, %arg8 : index
    %cst_12 = arith.constant 0.000000e+00 : f32
    %cst_13 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %5 = affine.apply #map(%dim_9, %dim_11, %c1_7)
    affine.parallel (%arg12) = (0) to (symbol(%dim_8)) {
      affine.parallel (%arg13) = (0) to (symbol(%dim_10)) {
        affine.parallel (%arg14) = (0) to (symbol(%dim_9)) step (64) {
          affine.parallel (%arg15) = (0) to (symbol(%dim_11)) {
            %6 = arith.addi %arg12, %arg13 : index
            %7 = arith.addi %arg14, %arg15 : index
            %8 = arith.subi %6, %arg9 : index
            %9 = arith.subi %7, %arg8 : index
            %10 = arith.addi %7, %c64 : index
            %11 = arith.cmpi slt, %6, %arg9 : index
          }
        }
      }
    }
    affine.for %arg12 = 0 to %arg10 {
      %6 = arith.cmpi sge, %arg12, %c1 : index
      scf.if %6 {
        memref.copy %arg4, %arg5 : memref<?x?xf32> to memref<?x?xf32>
      }
      memref.copy %arg7, %arg4 : memref<?x?xf32> to memref<?x?xf32>
    }
    %dim_14 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %dim_15 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %cst_16 = arith.constant 0.000000e+00 : f32
    %cst_17 = arith.constant dense<0.000000e+00> : vector<64xf32>
    affine.for %arg12 = 0 to %dim_14 {
      affine.for %arg13 = 0 to %dim_15 step 64 {
        %6 = arith.addi %arg13, %c64 : index
        %7 = arith.subi %6, %c1 : index
        %8 = arith.cmpi sgt, %7, %dim_15 : index
        scf.if %8 {
          %9 = arith.subi %7, %dim_15 : index
          %10 = arith.subi %c64, %9 : index
          %11 = vector.create_mask %10 : vector<64xi1>
          %12 = vector.maskedload %arg3[%arg12, %arg13], %11, %cst_17 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %13 = vector.maskedload %arg4[%arg12, %arg13], %11, %cst_17 : memref<?x?xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %14 = arith.subf %12, %13 : vector<64xf32>
          vector.maskedstore %arg2[%arg12, %arg13], %11, %14 : memref<?x?xf32>, vector<64xi1>, vector<64xf32>
        } else {
          %9 = vector.load %arg3[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
          %10 = vector.load %arg4[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
          %11 = arith.subf %9, %10 : vector<64xf32>
          vector.store %11, %arg2[%arg12, %arg13] : memref<?x?xf32>, vector<64xf32>
        }
      }
    }
    return
  }
}

