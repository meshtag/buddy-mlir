#map0 = affine_map<(d0, d1, d2) -> (d0 + d1 - d2)>
#map1 = affine_map<(d0) -> (d0)>
module {
  func @corr_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
    %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %2 = memref.dim %arg1, %c0 : memref<?x?xf32>
    %3 = arith.addi %0, %arg4 : index
    %4 = arith.addi %1, %arg3 : index
    %cst = arith.constant 0.000000e+00 : f32
    %5 = vector.broadcast %cst : f32 to vector<6xf32>
    %6 = affine.apply #map0(%1, %2, %c1)
    affine.for %arg6 = #map1(%c0) to #map1(%0) {
      affine.for %arg7 = #map1(%c0) to #map1(%2) {
        affine.for %arg8 = #map1(%c0) to #map1(%1) step 6 {
          affine.for %arg9 = #map1(%c0) to #map1(%2) {
            %7 = arith.addi %arg6, %arg7 : index
            %8 = arith.addi %arg8, %arg9 : index
            %9 = memref.load %arg1[%arg7, %arg9] : memref<?x?xf32>
            %10 = vector.broadcast %9 : f32 to vector<6xf32>
            %11 = arith.subi %7, %arg4 : index
            %12 = arith.subi %8, %arg3 : index
            %13 = arith.addi %8, %c6 : index
            %14 = arith.cmpi slt, %7, %arg4 : index
            scf.if %14 {
              %15 = arith.cmpi slt, %8, %arg3 : index
              scf.if %15 {
                %16 = arith.subi %arg3, %8 : index
                %17 = vector.create_mask %16 : vector<6xi1>
                %18 = vector.create_mask %c6 : vector<6xi1>
                %19 = arith.subi %18, %17 : vector<6xi1>
                %20 = memref.load %arg0[%c0, %c0] : memref<?x?xf32>
                %21 = vector.broadcast %20 : f32 to vector<6xf32>
                %22 = arith.subi %c0, %16 : index
                %23 = vector.maskedload %arg0[%c0, %22], %19, %21 : memref<?x?xf32>, vector<6xi1>, vector<6xf32> into vector<6xf32>
                %24 = vector.load %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                %25 = vector.fma %23, %10, %24 : vector<6xf32>
                vector.store %25, %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
              } else {
                %16 = arith.cmpi sle, %13, %4 : index
                scf.if %16 {
                  %17 = vector.load %arg0[%c0, %12] : memref<?x?xf32>, vector<6xf32>
                  %18 = vector.load %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                  %19 = vector.fma %17, %10, %18 : vector<6xf32>
                  vector.store %19, %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                } else {
                  %17 = arith.subi %13, %4 : index
                  %18 = arith.subi %c6, %17 : index
                  %19 = vector.create_mask %18 : vector<6xi1>
                  %20 = arith.subi %1, %c1 : index
                  %21 = memref.load %arg0[%c0, %20] : memref<?x?xf32>
                  %22 = vector.broadcast %21 : f32 to vector<6xf32>
                  %23 = vector.maskedload %arg0[%c0, %12], %19, %22 : memref<?x?xf32>, vector<6xi1>, vector<6xf32> into vector<6xf32>
                  %24 = affine.apply #map0(%c6, %2, %c1)
                  %25 = arith.subi %6, %arg8 : index
                  %26 = arith.cmpi sge, %25, %24 : index
                  scf.if %26 {
                    %27 = vector.load %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                    %28 = vector.fma %23, %10, %27 : vector<6xf32>
                    vector.store %28, %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                  } else {
                    %27 = arith.subi %1, %arg8 : index
                    %28 = vector.create_mask %27 : vector<6xi1>
                    %29 = vector.maskedload %arg2[%arg6, %arg8], %28, %5 : memref<?x?xf32>, vector<6xi1>, vector<6xf32> into vector<6xf32>
                    %30 = vector.fma %23, %10, %29 : vector<6xf32>
                    vector.maskedstore %arg2[%arg6, %arg8], %28, %30 : memref<?x?xf32>, vector<6xi1>, vector<6xf32>
                  }
                }
              }
            } else {
              %15 = arith.cmpi slt, %7, %3 : index
              scf.if %15 {
                %16 = arith.cmpi slt, %8, %arg3 : index
                scf.if %16 {
                  %17 = arith.subi %arg3, %8 : index
                  %18 = vector.create_mask %17 : vector<6xi1>
                  %19 = vector.create_mask %c6 : vector<6xi1>
                  %20 = arith.subi %19, %18 : vector<6xi1>
                  %21 = memref.load %arg0[%11, %c0] : memref<?x?xf32>
                  %22 = vector.broadcast %21 : f32 to vector<6xf32>
                  %23 = arith.subi %c0, %17 : index
                  %24 = vector.maskedload %arg0[%11, %23], %20, %22 : memref<?x?xf32>, vector<6xi1>, vector<6xf32> into vector<6xf32>
                  %25 = vector.load %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                  %26 = vector.fma %24, %10, %25 : vector<6xf32>
                  vector.store %26, %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                } else {
                  %17 = arith.cmpi sle, %13, %4 : index
                  scf.if %17 {
                    %18 = vector.load %arg0[%11, %12] : memref<?x?xf32>, vector<6xf32>
                    %19 = vector.load %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                    %20 = vector.fma %18, %10, %19 : vector<6xf32>
                    vector.store %20, %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                  } else {
                    %18 = arith.subi %13, %4 : index
                    %19 = arith.subi %c6, %18 : index
                    %20 = vector.create_mask %19 : vector<6xi1>
                    %21 = arith.subi %1, %c1 : index
                    %22 = memref.load %arg0[%11, %21] : memref<?x?xf32>
                    %23 = vector.broadcast %22 : f32 to vector<6xf32>
                    %24 = vector.maskedload %arg0[%11, %12], %20, %23 : memref<?x?xf32>, vector<6xi1>, vector<6xf32> into vector<6xf32>
                    %25 = affine.apply #map0(%c6, %2, %c1)
                    %26 = arith.subi %6, %arg8 : index
                    %27 = arith.cmpi sge, %26, %25 : index
                    scf.if %27 {
                      %28 = vector.load %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                      %29 = vector.fma %24, %10, %28 : vector<6xf32>
                      vector.store %29, %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                    } else {
                      %28 = arith.subi %1, %arg8 : index
                      %29 = vector.create_mask %28 : vector<6xi1>
                      %30 = vector.maskedload %arg2[%arg6, %arg8], %29, %5 : memref<?x?xf32>, vector<6xi1>, vector<6xf32> into vector<6xf32>
                      %31 = vector.fma %24, %10, %30 : vector<6xf32>
                      vector.maskedstore %arg2[%arg6, %arg8], %29, %31 : memref<?x?xf32>, vector<6xi1>, vector<6xf32>
                    }
                  }
                }
              } else {
                %16 = arith.cmpi slt, %8, %arg3 : index
                scf.if %16 {
                  %17 = arith.subi %0, %c1 : index
                  %18 = arith.subi %arg3, %8 : index
                  %19 = vector.create_mask %18 : vector<6xi1>
                  %20 = vector.create_mask %c6 : vector<6xi1>
                  %21 = arith.subi %20, %19 : vector<6xi1>
                  %22 = memref.load %arg0[%17, %c0] : memref<?x?xf32>
                  %23 = vector.broadcast %22 : f32 to vector<6xf32>
                  %24 = arith.subi %c0, %18 : index
                  %25 = vector.maskedload %arg0[%17, %24], %21, %23 : memref<?x?xf32>, vector<6xi1>, vector<6xf32> into vector<6xf32>
                  %26 = vector.load %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                  %27 = vector.fma %25, %10, %26 : vector<6xf32>
                  vector.store %27, %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                } else {
                  %17 = arith.cmpi sle, %13, %4 : index
                  scf.if %17 {
                    %18 = arith.subi %0, %c1 : index
                    %19 = vector.load %arg0[%18, %12] : memref<?x?xf32>, vector<6xf32>
                    %20 = vector.load %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                    %21 = vector.fma %19, %10, %20 : vector<6xf32>
                    vector.store %21, %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                  } else {
                    %18 = arith.subi %13, %4 : index
                    %19 = arith.subi %c6, %18 : index
                    %20 = vector.create_mask %19 : vector<6xi1>
                    %21 = arith.subi %0, %c1 : index
                    %22 = arith.subi %1, %c1 : index
                    %23 = memref.load %arg0[%21, %22] : memref<?x?xf32>
                    %24 = vector.broadcast %23 : f32 to vector<6xf32>
                    %25 = vector.maskedload %arg0[%21, %12], %20, %24 : memref<?x?xf32>, vector<6xi1>, vector<6xf32> into vector<6xf32>
                    %26 = affine.apply #map0(%c6, %2, %c1)
                    %27 = arith.subi %6, %arg8 : index
                    %28 = arith.cmpi sge, %27, %26 : index
                    scf.if %28 {
                      %29 = vector.load %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                      %30 = vector.fma %25, %10, %29 : vector<6xf32>
                      vector.store %30, %arg2[%arg6, %arg8] : memref<?x?xf32>, vector<6xf32>
                    } else {
                      %29 = arith.subi %1, %arg8 : index
                      %30 = vector.create_mask %29 : vector<6xi1>
                      %31 = vector.maskedload %arg2[%arg6, %arg8], %30, %5 : memref<?x?xf32>, vector<6xi1>, vector<6xf32> into vector<6xf32>
                      %32 = vector.fma %25, %10, %31 : vector<6xf32>
                      vector.maskedstore %arg2[%arg6, %arg8], %30, %32 : memref<?x?xf32>, vector<6xi1>, vector<6xf32>
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
  func @rotate_2d(%arg0: memref<?x?xf32>, %arg1: index, %arg2: memref<?x?xf32>) {
    %c6 = arith.constant 6 : index
    %cst = arith.constant 3.140000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
    %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %2 = arith.index_cast %0 : index to i32
    %3 = arith.sitofp %2 : i32 to f32
    %4 = vector.splat %3 : vector<6xf32>
    %5 = arith.index_cast %1 : index to i32
    %6 = arith.sitofp %5 : i32 to f32
    %7 = vector.splat %6 : vector<6xf32>
    %8 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %9 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %10 = arith.index_cast %0 : index to i32
    %11 = arith.sitofp %10 : i32 to f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %12 = arith.addf %11, %cst_0 : f32
    %13 = arith.divf %12, %cst_1 : f32
    %14 = arith.subf %13, %cst_0 : f32
    %15 = math.ceil %14 : f32
    %16 = math.floor %14 : f32
    %17 = arith.subf %15, %14 : f32
    %18 = arith.subf %14, %16 : f32
    %19 = arith.cmpf ogt, %17, %18 : f32
    %20 = arith.select %19, %16, %15 : f32
    %21 = arith.fptoui %20 : f32 to i32
    %22 = arith.index_cast %21 : i32 to index
    %23 = arith.index_cast %1 : index to i32
    %24 = arith.sitofp %23 : i32 to f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 2.000000e+00 : f32
    %25 = arith.addf %24, %cst_2 : f32
    %26 = arith.divf %25, %cst_3 : f32
    %27 = arith.subf %26, %cst_2 : f32
    %28 = math.ceil %27 : f32
    %29 = math.floor %27 : f32
    %30 = arith.subf %28, %27 : f32
    %31 = arith.subf %27, %29 : f32
    %32 = arith.cmpf ogt, %30, %31 : f32
    %33 = arith.select %32, %29, %28 : f32
    %34 = arith.fptoui %33 : f32 to i32
    %35 = arith.index_cast %34 : i32 to index
    %36 = arith.index_cast %22 : index to i32
    %37 = arith.sitofp %36 : i32 to f32
    %38 = vector.splat %37 : vector<6xf32>
    %39 = arith.index_cast %35 : index to i32
    %40 = arith.sitofp %39 : i32 to f32
    %41 = vector.splat %40 : vector<6xf32>
    %42 = arith.index_cast %8 : index to i32
    %43 = arith.sitofp %42 : i32 to f32
    %cst_4 = arith.constant 1.000000e+00 : f32
    %cst_5 = arith.constant 2.000000e+00 : f32
    %44 = arith.addf %43, %cst_4 : f32
    %45 = arith.divf %44, %cst_5 : f32
    %46 = arith.subf %45, %cst_4 : f32
    %47 = math.ceil %46 : f32
    %48 = math.floor %46 : f32
    %49 = arith.subf %47, %46 : f32
    %50 = arith.subf %46, %48 : f32
    %51 = arith.cmpf ogt, %49, %50 : f32
    %52 = arith.select %51, %48, %47 : f32
    %53 = arith.fptoui %52 : f32 to i32
    %54 = arith.index_cast %53 : i32 to index
    %55 = arith.index_cast %9 : index to i32
    %56 = arith.sitofp %55 : i32 to f32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 2.000000e+00 : f32
    %57 = arith.addf %56, %cst_6 : f32
    %58 = arith.divf %57, %cst_7 : f32
    %59 = arith.subf %58, %cst_6 : f32
    %60 = math.ceil %59 : f32
    %61 = math.floor %59 : f32
    %62 = arith.subf %60, %59 : f32
    %63 = arith.subf %59, %61 : f32
    %64 = arith.cmpf ogt, %62, %63 : f32
    %65 = arith.select %64, %61, %60 : f32
    %66 = arith.fptoui %65 : f32 to i32
    %67 = arith.index_cast %66 : i32 to index
    %68 = arith.index_cast %54 : index to i32
    %69 = arith.sitofp %68 : i32 to f32
    %70 = vector.splat %69 : vector<6xf32>
    %71 = arith.index_cast %67 : index to i32
    %72 = arith.sitofp %71 : i32 to f32
    %73 = vector.splat %72 : vector<6xf32>
    %cst_8 = arith.constant 1.000000e+00 : f32
    %74 = vector.splat %cst_8 : vector<6xf32>
    %75 = math.sin %cst : f32
    %76 = vector.broadcast %75 : f32 to vector<6xf32>
    %cst_9 = arith.constant 2.000000e+00 : f32
    %77 = arith.divf %cst, %cst_9 : f32
    %78 = math.sin %77 : f32
    %79 = math.cos %77 : f32
    %80 = arith.divf %78, %79 : f32
    %81 = vector.broadcast %80 : f32 to vector<6xf32>
    affine.for %arg3 = #map1(%c0) to #map1(%0) {
      affine.for %arg4 = #map1(%c0) to #map1(%1) step 6 {
        %82 = arith.divui %arg4, %c6 : index
        %83 = arith.muli %c6, %82 : index
        %84 = arith.index_cast %arg3 : index to i32
        %85 = arith.sitofp %84 : i32 to f32
        %86 = vector.splat %85 : vector<6xf32>
        %c0_10 = arith.constant 0 : index
        %cst_11 = arith.constant 0.000000e+00 : f32
        %87 = vector.splat %cst_11 : vector<6xf32>
        %c1_12 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %88 = arith.addi %83, %c1_12 : index
        %89 = arith.addi %83, %c2 : index
        %90 = arith.addi %83, %c3 : index
        %91 = arith.addi %83, %c4 : index
        %92 = arith.addi %83, %c5 : index
        %93 = arith.index_cast %83 : index to i32
        %94 = arith.sitofp %93 : i32 to f32
        %95 = arith.index_cast %88 : index to i32
        %96 = arith.sitofp %95 : i32 to f32
        %97 = arith.index_cast %89 : index to i32
        %98 = arith.sitofp %97 : i32 to f32
        %99 = arith.index_cast %90 : index to i32
        %100 = arith.sitofp %99 : i32 to f32
        %101 = arith.index_cast %91 : index to i32
        %102 = arith.sitofp %101 : i32 to f32
        %103 = arith.index_cast %92 : index to i32
        %104 = arith.sitofp %103 : i32 to f32
        %105 = vector.insertelement %94, %87[%c0_10 : index] : vector<6xf32>
        %106 = vector.insertelement %96, %105[%c1_12 : index] : vector<6xf32>
        %107 = vector.insertelement %98, %106[%c2 : index] : vector<6xf32>
        %108 = vector.insertelement %100, %107[%c3 : index] : vector<6xf32>
        %109 = vector.insertelement %102, %108[%c4 : index] : vector<6xf32>
        %110 = vector.insertelement %104, %109[%c5 : index] : vector<6xf32>
        %111 = vector.splat %cst_11 : vector<6xf32>
        %112 = affine.for %arg5 = #map1(%c0_10) to #map1(%c6) iter_args(%arg6 = %111) -> (vector<6xf32>) {
          %145 = arith.index_cast %arg5 : index to i32
          %146 = arith.sitofp %145 : i32 to f32
          %147 = vector.insertelement %146, %arg6[%arg5 : index] : vector<6xf32>
          // vector.print %111 : vector<6xf32>
          // vector.print %147 : vector<6xf32>
          // vector.print %arg6 : vector<6xf32>
          affine.yield %147 : vector<6xf32>
        }

        // vector.print %112 : vector<6xf32>

        // vector.print %111 : vector<6xf32>
        %113 = arith.subf %4, %86 : vector<6xf32>
        %114 = arith.subf %113, %38 : vector<6xf32>
        %115 = arith.subf %114, %74 : vector<6xf32>
        %116 = arith.subf %7, %110 : vector<6xf32>
        %117 = arith.subf %116, %41 : vector<6xf32>
        %118 = arith.subf %117, %74 : vector<6xf32>
        %119 = arith.mulf %81, %115 : vector<6xf32>
        %120 = arith.subf %118, %119 : vector<6xf32>
        %121 = math.ceil %120 : vector<6xf32>
        %122 = math.floor %120 : vector<6xf32>
        %123 = arith.subf %121, %120 : vector<6xf32>
        %124 = arith.subf %120, %122 : vector<6xf32>
        %125 = arith.cmpf ogt, %123, %124 : vector<6xf32>
        %126 = arith.select %125, %122, %121 : vector<6xi1>, vector<6xf32>
        %127 = arith.mulf %126, %76 : vector<6xf32>
        %128 = arith.addf %127, %115 : vector<6xf32>
        %129 = math.ceil %128 : vector<6xf32>
        %130 = math.floor %128 : vector<6xf32>
        %131 = arith.subf %129, %128 : vector<6xf32>
        %132 = arith.subf %128, %130 : vector<6xf32>
        %133 = arith.cmpf ogt, %131, %132 : vector<6xf32>
        %134 = arith.select %133, %130, %129 : vector<6xi1>, vector<6xf32>
        %135 = arith.mulf %134, %81 : vector<6xf32>
        %136 = arith.subf %126, %135 : vector<6xf32>
        %137 = math.ceil %136 : vector<6xf32>
        %138 = math.floor %136 : vector<6xf32>
        %139 = arith.subf %137, %136 : vector<6xf32>
        %140 = arith.subf %136, %138 : vector<6xf32>
        %141 = arith.cmpf ogt, %139, %140 : vector<6xf32>
        %142 = arith.select %141, %138, %137 : vector<6xi1>, vector<6xf32>
        %143 = arith.subf %70, %134 : vector<6xf32>
        %144 = arith.subf %73, %142 : vector<6xf32>
        affine.for %arg5 = 0 to 6 {
          %145 = vector.extractelement %144[%arg5 : index] : vector<6xf32>
          %146 = vector.extractelement %143[%arg5 : index] : vector<6xf32>
          %147 = arith.fptoui %145 : f32 to i32
          %148 = arith.index_cast %147 : i32 to index
          %149 = arith.fptoui %146 : f32 to i32
          %150 = arith.index_cast %149 : i32 to index
          %151 = vector.extractelement %110[%arg5 : index] : vector<6xf32>
          %152 = vector.extractelement %86[%arg5 : index] : vector<6xf32>
          %153 = arith.fptoui %151 : f32 to i32
          %154 = arith.index_cast %153 : i32 to index
          %155 = arith.fptoui %152 : f32 to i32
          %156 = arith.index_cast %155 : i32 to index
          %157 = memref.load %arg0[%154, %156] : memref<?x?xf32>
          memref.store %157, %arg2[%148, %150] : memref<?x?xf32>
        }
      }
    }
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
  func @changeImage(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = memref.alloc(%arg1, %arg2) : memref<?x?xf32>
    scf.for %arg3 = %c1 to %arg1 step %c2 {
      scf.for %arg4 = %c0 to %arg2 step %c1 {
        %2 = memref.load %arg0[%arg3, %arg4] : memref<?x?xf32>
        memref.store %2, %0[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    scf.for %arg3 = %c0 to %arg1 step %c2 {
      scf.for %arg4 = %c0 to %arg2 step %c1 {
        %cst_0 = arith.constant 1.000000e+00 : f32
        %2 = memref.load %arg0[%arg3, %arg4] : memref<?x?xf32>
        %3 = arith.addf %2, %cst_0 : f32
        memref.store %3, %0[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    %1 = arith.subi %arg1, %c1 : index
    %cst = arith.constant 7.000000e+00 : f32
    scf.for %arg3 = %c0 to %arg2 step %c1 {
      %2 = memref.load %arg0[%arg3, %1] : memref<?x?xf32>
      memref.store %cst, %0[%arg3, %1] : memref<?x?xf32>
    }
    return %0 : memref<?x?xf32>
  }
  func @DIPCorr2D(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) -> memref<?x?xf32> {
    %c19 = arith.constant 19 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = call @alloc_2d_filled_f32(%c19, %c19, %cst) : (index, index, f32) -> memref<?x?xf32>
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c90 = arith.constant 90 : index
    %c6 = arith.constant 6 : index
    %cst_1 = arith.constant 3.140000e+00 : f32
    %c0_2 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = memref.dim %arg0, %c0_2 : memref<?x?xf32>
    %2 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %3 = arith.index_cast %1 : index to i32
    %4 = arith.sitofp %3 : i32 to f32
    %5 = vector.splat %4 : vector<6xf32>
    %6 = arith.index_cast %2 : index to i32
    %7 = arith.sitofp %6 : i32 to f32
    %8 = vector.splat %7 : vector<6xf32>
    %9 = memref.dim %0, %c0_2 : memref<?x?xf32>
    %10 = memref.dim %0, %c1 : memref<?x?xf32>
    %11 = arith.index_cast %1 : index to i32
    %12 = arith.sitofp %11 : i32 to f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant 2.000000e+00 : f32
    %13 = arith.addf %12, %cst_3 : f32
    %14 = arith.divf %13, %cst_4 : f32
    %15 = arith.subf %14, %cst_3 : f32
    %16 = math.ceil %15 : f32
    %17 = math.floor %15 : f32
    %18 = arith.subf %16, %15 : f32
    %19 = arith.subf %15, %17 : f32
    %20 = arith.cmpf ogt, %18, %19 : f32
    %21 = arith.select %20, %17, %16 : f32
    %22 = arith.fptoui %21 : f32 to i32
    %23 = arith.index_cast %22 : i32 to index
    %24 = arith.index_cast %2 : index to i32
    %25 = arith.sitofp %24 : i32 to f32
    %cst_5 = arith.constant 1.000000e+00 : f32
    %cst_6 = arith.constant 2.000000e+00 : f32
    %26 = arith.addf %25, %cst_5 : f32
    %27 = arith.divf %26, %cst_6 : f32
    %28 = arith.subf %27, %cst_5 : f32
    %29 = math.ceil %28 : f32
    %30 = math.floor %28 : f32
    %31 = arith.subf %29, %28 : f32
    %32 = arith.subf %28, %30 : f32
    %33 = arith.cmpf ogt, %31, %32 : f32
    %34 = arith.select %33, %30, %29 : f32
    %35 = arith.fptoui %34 : f32 to i32
    %36 = arith.index_cast %35 : i32 to index
    %37 = arith.index_cast %23 : index to i32
    %38 = arith.sitofp %37 : i32 to f32
    %39 = vector.splat %38 : vector<6xf32>
    %40 = arith.index_cast %36 : index to i32
    %41 = arith.sitofp %40 : i32 to f32
    %42 = vector.splat %41 : vector<6xf32>
    %43 = arith.index_cast %9 : index to i32
    %44 = arith.sitofp %43 : i32 to f32
    %cst_7 = arith.constant 1.000000e+00 : f32
    %cst_8 = arith.constant 2.000000e+00 : f32
    %45 = arith.addf %44, %cst_7 : f32
    %46 = arith.divf %45, %cst_8 : f32
    %47 = arith.subf %46, %cst_7 : f32
    %48 = math.ceil %47 : f32
    %49 = math.floor %47 : f32
    %50 = arith.subf %48, %47 : f32
    %51 = arith.subf %47, %49 : f32
    %52 = arith.cmpf ogt, %50, %51 : f32
    %53 = arith.select %52, %49, %48 : f32
    %54 = arith.fptoui %53 : f32 to i32
    %55 = arith.index_cast %54 : i32 to index
    %56 = arith.index_cast %10 : index to i32
    %57 = arith.sitofp %56 : i32 to f32
    %cst_9 = arith.constant 1.000000e+00 : f32
    %cst_10 = arith.constant 2.000000e+00 : f32
    %58 = arith.addf %57, %cst_9 : f32
    %59 = arith.divf %58, %cst_10 : f32
    %60 = arith.subf %59, %cst_9 : f32
    %61 = math.ceil %60 : f32
    %62 = math.floor %60 : f32
    %63 = arith.subf %61, %60 : f32
    %64 = arith.subf %60, %62 : f32
    %65 = arith.cmpf ogt, %63, %64 : f32
    %66 = arith.select %65, %62, %61 : f32
    %67 = arith.fptoui %66 : f32 to i32
    %68 = arith.index_cast %67 : i32 to index
    %69 = arith.index_cast %55 : index to i32
    %70 = arith.sitofp %69 : i32 to f32
    %71 = vector.splat %70 : vector<6xf32>
    %72 = arith.index_cast %68 : index to i32
    %73 = arith.sitofp %72 : i32 to f32
    %74 = vector.splat %73 : vector<6xf32>
    %cst_11 = arith.constant 1.000000e+00 : f32
    %75 = vector.splat %cst_11 : vector<6xf32>
    %76 = math.sin %cst_1 : f32
    %77 = vector.broadcast %76 : f32 to vector<6xf32>
    %cst_12 = arith.constant 2.000000e+00 : f32
    %78 = arith.divf %cst_1, %cst_12 : f32
    %79 = math.sin %78 : f32
    %80 = math.cos %78 : f32
    %81 = arith.divf %79, %80 : f32
    %82 = vector.broadcast %81 : f32 to vector<6xf32>
    affine.for %arg2 = #map1(%c0_2) to #map1(%1) {
      affine.for %arg3 = #map1(%c0_2) to #map1(%2) step 6 {
        %84 = arith.divui %arg3, %c6 : index
        %85 = arith.muli %c6, %84 : index
        %86 = arith.index_cast %arg2 : index to i32
        %87 = arith.sitofp %86 : i32 to f32
        %88 = vector.splat %87 : vector<6xf32>
        %c0_13 = arith.constant 0 : index
        %cst_14 = arith.constant 0.000000e+00 : f32
        %89 = vector.splat %cst_14 : vector<6xf32>
        %c1_15 = arith.constant 1 : index
        %c2_16 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %90 = arith.addi %85, %c1_15 : index
        %91 = arith.addi %85, %c2_16 : index
        %92 = arith.addi %85, %c3 : index
        %93 = arith.addi %85, %c4 : index
        %94 = arith.addi %85, %c5 : index
        %95 = arith.index_cast %85 : index to i32
        %96 = arith.sitofp %95 : i32 to f32
        %97 = arith.index_cast %90 : index to i32
        %98 = arith.sitofp %97 : i32 to f32
        %99 = arith.index_cast %91 : index to i32
        %100 = arith.sitofp %99 : i32 to f32
        %101 = arith.index_cast %92 : index to i32
        %102 = arith.sitofp %101 : i32 to f32
        %103 = arith.index_cast %93 : index to i32
        %104 = arith.sitofp %103 : i32 to f32
        %105 = arith.index_cast %94 : index to i32
        %106 = arith.sitofp %105 : i32 to f32
        %107 = vector.insertelement %96, %89[%c0_13 : index] : vector<6xf32>
        %108 = vector.insertelement %98, %107[%c1_15 : index] : vector<6xf32>
        %109 = vector.insertelement %100, %108[%c2_16 : index] : vector<6xf32>
        %110 = vector.insertelement %102, %109[%c3 : index] : vector<6xf32>
        %111 = vector.insertelement %104, %110[%c4 : index] : vector<6xf32>
        %112 = vector.insertelement %106, %111[%c5 : index] : vector<6xf32>
        %113 = vector.splat %cst_14 : vector<6xf32>
        %114 = affine.for %arg4 = #map1(%c0_13) to #map1(%c6) iter_args(%arg5 = %113) -> (vector<6xf32>) {
          %147 = arith.index_cast %arg4 : index to i32
          %148 = arith.sitofp %147 : i32 to f32
          %149 = vector.insertelement %148, %arg5[%arg4 : index] : vector<6xf32>
          vector.print %113 : vector<6xf32>
          vector.print %149 : vector<6xf32>
          vector.print %arg5 : vector<6xf32>
          affine.yield %149 : vector<6xf32>
        }

        vector.print %114 : vector<6xf32>

        // vector.print %113 : vector<6xf32>
        %115 = arith.subf %5, %88 : vector<6xf32>
        %116 = arith.subf %115, %39 : vector<6xf32>
        %117 = arith.subf %116, %75 : vector<6xf32>
        %118 = arith.subf %8, %112 : vector<6xf32>
        %119 = arith.subf %118, %42 : vector<6xf32>
        %120 = arith.subf %119, %75 : vector<6xf32>
        %121 = arith.mulf %82, %117 : vector<6xf32>
        %122 = arith.subf %120, %121 : vector<6xf32>
        %123 = math.ceil %122 : vector<6xf32>
        %124 = math.floor %122 : vector<6xf32>
        %125 = arith.subf %123, %122 : vector<6xf32>
        %126 = arith.subf %122, %124 : vector<6xf32>
        %127 = arith.cmpf ogt, %125, %126 : vector<6xf32>
        %128 = arith.select %127, %124, %123 : vector<6xi1>, vector<6xf32>
        %129 = arith.mulf %128, %77 : vector<6xf32>
        %130 = arith.addf %129, %117 : vector<6xf32>
        %131 = math.ceil %130 : vector<6xf32>
        %132 = math.floor %130 : vector<6xf32>
        %133 = arith.subf %131, %130 : vector<6xf32>
        %134 = arith.subf %130, %132 : vector<6xf32>
        %135 = arith.cmpf ogt, %133, %134 : vector<6xf32>
        %136 = arith.select %135, %132, %131 : vector<6xi1>, vector<6xf32>
        %137 = arith.mulf %136, %82 : vector<6xf32>
        %138 = arith.subf %128, %137 : vector<6xf32>
        %139 = math.ceil %138 : vector<6xf32>
        %140 = math.floor %138 : vector<6xf32>
        %141 = arith.subf %139, %138 : vector<6xf32>
        %142 = arith.subf %138, %140 : vector<6xf32>
        %143 = arith.cmpf ogt, %141, %142 : vector<6xf32>
        %144 = arith.select %143, %140, %139 : vector<6xi1>, vector<6xf32>
        %145 = arith.subf %71, %136 : vector<6xf32>
        %146 = arith.subf %74, %144 : vector<6xf32>
        affine.for %arg4 = 0 to 6 {
          %147 = vector.extractelement %146[%arg4 : index] : vector<6xf32>
          %148 = vector.extractelement %145[%arg4 : index] : vector<6xf32>
          %149 = arith.fptoui %147 : f32 to i32
          %150 = arith.index_cast %149 : i32 to index
          %151 = arith.fptoui %148 : f32 to i32
          %152 = arith.index_cast %151 : i32 to index
          %153 = vector.extractelement %112[%arg4 : index] : vector<6xf32>
          %154 = vector.extractelement %88[%arg4 : index] : vector<6xf32>
          %155 = arith.fptoui %153 : f32 to i32
          %156 = arith.index_cast %155 : i32 to index
          %157 = arith.fptoui %154 : f32 to i32
          %158 = arith.index_cast %157 : i32 to index
          %159 = memref.load %arg0[%156, %158] : memref<?x?xf32>
          memref.store %159, %0[%150, %152] : memref<?x?xf32>
        }
      }
    }
    %83 = memref.cast %0 : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%83) : (memref<*xf32>) -> ()
    return %0 : memref<?x?xf32>
  }
  func @main() {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c18 = arith.constant 18 : index
    %c3 = arith.constant 3 : index
    %0 = call @alloc_2d_filled_f32(%c18, %c18, %cst) : (index, index, f32) -> memref<?x?xf32>
    %1 = call @alloc_2d_filled_f32(%c3, %c3, %cst_0) : (index, index, f32) -> memref<?x?xf32>
    %2 = call @changeImage(%0, %c18, %c18) : (memref<?x?xf32>, index, index) -> memref<?x?xf32>
    %3 = call @DIPCorr2D(%2, %1) : (memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
    %4 = memref.cast %2 : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%4) : (memref<*xf32>) -> ()
    %5 = memref.cast %1 : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
    return
  }
}
