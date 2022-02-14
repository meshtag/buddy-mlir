module attributes {llvm.data_layout = ""} {
  llvm.func @printNewline()
  llvm.func @printClose()
  llvm.func @printComma()
  llvm.func @printOpen()
  llvm.func @printF32(f32)
  llvm.mlir.global private @gv(dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01], [2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01], [3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01]]> : tensor<4x4xf32>) : !llvm.array<4 x array<4 x f32>>
  llvm.func @main() {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(dense<0.000000e+00> : vector<4x4xf32>) : !llvm.array<4 x vector<4xf32>>
    %3 = llvm.mlir.constant(4 : index) : i64
    %4 = llvm.mlir.constant(4 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(16 : index) : i64
    %7 = llvm.mlir.null : !llvm.ptr<f32>
    %8 = llvm.getelementptr %7[%6] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %9 = llvm.ptrtoint %8 : !llvm.ptr<f32> to i64
    %10 = llvm.mlir.addressof @gv : !llvm.ptr<array<4 x array<4 x f32>>>
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = llvm.getelementptr %10[%11, %11, %11] : (!llvm.ptr<array<4 x array<4 x f32>>>, i64, i64, i64) -> !llvm.ptr<f32>
    %13 = llvm.mlir.constant(3735928559 : index) : i64
    %14 = llvm.inttoptr %13 : i64 to !llvm.ptr<f32>
    %15 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %12, %16[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.insertvalue %3, %19[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.insertvalue %4, %20[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.insertvalue %4, %21[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.insertvalue %5, %22[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.mlir.constant(4 : index) : i64
    %25 = llvm.mul %1, %24  : i64
    %26 = llvm.add %25, %1  : i64
    %27 = llvm.getelementptr %12[%26] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %28 = llvm.load %27 : !llvm.ptr<f32>
    %29 = llvm.mlir.undef : vector<4xf32>
    %30 = llvm.mlir.constant(0 : i32) : i32
    %31 = llvm.insertelement %28, %29[%30 : i32] : vector<4xf32>
    %32 = llvm.shufflevector %31, %29 [0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<4xf32>, vector<4xf32>
    llvm.call @printOpen() : () -> ()
    %33 = llvm.mlir.constant(0 : index) : i64
    %34 = llvm.extractelement %32[%33 : i64] : vector<4xf32>
    llvm.call @printF32(%34) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %35 = llvm.mlir.constant(1 : index) : i64
    %36 = llvm.extractelement %32[%35 : i64] : vector<4xf32>
    llvm.call @printF32(%36) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %37 = llvm.mlir.constant(2 : index) : i64
    %38 = llvm.extractelement %32[%37 : i64] : vector<4xf32>
    llvm.call @printF32(%38) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %39 = llvm.mlir.constant(3 : index) : i64
    %40 = llvm.extractelement %32[%39 : i64] : vector<4xf32>
    llvm.call @printF32(%40) : (f32) -> ()
    llvm.call @printClose() : () -> ()
    llvm.call @printNewline() : () -> ()
    %41 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %42 = llvm.mlir.constant(4 : index) : i64
    %43 = llvm.mul %0, %42  : i64
    %44 = llvm.add %43, %0  : i64
    %45 = llvm.getelementptr %41[%44] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %46 = llvm.bitcast %45 : !llvm.ptr<f32> to !llvm.ptr<vector<4xf32>>
    %47 = llvm.load %46 {alignment = 4 : i64} : !llvm.ptr<vector<4xf32>>
    %48 = llvm.insertvalue %47, %2[0] : !llvm.array<4 x vector<4xf32>>
    %49 = llvm.insertvalue %47, %48[1] : !llvm.array<4 x vector<4xf32>>
    %50 = llvm.insertvalue %47, %49[2] : !llvm.array<4 x vector<4xf32>>
    %51 = llvm.insertvalue %47, %50[3] : !llvm.array<4 x vector<4xf32>>
    llvm.call @printOpen() : () -> ()
    llvm.call @printOpen() : () -> ()
    %52 = llvm.mlir.constant(0 : index) : i64
    %53 = llvm.extractelement %47[%52 : i64] : vector<4xf32>
    llvm.call @printF32(%53) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %54 = llvm.mlir.constant(1 : index) : i64
    %55 = llvm.extractelement %47[%54 : i64] : vector<4xf32>
    llvm.call @printF32(%55) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %56 = llvm.mlir.constant(2 : index) : i64
    %57 = llvm.extractelement %47[%56 : i64] : vector<4xf32>
    llvm.call @printF32(%57) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %58 = llvm.mlir.constant(3 : index) : i64
    %59 = llvm.extractelement %47[%58 : i64] : vector<4xf32>
    llvm.call @printF32(%59) : (f32) -> ()
    llvm.call @printClose() : () -> ()
    llvm.call @printComma() : () -> ()
    llvm.call @printOpen() : () -> ()
    %60 = llvm.mlir.constant(0 : index) : i64
    %61 = llvm.extractelement %47[%60 : i64] : vector<4xf32>
    llvm.call @printF32(%61) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %62 = llvm.mlir.constant(1 : index) : i64
    %63 = llvm.extractelement %47[%62 : i64] : vector<4xf32>
    llvm.call @printF32(%63) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %64 = llvm.mlir.constant(2 : index) : i64
    %65 = llvm.extractelement %47[%64 : i64] : vector<4xf32>
    llvm.call @printF32(%65) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %66 = llvm.mlir.constant(3 : index) : i64
    %67 = llvm.extractelement %47[%66 : i64] : vector<4xf32>
    llvm.call @printF32(%67) : (f32) -> ()
    llvm.call @printClose() : () -> ()
    llvm.call @printComma() : () -> ()
    llvm.call @printOpen() : () -> ()
    %68 = llvm.mlir.constant(0 : index) : i64
    %69 = llvm.extractelement %47[%68 : i64] : vector<4xf32>
    llvm.call @printF32(%69) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %70 = llvm.mlir.constant(1 : index) : i64
    %71 = llvm.extractelement %47[%70 : i64] : vector<4xf32>
    llvm.call @printF32(%71) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %72 = llvm.mlir.constant(2 : index) : i64
    %73 = llvm.extractelement %47[%72 : i64] : vector<4xf32>
    llvm.call @printF32(%73) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %74 = llvm.mlir.constant(3 : index) : i64
    %75 = llvm.extractelement %47[%74 : i64] : vector<4xf32>
    llvm.call @printF32(%75) : (f32) -> ()
    llvm.call @printClose() : () -> ()
    llvm.call @printComma() : () -> ()
    llvm.call @printOpen() : () -> ()
    %76 = llvm.mlir.constant(0 : index) : i64
    %77 = llvm.extractelement %47[%76 : i64] : vector<4xf32>
    llvm.call @printF32(%77) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %78 = llvm.mlir.constant(1 : index) : i64
    %79 = llvm.extractelement %47[%78 : i64] : vector<4xf32>
    llvm.call @printF32(%79) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %80 = llvm.mlir.constant(2 : index) : i64
    %81 = llvm.extractelement %47[%80 : i64] : vector<4xf32>
    llvm.call @printF32(%81) : (f32) -> ()
    llvm.call @printComma() : () -> ()
    %82 = llvm.mlir.constant(3 : index) : i64
    %83 = llvm.extractelement %47[%82 : i64] : vector<4xf32>
    llvm.call @printF32(%83) : (f32) -> ()
    llvm.call @printClose() : () -> ()
    llvm.call @printClose() : () -> ()
    llvm.call @printNewline() : () -> ()
    llvm.return
  }
}

