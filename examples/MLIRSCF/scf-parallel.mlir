func.func @main() {
  %c0 = arith.constant 10.0 : f32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %c0_vec = vector.splat %c0 : vector<4xf32>

  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index

  %A = memref.alloc() : memref<8x8xf32>
  %U = memref.cast %A :  memref<8x8xf32> to memref<*xf32>

  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c4) {
    // memref.store %c0, %A[%i, %j] : memref<8x8xf32>
    vector.store %c0_vec, %A[%i, %j] : memref<8x8xf32>, vector<4xf32>
  }

  // scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c1) {
  //   %0 = arith.muli %i, %c8 : index
  //   %1 = arith.addi %j, %0  : index
  //   %2 = arith.index_cast %1 : index to i32
  //   %3 = arith.sitofp %2 : i32 to f32
  //   %8 = memref.load %A[%i, %j] : memref<8x8xf32>
  //   %5 = arith.addf %3, %8 : f32



  //   memref.store %5, %A[%i, %j] : memref<8x8xf32>
  // }

  vector.print %lb : index

  // %print_output = memref.cast %output : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%U) : (memref<*xf32>) -> ()

  memref.dealloc %A : memref<8x8xf32>

  return
}

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
