//
// x86
//
// RUN: buddy-opt %s -lower-dip="DIP-strip-mining=64" -arith-expand --convert-vector-to-scf --lower-affine --convert-scf-to-cf --convert-vector-to-llvm \
// RUN: --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts  \
// RUN: | mlir-cpu-runner -O0 -e main -entry-point-result=i32 \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @global_input : memref<3x3xf32> = dense<[[0. , 1. , 2. ],
                                                                 [10., 11., 12.],
                                                                 [20., 21., 22.]]>

memref.global "private" @global_identity : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                    [0., 1., 0.],
                                                                    [0., 0., 0.]]>

memref.global "private" @global_outputerosion : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputdilation : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>   
memref.global "private" @global_outputopening : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>  
memref.global "private" @global_outputopeninginter : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>  
memref.global "private" @global_outputclosing : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>  
memref.global "private" @global_outputclosinginter : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>   
memref.global "private" @global_outputtophat : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>  
memref.global "private" @global_outputtophatinter : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputtophatinter1 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>  

memref.global "private" @global_inputtophatinter : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputbottomhat : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>  
memref.global "private" @global_outputbottomhatinter : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputbottomhatinter1 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>  

memref.global "private" @global_inputbottomhatinter : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>                                                                  



func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @main() -> i32 {
  %input = memref.get_global @global_input : memref<3x3xf32>
  %identity = memref.get_global @global_identity : memref<3x3xf32>
  %outpute = memref.get_global @global_outputerosion : memref<3x3xf32>
  %outputd = memref.get_global @global_outputerosion : memref<3x3xf32>
  %outputopening = memref.get_global @global_outputopening : memref<3x3xf32>
  %outputopening1 = memref.get_global @global_outputopeninginter : memref<3x3xf32>
  %outputclosing = memref.get_global @global_outputclosing : memref<3x3xf32>
  %outputclosing1 = memref.get_global @global_outputclosinginter : memref<3x3xf32>
  %outputtophat = memref.get_global @global_outputtophat : memref<3x3xf32>
  %outputtophat1 = memref.get_global @global_outputtophatinter : memref<3x3xf32>
  %outputtophat2 = memref.get_global @global_outputtophatinter1 : memref<3x3xf32>
  %inputtophat1 = memref.get_global @global_inputtophatinter : memref<3x3xf32>
  %outputbottomhat = memref.get_global @global_outputbottomhat : memref<3x3xf32>
  %outputbottomhat1 = memref.get_global @global_outputbottomhatinter : memref<3x3xf32>
  %outputbottomhat2 = memref.get_global @global_outputbottomhatinter1 : memref<3x3xf32>
  %inputbottomhat1 = memref.get_global @global_inputbottomhatinter : memref<3x3xf32>

  %kernelAnchorX = arith.constant 1 : index
  %kernelAnchorY = arith.constant 1 : index
  %iterations = arith.constant 1 : index
  %c = arith.constant 0. : f32 
  dip.erosion_2d <CONSTANT_PADDING> %input, %identity, %outpute, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  dip.dilation_2d <REPLICATE_PADDING> %input, %identity, %outputd, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  dip.opening_2d <CONSTANT_PADDING> %input, %identity, %outputopening, %outputopening1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  dip.closing_2d <REPLICATE_PADDING> %input, %identity, %outputclosing, %outputclosing1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  dip.tophat_2d <REPLICATE_PADDING> %input, %identity, %outputtophat, %outputtophat1,%outputtophat2, %inputtophat1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>,memref<3x3xf32>,memref<3x3xf32>,memref<3x3xf32>, index, index, index, f32
  dip.bottomhat_2d <CONSTANT_PADDING> %input, %identity, %outputbottomhat, %outputbottomhat1,%outputbottomhat2, %inputbottomhat1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>,memref<3x3xf32>,memref<3x3xf32>,memref<3x3xf32>, index, index, index, f32
  %printed_outpute = memref.cast %outpute : memref<3x3xf32> to memref<*xf32>
  %printed_outputd = memref.cast %outputd : memref<3x3xf32> to memref<*xf32>
  %printed_outputo = memref.cast %outputopening : memref<3x3xf32> to memref<*xf32>
  %printed_outputc = memref.cast %outputclosing : memref<3x3xf32> to memref<*xf32>
  %printed_outputt = memref.cast %outputtophat : memref<3x3xf32> to memref<*xf32>
  %printed_outputb = memref.cast %outputbottomhat : memref<3x3xf32> to memref<*xf32>
  call @printMemrefF32(%printed_outpute) : (memref<*xf32>) -> ()
  call @printMemrefF32(%printed_outputd) : (memref<*xf32>) -> ()
  call @printMemrefF32(%printed_outputo) : (memref<*xf32>) -> ()
  call @printMemrefF32(%printed_outputc) : (memref<*xf32>) -> () 
  call @printMemrefF32(%printed_outputt) : (memref<*xf32>) -> ()
  call @printMemrefF32(%printed_outputb) : (memref<*xf32>) -> () 
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[3, 3\] strides = \[3, 1\] data =}}
  // CHECK{LITERAL}: [[0, 1, 2],
  // CHECK{LITERAL}: [10, 11, 12],
  // CHECK{LITERAL}: [20, 21, 22]]

  %ret = arith.constant 0 : i32
  return %ret : i32
}
