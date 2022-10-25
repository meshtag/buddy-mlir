//
// x86
//
// RUN: buddy-opt %s -lower-dip="DIP-strip-mining=64" -arith-expand --convert-vector-to-scf --lower-affine --convert-scf-to-cf --convert-vector-to-llvm \
// RUN: --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts  \
// RUN: | mlir-cpu-runner -O0 -e main -entry-point-result=i32 \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @global_input : memref<3x3xf64> = dense<[[0. , 1. , 2. ],
                                                                 [10., 11., 12.],
                                                                 [20., 21., 22.]]>

memref.global "private" @global_identity : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                    [0., 1., 0.],
                                                                    [0., 0., 0.]]>

memref.global "private" @global_outputerosion : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>

memref.global "private" @global_outputdilation : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]> 

memref.global "private" @global_outputopening : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>   
memref.global "private" @global_outputopeninginter : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>  

memref.global "private" @global_outputclosing : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>   
memref.global "private" @global_outputclosinginter : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]> 
memref.global "private" @global_outputtophat : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>  
memref.global "private" @global_outputtophatinter : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputtophatinter1 : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>  

memref.global "private" @global_inputtophatinter : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputbottomhat : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>  
memref.global "private" @global_outputbottomhatinter : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputbottomhatinter1 : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>  

memref.global "private" @global_inputbottomhatinter : memref<3x3xf64> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>   

memref.global "private" @global_kernel : memref<3x3xf64> = dense<[[12., 22., 33.],
                                                                    [45., 44., 0.],
                                                                    [90., 11., 10.]]>    

memref.global "private" @global_kernel1 : memref<3x3xf64> = dense<[[0., 0., 11.],
                                                                    [4., 44., 10.],
                                                                    [9., 100., 10.]]>

memref.global "private" @global_kernel2 : memref<3x3xf64> = dense<[[1., 0., 0.],
                                                                    [0., 225., 0.],
                                                                    [0., 11., 10.]]>  

memref.global "private" @global_kernel3 : memref<3x3xf64> = dense<[[100., 0., 0.],
                                                                    [0., 0., 110.],
                                                                    [190., 0., 0.]]>   

memref.global "private" @global_copymemref1 : memref<3x3xf64> = dense<[[-1., -1., -1.],
                                                                    [-1., -1., -1.],
                                                                    [-1., -1., -1.]]> 

memref.global "private" @global_copymemref2 : memref<3x3xf64> = dense<[[256., 256., 256.],
                                                                    [256., 256., 256.],
                                                                    [256., 256., 256.]]>                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                             

func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }

func.func @main() -> i32 {
  %input = memref.get_global @global_input : memref<3x3xf64>
  %identity = memref.get_global @global_identity : memref<3x3xf64>
  %kernel = memref.get_global @global_kernel : memref<3x3xf64>
  %kernel1 = memref.get_global @global_kernel1 : memref<3x3xf64>
  %kernel2 = memref.get_global @global_kernel2 : memref<3x3xf64>
  %kernel3 = memref.get_global @global_kernel3 : memref<3x3xf64>
  %outputerosion = memref.get_global @global_outputerosion : memref<3x3xf64>
  %outputdilation = memref.get_global @global_outputdilation : memref<3x3xf64>
  %outputopening = memref.get_global @global_outputopening : memref<3x3xf64>
  %outputopening1 = memref.get_global @global_outputopeninginter : memref<3x3xf64>
  %outputclosing = memref.get_global @global_outputclosing : memref<3x3xf64>
  %outputclosing1 = memref.get_global @global_outputclosinginter : memref<3x3xf64>
  %outputtophat = memref.get_global @global_outputtophat : memref<3x3xf64>
  %outputtophat1 = memref.get_global @global_outputtophatinter : memref<3x3xf64>
  %outputtophat2 = memref.get_global @global_outputtophatinter1 : memref<3x3xf64>
  %inputtophat1 = memref.get_global @global_inputtophatinter : memref<3x3xf64>
  %outputbottomhat = memref.get_global @global_outputbottomhat : memref<3x3xf64>
  %outputbottomhat1 = memref.get_global @global_outputbottomhatinter : memref<3x3xf64>
  %outputbottomhat2 = memref.get_global @global_outputbottomhatinter1 : memref<3x3xf64>
  %inputbottomhat1 = memref.get_global @global_inputbottomhatinter : memref<3x3xf64>
  %copymemref1 = memref.get_global @global_copymemref1 : memref<3x3xf64>
  %copymemref2 = memref.get_global @global_copymemref2 : memref<3x3xf64>
  %kernelAnchorX = arith.constant 1 : index
  %kernelAnchorY = arith.constant 1 : index
  %iterations = arith.constant 1 : index
  %c = arith.constant 0. : f64 
  
  dip.erosion_2d <CONSTANT_PADDING> %input, %identity, %outputerosion, %copymemref2, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, index, index, index, f64
  dip.dilation_2d <CONSTANT_PADDING> %input, %kernel, %outputdilation, %copymemref1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, index, index, index, f64
  dip.opening_2d <REPLICATE_PADDING> %input, %kernel1, %outputopening, %outputopening1, %copymemref2, %copymemref1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, index, index, index, f64
  dip.closing_2d <CONSTANT_PADDING> %input, %kernel2, %outputclosing, %outputclosing1, %copymemref1, %copymemref2, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, index, index, index, f64
  dip.tophat_2d <REPLICATE_PADDING> %input, %kernel, %outputtophat, %outputtophat1,%outputtophat2, %inputtophat1, %copymemref2, %copymemref1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, index, index, index, f64
  dip.bottomhat_2d <CONSTANT_PADDING> %input, %kernel3, %outputbottomhat, %outputbottomhat1,%outputbottomhat2, %inputbottomhat1, %copymemref1, %copymemref2, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, memref<3x3xf64>, index, index, index, f64

  %printed_outpute = memref.cast %outputerosion : memref<3x3xf64> to memref<*xf64>
  %printed_outputd = memref.cast %outputdilation : memref<3x3xf64> to memref<*xf64>
  %printed_outputo = memref.cast %outputopening : memref<3x3xf64> to memref<*xf64>
  %printed_outputc = memref.cast %outputclosing : memref<3x3xf64> to memref<*xf64>
  %printed_outputt = memref.cast %outputtophat : memref<3x3xf64> to memref<*xf64>
  %printed_outputb = memref.cast %outputbottomhat : memref<3x3xf64> to memref<*xf64>
  call @printMemrefF64(%printed_outpute) : (memref<*xf64>) -> ()
  call @printMemrefF64(%printed_outputd) : (memref<*xf64>) -> ()
  call @printMemrefF64(%printed_outputo) : (memref<*xf64>) -> ()
  call @printMemrefF64(%printed_outputc) : (memref<*xf64>) -> ()
  call @printMemrefF64(%printed_outputt) : (memref<*xf64>) -> ()
  call @printMemrefF64(%printed_outputb) : (memref<*xf64>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[3, 3\] strides = \[3, 1\] data =}}
  // CHECK{LITERAL}: [[0, 1, 2],
  // CHECK{LITERAL}: [10, 11, 12],
  // CHECK{LITERAL}: [20, 21, 22]]

  %ret = arith.constant 0 : i32
  return %ret : i32
}
