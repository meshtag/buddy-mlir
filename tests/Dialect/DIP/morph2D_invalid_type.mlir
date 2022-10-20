//
// x86
//
// RUN: buddy-opt %s -lower-dip="DIP-strip-mining=64" -arith-expand --convert-vector-to-scf --lower-affine --convert-scf-to-cf --convert-vector-to-llvm \
// RUN: --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts 2>&1 | FileCheck %s

memref.global "private" @global_input_f32 : memref<3x3xf32> = dense<[[0. , 1. , 2. ],
                                                                     [10., 11., 12.],
                                                                     [20., 21., 22.]]>

memref.global "private" @global_identity_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>

memref.global "private" @global_output_erosion_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>
memref.global "private" @global_output_dilation_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>

memref.global "private" @global_output_opening_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>
memref.global "private" @global_output_openinginter_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>

memref.global "private" @global_output_closing_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>
memref.global "private" @global_output_closinginter_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>

memref.global "private" @global_input_i32 : memref<3x3xi32> = dense<[[0 , 1 , 2 ],
                                                                     [10, 11, 12],
                                                                     [20, 21, 22]]>

memref.global "private" @global_identity_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                        [0, 1, 0],
                                                                        [0, 0, 0]]>

memref.global "private" @global_output_erosion_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>

memref.global "private" @global_output_dilation_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>
memref.global "private" @global_output_opening_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>

memref.global "private" @global_output_openinginter_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>
memref.global "private" @global_output_closing_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>

memref.global "private" @global_output_closinginter_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>

memref.global "private" @global_input_f128 : memref<3x3xf128> = dense<[[0. , 1. , 2. ],
                                                                       [10., 11., 12.],
                                                                       [20., 21., 22.]]>

memref.global "private" @global_output_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>
memref.global "private" @global_output_erosion_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>
memref.global "private" @global_output_dilation_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>
memref.global "private" @global_output_opening_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>
memref.global "private" @global_output_openinginter_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>
memref.global "private" @global_output_closing_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>
memref.global "private" @global_output_closinginter_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>

memref.global "private" @global_identity_f128 : memref<3x3xf128> = dense<[[0., 0., 0.],
                                                                          [0., 0., 0.],
                                                                          [0., 0., 0.]]>

memref.global "private" @global_copymemref1_f128 : memref<3x3xf128> = dense<[[-1., -1., -1.],
                                                                    [-1., -1., -1.],
                                                                    [-1., -1., -1.]]>

memref.global "private" @global_copymemref2_f128 : memref<3x3xf128> = dense<[[256., 256., 256.],
                                                                    [256., 256., 256.],
                                                                    [256., 256., 256.]]>

memref.global "private" @global_copymemref1_f32 : memref<3x3xf32> = dense<[[-1., -1., -1.],
                                                                    [-1., -1., -1.],
                                                                    [-1., -1., -1.]]>

memref.global "private" @global_copymemref2_f32 : memref<3x3xf32> = dense<[[256., 256., 256.],
                                                                    [256., 256., 256.],
                                                                    [256., 256., 256.]]>

memref.global "private" @global_copymemref1_i32 : memref<3x3xi32> = dense<[[-1, -1, -1],
                                                                    [-1, -1, -1],
                                                                    [-1, -1, -1]]>

memref.global "private" @global_copymemref2_i32 : memref<3x3xi32> = dense<[[256, 256, 256],
                                                                    [256, 256, 256],
                                                                    [256, 256, 256]]>

func.func @main() -> i32 {
  %input_f32 = memref.get_global @global_input_f32 : memref<3x3xf32>
  %identity_f32 = memref.get_global @global_identity_f32 : memref<3x3xf32>
  %output_f32erosion = memref.get_global @global_output_erosion_f32 : memref<3x3xf32>
  %output_f32dilation = memref.get_global @global_output_dilation_f32 : memref<3x3xf32>
  %output_f32opening = memref.get_global @global_output_opening_f32 : memref<3x3xf32>
  %output_f32opening1 = memref.get_global @global_output_openinginter_f32 : memref<3x3xf32>
  %output_f32closing = memref.get_global @global_output_closing_f32 : memref<3x3xf32>
  %output_f32closing1 = memref.get_global @global_output_closinginter_f32 : memref<3x3xf32>
  %copymemreff32_1 = memref.get_global @global_copymemref1_f32 : memref<3x3xf32>
  %copymemreff32_2 = memref.get_global @global_copymemref2_f32 : memref<3x3xf32>
  %c_f32 = arith.constant 0. : f32

  %input_i32 = memref.get_global @global_input_i32 : memref<3x3xi32>
  %identity_i32 = memref.get_global @global_identity_i32 : memref<3x3xi32>
  %output_i32erosion = memref.get_global @global_output_erosion_i32 : memref<3x3xi32>
  %output_i32dilation = memref.get_global @global_output_dilation_i32 : memref<3x3xi32>
  %output_i32opening = memref.get_global @global_output_opening_i32 : memref<3x3xi32>
  %output_i32opening1 = memref.get_global @global_output_openinginter_i32 : memref<3x3xi32>
  %output_i32closing = memref.get_global @global_output_closing_i32 : memref<3x3xi32>
  %output_i32closing1 = memref.get_global @global_output_closinginter_i32 : memref<3x3xi32>
  %copymemrefi32_1 = memref.get_global @global_copymemref1_i32 : memref<3x3xi32>
  %copymemrefi32_2 = memref.get_global @global_copymemref2_i32 : memref<3x3xi32>
  %c_i32 = arith.constant 0 : i32

  %input_f128 = memref.get_global @global_input_f128 : memref<3x3xf128>
  %identity_f128 = memref.get_global @global_identity_f128 : memref<3x3xf128>
  %output_f128erosion = memref.get_global @global_output_erosion_f128 : memref<3x3xf128>
  %output_f128dilation = memref.get_global @global_output_dilation_f128 : memref<3x3xf128>
  %output_f128opening = memref.get_global @global_output_opening_f128 : memref<3x3xf128>
  %output_f128opening1 = memref.get_global @global_output_openinginter_f128 : memref<3x3xf128>
  %output_f128closing = memref.get_global @global_output_closing_f128 : memref<3x3xf128>
  %output_f128closing1 = memref.get_global @global_output_closinginter_f128 : memref<3x3xf128>
  %copymemref128_1 = memref.get_global @global_copymemref1_f128 : memref<3x3xf128>
  %copymemreff128_2 = memref.get_global @global_copymemref2_f128 : memref<3x3xf128>
  %c_f128 = arith.constant 0. : f128

  %kernelAnchorX = arith.constant 1 : index
  %kernelAnchorY = arith.constant 1 : index
  %iterations = arith.constant 1 : index

  dip.erosion_2d <CONSTANT_PADDING> %input_i32, %identity_f32, %output_f32erosion, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32, %copymemreff32_2 : memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32, memref<3x3xf32>
  // CHECK: 'dip.erosion_2d' op input, kernel, output and constant must have the same element type

  dip.erosion_2d <CONSTANT_PADDING> %input_f32, %identity_i32, %output_f32erosion, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32, %copymemreff32_2 : memref<3x3xf32>, memref<3x3xi32>, memref<3x3xf32>, index, index, index, f32, memref<3x3xf32>
  // CHECK: 'dip.erosion_2d' op input, kernel, output and constant must have the same element type

  dip.erosion_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_i32erosion, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32, %copymemrefi32_2 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xi32>, index, index,index, f32, memref<3x3xi32>
  // CHECK: 'dip.erosion_2d' op input, kernel, output and constant must have the same element type

  dip.erosion_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_f32erosion, %kernelAnchorX, %kernelAnchorY, %iterations, %c_i32, %copymemreff32_2 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index,index, i32, memref<3x3xf32>
  // CHECK: 'dip.erosion_2d' op input, kernel, output and constant must have the same element type

  dip.erosion_2d <CONSTANT_PADDING> %input_f128, %identity_f128, %output_f128erosion, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f128, %copymemreff128_2 : memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, index, index,index, f128, memref<3x3xf128>
  // CHECK: 'dip.erosion_2d' op supports only f32, f64 and integer types. 'f128'is passed

  dip.dilation_2d <CONSTANT_PADDING> %input_i32, %identity_f32, %output_f32dilation, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32, %copymemreff32_1 : memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32, memref<3x3xf32>
  // CHECK: 'dip.dilation_2d' op input, kernel, output and constant must have the same element type

  dip.dilation_2d <CONSTANT_PADDING> %input_f32, %identity_i32, %output_f32dilation, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32, %copymemreff32_1 : memref<3x3xf32>, memref<3x3xi32>, memref<3x3xf32>, index, index, index, f32, memref<3x3xf32>
  // CHECK: 'dip.dilation_2d' op input, kernel, output and constant must have the same element type

  dip.dilation_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_i32dilation, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32, %copymemreff32_1 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xi32>, index, index,index, f32, memref<3x3xf32>
  // CHECK: 'dip.dilation_2d' op input, kernel, output and constant must have the same element type

  dip.dilation_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_f32dilation, %kernelAnchorX, %kernelAnchorY, %iterations, %c_i32, %copymemreff32_1 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index,index, i32, memref<3x3xf32>
  // CHECK: 'dip.dilation_2d' op input, kernel, output and constant must have the same element type

  dip.dilation_2d <CONSTANT_PADDING> %input_f128, %identity_f128, %output_f128dilation, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f128, %copymemref128_1 : memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, index, index,index, f128, memref<3x3xf128>
  // CHECK: 'dip.dilation_2d' op supports only f32, f64 and integer types. 'f128'is passed

 dip.opening_2d <CONSTANT_PADDING> %input_i32, %identity_f32, %output_f32opening, %output_f32opening1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32, %copymemreff32_2, %copymemreff32_1 : memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32, memref<3x3xf32>, memref<3x3xf32>
  // CHECK: 'dip.opening_2d' op input, kernel, output and constant must have the same element type

  dip.opening_2d <CONSTANT_PADDING> %input_f32, %identity_i32, %output_f32opening, %output_f32opening1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32, %copymemreff32_2, %copymemreff32_1 : memref<3x3xf32>, memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32, memref<3x3xf32>, memref<3x3xf32>
  // CHECK: 'dip.opening_2d' op input, kernel, output and constant must have the same element type

  dip.opening_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_i32opening, %output_i32opening1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32, %copymemrefi32_2, %copymemrefi32_1 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xi32>, memref<3x3xi32>, index, index,index, f32, memref<3x3xi32>, memref<3x3xi32>
  // CHECK: 'dip.opening_2d' op input, kernel, output and constant must have the same element type

  dip.opening_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_f32opening, %output_f32opening1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_i32, %copymemreff32_2, %copymemreff32_1 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index,index, i32, memref<3x3xf32>, memref<3x3xf32>
  // CHECK: 'dip.opening_2d' op input, kernel, output and constant must have the same element type

  dip.opening_2d <CONSTANT_PADDING> %input_f128, %identity_f128, %output_f128opening, %output_f128opening1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f128, %copymemreff128_2, %copymemref128_1 : memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, index, index,index, f128, memref<3x3xf128>, memref<3x3xf128>
  // CHECK: 'dip.opening_2d' op supports only f32, f64 and integer types. 'f128'is passed

  dip.closing_2d <CONSTANT_PADDING> %input_i32, %identity_f32, %output_f32closing, %output_f32closing1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32, %copymemreff32_1, %copymemreff32_2 : memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32, memref<3x3xf32>, memref<3x3xf32>
  // CHECK: 'dip.closing_2d' op input, kernel, output and constant must have the same element type

  dip.closing_2d <CONSTANT_PADDING> %input_f32, %identity_i32, %output_f32closing, %output_f32closing1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32, %copymemreff32_1, %copymemreff32_2 : memref<3x3xf32>, memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32, memref<3x3xf32>, memref<3x3xf32>
  // CHECK: 'dip.closing_2d' op input, kernel, output and constant must have the same element type

  dip.closing_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_i32closing, %output_i32closing1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32, %copymemrefi32_1, %copymemrefi32_2 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xi32>, memref<3x3xi32>, index, index,index, f32, memref<3x3xi32>, memref<3x3xi32>
  // CHECK: 'dip.closing_2d' op input, kernel, output and constant must have the same element type

  dip.closing_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_f32closing, %output_f32closing1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_i32, %copymemreff32_1, %copymemreff32_2 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index,index, i32, memref<3x3xf32>, memref<3x3xf32>
  // CHECK: 'dip.closing_2d' op input, kernel, output and constant must have the same element type

  dip.closing_2d <CONSTANT_PADDING> %input_f128, %identity_f128, %output_f128closing, %output_f128closing1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f128, %copymemref128_1, %copymemreff128_2 : memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, index, index,index, f128, memref<3x3xf128>, memref<3x3xf128>
  // CHECK: 'dip.closing_2d' op supports only f32, f64 and integer types. 'f128'is passed
  %ret = arith.constant 0 : i32
  return %ret : i32
}