//====- LowerdipPass.cpp - dip Dialect Lowering Pass  ---------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines dip dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"

#include "dip/dipDialect.h"
#include "dip/dipOps.h"

using namespace mlir;
using namespace Buddy;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

// Calculate result of FMA and store it in output memref
void calcAndStoreFMA(OpBuilder &builder, Location loc, VectorType vecType,
                     Value inputVec, Value kernelVec, Value output,
                     Value index1, Value index2) {
//   builder.create<scf::IfOp>(
//       loc, tailCond,
//       [&](OpBuilder &builder, Location loc) {
//         Value outputVec =
//             builder.create<LoadOp>(loc, vecType, output, ValueRange{index1, index2});
//         Value resVec =
//             builder.create<FMAOp>(loc, inputVec, kernelVec, outputVec);
//         builder.create<StoreOp>(loc, resVec, output, ValueRange{index1, index2});

//         builder.create<scf::YieldOp>(loc);
//       },
//       [&](OpBuilder &builder, Location loc) {
//         Value outputVec =
//             builder.create<MaskedLoadOp>(loc, vecType, output, 
//             ValueRange{index1, index2}, extraElemMask, zeroPadding);
//         Value resVec =
//             builder.create<FMAOp>(loc, inputVec, kernelVec, outputVec);
//         builder.create<MaskedStoreOp>(loc, output, ValueRange{index1, index2},
//             extraElemMask, resVec);

//         builder.create<scf::YieldOp>(loc);
//       });


                Value outputVec =
            builder.create<LoadOp>(loc, vecType, output, ValueRange{index1, index2});
        Value resVec =
            builder.create<FMAOp>(loc, inputVec, kernelVec, outputVec);
        builder.create<StoreOp>(loc, resVec, output, ValueRange{index1, index2});
}

void calcAndStoreFMATail(OpBuilder &builder, Location loc, VectorType vecType,
                     Value inputVec, Value kernelVec, Value output,
                     Value index1, Value index2,
                     Value extraElemMask, Value zeroPadding) {
          Value outputVec =
            builder.create<MaskedLoadOp>(loc, vecType, output, 
            ValueRange{index1, index2}, extraElemMask, zeroPadding);
        Value resVec =
            builder.create<FMAOp>(loc, inputVec, kernelVec, outputVec);
        builder.create<MaskedStoreOp>(loc, output, ValueRange{index1, index2},
            extraElemMask, resVec); 
    }


// Create an inverted mask having all 1's shifted to right side
Value createInvertedMask(OpBuilder &builder, Location loc, Value strideVal,
                         VectorType vectorMask, Value leftIndex) {
  Value leftMask = builder.create<CreateMaskOp>(loc, vectorMask, leftIndex);
  Value maskInverter = builder.create<CreateMaskOp>(loc, vectorMask, strideVal);
  Value rightMask = builder.create<SubIOp>(loc, maskInverter, leftMask);
  return rightMask;
}

class dipCorr2DLowering : public OpRewritePattern<dip::Corr2DOp> {
public:
  using OpRewritePattern<dip::Corr2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(dip::Corr2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Register operand values
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value centerX = op->getOperand(3);
    Value centerY = op->getOperand(4);
    // Value boundaryOptionVal = op->getOperand(5);
    unsigned int boundaryOption = 1;

    unsigned int stride = 3;
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    FloatType f32 = FloatType::getF32(ctx);
    IntegerType i1 = IntegerType::get(ctx, 1);

    // Create DimOp.
    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);
    Value kernelSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    // Variables used for detecting rowMid, rowDown, colMid and colRight regions
    Value rowMidHelper = rewriter.create<AddIOp>(loc, inputRow, centerY);
    Value colMidHelper = rewriter.create<AddIOp>(loc, inputCol, centerX);

    SmallVector<Value, 8> lowerBounds(4, c0);
    SmallVector<Value, 8> uperBounds{inputRow, kernelSize, inputCol,
                                     kernelSize};
    SmallVector<int64_t, 8> steps{1, 1, stride, 1};

    VectorType vectorTy32 = VectorType::get({stride}, f32);
    VectorType vectorMask = VectorType::get({stride}, i1);

    // Improve this flow for constant padding option
    Value zeroPaddingElem =
        rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)0, f32);
    Value zeroPadding =
        rewriter.create<BroadcastOp>(loc, vectorTy32, zeroPaddingElem);

    AffineExpr a, b, c;
    bindDims(ctx, a, b, c);
    AffineMap calcHelper = AffineMap::get(3, 0, {a + b - c}, ctx);

    Value pseudoCol = rewriter.create<AffineApplyOp>(
        loc, calcHelper, ValueRange{inputCol, kernelSize, c1});

    buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        //   Value tailChecker = builder.create<AffineApplyOp>(
        //       loc, calcHelper, ValueRange{strideVal, kernelSize, c1});
        //   Value colEndDistance = builder.create<SubIOp>(loc, pseudoCol, ivs[2]);
        //   Value tailCond = rewriter.create<CmpIOp>(loc, CmpIPredicate::sge,
        //                                            colEndDistance, tailChecker);

        //   Value extraElemCount = builder.create<SubIOp>(loc, inputCol, ivs[2]);
        //   Value extraElemMask = builder.create<CreateMaskOp>(loc, vectorMask, extraElemCount);

          // Indices of current pixel with respect to pseudo image containing
          // extrapolated boundaries
          Value currRow = builder.create<AddIOp>(loc, ivs[0], ivs[1]);
          Value currCol = builder.create<AddIOp>(loc, ivs[2], ivs[3]);

          Value kernelValue = builder.create<memref::LoadOp>(
              loc, kernel, ValueRange{ivs[1], ivs[3]});
          Value kernelVec = builder.create<BroadcastOp>(loc, vectorTy32, kernelValue);

          // Pixel indices with respect to the actual image
          Value imRow = builder.create<SubIOp>(loc, currRow, centerY);
          Value imCol = builder.create<SubIOp>(loc, currCol, centerX);

          // Index of pixel used for determining right region
          Value colLastElem = builder.create<AddIOp>(loc, currCol, strideVal);

          Value rowUpCond =
              builder.create<CmpIOp>(loc, CmpIPredicate::slt, currRow, centerY);

          builder.create<scf::IfOp>(
              loc, rowUpCond,
              [&](OpBuilder &builder, Location loc) {
                // rowUp
                if (!boundaryOption) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               zeroPaddingElem);

                  calcAndStoreFMA(builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);
                } else {
                  Value colLeftCond = builder.create<CmpIOp>(
                      loc, CmpIPredicate::slt, currCol, centerX);

                  builder.create<scf::IfOp>(
                      loc, colLeftCond,
                      [&](OpBuilder &builder, Location loc) {
                        // colLeft & rowUp
                        Value inputVec;
                        Value leftMaskElem =
                            builder.create<SubIOp>(loc, centerX, currCol);
                        Value leftMask = createInvertedMask(
                            builder, loc, strideVal, vectorMask, leftMaskElem);

                        if (boundaryOption == 1) {
                          Value paddingVal = builder.create<memref::LoadOp>(
                              loc, input, ValueRange{c0, c0});
                          Value padding = builder.create<BroadcastOp>(
                              loc, vectorTy32, paddingVal);

                          Value c11 =
                              builder.create<SubIOp>(loc, c0, leftMaskElem);
                          inputVec = builder.create<vector::MaskedLoadOp>(
                              loc, vectorTy32, input, ValueRange{c0, c11},
                              leftMask, padding);
                        }
                        calcAndStoreFMA(builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2]);

                        builder.create<scf::YieldOp>(loc);
                      },
                      [&](OpBuilder &builder, Location loc) {
                        // (colMid or colRight) & rowUp
                        Value colMidCond = builder.create<CmpIOp>(
                            loc, CmpIPredicate::sle, colLastElem, colMidHelper);

                        builder.create<scf::IfOp>(
                            loc, colMidCond,
                            [&](OpBuilder &builder, Location loc) {
                              // colMid & rowUp
                              Value inputVec;
                              if (boundaryOption == 1) {
                                inputVec = builder.create<LoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol});
                              }
                              calcAndStoreFMA(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // colRight & rowUp

                              Value tailChecker = builder.create<AffineApplyOp>(
              loc, calcHelper, ValueRange{strideVal, kernelSize, c1});
          Value colEndDistance = builder.create<SubIOp>(loc, pseudoCol, ivs[2]);
          Value tailCond = rewriter.create<CmpIOp>(loc, CmpIPredicate::sge,
                                                   colEndDistance, tailChecker);

          Value extraElemCount = builder.create<SubIOp>(loc, inputCol, ivs[2]);
          Value extraElemMask = builder.create<CreateMaskOp>(loc, vectorMask, extraElemCount);

                              Value inputVec;
                              Value rightMaskHelper = builder.create<SubIOp>(
                                  loc, colLastElem, colMidHelper);
                              Value rightMaskElem = builder.create<SubIOp>(
                                  loc, kernelSize, rightMaskHelper);
                              Value rightMask = builder.create<CreateMaskOp>(
                                  loc, vectorMask, rightMaskElem);

                              if (boundaryOption == 1) {
                                Value rightRange =
                                    builder.create<SubIOp>(loc, inputCol, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{c0, rightRange});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{c0, imCol}, rightMask, padding);
                              }

                              builder.create<scf::IfOp>(loc, tailCond, 
                                  [&](OpBuilder &builder, Location loc){
                                      calcAndStoreFMA(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);

                                      builder.create<scf::YieldOp>(loc);
                                  }, [&](OpBuilder &builder, Location loc){
                                      calcAndStoreFMATail(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2], extraElemMask, zeroPadding);

                                      builder.create<scf::YieldOp>(loc);
                                  });

                              builder.create<scf::YieldOp>(loc);
                            });
                        builder.create<scf::YieldOp>(loc);
                      });
                }
                builder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &builder, Location loc) {
                // rowMid or rowDown
                Value rowMidCond = builder.create<CmpIOp>(
                    loc, CmpIPredicate::slt, currRow, rowMidHelper);

                builder.create<scf::IfOp>(
                    loc, rowMidCond,
                    [&](OpBuilder &builder, Location loc) {
                      // rowMid
                      Value colLeftCond = builder.create<CmpIOp>(
                          loc, CmpIPredicate::slt, currCol, centerX);

                      builder.create<scf::IfOp>(
                          loc, colLeftCond,
                          [&](OpBuilder &builder, Location loc) {
                            // colLeft & rowMid
                            Value inputVec;
                            Value leftMaskElem =
                                builder.create<SubIOp>(loc, centerX, currCol);
                            Value leftMask =
                                createInvertedMask(builder, loc, strideVal,
                                                   vectorMask, leftMaskElem);

                            if (!boundaryOption) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, zeroPaddingElem);

                              Value c11 =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, c11}, leftMask, padding);
                            } else if (boundaryOption == 1) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, input, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value c11 =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, c11}, leftMask, padding);
                            }
                            calcAndStoreFMA(builder, loc, vectorTy32, inputVec,
                                            kernelVec, output, ivs[0], ivs[2]);

                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // (colMid or colRight) & rowMid
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, CmpIPredicate::sle, colLastElem,
                                colMidHelper);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec = builder.create<LoadOp>(
                                      loc, vectorTy32, input,
                                      ValueRange{imRow, imCol});
                                  calcAndStoreFMA(builder, loc, vectorTy32,
                                                  inputVec, kernelVec, output,
                                                  ivs[0], ivs[2]);

                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid

                                  Value tailChecker = builder.create<AffineApplyOp>(
              loc, calcHelper, ValueRange{strideVal, kernelSize, c1});
          Value colEndDistance = builder.create<SubIOp>(loc, pseudoCol, ivs[2]);
          Value tailCond = rewriter.create<CmpIOp>(loc, CmpIPredicate::sge,
                                                   colEndDistance, tailChecker);

          Value extraElemCount = builder.create<SubIOp>(loc, inputCol, ivs[2]);
          Value extraElemMask = builder.create<CreateMaskOp>(loc, vectorMask, extraElemCount);

                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, kernelSize, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMask, rightMaskElem);

                                  if (!boundaryOption) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, zeroPaddingElem);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOption == 1) {
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);
                                    Value paddingVal =
                                        builder.create<memref::LoadOp>(
                                            loc, input,
                                            ValueRange{imRow, rightRange});
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, paddingVal);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  }
                                  builder.create<scf::IfOp>(loc, tailCond, 
                                  [&](OpBuilder &builder, Location loc){
                                      calcAndStoreFMA(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);

                                      builder.create<scf::YieldOp>(loc);
                                  }, [&](OpBuilder &builder, Location loc){
                                      calcAndStoreFMATail(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2], extraElemMask, zeroPadding);

                                      builder.create<scf::YieldOp>(loc);
                                  });

                                  builder.create<scf::YieldOp>(loc);
                                });
                            builder.create<scf::YieldOp>(loc);
                          });
                      builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &builder, Location loc) {
                      // rowDown
                      if (!boundaryOption) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, zeroPaddingElem);

                        calcAndStoreFMA(builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2]);
                      } else {
                        Value colLeftCond = builder.create<CmpIOp>(
                            loc, CmpIPredicate::slt, currCol, centerX);

                        builder.create<scf::IfOp>(
                            loc, colLeftCond,
                            [&](OpBuilder &builder, Location loc) {
                              // colLeft & rowDown
                              Value inputVec;
                              Value downRange =
                                  builder.create<SubIOp>(loc, inputRow, c1);
                              Value leftMaskElem =
                                  builder.create<SubIOp>(loc, centerX, currCol);
                              Value leftMask =
                                  createInvertedMask(builder, loc, strideVal,
                                                     vectorMask, leftMaskElem);

                              if (boundaryOption == 1) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value c11 = builder.create<SubIOp>(
                                    loc, c0, leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{downRange, c11}, leftMask,
                                    padding);
                              }
                              calcAndStoreFMA(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // (colMid or colRight) & rowDown
                              Value colMidCond = builder.create<CmpIOp>(
                                  loc, CmpIPredicate::sle, colLastElem,
                                  colMidHelper);

                              builder.create<scf::IfOp>(
                                  loc, colMidCond,
                                  [&](OpBuilder &builder, Location loc) {
                                    // colMid & rowDown
                                    Value inputVec;
                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    if (boundaryOption == 1) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol});
                                    } else if (boundaryOption == 2) {
                                      Value refRowHelper =
                                          builder.create<SubIOp>(loc, currRow,
                                                                 rowMidHelper);
                                      Value refRow = builder.create<SubIOp>(
                                          loc, downRange, refRowHelper);

                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{refRow, imCol});
                                    }
                                    calcAndStoreFMA(builder, loc, vectorTy32,
                                                    inputVec, kernelVec, output,
                                                    ivs[0], ivs[2]);

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown

                                    Value tailChecker = builder.create<AffineApplyOp>(
              loc, calcHelper, ValueRange{strideVal, kernelSize, c1});
          Value colEndDistance = builder.create<SubIOp>(loc, pseudoCol, ivs[2]);
          Value tailCond = rewriter.create<CmpIOp>(loc, CmpIPredicate::sge,
                                                   colEndDistance, tailChecker);

          Value extraElemCount = builder.create<SubIOp>(loc, inputCol, ivs[2]);
          Value extraElemMask = builder.create<CreateMaskOp>(loc, vectorMask, extraElemCount);

                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, kernelSize,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMask, rightMaskElem);

                                    Value downRange = builder.create<SubIOp>(
                                        loc, inputRow, c1);
                                    Value rightRange = builder.create<SubIOp>(
                                        loc, inputCol, c1);

                                    if (boundaryOption == 1) {
                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, input,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec = builder.create<MaskedLoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol},
                                          rightMask, padding);
                                    }
                                    builder.create<scf::IfOp>(loc, tailCond, 
                                  [&](OpBuilder &builder, Location loc){
                                      calcAndStoreFMA(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);

                                      builder.create<scf::YieldOp>(loc);
                                  }, [&](OpBuilder &builder, Location loc){
                                      calcAndStoreFMATail(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2], extraElemMask, zeroPadding);

                                      builder.create<scf::YieldOp>(loc);
                                  });

                                    builder.create<scf::YieldOp>(loc);
                                  });
                              builder.create<scf::YieldOp>(loc);
                            });
                      }
                      builder.create<scf::YieldOp>(loc);
                    });
                builder.create<scf::YieldOp>(loc);
              });
        });
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }
};
} // end anonymous namespace

void populateLowerdipConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<dipCorr2DLowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// LowerdipPass
//===----------------------------------------------------------------------===//

namespace {
class LowerdipPass : public PassWrapper<LowerdipPass, OperationPass<ModuleOp>> {
public:
  LowerdipPass() = default;
  LowerdipPass(const LowerdipPass &) {}

  StringRef getArgument() const final { return "lower-dip"; }
  StringRef getDescription() const final { return "Lower dip Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Buddy::dip::dipDialect, StandardOpsDialect,
                    memref::MemRefDialect, scf::SCFDialect, VectorDialect,
                    AffineDialect>();
  }
};
} // end anonymous namespace.

void LowerdipPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, scf::SCFDialect, StandardOpsDialect,
                         memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ReturnOp>();

  RewritePatternSet patterns(context);
  populateLowerdipConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace Buddy {
void registerLowerdipPass() { PassRegistration<LowerdipPass>(); }
} // namespace Buddy
} // namespace mlir
