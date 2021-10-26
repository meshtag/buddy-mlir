//====- LowerDIPPass.cpp - DIP Dialect Lowering Pass  ---------------------===//
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
// This file defines DIP dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"

#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include <numeric>

#include <iostream>

using namespace mlir;
using namespace Buddy;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

void calcAndStoreFMA(OpBuilder &builder, Location loc, VectorType vecType,
                     Value inputVec, Value kernelVec, Value output,
                     ValueRange indices) {
  Value outputVec =
      builder.create<vector::LoadOp>(loc, vecType, output, indices);
  Value resVec =
      builder.create<vector::FMAOp>(loc, inputVec, kernelVec, outputVec);
  builder.create<vector::StoreOp>(loc, resVec, output, indices);
}

unsigned int calcValue(OpBuilder &builder, Location loc, Value numVal)
{
  Value c0 = builder.create<ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<ConstantIndexOp>(loc, 1);
  unsigned int val = 0;

  // Value whileCond = builder.create<CmpIOp>(
  //                     loc, mlir::CmpIPredicate::slt, builder.create<ConstantIndexOp>(loc, val),
  //                     numVal);

  // builder.create<scf::WhileOp>(loc, whileCond, [&](unsigned int val){
  //   // ++val;
  // });

  buildAffineLoopNest(
        builder, loc, {c0}, {numVal}, {1},
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          ++val;
        // builder.create<AffineYieldOp>(loc);
  });
  return val;
}

class DIPCorr2DLowering : public OpRewritePattern<DIP::Corr2DOp> {
public:
  using OpRewritePattern<DIP::Corr2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DIP::Corr2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value centerX = op->getOperand(3);
    Value centerY = op->getOperand(4);

    // Value boundaryOption = op->getOperand(5);
    unsigned int boundaryOption = 0;
    unsigned int stride = 3;
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);
    FloatType f32 = mlir::FloatType::getF32(ctx);
    IntegerType i1 = mlir::IntegerType::get(ctx, 1);

    // Improve this flow for constant padding option
    Value constantPadding =
        rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)0, f32);

    // Create DimOp.
    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);
    Value kernelSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    Value rowMidHelper =
        rewriter.create<AddIOp>(loc, inputRow, centerY);
    Value colMidHelper =
        rewriter.create<AddIOp>(loc, inputCol, centerX);

    SmallVector<Value, 8> lowerBounds(4, c0);
    SmallVector<Value, 8> uperBounds{inputRow, kernelSize, inputCol,
                                     kernelSize};
    SmallVector<int64_t, 8> steps{1, 1, stride, 1};

    VectorType vectorTy1 = mlir::VectorType::get({1}, f32);
    VectorType vectorTy32 = mlir::VectorType::get({stride}, f32);
    VectorType vectorMask = mlir::VectorType::get({stride}, i1);

    buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Broadcast element of the kernel.
          Value kernelValue = builder.create<LoadOp>(
              loc, vectorTy1, kernel, ValueRange{ivs[1], ivs[3]});
          Value kernelVec =
              builder.create<BroadcastOp>(loc, vectorTy32, kernelValue);

          Value currRow = builder.create<AddIOp>(loc, ivs[0], ivs[1]);
          Value currCol = builder.create<AddIOp>(loc, ivs[2], ivs[3]);

          Value imRow = builder.create<SubIOp>(loc, currRow, centerY);
          Value imCol = builder.create<SubIOp>(loc, currCol, centerX);

          Value colLastElem = builder.create<AddIOp>(loc, currCol, strideVal);

          Value rowUpCond = builder.create<CmpIOp>(
              loc, mlir::CmpIPredicate::slt, currRow, centerY);

          builder.create<scf::IfOp>(
              loc, rowUpCond,
              [&](OpBuilder &builder, Location loc) {
                // rowUp
                if (!boundaryOption) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantPadding);

                  calcAndStoreFMA(builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ValueRange{ivs[0], ivs[2]});
                } else {
                  Value colLeftCond = builder.create<CmpIOp>(
                      loc, mlir::CmpIPredicate::slt, currCol, centerX);

                  builder.create<scf::IfOp>(
                      loc, colLeftCond,
                      [&](OpBuilder &builder, Location loc) {
                        // colLeft & rowUp

                        builder.create<scf::YieldOp>(loc);
                      },
                      [&](OpBuilder &builder, Location loc) {
                        // colMid or colRight
                        Value colMidCond = builder.create<CmpIOp>(
                            loc, mlir::CmpIPredicate::sle, colLastElem,
                            colMidHelper);

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
                              else if (boundaryOption == 2) {
                                Value refRowHelper = builder.create<SubIOp>(loc, 
                                    centerY, currRow);
                                Value refRow = builder.create<SubIOp>(loc, refRowHelper, c1);

                                inputVec = builder.create<LoadOp>(loc, vectorTy32, input, 
                                    ValueRange{refRow, imCol});
                              }

                              calcAndStoreFMA(builder, loc, vectorTy32,
                                                inputVec, kernelVec, output,
                                                ValueRange{ivs[0], ivs[2]});

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // colRight & rowUp
                              Value inputVec;
                              Value rightMaskHelper =
                                  builder.create<mlir::SubIOp>(loc, colLastElem,
                                                               colMidHelper);
                              Value rightMaskElem =
                                  builder.create<mlir::SubIOp>(loc, kernelSize,
                                                               rightMaskHelper);
                              Value rightMask =
                                  builder.create<vector::CreateMaskOp>(
                                      loc, vectorMask, rightMaskElem);

                              if (boundaryOption == 1) {
                                Value rightRange = builder.create<mlir::SubIOp>(
                                    loc, inputCol, c1);
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{c0, rightRange});
                                Value padding =
                                    builder.create<vector::BroadcastOp>(
                                        loc, vectorTy32, paddingVal);

                                inputVec =
                                    builder.create<vector::MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{currRow, imCol}, rightMask, // currRow seems doubtful
                                        padding);
                              }
                              else if (boundaryOption == 2) {
                                Value refRowHelper = builder.create<SubIOp>(loc, 
                                    centerY, currRow);
                                Value refRow = builder.create<SubIOp>(loc, refRowHelper, c1);

                                Value refVecElem = 
                                    builder.create<SubIOp>(loc, inputCol, strideVal);

                                Value paddingHelper = 
                                    builder.create<LoadOp>(loc, vectorTy32, input,
                                    ValueRange{refRow, refVecElem});

                                Value shuffleMask = builder.create<BroadcastOp>(
                                    loc, vectorTy32, kernelValue);
                                SmallVector<int64_t, 8> shuffleVals(stride);
                                std::iota(shuffleVals.begin(), shuffleVals.end(), 0);
                                // std::reverse(shuffleVals.begin() + 1, shuffleVals.end());

                                Value padding =
                                    builder.create<ShuffleOp>(loc,
                                    paddingHelper, shuffleMask, shuffleVals);

                                inputVec =
                                    builder.create<vector::MaskedLoadOp>(
                                      loc, vectorTy32, input,
                                      ValueRange{currRow, imCol}, rightMask,
                                      padding);
                              }

                              calcAndStoreFMA(builder, loc, vectorTy32,
                                                inputVec, kernelVec, output,
                                                ValueRange{ivs[0], ivs[2]});

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
                    loc, mlir::CmpIPredicate::slt, currRow, rowMidHelper);

                builder.create<scf::IfOp>(
                    loc, rowMidCond,
                    [&](OpBuilder &builder, Location loc) {
                      // rowMid
                      Value colLeftCond = builder.create<CmpIOp>(
                          loc, mlir::CmpIPredicate::slt, currCol, centerX);

                      builder.create<scf::IfOp>(
                          loc, colLeftCond,
                          [&](OpBuilder &builder, Location loc) {
                            // colLeft & rowMid
                            Value inputVec;
                            Value leftMaskHelper =
                                builder.create<SubIOp>(loc, centerX, currCol);
                            Value leftMaskElem = builder.create<SubIOp>(
                                loc, strideVal, leftMaskHelper);
                            Value leftMask = builder.create<CreateMaskOp>(
                                loc, vectorMask, leftMaskElem);

                            if (!boundaryOption)
                            {
                              Value padding = builder.create<BroadcastOp>(
                                loc, vectorTy32, constantPadding);

                              Value inputVecHelper = builder.create<MaskedLoadOp>(
                                loc, vectorTy32, input, ValueRange{imRow, c0},
                                leftMask, padding);

                              Value p =
                                      rewriter.create<ConstantFloatOp>(loc, (APFloat)(float)100, f32);
                              Value shuffleMask = builder.create<BroadcastOp>(
                                loc, vectorTy32, p);
                            
                              SmallVector<int64_t, 8> shuffleVals(stride);
                              std::iota(shuffleVals.begin(), shuffleVals.end(), 0);
                              // unsigned int p = 
                              // Value a = builder.create<ConstantIndexOp>(loc, leftMaskElem);
                              // unsigned int rotateOffset = 
                              //     calcValue(builder, loc, dyn_cast<memref::DimOp>(leftMaskElem.getDefiningOp()));
                              // // unsigned int p = static_cast<IntegerAttr>(*leftMaskElem.getImpl());
                              // auto h1 = dyn_cast<ConstantIndexOp>(leftMaskElem.getDefiningOp());
                              // auto h = h1.getValue();
                              
                              // ConstantIndexOp j1= builder.create<ConstantIndexOp>(loc, 101);
                              // unsigned j2 = j1.getValue();

                              std::rotate(shuffleVals.begin(), shuffleVals.begin() + 2,
                                shuffleVals.end());

                              

                              inputVec =
                                      builder.create<ShuffleOp>(loc,
                                      inputVecHelper, shuffleMask, shuffleVals);

                              // builder.create<PrintOp>(loc, shuffleMask);
                              // // builder.create<PrintOp>(loc, shuffleVals);
                              // builder.create<PrintOp>(loc, inputVec);
                              // builder.create<PrintOp>(loc, ivs[0]);
                            }

                            // calcAndStoreFMA(builder, loc, vectorTy32,
                            //                     inputVec, kernelVec, output,
                            //                     ValueRange{ivs[0], ivs[2]});

                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            // colMid or colRight
                            Value colMidCond = builder.create<CmpIOp>(
                                loc, mlir::CmpIPredicate::sle, colLastElem,
                                colMidHelper);

                            builder.create<scf::IfOp>(
                                loc, colMidCond,
                                [&](OpBuilder &builder, Location loc) {
                                  // colMid & rowMid
                                  Value inputVec =
                                      builder.create<LoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{imRow, imCol});

                                  calcAndStoreFMA(builder, loc, vectorTy32,
                                                  inputVec, kernelVec, output,
                                                  ValueRange{ivs[0], ivs[2]});

                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
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
                                        loc, vectorTy32, constantPadding);

                                    inputVec =
                                        builder.create<MaskedLoadOp>(
                                            loc, vectorTy32, input,
                                            ValueRange{imRow, imCol}, rightMask,
                                            padding);
                                  } else if (boundaryOption == 1) {
                                    Value rightRange =
                                        builder.create<mlir::SubIOp>(
                                            loc, inputCol, c1);
                                    Value paddingVal =
                                        builder.create<memref::LoadOp>(
                                            loc, input,
                                            ValueRange{imRow, rightRange});
                                    Value padding =
                                        builder.create<vector::BroadcastOp>(
                                            loc, vectorTy32, paddingVal);

                                    inputVec =
                                        builder.create<vector::MaskedLoadOp>(
                                            loc, vectorTy32, input,
                                            ValueRange{imRow, imCol}, rightMask,
                                            padding);
                                  } 
                                  else if (boundaryOption == 2) {
                                    Value refVecElem = 
                                        builder.create<SubIOp>(loc, inputCol, strideVal);
                                  
                                    Value padding = 
                                        builder.create<LoadOp>(loc, vectorTy32, input,
                                        ValueRange{imRow, refVecElem});

                                    inputVec =
                                        builder.create<vector::MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  }

                                  calcAndStoreFMA(builder, loc, vectorTy32,
                                                    inputVec, kernelVec, output,
                                                    ValueRange{ivs[0], ivs[2]});

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
                            loc, vectorTy32, constantPadding);

                        calcAndStoreFMA(builder, loc, vectorTy32, inputVec,
                                        kernelVec, output,
                                        ValueRange{ivs[0], ivs[2]});
                      } else {
                        Value colLeftCond = builder.create<CmpIOp>(
                            loc, mlir::CmpIPredicate::slt, currCol, centerX);

                        builder.create<mlir::scf::IfOp>(
                            loc, colLeftCond,
                            [&](OpBuilder &builder, Location loc) {
                              // colLeft & rowDown
                              if (boundaryOption == 1) {
                                Value downRange = builder.create<mlir::SubIOp>(
                                    loc, inputRow, c1);
                              }

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // (colMid or colRight) & rowDown
                              Value colMidCond = builder.create<CmpIOp>(
                                  loc, mlir::CmpIPredicate::sle, colLastElem,
                                  colMidHelper);

                              builder.create<mlir::scf::IfOp>(
                                  loc, colMidCond,
                                  [&](OpBuilder &builder, Location loc) {
                                    // colMid & rowDown
                                    Value inputVec;
                                    Value downRange =
                                          builder.create<SubIOp>(
                                              loc, inputRow, c1);
                                    if (boundaryOption == 1) {
                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, input,
                                          ValueRange{downRange, imCol});
                                    } else if (boundaryOption == 2) {
                                      Value refRowHelper = 
                                          builder.create<SubIOp>(loc, currRow, rowMidHelper);
                                      Value refRow = 
                                          builder.create<SubIOp>(loc, downRange, refRowHelper);

                                      inputVec = builder.create<LoadOp>(
                                          loc, vectorTy32, input, 
                                          ValueRange{refRow, imCol});
                                    }

                                    calcAndStoreFMA(
                                          builder, loc, vectorTy32, inputVec,
                                          kernelVec, output,
                                          ValueRange{ivs[0], ivs[2]});

                                    builder.create<mlir::scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
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

                                    Value downRange =
                                        builder.create<mlir::SubIOp>(loc, inputRow, c1);
                                    Value rightRange = 
                                        builder.create<SubIOp>(loc, inputCol, c1);

                                    if (boundaryOption == 1) {
                                      Value paddingVal =
                                          builder.create<memref::LoadOp>(
                                              loc, input,
                                              ValueRange{downRange,
                                                         rightRange});
                                      Value padding =
                                          builder.create<vector::BroadcastOp>(
                                              loc, vectorTy32, paddingVal);

                                      inputVec =
                                          builder.create<vector::MaskedLoadOp>(
                                              loc, vectorTy32, input,
                                              ValueRange{downRange, imCol},
                                              rightMask, padding);
                                    } else if (boundaryOption == 2) {
                                      Value refRowHelper = 
                                          builder.create<SubIOp>(loc, currRow, rowMidHelper);
                                      Value refRow = 
                                          builder.create<SubIOp>(loc, downRange, refRowHelper);

                                      Value refVecElem = 
                                          builder.create<SubIOp>(loc, inputCol, strideVal);

                                      Value padding = 
                                          builder.create<LoadOp>(loc, vectorTy32, input,
                                          ValueRange{refRow, refVecElem});

                                      inputVec =
                                          builder.create<vector::MaskedLoadOp>(
                                              loc, vectorTy32, input,
                                              ValueRange{refRow, imCol},
                                              rightMask, padding);
                                    }

                                    calcAndStoreFMA(
                                          builder, loc, vectorTy32, inputVec,
                                          kernelVec, output,
                                          ValueRange{ivs[0], ivs[2]});

                                    builder.create<mlir::scf::YieldOp>(loc);
                                  });
                              builder.create<mlir::scf::YieldOp>(loc);
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

void populateLowerDIPConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<DIPCorr2DLowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// LowerDIPPass
//===----------------------------------------------------------------------===//

namespace {
class LowerDIPPass : public PassWrapper<LowerDIPPass, OperationPass<ModuleOp>> {
public:
  LowerDIPPass() = default;
  LowerDIPPass(const LowerDIPPass &) {}

  StringRef getArgument() const final { return "lower-DIP"; }
  StringRef getDescription() const final { return "Lower DIP Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Buddy::DIP::DIPDialect, StandardOpsDialect,
                    memref::MemRefDialect, scf::SCFDialect, VectorDialect,
                    AffineDialect>();
  }
};
} // end anonymous namespace.

void LowerDIPPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, scf::SCFDialect, StandardOpsDialect,
                         memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ReturnOp>();

  RewritePatternSet patterns(context);
  populateLowerDIPConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace Buddy {
void registerLowerDIPPass() { PassRegistration<LowerDIPPass>(); }
} // namespace Buddy
} // namespace mlir
