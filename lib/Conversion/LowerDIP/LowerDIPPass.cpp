//====- LowerDIPPass.cpp - dip Dialect Lowering Pass  ---------------------===//
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"

#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include <vector>

using namespace mlir;
using namespace buddy;
using namespace vector;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
// Calculate result of FMA and store it in output memref. This function cannot
// handle tail processing.
void calcAndStoreFMAwoTailProcessing(OpBuilder &builder, Location loc,
                                     VectorType vecType, Value inputVec,
                                     Value kernelVec, Value output,
                                     Value beginIdx, Value endIdx) {
  Value outputVec = builder.create<LoadOp>(loc, vecType, output,
                                           ValueRange{beginIdx, endIdx});
  Value resVec = builder.create<FMAOp>(loc, inputVec, kernelVec, outputVec);
  builder.create<StoreOp>(loc, resVec, output, ValueRange{beginIdx, endIdx});
}

Value tailChecker(OpBuilder &builder, Location loc, AffineMap calcHelper,
                  Value strideVal, Value kernelSize, Value c1, Value pseudoCol,
                  Value colPivot) {
  Value tailChecker = builder.create<AffineApplyOp>(
      loc, calcHelper, ValueRange{strideVal, kernelSize, c1});
  Value colEndDistance = builder.create<SubIOp>(loc, pseudoCol, colPivot);
  Value tailCond = builder.create<CmpIOp>(loc, CmpIPredicate::sge,
                                          colEndDistance, tailChecker);
  return tailCond;
}

Value tailMaskCreator(OpBuilder &builder, Location loc, Value inputCol,
                      Value colPivot, VectorType vectorMaskTy) {
  Value extraElemCount = builder.create<SubIOp>(loc, inputCol, colPivot);
  Value tailMask =
      builder.create<CreateMaskOp>(loc, vectorMaskTy, extraElemCount);
  return tailMask;
}

// Calculate result of FMA and store it in output memref. This function can
// handle tail processing.
void calcAndStoreFMAwTailProcessing(OpBuilder &builder, Location loc,
                                    VectorType vecType, Value inputVec,
                                    Value kernelVec, Value output,
                                    Value beginIdx, Value endIdx,
                                    Value tailCond, Value zeroPadding,
                                    Value inputCol, VectorType vectorMaskTy) {
  builder.create<scf::IfOp>(
      loc, tailCond,
      [&](OpBuilder &builder, Location loc) {
        Value outputVec = builder.create<LoadOp>(loc, vecType, output,
                                                 ValueRange{beginIdx, endIdx});
        Value resVec =
            builder.create<FMAOp>(loc, inputVec, kernelVec, outputVec);
        builder.create<StoreOp>(loc, resVec, output,
                                ValueRange{beginIdx, endIdx});

        builder.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &builder, Location loc) {
        Value extraElemMask =
            tailMaskCreator(builder, loc, inputCol, endIdx, vectorMaskTy);
        Value outputVec = builder.create<MaskedLoadOp>(
            loc, vecType, output, ValueRange{beginIdx, endIdx}, extraElemMask,
            zeroPadding);
        Value resVec =
            builder.create<FMAOp>(loc, inputVec, kernelVec, outputVec);
        builder.create<MaskedStoreOp>(loc, output, ValueRange{beginIdx, endIdx},
                                      extraElemMask, resVec);

        builder.create<scf::YieldOp>(loc);
      });
}

// Create an inverted mask having all 1's shifted to right side.
Value createInvertedMask(OpBuilder &builder, Location loc, Value strideVal,
                         VectorType vectorMaskTy, Value leftIndex) {
  Value leftMask = builder.create<CreateMaskOp>(loc, vectorMaskTy, leftIndex);
  Value maskInverter =
      builder.create<CreateMaskOp>(loc, vectorMaskTy, strideVal);
  Value rightMask = builder.create<SubIOp>(loc, maskInverter, leftMask);
  return rightMask;
}

class DIPCorr2DLowering : public OpRewritePattern<dip::Corr2DOp> {
public:
  using OpRewritePattern<dip::Corr2DOp>::OpRewritePattern;

  explicit DIPCorr2DLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Corr2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value centerX = op->getOperand(3);
    Value centerY = op->getOperand(4);
    // Value boundaryOptionVal = op->getOperand(5);
    unsigned int boundaryOption = 1;
    // ToDo : Make boundaryOption an attribute.
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    FloatType f32 = FloatType::getF32(ctx);
    IntegerType i1 = IntegerType::get(ctx, 1);

    // Create DimOp.
    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);
    Value kernelSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    // Variables used for detecting rowMid, rowDown, colMid and colRight
    // regions.
    Value rowMidHelper = rewriter.create<AddIOp>(loc, inputRow, centerY);
    Value colMidHelper = rewriter.create<AddIOp>(loc, inputCol, centerX);

    SmallVector<Value, 8> lowerBounds(4, c0);
    SmallVector<Value, 8> uperBounds{inputRow, kernelSize, inputCol,
                                     kernelSize};
    SmallVector<int64_t, 8> steps{1, 1, stride, 1};

    VectorType vectorTy32 = VectorType::get({stride}, f32);
    VectorType vectorMaskTy = VectorType::get({stride}, i1);

    // Improve this flow for constant padding option.
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
          // Indices of current pixel with respect to pseudo image containing
          // extrapolated boundaries.
          Value currRow = builder.create<AddIOp>(loc, ivs[0], ivs[1]);
          Value currCol = builder.create<AddIOp>(loc, ivs[2], ivs[3]);

          Value kernelValue = builder.create<memref::LoadOp>(
              loc, kernel, ValueRange{ivs[1], ivs[3]});
          Value kernelVec =
              builder.create<BroadcastOp>(loc, vectorTy32, kernelValue);

          // Pixel indices with respect to the actual image.
          Value imRow = builder.create<SubIOp>(loc, currRow, centerY);
          Value imCol = builder.create<SubIOp>(loc, currCol, centerX);

          // Index of pixel used for determining right region.
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

                  calcAndStoreFMAwoTailProcessing(builder, loc, vectorTy32,
                                                  inputVec, kernelVec, output,
                                                  ivs[0], ivs[2]);
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
                        Value leftMask =
                            createInvertedMask(builder, loc, strideVal,
                                               vectorMaskTy, leftMaskElem);

                        if (boundaryOption == 1) {
                          Value paddingVal = builder.create<memref::LoadOp>(
                              loc, input, ValueRange{c0, c0});
                          Value padding = builder.create<BroadcastOp>(
                              loc, vectorTy32, paddingVal);

                          Value leftPaddingOffset =
                              builder.create<SubIOp>(loc, c0, leftMaskElem);
                          inputVec = builder.create<vector::MaskedLoadOp>(
                              loc, vectorTy32, input,
                              ValueRange{c0, leftPaddingOffset}, leftMask,
                              padding);
                        }
                        calcAndStoreFMAwoTailProcessing(
                            builder, loc, vectorTy32, inputVec, kernelVec,
                            output, ivs[0], ivs[2]);

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
                              calcAndStoreFMAwoTailProcessing(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2]);

                              builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &builder, Location loc) {
                              // colRight & rowUp
                              Value inputVec;
                              Value rightMaskHelper = builder.create<SubIOp>(
                                  loc, colLastElem, colMidHelper);
                              Value rightMaskElem = builder.create<SubIOp>(
                                  loc, strideVal, rightMaskHelper);
                              Value rightMask = builder.create<CreateMaskOp>(
                                  loc, vectorMaskTy, rightMaskElem);

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
                              Value tailCond = tailChecker(
                                  builder, loc, calcHelper, strideVal,
                                  kernelSize, c1, pseudoCol, ivs[2]);
                              calcAndStoreFMAwTailProcessing(
                                  builder, loc, vectorTy32, inputVec, kernelVec,
                                  output, ivs[0], ivs[2], tailCond, zeroPadding,
                                  inputCol, vectorMaskTy);

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
                                                   vectorMaskTy, leftMaskElem);

                            if (!boundaryOption) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, zeroPaddingElem);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            } else if (boundaryOption == 1) {
                              Value paddingVal = builder.create<memref::LoadOp>(
                                  loc, input, ValueRange{imRow, c0});
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, paddingVal);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            }
                            calcAndStoreFMAwoTailProcessing(
                                builder, loc, vectorTy32, inputVec, kernelVec,
                                output, ivs[0], ivs[2]);

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
                                  calcAndStoreFMAwoTailProcessing(
                                      builder, loc, vectorTy32, inputVec,
                                      kernelVec, output, ivs[0], ivs[2]);

                                  builder.create<scf::YieldOp>(loc);
                                },
                                [&](OpBuilder &builder, Location loc) {
                                  // colRight & rowMid
                                  Value inputVec;
                                  Value rightMaskHelper =
                                      builder.create<SubIOp>(loc, colLastElem,
                                                             colMidHelper);
                                  Value rightMaskElem = builder.create<SubIOp>(
                                      loc, strideVal, rightMaskHelper);
                                  Value rightMask =
                                      builder.create<CreateMaskOp>(
                                          loc, vectorMaskTy, rightMaskElem);

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
                                  Value tailCond = tailChecker(
                                      builder, loc, calcHelper, strideVal,
                                      kernelSize, c1, pseudoCol, ivs[2]);
                                  calcAndStoreFMAwTailProcessing(
                                      builder, loc, vectorTy32, inputVec,
                                      kernelVec, output, ivs[0], ivs[2],
                                      tailCond, zeroPadding, inputCol,
                                      vectorMaskTy);

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

                        calcAndStoreFMAwoTailProcessing(
                            builder, loc, vectorTy32, inputVec, kernelVec,
                            output, ivs[0], ivs[2]);
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
                              Value leftMask = createInvertedMask(
                                  builder, loc, strideVal, vectorMaskTy,
                                  leftMaskElem);

                              if (boundaryOption == 1) {
                                Value paddingVal =
                                    builder.create<memref::LoadOp>(
                                        loc, input, ValueRange{downRange, c0});
                                Value padding = builder.create<BroadcastOp>(
                                    loc, vectorTy32, paddingVal);

                                Value leftPaddingOffset =
                                    builder.create<SubIOp>(loc, c0,
                                                           leftMaskElem);
                                inputVec = builder.create<MaskedLoadOp>(
                                    loc, vectorTy32, input,
                                    ValueRange{downRange, leftPaddingOffset},
                                    leftMask, padding);
                              }
                              calcAndStoreFMAwoTailProcessing(
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
                                    }
                                    calcAndStoreFMAwoTailProcessing(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2]);

                                    builder.create<scf::YieldOp>(loc);
                                  },
                                  [&](OpBuilder &builder, Location loc) {
                                    // colRight & rowDown
                                    Value inputVec;
                                    Value rightMaskHelper =
                                        builder.create<SubIOp>(loc, colLastElem,
                                                               colMidHelper);
                                    Value rightMaskElem =
                                        builder.create<SubIOp>(loc, strideVal,
                                                               rightMaskHelper);
                                    Value rightMask =
                                        builder.create<CreateMaskOp>(
                                            loc, vectorMaskTy, rightMaskElem);

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
                                    Value tailCond = tailChecker(
                                        builder, loc, calcHelper, strideVal,
                                        kernelSize, c1, pseudoCol, ivs[2]);
                                    calcAndStoreFMAwTailProcessing(
                                        builder, loc, vectorTy32, inputVec,
                                        kernelVec, output, ivs[0], ivs[2],
                                        tailCond, zeroPadding, inputCol,
                                        vectorMaskTy);

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

private:
  int64_t stride;
};

Value indexToF32(OpBuilder &builder, Location loc, Value val)
{
    Value interm1 = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), val);
    Value res = builder.create<arith::SIToFPOp>(loc, builder.getF32Type(), interm1);

    return res;
}

Value F32ToIndex(OpBuilder &builder, Location loc, Value val)
{
    Value interm1 = builder.create<arith::FPToUIOp>(loc, builder.getI32Type(), val);
    Value res = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), interm1);

    return res;
}

Value roundOff(OpBuilder &builder, Location loc, Value val, VectorType vecTypeI, VectorType vecTypeF)
{
    Value interm1 = builder.create<arith::FPToSIOp>(loc, vecTypeI, val);
    Value interm2 = builder.create<arith::SIToFPOp>(loc, vecTypeF, interm1);

    return interm2;
}

std::vector<Value> shearTransform(OpBuilder &builder, Location loc, Value originalX, Value originalY, 
                            Value sinVec, Value tanVec, VectorType vecTypeI, VectorType vecTypeF)
{
    Value yTan1 = builder.create<arith::MulFOp>(loc, tanVec, originalY);
    Value xIntermediate1 = builder.create<arith::SubFOp>(loc, originalX, yTan1);
    Value xIntermediate = roundOff(builder, loc, xIntermediate1, vecTypeI, vecTypeF);
    // round xIntermediate

    Value xSin = builder.create<arith::MulFOp>(loc, xIntermediate, sinVec);
    Value newY1 = builder.create<arith::AddFOp>(loc, xSin, originalY);
    Value newY = roundOff(builder, loc, newY1, vecTypeI, vecTypeF);
    // round newY
    
    Value yTan2 = builder.create<arith::MulFOp>(loc, newY, tanVec);
    Value newX1 = builder.create<arith::SubFOp>(loc, xIntermediate, yTan2);
    Value newX = roundOff(builder, loc, newX1, vecTypeI, vecTypeF);
    // round newX

    return {newX, newY};
}

Value getCenter(OpBuilder &builder, Location loc, MLIRContext *ctx, Value dim)
{
    Value c1 = builder.create<ConstantIndexOp>(loc, 1);
    Value c2 = builder.create<ConstantIndexOp>(loc, 2);
    
    Value temp1 = builder.create<arith::AddIOp>(loc, dim, c1);
    Value temp2 = builder.create<arith::DivUIOp>(loc, temp1, c2);
    Value center = builder.create<arith::SubIOp>(loc, temp2, c1);
    // round center

    return center; 
}

Value iotaVec(OpBuilder &builder, Location loc, MLIRContext *ctx, Value lowerBound, 
              Value strideVal, VectorType vecType, FloatType f32)
{
    Value c0 = builder.create<ConstantIndexOp>(loc, 0);
    Value c0f32 = builder.create<ConstantFloatOp>(loc, (llvm::APFloat)(float)0, f32);
    Value tempVec = builder.create<vector::SplatOp>(loc, vecType, c0f32);

    Value c1 = builder.create<ConstantIndexOp>(loc, 1);
    Value c2 = builder.create<ConstantIndexOp>(loc, 2);
    Value c3 = builder.create<ConstantIndexOp>(loc, 3);
    Value c4 = builder.create<ConstantIndexOp>(loc, 4);
    Value c5 = builder.create<ConstantIndexOp>(loc, 5);

    Value p11 = builder.create<arith::AddIOp>(loc, lowerBound, c1);
    Value p22 = builder.create<arith::AddIOp>(loc, lowerBound, c2);
    Value p33 = builder.create<arith::AddIOp>(loc, lowerBound, c3);
    Value p44 = builder.create<arith::AddIOp>(loc, lowerBound, c4);
    Value p55 = builder.create<arith::AddIOp>(loc, lowerBound, c5);

    Value p0 = indexToF32(builder, loc, lowerBound);
    Value p1 = indexToF32(builder, loc, p11);
    Value p2 = indexToF32(builder, loc, p22);
    Value p3 = indexToF32(builder, loc, p33);
    Value p4 = indexToF32(builder, loc, p44);
    Value p5 = indexToF32(builder, loc, p55);
    
    tempVec = builder.create<vector::InsertElementOp>(loc, p0, tempVec, c0);
    tempVec = builder.create<vector::InsertElementOp>(loc, p1, tempVec, c1);
    tempVec = builder.create<vector::InsertElementOp>(loc, p2, tempVec, c2);
    tempVec = builder.create<vector::InsertElementOp>(loc, p3, tempVec, c3);
    tempVec = builder.create<vector::InsertElementOp>(loc, p4, tempVec, c4);
    tempVec = builder.create<vector::InsertElementOp>(loc, p5, tempVec, c5);

    // builder.create<vector::PrintOp>(loc, tempVec);
    // builder.create<vector::PrintOp>(loc, lowerBound);

    return tempVec;
}

Value pixelScaling(OpBuilder &builder, Location loc, Value imageDImF32Vec, Value coordVec,
                   Value imageCenterF32Vec, Value c1f32Vec)
{
    Value interm1 = builder.create<arith::SubFOp>(loc, imageDImF32Vec, coordVec);
    Value interm2 = builder.create<arith::SubFOp>(loc, interm1, imageCenterF32Vec);
    Value res = builder.create<arith::SubFOp>(loc, interm2, c1f32Vec);

    return res;
}

void fillPixels(OpBuilder &builder, Location loc, Value resXVec, Value resYVec, 
                Value xVec, Value yVec, Value input, Value output, Value c0, Value strideVal)
{
    SmallVector<Value, 8> lowerBounds{c0};
    SmallVector<Value, 8> upperBounds{strideVal};
    SmallVector<intptr_t, 8> steps{1};
    // check vector usage for loading and storing

    buildAffineLoopNest(
        builder, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
            Value resXPos = builder.create<vector::ExtractElementOp>(loc, resXVec, ivs[0]);
            Value resYPos = builder.create<vector::ExtractElementOp>(loc, resYVec, ivs[0]);

            Value resXPosIndex = F32ToIndex(builder, loc, resXPos);
            Value resYPosIndex = F32ToIndex(builder, loc, resYPos);

            Value xPos = builder.create<vector::ExtractElementOp>(loc, xVec, ivs[0]);
            Value yPos = builder.create<vector::ExtractElementOp>(loc, yVec, ivs[0]);

            Value xPosIndex = F32ToIndex(builder, loc, xPos);
            Value yPosIndex = F32ToIndex(builder, loc, yPos);

            Value pixelVal = builder.create<memref::LoadOp>(loc, builder.getF32Type(), input, 
                                ValueRange{xPosIndex, yPosIndex});
            // builder.create<memref::StoreOp>(loc, pixelVal, output, 
            //                     ValueRange{resXPosIndex, resYPosIndex});
    });
}

Value castAndExpand(OpBuilder &builder, Location loc, Value val, VectorType vecType)
{
    Value interm1 = indexToF32(builder, loc, val);
    Value interm2 = builder.create<vector::SplatOp>(loc, vecType, interm1);

    return interm2;
}

class DIPRotate2DOpLowering : public OpRewritePattern<dip::Rotate2DOp> {
public:
  using OpRewritePattern<dip::Rotate2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(dip::Rotate2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value output = op->getOperand(2);

    Value strideVal = rewriter.create<ConstantIndexOp>(loc, 6);

    FloatType f32 = FloatType::getF32(ctx);
    VectorType vectorTy32F = VectorType::get({6}, f32);
    VectorType vectorTy32I = VectorType::get({6}, rewriter.getI32Type());

    // Value angleVal = op->getOperand(1);
    // float angle = 90;
    Value angleVal = rewriter.create<ConstantFloatOp>(loc, (llvm::APFloat)0.0f, f32);
    // Value angleVal = rewriter.create<ConstantIndexOp>(loc, 90);

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);

    Value inputRowF32Vec = castAndExpand(rewriter, loc, inputRow, vectorTy32F);
    Value inputColF32Vec = castAndExpand(rewriter, loc, inputCol, vectorTy32F);

    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c1);

    SmallVector<Value, 8> lowerBounds(2, c0);
    SmallVector<Value, 8> upperBounds{inputRow, inputCol};
    // SmallVector<Value, 8> upperBounds{c1, c1};
    SmallVector<intptr_t, 8> steps(2, 6);

    Value inputCenterY = getCenter(rewriter, loc, ctx, inputRow);
    Value inputCenterX = getCenter(rewriter, loc, ctx, inputCol);

    Value inputCenterYF32Vec = castAndExpand(rewriter, loc, inputCenterY, vectorTy32F);
    Value inputCenterXF32Vec = castAndExpand(rewriter, loc, inputCenterX, vectorTy32F);

    Value outputCenterY = getCenter(rewriter, loc, ctx, outputRow);
    Value outputCenterX = getCenter(rewriter, loc, ctx, outputCol);

    Value outputCenterYF32Vec = castAndExpand(rewriter, loc, outputCenterY, vectorTy32F);
    Value outputCenterXF32Vec = castAndExpand(rewriter, loc, outputCenterX, vectorTy32F);

    Value c1f32 = rewriter.create<ConstantFloatOp>(loc, (llvm::APFloat)(float)1, f32);
    Value c1f32Vec = rewriter.create<vector::SplatOp>(loc, vectorTy32F, c1f32);

    Value sinVal = rewriter.create<math::SinOp>(loc, angleVal);
    Value sinVec = rewriter.create<vector::BroadcastOp>(loc, vectorTy32F, sinVal);

    Value cosVal = rewriter.create<math::CosOp>(loc, angleVal);
    Value tanVal = rewriter.create<arith::DivFOp>(loc, sinVal, cosVal);
    Value tanVec = rewriter.create<vector::BroadcastOp>(loc, vectorTy32F, tanVal);

    buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
            Value yLowerBoundMult = builder.create<arith::DivUIOp>(loc, ivs[0], strideVal);
            Value xLowerBoundMult = builder.create<arith::DivUIOp>(loc, ivs[1], strideVal);

            Value yLowerBound = builder.create<arith::MulIOp>(loc, strideVal, yLowerBoundMult);
            Value xLowerBound = builder.create<arith::MulIOp>(loc, strideVal, xLowerBoundMult);

            Value yVec = 
                iotaVec(builder, loc, ctx, yLowerBound, strideVal, vectorTy32F, f32);
            Value xVec = 
                iotaVec(builder, loc, ctx, xLowerBound, strideVal, vectorTy32F, f32);

            Value yVecModified = pixelScaling(builder, loc, inputRowF32Vec, yVec, 
                                              inputCenterYF32Vec, c1f32Vec);
            Value xVecModified = pixelScaling(builder, loc, inputColF32Vec, xVec, 
                                              inputCenterXF32Vec, c1f32Vec);

            std::vector<Value> resIndices = 
                shearTransform(builder, loc, xVecModified, yVecModified, sinVec, tanVec, 
                               vectorTy32I, vectorTy32F);

            Value resYVec = builder.create<arith::SubFOp>(loc, outputCenterYF32Vec, resIndices[1]);
            Value resXVec = builder.create<arith::SubFOp>(loc, outputCenterXF32Vec, resIndices[0]);

            builder.create<vector::PrintOp>(loc, resYVec);
            builder.create<vector::PrintOp>(loc, resXVec);

            // fillPixels(builder, loc, resXVec, resYVec, xVec, yVec, input, output, c0, strideVal);
    });

    // Remove the origin rotation operation.
    rewriter.eraseOp(op);
    return success();
  }
};
} // end anonymous namespace

void populateLowerDIPConversionPatterns(RewritePatternSet &patterns,
                                        int64_t stride) {
  patterns.add<DIPCorr2DLowering>(patterns.getContext(), stride);
  patterns.add<DIPRotate2DOpLowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// LowerDIPPass
//===----------------------------------------------------------------------===//

namespace {
class LowerDIPPass : public PassWrapper<LowerDIPPass, OperationPass<ModuleOp>> {
public:
  LowerDIPPass() = default;
  LowerDIPPass(const LowerDIPPass &) {}
  explicit LowerDIPPass(int64_t strideParam) { stride = strideParam; }

  StringRef getArgument() const final { return "lower-dip"; }
  StringRef getDescription() const final { return "Lower DIP Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<buddy::dip::DIPDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, VectorDialect,
                    AffineDialect, arith::ArithmeticDialect, math::MathDialect>();
  }

  Option<int64_t> stride{*this, "DIP-strip-mining",
                         llvm::cl::desc("Strip mining size."),
                         llvm::cl::init(32)};
};
} // end anonymous namespace.

void LowerDIPPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, scf::SCFDialect, func::FuncDialect,
                         memref::MemRefDialect, VectorDialect,
                         arith::ArithmeticDialect, math::MathDialect>();
  target.addLegalOp<ModuleOp, FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  populateLowerDIPConversionPatterns(patterns, stride);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerDIPPass() { PassRegistration<LowerDIPPass>(); }
} // namespace buddy
} // namespace mlir


// ./buddy-opt ../../examples/DIPDialect/TestCorr2D.mlir -lower-dip -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | ../../llvm/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=../../llvm/build/lib/libmlir_c_runner_utils.so -shared-libs=../../llvm/build/lib/libmlir_runner_utils.so

// ./buddy-opt ../../examples/DIPDialect/corr2d.mlir -lower-dip="DIP-strip-mining=6" -lower-affine -convert-scf-to-cf --convert-math-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -convert-func-to-llvm='emit-c-wrappers=1' -reconcile-unrealized-casts | ../../llvm/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=../../llvm/build/lib/libmlir_c_runner_utils.so -shared-libs=../../llvm/build/lib/libmlir_runner_utils.so
