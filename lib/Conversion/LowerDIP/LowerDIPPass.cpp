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
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include "Utils/DIPUtils.h"
#include "Utils/Utils.h"
#include <vector>

using namespace mlir;
using namespace buddy;
using namespace vector;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class DIPCorr2DOpLowering : public OpRewritePattern<dip::Corr2DOp> {
public:
  using OpRewritePattern<dip::Corr2DOp>::OpRewritePattern;

  explicit DIPCorr2DOpLowering(MLIRContext *context, int64_t strideParam)
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
    Value constantValue = op->getOperand(5);
    auto boundaryOptionAttr = op.boundary_option();
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
                if (boundaryOptionAttr ==
                    dip::BoundaryOption::ConstantPadding) {
                  Value inputVec = builder.create<BroadcastOp>(loc, vectorTy32,
                                                               constantValue);

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

                        if (boundaryOptionAttr ==
                            dip::BoundaryOption::ReplicatePadding) {
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
                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
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

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
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

                            if (boundaryOptionAttr ==
                                dip::BoundaryOption::ConstantPadding) {
                              Value padding = builder.create<BroadcastOp>(
                                  loc, vectorTy32, constantValue);

                              Value leftPaddingOffset =
                                  builder.create<SubIOp>(loc, c0, leftMaskElem);
                              inputVec = builder.create<MaskedLoadOp>(
                                  loc, vectorTy32, input,
                                  ValueRange{imRow, leftPaddingOffset},
                                  leftMask, padding);
                            } else if (boundaryOptionAttr ==
                                       dip::BoundaryOption::ReplicatePadding) {
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

                                  if (boundaryOptionAttr ==
                                      dip::BoundaryOption::ConstantPadding) {
                                    Value padding = builder.create<BroadcastOp>(
                                        loc, vectorTy32, constantValue);

                                    inputVec = builder.create<MaskedLoadOp>(
                                        loc, vectorTy32, input,
                                        ValueRange{imRow, imCol}, rightMask,
                                        padding);
                                  } else if (boundaryOptionAttr ==
                                             dip::BoundaryOption::
                                                 ReplicatePadding) {
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
                      if (boundaryOptionAttr ==
                          dip::BoundaryOption::ConstantPadding) {
                        Value inputVec = builder.create<BroadcastOp>(
                            loc, vectorTy32, constantValue);

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

                              if (boundaryOptionAttr ==
                                  dip::BoundaryOption::ReplicatePadding) {
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
                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {
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

                                    if (boundaryOptionAttr ==
                                        dip::BoundaryOption::ReplicatePadding) {

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

class DIPRotate2DOpLowering : public OpRewritePattern<dip::Rotate2DOp> {
public:
  using OpRewritePattern<dip::Rotate2DOp>::OpRewritePattern;

  explicit DIPRotate2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Rotate2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value angleVal = op->getOperand(1);
    Value output = op->getOperand(2);

    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);
    FloatType f32 = FloatType::getF32(ctx);
    VectorType vectorTy32 = VectorType::get({stride}, f32);

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    Value c0F32 = indexToF32(rewriter, loc, c0);
    Value c1F32 = indexToF32(rewriter, loc, c1);
    Value c1F32Vec = rewriter.create<vector::SplatOp>(loc, vectorTy32, c1F32);

    // Get input image dimensions.
    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);

    // Create f32 type vectors from input dimensions.
    Value inputRowF32Vec = castAndExpand(rewriter, loc, inputRow, vectorTy32);
    Value inputColF32Vec = castAndExpand(rewriter, loc, inputCol, vectorTy32);

    // Get output image dimensions.
    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c1);

    // Obtain extreme allocatable value(s) in output for bounding purpose.
    Value outputRowLastElem =
        rewriter.create<arith::SubIOp>(loc, outputRow, c1);
    Value outputRowLastElemF32 = indexToF32(rewriter, loc, outputRowLastElem);

    Value outputColLastElem =
        rewriter.create<arith::SubIOp>(loc, outputCol, c1);
    Value outputColLastElemF32 = indexToF32(rewriter, loc, outputColLastElem);

    // Determine lower bound for second call of rotation function (this is done
    // for efficient tail processing).
    Value inputColStrideRatio =
        rewriter.create<arith::DivUIOp>(loc, inputCol, strideVal);
    Value inputColMultiple =
        rewriter.create<arith::MulIOp>(loc, strideVal, inputColStrideRatio);

    // Bounds for first call to rotation function (doesn't involve tail
    // processing).
    SmallVector<Value, 8> lowerBounds1(2, c0);
    SmallVector<Value, 8> upperBounds1{inputRow, inputColMultiple};

    // Bounds for second call to rotation function (involves tail processing).
    SmallVector<Value, 8> lowerBounds2{c0, inputColMultiple};
    SmallVector<Value, 8> upperBounds2{inputRow, inputCol};

    SmallVector<int64_t, 8> steps{1, stride};
    Value strideTailVal =
        rewriter.create<arith::SubIOp>(loc, inputCol, inputColMultiple);

    // Get input image center.
    Value inputCenterY = getCenter(rewriter, loc, ctx, inputRow);
    Value inputCenterX = getCenter(rewriter, loc, ctx, inputCol);

    Value inputCenterYF32Vec =
        castAndExpand(rewriter, loc, inputCenterY, vectorTy32);
    Value inputCenterXF32Vec =
        castAndExpand(rewriter, loc, inputCenterX, vectorTy32);

    // Get output image center.
    Value outputCenterY = getCenter(rewriter, loc, ctx, outputRow);
    Value outputCenterX = getCenter(rewriter, loc, ctx, outputCol);

    Value outputCenterYF32Vec =
        castAndExpand(rewriter, loc, outputCenterY, vectorTy32);
    Value outputCenterXF32Vec =
        castAndExpand(rewriter, loc, outputCenterX, vectorTy32);

    // Get sin(angle) which will be used in further calculations.
    Value sinVal = rewriter.create<math::SinOp>(loc, angleVal);
    Value sinVec =
        rewriter.create<vector::BroadcastOp>(loc, vectorTy32, sinVal);

    // Get tan(angle / 2) which will be used in further calculations.
    Value tanVal = customTanVal(rewriter, loc, angleVal);
    Value tanVec =
        rewriter.create<vector::BroadcastOp>(loc, vectorTy32, tanVal);

    // Determine the condition for chosing ideal rotation strategy.
    Value tanBound =
        rewriter.create<ConstantFloatOp>(loc, (llvm::APFloat)8.10f, f32);
    Value tanValAbs = rewriter.create<math::AbsOp>(loc, tanVal);
    Value transformCond = rewriter.create<arith::CmpFOp>(
        loc, CmpFPredicate::OGT, tanBound, tanValAbs);

    // For both rotation strategies, tail processing is handled in second call.
    rewriter.create<scf::IfOp>(
        loc, transformCond,
        [&](OpBuilder &builder, Location loc) {
          shearTransformController(
              builder, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
              input, output, sinVec, tanVec, inputRowF32Vec, inputColF32Vec,
              inputCenterYF32Vec, inputCenterXF32Vec, outputCenterYF32Vec,
              outputCenterXF32Vec, outputRowLastElemF32, outputColLastElemF32,
              c0, c0F32, c1F32Vec, vectorTy32, stride, f32);

          shearTransformController(
              builder, loc, ctx, lowerBounds2, upperBounds2, steps,
              strideTailVal, input, output, sinVec, tanVec, inputRowF32Vec,
              inputColF32Vec, inputCenterYF32Vec, inputCenterXF32Vec,
              outputCenterYF32Vec, outputCenterXF32Vec, outputRowLastElemF32,
              outputColLastElemF32, c0, c0F32, c1F32Vec, vectorTy32, stride,
              f32);

          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          standardRotateController(
              builder, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
              input, output, sinVec, angleVal, inputRowF32Vec, inputColF32Vec,
              inputCenterYF32Vec, inputCenterXF32Vec, outputCenterYF32Vec,
              outputCenterXF32Vec, outputRowLastElemF32, outputColLastElemF32,
              c0, c0F32, c1F32Vec, vectorTy32, stride, f32);

          standardRotateController(
              builder, loc, ctx, lowerBounds2, upperBounds2, steps,
              strideTailVal, input, output, sinVec, angleVal, inputRowF32Vec,
              inputColF32Vec, inputCenterYF32Vec, inputCenterXF32Vec,
              outputCenterYF32Vec, outputCenterXF32Vec, outputRowLastElemF32,
              outputColLastElemF32, c0, c0F32, c1F32Vec, vectorTy32, stride,
              f32);

          builder.create<scf::YieldOp>(loc);
        });

    // Remove the origin rotation operation.
    rewriter.eraseOp(op);
    return success();
  }

  int64_t stride;
};
} // end anonymous namespace

void NearestNeighbourInterpolationResizing(
  OpBuilder &builder, Location loc, MLIRContext *ctx, 
  SmallVector<Value, 8> lowerBounds, SmallVector<Value, 8> upperBounds, 
  SmallVector<int64_t, 8> steps, Value strideVal, Value input, Value output, 
  Value horizontalScalingFactorVec, Value verticalScalingFactorVec, 
  Value outputRowLastElemF32, Value outputColLastElemF32, Value inputRowLastElemF32, 
  Value inputColLastElemF32, VectorType vectorTy32, 
  int64_t stride, Value c0, Value c0F32)
{
  buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
    Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
    Value yVec = builder.create<vector::SplatOp>(loc, vectorTy32, ivs0F32);
    Value xVec = iotaVec(builder, loc, ctx, ivs[1], strideVal,
                         vectorTy32, c0, stride);

    Value resXVecInterm = builder.create<arith::MulFOp>(loc, xVec, horizontalScalingFactorVec);
    Value resYVecInterm = builder.create<arith::MulFOp>(loc, yVec, verticalScalingFactorVec);

    Value resXVec = roundOff(builder, loc, resXVecInterm);
    Value resYVec = roundOff(builder, loc, resYVecInterm);

    fillPixels_check(builder, loc, xVec, yVec, resXVec, resYVec, input, output, 
               c0, strideVal, outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32, inputColLastElemF32,
               c0F32);
  });
}

std::vector<Value> extractIndices(
    OpBuilder &builder, Location loc, Value xVec, Value yVec, 
    Value vecIndex, Value xUpperBound, Value yUpperBound, Value c0F32) {
  Value xPos = builder.create<vector::ExtractElementOp>(loc, xVec, vecIndex);
  Value yPos = builder.create<vector::ExtractElementOp>(loc, yVec, vecIndex);

  Value xPosBound =
        valBound(builder, loc, xPos, xUpperBound, c0F32);
  Value yPosBound = 
        valBound(builder, loc, yPos, yUpperBound, c0F32);

  return {F32ToIndex(builder, loc, xPosBound), F32ToIndex(builder, loc, yPosBound)};
}

void fillPixelsBillinearInterpolate(
    OpBuilder &builder, Location loc, Value resXVec, Value resYVec,
    Value xVec_L, Value yVec_L, Value xVec_H, Value yVec_H, Value input, Value output, Value c0,
    Value strideVal, Value outputRowLastElemF32, Value xVecWeight, Value yVecWeight,
    Value outputColLastElemF32, Value inputRowLastElemF32,
    Value inputColLastElemF32, Value c0F32, Value c1F32) {
  builder.create<AffineForOp>(
      loc, ValueRange{c0}, builder.getDimIdentityMap(), ValueRange{strideVal},
      builder.getDimIdentityMap(), /*step*/ 1, llvm::None,
      [&](OpBuilder &builder, Location loc, ValueRange ivs,
          ValueRange iterArg) {
    std::vector<Value> resIndices = extractIndices(builder, loc, resXVec, resYVec, ivs[0], 
                                        outputColLastElemF32, outputRowLastElemF32, c0F32);

    std::vector<Value> inputIndices_L = extractIndices(builder, loc, xVec_L, yVec_L, ivs[0], 
                                            inputColLastElemF32, inputRowLastElemF32, c0F32);
    std::vector<Value> inputIndices_H = extractIndices(builder, loc, xVec_H, yVec_H, ivs[0], 
                                            inputColLastElemF32, inputRowLastElemF32, c0F32);

    std::vector<Value> indexWeights_temp = extractIndices(builder, loc, xVecWeight, yVecWeight, ivs[0], 
                                            inputColLastElemF32, inputRowLastElemF32, c0F32);

    std::vector<Value> indexWeights = {indexToF32(builder, loc, indexWeights_temp[0]), 
                                       indexToF32(builder, loc, indexWeights_temp[1])};

    Value indexWeights_0_temp = builder.create<arith::SubFOp>(loc, c1F32, indexWeights[0]);
    Value indexWeights_1_temp = builder.create<arith::SubFOp>(loc, c1F32, indexWeights[1]);

    Value pixelVal_a = builder.create<memref::LoadOp>(
            loc, builder.getF32Type(), input, ValueRange{inputIndices_L[0], inputIndices_L[1]});
    Value pixelVal_b = builder.create<memref::LoadOp>(
            loc, builder.getF32Type(), input, ValueRange{inputIndices_H[0], inputIndices_L[1]});
    Value pixelVal_c = builder.create<memref::LoadOp>(
            loc, builder.getF32Type(), input, ValueRange{inputIndices_L[0], inputIndices_H[1]});
    Value pixelVal_d = builder.create<memref::LoadOp>(
            loc, builder.getF32Type(), input, ValueRange{inputIndices_H[0], inputIndices_H[1]});

    Value weightVal1 = builder.create<arith::MulFOp>(loc, indexWeights_0_temp, indexWeights_1_temp);
    Value weightVal2 = builder.create<arith::MulFOp>(loc, indexWeights[0], indexWeights_1_temp);
    Value weightVal3 = builder.create<arith::MulFOp>(loc, indexWeights[1], indexWeights_0_temp);
    Value weightVal4 = builder.create<arith::MulFOp>(loc, indexWeights[0], indexWeights[1]);

    Value interm1 = builder.create<arith::MulFOp>(loc, pixelVal_a, weightVal1);
    Value interm2 = builder.create<arith::MulFOp>(loc, pixelVal_b, weightVal2);
    Value interm3 = builder.create<arith::MulFOp>(loc, pixelVal_c, weightVal3);
    Value interm4 = builder.create<arith::MulFOp>(loc, pixelVal_d, weightVal4);

    Value pixel_interm1 = builder.create<arith::AddFOp>(loc, interm1, interm2);
    Value pixel_interm2 = builder.create<arith::AddFOp>(loc, interm3, interm4);
    Value pixel_interm3 = builder.create<arith::AddFOp>(loc, pixel_interm1, pixel_interm2);

    Value pixelVal = roundOff(builder, loc, pixel_interm3);

    builder.create<memref::StoreOp>(loc, pixelVal, output,
                                        ValueRange{resIndices[0], resIndices[1]});

    builder.create<AffineYieldOp>(loc);
  });
}

void BillinearInterpolationResizing(
  OpBuilder &builder, Location loc, MLIRContext *ctx, 
  SmallVector<Value, 8> lowerBounds, SmallVector<Value, 8> upperBounds, 
  SmallVector<int64_t, 8> steps, Value strideVal, Value input, Value output, 
  Value horizontalScalingFactorVec, Value verticalScalingFactorVec, 
  Value outputRowLastElemF32, Value outputColLastElemF32, Value inputRowLastElemF32, 
  Value inputColLastElemF32, VectorType vectorTy32, 
  int64_t stride, Value c0, Value c0F32) {
  buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
    Value ivs0F32 = indexToF32(builder, loc, ivs[0]);
    Value yVec = builder.create<vector::SplatOp>(loc, vectorTy32, ivs0F32);
    Value xVec = iotaVec(builder, loc, ctx, ivs[1], strideVal,
                         vectorTy32, c0, stride);

    Value c1 = builder.create<ConstantIndexOp>(loc, 1);
    Value c1F32 = indexToF32(builder, loc, c1);

    Value xVecInterm = builder.create<arith::MulFOp>(loc, xVec, horizontalScalingFactorVec);
    Value yVecInterm = builder.create<arith::MulFOp>(loc, yVec, verticalScalingFactorVec);

    Value xVecInterm_L = builder.create<math::FloorOp>(loc, xVecInterm);
    Value xVecInterm_H = builder.create<math::CeilOp>(loc, xVecInterm);

    Value yVecInterm_L = builder.create<math::FloorOp>(loc, yVecInterm);
    Value yVecInterm_H = builder.create<math::CeilOp>(loc, yVecInterm);

    Value xVecWeight = builder.create<arith::SubFOp>(loc, xVecInterm, xVecInterm_L);
    Value yVecWeight = builder.create<arith::SubFOp>(loc, yVecInterm, yVecInterm_L);

    fillPixelsBillinearInterpolate(builder, loc, xVec, yVec, xVecInterm_L, yVecInterm_L, 
        xVecInterm_H, yVecInterm_H, input, output, c0, strideVal, outputRowLastElemF32, 
        xVecWeight, yVecWeight, outputColLastElemF32, inputRowLastElemF32, inputColLastElemF32, 
        c0F32, c1F32);
  });
}

class DIPResize2DOpLowering : public OpRewritePattern<dip::Resize2DOp> {
public:
  using OpRewritePattern<dip::Resize2DOp>::OpRewritePattern;

  explicit DIPResize2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
  }

  LogicalResult matchAndRewrite(dip::Resize2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value horizontalScalingFactor = op->getOperand(1);
    Value verticalScalingFactor = op->getOperand(2);
    Value output = op->getOperand(3);
    // auto interpolationAttr = op.interpolation_type();
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c0F32 = indexToF32(rewriter, loc, c0);

    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);

    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0); // check usability
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c1);

    // Determine lower bound for second call of rotation function (this is done
    // for efficient tail processing).
    Value outputColStrideRatio =
        rewriter.create<arith::DivUIOp>(loc, outputCol, strideVal);
    Value outputColMultiple =
        rewriter.create<arith::MulIOp>(loc, strideVal, outputColStrideRatio);

    SmallVector<Value, 8> lowerBounds1{c0, c0};
    SmallVector<Value, 8> upperBounds1{outputRow, outputColMultiple};

    SmallVector<int64_t, 8> steps{1, stride};
    Value strideTailVal =
        rewriter.create<arith::SubIOp>(loc, outputCol, outputColMultiple);

    SmallVector<Value, 8> lowerBounds2{c0, outputColMultiple};
    SmallVector<Value, 8> upperBounds2{outputRow, outputCol};

    FloatType f32 = FloatType::getF32(ctx);
    VectorType vectorTy32 = VectorType::get({stride}, f32);

    Value horizontalScalingFactorVec = rewriter.create<vector::SplatOp>(loc, vectorTy32, 
                                            horizontalScalingFactor);
    Value verticalScalingFactorVec = rewriter.create<vector::SplatOp>(loc, vectorTy32, 
                                            verticalScalingFactor);

    // Obtain extreme allocatable value(s) in input/output for bounding purpose.
    Value inputRowLastElem = rewriter.create<arith::SubIOp>(loc, inputRow, c1);
    Value inputRowLastElemF32 = indexToF32(rewriter, loc, inputRowLastElem);

    Value inputColLastElem = rewriter.create<arith::SubIOp>(loc, inputCol, c1);
    Value inputColLastElemF32 = indexToF32(rewriter, loc, inputColLastElem);

    Value outputRowLastElem =
        rewriter.create<arith::SubIOp>(loc, outputRow, c1);
    Value outputRowLastElemF32 = indexToF32(rewriter, loc, outputRowLastElem);

    Value outputColLastElem =
        rewriter.create<arith::SubIOp>(loc, outputCol, c1);
    Value outputColLastElemF32 = indexToF32(rewriter, loc, outputColLastElem);

    // NearestNeighbourInterpolationResizing(rewriter, loc, ctx, lowerBounds1, upperBounds1, 
    //                                       steps, strideVal, input, output, 
    //                                       horizontalScalingFactorVec, verticalScalingFactorVec, 
    //                                       outputRowLastElemF32, outputColLastElemF32, 
    //                                       inputRowLastElemF32, inputColLastElemF32, 
    //                                       vectorTy32, stride, c0, c0F32);

    // NearestNeighbourInterpolationResizing(rewriter, loc, ctx, lowerBounds2, upperBounds2, 
    //                                       steps, strideTailVal, input, output, 
    //                                       horizontalScalingFactorVec, verticalScalingFactorVec, 
    //                                       outputRowLastElemF32, outputColLastElemF32, 
    //                                       inputRowLastElemF32, inputColLastElemF32, 
    //                                       vectorTy32, stride, c0, c0F32);


    BillinearInterpolationResizing(rewriter, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal, 
                                   input, output, horizontalScalingFactorVec, verticalScalingFactorVec, 
                                   outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32, 
                                   inputColLastElemF32, vectorTy32, stride, c0, c0F32);

     BillinearInterpolationResizing(rewriter, loc, ctx, lowerBounds2, upperBounds2, steps, strideTailVal, 
                                   input, output, horizontalScalingFactorVec, verticalScalingFactorVec, 
                                   outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32, 
                                   inputColLastElemF32, vectorTy32, stride, c0, c0F32);


     // Remove the original resize operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

void populateLowerDIPConversionPatterns(RewritePatternSet &patterns,
                                        int64_t stride) {
  patterns.add<DIPCorr2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPRotate2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPResize2DOpLowering>(patterns.getContext(), stride);
}

//===----------------------------------------------------------------------===//
// LowerDIPPass
//===----------------------------------------------------------------------===//

namespace {
class LowerDIPPass : public PassWrapper<LowerDIPPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerDIPPass)
  LowerDIPPass() = default;
  LowerDIPPass(const LowerDIPPass &) {}
  explicit LowerDIPPass(int64_t strideParam) { stride = strideParam; }

  StringRef getArgument() const final { return "lower-dip"; }
  StringRef getDescription() const final { return "Lower DIP Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<buddy::dip::DIPDialect, func::FuncDialect,
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
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

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
