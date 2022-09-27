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
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"

#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"
#include "Utils/DIPUtils.h"
#include "Utils/Utils.h"
#include <vector>

#include <cmath>
#include <iostream>

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
    auto *ctx = op->getContext();

    // Register operand values.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    Value centerX = op->getOperand(3);
    Value centerY = op->getOperand(4);
    Value constantValue = op->getOperand(5);
    dip::BoundaryOption boundaryOptionAttr = op.getBoundaryOption();
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    auto inElemTy = input.getType().cast<MemRefType>().getElementType();
    DIP_ERROR error = checkDIPCommonTypes<dip::Corr2DOp>(op, input, kernel,
                                                         output, constantValue);

    if (error == DIP_ERROR::INCONSISTENT_INPUT_KERNEL_OUTPUT_TYPES) {
      return op->emitOpError() << "input, kernel, output and constant must "
                                  "have the same element type";
    } else if (error == DIP_ERROR::UNSUPPORTED_TYPE) {
      return op->emitOpError() << "supports only f32, f64 and integer types. "
                               << inElemTy << "is passed";
    }

    traverseImagewBoundaryExtrapolation(rewriter, loc, ctx, input, kernel,
                                        output, centerX, centerY, constantValue,
                                        strideVal, inElemTy, boundaryOptionAttr,
                                        stride, DIP_OP::CORRELATION_2D);
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

    // Obtain extreme allocatable value(s) in input and output for bounding
    // purpose.
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
    Value tanValAbs = rewriter.create<math::AbsFOp>(loc, tanVal);
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
              inputRowLastElemF32, inputColLastElemF32, c0, c0F32, c1F32Vec,
              vectorTy32, stride, f32);

          shearTransformController(
              builder, loc, ctx, lowerBounds2, upperBounds2, steps,
              strideTailVal, input, output, sinVec, tanVec, inputRowF32Vec,
              inputColF32Vec, inputCenterYF32Vec, inputCenterXF32Vec,
              outputCenterYF32Vec, outputCenterXF32Vec, outputRowLastElemF32,
              outputColLastElemF32, inputRowLastElemF32, inputColLastElemF32,
              c0, c0F32, c1F32Vec, vectorTy32, stride, f32);

          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          standardRotateController(
              builder, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
              input, output, sinVec, angleVal, inputRowF32Vec, inputColF32Vec,
              inputCenterYF32Vec, inputCenterXF32Vec, outputCenterYF32Vec,
              outputCenterXF32Vec, outputRowLastElemF32, outputColLastElemF32,
              inputRowLastElemF32, inputColLastElemF32, c0, c0F32, c1F32Vec,
              vectorTy32, stride, f32);

          standardRotateController(
              builder, loc, ctx, lowerBounds2, upperBounds2, steps,
              strideTailVal, input, output, sinVec, angleVal, inputRowF32Vec,
              inputColF32Vec, inputCenterYF32Vec, inputCenterXF32Vec,
              outputCenterYF32Vec, outputCenterXF32Vec, outputRowLastElemF32,
              outputColLastElemF32, inputRowLastElemF32, inputColLastElemF32,
              c0, c0F32, c1F32Vec, vectorTy32, stride, f32);

          builder.create<scf::YieldOp>(loc);
        });

    // Remove the origin rotation operation.
    rewriter.eraseOp(op);
    return success();
  }

  int64_t stride;
};

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
    auto interpolationAttr = op.getInterpolationType();
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c0F32 = indexToF32(rewriter, loc, c0);

    Value inputRow = rewriter.create<memref::DimOp>(loc, input, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, input, c1);

    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0);
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

    Value horizontalScalingFactorVec = rewriter.create<vector::SplatOp>(
        loc, vectorTy32, horizontalScalingFactor);
    Value verticalScalingFactorVec = rewriter.create<vector::SplatOp>(
        loc, vectorTy32, verticalScalingFactor);

    // Obtain extreme allocatable value(s) in input and output for bounding
    // purpose.
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

    if (interpolationAttr ==
        dip::InterpolationType::NearestNeighbourInterpolation) {
      NearestNeighbourInterpolationResizing(
          rewriter, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
          input, output, horizontalScalingFactorVec, verticalScalingFactorVec,
          outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
          inputColLastElemF32, vectorTy32, stride, c0, c0F32);

      NearestNeighbourInterpolationResizing(
          rewriter, loc, ctx, lowerBounds2, upperBounds2, steps, strideTailVal,
          input, output, horizontalScalingFactorVec, verticalScalingFactorVec,
          outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
          inputColLastElemF32, vectorTy32, stride, c0, c0F32);
    } else if (interpolationAttr ==
               dip::InterpolationType::BilinearInterpolation) {
      Value c1F32 = indexToF32(rewriter, loc, c1);

      BilinearInterpolationResizing(
          rewriter, loc, ctx, lowerBounds1, upperBounds1, steps, strideVal,
          input, output, horizontalScalingFactorVec, verticalScalingFactorVec,
          outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
          inputColLastElemF32, vectorTy32, stride, c0, c0F32, c1F32);

      BilinearInterpolationResizing(
          rewriter, loc, ctx, lowerBounds2, upperBounds2, steps, strideTailVal,
          input, output, horizontalScalingFactorVec, verticalScalingFactorVec,
          outputRowLastElemF32, outputColLastElemF32, inputRowLastElemF32,
          inputColLastElemF32, vectorTy32, stride, c0, c0F32, c1F32);
    }

    // Remove the original resize operation.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};

// Function for calculating complex addition of 2 input 1D complex vectors. 
// Separate vectors for real and imaginary parts are expected.
inline std::vector<Value> complexVecAddI(OpBuilder &builder, Location loc, 
                                 Value vec1Real, Value vec1Imag, Value vec2Real, Value vec2Imag)
{
  return {builder.create<arith::AddFOp>(loc, vec1Real, vec2Real),
          builder.create<arith::AddFOp>(loc, vec1Imag, vec2Imag)};
}

// Function for calculating complex subtraction of 2 input 1D complex vectors. 
// Separate vectors for real and imaginary parts are expected.
inline std::vector<Value> complexVecSubI(OpBuilder &builder, Location loc, 
                                 Value vec1Real, Value vec1Imag, Value vec2Real, Value vec2Imag)
{
  return {builder.create<arith::SubFOp>(loc, vec1Real, vec2Real),
          builder.create<arith::SubFOp>(loc, vec1Imag, vec2Imag)};
}

// Function for calculating complex product of 2 input 1D complex vectors. 
// Separate vectors for real and imaginary parts are expected.
std::vector<Value> complexVecMulI(OpBuilder &builder, Location loc, 
                                  Value vec1Real, Value vec1Imag, Value vec2Real, Value vec2Imag)
{
  Value int1 = builder.create<arith::MulFOp>(loc, vec1Real, vec2Real);
  Value int2 = builder.create<arith::MulFOp>(loc, vec1Imag, vec2Imag);
  Value int3 = builder.create<arith::MulFOp>(loc, vec1Real, vec2Imag);
  Value int4 = builder.create<arith::MulFOp>(loc, vec1Imag, vec2Real);

  return {builder.create<arith::SubFOp>(loc, int1, int2),
          builder.create<arith::AddFOp>(loc, int3, int4)};
}

// Function for calculating Transpose of 2D input MemRef.
void scalar2DMemRefTranspose(OpBuilder &builder, Location loc, Value memref1, Value memref2, 
    Value memref1NumRows, Value memref1NumCols,
    Value memref2NumRows, Value memref2NumCols, Value c0)
{
  SmallVector<Value, 8> lowerBounds(2, c0);
  SmallVector<Value, 8> upperBounds{memref1NumRows, memref1NumCols};
  SmallVector<int64_t, 8> steps(2, 1);

  buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value pixelVal = builder.create<memref::LoadOp>(
            loc, builder.getF32Type(), memref1,
            ValueRange{ivs[0], ivs[1]});

        builder.create<memref::StoreOp>(loc, pixelVal, memref2, ValueRange{ivs[1], ivs[0]});
  });
}

// Function for calculating Hadamard product of complex type 2D MemRefs. Separate MemRefs 
// for real and imaginary parts are expected.
void vector2DMemRefMultiply(OpBuilder &builder, Location loc, 
    Value memRef1Real, Value memRef1Imag, Value memRef2Real, Value memRef2Imag,
    Value memRef3Real, Value memRef3Imag, Value memRefNumRows, Value memRefNumCols, Value c0, VectorType vecType)
{
  SmallVector<Value, 8> lowerBounds(2, c0);
  SmallVector<Value, 8> upperBounds{memRefNumRows, memRefNumCols};
  SmallVector<int64_t, 8> steps(2, 1);

  buildAffineLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        Value pixelVal1Real = builder.create<LoadOp>(
            loc, vecType, memRef1Real,
            ValueRange{ivs[0], ivs[1]});
        Value pixelVal1Imag = builder.create<LoadOp>(
            loc, vecType, memRef1Imag,
            ValueRange{ivs[0], ivs[1]});

        Value pixelVal2Real = builder.create<LoadOp>(
            loc, vecType, memRef2Real,
            ValueRange{ivs[0], ivs[1]});
        Value pixelVal2Imag = builder.create<LoadOp>(
            loc, vecType, memRef2Imag,
            ValueRange{ivs[0], ivs[1]});

        std::vector<Value> resVecs = complexVecMulI(builder, loc, pixelVal1Real, pixelVal1Imag,
                                                    pixelVal2Real, pixelVal2Imag);

        builder.create<StoreOp>(loc, resVecs[0], memRef3Real, ValueRange{ivs[0], ivs[1]});
        builder.create<StoreOp>(loc, resVecs[1], memRef3Imag, ValueRange{ivs[0], ivs[1]});
  });
}

// Function for implementing Cooley Tukey Butterfly algortihm for calculating inverse of 
// discrete Fourier transform of invidiual 1D components of 2D input MemRef. 
// Separate MemRefs for real and imaginary parts are expected.
void idft1DCooleyTukeyButterfly(OpBuilder &builder, Location loc, Value memRefReal2D,
             Value memRefImag2D, Value memRefLength, Value strideVal, VectorType vecType,
             Value rowIndex, Value c0, Value c1, int64_t step)
{
  // Cooley Tukey Butterfly algorithm implementation.
  Value subProbs = builder.create<arith::ShRSIOp>(loc, memRefLength, c1);
  Value subProbSize, half = c1, i, jBegin, jEnd, j, angle;
  Value wStepReal, wStepImag, wReal, wImag, tmp1Real, tmp1Imag, tmp2Real, tmp2Imag;
  Value wRealVec, wImagVec, wStepRealVec, wStepImagVec;
  Value tmp2RealTemp, tmp2ImagTemp;

  Value upperBound = F32ToIndex(builder, loc,
                     builder.create<math::Log2Op>(loc, indexToF32(builder, loc, memRefLength)));
  Value pos2MPI = builder.create<ConstantFloatOp>(loc,
                  (llvm::APFloat)(float)(2.0*M_PI), builder.getF32Type());

  builder.create<scf::ForOp>(loc, c0, upperBound, c1, ValueRange{subProbs, half},
    [&](OpBuilder &builder, Location loc, ValueRange iv, ValueRange outerIterVR) {
      subProbSize = builder.create<arith::ShLIOp>(loc, outerIterVR[1], c1);
      angle = builder.create<arith::DivFOp>(loc, pos2MPI, indexToF32(builder, loc, subProbSize));

      wStepReal = builder.create<math::CosOp>(loc, angle);
      wStepRealVec = builder.create<vector::BroadcastOp>(loc, vecType, wStepReal);

      wStepImag = builder.create<math::SinOp>(loc, angle);
      wStepImagVec = builder.create<vector::BroadcastOp>(loc, vecType, wStepImag);

      builder.create<scf::ForOp>(loc, c0, outerIterVR[0], c1, ValueRange{}, 
        [&](OpBuilder &builder, Location loc, ValueRange iv1, ValueRange) {
          jBegin = builder.create<arith::MulIOp>(loc, iv1[0], subProbSize);
          jEnd = builder.create<arith::AddIOp>(loc, jBegin, outerIterVR[1]);
          wReal = builder.create<ConstantFloatOp>(loc, (llvm::APFloat)1.0f, builder.getF32Type());
          wImag = builder.create<ConstantFloatOp>(loc, (llvm::APFloat)0.0f, builder.getF32Type());

          wRealVec = builder.create<vector::BroadcastOp>(loc, vecType, wReal);
          wImagVec = builder.create<vector::BroadcastOp>(loc, vecType, wImag);

          // Vectorize stuff inside this loop (take care of tail processing as well)
          builder.create<scf::ForOp>(loc, jBegin, jEnd, strideVal, ValueRange{wRealVec, wImagVec}, 
            [&](OpBuilder &builder, Location loc, ValueRange iv2, 
                ValueRange wVR) {
              tmp1Real = builder.create<LoadOp>(loc, vecType, memRefReal2D, 
                                                ValueRange{rowIndex, iv2[0]});
              tmp1Imag = builder.create<LoadOp>(loc, vecType, memRefImag2D, 
                                                ValueRange{rowIndex, iv2[0]});

              Value secondIndex = builder.create<arith::AddIOp>(loc, iv2[0], outerIterVR[1]);
              tmp2RealTemp = builder.create<LoadOp>(loc, vecType, memRefReal2D, 
                                                    ValueRange{rowIndex, secondIndex});
              tmp2ImagTemp = builder.create<LoadOp>(loc, vecType, memRefImag2D, 
                                                    ValueRange{rowIndex, secondIndex});

              std::vector<Value> tmp2Vec = 
                  complexVecMulI(builder, loc, tmp2RealTemp, tmp2ImagTemp, wVR[0], wVR[1]);

              std::vector<Value> int1Vec = 
                  complexVecAddI(builder, loc, tmp1Real, tmp1Imag, tmp2Vec[0], tmp2Vec[1]);
              builder.create<StoreOp>(loc, int1Vec[0], memRefReal2D, ValueRange{rowIndex, iv2[0]});
              builder.create<StoreOp>(loc, int1Vec[1], memRefImag2D, ValueRange{rowIndex, iv2[0]});

              std::vector<Value> int2Vec = 
                  complexVecSubI(builder, loc, tmp1Real, tmp1Imag, tmp2Vec[0], tmp2Vec[1]);
              builder.create<StoreOp>(loc, int2Vec[0], memRefReal2D, 
                  ValueRange{rowIndex, secondIndex});
              builder.create<StoreOp>(loc, int2Vec[1], memRefImag2D, 
                  ValueRange{rowIndex, secondIndex});

              std::vector<Value> wUpdate = 
                  complexVecMulI(builder, loc, wVR[0], wVR[1], wStepRealVec, wStepImagVec);

              builder.create<scf::YieldOp>(loc, ValueRange{wUpdate[0], wUpdate[1]});
            });

          builder.create<scf::YieldOp>(loc);
        });
      Value updatedSubProbs = builder.create<arith::ShRSIOp>(loc, outerIterVR[0], c1);

      builder.create<scf::YieldOp>(loc, ValueRange{updatedSubProbs, subProbSize});
    }); 

    Value memRefLengthVec = builder.create<vector::BroadcastOp>(loc, vecType,
                            indexToF32(builder, loc, memRefLength));

    builder.create<scf::ForOp>(loc, c0, memRefLength, strideVal, ValueRange{}, 
      [&](OpBuilder &builder, Location loc, ValueRange iv, ValueRange) {
        Value tempVecReal = builder.create<LoadOp>(loc, vecType, memRefReal2D, 
                                               ValueRange{rowIndex, iv[0]});
        Value tempResVecReal = builder.create<arith::DivFOp>(loc, tempVecReal, memRefLengthVec);
        builder.create<StoreOp>(loc, tempResVecReal, memRefReal2D, ValueRange{rowIndex, iv[0]});

        Value tempVecImag = builder.create<LoadOp>(loc, vecType, memRefImag2D, 
                                               ValueRange{rowIndex, iv[0]});
        Value tempResVecImag = builder.create<arith::DivFOp>(loc, tempVecImag, memRefLengthVec);
        builder.create<StoreOp>(loc, tempResVecImag, memRefImag2D, ValueRange{rowIndex, iv[0]});

        builder.create<scf::YieldOp>(loc);
      });

}

// Function for implementing Gentleman Sande Butterfly algortihm for calculating discrete 
// Fourier transform of invidiual 1D components of 2D input MemRef. 
// Separate MemRefs for real and imaginary parts are expected.
void dft1DGentlemanSandeButterfly(OpBuilder &builder, Location loc, Value memRefReal2D,
            Value memRefImag2D, Value memRefLength, Value strideVal, VectorType vecType,
            Value rowIndex, Value c0, Value c1, int64_t step)
{
  // Gentleman Sande Butterfly algorithm implementation.
  Value subProbs = c1, subProbSize = memRefLength, i, jBegin, jEnd, j, half, angle;
  Value wStepReal, wStepImag, wReal, wImag, tmp1Real, tmp1Imag, tmp2Real, tmp2Imag;
  Value wRealVec, wImagVec, wStepRealVec, wStepImagVec;

  Value upperBound = F32ToIndex(builder, loc,
                     builder.create<math::Log2Op>(loc, indexToF32(builder, loc, memRefLength)));
  Value neg2MPI = builder.create<ConstantFloatOp>(loc,
                  (llvm::APFloat)(float)(-2.0*M_PI), builder.getF32Type());

  builder.create<scf::ForOp>(loc, c0, upperBound, c1, ValueRange{subProbs, subProbSize},
    [&](OpBuilder &builder, Location loc, ValueRange iv, ValueRange outerIterVR) {
      half = builder.create<arith::ShRSIOp>(loc, outerIterVR[1], c1);
      angle = builder.create<arith::DivFOp>(loc, neg2MPI, indexToF32(builder, loc, outerIterVR[1]));

      wStepReal = builder.create<math::CosOp>(loc, angle);
      wStepRealVec = builder.create<vector::BroadcastOp>(loc, vecType, wStepReal);

      wStepImag = builder.create<math::SinOp>(loc, angle);
      wStepImagVec = builder.create<vector::BroadcastOp>(loc, vecType, wStepImag);

      builder.create<scf::ForOp>(loc, c0, outerIterVR[0], c1, ValueRange{}, 
        [&](OpBuilder &builder, Location loc, ValueRange iv1, ValueRange) {
          jBegin = builder.create<arith::MulIOp>(loc, iv1[0], outerIterVR[1]);
          jEnd = builder.create<arith::AddIOp>(loc, jBegin, half);
          wReal = builder.create<ConstantFloatOp>(loc, (llvm::APFloat)1.0f, builder.getF32Type());
          wImag = builder.create<ConstantFloatOp>(loc, (llvm::APFloat)0.0f, builder.getF32Type());

          wRealVec = builder.create<vector::BroadcastOp>(loc, vecType, wReal);
          wImagVec = builder.create<vector::BroadcastOp>(loc, vecType, wImag);

          // Vectorize stuff inside this loop (take care of tail processing as well)
          builder.create<scf::ForOp>(loc, jBegin, jEnd, strideVal, ValueRange{wRealVec, wImagVec}, 
            [&](OpBuilder &builder, Location loc, ValueRange iv2, 
                ValueRange wVR) {
              tmp1Real = builder.create<LoadOp>(loc, vecType, memRefReal2D, 
                                                ValueRange{rowIndex, iv2[0]});
              tmp1Imag = builder.create<LoadOp>(loc, vecType, memRefImag2D, 
                                                ValueRange{rowIndex, iv2[0]});

              Value secondIndex = builder.create<arith::AddIOp>(loc, iv2[0], half);
              tmp2Real = builder.create<LoadOp>(loc, vecType, memRefReal2D, 
                                                ValueRange{rowIndex, secondIndex});
              tmp2Imag = builder.create<LoadOp>(loc, vecType, memRefImag2D, 
                                                ValueRange{rowIndex, secondIndex});

              std::vector<Value> int1Vec = 
                  complexVecAddI(builder, loc, tmp1Real, tmp1Imag, tmp2Real, tmp2Imag);
              builder.create<StoreOp>(loc, int1Vec[0], memRefReal2D, ValueRange{rowIndex, iv2[0]});
              builder.create<StoreOp>(loc, int1Vec[1], memRefImag2D, ValueRange{rowIndex, iv2[0]});

              std::vector<Value> int2Vec = 
                  complexVecSubI(builder, loc, tmp1Real, tmp1Imag, tmp2Real, tmp2Imag);
              std::vector<Value> int3Vec = 
                  complexVecMulI(builder, loc, int2Vec[0], int2Vec[1], wVR[0], wVR[1]);

              builder.create<StoreOp>(loc, int3Vec[0], memRefReal2D, 
                  ValueRange{rowIndex, secondIndex});
              builder.create<StoreOp>(loc, int3Vec[1], memRefImag2D, 
                  ValueRange{rowIndex, secondIndex});

              std::vector<Value> wUpdate = 
                  complexVecMulI(builder, loc, wVR[0], wVR[1], wStepRealVec, wStepImagVec);

              builder.create<scf::YieldOp>(loc, ValueRange{wUpdate[0], wUpdate[1]});
            });

          builder.create<scf::YieldOp>(loc);
        });
      Value updatedSubProbs = builder.create<arith::ShLIOp>(loc, outerIterVR[0], c1);

      builder.create<scf::YieldOp>(loc, ValueRange{updatedSubProbs, half});
    }); 
}

// Function for applying inverse of discrete fourier transform on a 2D MemRef.
// Separate MemRefs for real and imaginary parts are expected.
void idft2D(OpBuilder &builder, Location loc, Value container2DReal,
             Value container2DImag, Value container2DRows, Value container2DCols,
             Value intermediateReal, Value intermediateImag, Value c0, Value c1,
             Value strideVal, VectorType vecType)
{
  builder.create<AffineForOp>(
        loc, ValueRange{c0}, builder.getDimIdentityMap(),
        ValueRange{container2DRows}, builder.getDimIdentityMap(), 1, llvm::None,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv, ValueRange itrArg) {
          idft1DCooleyTukeyButterfly(builder, loc, container2DReal, container2DImag,
                                     container2DCols, strideVal, vecType, iv, c0, c1, 1);

          nestedBuilder.create<AffineYieldOp>(nestedLoc);
    });

  scalar2DMemRefTranspose(builder, loc, container2DReal, intermediateReal, container2DRows, container2DCols,
                          container2DCols, container2DRows, c0);
  scalar2DMemRefTranspose(builder, loc, container2DImag, intermediateImag, container2DRows, container2DCols,
                          container2DCols, container2DRows, c0);

  builder.create<AffineForOp>(
        loc, ValueRange{c0}, builder.getDimIdentityMap(),
        ValueRange{container2DCols}, builder.getDimIdentityMap(), 1, llvm::None,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv, ValueRange itrArg) {
          idft1DCooleyTukeyButterfly(builder, loc, intermediateReal, intermediateImag,
                                     container2DRows, strideVal, vecType, iv, c0, c1, 1);

          nestedBuilder.create<AffineYieldOp>(nestedLoc);
    });

    Value transposeCond = builder.create<CmpIOp>(loc, CmpIPredicate::ne, container2DRows, container2DCols);
    builder.create<scf::IfOp>(loc, transposeCond, 
      [&](OpBuilder &builder, Location loc) {
        scalar2DMemRefTranspose(builder, loc, intermediateReal, container2DReal, container2DCols, container2DRows,
                            container2DRows, container2DCols, c0);
        scalar2DMemRefTranspose(builder, loc, intermediateImag, container2DImag, container2DCols, container2DRows,
                            container2DRows, container2DCols, c0);

        builder.create<scf::YieldOp>(loc);
      }, [&](OpBuilder &builder, Location loc) {
        builder.create<memref::CopyOp>(loc, intermediateReal, container2DReal);
        builder.create<memref::CopyOp>(loc, intermediateImag, container2DImag);

        builder.create<scf::YieldOp>(loc);
      });
}

// Function for applying discrete fourier transform on a 2D MemRef. Separate MemRefs 
// for real and imaginary parts are expected.
void dft2D(OpBuilder &builder, Location loc, Value container2DReal,
            Value container2DImag, Value container2DRows, Value container2DCols,
            Value intermediateReal, Value intermediateImag, Value c0, Value c1,
            Value strideVal, VectorType vecType)
{
  builder.create<AffineForOp>(
        loc, ValueRange{c0}, builder.getDimIdentityMap(),
        ValueRange{container2DRows}, builder.getDimIdentityMap(), 1, llvm::None,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv, ValueRange itrArg) {
          dft1DGentlemanSandeButterfly(builder, loc, container2DReal, container2DImag,
                                       container2DCols, strideVal, vecType, iv, c0, c1, 1);

          nestedBuilder.create<AffineYieldOp>(nestedLoc);
    });

  scalar2DMemRefTranspose(builder, loc, container2DReal, intermediateReal, container2DRows, container2DCols,
                          container2DCols, container2DRows, c0);
  scalar2DMemRefTranspose(builder, loc, container2DImag, intermediateImag, container2DRows, container2DCols,
                          container2DCols, container2DRows, c0);

  builder.create<AffineForOp>(
        loc, ValueRange{c0}, builder.getDimIdentityMap(),
        ValueRange{container2DCols}, builder.getDimIdentityMap(), 1, llvm::None,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv, ValueRange itrArg) {
          dft1DGentlemanSandeButterfly(builder, loc, intermediateReal, intermediateImag,
                                       container2DRows, strideVal, vecType, iv, c0, c1, 1);

          nestedBuilder.create<AffineYieldOp>(nestedLoc);
    });

    Value transposeCond = builder.create<CmpIOp>(loc, CmpIPredicate::ne, container2DRows, container2DCols);
    builder.create<scf::IfOp>(loc, transposeCond, 
      [&](OpBuilder &builder, Location loc) {
        scalar2DMemRefTranspose(builder, loc, intermediateReal, container2DReal, container2DCols, container2DRows,
                            container2DRows, container2DCols, c0);
        scalar2DMemRefTranspose(builder, loc, intermediateImag, container2DImag, container2DCols, container2DRows,
                            container2DRows, container2DCols, c0);

        builder.create<scf::YieldOp>(loc);
      }, [&](OpBuilder &builder, Location loc) {
        builder.create<memref::CopyOp>(loc, intermediateReal, container2DReal);
        builder.create<memref::CopyOp>(loc, intermediateImag, container2DImag);

        builder.create<scf::YieldOp>(loc);
      });
}

class DIPCorrFFT2DOpLowering : public OpRewritePattern<dip::CorrFFT2DOp> {
public:
  using OpRewritePattern<dip::CorrFFT2DOp>::OpRewritePattern;

  explicit DIPCorrFFT2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    // stride = strideParam;
    stride = 1;
    // stride = 2;
  }

  LogicalResult matchAndRewrite(dip::CorrFFT2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Create constant indices.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);

    // Register operand values.
    Value inputReal = op->getOperand(0);
    Value inputImag = op->getOperand(1);
    Value kernelReal = op->getOperand(2);
    Value kernelImag = op->getOperand(3);
    Value intermediateReal = op->getOperand(4);
    Value intermediateImag = op->getOperand(5);
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    // Create DimOp for padded input image.
    Value inputRow = rewriter.create<memref::DimOp>(loc, inputReal, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, inputReal, c1);

    // Create DimOp for padded original kernel.
    Value kernelRow = rewriter.create<memref::DimOp>(loc, kernelReal, c0);
    Value kernelCol = rewriter.create<memref::DimOp>(loc, kernelReal, c1);

    FloatType f32 = FloatType::getF32(ctx);
    VectorType vectorTy32 = VectorType::get({stride}, f32);

    dft2D(rewriter, loc, inputReal, inputImag, inputRow, inputCol, intermediateReal,
          intermediateImag, c0, c1, strideVal, vectorTy32);

    dft2D(rewriter, loc, kernelReal, kernelImag, kernelRow, kernelCol, intermediateReal,
          intermediateImag, c0, c1, strideVal, vectorTy32);

    vector2DMemRefMultiply(rewriter, loc, inputReal, inputImag, kernelReal, kernelImag,
                           inputReal, inputImag, inputRow, inputCol, c0, vectorTy32);

    idft2D(rewriter, loc, inputReal, inputImag, inputRow, inputCol, intermediateReal,
           intermediateImag, c0, c1, strideVal, vectorTy32);

    // Remove the origin convolution operation involving FFT.
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stride;
};
} // end anonymous namespace

void populateLowerDIPConversionPatterns(RewritePatternSet &patterns,
                                        int64_t stride) {
  patterns.add<DIPCorr2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPRotate2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPResize2DOpLowering>(patterns.getContext(), stride);
  patterns.add<DIPCorrFFT2DOpLowering>(patterns.getContext(), stride);
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
                AffineDialect, arith::ArithmeticDialect, math::MathDialect,
                complex::ComplexDialect>();
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
                         arith::ArithmeticDialect, math::MathDialect,
                         complex::ComplexDialect>();
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
