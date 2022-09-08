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
    auto kElemTy = kernel.getType().cast<MemRefType>().getElementType();
    auto outElemTy = output.getType().cast<MemRefType>().getElementType();
    auto constElemTy = constantValue.getType();
    if (inElemTy != kElemTy || kElemTy != outElemTy ||
        outElemTy != constElemTy) {
      return op->emitOpError() << "input, kernel, output and constant must "
                                  "have the same element type";
    }
    // NB: we can infer element type for all operation to be the same as input
    // since we verified that the operand types are the same
    auto elemTy = inElemTy;
    auto bitWidth = elemTy.getIntOrFloatBitWidth();
    if (!elemTy.isF64() && !elemTy.isF32() && !elemTy.isInteger(bitWidth)) {
      return op->emitOpError() << "supports only f32, f64 and integer types. "
                               << elemTy << "is passed";
    }

    traverseImagewBoundaryExtrapolation(rewriter, loc, ctx, input, kernel,
                                        output, centerX, centerY, constantValue,
                                        strideVal, elemTy, boundaryOptionAttr,
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

inline std::vector<Value> complexVecAddI(OpBuilder &builder, Location loc, 
                                 Value vec1Real, Value vec1Imag, Value vec2Real, Value vec2Imag)
{
  return {builder.create<arith::AddIOp>(loc, vec1Real, vec2Real),
          builder.create<arith::AddIOp>(loc, vec1Imag, vec2Imag)};
}

inline std::vector<Value> complexVecSubI(OpBuilder &builder, Location loc, 
                                 Value vec1Real, Value vec1Imag, Value vec2Real, Value vec2Imag)
{
  return {builder.create<arith::SubIOp>(loc, vec1Real, vec2Real),
          builder.create<arith::SubIOp>(loc, vec1Imag, vec2Imag)};
}

std::vector<Value> complexVecMulI(OpBuilder &builder, Location loc, 
                                  Value vec1Real, Value vec1Imag, Value vec2Real, Value vec2Imag)
{
  // (ac - bd) + i(ad + bc)
  Value int1 = builder.create<arith::MulIOp>(loc, vec1Real, vec2Real);
  Value int2 = builder.create<arith::MulIOp>(loc, vec1Imag, vec2Imag);
  Value int3 = builder.create<arith::MulIOp>(loc, vec1Real, vec2Imag);
  Value int4 = builder.create<arith::MulIOp>(loc, vec1Imag, vec2Real);

  return {builder.create<arith::SubIOp>(loc, int1, int2),
          builder.create<arith::AddIOp>(loc, int3, int4)};
}

void fft_1d(OpBuilder &builder, Location loc, Value memRefReal,
            Value memRefImag, Value resMemRefReal, Value resMemRefImag, Value lowerBound,
            Value MemRefLength, Value strideVal, VectorType vecType, Value c1, int64_t step)
{
  Value subProbs = c1, subProbSize = MemRefLength, i, jBegin, jEnd, j, half, angle;
  Value wStepReal, wStepImag, wReal, wImag, tmp1Real, tmp1Imag, tmp2Real, tmp2Imag;

  Value upperBound = F32ToIndex(builder, loc,
                     builder.create<math::Log2Op>(loc, indexToF32(builder, loc, MemRefLength)));
  Value neg2MPI = builder.create<ConstantFloatOp>(loc,
                  (llvm::APFloat)(float)(-2.0*M_PI), builder.getF32Type());

  builder.create<scf::ForOp>(loc, lowerBound, upperBound, c1, ValueRange{}, 
    [&](OpBuilder &builder, Location loc, ValueRange iv, ValueRange) {
      half = builder.create<arith::ShRSIOp>(loc, subProbSize, c1);
      angle = builder.create<arith::DivFOp>(loc, neg2MPI, indexToF32(builder, loc, subProbSize));
      wStepReal = builder.create<math::CosOp>(loc, angle);
      wStepImag = builder.create<math::SinOp>(loc, angle);

      builder.create<scf::ForOp>(loc, lowerBound, subProbs, c1, ValueRange{}, 
        [&](OpBuilder &builder, Location loc, ValueRange iv1, ValueRange) {
          jBegin = builder.create<arith::MulIOp>(loc, iv1[0], subProbSize);
          jEnd = builder.create<arith::AddIOp>(loc, jBegin, half);
          wReal = builder.create<ConstantFloatOp>(loc, (llvm::APFloat)1.0f, builder.getF32Type());
          wImag = builder.create<ConstantFloatOp>(loc, (llvm::APFloat)0.0f, builder.getF32Type());

          // Vectorize stuff inside this loop (take care of tail processing as well)
          builder.create<scf::ForOp>(loc, jBegin, jEnd, strideVal, ValueRange{}, 
            [&](OpBuilder &builder, Location loc, ValueRange iv2, ValueRange) {
        //       tmp1 = X[j];
				// tmp2 = X[j+half];
				// X[j] = tmp1+tmp2;
				// X[j+half] = (tmp1-tmp2)*w;
				// w *= w_step;

              // tmp1Real = builder.create<LoadOp>(loc, vecType, memRefReal, 
              //                                   ValueRange{lowerBound, lowerBound});
              // tmp1Imag = builder.create<LoadOp>(loc, vecType, memRefImag, 
              //                                   ValueRange{lowerBound, iv2[0]});

              // Value secondIndex = builder.create<arith::AddIOp>(loc, iv2[0], half);
              // tmp2Real = builder.create<LoadOp>(loc, vecType, memRefReal, 
              //                                   ValueRange{lowerBound, secondIndex});
              // tmp2Imag = builder.create<LoadOp>(loc, vecType, memRefImag, 
              //                                   ValueRange{lowerBound, secondIndex});

              // std::vector<Value> int1Vec = 
              //     complexVecAddI(builder, loc, tmp1Real, tmp1Imag, tmp2Real, tmp2Imag);
              // builder.create<StoreOp>(loc, int1Vec[0], memRefReal, ValueRange{lowerBound, iv2[0]});
              // builder.create<StoreOp>(loc, int1Vec[1], memRefImag, ValueRange{lowerBound, iv2[0]});

              // std::vector<Value> int2Vec = 
              //     complexVecSubI(builder, loc, tmp1Real, tmp1Imag, tmp2Real, tmp2Imag);
              // std::vector<Value> int3Vec = 
              //     complexVecMulI(builder, loc, int2Vec[0], int2Vec[1], wReal, wImag);
              // builder.create<StoreOp>(loc, int3Vec[0], memRefReal, 
              //     ValueRange{lowerBound, secondIndex});
              // builder.create<StoreOp>(loc, int3Vec[1], memRefImag, 
              //     ValueRange{lowerBound, secondIndex});

              builder.create<scf::YieldOp>(loc);
            });

          builder.create<scf::YieldOp>(loc);
        });

      builder.create<scf::YieldOp>(loc);
    }); 

}

void dft_2d(OpBuilder &builder, Location loc, Value container2DReal,
            Value container2DImag, Value container2DRows, Value container2DCols,
            Value intermediateReal, Value intermediateImag, Value c0, Value c1,
            Value strideVal, VectorType vecType)
{

  builder.create<memref::CopyOp>(loc, container2DReal, intermediateReal);

  builder.create<AffineForOp>(
        loc, ValueRange{c0}, builder.getDimIdentityMap(),
        ValueRange{container2DRows}, builder.getDimIdentityMap(), 1, llvm::None,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv, ValueRange itrArg) {
        Value origSubMemRefReal = builder.create<memref::SubViewOp>(loc, container2DReal, 
          ValueRange{iv, c0}, ValueRange{container2DRows, c1}, ValueRange{c1, c1});
        Value origSubMemRefImag = builder.create<memref::SubViewOp>(loc, container2DImag, 
                          ValueRange{iv, c0}, ValueRange{c1, container2DCols}, ValueRange{c1, c1});
        Value intermediateSubMemRefReal = builder.create<memref::SubViewOp>(loc, intermediateReal,
                          ValueRange{iv, c0}, ValueRange{c1, container2DCols}, ValueRange{c1, c1});
        Value intermediateSubMemRefImag = builder.create<memref::SubViewOp>(loc, intermediateImag,
                          ValueRange{iv, c0}, ValueRange{c1, container2DCols}, ValueRange{c1, c1});

        // builder.create<vector::PrintOp>(loc, c1);
        // MemRefType checkTy = MemRefType::get(container2DCols, builder.getF32Type());
        // builder.create<vector::PrintOp>(loc, container2DReal.getType().getShape());

        fft_1d(builder, loc, origSubMemRefReal, origSubMemRefImag, intermediateSubMemRefReal,
               intermediateSubMemRefImag, c0, container2DCols, strideVal, vecType, c1, 1);

        nestedBuilder.create<AffineYieldOp>(nestedLoc);
    });

}

class DIPCorrFFT2DOpLowering : public OpRewritePattern<dip::CorrFFT2DOp> {
public:
  using OpRewritePattern<dip::CorrFFT2DOp>::OpRewritePattern;

  explicit DIPCorrFFT2DOpLowering(MLIRContext *context, int64_t strideParam)
      : OpRewritePattern(context) {
    stride = strideParam;
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
    Value outputReal = op->getOperand(4);
    Value outputImag = op->getOperand(5);
    Value intermediateReal = op->getOperand(6);
    Value intermediateImag = op->getOperand(7);
    Value centerX = op->getOperand(8);
    Value centerY = op->getOperand(9);
    Value constantValue = op->getOperand(10);
    Value strideVal = rewriter.create<ConstantIndexOp>(loc, stride);

    // Create DimOp.
    Value inputRow = rewriter.create<memref::DimOp>(loc, inputReal, c0);
    Value inputCol = rewriter.create<memref::DimOp>(loc, inputReal, c1);

    FloatType f32 = FloatType::getF32(ctx);
    VectorType vectorTy32 = VectorType::get({stride}, f32);

    dft_2d(rewriter, loc, inputReal, inputImag, inputRow, inputCol, intermediateReal,
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
