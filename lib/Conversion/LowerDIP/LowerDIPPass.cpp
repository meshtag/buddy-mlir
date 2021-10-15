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

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"

#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"

#include <iostream>

using namespace mlir;
using namespace Buddy;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class DIPCorr2DLowering : public ConversionPattern {
public:
  explicit DIPCorr2DLowering(MLIRContext *context)
      : ConversionPattern(DIP::Corr2DOp::getOperationName(), 1, context) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();
    // Create constant index.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    // Get input, kernel and output.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);

    // Create DimOp.
    Value kernelRow = rewriter.create<memref::DimOp>(loc, kernel, c0);
    Value kernelCol = rewriter.create<memref::DimOp>(loc, kernel, c1);
    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c1);
    // Size of strip mining.
    AffineExpr d0;
    bindDims(ctx, d0);

    AffineMap stripMap = AffineMap::get(1, 0, {d0.ceilDiv(5)}, ctx);
    SmallVector<Value, 8> lowerBounds(3, c0);
    SmallVector<Value, 8> uperBounds{outputRow, kernelRow, kernelCol};
    SmallVector<int64_t, 8> steps(3, /*Value=*/1);

    buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Create strip mining loop.
          builder.create<AffineForOp>(
              loc, ValueRange{c0}, builder.getDimIdentityMap(),
              ValueRange{outputCol}, stripMap, /*Step=*/1, llvm::None,
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                  ValueRange itrArgs) {

                nestedBuilder.create<PrintOp>(nestedLoc, iv);

                // Vectorize the kernel.
                // Define `*Type`.
                FloatType f32 = mlir::FloatType::getF32(ctx);
                VectorType vectorTy1 = mlir::VectorType::get({1}, f32);
                VectorType vectorTy32 = mlir::VectorType::get({5}, f32);
                // Broadcast element of the kernel.
                Value kernelValue = builder.create<AffineVectorLoadOp>(
                    loc, vectorTy1, kernel, ValueRange{ivs[1], ivs[2]});
                Value kernelVector =
                    builder.create<BroadcastOp>(loc, vectorTy32, kernelValue);
                // Load input vector from memref.
                AffineExpr m, n, k, j;
                bindDims(ctx, m, n, k, j);
                AffineMap inputVectorMap = AffineMap::get(
                    /*dimCount=*/4, /*symbolCount=*/0, {m + n, k + j * 5},
                    ctx);
                Value inputVector = nestedBuilder.create<AffineVectorLoadOp>(
                    loc, vectorTy32, input, inputVectorMap,
                    ValueRange{ivs[0], ivs[1], ivs[2], iv});
                // Define AffineMap.
                // The `outputVector` and `resultVector` share the same
                // AffineMap.
                AffineExpr x, y;
                bindDims(ctx, x, y);
                AffineMap outputVectorMap = AffineMap::get(
                    /*dimCount=*/2, /*symbolCount=*/0, {x, y * 5}, ctx);
                Value outputVector = nestedBuilder.create<AffineVectorLoadOp>(
                    loc, vectorTy32, output, outputVectorMap,
                    ValueRange{ivs[0], iv});
                // FMA = Fused Multiply + Add
                Value resultVector = nestedBuilder.create<FMAOp>(
                    loc, inputVector, kernelVector, outputVector);
                nestedBuilder.create<AffineVectorStoreOp>(
                    loc, resultVector, output, outputVectorMap,
                    ValueRange{ivs[0], iv});
                nestedBuilder.create<AffineYieldOp>(nestedLoc);
              });
        });
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }
};
} // end anonymous namespace

void populateLowerDIPConversionPatterns(RewritePatternSet &patterns)
{
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
    registry.insert<Buddy::DIP::DIPDialect, StandardOpsDialect, memref::MemRefDialect, 
                      scf::SCFDialect, VectorDialect>();
  }
  // void getDependentDialects(DialectRegistry &registry) const override {
  //   registry.insert<scf::SCFDialect, AffineDialect,
  //                   VectorDialect, StandardOpsDialect>();
  // }
};
} // end anonymous namespace.

void LowerDIPPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect,
                         scf::SCFDialect, StandardOpsDialect,
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
