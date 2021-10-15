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

#include "DIP/DIPDialect.h"
#include "DIP/DIPOps.h"

#include <iostream>

using namespace mlir;
using namespace Buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class DIPCorr2DLowering : public OpRewritePattern<DIP::Corr2DOp> {
public:
  using OpRewritePattern<DIP::Corr2DOp>::OpRewritePattern;

  explicit DIPCorr2DLowering(MLIRContext *context , std::ptrdiff_t centerX, 
              std::ptrdiff_t centerY, unsigned int boundaryOption) : OpRewritePattern(context)
  {
    this->centerX = centerX;
    this->centerY = centerY;
    this->boundaryOption = boundaryOption;
    std::cout << this->centerX << "\n";
    std::cout << this->centerY << "\n";
    std::cout << this->boundaryOption << "\n";
    std::cout << "Here\n";
  }

  LogicalResult matchAndRewrite(DIP::Corr2DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    return success();
  }

private:
  std::ptrdiff_t centerX, centerY;
  unsigned int boundaryOption = 0;
};
} // end anonymous namespace

void populateLowerDIPConversionPatterns(RewritePatternSet &patterns, std::ptrdiff_t centerX, 
                                        std::ptrdiff_t centerY, unsigned int boundaryOption)
{
  patterns.add<DIPCorr2DLowering>(patterns.getContext(), centerX, centerY, boundaryOption);
}

//===----------------------------------------------------------------------===//
// LowerDIPPass
//===----------------------------------------------------------------------===//

namespace {
class LowerDIPPass : public PassWrapper<LowerDIPPass, OperationPass<ModuleOp>> {
public:
  LowerDIPPass() = default;
  LowerDIPPass(const LowerDIPPass &) {}
  explicit LowerDIPPass(std::ptrdiff_t centerXParam, std::ptrdiff_t centerYParam,
                        unsigned int boundaryOptionParam = 0)
  { 
    centerX = centerXParam;
    centerY = centerYParam;
    boundaryOption = boundaryOptionParam;
  }

  StringRef getArgument() const final { return "lower-DIP"; }
  StringRef getDescription() const final { return "Lower DIP Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Buddy::DIP::DIPDialect, StandardOpsDialect>();
  }

  Option<std::ptrdiff_t> centerX{*this, "centerX",
                         llvm::cl::desc("X co-ordinate of anchor point"),
                         llvm::cl::init(32)};
  Option<std::ptrdiff_t> centerY{*this, "centerY",
                         llvm::cl::desc("Y co-ordinate of anchor point"),
                         llvm::cl::init(32)};
  Option<unsigned int> boundaryOption{*this, "boundaryOption",
                         llvm::cl::desc("Method for boundary extrapolation"),
                         llvm::cl::init(32)};
};
} // end anonymous namespace.

void LowerDIPPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ReturnOp>();

  RewritePatternSet patterns(context);
  populateLowerDIPConversionPatterns(patterns, centerX, centerY, boundaryOption);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace Buddy {
void registerLowerDIPPass() { PassRegistration<LowerDIPPass>(); }
} // namespace Buddy
} // namespace mlir
