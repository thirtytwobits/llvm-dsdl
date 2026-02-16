#include "llvmdsdl/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace llvmdsdl {

namespace {

std::int64_t nonNegative(const std::int64_t value) {
  return std::max<std::int64_t>(value, 0);
}

std::int64_t intAttrOrDefault(mlir::Operation *op, llvm::StringRef name,
                              const std::int64_t fallback) {
  if (const auto attr = op->getAttrOfType<mlir::IntegerAttr>(name)) {
    return attr.getInt();
  }
  return fallback;
}

void setI64Attr(mlir::Operation *op, llvm::StringRef name,
                const std::int64_t value, mlir::Builder &builder) {
  op->setAttr(name, builder.getI64IntegerAttr(value));
}

void setI32Attr(mlir::Operation *op, llvm::StringRef name,
                const std::int64_t value, mlir::Builder &builder) {
  op->setAttr(name, builder.getI32IntegerAttr(static_cast<std::int32_t>(value)));
}

mlir::LogicalResult canonicalizePlan(mlir::Operation *plan, mlir::Builder &builder) {
  if (plan->getNumRegions() == 0 || plan->getRegion(0).empty()) {
    return plan->emitOpError("must contain a non-empty body region");
  }

  auto &body = plan->getRegion(0).front();
  std::vector<mlir::Operation *> eraseOps;
  std::int64_t stepIndex = 0;
  std::int64_t alignCount = 0;
  std::int64_t fieldCount = 0;
  std::int64_t paddingCount = 0;
  std::int64_t inferredUnionTagBits = 0;
  std::set<std::int64_t> unionOptionIndexes;

  for (mlir::Operation &op : body) {
    const auto opName = op.getName().getStringRef();
    if (opName == "dsdl.align") {
      const std::int64_t bits =
          nonNegative(intAttrOrDefault(&op, "bits", /*fallback=*/1));
      if (bits <= 1) {
        eraseOps.push_back(&op);
        continue;
      }
      setI32Attr(&op, "bits", bits, builder);
      setI64Attr(&op, "step_index", stepIndex++, builder);
      ++alignCount;
      continue;
    }

    if (opName == "dsdl.io") {
      const auto kindAttr = op.getAttrOfType<mlir::StringAttr>("kind");
      const std::string kind = kindAttr ? kindAttr.getValue().str() : "field";
      if (!kindAttr) {
        op.setAttr("kind", builder.getStringAttr(kind));
      }

      std::int64_t minBits =
          nonNegative(intAttrOrDefault(&op, "min_bits", /*fallback=*/0));
      std::int64_t maxBits = nonNegative(
          intAttrOrDefault(&op, "max_bits", /*fallback=*/minBits));
      if (maxBits < minBits) {
        maxBits = minBits;
      }

      if (kind == "padding") {
        ++paddingCount;
        if (maxBits == 0) {
          eraseOps.push_back(&op);
          continue;
        }
      }

      setI64Attr(&op, "min_bits", minBits, builder);
      setI64Attr(&op, "max_bits", maxBits, builder);
      setI64Attr(&op, "lowered_bits", maxBits, builder);
      setI64Attr(&op, "step_index", stepIndex++, builder);

      setI64Attr(&op, "bit_length",
                 nonNegative(intAttrOrDefault(&op, "bit_length", 0)), builder);
      setI64Attr(&op, "array_capacity",
                 nonNegative(intAttrOrDefault(&op, "array_capacity", 0)),
                 builder);
      setI64Attr(
          &op, "array_length_prefix_bits",
          nonNegative(intAttrOrDefault(&op, "array_length_prefix_bits", 0)),
          builder);
      setI64Attr(&op, "alignment_bits",
                 std::max<std::int64_t>(
                     nonNegative(intAttrOrDefault(&op, "alignment_bits", 1)), 1),
                 builder);
      setI64Attr(&op, "union_option_index",
                 nonNegative(intAttrOrDefault(&op, "union_option_index", 0)),
                 builder);
      setI64Attr(&op, "union_tag_bits",
                 nonNegative(intAttrOrDefault(&op, "union_tag_bits", 0)),
                 builder);

      inferredUnionTagBits = std::max<std::int64_t>(
          inferredUnionTagBits, intAttrOrDefault(&op, "union_tag_bits", 0));
      if (kind != "padding") {
        unionOptionIndexes.insert(intAttrOrDefault(&op, "union_option_index", 0));
      }

      if (kind != "padding") {
        ++fieldCount;
      }
      continue;
    }

    return op.emitError("unsupported operation in serialization plan body");
  }

  for (mlir::Operation *op : eraseOps) {
    op->erase();
  }

  std::int64_t minBits = nonNegative(intAttrOrDefault(plan, "min_bits", 0));
  std::int64_t maxBits =
      nonNegative(intAttrOrDefault(plan, "max_bits", minBits));
  if (maxBits < minBits) {
    maxBits = minBits;
  }
  setI64Attr(plan, "min_bits", minBits, builder);
  setI64Attr(plan, "max_bits", maxBits, builder);
  setI64Attr(plan, "lowered_min_bits", minBits, builder);
  setI64Attr(plan, "lowered_max_bits", maxBits, builder);
  setI64Attr(plan, "lowered_step_count", stepIndex, builder);
  setI64Attr(plan, "lowered_field_count", fieldCount, builder);
  setI64Attr(plan, "lowered_padding_count", paddingCount, builder);
  setI64Attr(plan, "lowered_align_count", alignCount, builder);
  plan->setAttr("lowered", builder.getUnitAttr());

  if (plan->hasAttr("is_union")) {
    std::int64_t unionTagBits =
        nonNegative(intAttrOrDefault(plan, "union_tag_bits", 0));
    if (unionTagBits <= 0) {
      unionTagBits = std::max<std::int64_t>(8, nonNegative(inferredUnionTagBits));
    }
    setI64Attr(plan, "union_tag_bits", unionTagBits, builder);

    std::int64_t optionCount = nonNegative(
        intAttrOrDefault(plan, "union_option_count", static_cast<std::int64_t>(
                                                      unionOptionIndexes.size())));
    if (!unionOptionIndexes.empty()) {
      optionCount = static_cast<std::int64_t>(unionOptionIndexes.size());
    }
    setI64Attr(plan, "union_option_count", optionCount, builder);
  }

  return mlir::success();
}

struct LowerDSDLSerializationPass
    : public mlir::PassWrapper<LowerDSDLSerializationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  llvm::StringRef getArgument() const final { return "lower-dsdl-serialization"; }
  llvm::StringRef getDescription() const final {
    return "Lower DSDL serialization-plan ops into canonical control-flow form";
  }

  void runOnOperation() override {
    auto module = getOperation();
    mlir::Builder builder(module.getContext());

    for (mlir::Operation &op : module.getBodyRegion().front()) {
      if (op.getName().getStringRef() != "dsdl.schema") {
        continue;
      }
      if (op.getNumRegions() == 0 || op.getRegion(0).empty()) {
        op.emitOpError("must contain a non-empty body region");
        signalPassFailure();
        return;
      }

      for (mlir::Operation &child : op.getRegion(0).front()) {
        if (child.getName().getStringRef() != "dsdl.serialization_plan") {
          continue;
        }
        if (mlir::failed(canonicalizePlan(&child, builder))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerDSDLSerializationPass() {
  return std::make_unique<LowerDSDLSerializationPass>();
}

void registerDSDLPasses() {
  static bool once = false;
  if (once) {
    return;
  }
  once = true;
  static mlir::PassRegistration<LowerDSDLSerializationPass> reg;
}

} // namespace llvmdsdl
