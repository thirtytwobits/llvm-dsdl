#include "llvmdsdl/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

void registerDSDLConvertPasses();

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

std::string sectionSuffix(llvm::StringRef sectionName) {
  if (sectionName == "request") {
    return "__request";
  }
  if (sectionName == "response") {
    return "__response";
  }
  return "";
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

mlir::LogicalResult createPlanCapacityCheckFunction(mlir::ModuleOp module,
                                                    mlir::Operation *plan,
                                                    mlir::OpBuilder &builder) {
  auto *schema = plan->getParentOp();
  if (!schema || schema->getName().getStringRef() != "dsdl.schema") {
    return plan->emitOpError("must be nested under dsdl.schema");
  }
  const auto schemaSym = schema->getAttrOfType<mlir::StringAttr>("sym_name");
  if (!schemaSym) {
    return schema->emitOpError("missing required sym_name attribute");
  }
  const auto sectionAttr = plan->getAttrOfType<mlir::StringAttr>("section");
  const std::string section = sectionAttr ? sectionAttr.getValue().str() : "";
  const std::string funcName = "__llvmdsdl_plan_capacity_check__" +
                               schemaSym.getValue().str() + sectionSuffix(section);
  if (module.lookupSymbol<mlir::func::FuncOp>(funcName)) {
    return mlir::success();
  }

  mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(&module.getBodyRegion().front());

  const mlir::Location loc = plan->getLoc();
  auto i64Ty = builder.getIntegerType(64);
  auto i8Ty = builder.getIntegerType(8);
  auto fnType = builder.getFunctionType(mlir::TypeRange{i64Ty},
                                        mlir::TypeRange{i8Ty});
  auto fn = builder.create<mlir::func::FuncOp>(loc, funcName, fnType);
  fn->setAttr("llvmdsdl.plan_capacity_check", builder.getUnitAttr());
  fn->setAttr("llvmdsdl.schema_sym", schemaSym);
  if (sectionAttr) {
    fn->setAttr("llvmdsdl.section", sectionAttr);
  }
  fn->setAttr("llvmdsdl.plan_origin", builder.getStringAttr("lower-dsdl-serialization"));

  mlir::Block *entry = fn.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  mlir::Value capacityBits = entry->getArgument(0);
  mlir::Value offsetBits =
      builder.create<mlir::arith::ConstantIntOp>(loc, 0, 64).getResult();

  for (mlir::Operation &op : plan->getRegion(0).front()) {
    const auto opName = op.getName().getStringRef();
    if (opName == "dsdl.align") {
      const std::int64_t bits = nonNegative(intAttrOrDefault(&op, "bits", 1));
      if (bits > 1) {
        auto addend =
            builder.create<mlir::arith::ConstantIntOp>(loc, bits - 1, 64)
                .getResult();
        auto bitsValue =
            builder.create<mlir::arith::ConstantIntOp>(loc, bits, 64).getResult();
        auto rounded = builder.create<mlir::arith::AddIOp>(loc, offsetBits, addend);
        auto div = builder.create<mlir::arith::DivUIOp>(loc, rounded, bitsValue);
        offsetBits = builder.create<mlir::arith::MulIOp>(loc, div, bitsValue);
      }
      continue;
    }
    if (opName == "dsdl.io") {
      const std::int64_t stepBits =
          nonNegative(intAttrOrDefault(&op, "lowered_bits",
                                       intAttrOrDefault(&op, "max_bits", 0)));
      if (stepBits > 0) {
        auto bitsValue =
            builder.create<mlir::arith::ConstantIntOp>(loc, stepBits, 64)
                .getResult();
        offsetBits = builder.create<mlir::arith::AddIOp>(loc, offsetBits, bitsValue);
      }
      continue;
    }
    return op.emitError("unsupported operation in serialization plan body");
  }

  auto cond = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ugt, offsetBits, capacityBits);
  auto status =
      builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{i8Ty}, cond, true);
  {
    mlir::OpBuilder thenBuilder = status.getThenBodyBuilder();
    auto fail =
        thenBuilder.create<mlir::arith::ConstantIntOp>(loc, -3, 8).getResult();
    thenBuilder.create<mlir::scf::YieldOp>(loc, fail);
  }
  {
    mlir::OpBuilder elseBuilder = status.getElseBodyBuilder();
    auto ok =
        elseBuilder.create<mlir::arith::ConstantIntOp>(loc, 0, 8).getResult();
    elseBuilder.create<mlir::scf::YieldOp>(loc, ok);
  }
  builder.create<mlir::func::ReturnOp>(loc, status.getResults());

  return mlir::success();
}

mlir::LogicalResult createUnionTagValidationFunction(mlir::ModuleOp module,
                                                     mlir::Operation *plan,
                                                     mlir::OpBuilder &builder) {
  if (!plan->hasAttr("is_union")) {
    return mlir::success();
  }

  auto *schema = plan->getParentOp();
  if (!schema || schema->getName().getStringRef() != "dsdl.schema") {
    return plan->emitOpError("must be nested under dsdl.schema");
  }
  const auto schemaSym = schema->getAttrOfType<mlir::StringAttr>("sym_name");
  if (!schemaSym) {
    return schema->emitOpError("missing required sym_name attribute");
  }
  const auto sectionAttr = plan->getAttrOfType<mlir::StringAttr>("section");
  const std::string section = sectionAttr ? sectionAttr.getValue().str() : "";
  const std::string funcName = "__llvmdsdl_plan_validate_union_tag__" +
                               schemaSym.getValue().str() + sectionSuffix(section);
  if (module.lookupSymbol<mlir::func::FuncOp>(funcName)) {
    return mlir::success();
  }

  std::set<std::int64_t> optionIndexes;
  if (plan->getNumRegions() > 0 && !plan->getRegion(0).empty()) {
    for (mlir::Operation &op : plan->getRegion(0).front()) {
      if (op.getName().getStringRef() != "dsdl.io") {
        continue;
      }
      const auto kindAttr = op.getAttrOfType<mlir::StringAttr>("kind");
      const auto kind = kindAttr ? kindAttr.getValue() : llvm::StringRef("field");
      if (kind == "padding") {
        continue;
      }
      optionIndexes.insert(nonNegative(
          intAttrOrDefault(&op, "union_option_index", /*fallback=*/0)));
    }
  }
  if (optionIndexes.empty()) {
    return plan->emitOpError("union plan has no selectable options");
  }

  mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(&module.getBodyRegion().front());

  const mlir::Location loc = plan->getLoc();
  auto i64Ty = builder.getIntegerType(64);
  auto i8Ty = builder.getIntegerType(8);
  auto fnType = builder.getFunctionType(mlir::TypeRange{i64Ty},
                                        mlir::TypeRange{i8Ty});
  auto fn = builder.create<mlir::func::FuncOp>(loc, funcName, fnType);
  fn->setAttr("llvmdsdl.union_tag_validate", builder.getUnitAttr());
  fn->setAttr("llvmdsdl.schema_sym", schemaSym);
  if (sectionAttr) {
    fn->setAttr("llvmdsdl.section", sectionAttr);
  }
  fn->setAttr("llvmdsdl.plan_origin", builder.getStringAttr("lower-dsdl-serialization"));

  mlir::Block *entry = fn.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  mlir::Value tagValue = entry->getArgument(0);
  mlir::Value anyMatch =
      builder.create<mlir::arith::ConstantIntOp>(loc, 0, 1).getResult();
  for (const std::int64_t option : optionIndexes) {
    auto optConst =
        builder.create<mlir::arith::ConstantIntOp>(loc, option, 64).getResult();
    auto match = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, tagValue, optConst);
    anyMatch = builder.create<mlir::arith::OrIOp>(loc, anyMatch, match);
  }

  auto status =
      builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{i8Ty}, anyMatch, true);
  {
    mlir::OpBuilder thenBuilder = status.getThenBodyBuilder();
    auto ok =
        thenBuilder.create<mlir::arith::ConstantIntOp>(loc, 0, 8).getResult();
    thenBuilder.create<mlir::scf::YieldOp>(loc, ok);
  }
  {
    mlir::OpBuilder elseBuilder = status.getElseBodyBuilder();
    auto fail =
        elseBuilder.create<mlir::arith::ConstantIntOp>(loc, -11, 8).getResult();
    elseBuilder.create<mlir::scf::YieldOp>(loc, fail);
  }
  builder.create<mlir::func::ReturnOp>(loc, status.getResults());

  return mlir::success();
}

mlir::LogicalResult createScalarUnsignedFieldHelpers(mlir::ModuleOp module,
                                                     mlir::Operation *plan,
                                                     mlir::OpBuilder &builder) {
  auto *schema = plan->getParentOp();
  if (!schema || schema->getName().getStringRef() != "dsdl.schema") {
    return plan->emitOpError("must be nested under dsdl.schema");
  }
  const auto schemaSym = schema->getAttrOfType<mlir::StringAttr>("sym_name");
  if (!schemaSym) {
    return schema->emitOpError("missing required sym_name attribute");
  }
  const auto sectionAttr = plan->getAttrOfType<mlir::StringAttr>("section");
  const std::string section = sectionAttr ? sectionAttr.getValue().str() : "";

  if (plan->getNumRegions() == 0 || plan->getRegion(0).empty()) {
    return mlir::success();
  }

  for (mlir::Operation &op : plan->getRegion(0).front()) {
    if (op.getName().getStringRef() != "dsdl.io") {
      continue;
    }
    const auto kindAttr = op.getAttrOfType<mlir::StringAttr>("kind");
    const auto kind = kindAttr ? kindAttr.getValue() : llvm::StringRef("field");
    if (kind != "field") {
      continue;
    }
    const auto arrayKindAttr = op.getAttrOfType<mlir::StringAttr>("array_kind");
    const auto arrayKind =
        arrayKindAttr ? arrayKindAttr.getValue() : llvm::StringRef("none");
    if (arrayKind != "none") {
      continue;
    }
    const auto scalarAttr = op.getAttrOfType<mlir::StringAttr>("scalar_category");
    const auto scalar =
        scalarAttr ? scalarAttr.getValue() : llvm::StringRef("unsigned");
    if (scalar != "unsigned" && scalar != "byte") {
      continue;
    }
    const std::int64_t bitLength =
        nonNegative(intAttrOrDefault(&op, "bit_length", /*fallback=*/0));
    if (bitLength <= 0 || bitLength > 63) {
      continue;
    }
    const std::int64_t stepIndex =
        nonNegative(intAttrOrDefault(&op, "step_index", /*fallback=*/0));
    const auto castModeAttr = op.getAttrOfType<mlir::StringAttr>("cast_mode");
    const auto castMode =
        castModeAttr ? castModeAttr.getValue() : llvm::StringRef("truncated");

    const std::string symbolStem = "__llvmdsdl_plan_scalar_unsigned__" +
                                   schemaSym.getValue().str() +
                                   sectionSuffix(section) + "__" +
                                   std::to_string(stepIndex);
    const std::string serName = symbolStem + "__ser";
    const std::string deserName = symbolStem + "__deser";
    op.setAttr("lowered_ser_unsigned_helper", builder.getStringAttr(serName));
    op.setAttr("lowered_deser_unsigned_helper", builder.getStringAttr(deserName));

    const std::uint64_t mask =
        (bitLength == 64)
            ? UINT64_MAX
            : ((UINT64_C(1) << static_cast<unsigned>(bitLength)) - UINT64_C(1));
    const auto maskSigned = static_cast<std::int64_t>(mask);

    if (!module.lookupSymbol<mlir::func::FuncOp>(serName)) {
      mlir::OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&module.getBodyRegion().front());
      const mlir::Location loc = op.getLoc();
      auto i64Ty = builder.getIntegerType(64);
      auto fnType = builder.getFunctionType(mlir::TypeRange{i64Ty},
                                            mlir::TypeRange{i64Ty});
      auto fn = builder.create<mlir::func::FuncOp>(loc, serName, fnType);
      fn->setAttr("llvmdsdl.scalar_unsigned_helper", builder.getUnitAttr());
      fn->setAttr("llvmdsdl.scalar_unsigned_helper_kind",
                  builder.getStringAttr("serialize"));
      fn->setAttr("llvmdsdl.schema_sym", schemaSym);
      if (sectionAttr) {
        fn->setAttr("llvmdsdl.section", sectionAttr);
      }
      auto *entry = fn.addEntryBlock();
      builder.setInsertionPointToStart(entry);
      auto value = entry->getArgument(0);
      auto maskConst =
          builder.create<mlir::arith::ConstantIntOp>(loc, maskSigned, 64);
      mlir::Value result = value;
      if (castMode == "saturated") {
        auto over = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ugt, value, maskConst);
        result =
            builder.create<mlir::arith::SelectOp>(loc, over, maskConst, value)
                .getResult();
      } else {
        result =
            builder.create<mlir::arith::AndIOp>(loc, value, maskConst).getResult();
      }
      builder.create<mlir::func::ReturnOp>(loc, result);
    }

    if (!module.lookupSymbol<mlir::func::FuncOp>(deserName)) {
      mlir::OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&module.getBodyRegion().front());
      const mlir::Location loc = op.getLoc();
      auto i64Ty = builder.getIntegerType(64);
      auto fnType = builder.getFunctionType(mlir::TypeRange{i64Ty},
                                            mlir::TypeRange{i64Ty});
      auto fn = builder.create<mlir::func::FuncOp>(loc, deserName, fnType);
      fn->setAttr("llvmdsdl.scalar_unsigned_helper", builder.getUnitAttr());
      fn->setAttr("llvmdsdl.scalar_unsigned_helper_kind",
                  builder.getStringAttr("deserialize"));
      fn->setAttr("llvmdsdl.schema_sym", schemaSym);
      if (sectionAttr) {
        fn->setAttr("llvmdsdl.section", sectionAttr);
      }
      auto *entry = fn.addEntryBlock();
      builder.setInsertionPointToStart(entry);
      auto value = entry->getArgument(0);
      auto maskConst =
          builder.create<mlir::arith::ConstantIntOp>(loc, maskSigned, 64);
      auto masked =
          builder.create<mlir::arith::AndIOp>(loc, value, maskConst).getResult();
      builder.create<mlir::func::ReturnOp>(loc, masked);
    }
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
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    std::vector<mlir::Operation *> plans;

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
        plans.push_back(&child);
      }
    }

    for (mlir::Operation *plan : plans) {
      if (mlir::failed(createPlanCapacityCheckFunction(module, plan, builder))) {
        signalPassFailure();
        return;
      }
      if (mlir::failed(createUnionTagValidationFunction(module, plan, builder))) {
        signalPassFailure();
        return;
      }
      if (mlir::failed(createScalarUnsignedFieldHelpers(module, plan, builder))) {
        signalPassFailure();
        return;
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
  registerDSDLConvertPasses();
}

} // namespace llvmdsdl
