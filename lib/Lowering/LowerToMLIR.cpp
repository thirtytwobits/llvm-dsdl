#include "llvmdsdl/Lowering/LowerToMLIR.h"

#include "llvmdsdl/IR/DSDLOps.h"

#include "mlir/IR/Builders.h"

#include <algorithm>

namespace llvmdsdl {
namespace {

std::string mangleSymbol(std::string fullName, std::uint32_t major,
                         std::uint32_t minor) {
  for (char &c : fullName) {
    if (c == '.') {
      c = '_';
    }
  }
  return fullName + "_" + std::to_string(major) + "_" +
         std::to_string(minor);
}

std::string fieldKind(const SemanticField &f) {
  return f.isPadding ? "padding" : "field";
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
lowerToMLIR(const SemanticModule &module, mlir::MLIRContext &context,
            DiagnosticEngine &diagnostics) {
  mlir::OpBuilder builder(&context);
  auto m = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(&m->getRegion(0).front());

  for (const auto &def : module.definitions) {
    builder.setInsertionPointToEnd(&m->getRegion(0).front());
    const auto loc = builder.getUnknownLoc();

    mlir::OperationState state(loc, "dsdl.schema");
    state.addAttribute("sym_name",
                       builder.getStringAttr(mangleSymbol(
                           def.info.fullName, def.info.majorVersion,
                           def.info.minorVersion)));
    state.addAttribute("full_name", builder.getStringAttr(def.info.fullName));
    state.addAttribute("major", builder.getI32IntegerAttr(def.info.majorVersion));
    state.addAttribute("minor", builder.getI32IntegerAttr(def.info.minorVersion));
    if (def.request.sealed) {
      state.addAttribute("sealed", builder.getUnitAttr());
    }
    if (def.request.extentBits) {
      state.addAttribute("extent_bits",
                         builder.getI64IntegerAttr(*def.request.extentBits));
    }
    if (def.info.fixedPortId) {
      state.addAttribute("fixed_port_id",
                         builder.getI64IntegerAttr(*def.info.fixedPortId));
    }
    if (def.isService) {
      state.addAttribute("service", builder.getUnitAttr());
    }
    if (def.request.deprecated) {
      state.addAttribute("deprecated", builder.getUnitAttr());
    }
    state.addRegion();

    auto *schema = builder.create(state);
    auto &schemaBody = schema->getRegion(0);
    schemaBody.push_back(new mlir::Block());

    builder.setInsertionPointToStart(&schemaBody.front());

    auto emitSection = [&](const SemanticSection &section,
                           llvm::StringRef sectionName) {
      for (const auto &field : section.fields) {
        mlir::OperationState fieldState(loc, "dsdl.field");
        fieldState.addAttribute("name", builder.getStringAttr(field.name));
        fieldState.addAttribute("type_name",
                                builder.getStringAttr(field.type.str()));
        if (field.isPadding) {
          fieldState.addAttribute("padding", builder.getUnitAttr());
        }
        if (!sectionName.empty()) {
          fieldState.addAttribute("section", builder.getStringAttr(sectionName));
        }
        (void)builder.create(fieldState);
      }

      for (const auto &constant : section.constants) {
        mlir::OperationState constState(loc, "dsdl.constant");
        constState.addAttribute("name", builder.getStringAttr(constant.name));
        constState.addAttribute("type_name",
                                builder.getStringAttr(constant.type.str()));
        constState.addAttribute("value_text",
                                builder.getStringAttr(constant.value.str()));
        if (!sectionName.empty()) {
          constState.addAttribute("section", builder.getStringAttr(sectionName));
        }
        (void)builder.create(constState);
      }

      mlir::OperationState planState(loc, "dsdl.serialization_plan");
      if (!sectionName.empty()) {
        planState.addAttribute("section", builder.getStringAttr(sectionName));
      }
      planState.addRegion();
      auto *plan = builder.create(planState);
      auto &planRegion = plan->getRegion(0);
      planRegion.push_back(new mlir::Block());

      builder.setInsertionPointToStart(&planRegion.front());
      for (const auto &field : section.fields) {
        mlir::OperationState alignState(loc, "dsdl.align");
        alignState.addAttribute("bits", builder.getI32IntegerAttr(8));
        (void)builder.create(alignState);

        mlir::OperationState ioState(loc, "dsdl.io");
        ioState.addAttribute("kind", builder.getStringAttr(fieldKind(field)));
        ioState.addAttribute("name", builder.getStringAttr(field.name));
        ioState.addAttribute("type_name", builder.getStringAttr(field.type.str()));
        (void)builder.create(ioState);
      }

      builder.setInsertionPointAfter(plan);
    };

    emitSection(def.request, def.isService ? "request" : "");
    if (def.response) {
      emitSection(*def.response, "response");
    }

  }

  if (diagnostics.hasErrors()) {
    return nullptr;
  }
  return m;
}

} // namespace llvmdsdl
