//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Implements shared native-emitter function skeleton orchestration.
///
/// This utility sequences helper binding emission, helper-contract validation,
/// contract preflight, lowered traversal dispatch, and epilogue emission for
/// native backends.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/NativeFunctionSkeleton.h"

#include "llvmdsdl/CodeGen/LoweredRenderIR.h"
#include "llvmdsdl/CodeGen/NativeHelperContract.h"

namespace llvmdsdl
{

bool emitNativeFunctionSkeleton(const SemanticSection&                 section,
                                const LoweredSectionFacts* const       sectionFacts,
                                const HelperBindingDirection           direction,
                                const NativeFunctionSkeletonCallbacks& callbacks)
{
    const auto renderIR = buildLoweredBodyRenderIR(section, sectionFacts, direction);
    if (callbacks.emitHelperBindings)
    {
        callbacks.emitHelperBindings(renderIR.helperBindings);
    }

    std::string missingHelperRequirement;
    if (!validateNativeSectionHelperContract(section,
                                             sectionFacts,
                                             direction,
                                             renderIR.helperBindings,
                                             &missingHelperRequirement))
    {
        if (callbacks.emitMissingHelperContract)
        {
            callbacks.emitMissingHelperContract(missingHelperRequirement);
        }
        return false;
    }

    if (callbacks.emitContractPreflight)
    {
        callbacks.emitContractPreflight(renderIR.helperBindings);
    }
    if (callbacks.makeTraversalCallbacks)
    {
        const auto traversalCallbacks = callbacks.makeTraversalCallbacks(renderIR);
        forEachNativeEmitterRenderStep(renderIR, traversalCallbacks);
    }
    if (callbacks.emitEpilogue)
    {
        callbacks.emitEpilogue();
    }
    return true;
}

}  // namespace llvmdsdl
