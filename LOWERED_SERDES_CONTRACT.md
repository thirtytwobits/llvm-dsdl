# Lowered SerDes Contract (MLIR)

This document defines the backend-facing contract produced by `lower-dsdl-serialization`.

## Versioning

1. Contract version: `1`
2. Producer marker: `"lower-dsdl-serialization"`
3. Required attributes:
   1. Module: `llvmdsdl.lowered_contract_version`, `llvmdsdl.lowered_contract_producer`
   2. Every `dsdl.serialization_plan`: `llvmdsdl.lowered_contract_version`, `llvmdsdl.lowered_contract_producer`

Consumers (`collectLoweredFactsFromMlir`, `convert-dsdl-to-emitc`) fail hard if these markers are missing or mismatched.

## Required Plan-Level Attributes

Each `dsdl.serialization_plan` is expected to carry:

1. `lowered` (unit attr)
2. `lowered_min_bits` / `lowered_max_bits` (i64)
3. `lowered_step_count` / `lowered_field_count` / `lowered_padding_count` / `lowered_align_count` (i64)
4. `lowered_capacity_check_helper` (symbol name)

Union plans additionally require:

1. `union_tag_bits` (i64)
2. `union_option_count` (i64)
3. `lowered_union_tag_validate_helper`
4. `lowered_ser_union_tag_helper`
5. `lowered_deser_union_tag_helper`

## Required Step-Level Attributes

Canonicalized `dsdl.align`:

1. `bits` (i32, no no-op aligners)
2. `step_index` (i64)

Canonicalized `dsdl.io`:

1. Core metadata: `kind`, `scalar_category`, `array_kind`, `bit_length`, `alignment_bits`
2. Canonicalized sizing: `min_bits`, `max_bits`, `lowered_bits`
3. Ordering: `step_index`

Conditional helper bindings:

1. Variable arrays:
   1. `array_length_prefix_bits > 0`
   2. `lowered_ser_array_length_prefix_helper`
   3. `lowered_deser_array_length_prefix_helper`
   4. `lowered_array_length_validate_helper`
2. Scalar unsigned/byte fields:
   1. `lowered_ser_unsigned_helper`
   2. `lowered_deser_unsigned_helper`
3. Scalar signed fields:
   1. `lowered_ser_signed_helper`
   2. `lowered_deser_signed_helper`
4. Scalar float fields:
   1. `lowered_ser_float_helper`
   2. `lowered_deser_float_helper`
5. Delimited composite fields:
   1. `lowered_delimiter_validate_helper`

## Contract Enforcement

1. `lower-dsdl-serialization` stamps contract attrs on module and plan ops.
2. `convert-dsdl-to-emitc` rejects modules/plans that do not satisfy contract version/producer requirements.
3. Backend lowered-fact collection rejects missing or mismatched contract attrs before extracting helper bindings.

## Upgrade Rule

Any incompatible schema/attribute change to this contract must:

1. Increment `kLoweredSerDesContractVersion`.
2. Update this document.
3. Update contract validation tests.
