// RUN: %dsdl-opt --pass-pipeline='builtin.module(lower-dsdl-serialization)' %s | FileCheck %s

module {
  dsdl.schema @test_Type_1_0 attributes {full_name = "test.Type", major = 1 : i32, minor = 0 : i32, sealed} {
    dsdl.serialization_plan attributes {c_deserialize_symbol = "test__Type__deserialize_", c_serialize_symbol = "test__Type__serialize_", c_type_name = "test__Type", is_union, max_bits = 16 : i64, min_bits = 8 : i64} {
      dsdl.align {bits = 1 : i32}
      dsdl.align {bits = 8 : i32}
      dsdl.io {alignment_bits = 8 : i64, array_capacity = 0 : i64, array_kind = "none", array_length_prefix_bits = 0 : i64, bit_length = 8 : i64, c_name = "opt_a", cast_mode = "saturated", kind = "field", max_bits = 8 : i64, min_bits = -2 : i64, name = "opt_a", scalar_category = "unsigned", type_name = "saturated uint8", union_option_index = 3 : i64, union_tag_bits = 0 : i64}
      dsdl.io {alignment_bits = 1 : i64, array_capacity = 0 : i64, array_kind = "none", array_length_prefix_bits = 0 : i64, bit_length = 0 : i64, c_name = "_pad", cast_mode = "saturated", kind = "padding", max_bits = 0 : i64, min_bits = 0 : i64, name = "_pad", scalar_category = "void", type_name = "void0", union_option_index = 0 : i64, union_tag_bits = 0 : i64}
    }
  }
}

// CHECK: dsdl.serialization_plan attributes {
// CHECK: lowered
// CHECK: lowered_align_count = 1 : i64
// CHECK: lowered_field_count = 1 : i64
// CHECK: lowered_max_bits = 16 : i64
// CHECK: lowered_min_bits = 8 : i64
// CHECK: lowered_padding_count = 1 : i64
// CHECK: lowered_step_count = 2 : i64
// CHECK: union_option_count = 1 : i64
// CHECK: union_tag_bits = 8 : i64
// CHECK: dsdl.align {bits = 8 : i32, step_index = 0 : i64}
// CHECK: dsdl.io
// CHECK-SAME: lowered_bits = 8 : i64
// CHECK-SAME: min_bits = 0 : i64
// CHECK-SAME: step_index = 1 : i64
// CHECK-NOT: dsdl.align {bits = 1 : i32
// CHECK-NOT: kind = "padding"
