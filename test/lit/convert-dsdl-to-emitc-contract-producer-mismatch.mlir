// RUN: not %dsdl-opt --pass-pipeline='builtin.module(convert-dsdl-to-emitc)' %s 2>&1 | FileCheck %s

module attributes {llvmdsdl.lowered_contract_producer = "wrong-pass", llvmdsdl.lowered_contract_version = 1 : i64} {
  dsdl.schema @test_ContractProducerMismatch_1_0 attributes {full_name = "test.ContractProducerMismatch", major = 1 : i32, minor = 0 : i32, sealed} {
    dsdl.serialization_plan attributes {c_deserialize_symbol = "test__ContractProducerMismatch__deserialize_", c_serialize_symbol = "test__ContractProducerMismatch__serialize_", c_type_name = "test__ContractProducerMismatch", max_bits = 8 : i64, min_bits = 8 : i64} {
      dsdl.io {alignment_bits = 8 : i64, array_capacity = 0 : i64, array_kind = "none", array_length_prefix_bits = 0 : i64, bit_length = 8 : i64, c_name = "value", cast_mode = "truncated", kind = "field", max_bits = 8 : i64, min_bits = 8 : i64, name = "value", scalar_category = "unsigned", type_name = "uint8", union_option_index = 0 : i64, union_tag_bits = 0 : i64}
    }
  }
}

// CHECK: error: lowered SerDes contract producer mismatch: expected 'lower-dsdl-serialization'; run lower-dsdl-serialization before convert-dsdl-to-emitc
