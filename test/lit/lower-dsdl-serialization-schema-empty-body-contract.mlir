// RUN: not %dsdl-opt --pass-pipeline='builtin.module(lower-dsdl-serialization)' %s 2>&1 | FileCheck %s

module {
  dsdl.schema @test_EmptySchemaBody_1_0 attributes {full_name = "test.EmptySchemaBody", major = 1 : i32, minor = 0 : i32, sealed} {
  }
}

// CHECK: error: 'dsdl.schema' op must contain a non-empty body region
