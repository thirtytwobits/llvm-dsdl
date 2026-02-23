//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>

#include "llvmdsdl/CodeGen/NamingPolicy.h"

bool runNamingPolicyTests()
{
    using llvmdsdl::CodegenNamingLanguage;
    using llvmdsdl::codegenSanitizeIdentifier;
    using llvmdsdl::codegenToPascalCaseIdentifier;
    using llvmdsdl::codegenToSnakeCaseIdentifier;
    using llvmdsdl::codegenToUpperSnakeCaseIdentifier;

    if (codegenSanitizeIdentifier(CodegenNamingLanguage::TypeScript, "class") != "class_")
    {
        std::cerr << "TypeScript keyword sanitization mismatch\n";
        return false;
    }
    if (codegenSanitizeIdentifier(CodegenNamingLanguage::Python, "def") != "def_")
    {
        std::cerr << "Python keyword sanitization mismatch\n";
        return false;
    }
    if (codegenSanitizeIdentifier(CodegenNamingLanguage::Rust, "self") != "self_")
    {
        std::cerr << "Rust keyword sanitization mismatch\n";
        return false;
    }
    if (codegenSanitizeIdentifier(CodegenNamingLanguage::Go, "map") != "map_")
    {
        std::cerr << "Go keyword sanitization mismatch\n";
        return false;
    }
    if (codegenSanitizeIdentifier(CodegenNamingLanguage::Cpp, "namespace") != "namespace_")
    {
        std::cerr << "C++ keyword sanitization mismatch\n";
        return false;
    }
    if (codegenSanitizeIdentifier(CodegenNamingLanguage::C, "int") != "int_")
    {
        std::cerr << "C keyword sanitization mismatch\n";
        return false;
    }

    if (codegenToSnakeCaseIdentifier(CodegenNamingLanguage::TypeScript, "FlightControlMode") != "flight_control_mode")
    {
        std::cerr << "snake_case projection mismatch\n";
        return false;
    }
    if (codegenToSnakeCaseIdentifier(CodegenNamingLanguage::Python, "9AxisIMU") != "_9axis_imu")
    {
        std::cerr << "snake_case digit-prefix projection mismatch\n";
        return false;
    }
    if (codegenToPascalCaseIdentifier(CodegenNamingLanguage::Python, "vslam_pose_update") != "VslamPoseUpdate")
    {
        std::cerr << "PascalCase projection mismatch\n";
        return false;
    }
    if (codegenToUpperSnakeCaseIdentifier(CodegenNamingLanguage::Go, "OpticalFlowRate") != "OPTICAL_FLOW_RATE")
    {
        std::cerr << "UPPER_SNAKE_CASE projection mismatch\n";
        return false;
    }

    return true;
}
