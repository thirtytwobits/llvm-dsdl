//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>

#include "llvmdsdl/CodeGen/ConstantLiteralRender.h"
#include "llvmdsdl/Support/Rational.h"

bool runConstantLiteralRenderTests()
{
    using llvmdsdl::ConstantLiteralLanguage;
    using llvmdsdl::Value;
    using llvmdsdl::renderConstantLiteral;

    const Value booleanValue{true};
    if (renderConstantLiteral(ConstantLiteralLanguage::Python, booleanValue) != "True" ||
        renderConstantLiteral(ConstantLiteralLanguage::TypeScript, booleanValue) != "true")
    {
        std::cerr << "boolean literal rendering mismatch\n";
        return false;
    }

    const Value integerValue{llvmdsdl::Rational(42, 1)};
    if (renderConstantLiteral(ConstantLiteralLanguage::C, integerValue) != "42" ||
        renderConstantLiteral(ConstantLiteralLanguage::Rust, integerValue) != "42")
    {
        std::cerr << "integer literal rendering mismatch\n";
        return false;
    }

    const Value fractionValue{llvmdsdl::Rational(3, 2)};
    if (renderConstantLiteral(ConstantLiteralLanguage::C, fractionValue) != "((double)3/(double)2)")
    {
        std::cerr << "C fractional literal rendering mismatch\n";
        return false;
    }
    if (renderConstantLiteral(ConstantLiteralLanguage::Rust, fractionValue) != "(3f64 / 2f64)")
    {
        std::cerr << "Rust fractional literal rendering mismatch\n";
        return false;
    }
    if (renderConstantLiteral(ConstantLiteralLanguage::Python, fractionValue) != "(3 / 2)")
    {
        std::cerr << "Python fractional literal rendering mismatch\n";
        return false;
    }

    const Value stringValue{std::string("a\"b\\c")};
    if (renderConstantLiteral(ConstantLiteralLanguage::TypeScript, stringValue) != "\"a\\\"b\\\\c\"")
    {
        std::cerr << "string escaping mismatch\n";
        return false;
    }

    const Value charValue{std::string("x")};
    if (renderConstantLiteral(ConstantLiteralLanguage::C, charValue) != "'x'" ||
        renderConstantLiteral(ConstantLiteralLanguage::Cpp, charValue) != "'x'" ||
        renderConstantLiteral(ConstantLiteralLanguage::Go, charValue) != "\"x\"")
    {
        std::cerr << "single-character literal rendering mismatch\n";
        return false;
    }

    return true;
}
