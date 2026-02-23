//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Implements shared constant-literal rendering helpers for code generation.
///
/// This module renders semantic constant values into language-specific literal
/// expressions while preserving backend behavioral parity.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/ConstantLiteralRender.h"

#include <sstream>

#include "llvmdsdl/Support/Rational.h"

namespace llvmdsdl
{
namespace
{

std::string quoteStringLiteral(const std::string& value)
{
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (const char c : value)
    {
        if (c == '\\' || c == '"')
        {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::string quoteCharLiteral(const char value)
{
    if (value == '\\' || value == '\'')
    {
        return std::string("'\\") + value + "'";
    }
    return std::string("'") + value + "'";
}

}  // namespace

std::string renderConstantLiteral(const ConstantLiteralLanguage language, const Value& value)
{
    if (const auto* booleanValue = std::get_if<bool>(&value.data))
    {
        switch (language)
        {
        case ConstantLiteralLanguage::Python:
            return *booleanValue ? "True" : "False";
        default:
            return *booleanValue ? "true" : "false";
        }
    }

    if (const auto* rationalValue = std::get_if<Rational>(&value.data))
    {
        if (rationalValue->isInteger())
        {
            return std::to_string(rationalValue->asInteger().value());
        }
        std::ostringstream out;
        switch (language)
        {
        case ConstantLiteralLanguage::C:
        case ConstantLiteralLanguage::Cpp:
            out << "((double)" << rationalValue->numerator() << "/(double)" << rationalValue->denominator() << ")";
            return out.str();
        case ConstantLiteralLanguage::Rust:
            out << "(" << rationalValue->numerator() << "f64 / " << rationalValue->denominator() << "f64)";
            return out.str();
        case ConstantLiteralLanguage::Go:
        case ConstantLiteralLanguage::TypeScript:
        case ConstantLiteralLanguage::Python:
            out << "(" << rationalValue->numerator() << " / " << rationalValue->denominator() << ")";
            return out.str();
        }
    }

    if (const auto* stringValue = std::get_if<std::string>(&value.data))
    {
        if ((language == ConstantLiteralLanguage::C || language == ConstantLiteralLanguage::Cpp) &&
            stringValue->size() == 1U)
        {
            return quoteCharLiteral((*stringValue)[0]);
        }
        return quoteStringLiteral(*stringValue);
    }

    return value.str();
}

}  // namespace llvmdsdl
