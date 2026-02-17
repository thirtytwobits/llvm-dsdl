#ifndef LLVMDSDL_FRONTEND_AST_PRINTER_H
#define LLVMDSDL_FRONTEND_AST_PRINTER_H

#include "llvmdsdl/Frontend/AST.h"

#include <string>

namespace llvmdsdl {

std::string printAST(const ASTModule &module);

} // namespace llvmdsdl

#endif // LLVMDSDL_FRONTEND_AST_PRINTER_H
