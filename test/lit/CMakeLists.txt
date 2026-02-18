find_program(LLVM_LIT_EXECUTABLE
  NAMES llvm-lit lit
  HINTS ${LLVM_TOOLS_BINARY_DIR} /opt/homebrew/opt/llvm/bin
  NO_CACHE)

set(LLVM_LIT_COMMAND "")
if(LLVM_LIT_EXECUTABLE)
  set(LLVM_LIT_COMMAND "${LLVM_LIT_EXECUTABLE}")
else()
  find_program(PYTHON3_EXECUTABLE python3)
  if(PYTHON3_EXECUTABLE)
    execute_process(
      COMMAND ${PYTHON3_EXECUTABLE} -c "import lit; print('ok')"
      RESULT_VARIABLE _lit_module_ok
      OUTPUT_QUIET
      ERROR_QUIET)
    if(_lit_module_ok EQUAL 0)
      set(LLVM_LIT_COMMAND "${PYTHON3_EXECUTABLE};-m;lit.main")
    endif()
  endif()
endif()

if(NOT LLVM_LIT_COMMAND)
  message(WARNING "Skipping lit tests because neither llvm-lit, lit, nor python -m lit.main is available")
  return()
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py @ONLY)

add_test(NAME llvmdsdl-lit
  COMMAND ${LLVM_LIT_COMMAND} -sv ${CMAKE_CURRENT_BINARY_DIR})
