find_program(LLVM_LIT llvm-lit HINTS ${LLVM_TOOLS_BINARY_DIR} /opt/homebrew/opt/llvm/bin)
if(NOT LLVM_LIT)
  find_program(PYTHON3_EXECUTABLE python3)
  if(PYTHON3_EXECUTABLE)
    execute_process(
      COMMAND ${PYTHON3_EXECUTABLE} -c "import lit; print('ok')"
      RESULT_VARIABLE _lit_module_ok
      OUTPUT_QUIET
      ERROR_QUIET)
    if(_lit_module_ok EQUAL 0)
      set(LLVM_LIT "${PYTHON3_EXECUTABLE};-m;lit.main")
    endif()
  endif()
endif()

if(NOT LLVM_LIT)
  message(WARNING "Skipping lit tests because llvm-lit/lit is unavailable")
  return()
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py @ONLY)

add_test(NAME llvmdsdl-lit
  COMMAND ${LLVM_LIT} -sv ${CMAKE_CURRENT_BINARY_DIR})
