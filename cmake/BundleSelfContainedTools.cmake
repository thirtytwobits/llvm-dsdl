cmake_minimum_required(VERSION 3.24)

foreach(var TOOL_DSDLC TOOL_DSDLOPT OUTPUT_DIR)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

foreach(tool "${TOOL_DSDLC}" "${TOOL_DSDLOPT}")
  if(NOT EXISTS "${tool}")
    message(FATAL_ERROR "Tool executable not found: ${tool}")
  endif()
endforeach()

function(_llvmdsdl_collect_macos_deps target_file out_deps)
  execute_process(
    COMMAND otool -L "${target_file}"
    RESULT_VARIABLE otool_result
    OUTPUT_VARIABLE otool_stdout
    ERROR_VARIABLE otool_stderr
  )
  if(NOT otool_result EQUAL 0)
    message(FATAL_ERROR "otool -L failed for ${target_file}: ${otool_stderr}")
  endif()

  string(REPLACE "\n" ";" dep_lines "${otool_stdout}")
  set(deps "")
  foreach(dep_line IN LISTS dep_lines)
    string(STRIP "${dep_line}" dep_line)
    if(dep_line STREQUAL "" OR dep_line MATCHES ":$")
      continue()
    endif()
    string(REGEX REPLACE "^([^ ]+).*" "\\1" dep_path "${dep_line}")
    if(dep_path MATCHES "^/System/Library/" OR dep_path MATCHES "^/usr/lib/")
      continue()
    endif()
    list(APPEND deps "${dep_path}")
  endforeach()
  list(REMOVE_DUPLICATES deps)
  set(${out_deps} "${deps}" PARENT_SCOPE)
endfunction()

function(_llvmdsdl_codesign_macos target_file)
  if(NOT EXISTS "${target_file}" OR IS_SYMLINK "${target_file}")
    return()
  endif()

  if(NOT DEFINED CODESIGN_EXECUTABLE)
    find_program(CODESIGN_EXECUTABLE codesign)
    if(NOT CODESIGN_EXECUTABLE)
      message(FATAL_ERROR
        "macOS self-contained tool bundling requires 'codesign'.")
    endif()
  endif()

  execute_process(
    COMMAND "${CODESIGN_EXECUTABLE}" --force --sign - --timestamp=none "${target_file}"
    RESULT_VARIABLE codesign_result
    OUTPUT_QUIET
    ERROR_VARIABLE codesign_stderr
  )
  if(NOT codesign_result EQUAL 0)
    message(FATAL_ERROR
      "codesign failed for ${target_file}: ${codesign_stderr}")
  endif()
endfunction()

function(_llvmdsdl_is_linux_system_dep dep_path out_result)
  if(dep_path MATCHES "^/lib/" OR dep_path MATCHES "^/lib64/" OR
     dep_path MATCHES "^/usr/lib/" OR dep_path MATCHES "^/usr/lib64/" OR
     dep_path MATCHES "^/usr/libexec/")
    set(${out_result} TRUE PARENT_SCOPE)
  else()
    set(${out_result} FALSE PARENT_SCOPE)
  endif()
endfunction()

set(bundle_dir "${OUTPUT_DIR}")
file(REMOVE_RECURSE "${bundle_dir}")
file(MAKE_DIRECTORY "${bundle_dir}")

set(bundled_items "")
foreach(tool "${TOOL_DSDLC}" "${TOOL_DSDLOPT}")
  file(COPY "${tool}" DESTINATION "${bundle_dir}")
  get_filename_component(tool_name "${tool}" NAME)
  set(dst "${bundle_dir}/${tool_name}")
  file(CHMOD "${dst}"
       PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                   GROUP_READ GROUP_EXECUTE
                   WORLD_READ WORLD_EXECUTE)
  list(APPEND bundled_items "${dst}")
endforeach()

file(GET_RUNTIME_DEPENDENCIES
  EXECUTABLES "${TOOL_DSDLC}" "${TOOL_DSDLOPT}"
  RESOLVED_DEPENDENCIES_VAR resolved_deps
  UNRESOLVED_DEPENDENCIES_VAR unresolved_deps
)

if(unresolved_deps)
  message(WARNING "Unresolved runtime dependencies: ${unresolved_deps}")
endif()

set(copied_deps "")

if(APPLE)
  foreach(dep IN LISTS resolved_deps)
    if(EXISTS "${dep}")
      file(COPY "${dep}" DESTINATION "${bundle_dir}" FOLLOW_SYMLINK_CHAIN)
    endif()
  endforeach()

  file(GLOB copied_deps "${bundle_dir}/*.dylib")
  list(APPEND bundled_items ${copied_deps})
  list(REMOVE_DUPLICATES bundled_items)

  foreach(item IN LISTS bundled_items)
    _llvmdsdl_collect_macos_deps("${item}" item_deps)
    if(item MATCHES "\\.dylib$")
      get_filename_component(item_name "${item}" NAME)
      execute_process(
        COMMAND install_name_tool -id "@loader_path/${item_name}" "${item}"
        RESULT_VARIABLE id_result
        OUTPUT_QUIET
        ERROR_VARIABLE id_stderr
      )
      if(NOT id_result EQUAL 0)
        message(FATAL_ERROR "install_name_tool -id failed for ${item}: ${id_stderr}")
      endif()
      set(prefix "@loader_path")
    else()
      set(prefix "@executable_path")
    endif()

    foreach(dep IN LISTS item_deps)
      get_filename_component(dep_name "${dep}" NAME)
      if(EXISTS "${bundle_dir}/${dep_name}")
        execute_process(
          COMMAND install_name_tool -change "${dep}" "${prefix}/${dep_name}" "${item}"
          RESULT_VARIABLE ch_result
          OUTPUT_QUIET
          ERROR_VARIABLE ch_stderr
        )
        if(NOT ch_result EQUAL 0)
          message(FATAL_ERROR
            "install_name_tool -change failed for ${item}: ${dep} -> ${prefix}/${dep_name}\n${ch_stderr}")
        endif()
      endif()
    endforeach()
  endforeach()

  foreach(item IN LISTS bundled_items)
    _llvmdsdl_codesign_macos("${item}")
  endforeach()
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  find_program(PATCHELF_EXECUTABLE patchelf)
  if(NOT PATCHELF_EXECUTABLE)
    message(FATAL_ERROR
      "Linux self-contained tool bundling requires 'patchelf'. "
      "Install patchelf and re-run the Release bundle target.")
  endif()

  find_program(LDD_EXECUTABLE ldd)

  foreach(dep IN LISTS resolved_deps)
    if(NOT EXISTS "${dep}")
      continue()
    endif()
    _llvmdsdl_is_linux_system_dep("${dep}" is_system_dep)
    if(is_system_dep)
      continue()
    endif()
    file(COPY "${dep}" DESTINATION "${bundle_dir}" FOLLOW_SYMLINK_CHAIN)
  endforeach()

  file(GLOB copied_deps "${bundle_dir}/*.so" "${bundle_dir}/*.so.*")
  list(APPEND bundled_items ${copied_deps})
  list(REMOVE_DUPLICATES bundled_items)

  foreach(item IN LISTS bundled_items)
    execute_process(
      COMMAND "${PATCHELF_EXECUTABLE}" --set-rpath "$ORIGIN" "${item}"
      RESULT_VARIABLE rpath_result
      OUTPUT_QUIET
      ERROR_VARIABLE rpath_stderr
    )
    if(NOT rpath_result EQUAL 0)
      message(FATAL_ERROR "patchelf --set-rpath failed for ${item}: ${rpath_stderr}")
    endif()

    execute_process(
      COMMAND "${PATCHELF_EXECUTABLE}" --print-needed "${item}"
      RESULT_VARIABLE needed_result
      OUTPUT_VARIABLE needed_stdout
      ERROR_VARIABLE needed_stderr
    )
    if(NOT needed_result EQUAL 0)
      message(FATAL_ERROR "patchelf --print-needed failed for ${item}: ${needed_stderr}")
    endif()

    string(REPLACE "\n" ";" needed_lines "${needed_stdout}")
    foreach(needed IN LISTS needed_lines)
      string(STRIP "${needed}" needed)
      if(needed STREQUAL "")
        continue()
      endif()
      if(needed MATCHES "^/")
        get_filename_component(needed_name "${needed}" NAME)
        if(EXISTS "${bundle_dir}/${needed_name}")
          execute_process(
            COMMAND "${PATCHELF_EXECUTABLE}" --replace-needed "${needed}" "${needed_name}" "${item}"
            RESULT_VARIABLE replace_result
            OUTPUT_QUIET
            ERROR_VARIABLE replace_stderr
          )
          if(NOT replace_result EQUAL 0)
            message(FATAL_ERROR
              "patchelf --replace-needed failed for ${item}: ${needed} -> ${needed_name}\n${replace_stderr}")
          endif()
        endif()
      endif()
    endforeach()
  endforeach()
else()
  message(FATAL_ERROR
    "BundleSelfContainedTools.cmake currently supports macOS and Linux only.")
endif()

set(manifest "${bundle_dir}/MANIFEST.txt")
file(WRITE "${manifest}" "Self-contained llvm-dsdl tools bundle\n")
file(APPEND "${manifest}" "Output directory: ${bundle_dir}\n")
file(APPEND "${manifest}" "Bundled executables:\n")
file(APPEND "${manifest}" "  ${bundle_dir}/dsdlc\n")
file(APPEND "${manifest}" "  ${bundle_dir}/dsdl-opt\n\n")
file(APPEND "${manifest}" "Bundled shared libraries:\n")
foreach(dep IN LISTS copied_deps)
  file(APPEND "${manifest}" "  ${dep}\n")
endforeach()

if(APPLE)
  file(APPEND "${manifest}" "\nRuntime links after rewrite (otool -L):\n")
  foreach(item "${bundle_dir}/dsdlc" "${bundle_dir}/dsdl-opt")
    if(EXISTS "${item}")
      execute_process(
        COMMAND otool -L "${item}"
        OUTPUT_VARIABLE otool_out
        RESULT_VARIABLE otool_result
      )
      if(otool_result EQUAL 0)
        file(APPEND "${manifest}" "\n${item}:\n${otool_out}\n")
      endif()
    endif()
  endforeach()
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  file(APPEND "${manifest}" "\nRuntime links after rewrite:\n")
  foreach(item "${bundle_dir}/dsdlc" "${bundle_dir}/dsdl-opt")
    if(NOT EXISTS "${item}")
      continue()
    endif()
    execute_process(
      COMMAND "${PATCHELF_EXECUTABLE}" --print-rpath "${item}"
      OUTPUT_VARIABLE rpath_out
      RESULT_VARIABLE rpath_result
      ERROR_VARIABLE rpath_stderr
    )
    if(rpath_result EQUAL 0)
      string(STRIP "${rpath_out}" rpath_out)
      file(APPEND "${manifest}" "\n${item}:\n  rpath=${rpath_out}\n")
    else()
      file(APPEND "${manifest}" "\n${item}:\n  rpath=<error: ${rpath_stderr}>\n")
    endif()

    if(LDD_EXECUTABLE)
      execute_process(
        COMMAND "${LDD_EXECUTABLE}" "${item}"
        OUTPUT_VARIABLE ldd_out
        RESULT_VARIABLE ldd_result
        ERROR_VARIABLE ldd_stderr
      )
      if(ldd_result EQUAL 0)
        file(APPEND "${manifest}" "${ldd_out}\n")
      else()
        file(APPEND "${manifest}" "  ldd failed: ${ldd_stderr}\n")
      endif()
    else()
      file(APPEND "${manifest}" "  ldd unavailable on PATH\n")
    endif()
  endforeach()
endif()

message(STATUS "Wrote self-contained tool bundle: ${bundle_dir}")
