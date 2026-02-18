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

if(NOT APPLE)
  message(FATAL_ERROR
    "BundleSelfContainedTools.cmake currently supports macOS only.")
endif()

function(_llvmdsdl_collect_deps target_file out_deps)
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

set(bundle_dir "${OUTPUT_DIR}")
file(REMOVE_RECURSE "${bundle_dir}")
file(MAKE_DIRECTORY "${bundle_dir}")

# Copy tools first.
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

# Resolve full runtime closure from the tools and copy dependencies
# with symlink chains preserved.
file(GET_RUNTIME_DEPENDENCIES
  EXECUTABLES "${TOOL_DSDLC}" "${TOOL_DSDLOPT}"
  RESOLVED_DEPENDENCIES_VAR resolved_deps
  UNRESOLVED_DEPENDENCIES_VAR unresolved_deps
  PRE_EXCLUDE_REGEXES "^/System/Library/" "^/usr/lib/"
  POST_EXCLUDE_REGEXES "^/System/Library/" "^/usr/lib/"
)

if(unresolved_deps)
  message(WARNING "Unresolved runtime dependencies: ${unresolved_deps}")
endif()

foreach(dep IN LISTS resolved_deps)
  if(EXISTS "${dep}")
    file(COPY "${dep}" DESTINATION "${bundle_dir}" FOLLOW_SYMLINK_CHAIN)
  endif()
endforeach()

file(GLOB copied_deps "${bundle_dir}/*.dylib")
list(APPEND bundled_items ${copied_deps})
list(REMOVE_DUPLICATES bundled_items)

# Rewrite dependency load paths to reference local copies only.
foreach(item IN LISTS bundled_items)
  _llvmdsdl_collect_deps("${item}" item_deps)
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
file(APPEND "${manifest}" "\nRuntime links after rewrite:\n")
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

message(STATUS "Wrote self-contained tool bundle: ${bundle_dir}")
