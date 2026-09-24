cmake_minimum_required(VERSION 3.20)

# Also usable as `cmake -S docs -B build-docs`, without a compiler or runtime dependencies.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(llamad_docs LANGUAGES NONE)
endif()

find_package(Doxygen 1.17 REQUIRED COMPONENTS doxygen)
get_filename_component(LLAMAD_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

set(DOXYGEN_PROJECT_NAME llamad)
set(DOXYGEN_PROJECT_BRIEF "Local streaming inference over a Unix domain socket")
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${LLAMAD_SOURCE_DIR}/README.md")
set(DOXYGEN_STRIP_FROM_PATH "${LLAMAD_SOURCE_DIR}")
set(DOXYGEN_EXAMPLE_PATH "${LLAMAD_SOURCE_DIR}")
set(DOXYGEN_FILE_PATTERNS *.h *.cpp *.md *.dox)
set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
set(DOXYGEN_EXTRACT_ALL NO)
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_EXTRACT_STATIC YES)
set(DOXYGEN_EXTRACT_ANON_NSPACES YES)
set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_INLINE_SOURCES NO)
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_HTML_TIMESTAMP NO)
set(DOXYGEN_HAVE_DOT NO)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_WARN_AS_ERROR YES)
set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
set(DOXYGEN_WARN_IF_UNDOC_ENUM_VAL YES)
set(DOXYGEN_WARN_IF_DOC_ERROR YES)

# Explicit roots keep vendored and generated code out of the reference.
doxygen_add_docs(llamad_docs
    "${LLAMAD_SOURCE_DIR}/README.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/reference.dox"
    "${CMAKE_CURRENT_SOURCE_DIR}/building.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/daemon.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/client.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/design.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/performance.md"
    "${LLAMAD_SOURCE_DIR}/src"
    "${LLAMAD_SOURCE_DIR}/client"
    "${LLAMAD_SOURCE_DIR}/proto/llamad/v1/convert.h"
    "${LLAMAD_SOURCE_DIR}/tests/engine_smoke.cpp"
    "${LLAMAD_SOURCE_DIR}/tests/engine_bench.cpp"
    WORKING_DIRECTORY "${LLAMAD_SOURCE_DIR}"
    COMMENT "Generating llamad API documentation")
