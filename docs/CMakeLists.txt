cmake_minimum_required(VERSION 3.20)

# Also usable as `cmake -S docs -B build-docs`, without a compiler or runtime dependencies.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(llamad_docs LANGUAGES NONE)
endif()

find_package(Doxygen 1.17 REQUIRED COMPONENTS doxygen)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
get_filename_component(LLAMAD_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(SITE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/site")

# The wire protocol page is generated from the proto, so it cannot drift from it.
set(WIRE_PROTOCOL_PAGE "${CMAKE_CURRENT_BINARY_DIR}/wire_protocol.dox")
add_custom_command(
    OUTPUT "${WIRE_PROTOCOL_PAGE}"
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/proto_reference.py"
            "${LLAMAD_SOURCE_DIR}/proto/llamad/v1/llamad.proto" "${WIRE_PROTOCOL_PAGE}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/proto_reference.py"
            "${LLAMAD_SOURCE_DIR}/proto/llamad/v1/llamad.proto"
    COMMENT "Generating the wire protocol page from llamad.proto")
add_custom_target(llamad_wire_protocol_page DEPENDS "${WIRE_PROTOCOL_PAGE}")

set(DOXYGEN_PROJECT_NAME llamad)
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
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
set(DOXYGEN_GENERATE_TREEVIEW NO)
set(DOXYGEN_HTML_TIMESTAMP NO)
set(DOXYGEN_HTML_COLORSTYLE AUTO_LIGHT)
set(DOXYGEN_HTML_EXTRA_STYLESHEET "${SITE_DIR}/llamad.css")
set(DOXYGEN_HTML_HEADER "${SITE_DIR}/header.html")
set(DOXYGEN_HTML_FOOTER "${SITE_DIR}/footer.html")
set(DOXYGEN_LAYOUT_FILE "${SITE_DIR}/DoxygenLayout.xml")
set(DOXYGEN_HAVE_DOT NO)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_WARN_AS_ERROR YES)
set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
set(DOXYGEN_WARN_IF_UNDOC_ENUM_VAL YES)
set(DOXYGEN_WARN_IF_DOC_ERROR YES)

# Explicit roots keep vendored and generated code out of the reference. design.md is included
# into the architecture page rather than listed here.
doxygen_add_docs(llamad_docs
    "${SITE_DIR}/overview.dox"
    "${SITE_DIR}/architecture.dox"
    "${WIRE_PROTOCOL_PAGE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/protocol.md"
    "${LLAMAD_SOURCE_DIR}/src"
    "${LLAMAD_SOURCE_DIR}/tests/engine_smoke.cpp"
    "${LLAMAD_SOURCE_DIR}/tests/engine_bench.cpp"
    WORKING_DIRECTORY "${LLAMAD_SOURCE_DIR}"
    COMMENT "Generating llamad API documentation")
add_dependencies(llamad_docs llamad_wire_protocol_page)
