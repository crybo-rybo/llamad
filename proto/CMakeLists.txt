# Generate from the proto root: config- and module-mode protobuf_generate both preserve the
# canonical llamad/v1 include paths and descriptor name relative to this directory.
add_library(llamad_proto STATIC llamad/v1/llamad.proto)
add_library(llamad::proto ALIAS llamad_proto)
target_compile_features(llamad_proto PUBLIC cxx_std_17)
target_link_libraries(llamad_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
target_include_directories(llamad_proto PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gen)
protobuf_generate(TARGET llamad_proto
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
protobuf_generate(TARGET llamad_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
