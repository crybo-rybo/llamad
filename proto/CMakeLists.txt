# llamad.proto's generated C++ messages and gRPC service. Generated from this directory, so that
# both CMake's FindProtobuf and Protobuf's own config place the output under gen/llamad/v1/, which
# is how every source spells it. protoc's headers are included as system headers.
set(gen ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${gen})

add_library(llamad_proto STATIC llamad/v1/llamad.proto)
target_compile_features(llamad_proto PUBLIC cxx_std_17)
target_link_libraries(llamad_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
target_include_directories(llamad_proto SYSTEM PUBLIC ${gen})

protobuf_generate(TARGET llamad_proto
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    PROTOC_OUT_DIR ${gen})
protobuf_generate(TARGET llamad_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    PROTOC_OUT_DIR ${gen})
