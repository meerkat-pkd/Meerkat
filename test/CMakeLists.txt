cuda_add_executable(GraphTest GraphTest.cu
  OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})
target_include_directories(GraphTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(GraphTest gtest)