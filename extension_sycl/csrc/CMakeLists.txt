set(SYCL_SRC_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/add_fp16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass_gemm.cpp
)

# build sycl target as a object
# this object will be linked to the main library along with host target
add_library(${SYCL_TARGET_NAME} OBJECT ${SYCL_SRC_FILES})
