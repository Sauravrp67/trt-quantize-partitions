cmake_minimum_required(VERSION 3.18)
project(rtdetr_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(CUDAToolkit REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation prefix")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS
    "${TENSORRT_ROOT}/include" /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY nvinfer HINTS
    "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64"
    /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin HINTS
    "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64"
    /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT C++ headers/library not found; set -DTENSORRT_ROOT=...")
endif()

add_library(rtdetr_runtime src/runtime.cpp src/model.cpp src/preprocess.cpp)
target_include_directories(rtdetr_runtime PUBLIC include ${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(rtdetr_runtime PUBLIC ${OpenCV_LIBS} CUDA::cudart ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY})
target_compile_options(rtdetr_runtime PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(rtdetr_runtime PROPERTIES BUILD_RPATH "${TENSORRT_ROOT}/lib")

add_executable(rtdetr-infer src/main.cpp)
target_link_libraries(rtdetr-infer PRIVATE rtdetr_runtime)

install(TARGETS rtdetr_runtime rtdetr-infer)
install(DIRECTORY include/rtdetr DESTINATION include)
install(DIRECTORY models/ DESTINATION share/rtdetr-cpp/models FILES_MATCHING PATTERN "*.json" PATTERN "*.yaml")

include(CTest)
if(BUILD_TESTING)
    add_executable(rtdetr-contract-test tests/test_contract.cpp)
    target_link_libraries(rtdetr-contract-test PRIVATE rtdetr_runtime)
    add_test(NAME contract COMMAND rtdetr-contract-test "${CMAKE_SOURCE_DIR}/models")
    add_executable(rtdetr-preprocess-test tests/test_preprocess.cpp)
    target_link_libraries(rtdetr-preprocess-test PRIVATE rtdetr_runtime)
    add_test(NAME preprocess COMMAND rtdetr-preprocess-test "${CMAKE_SOURCE_DIR}/tests/data")
endif()
