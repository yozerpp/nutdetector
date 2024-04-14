cmake_minimum_required(VERSION 3.8)
project(img1 LANGUAGES C CXX CUDA)
find_package(CUDA REQUIRED)
set(OPENCV_DIR "/usr/lib/opencv")
set(Boost_ADDITIONAL_VERSIONS "1.74.0")
#set(BOOST_DIR "/usr/local/boost")
#add_definitions( -DBOOST_ALL_NO_LIB )
find_package(Boost 1.74 COMPONENTS thread system program_options REQUIRED)
find_package(OpenCV REQUIRED)
set(CUDA_HOST_COMPILER g++)
SET(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

SET(CUDA_NVCC_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo; -rdc=true; --extended-lambda; -std=c++17; -gencode arch=compute_75,code=sm_75; -gencode=arch=compute_75,code=compute_75; -arch=compute_75")
list(APPEND CUDA_FILES "src/kernel/Kernel.cu" "src/pipeline/Extractor.cu")
cuda_compile(CU_O ${CUDA_FILES} ${CUDA_NVCC_FLAGS})
#cuda_add_library(CU_L ${CU_O})
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fpermissive")
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} "-Wl,-z,stack-size=100000000000000000" )
#set(CMAKE_STATIC_LINKER_FLAGS "-lboost_thread -lboost_system")
include_directories(src/include)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${Boost_INCLUDE_DIRS})
#include(FetchContent)
#link_libraries(yaml-cpp::yaml-cpp) # The library or executable that require
link_libraries( ${OpenCV_LIBS} )
link_libraries(Boost::thread Boost::system Boost::program_options)
#link_libraries(CU_L)

cuda_add_executable(img1
        src/pipeline/Preprocessor.cpp
        src/common/Image.cpp
        src/include/stb_image_impl.c
        src/include/stb_image_write_impl.c
        src/main.cpp
        src/common/common.cpp
        src/pipeline/Detector.cpp
        src/pipeline/Model.cpp
        ${CU_O}
#        src/kernel/Kernel.cu
#        src/pipeline/Extractor.cu
)

#FetchContent_Declare(
#        yaml-cpp
#        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
#        GIT_TAG 0.8.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
#)
#FetchContent_GetProperties(yaml-cpp)
#
#if(NOT yaml-cpp_POPULATED)
#    message(STATUS "Fetching yaml-cpp...")
#    FetchContent_Populate(yaml-cpp)
#    add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
#endif()
#add_executable(img1
#        src/common/Image.cu
#        src/common/Preprocessing.h
#        src/include/stb_image_impl.c
#        src/include/stb_image_write_impl.c
#        src/main.cpp
#        src/include/Common.cpp
#        src/include/Common.h
#        src/common/Processor.cuh
#        src/common/Processor.cu
#        ${cuda})