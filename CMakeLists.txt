cmake_minimum_required(VERSION 3.8)
project(img1 LANGUAGES C CXX CUDA)
find_package(CUDA REQUIRED)
#option(CV_DIR "OPENCV_DIR" "/usr/lib/opencv")
#set(OPENCV_DIR CV_DIR)
#find_package(OpenCV REQUIRED)
set(CUDA_HOST_COMPILER g++)
SET(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_NVCC_FLAGS ${CMAKE_CUDA_FLAGS} "-g; -G; -rdc=true; --extended-lambda; -std=c++17; -gencode=arch=compute_89,code=compute_89")
list(APPEND CUDA_FILES "src/pipeline/Extractor.cu")
cuda_compile(CU_O ${CUDA_FILES} ${CUDA_NVCC_FLAGS})
add_library(Extractor STATIC ${CU_O})
list(APPEND CU_F_TEST "src/kernel/Kernel.cu")
cuda_compile(CU_O_TEST ${CU_F_TEST} ${CUDA_NVCC_FLAGS})
add_library(Kernel STATIC ${CU_O_TEST})
#cuda_add_library(CU_L ${CU_O})
#set(CMAKE_STATIC_LINKER_FLAGS "-lboost_thread -lboost_system")
include_directories(src/include)
#include_directories(/usr/local/cuda/include)
#include_directories( ${OpenCV_INCLUDE_DIRS} )
#link_libraries( ${OpenCV_LIBS} )
cuda_add_executable(img1
        src/pipeline/Preprocessor.cpp
        src/common/Image.cpp
        src/include/stb_image_impl.c
        src/include/stb_image_write_impl.c
        src/main.cpp
        src/pipeline/Detector.cpp
        src/pipeline/Model.cpp
        src/common/Matrix.cpp
        #        src/kernel/Kernel.cu
#        src/pipeline/Extractor.cu
)
target_link_libraries(img1 Kernel)
target_link_libraries(img1 Extractor)
add_executable( test
            src/test.cpp
            src/pipeline/Preprocessor.cpp
            src/common/Image.cpp
            src/include/stb_image_impl.c
            src/include/stb_image_write_impl.c
            src/common/Matrix.cpp
)
target_link_libraries(test Kernel)
#target_link_libraries(test ${OpenCV_LIBS})
#target_include_directories(test PUBLIC ${OpenCV_INCLUDE_DIRS} )