cmake_minimum_required(VERSION 2.6)

project(lane_det)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_61;code=sm_61)

include_directories(${PROJECT_SOURCE_DIR}/include)

#tensort
include_directories(/home/TensorRT-6.0.1.5/include)
link_directories(/home/TensorRT-6.0.1.5/lib)
link_directories(/home/TensorRT-6.0.1.5/include/lib/stubs)

#opencv
include_directories(/home/opencv-3.4.2/include)
link_directories(/home/opencv-3.4.2/lib)

set(OPENCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_calib3d
                opencv_objdetect opencv_video opencv_features2d opencv_imgcodecs)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

set(CUDA_LIBS cudart cudnn cublas rt dl cudadevrt)
set(TRT_LIBS  nvparsers nvinfer nvonnxparser)

link_libraries(${CUDA_LIBS} ${TRT_LIBS} ${OPENCV_LIBS})

add_executable(lane_det ${PROJECT_SOURCE_DIR}/main.cpp)

target_link_libraries(lane_det ${CUDA_LIBS} ${OPENCV_LIBS} ${TRT_LIBS})
