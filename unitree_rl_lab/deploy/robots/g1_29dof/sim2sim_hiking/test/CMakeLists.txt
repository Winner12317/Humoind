cmake_minimum_required(VERSION 3.12)
project(hiking_obs_test)
set(CMAKE_CXX_STANDARD 17)
find_package(yaml-cpp REQUIRED)
set(DEPLOY ${PROJECT_SOURCE_DIR}/../../../..)
include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  ${DEPLOY}/thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${DEPLOY}/include
)
add_executable(hiking_obs_test hiking_obs_test.cpp)
target_link_libraries(hiking_obs_test unitree_sdk2 ddsc ddscxx rt pthread libyaml-cpp.a fmt
  ${DEPLOY}/thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0)
