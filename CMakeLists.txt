cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include_directories(${CMAKE_CURRENT_BINARY_DIR})


project(${PACKAGE_NAME} C CXX)

find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(GTest REQUIRED) 
find_package(GFlags REQUIRED)
find_package(GLog REQUIRED) 

#link_libraries(-static-libstdc++ -static-libgcc)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

set(CMAKE_CXX_STANDARD 14)
add_definitions("-D_GLIBCXX_USE_CXX11_ABI=1")

add_subdirectory("src")
add_subdirectory("tools/random-program-generator")
