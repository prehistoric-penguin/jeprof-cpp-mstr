cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include_directories(${CMAKE_CURRENT_BINARY_DIR})


project(${PACKAGE_NAME} C CXX)

find_package(Boost COMPONENTS system filesystem regex REQUIRED)
find_package(GTest REQUIRED) 
find_package(GFlags REQUIRED)
find_package(GLog REQUIRED) 
find_package(TCmalloc REQUIRED)

#link_libraries(-static-libstdc++ -static-libgcc)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE 
  "${CMAKE_CXX_FLAGS_RELEASE} -Wall -g3 -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")

if ("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
  message(STATUS "** It's debug mode **")
endif()

set(CMAKE_CXX_STANDARD 14)
add_definitions("-D_GLIBCXX_USE_CXX11_ABI=1")

add_subdirectory("src")
add_subdirectory("tools/random-program-generator")
