cmake_minimum_required(VERSION 3.11)
project(PATCHWORK VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 14)
set(PYTHON_EXECUTABLE python3)
set(CMAKE_BUILD_TYPE Release)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

if(CMAKE_VERSION VERSION_LESS "3.15")
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ListPrepend.cmake")
  list_prepend(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
else()
  list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()


list(APPEND Open3D_LIBRARIES dl)

add_subdirectory(patchworkpp)

set(INCLUDE_PYTHON_WRAPPER OFF CACHE BOOL "Build Python wrapper")
set(INCLUDE_EXAMPLES ON CACHE BOOL "Build examples")

if (INCLUDE_PYTHON_WRAPPER)

  message(STATUS "Building Python wrapper")
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  include(FetchContent)
  include(pybind11)
  add_subdirectory(python_wrapper)
  
endif()
