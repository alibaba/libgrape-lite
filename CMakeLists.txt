project(GRAPEGPU LANGUAGES CUDA CXX)
cmake_minimum_required(VERSION 3.16)

set(GRAPEGPU_MAJOR_VERSION 0)
set(GRAPEGPU_MINOR_VERSION 1)
set(GRAPEGPU_PATCH_VERSION 0)
set(GRAPEGPU_VERSION ${GRAPEGPU_MAJOR_VERSION}.${GRAPEGPU_MINOR_VERSION}.${GRAPEGPU_PATCH_VERSION})

# ------------------------------------------------------------------------------
# cmake options
# ------------------------------------------------------------------------------
option(USE_JEMALLOC "Whether to use jemalloc." OFF)
option(USE_HUGEPAGES "Whether to use hugepages." OFF)
option(BUILD_SHARED_LIBS "Whether to build libgrape-lite as shared library" ON)
option(PROFILING "Whether to enable profiling" OFF)
option(WITH_ASAN "Build with Address Sanitizer" OFF)

if (USE_HUGEPAGES AND LINUX)
    add_definitions(-DUSE_HUGEPAGES)
endif ()

if (PROFILING)
    message("-- Enable profiling")
    add_definitions(-DPROFILING)
endif ()


# find glog---------------------------------------------------------------------
include("cmake/FindGlog.cmake")
if (NOT GLOG_FOUND)
    message(FATAL_ERROR "glog not found, please install the glog library")
else ()
    include_directories(BEFORE SYSTEM ${GLOG_INCLUDE_DIRS})
endif ()

# find gflags-------------------------------------------------------------------
include("cmake/FindGFlags.cmake")

if (NOT GFLAGS_FOUND)
    message(STATUS "gflags not found, build without gflags")
else ()
    include_directories(BEFORE SYSTEM ${GFLAGS_INCLUDE_DIRS})
endif ()

include_directories(thirdparty)
include_directories(thirdparty/cub)
include_directories(thirdparty/thrust)
include_directories(thirdparty/libgrape-lite)
include_directories(thirdparty/libgrape-lite/thirdparty)
include_directories(thirdparty/moderngpu/src)
include_directories(thirdparty/variadic_table/include)

# ------------------------------------------------------------------------------
# setting default cmake type to Release
# ------------------------------------------------------------------------------

set(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
            STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

message("[GRAPEGPU] will build in type: ${CMAKE_BUILD_TYPE}")

# ------------------------------------------------------------------------------
# cmake configs
# ------------------------------------------------------------------------------

include(CheckLibraryExists)
include(GNUInstallDirs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# reference: https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#always-full-rpath
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -fmax-errors=1 -fopenmp -Werror -Wl,-rpath,$ORIGIN")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g")

if (WITH_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -O1")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif ()

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -rdc=true")
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70)
endif ()
message("CUDA ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# ------------------------------------------------------------------------------
# find_libraries
# ------------------------------------------------------------------------------
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

find_package(CUDAToolkit)

# find Threads------------------------------------------------------------------
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)


# find NCCL---------------------------------------------------------------------
# 2.7 at least
include("cmake/FindNCCL.cmake")
include_directories(SYSTEM ${NCCL_INCLUDE_DIRS})
if (NCCL_MAJOR_VERSION VERSION_LESS "2" OR NCCL_MINOR_VERSION VERSION_LESS "7")
    message(FATAL_ERROR "The version of NCCL is required at least 2.7")
endif ()

# find gtest-------------------------------------------------------------------
#include("cmake/FindGTest.cmake")
#if (NOT GTEST_FOUND)
#    message(STATUS "gtest not found, build without gtest")
#else ()
#    include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
#endif ()

# find jemalloc-----------------------------------------------------------------
if (USE_JEMALLOC)
    include("cmake/FindJemalloc.cmake")
    if (NOT JEMALLOC_FOUND)
        message(STATUS "jemalloc not found, build without jemalloc")
    else ()
        add_definitions(-DUSE_JEMALLOC)
        include_directories(SYSTEM ${JEMALLOC_INCLUDE_DIRS})
    endif ()
endif ()

# ------------------------------------------------------------------------------
# generate libgrape-lite
# ------------------------------------------------------------------------------
file(GLOB_RECURSE CORE_SRC_FILES "thirdparty/libgrape-lite/grape/*.cc")
add_library(grape-lite ${CORE_SRC_FILES})

target_link_libraries(grape-lite ${MPI_CXX_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${GLOG_LIBRARIES})

if (JEMALLOC_FOUND)
    target_link_libraries(grape-lite ${JEMALLOC_LIBRARIES})
endif ()


get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
    message(STATUS "dir='${dir}'")
endforeach()

add_executable(lux_convert examples/lux_convert.cc)

add_executable(run_app examples/run_app.cu examples/flags.cc)
target_include_directories(run_app PRIVATE
        examples/analytical_apps)
set_target_properties(run_app PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(run_app grape-lite ${MPI_CXX_LIBRARIES}
        ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${CMAKE_DL_LIBS} ${NCCL_LIBRARIES} CUDA::nvToolsExt CUDA::cuda_driver crypto)


add_executable(test examples/test.cu)
set_target_properties(test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(test grape-lite ${MPI_CXX_LIBRARIES}
        ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${GTEST_LIBRARY}
        ${CMAKE_DL_LIBS} ${NCCL_LIBRARIES} CUDA::nvToolsExt CUDA::cuda_driver)

add_executable(gum examples/gum.cu examples/flags.cc)
target_include_directories(gum PRIVATE examples/pgas)
set_target_properties(gum PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(gum grape-lite ${MPI_CXX_LIBRARIES}
        ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${GTEST_LIBRARY}
        ${CMAKE_DL_LIBS} ${NCCL_LIBRARIES} CUDA::nvToolsExt CUDA::cuda_driver crypto)

message("GLOG, GFLAGS: ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES}")

# ------------------------------------------------------------------------------
# format code
# ------------------------------------------------------------------------------
file(GLOB_RECURSE FILES_NEED_FORMAT "grape_gpu/*.cc"
        "grape_gpu/*.cu"
        "grape_gpu/*.h"
        "examples/*.h"
        "examples/*.cc"
        "grape_gpu/*.cu")
foreach (file_path ${FILES_NEED_FORMAT})
    if (${file_path} MATCHES ".*thirdparty.*")
        list(REMOVE_ITEM FILES_NEED_FORMAT ${file_path})
    endif ()
endforeach ()

add_custom_target(clformat
        COMMAND clang-format --style=file -i ${FILES_NEED_FORMAT}
        COMMENT "Running clang-format."
        VERBATIM)

# ------------------------------------------------------------------------------
# cpplint, check for readability with Google style
# ------------------------------------------------------------------------------
add_custom_target(cpplint
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/misc/cpplint.py ${FILES_NEED_FORMAT}
        COMMENT "Running cpplint check."
        VERBATIM)

add_executable(d2ud examples/d2ud.cpp)
