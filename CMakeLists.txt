project(AutoInc C CXX)
cmake_minimum_required(VERSION 2.8)

# ------------------------------------------------------------------------------
# cmake options
# ------------------------------------------------------------------------------
option(USE_JEMALLOC "Whether to use jemalloc." false)
option(USE_HUGEPAGES "Whether to use hugepages." false)
option(BUILD_SHARED_LIBS "Whether to build libgrape-lite as shared library" ON)
option(PROFILING "Whether to enable profiling" false)
option(USE_CILK "" false) # GCC 7.5.0 is required to make cilk functional

if (USE_HUGEPAGES AND LINUX)
    add_definitions(-DUSE_HUGEPAGES)
endif ()

if (PROFILING)
    message("-- Enable profiling")
    add_definitions(-DPROFILING)
endif ()

include_directories(thirdparty)

# ------------------------------------------------------------------------------
# setting default cmake type to Release
# ------------------------------------------------------------------------------

SET(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
            STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

message("[libgrape-lite] will build in type: ${CMAKE_BUILD_TYPE}")

# ------------------------------------------------------------------------------
# cmake configs
# ------------------------------------------------------------------------------

include(CheckLibraryExists)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(${CMAKE_SOURCE_DIR})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
if (APPLE)
    SET(CMAKE_MACOSX_RPATH ON)
else ()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp  -Wl,-rpath,$ORIGIN")
endif ()

if (USE_CILK)
    message("Enable CILK")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcilkplus -lcilkrts -DCILK")
endif()

SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g") # -fprofile-arcs -ftest-coverage
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g")

# ------------------------------------------------------------------------------
# find_libraries
# ------------------------------------------------------------------------------
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

# find Threads------------------------------------------------------------------
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)

# find glog---------------------------------------------------------------------
include("cmake/FindGlog.cmake")
if (NOT GLOG_FOUND)
    message(FATAL_ERROR "glog not found, please install the glog library")
else ()
    include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
endif ()
find_package(Boost 1.65.0 COMPONENTS serialization mpi REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# find gflags-------------------------------------------------------------------
include("cmake/FindGFlags.cmake")
if (NOT GFLAGS_FOUND)
    message(STATUS "gflags not found, build without gflags")
else ()
    include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
endif ()

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
# generete libgrape-lite
# ------------------------------------------------------------------------------
file(GLOB_RECURSE CORE_SRC_FILES "grape/*.cc")
add_library(grape-lite ${CORE_SRC_FILES})

target_link_libraries(grape-lite ${MPI_CXX_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${GLOG_LIBRARIES})

if (JEMALLOC_FOUND)
    target_link_libraries(grape-lite ${JEMALLOC_LIBRARIES})
endif ()

if (NOT GFLAGS_FOUND)
    message(WARNING "Disable analytical_apps because gflags not found")
else ()
    add_executable(ingress examples/analytical_apps/flags.cc
            examples/analytical_apps/ingress.cc)
    target_include_directories(ingress PRIVATE
            examples/analytical_apps)
    set_target_properties(ingress PROPERTIES OUTPUT_NAME ingress)
    target_link_libraries(ingress grape-lite ${MPI_CXX_LIBRARIES}
            ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${CMAKE_DL_LIBS} -lboost_mpi -lboost_serialization)

    add_executable(tornado_pagerank examples/analytical_apps/flags.cc
            examples/analytical_apps/tornado_pagerank.cc)
    target_include_directories(tornado_pagerank PRIVATE
            examples/analytical_apps)
    target_link_libraries(tornado_pagerank grape-lite ${MPI_CXX_LIBRARIES}
            ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${CMAKE_DL_LIBS})

    add_executable(tornado_php examples/analytical_apps/flags.cc
            examples/analytical_apps/tornado_php.cc)
    target_include_directories(tornado_php PRIVATE
            examples/analytical_apps)
    target_link_libraries(tornado_php grape-lite ${MPI_CXX_LIBRARIES}
            ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${CMAKE_DL_LIBS})

    add_executable(d2ud examples/analytical_apps/flags.cc
            examples/analytical_apps/d2ud.cc)
    target_include_directories(d2ud PRIVATE
            examples/analytical_apps)
    target_link_libraries(d2ud grape-lite ${MPI_CXX_LIBRARIES}
            ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${CMAKE_DL_LIBS})

    add_executable(gen_updated examples/analytical_apps/flags.cc
            examples/analytical_apps/gen_updated.cc)
    target_include_directories(gen_updated PRIVATE
            examples/analytical_apps)
    target_link_libraries(gen_updated grape-lite ${MPI_CXX_LIBRARIES}
            ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${CMAKE_DL_LIBS})
endif ()

# ------------------------------------------------------------------------------
# format code
# ------------------------------------------------------------------------------
file(GLOB_RECURSE FILES_NEED_FORMAT "grape/*.cc"
        "grape/*.h"
        "examples/*.h"
        "examples/*.cc")
list(FILTER FILES_NEED_FORMAT EXCLUDE REGEX ".*thirdparty.*")

add_custom_target(clformat
        COMMAND clang-format --style=file -i ${FILES_NEED_FORMAT}
        COMMENT "Running clang-format."
        VERBATIM)

# ------------------------------------------------------------------------------
# cpplint, check for readability with Google style
# ------------------------------------------------------------------------------
add_custom_target(cpplint
        COMMAND ${CMAKE_SOURCE_DIR}/misc/cpplint.py ${FILES_NEED_FORMAT}
        COMMENT "Running cpplint check."
        VERBATIM)

# ------------------------------------------------------------------------------
# generate docs
# ------------------------------------------------------------------------------
add_custom_target(doc COMMAND doxygen "${CMAKE_SOURCE_DIR}/misc/doc-config"
        COMMENT "Generating docs."
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM)
if (APPLE)
    add_custom_command(TARGET ingress POST_BUILD
            COMMAND bash -c "find . -name \\*.gcda -print0 | xargs -0 rm"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Deleting .gcda files"
            VERBATIM)
endif ()
