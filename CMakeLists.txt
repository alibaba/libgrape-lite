project(libgrape-lite C CXX)
cmake_minimum_required(VERSION 2.8)

set(LIBGRAPELITE_MAJOR_VERSION 0)
set(LIBGRAPELITE_MINOR_VERSION 1)
set(LIBGRAPELITE_PATCH_VERSION 0)
set(LIBGRAPELITE_VERSION ${LIBGRAPELITE_MAJOR_VERSION}.${LIBGRAPELITE_MINOR_VERSION}.${LIBGRAPELITE_PATCH_VERSION})

# ------------------------------------------------------------------------------
# cmake options
# ------------------------------------------------------------------------------
option(USE_JEMALLOC "Whether to use jemalloc." OFF)
option(USE_HUGEPAGES "Whether to use hugepages." OFF)
option(BUILD_SHARED_LIBS "Whether to build libgrape-lite as shared library" ON)
option(PROFILING "Whether to enable profiling" OFF)
option(WITH_ASAN "Build with Address Sanitizer" OFF)
option(BUILD_LIBGRAPELITE_DOCS "Build libgrape-lite documentation" ON)

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

set(DEFAULT_BUILD_TYPE "Release")
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
include(GNUInstallDirs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# reference: https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#always-full-rpath
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Werror -Wl,-rpath,$ORIGIN")
endif ()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g")

if (WITH_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -O1")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

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

# find rdkafka---------------------------------------------------------------------
include("cmake/FindRdkafka.cmake")
if (NOT RDKAFKA_FOUND)
    message(STATUS "rdkafka not found, build without rdkafka")
endif ()

macro(install_libgrapelite_target target)
    # install
    install(TARGETS ${target}
        EXPORT libgrapelite-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endmacro()

# ------------------------------------------------------------------------------
# generate libgrape-lite
# ------------------------------------------------------------------------------
file(GLOB_RECURSE CORE_SRC_FILES "grape/*.cc")
add_library(grape-lite ${CORE_SRC_FILES})
install_libgrapelite_target(grape-lite)

target_link_libraries(grape-lite ${MPI_CXX_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${GLOG_LIBRARIES})

if (JEMALLOC_FOUND)
    target_link_libraries(grape-lite ${JEMALLOC_LIBRARIES})
endif ()

if (NOT GFLAGS_FOUND)
    message(WARNING "Disable analytical_apps because gflags not found")
else ()
    add_executable(analytical_apps examples/analytical_apps/flags.cc
            examples/analytical_apps/run_app.cc)
    target_include_directories(analytical_apps PRIVATE
            examples/analytical_apps)
    set_target_properties(analytical_apps PROPERTIES OUTPUT_NAME run_app)
    target_link_libraries(analytical_apps grape-lite ${MPI_CXX_LIBRARIES}
            ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${CMAKE_DL_LIBS})
    install_libgrapelite_target(analytical_apps)
endif ()

if (NOT GFLAGS_FOUND)
    message(WARNING "Disable gnn_sampler because gflags not found")
elseif (NOT RDKAFKA_FOUND)
    message(WARNING "Disable gnn_sampler because rdkafka not found")
else ()
    add_executable(gnn_sampler examples/gnn_sampler/run_sampler.cc)
    set_target_properties(gnn_sampler PROPERTIES OUTPUT_NAME run_sampler)
    target_include_directories(gnn_sampler PRIVATE
            examples/gnn_sampler
            examples/gnn_sampler/thirdparty
            ${RDKAFKA_INCLUDE_DIR})
    target_link_libraries(gnn_sampler grape-lite  ${MPI_CXX_LIBRARIES}
            ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${CMAKE_DL_LIBS} ${RDKAFKA_LIBRARIES})
    install_libgrapelite_target(gnn_sampler)
endif ()

set(EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)

if (PROFILING)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/thirdparty/atlarge-research-granula
        DESTINATION include # target directory
        FILES_MATCHING      # install only matched files
        PATTERN "*.h"       # select header files
)        
endif()
install(DIRECTORY ${PROJECT_SOURCE_DIR}/thirdparty/flat_hash_map
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/grape
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
)
install(DIRECTORY ${EXAMPLES_DIR}/analytical_apps/bfs
                  ${EXAMPLES_DIR}/analytical_apps/cdlp
                  ${EXAMPLES_DIR}/analytical_apps/lcc
                  ${EXAMPLES_DIR}/analytical_apps/pagerank
                  ${EXAMPLES_DIR}/analytical_apps/sssp
                  ${EXAMPLES_DIR}/analytical_apps/wcc
        DESTINATION include/grape/analytical_apps
        FILES_MATCHING
        PATTERN "*.h"
)
install(DIRECTORY ${EXAMPLES_DIR}/gnn_sampler
        DESTINATION include/grape/analytical_apps
        FILES_MATCHING
        PATTERN "*.h"
)

configure_file(libgrapelite-config.in.cmake
               "${PROJECT_BINARY_DIR}/libgrapelite-config.cmake" @ONLY
)
configure_file(libgrapelite-config-version.in.cmake
               "${PROJECT_BINARY_DIR}/libgrapelite-config-version.cmake" @ONLY
)
install(FILES "${PROJECT_BINARY_DIR}/libgrapelite-config.cmake"
              "${PROJECT_BINARY_DIR}/libgrapelite-config-version.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libgrapelite
)
install(EXPORT libgrapelite-targets
        FILE libgrapelite-targets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libgrapelite
)

# ------------------------------------------------------------------------------
# format code
# ------------------------------------------------------------------------------
file(GLOB_RECURSE FILES_NEED_FORMAT "grape/*.cc"
        "grape/*.h"
        "examples/*.h"
        "examples/*.cc")
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

# ------------------------------------------------------------------------------
# generate docs
# ------------------------------------------------------------------------------
if(BUILD_LIBGRAPELITE_DOCS)
    add_custom_target(doc COMMAND doxygen "${CMAKE_CURRENT_SOURCE_DIR}/misc/doc-config"
            COMMENT "Generating docs."
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            VERBATIM)
endif()

