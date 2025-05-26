cmake_minimum_required(VERSION 2.8)

project(libgrape-lite C CXX)

set(LIBGRAPELITE_MAJOR_VERSION 0)
set(LIBGRAPELITE_MINOR_VERSION 3)
set(LIBGRAPELITE_PATCH_VERSION 5)
set(LIBGRAPELITE_VERSION ${LIBGRAPELITE_MAJOR_VERSION}.${LIBGRAPELITE_MINOR_VERSION}.${LIBGRAPELITE_PATCH_VERSION})

# ------------------------------------------------------------------------------
# cmake options
# ------------------------------------------------------------------------------
option(USE_JEMALLOC "Whether to use jemalloc." OFF)
option(USE_HUGEPAGES "Whether to use hugepages." OFF)
option(BUILD_SHARED_LIBS "Whether to build libgrape-lite as shared library" ON)
option(PROFILING "Whether to enable profiling" OFF)
option(TRACKING_MEMORY "Whether to enable memory tracking" OFF)
option(WITH_ASAN "Build with Address Sanitizer" OFF)
option(BUILD_LIBGRAPELITE_DOCS "Build libgrape-lite documentation" ON)
option(BUILD_LIBGRAPELITE_TESTS "Build libgrape-lite test cases" ON)
option(WCC_USE_GID "Use global id as wcc component id" OFF)
option(USE_SIMD "Whether to use SIMD instructions" OFF)

if (USE_HUGEPAGES)
  if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Enable hugepage")
    add_definitions(-DUSE_HUGEPAGES)
  endif ()
endif ()

if (USE_SIMD)
  message(STATUS "Enable SIMD")
  add_definitions(-DUSE_BMISS_STTNI_INTERSECT)
  add_definitions(-DUSE_SIMD_SORT)
endif ()

if (PROFILING)
  message(STATUS "Enable profiling")
  add_definitions(-DPROFILING)
endif ()

if (TRACKING_MEMORY)
  message(STATUS "Enable memory tracking")
  add_definitions(-DTRACKING_MEMORY)
endif ()

if (WCC_USE_GID)
  add_definitions(-DWCC_USE_GID)
endif ()

include_directories(SYSTEM thirdparty)

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

message(STATUS "[libgrape-lite] will build in type: ${CMAKE_BUILD_TYPE}")

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Werror -Wl,-rpath,$ORIGIN -march=native")
endif ()
if (USE_SIMD)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif ()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g")

if (WITH_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -O1")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif ()


# find CUDA, 9.0 at least
find_package(CUDA 9 QUIET)
if (CUDA_VERSION VERSION_LESS "9.0")
  message(WARNING "Disable gpu support because cuda >= 9.0 not found")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  # find NCCL, 2.7 at least
  include("cmake/FindNCCL.cmake")
  set(NCCL_VERSION "${NCCL_MAJOR_VERSION}.${NCCL_MINOR_VERSION}")
  if (NCCL_VERSION VERSION_LESS "2.7")
    message(WARNING "Disable GPU support because NCCL >= 2.7 not found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  else ()
    option(WITH_CUDA "Whether to enable cuda support" ON)
    message(STATUS "Build with CUDA support")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

    include_directories(${CUDA_TOOLKIT_INCLUDE})
    set(CUDA_LIBS ${CUDA_TOOLKIT_TARGET_DIR}/lib64/stubs/libcuda.so
                  ${CUDA_TOOLKIT_TARGET_DIR}/lib64/libnvToolsExt.so
                  ${CUDA_TOOLKIT_TARGET_DIR}/lib64/libcudart.so)

    if(${CMAKE_VERSION} VERSION_LESS_EQUAL "3.13.4")
      cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
      message(STATUS "CUDA_ARCH = ${ARCH_FLAGS}")
      string(REPLACE "-gencode;" "--generate-code=" ARCH_FLAGS "${ARCH_FLAGS}")
      string(APPEND CMAKE_CUDA_FLAGS "${ARCH_FLAGS}")
    else()
      include(FindCUDA/select_compute_arch)
      CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
      string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
      string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
      string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
      SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
      message(STATUS "CUDA_ARCH = ${CUDA_ARCH_LIST}")
      set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
      list(GET CUDA_ARCH_LIST 0 CUDA_ARCH_CODE)
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CUDA_ARCH_CODE}")
    endif()

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-lineinfo;-Xcompiler;-ggdb;-std=c++14;--expt-extended-lambda)
    else ()
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++14;--expt-extended-lambda)
    endif ()

    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_SEPARABLE_COMPILATION OFF)
    message(STATUS "Host Compiler: ${CUDA_HOST_COMPILER}")

    include_directories(SYSTEM ${NCCL_INCLUDE_DIRS})
  endif()
endif ()

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

# install, and export
macro(install_libgrapelite_target target)
  install(TARGETS ${target}
          EXPORT libgrapelite-targets
          ARCHIVE DESTINATION lib
          LIBRARY DESTINATION lib
          RUNTIME DESTINATION bin
  )
endmacro()

# install, without export, as when we `find_package(libgrape-lite)`, we only want
# the library and headers exist.
macro(install_without_export_libgrapelite_target target)
  install(TARGETS ${target}
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
  install_without_export_libgrapelite_target(analytical_apps)

  file(GLOB TEST_FILES RELATIVE "${PROJECT_SOURCE_DIR}/tests" "${PROJECT_SOURCE_DIR}/tests/*.cc")
  foreach(f ${TEST_FILES})
    string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${f})
    set(T_NAME ${CMAKE_MATCH_1})
    message(STATUS "Found test - " ${T_NAME})
    if(BUILD_LIBGRAPELITE_TESTS)
      add_executable(${T_NAME} tests/${T_NAME}.cc)
    else()
      add_executable(${T_NAME} EXCLUDE_FROM_ALL tests/${T_NAME}.cc)
    endif()
    target_include_directories(${T_NAME} PRIVATE examples/analytical_apps)
    target_link_libraries(${T_NAME} PRIVATE grape-lite ${MPI_CXX_LIBRARIES}
                          ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${CMAKE_DL_LIBS})
  endforeach()

  if (WITH_CUDA)
    cuda_add_executable(gpu_analytical_apps examples/analytical_apps/flags.cc examples/analytical_apps/run_cuda_app.cu)
    target_include_directories(gpu_analytical_apps PRIVATE examples/analytical_apps)
    set_target_properties(gpu_analytical_apps PROPERTIES OUTPUT_NAME run_cuda_app)
    target_link_libraries(gpu_analytical_apps grape-lite ${GFLAGS_LIBRARIES} ${CUDA_LIBS} ${NCCL_LIBRARIES} ${CMAKE_DL_LIBS})
    install_without_export_libgrapelite_target(gpu_analytical_apps)
  endif ()
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
  target_link_libraries(gnn_sampler grape-lite ${MPI_CXX_LIBRARIES}
                        ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${CMAKE_DL_LIBS} ${RDKAFKA_LIBRARIES})
  install_without_export_libgrapelite_target(gnn_sampler)
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
                  ${PROJECT_SOURCE_DIR}/thirdparty/string_view
                  ${PROJECT_SOURCE_DIR}/thirdparty/pthash
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
)

if (WITH_CUDA)
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/thirdparty/cuda_hashmap
          DESTINATION include/cuda_hashmap
          FILES_MATCHING
          PATTERN "*.h"
  )
endif ()

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
                  ${EXAMPLES_DIR}/analytical_apps/cuda
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
                                    "examples/*.cc"
                                    "tests/*.h"
                                    "tests/*.cc")
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

