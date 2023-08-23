project (libgrape-ldbc CXX)
cmake_minimum_required (VERSION 2.8)

if (NOT LIBGRAPE_HOME)
    set(LIBGRAPE_HOME $ENV{LIBGRAPE_HOME})
endif ()

if (NOT LIBGRAPE_HOME)
    message(FATAL_ERROR "Before running cmake, please specify $LIBGRAPE_HOME in the environment or using -D")
endif ()

add_definitions(-DLIBGRAPE_HOME=${LIBGRAPE_HOME})

if (GRANULA)
    add_definitions(-DGRANULA=1)
endif ()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -ldl -pthread")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pedantic -Wextra -g3")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

include("${LIBGRAPE_HOME}/cmake/FindGlog.cmake")
include_directories(SYSTEM ${GLOG_INCLUDE_DIR})

include("${LIBGRAPE_HOME}/cmake/FindGFlags.cmake")
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIR})

set (CMAKE_EXPORT_COMPILE_COMMANDS 1)

include_directories (SYSTEM ${LIBGRAPE_HOME})
include_directories (SYSTEM ${LIBGRAPE_HOME}/thirdparty)
link_directories (${LIBGRAPE_HOME}/build)

add_executable(libgrape-ldbc ${LIBGRAPE_HOME}/examples/analytical_apps/flags.cc
                             ${LIBGRAPE_HOME}/examples/analytical_apps/run_app.cc)
target_include_directories(libgrape-ldbc PRIVATE
                           ${LIBGRAPE_HOME}/examples/analytical_apps)
target_link_libraries(libgrape-ldbc grape-lite ${MPI_CXX_LIBRARIES}
                      ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${CMAKE_DL_LIBS})
