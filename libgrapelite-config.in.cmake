# - Config file for the libgrape-lite package
#
# It defines the following variables
#
#  LIBGRAPELITE_INCLUDE_DIR         - include directory for libgrape-lite
#  LIBGRAPELITE_INCLUDE_DIRS        - include directories for libgrape-lite
#  LIBGRAPELITE_LIBRARIES           - libraries to link against

set(LIBGRAPELITE_HOME "${CMAKE_CURRENT_LIST_DIR}/../../..")
include("${CMAKE_CURRENT_LIST_DIR}/libgrapelite-targets.cmake")

set(LIBGRAPELITE_LIBRARIES grape-lite)
set(LIBGRAPELITE_INCLUDE_DIR "${LIBGRAPELITE_HOME}/include")
set(LIBGRAPELITE_INCLUDE_DIRS "${LIBGRAPELITE_INCLUDE_DIR}")
