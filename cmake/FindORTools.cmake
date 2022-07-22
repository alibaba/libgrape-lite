set(ORTOOLS_ROOT_DIR $ENV{ORTOOLS_ROOT_DIR} CACHE PATH "Folder contains Google OR-Tools")
message(STATUS "ortools root here : ${ORTOOLS_ROOT_DIR}")

if(NOT TARGET ortools::ortools)
    # Include directories
    find_path(ORTOOLS_INCLUDE_DIRS 
              NAME ortools 
              HINTS
              ${ORTOOLS_ROOT_DIR}
              PATH_SUFFIXES include/)
    message(STATUS "ortools found here : ${ORTOOLS_INCLUDE_DIRS}")

    ## Libraries
    set(LIB_TO_FIND
        CbcSolver #CBC_LNK
        Cbc
        OsiCbc
        Cgl
        ClpSolver
        Clp
        OsiClp
        Osi
        CoinUtils
        absl_bad_any_cast_impl #ABSL_LNK
        absl_bad_optional_access
        absl_bad_variant_access
        absl_base
        absl_city
        absl_civil_time
        absl_debugging_internal
        absl_demangle_internal
        #absl_dynamic_annotations
        absl_examine_stack
        absl_failure_signal_handler
        absl_graphcycles_internal
        absl_hash
        absl_hashtablez_sampler
        absl_int128
        absl_leak_check
        absl_malloc_internal
        #absl_optional
        absl_raw_hash_set
        absl_spinlock_wait
        absl_stacktrace
        absl_str_format_internal
        absl_strings
        absl_strings_internal
        absl_symbolize
        absl_synchronization
        absl_throw_delegate
        absl_time
        absl_time_zone
        protobuf #protobuf
        glog #glog
        gflags #gflags
        ortools #ortools
    )

    foreach(X ${LIB_TO_FIND})
        find_library(LIB_${X} 
                     NAME ${X} 
                     HINTS
                     ${ORTOOLS_ROOT_DIR}
                     PATH_SUFFIXES lib/)
        message(STATUS "${X} lib found here : ${LIB_${X}}")
        set(ORTOOLS_LIBRARIES ${ORTOOLS_LIBRARIES} ${LIB_${X}})
    endforeach()

    # Definitions
    set(ORTOOLS_DEFINITIONS -DUSE_CBC -DUSE_CLP -DUSE_BOP -DUSE_GLOP)

    add_library(ortools INTERFACE)
    add_library(ortools::ortools ALIAS ortools)
    target_include_directories(ortools INTERFACE ${ORTOOLS_INCLUDE_DIRS})
    target_link_libraries(ortools INTERFACE ${ORTOOLS_LIBRARIES})
    target_compile_options(ortools INTERFACE ${ORTOOLS_DEFINITIONS})

    include(FindPackageHandleStandardArgs)
    # handle the QUIETLY and REQUIRED arguments and set ortools to TRUE
    # if all listed variables are TRUE
    find_package_handle_standard_args(ortools
        REQUIRED_VARS ORTOOLS_LIBRARIES ORTOOLS_INCLUDE_DIRS)
endif()
