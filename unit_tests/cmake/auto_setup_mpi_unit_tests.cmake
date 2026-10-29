# Removes Trilinos' bundled Google Test from the entire (transitive) link
# interface of a target. Our own GoogleTest 1.17.0 (GTest::*) is left
# untouched; only Trilinos targets "Gtest::*" and file paths to
# libgtest*/libgmock* are filtered out.
function(meltpooldg_strip_trilinos_gtest start_target)
  set(_queue ${start_target})
  set(_visited "")

  while(_queue)
    list(POP_FRONT _queue _tgt)

    # Properties can only be set on the original target, not on an alias.
    get_target_property(_aliased ${_tgt} ALIASED_TARGET)
    if(_aliased)
      set(_tgt ${_aliased})
    endif()
    if(_tgt IN_LIST _visited)
      continue()
    endif()
    list(APPEND _visited ${_tgt})

    set(_props INTERFACE_LINK_LIBRARIES)
    get_target_property(_imported ${_tgt} IMPORTED)
    get_target_property(_type ${_tgt} TYPE)
    if(NOT _imported AND NOT _type STREQUAL "INTERFACE_LIBRARY")
      list(APPEND _props LINK_LIBRARIES)
    endif()

    foreach(_prop IN LISTS _props)
      get_target_property(_libs ${_tgt} ${_prop})
      if(NOT _libs)
        continue()
      endif()

      set(_kept "")
      foreach(_lib IN LISTS _libs)
        # Unwrap $<LINK_ONLY:x>; leave other generator expressions untouched.
        string(REGEX REPLACE "^\\$<LINK_ONLY:([^$<>]*)>$" "\\1" _item "${_lib}")
        if(NOT _item MATCHES "[<>]" AND
           _item MATCHES "^Gtest::|/libg(test|mock)[._][^/]*$")
          message(STATUS "[gtest-fix] ${_tgt} (${_prop}): removing ${_lib}")
        else()
          list(APPEND _kept "${_lib}")
          if(TARGET "${_item}")
            list(APPEND _queue "${_item}")
          endif()
        endif()
      endforeach()

      if(NOT "${_kept}" STREQUAL "${_libs}")
        set_property(TARGET ${_tgt} PROPERTY ${_prop} "${_kept}")
      endif()
    endforeach()
  endwhile()
endfunction()

# Creates MPI-enabled unit test executables using GoogleTest. Each source file 
# is compiled into its own executable, linked against GTest, GMock, deal.II, 
# and the meltpooldg library, and registered with CTest to be run via mpiexec.
#
# Example:
#   mpdg_setup_mpi_unit_tests(
#     N_MPI_PROCESSES 4
#     TEST_SOURCE_FILES
#       test_1.cpp
#       test_2.cpp
#   )
function(mpdg_setup_mpi_unit_tests)
    set(OPTIONS "")
    set(ONE_VALUE_ARGS N_MPI_PROCESSES)
    set(MULTI_VALUE_ARGS TEST_SOURCE_FILES)
    set(ARG_PREFIX "ARG")
    cmake_parse_arguments(PARSE_ARGV 0 "${ARG_PREFIX}" "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}")

    # Check validity of number of MPI processes
    if(NOT ARG_N_MPI_PROCESSES MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "N_MPI_PROCESSES must be a positive integral value.")
    endif()

    # Check if file list is not empty assuming that the user might not want to call this function with an empty list
    if(NOT ARG_TEST_SOURCE_FILES)
        message(WARNING "An empty list of test source files was provided to mpdg_setup_mpi_unit_tests. This call will therefore not create any tests and is obsolete.")
    endif()

    foreach(TEST_SRC ${ARG_TEST_SOURCE_FILES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME)
        string(REPLACE ".cpp" "" TEST_TARGET ${TEST_NAME})
        add_executable(${TEST_TARGET} ${CMAKE_SOURCE_DIR}/unit_tests/test_utils/gtest_mpi_main.cpp ${TEST_SRC})

        target_link_libraries(${TEST_TARGET} GTest::gtest GTest::gmock)
        deal_ii_setup_target(${TEST_TARGET})
        target_link_libraries(${TEST_TARGET} ${MELTPOOLDG_LIB})

        # Drop Trilinos' GTest from the link line and do not export the
        # symbols of the static GTest archives (see top-level CMakeLists.txt).
        meltpooldg_strip_trilinos_gtest(${TEST_TARGET})
        target_link_options(${TEST_TARGET} PRIVATE
          "LINKER:--exclude-libs,libgtest.a:libgmock.a")

        set(MPI_ARGUMENTS
            ${MPIEXEC_NUMPROC_FLAG}
            1
            $<TARGET_FILE:${TEST_TARGET}>
            --gtest_output=xml:unittest_reports/${TEST_TARGET}_report.xml)

        if(ARG_N_MPI_PROCESSES GREATER 1)
            math(EXPR _N_REMAINING_PROCESSES "${ARG_N_MPI_PROCESSES} - 1")
            list(APPEND MPI_ARGUMENTS ":" ${MPIEXEC_NUMPROC_FLAG} ${_N_REMAINING_PROCESSES} $<TARGET_FILE:${TEST_TARGET}>)
        endif()

        # We (at least for now) do not use gtest_discover_tests here because it was not designed for MPI
        add_test(
            NAME ${TEST_TARGET}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPI_ARGUMENTS}
        )

        set_property(TEST ${TEST_TARGET} PROPERTY LABELS UnitTest MPI)
        set_property(TEST ${TEST_TARGET} PROPERTY PROCESSORS ${ARG_N_MPI_PROCESSES})
    endforeach()
endfunction()
