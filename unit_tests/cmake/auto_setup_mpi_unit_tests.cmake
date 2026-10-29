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

        # Start all ranks in a single app context. The MPMD syntax (":")
        # crashes mpiexec in the Open MPI of Ubuntu 26.04 (resolute).
        # Only rank 0 writes the XML report, see gtest_mpi_main.cpp.
        # We (at least for now) do not use gtest_discover_tests here because it was not designed for MPI
        add_test(
            NAME ${TEST_TARGET}
            COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${ARG_N_MPI_PROCESSES}
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:${TEST_TARGET}>
                    --gtest_output=xml:unittest_reports/${TEST_TARGET}_report.xml
                    ${MPIEXEC_POSTFLAGS}
        )

        set_property(TEST ${TEST_TARGET} PROPERTY LABELS UnitTest MPI)
        set_property(TEST ${TEST_TARGET} PROPERTY PROCESSORS ${ARG_N_MPI_PROCESSES})
    endforeach()
endfunction()
