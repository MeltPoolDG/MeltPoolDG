set(TEST_LIBRARIES ${MELTPOOLDG_LIB})
set(TEST_TARGET ${MELTPOOLDG_LIB})
deal_ii_pickup_tests()

file(GLOB UNIT_TEST_CMAKE_COMMANDS "cmake/*.cmake")
foreach(CMAKE_COMMAND ${UNIT_TEST_CMAKE_COMMANDS})
    include(${CMAKE_COMMAND})
endforeach()

# In order to run tests with MPI, we need to know the MPI executable and its semantics.
find_package(MPI REQUIRED)

mpdg_setup_mpi_unit_tests(
   N_MPI_PROCESSES
   1
   TEST_SOURCE_FILES
   material_data.cpp
)

add_subdirectory(cut)
add_subdirectory(flow)
add_subdirectory(fluid_structure_interaction)
add_subdirectory(heat)
add_subdirectory(level_set)
add_subdirectory(linear_algebra)
add_subdirectory(particles)
add_subdirectory(phase_change)
add_subdirectory(post_processing)
add_subdirectory(species_transport)
add_subdirectory(time_integration)
add_subdirectory(utilities)
