mpdg_setup_serial_unit_tests(
    TEST_SOURCE_FILES
    dg_generic_convection_diffusion_worker.cpp
    limiters.cpp
    cell_monitor_01.cpp
)

mpdg_setup_mpi_unit_tests(
   N_MPI_PROCESSES
   4
   TEST_SOURCE_FILES
   triangulation_utils.cpp
)
