file(GLOB CUSTOM_BENCHMARK_CMAKE_COMMANDS "cmake/*.cmake")
foreach (BENCHMARK_COMMAND ${CUSTOM_BENCHMARK_CMAKE_COMMANDS})
    include(${BENCHMARK_COMMAND})
endforeach ()

# ##############################################################################
# benchmark library (utilities)
# ##############################################################################
set(BENCHMARK_LIBRARY "mpdg_benchmark_library")
file(GLOB BENCHMARK_SOURCES "${google_benchmark_SOURCE_DIR}/src/*.cc")
# needs to be removed as it is already a part of benchmark::benchmark
list(REMOVE_ITEM BENCHMARK_SOURCES "${google_benchmark_SOURCE_DIR}/src/benchmark_main.cc")
add_library(${BENCHMARK_LIBRARY} benchmark_util.cpp ${BENCHMARK_SOURCES})
deal_ii_setup_target(${BENCHMARK_LIBRARY})
target_link_libraries(${BENCHMARK_LIBRARY} benchmark::benchmark)

# ##############################################################################
# add benchmark executables
# ##############################################################################
add_subdirectory(flow)
add_subdirectory(fluid_structure_interaction)
add_subdirectory(particles)
add_subdirectory(utilities)
