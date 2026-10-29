message("_________________________________________________________________________________")
message("                                                                                 ")
message("                              MeltPoolDG                                         ")
message("                                                                                 ")
message("_________________________________________________________________________________")

# this is the standard deal.II search mechanism, including check for Trilinos
# and p4est
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

project(MeltPoolDG)

#
# avoid in-source compilation
set(DEFAULT_OUT_OF_SOURCE_FOLDER "build")
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(
        FATAL_ERROR
        "In-source builds not allowed. Change to ${DEFAULT_OUT_OF_SOURCE_FOLDER} folder and run then cmake")
endif()

# ------------------------------------------------------------------------------
# set default build type
# ------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
    message("-- Setting to default CMAKE_BUILD_TYPE Debug")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# find deal.II and check that it is properly configured
# ------------------------------------------------------------------------------
find_package(deal.II 9.6 QUIET
    HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
if(NOT ${deal.II_FOUND})
    message(
        FATAL_ERROR
        "\n"
        "*** Could not locate deal.II. ***\n\n"
        "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
        "or set an environment variable \"DEAL_II_DIR\" that contains this path.")
endif()

message("-- deal.II found: ${DEAL_II_DIR}")

if(NOT DEAL_II_WITH_P4EST)
    message(
        FATAL_ERROR
        "\nThis library requires a deal.II installation built with support for p4est but it appears to be missing!\n"
    )
endif()

if(NOT DEAL_II_WITH_TRILINOS)
    message(
        FATAL_ERROR
        "\nThis library requires a deal.II installation built with support for Trilinos but it appears to be missing!\n"
    )
endif()

# ------------------------------------------------------------------------------
#  find adaflo
# ------------------------------------------------------------------------------
find_library(ADAFLO adaflo HINTS ${ADAFLO_LIB})
if(ADAFLO)
    add_compile_definitions(ADAFLO_SUPPORT)
    set(DEAL_II_WITH_ADAFLO_SUPPORT ON)
    set(MPDG_ENABLE_ADAFLO TRUE)
    message(STATUS "Adaflo found: ${ADAFLO}")
else()
    set(MPDG_ENABLE_ADAFLO FALSE)
    message(STATUS "Adaflo not found. Setting up project without adaflo support.")
endif()

# ------------------------------------------------------------------------------
# build meltpooldg library
# ------------------------------------------------------------------------------
deal_ii_initialize_cached_variables()
deal_ii_query_git_information(MELTPOOLDG)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/meltpooldg/utilities/revision.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/meltpooldg/utilities/revision.hpp)

include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/include)

file(GLOB_RECURSE sources
    "${CMAKE_SOURCE_DIR}/source/**/*.cpp"
    "${CMAKE_SOURCE_DIR}/simulations/*.cpp")


set(MELTPOOLDG_LIB "meltpooldg")

add_library(${MELTPOOLDG_LIB} ${sources})

if(ADAFLO)
    target_compile_definitions(${MELTPOOLDG_LIB} PUBLIC MPDG_ENABLE_ADAFLO)
    target_include_directories(${MELTPOOLDG_LIB} PUBLIC ${ADAFLO_INCLUDE})
    target_link_libraries(${MELTPOOLDG_LIB} ${ADAFLO})
endif()

target_compile_definitions(${MELTPOOLDG_LIB} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

deal_ii_setup_target(${MELTPOOLDG_LIB})

if(${DEAL_II_VERSION_MAJOR} GREATER_EQUAL 10)
    message("-- Enabling early deprecation via #define PUBLIC DEAL_II_DEPRECATED_EARLY=[[deprecated]]")
    add_compile_definitions(${MELTPOOLDG_LIB} PUBLIC DEAL_II_DEPRECATED_EARLY=[[deprecated]])
endif()

# ------------------------------------------------------------------------------
# benchmarks
# ------------------------------------------------------------------------------
option(MPDG_ENABLE_BENCHMARKING "Configure benchmarks" OFF)
if(MPDG_ENABLE_BENCHMARKING)
    message(STATUS "Benchmarks: enabled")
    # Deactivate internal google benchmark tests
    set(BENCHMARK_ENABLE_TESTING OFF)
    # Ensure that google benchmark debug assertions are disabled
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -DNDEBUG")
    FetchContent_Declare(
        google_benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.3)

    FetchContent_MakeAvailable(google_benchmark)
    include_directories(${google_benchmark_SOURCE_DIR}/src)
    add_subdirectory(benchmarks)
else()
    message(STATUS "Benchmarks: disabled")
endif()

# ------------------------------------------------------------------------------
# unit tests of the library
# ------------------------------------------------------------------------------
option(MPDG_ENABLE_TESTING "Configure tests" ON)

if(MPDG_ENABLE_TESTING)
    message(STATUS "Tests: enabled")

    fetchcontent_declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
    )

    # Always build GoogleTest as a static library. Together with
    # --exclude-libs in unit_tests/cmake/auto_setup_mpi_unit_tests.cmake,
    # this prevents Trilinos' libgtest.so.14 (loaded transitively via
    # deal.II) from binding to our GTest symbols (double free at exit).
    set(_meltpooldg_saved_bsl ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
    fetchcontent_makeavailable(googletest)
    set(BUILD_SHARED_LIBS ${_meltpooldg_saved_bsl})

    include(GoogleTest)

    set(DEAL_II_HAVE_TESTS_DIRECTORY ON)
    set(_n_tests "0")
    enable_testing()
    add_subdirectory(unit_tests)
else()
    message(STATUS "Tests: disabled")
endif()


# ------------------------------------------------------------------------------
# applications
# ------------------------------------------------------------------------------
add_subdirectory(applications)
if (EXISTS ${CMAKE_SOURCE_DIR}/applications-dev/CMakeLists.txt)
    add_subdirectory(applications-dev)
endif ()
