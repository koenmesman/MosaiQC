# This CMakeLists is included from the project root.
# It builds the pybind11 extension module and (optionally) a METIS test executable.

pybind11_add_module(_core MODULE
    core.cpp
    device.cpp
    circuit.cpp
    circuit_graph.cpp
    topology_graph.cpp
    GED_weighted.cpp
    QAP_fw.cpp
    METIS_solver.cpp
    partition_refiner.cpp
)

target_compile_features(_core PRIVATE cxx_std_17)

# optional flags
if (MSVC)
  target_compile_options(_core PRIVATE /O2)
else()
  target_compile_options(_core PRIVATE -O3)
endif()

# ---- OpenMP (optional) ----
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(_core PRIVATE OPTCORE_HAS_OPENMP=1)
else()
  target_compile_definitions(_core PRIVATE OPTCORE_HAS_OPENMP=0)
endif()

# ---- Find METIS for _core (optional but enabled when found) ----
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(METIS_PC QUIET metis)
endif()

find_path(METIS_INCLUDE_DIR metis.h
  HINTS ${METIS_PC_INCLUDE_DIRS}
  PATHS /usr/include /usr/local/include $ENV{HOME}/.local/include
)

find_library(METIS_LIBRARY metis
  HINTS ${METIS_PC_LIBRARY_DIRS}
  PATHS
    /usr/lib /usr/local/lib $ENV{HOME}/.local/lib
    /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
)

if(METIS_INCLUDE_DIR AND METIS_LIBRARY)
  target_compile_definitions(_core PRIVATE OPTCORE_HAS_METIS=1)
  target_include_directories(_core PRIVATE ${METIS_INCLUDE_DIR})
  target_link_libraries(_core PRIVATE ${METIS_LIBRARY})
else()
  target_compile_definitions(_core PRIVATE OPTCORE_HAS_METIS=0)
endif()

# install into Python package
install(TARGETS _core
  LIBRARY DESTINATION opt_core
  RUNTIME DESTINATION opt_core
  ARCHIVE DESTINATION opt_core
)

option(OPTCORE_BUILD_METIS_TEST "Build metis_partition_test executable" OFF)

if(OPTCORE_BUILD_METIS_TEST)
  add_executable(metis_test
    metis_partition_test.cpp
    circuit_graph.cpp
  )
  # Make sure Python is found (at top-level is fine too)
  find_package(Python COMPONENTS Development REQUIRED)

  # Ensure metis_test gets Python include dirs (for Python.h)
  target_link_libraries(metis_test PRIVATE Python::Python)

  # If circuit_graph.cpp includes pybind11 headers, also add:
  find_package(pybind11 CONFIG REQUIRED)
  target_link_libraries(metis_test PRIVATE pybind11::headers)

  target_include_directories(metis_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(metis_test PRIVATE cxx_std_17)
  if (MSVC)
    target_compile_options(metis_test PRIVATE /O2)
  else()
    target_compile_options(metis_test PRIVATE -O3)
  endif()

  # ---- Find METIS (header + library) ----
  # In isolated pip/scikit-build-core builds, the compiler/linker won't
  # automatically pick up non-standard prefixes. We therefore search common
  # system locations and (if available) use pkg-config hints.
  if(NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARY)
    message(FATAL_ERROR "METIS not found. METIS_INCLUDE_DIR=${METIS_INCLUDE_DIR}, METIS_LIBRARY=${METIS_LIBRARY}")
  endif()

  target_include_directories(metis_test PRIVATE ${METIS_INCLUDE_DIR})
  target_link_libraries(metis_test PRIVATE ${METIS_LIBRARY})
endif()
