cmake_minimum_required(VERSION 3.18)
project(opt_core_refactor LANGUAGES CXX)

# Ensure we can locate Python and pybind11 in an isolated PEP 517 build.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Robustly locate pybind11's CMake config even when installed via pip in a build env.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE _PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(_PYBIND11_CMAKE_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${_PYBIND11_CMAKE_DIR}")
endif()

find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(src/cpp)
