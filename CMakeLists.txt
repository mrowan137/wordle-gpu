# Copyright (c) 2024 Michael E. Rowan.
#
# This file is part of Wordle-GPU.
#
# License: MIT.

cmake_minimum_required(VERSION 3.21)

project(wordle-gpu LANGUAGES CXX HIP)

set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCM installation")
list(PREPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

find_package(hip REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx906 CACHE STRING "HIP GPU architectures")
endif()

set(OTHER_FLAGS -Wall -Wextra -Werror -O3 -funsafe-math-optimizations -ffast-math)
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${OTHER_FLAGS}>")
add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:${OTHER_FLAGS}>")
include_directories(${CMAKE_SOURCE_DIR})

# Generate constants from input guesses & solutions
add_custom_command(
  OUTPUT ${CMAKE_SOURCE_DIR}/src/constants.h
  COMMAND ${CMAKE_SOURCE_DIR}/util/generate_constants.sh
  DEPENDS ${CMAKE_SOURCE_DIR}/src/guesses.txt
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Generating constants.h"
)

add_custom_target(generate_constants ALL
  DEPENDS ${CMAKE_SOURCE_DIR}/src/constants.h
)

# Get the wavefront dependent on architecture
if(CMAKE_HIP_ARCHITECTURES MATCHES "^gfx9")
  add_compile_definitions(DETECTED_WAVEFRONT_SIZE=64)
elseif(CMAKE_HIP_ARCHITECTURES MATCHES "^gfx1[01]")
  add_compile_definitions(DETECTED_WAVEFRONT_SIZE=32)
else()
  message(FATAL_ERROR "Unsupported architecture ${CMAKE_HIP_ARCHITECTURES}")
endif()

add_subdirectory(src)
