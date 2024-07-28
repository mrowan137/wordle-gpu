# Copyright (c) 2024 Michael E. Rowan.
#
# This file is part of Wordle-GPU.
#
# License: MIT.

cmake_minimum_required(VERSION 3.12)

project(wordle-gpu LANGUAGES CXX)

# Find HIP
find_package(HIP REQUIRED)
if(HIP_FOUND)
  include_directories(${HIP_INCLUDE_DIRS})
  set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(HIP_OFFLOAD_ARCH gfx90a)
  set(HIP_CXX_FLAGS "--offload-arch=${HIP_OFFLOAD_ARCH} -fgpu-rdc -O3 -funsafe-math-optimizations -ffast-math")
  set(OTHER_CXX_FLAGS "-Wall -Wextra -Werror")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CXX_FLAGS} ${OTHER_CXX_FLAGS}")
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
  if(${HIP_OFFLOAD_ARCH} MATCHES "gfx9.*")
    add_definitions(-DDETECTED_WAVEFRONT_SIZE=64)
  elseif(${HIP_OFFLOAD_ARCH} MATCHES "gfx10.*")
    add_definitions(-DDETECTED_WAVEFRONT_SIZE=32)
  else()
    message(FATAL_ERROR "Unsupported architecture ${HIP_OFFLOAD_ARCH}")
  endif()

  add_subdirectory(src)
endif()
