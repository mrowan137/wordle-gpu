# Copyright (c) 2024 Michael E. Rowan.
#
# This file is part of Wordle-GPU.
#
# License: MIT.

# Add project files
add_executable(
  wordle-gpu.exe
  main.cpp
  state.cpp
  clue.cpp)

# Link stdc++fs library to get absolute paths
target_link_libraries(wordle-gpu.exe stdc++fs)

# Link HIP libraries
if(HIP_FOUND)
  target_link_libraries(wordle-gpu.exe ${HIP_LIBRARIES})
endif()

# Put the executable in build directory
set_target_properties(wordle-gpu.exe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Executable depends on generated constants.h
add_dependencies(wordle-gpu.exe generate_constants)
