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

set_source_files_properties(main.cpp state.cpp clue.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(wordle-gpu.exe PRIVATE hip::device)

# Put the executable in build directory
set_target_properties(wordle-gpu.exe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_compile_options(wordle-gpu.exe PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>
)

target_link_options(wordle-gpu.exe PRIVATE
  -fgpu-rdc
)

# Executable depends on generated constants.h
add_dependencies(wordle-gpu.exe generate_constants)
