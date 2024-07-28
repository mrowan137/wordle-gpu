#!/bin/bash

# Copyright (c) 2024 Michael E. Rowan.
#
# This file is part of Wordle-GPU.
#
# License: MIT.

MODE=$1
export ROCR_VISIBLE_DEVICES=0
export HSA_XNACK=0
srun -n 1 -G 1 --unbuffered wordle-gpu.exe ../src/guesses.txt ../src/solutions.txt ${MODE}
