/* Copyright (c) 2024 Michael E. Rowan.
 *
 * This file is part of Wordle-GPU.
 *
 * License: MIT.
 */

#include "state.h"

int main(const int argc, const char *argv[]) {
  State state(argc, argv);

  while (!state.GameOver()) {
    state.QueryClueAndComputeBestGuess();
  }
  return 0;
}
