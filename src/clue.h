/* Copyright (c) 2024 Michael E. Rowan.
 *
 * This file is part of Wordle-GPU.
 *
 * License: MIT.
 */

#ifndef WORDLE_GPU_SRC_CLUE_H
#define WORDLE_GPU_SRC_CLUE_H

#include <hip/hip_runtime.h>

class Clue {
 public:
  __host__ __device__ explicit Clue(const char *input) noexcept;
  __host__ __device__ void UpdateData(const char *input) noexcept;
  __host__ __device__ bool Disallows(const char *word) const noexcept;
  static __host__ __device__ void EncodeClueInput(const char *guess,
                                                  const char *solution,
                                                  char *encoded_input) noexcept;

 private:
  static __host__ __device__ void EncodeLetters_(uint8_t *encoded_letters,
                                                 const char *letters) noexcept;
  char *k_input_;
  char *k_letters_;
  char *k_colors_;
};
#endif // WORDLE_GPU_SRC_CLUE_H
