/* Copyright (c) 2024 Michael E. Rowan.
 *
 * This file is part of Wordle-GPU.
 *
 * License: MIT.
 */

#include "clue.h"

#include <hip/hip_runtime.h>

#include <cstdint>

#include "state.h"

__host__ __device__ Clue::Clue(const char *input) noexcept
  : k_input_(const_cast<char *>(input)),
    k_letters_(const_cast<char *>(input + State::k_letters_per_word)),
    k_colors_(const_cast<char *>(input)) {}

__host__ __device__ void Clue::UpdateData(const char *input) noexcept {
  k_input_ = const_cast<char *>(input);
  k_colors_ = const_cast<char *>(input);
  k_letters_ = const_cast<char *>(input + State::k_letters_per_word);
}

__host__ __device__ bool Clue::Disallows(const char *word) const noexcept {
  uint8_t word_letter_counts[26] = {};
  EncodeLetters_(word_letter_counts, word);
  for (uint8_t i = 0; i < State::k_letters_per_word; ++i) {
    const uint8_t j = k_letters_[i] - 'a';
    const char word_letter = word[i];
    const char clue_letter = k_letters_[i];
    const char clue_color = k_colors_[i];

    if (clue_color == 'g') {
      if (word_letter != clue_letter) {
        return true;
      }
      word_letter_counts[j] -= 1;
    }
  }
#pragma unroll State::k_letters_per_word
  for (uint8_t i = 0; i < State::k_letters_per_word; ++i) {
    const uint8_t j = k_letters_[i] - 'a';
    const char word_letter = word[i];
    const char clue_letter = k_letters_[i];
    const char clue_color = k_colors_[i];
    if (clue_color == 'y') {
      if (word_letter == clue_letter || word_letter_counts[j] <= 0) {
        return true;
      }
      word_letter_counts[j] -= 1;
    } else if (clue_color == 'x') {
      if (word_letter_counts[j] > 0) {
        return true;
      }
    }
  }
  return false;
}

__host__ __device__ void Clue::EncodeClueInput(const char *guess,
                                               const char *solution,
                                               char *encoded_input) noexcept {
  uint8_t encoded_solution[26] = {};
  EncodeLetters_(encoded_solution, solution);
#pragma unroll State::k_letters_per_word
  for (uint8_t i = 0; i < State::k_letters_per_word; ++i) {
    const char guess_letter = guess[i];
    const char solution_letter = solution[i];
    const uint8_t j = guess_letter - 'a';
    encoded_input[i + State::k_letters_per_word] = guess_letter;
    if (encoded_solution[j] > 0) {
      encoded_input[i] = (guess_letter == solution_letter) ? 'g' : 'y';
      encoded_solution[j] -= 1;
    } else {
      encoded_input[i] = 'x';
    }
  }
}

__host__ __device__ void Clue::EncodeLetters_(uint8_t *encoded_letters,
                                              const char *letters) noexcept {
#pragma unroll State::k_letters_per_word
  for (uint8_t i = 0; i < State::k_letters_per_word; ++i) {
    encoded_letters[letters[i] - 'a'] += 1;
  }
}
