#!/bin/bash

# Copyright (c) 2024 Michael E. Rowan.
#
# This file is part of Wordle-GPU.
#
# License: MIT.

# Output
constants_header="./src/constants.h"

# Guesses
guesses="./src/guesses.txt"

if [[ ! -f "$guesses" ]]; then
  echo "Guesses file $guesses not found."
  exit 1
fi

words=()
while IFS= read -r line; do
  # Skip blank lines or comment
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  words+=("$line")
done < "$guesses"

if [[ ${#words[@]} -eq 0 ]]; then
  echo "Guesses file $guesses is empty."
  exit 1
fi

letters_per_word=${#words[0]}
number_of_words_in_guesses=${#words[@]}

for word in "${words[@]}"; do
  if [[ ${#word} -ne $letters_per_word ]]; then
    echo "All words in guesses file $guesses must be same length."
    exit 1
  fi
done

# Solutions
solutions="./src/solutions.txt"

if [[ ! -f "$solutions" ]]; then
  echo "Solutions file $solutions not found."
  exit 1
fi

words=()
while IFS= read -r line; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  words+=("$line")
done < "$solutions"

number_of_words_in_solutions=${#words[@]}

for word in "${words[@]}"; do
  if [[ ${#word} -ne $letters_per_word ]]; then
    echo "All words in solutions file $solutions must be same length as guesses in $guesses"
    exit 1
  fi
done

cat <<EOL > "$constants_header"
/* Copyright (c) 2024 Michael E. Rowan.
 *
 * This file is part of Wordle-GPU.
 *
 * License: MIT.
 */

#ifndef WORDLE_GPU_SRC_CONSTANTS_H
#define WORDLE_GPU_SRC_CONSTANTS_H

constexpr uint32_t LETTERS_PER_WORD = $letters_per_word;
constexpr uint32_t NUMBER_OF_GUESSES = $number_of_words_in_guesses;
constexpr uint32_t NUMBER_OF_SOLUTIONS = $number_of_words_in_solutions;

#endif // WORDLE_GPU_SRC_CONSTANTS_H
EOL
