/* Copyright (c) 2024 Michael E. Rowan.
 *
 * This file is part of Wordle-GPU.
 *
 * License: MIT.
 */

#include "state.h"

#include <hip/hip_runtime.h>

#include <cstdint>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "clue.h"
#include "host_device_buffer.h"
#include "utilities.h"

__global__ void CountDisallowedWords_(const uint16_t guess_idx,
                                      const char *__restrict__ guesses,
                                      const char *__restrict__ solutions,
                                      const uint16_t N,
                                      uint32_t *__restrict__ counts) {
  const uint32_t tid = threadIdx.x;
  const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t local_stride = blockDim.x;
  const uint32_t global_stride = blockDim.x * State::k_number_of_blocks;
  const uint32_t number_of_solution_letters = N;
  const uint32_t number_of_solutions =
    number_of_solution_letters / State::k_letters_per_word;
  extern __shared__ char local_solutions[];
  for (uint32_t i = tid; i < number_of_solution_letters; i += local_stride) {
    local_solutions[i] = solutions[i];
  }
  __syncthreads();

  const char *guess = &guesses[guess_idx * State::k_letters_per_word];
  char *test = nullptr;
  char *solution =
    &local_solutions[(gid % number_of_solutions) * State::k_letters_per_word];

  // Count words eliminated by guess: all combinations (guess, test, solution)
  uint16_t tally = 0;
  char encoded_input[2 * State::k_letters_per_word] = {};
  Clue::EncodeClueInput(guess, solution, encoded_input);
  Clue clue(encoded_input);
  for (uint32_t idx = gid; idx < number_of_solutions * number_of_solutions;
       idx += global_stride) {
    solution =
      &local_solutions[(idx % number_of_solutions) * State::k_letters_per_word];
    Clue::EncodeClueInput(guess, solution, encoded_input);
    clue.UpdateData(encoded_input);
    test =
      &local_solutions[(idx / number_of_solutions) * State::k_letters_per_word];
    tally += static_cast<int>(clue.Disallows(test));
  }

#pragma unroll utilities::LogBaseTwo(State::k_wavefront_size / 2)
  for (uint16_t w = State::k_wavefront_size / 2; w > 0; w >>= 1) {
    tally += __shfl_down(tally, w);
  }

  if (tid % State::k_wavefront_size == 0 &&
      gid < number_of_solutions * number_of_solutions) {
    atomicAdd(&counts[guess_idx], tally);
  }
}

State::State(const int argc, const char *argv[])
  : clues_({}), solution_index_map_({}), guesses_and_counts_({}) {
  Initialize_(argc, argv);
}

State::~State() {
  CleanupHipResource_(streams_, hipStreamDestroy);
  CleanupHipResource_(kernel_completion_events_, hipEventDestroy);
}

void State::QueryClueAndComputeBestGuess() {
  std::string clue;
  std::cout << '\n'
            << "[Round " << Round_()
            << (hard_mode_ ? " (hard mode)" : " (easy mode)") << "] " << '\n'
            << "  Enter current clue: ";
  std::cin >> clue;
  if (clue == "q") {
    quit_ = true;
    return;
  }

  if (ClueIsValid_(clue)) {
    ApplyClue_(clue);
    BestGuess_();
  }
}

bool State::GameOver() const {
  if (quit_) {
    return true;
  }

  if (solutions_.h_buffer.size() == k_letters_per_word || tries_ == 7) {
    std::cout << '\n';
    std::cout << "[End]" << '\n';
    std::cout << "  Game over!" << '\n';
    std::cout << "  Guesses: ";
    for (auto it = clues_.begin() + 1; it != clues_.end(); ++it) {
      std::cout << (*it) << " ";
    }
    std::cout << '\n';
    if (solutions_.h_buffer.size() == k_letters_per_word) {
      const std::string solution(solutions_.h_buffer.data(),
                                 k_letters_per_word);
      std::cout << "  Solution: " << solution << '\n';
    }
    std::cout << '\n';
    return true;
  }
  return false;
}

void State::Initialize_(const int argc, const char *argv[]) {
  CheckCommandLine_(argc, argv);
  std::cout << '\n' << "[Initialize]" << '\n';
  ParseCommandLine_(argc, argv);
  LoadWords_(path_to_guesses_, k_number_of_guesses, guesses_);
  LoadWords_(path_to_solutions_, k_number_of_solutions, solutions_);
  InitializeCounts_();
  InitializeHipResource_(streams_, k_number_of_streams, hipStreamCreate);
  InitializeHipResource_(kernel_completion_events_, NumberOfGuesses_(),
                         hipEventCreate);
}

void State::CheckCommandLine_(const int argc, const char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " /path/to/guesses.txt /path/to/solutions.txt [--hard]"
              << '\n';
    exit(EXIT_FAILURE);
  }
}

void State::ParseCommandLine_(const int argc, const char *argv[]) {
  path_to_guesses_ =
    std::filesystem::absolute(static_cast<std::filesystem::path>(argv[1]));
  path_to_solutions_ =
    std::filesystem::absolute(static_cast<std::filesystem::path>(argv[2]));
  hard_mode_ = ((argc == 4) && (static_cast<std::string>(argv[3]) == "--hard"));
  std::cout << "  Difficulty: " << (hard_mode_ ? "hard" : "easy") << '\n';
}

void State::LoadWords_(const std::string &path_to_words,
                       const uint32_t number_of_words,
                       HostDeviceBuffer<char> &words) {
  std::ifstream words_file(path_to_words);
  if (!words_file.is_open()) {
    std::cerr << "Failed to open file: " << path_to_words << '\n';
    exit(EXIT_FAILURE);
  }

  std::string line;
  words.h_buffer.reserve(k_letters_per_word * number_of_words);
  while (std::getline(words_file, line)) {
    // Skip blank lines or comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Line with correct length is a word
    if (line.length() == k_letters_per_word) {
      std::copy(line.begin(), line.end(), std::back_inserter(words.h_buffer));
    }
  }

  words.SyncHostToDevice();
  std::cout << "  Loaded words from: " << path_to_words << '\n';
}

void State::InitializeCounts_() {
  counts_.h_buffer.reserve(NumberOfGuesses_());
  counts_.h_buffer.resize(NumberOfGuesses_());
  counts_.SyncHostToDevice();
}

void State::ResetCounts_() {
  counts_.h_buffer.resize(NumberOfGuesses_());
  std::fill(counts_.h_buffer.begin(), counts_.h_buffer.end(), 0);
  counts_.SyncHostToDevice();
}

uint16_t State::NumberOfGuesses_() const noexcept {
  return guesses_.h_buffer.size() / k_letters_per_word;
}
uint16_t State::NumberOfSolutions_() const noexcept {
  return solutions_.h_buffer.size() / k_letters_per_word;
}

bool State::ClueIsValid_(const std::string &clue) const {
  // Length check: 5 colors, 5 letters
  if (clue.length() != 2 * k_letters_per_word) {
    PrintClueHelp_();
    return false;
  }

  // First round accepts '??????????'
  if (Round_() == 0) {
    bool all_unknown = true;
    for (uint8_t i = 0; i < 2 * k_letters_per_word; ++i) {
      all_unknown &= (clue[i] == '?');
    }
    if (!all_unknown) {
      PrintClueHelp_();
      return false;
    }
    return true;
  }

  // Valid colors are 'x' (gray), 'y' (yellow), 'g' (green)
  for (uint8_t i = 0; i < k_letters_per_word; ++i) {
    if (clue[i] != 'x' && clue[i] != 'y' && clue[i] != 'g') {
      PrintClueHelp_();
      return false;
    }
  }

  // Clue must be from guess list
  bool word_found_in_guesses = false;
  uint32_t i = 0;
  while (!word_found_in_guesses && i < guesses_.h_buffer.size()) {
    uint32_t j = i;
    while (j < i + k_letters_per_word &&
           clue[j - i + k_letters_per_word] == guesses_.h_buffer[j]) {
      j += 1;
    }
    word_found_in_guesses = (j == i + k_letters_per_word);
    i += k_letters_per_word;
  }
  if (!word_found_in_guesses) {
    PrintClueHelp_();
    return false;
  }

  return true;
}

void State::PrintClueHelp_() {
  const std::string &help_message = utilities::StringFormat(
    R"(

  Help menu
  ---------

  Enter a clue to compute best guess. The valid clue for the 0th round is
  '%s', indicating that no colors or letters are yet known. For
  subsequent rounds, valid clue format is:
  - %u characters
  - each of first %u characters is 'x' (gray), 'y' (yellow), 'g' (green)
  - the last %u characters form a word from the allowed guesses

  Example:
  - guess = 'debug' and solution = 'beans' → clue = 'xgyxxdebug'
  - second letter is correct with correct position, third letter is correct
    with an incorrect position, and the other letters are incorrect

  Press q to exit the program.
  )",
    std::string(2 * k_letters_per_word, '?').data(), 2 * k_letters_per_word,
    k_letters_per_word, k_letters_per_word);
  std::cout << help_message << '\n';
}

void State::ApplyClue_(const std::string &input) noexcept {
  // Input is assumed valid
  const Clue clue(input.data());
  clues_.emplace_back(input.begin() + k_letters_per_word, input.end());
  ApplyClueToWords_(clue, solutions_);
  if (hard_mode_) {
    ApplyClueToWords_(clue, guesses_);
  }
  tries_ += 1;
}

void State::ApplyClueToWords_(const Clue &clue,
                              HostDeviceBuffer<char> &words) noexcept {
  auto it = words.h_buffer.begin();
  while (it != words.h_buffer.end()) {
    std::string word(it, it + k_letters_per_word);
    if (clue.Disallows(word.data())) {
      it = words.h_buffer.erase(it, it + k_letters_per_word);
    } else {
      it += k_letters_per_word;
    }
  }
  words.SyncHostToDevice();
}

uint32_t State::Round_() const noexcept { return tries_; }

void State::BestGuess_() {
  PrintSolutions_();
  ResetCounts_();

  if (NumberOfSolutions_() > 1) {
    std::cout << '\n' << "  Computing eliminations per guess..." << '\n';
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NumberOfGuesses_(); ++i) {
      const uint32_t required_shared_memory =
        (solutions_.h_buffer.size() * sizeof(char));
      CHECK_HIP_KERNEL(hipLaunchKernelGGL(
        CountDisallowedWords_, dim3(k_number_of_blocks),
        dim3(k_threads_per_block), required_shared_memory,
        streams_[i % k_number_of_streams], i, guesses_.d_buffer,
        solutions_.d_buffer, solutions_.h_buffer.size(), counts_.d_buffer));
      CHECK_HIP_API(hipEventRecord(kernel_completion_events_[i],
                                   streams_[i % k_number_of_streams]));
    }
    PrintKernelCompletionProgress_(start);
    auto end = std::chrono::high_resolution_clock::now();
    CHECK_HIP_API(hipDeviceSynchronize());

    ComputeAndPrintMetrics_(end - start);
  }
}

void State::PrintKernelCompletionProgress_(
  const std::chrono::time_point<std::chrono::high_resolution_clock> &start)
  const {
  bool all_kernels_complete = false;
  while (!all_kernels_complete) {
    size_t num_completed_kernels = 0;
    for (size_t i = 0; i < NumberOfGuesses_(); ++i) {
      num_completed_kernels += static_cast<size_t>(
        hipEventQuery(kernel_completion_events_[i]) == hipSuccess);
    }
    const float progress = static_cast<float>(num_completed_kernels) /
                           static_cast<float>(NumberOfGuesses_());
    utilities::UpdateProgressBar(progress, start);
    all_kernels_complete = (num_completed_kernels == NumberOfGuesses_());
  }
  std::cout << '\n';
}

void State::ComputeAndPrintMetrics_(
  const std::chrono::duration<double> &duration) {
  SortAndPrintEliminationsPerGuess_();
  PrintWordStats_();
  PrintTimeToBestGuess_(duration);
}

void State::SortAndPrintEliminationsPerGuess_() {
  counts_.SyncDeviceToHost();

  solution_index_map_.clear();
  for (uint32_t i = 0; i < NumberOfSolutions_(); ++i) {
    const std::string solution(&solutions_.h_buffer[k_letters_per_word * i],
                               k_letters_per_word);
    solution_index_map_[solution] = i;
  }
  for (uint32_t i = 0; i < NumberOfGuesses_(); ++i) {
    const std::string guess(&guesses_.h_buffer[k_letters_per_word * i],
                            k_letters_per_word);
    if (solution_index_map_.find(guess) == solution_index_map_.end())
      solution_index_map_[guess] = -1;
  }

  // Sort guesses by elimination count; in a tie, prefer a guess from solutions
  if (NumberOfSolutions_() != 1) {
    guesses_and_counts_.clear();
    for (uint16_t i = 0; i < counts_.h_buffer.size(); ++i) {
      guesses_and_counts_.emplace_back(
        std::string(&guesses_.h_buffer[i * State::k_letters_per_word],
                    State::k_letters_per_word),
        counts_.h_buffer[i]);
    }
    std::sort(guesses_and_counts_.begin(), guesses_and_counts_.end(),
              [this](const auto &a, const auto &b) {
                if (a.second == b.second) {
                  auto a_index = solution_index_map_[a.first];
                  auto b_index = solution_index_map_[b.first];
                  bool found_a = (a_index != -1);
                  bool found_b = (b_index != -1);

                  if ((found_a && found_b) || (!found_a && !found_b)) {
                    return a_index < b_index;
                  }
                  return !found_a && found_b;
                }
                return a.second < b.second;
              });

    uint32_t i = 0;
    std::cout << '\n' << "  Detailed eliminations per allowed guess: " << '\n';
    for (const auto &pair : guesses_and_counts_) {
      std::cout << "    " << pair.first << " : " << std::fixed
                << std::setprecision(2)
                << static_cast<float>(pair.second) / NumberOfSolutions_();
      const bool is_line_end = (i % k_guess_count_pairs_per_line_ ==
                                (k_guess_count_pairs_per_line_ - 1));
      const bool is_last_guess = (i == (NumberOfGuesses_() - 1));
      if (is_line_end || is_last_guess) {
        std::cout << '\n';
      }
      ++i;
    }
  }
}

void State::PrintWordStats_() {
  auto [min_indices, max_indices] = ProcessCounts_();

  // Worst guesses
  if (hard_mode_) {
    // Skip this output on easy mode because many words don't eliminate
    std::cout << '\n'
              << "  Worst guess" << (min_indices.size() > 1 ? "es" : "") << ": "
              << '\n';
    for (const auto &i : min_indices) {
      std::cout << "    ";
      std::cout << guesses_and_counts_[i].first << " ";
      std::cout << "(eliminates avg. of "
                << static_cast<float>(guesses_and_counts_[i].second) /
                     NumberOfSolutions_()
                << " / " << NumberOfSolutions_() << " words) " << '\n';
    }
  }

  // Best guesses
  std::cout << '\n'
            << "  Best guess" << (max_indices.size() > 1 ? "es" : "") << ": "
            << '\n';
  for (const auto &i : max_indices) {
    std::cout << "    ";
    std::cout << guesses_and_counts_[i].first << " ";
    std::cout << "(eliminates avg. of "
              << static_cast<float>(guesses_and_counts_[i].second) /
                   NumberOfSolutions_()
              << " / " << NumberOfSolutions_() << " words"
              << (solution_index_map_[guesses_and_counts_[i].first] == -1
                    ? ") "
                    : "; appears in remaining solutions) ")
              << '\n';
  }
  std::cout << '\n';
}

std::pair<std::vector<uint16_t>, std::vector<uint16_t>>
State::ProcessCounts_() const {
  // Process counts to get word(s) with max eliminations
  const uint32_t min_value = guesses_and_counts_.begin()->second;
  const uint32_t max_value = guesses_and_counts_.rbegin()->second;
  std::vector<uint16_t> min_indices;
  std::vector<uint16_t> max_indices;
  auto it = guesses_and_counts_.begin();
  while (it != guesses_and_counts_.end() && it->second == min_value) {
    min_indices.push_back(std::distance(guesses_and_counts_.begin(), it++));
  }
  auto rit = guesses_and_counts_.rbegin();
  while (rit != guesses_and_counts_.rend() && rit->second == max_value) {
    max_indices.push_back(
      std::distance(guesses_and_counts_.begin(), (rit++).base()) - 1);
  }

  return {min_indices, max_indices};
}

void State::PrintSolutions_() const {
  std::cout << '\n'
            << "  " << NumberOfSolutions_() << " / " << k_number_of_solutions
            << " possible solutions remaining: " << '\n';
  for (uint16_t i = 0; i < NumberOfSolutions_(); ++i) {
    const std::string solution(&solutions_.h_buffer[k_letters_per_word * i],
                               k_letters_per_word);
    std::cout << "    " << solution;
    const bool is_line_end =
      (i % k_solutions_per_line_ == (k_solutions_per_line_ - 1));
    const bool is_last_solution = (i == (NumberOfSolutions_() - 1));
    if (is_line_end || is_last_solution) {
      std::cout << '\n';
    }
  }
}

void State::PrintTimeToBestGuess_(
  const std::chrono::duration<double> &duration) {
  std::cout << std::fixed << std::setprecision(6)
            << "  Time to solution: " << duration.count() << " seconds" << '\n';
}
