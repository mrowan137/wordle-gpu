/* Copyright (c) 2024 Michael E. Rowan
 *
 * This file is part of Wordle-GPU.
 *
 * License: MIT
 */

#ifndef WORDLE_GPU_SRC_STATE_H
#define WORDLE_GPU_SRC_STATE_H

#include <hip/hip_runtime.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "clue.h"
#include "constants.h"
#include "host_device_buffer.h"

__global__ void CountDisallowedWords_(uint16_t guess_idx,
                                      const char *__restrict__ guesses,
                                      const char *__restrict__ solutions,
                                      uint16_t N, uint32_t *__restrict__ counts);

class State {
 public:
  State(const int argc, const char *argv[]);
  ~State();
  void QueryClueAndComputeBestGuess();
  bool GameOver() const;
  static constexpr uint32_t k_number_of_guesses = NUMBER_OF_GUESSES;
  static constexpr uint32_t k_number_of_solutions = NUMBER_OF_SOLUTIONS;
  static constexpr uint32_t k_letters_per_word = LETTERS_PER_WORD;
  static constexpr uint32_t k_threads_per_block = 1024;
  static constexpr uint32_t k_number_of_blocks = 512;
#ifndef DETECTED_WAVEFRONT_SIZE
#error                                                                         \
  "DETECTED_WAVEFRONT_SIZE is not defined. Check CMakeLists.txt to ensure HIP_OFFLOAD_ARCH is set."
#endif
  static constexpr uint32_t k_wavefront_size = DETECTED_WAVEFRONT_SIZE;
  static constexpr uint32_t k_number_of_streams = 256;

 private:
  void Initialize_(int argc, const char *argv[]);
  static void CheckCommandLine_(int argc, const char *argv[]);
  void ParseCommandLine_(int argc, const char *argv[]);
  static void LoadWords_(const std::string &path_to_words,
                         uint32_t number_of_words,
                         HostDeviceBuffer<char> &words);
  void InitializeCounts_();
  void ResetCounts_();
  template <typename T, typename InitFuncT>
  void InitializeHipResource_(std::vector<T> &resource, size_t size,
                              const InitFuncT &InitFunc);
  template <typename T, typename CleanupFuncT>
  void CleanupHipResource_(std::vector<T> &resource,
                           const CleanupFuncT &CleanupFunc);
  uint16_t NumberOfGuesses_() const noexcept;
  uint16_t NumberOfSolutions_() const noexcept;
  bool ClueIsValid_(const std::string &clue) const;
  static void PrintClueHelp_();
  void ApplyClue_(const std::string &input) noexcept;
  static void ApplyClueToWords_(const Clue &clue,
                                HostDeviceBuffer<char> &words) noexcept;
  uint32_t Round_() const noexcept;
  void BestGuess_();
  void PrintKernelCompletionProgress_(
    const std::chrono::time_point<std::chrono::high_resolution_clock> &start)
    const;
  void ComputeAndPrintMetrics_(const std::chrono::duration<double> &duration);
  void SortAndPrintEliminationsPerGuess_();
  void PrintWordStats_();
  void PrintSolutions_() const;
  static void
  PrintTimeToBestGuess_(const std::chrono::duration<double> &duration);
  std::pair<std::vector<uint16_t>, std::vector<uint16_t>>
  ProcessCounts_() const;
  HostDeviceBuffer<char> guesses_;
  HostDeviceBuffer<char> solutions_;
  HostDeviceBuffer<uint32_t> counts_;
  std::vector<std::string> clues_;
  std::unordered_map<std::string, int32_t> solution_index_map_;
  std::vector<std::pair<std::string, uint32_t>> guesses_and_counts_;
  uint16_t tries_{0};
  bool quit_{false};
  std::string path_to_guesses_;
  std::string path_to_solutions_;
  bool hard_mode_{};
  std::vector<hipStream_t> streams_;
  std::vector<hipEvent_t> kernel_completion_events_;
  static const uint16_t k_guess_count_pairs_per_line_ = 4;
  static const uint16_t k_solutions_per_line_ = 8;
};

template <typename T, typename InitFuncT>
void State::InitializeHipResource_(std::vector<T> &resource, size_t size,
                                   const InitFuncT &InitFunc) {
  resource.reserve(size);
  resource.resize(size);
  for (uint32_t i = 0; i < size; ++i) {
    CHECK_HIP_API(InitFunc(&resource[i]));
  }
}

template <typename T, typename CleanupFuncT>
void State::CleanupHipResource_(std::vector<T> &resource,
                                const CleanupFuncT &CleanupFunc) {
  for (uint32_t i = 0; i < resource.size(); ++i) {
    CHECK_HIP_API(CleanupFunc(resource[i]));
  }
}

#endif // WORDLE_GPU_SRC_STATE_H
