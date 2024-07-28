/* Copyright (c) 2024 Michael E. Rowan.
 *
 * This file is part of Wordle-GPU.
 *
 * License: MIT.
 */

#ifndef WORDLE_GPU_SRC_UTILITIES_H
#define WORDLE_GPU_SRC_UTILITIES_H

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#define CHECK_HIP_API(call)                                                    \
  utilities::CheckHipApi((call), #call, __FILE__, __LINE__)
#define CHECK_HIP_KERNEL(call)                                                 \
  call;                                                                        \
  do {                                                                         \
    hipError_t err = hipGetLastError();                                        \
    utilities::CheckHipApi(err, #call, __FILE__, __LINE__);                    \
  } while (0)

namespace utilities {
inline void CheckHipApi(hipError_t err, const char *function, const char *file,
                        const uint32_t line) {
  if (err != hipSuccess) {
    std::cerr << "HIP error: " << hipGetErrorString(err) << " calling "
              << function << " at " << file << ": " << line << '\n';
    exit(EXIT_FAILURE);
  }
}

template <typename... Args>
std::string StringFormat(const std::string &format, Args... args) {
  const int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  const uint32_t size = static_cast<uint32_t>(size_s);
  const std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);
}

inline void UpdateProgressBar(
  const float progress,
  const std::chrono::time_point<std::chrono::high_resolution_clock> &start,
  const uint32_t bar_width = 20) {
  const uint32_t pos = static_cast<uint32_t>(bar_width * progress);
  const std::chrono::duration<double> duration =
    std::chrono::high_resolution_clock::now() - start;
  std::cout << std::fixed << std::setprecision(6)
            << "  Kernel completion progress (" << duration.count()
            << " seconds elapsed): |";
  for (uint32_t i = 0; i < bar_width; ++i) {
    if (i < pos) {
      std::cout << "|";
    } else {
      std::cout << " ";
    }
  }
  std::cout << "| " << static_cast<uint32_t>(progress * 100.0) << "%\r";
  std::cout.flush();
}

constexpr uint32_t LogBaseTwo(const uint32_t n) {
  return (n <= 1) ? 0 : 1 + LogBaseTwo(n / 2);
}
} // namespace utilities

#endif // WORDLE_GPU_SRC_UTILITIES_H
