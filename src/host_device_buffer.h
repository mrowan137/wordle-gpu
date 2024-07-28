/* Copyright (c) 2024 Michael E. Rowan.
 *
 * This file is part of Wordle-GPU.
 *
 * License: MIT.
 */

#ifndef WORDLE_GPU_SRC_HOST_DEVICE_BUFFER_H
#define WORDLE_GPU_SRC_HOST_DEVICE_BUFFER_H

#include <hip/hip_runtime.h>

#include <vector>

#include "utilities.h"

template <typename T> struct HostDeviceBuffer {
  HostDeviceBuffer();
  ~HostDeviceBuffer();
  void SyncHostToDevice();
  void SyncDeviceToHost();
  std::vector<T> h_buffer;
  T *d_buffer;
};

template <typename T>
HostDeviceBuffer<T>::HostDeviceBuffer() : h_buffer({}), d_buffer(nullptr) {}

template <typename T> HostDeviceBuffer<T>::~HostDeviceBuffer() {
  if (d_buffer) {
    CHECK_HIP_API(hipFree(d_buffer));
    d_buffer = nullptr;
  }
}

template <typename T> void HostDeviceBuffer<T>::SyncHostToDevice() {
  if (d_buffer) {
    CHECK_HIP_API(hipFree(d_buffer));
    d_buffer = nullptr;
  }
  const uint32_t allocation_size = h_buffer.size() * sizeof(T);
  CHECK_HIP_API(hipMalloc((void **)&d_buffer, allocation_size));
  CHECK_HIP_API(hipMemcpy(d_buffer, h_buffer.data(), allocation_size,
                          hipMemcpyHostToDevice));
}

template <typename T> void HostDeviceBuffer<T>::SyncDeviceToHost() {
  const uint32_t allocation_size = h_buffer.size() * sizeof(T);
  CHECK_HIP_API(hipMemcpy(h_buffer.data(), d_buffer, allocation_size,
                          hipMemcpyDeviceToHost));
}

#endif // WORDLE_GPU_SRC_HOST_DEVICE_BUFFER_H
