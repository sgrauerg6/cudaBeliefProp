/*
 * MemoryManagement.h
 *
 * Class for memory management with functions defined
 * for standard memory allocation using CPU and can be
 * overridden to support other computation devices
 */

#ifndef MEMORY_MANAGEMENT_H_
#define MEMORY_MANAGEMENT_H_

#include <new>
#include <ranges>
#include "RunSettingsParams/RunSettings.h"
#include "RunEval/RunTypeConstraints.h"

/**
 * @brief Class for memory management with functions defined for standard memory allocation using CPU.
 * Class functions can be overridden to support other computation devices such as GPU.
 * 
 * @tparam T 
 */
template <RunData_t T>
class MemoryManagement
{
public:
  virtual T* AllocateMemoryOnDevice(std::size_t numData) const {
    return (new T[numData]);
  }

  virtual void FreeMemoryOnDevice(T* array_to_free) const {
    delete [] array_to_free;
  }

  virtual T* AllocateAlignedMemoryOnDevice(
    std::size_t numData,
    run_environment::AccSetting acc_setting) const
  {
#ifdef _WIN32
    T* memoryData = static_cast<T*>(_aligned_malloc(
      numData * sizeof(T), run_environment::GetNumDataAlignWidth(acc_setting) * sizeof(T)));
    return memoryData;
#else
    T* memoryData = static_cast<T*>(std::aligned_alloc(
      run_environment::GetNumDataAlignWidth(acc_setting) * sizeof(T), numData * sizeof(T)));
    return memoryData;
#endif
  }

  virtual void FreeAlignedMemoryOnDevice(T* memory_to_free) const
  {
#ifdef _WIN32
    _aligned_free(memory_to_free);
#else
    free(memory_to_free);
#endif
  }

  virtual void TransferDataFromDeviceToHost(
    T* dest_array,
    const T* in_array,
    std::size_t num_data_transfer) const
  {
    std::ranges::copy(in_array, in_array + num_data_transfer, dest_array);
  }

  virtual void TransferDataFromHostToDevice(
    T* dest_array,
    const T* in_array,
    std::size_t num_data_transfer) const
  {
    std::ranges::copy(in_array, in_array + num_data_transfer, dest_array);
  }
};

#endif //MEMORY_MANAGEMENT_H_
