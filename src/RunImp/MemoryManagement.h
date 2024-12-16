/*
Copyright (C) 2024 Scott Grauer-Gray

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/**
 * @file MemoryManagement.h
 * @author Scott Grauer-Gray
 * @brief Declares class for memory management with functions defined for
 * standard memory allocation using CPU and can be overridden to support
 * other computation devices
 * 
 * @copyright Copyright (c) 2024
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
