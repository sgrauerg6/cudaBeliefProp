/*
Copyright (C) 2026 Scott Grauer-Gray

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
 * @file MemoryManagementCPU.h
 * @author Scott Grauer-Gray
 * @brief Declares child class for memory management with functions defined for
 * standard memory allocation using CPU
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef MEMORY_MANAGEMENT_CPU_H_
#define MEMORY_MANAGEMENT_CPU_H_

#include <new>
#include <ranges>
#include "RunSettingsParams/RunSettings.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunImp/MemoryManagement.h"

/**
 * @brief Class for memory management with functions defined for standard
 * memory allocation using CPU.<br>
 * Class functions can be overridden to support other computation devices such
 * as GPU.
 * 
 * @tparam T
 * @tparam U
 */
template <RunData_t T, typename U = T>
class MemoryManagementCPU final : public MemoryManagement<T, U>
{
public:

  U* AllocateMemoryOnDevice(std::size_t numData) const override {
    return (new T[numData]);
  }

  void FreeMemoryOnDevice(U* array_to_free) const override {
    delete [] array_to_free;
  }

  U* AllocateAlignedMemoryOnDevice(
    std::size_t numData,
    run_environment::AccSetting acc_setting) const override
  {
#ifdef _WIN32
    U* memoryData = static_cast<U*>(_aligned_malloc(
      numData * sizeof(T), run_environment::GetBytesAlignMemory(acc_setting)));
    return memoryData;
#else
    U* memoryData = static_cast<U*>(std::aligned_alloc(
      run_environment::GetBytesAlignMemory(acc_setting), numData * sizeof(T)));
    return memoryData;
#endif
  }

  void FreeAlignedMemoryOnDevice(U* memory_to_free) const override
  {
#ifdef _WIN32
    _aligned_free(memory_to_free);
#else
    free(memory_to_free);
#endif
  }

  void TransferDataFromDeviceToHost(
    T* dest_array,
    U* in_array,
    std::size_t num_data_transfer) const override
  {
    std::ranges::copy(in_array, in_array + num_data_transfer, dest_array);
  }

  virtual void TransferDataFromHostToDevice(
    U* dest_array,
    T* in_array,
    std::size_t num_data_transfer) const override
  {
    std::ranges::copy(in_array, in_array + num_data_transfer, dest_array);
  }
};

#endif //MEMORY_MANAGEMENT_CPU_H_
