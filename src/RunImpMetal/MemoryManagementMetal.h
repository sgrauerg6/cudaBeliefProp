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
 * @file MemoryManagementMetal.h
 * @author Scott Grauer-Gray
 * @brief Declares and defines child class of MemoryManagement with overriden
 * member functions to manage memory on metal device including transferring data
 * between host and metal device.
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef MEMORY_MANAGEMENT_METAL_H_
#define MEMORY_MANAGEMENT_METAL_H_

#include "RunEval/RunTypeConstraints.h"
#include "RunImp/MemoryManagement.h"

/**
 * @brief Child class of MemoryManagement with overriden member functions to
 * manage memory on Metal device including transferring data between host and
 * Metal device.
 * 
 * @tparam T
 * @tparam U
 */
template <RunData_t T, typename U>
class MemoryManagementMetal final : public MemoryManagement<T, U>
{
public:
  /**
   * @brief Allocate specified amount of data of type T on Metal device
   * 
   * @param numData 
   * @return Pointer to allocated memory on Metal device
   */
  U* AllocateMemoryOnDevice(std::size_t numData) const override
  {
    U* array_to_allocate;
    array_to_allocate =
      mDevice->newBuffer(numData * sizeof(T), MTL::ResourceStorageModeShared);
    return array_to_allocate;
  }

  /**
   * @brief Free memory on Metal device
   * 
   * @param array_to_free 
   */
  void FreeMemoryOnDevice(U* array_to_free) const override
  {
    array_to_free.release();
  }

  /**
   * @brief Allocate aligned memory on Metal device (same as default allocating
   * of memory for Metal)
   * 
   * @param numData 
   * @param acc_setting 
   * @return Pointer to allocated memory on Metal device
   */
  U* AllocateAlignedMemoryOnDevice(
    std::size_t numData,
    run_environment::AccSetting acc_setting) const override
  {
    return AllocateMemoryOnDevice(numData);
  }

  /**
   * @brief Free aligned memory on Metal device (same as default free memory for
   * Metal)
   * 
   * @param memory_to_free 
   */
  void FreeAlignedMemoryOnDevice(U* memory_to_free) const override
  {
    FreeMemoryOnDevice(memory_to_free);
  }

  void TransferDataFromDeviceToHost(
    T* dest_array,
    const U* in_array,
    std::size_t num_data_transfer) const override
  {
    void* buffer_data = in_array->contents();
    memcpy(dest_array, buffer_data, num_data_transfer * sizeof(T));
  }

  void TransferDataFromHostToDevice(
    U* dest_array,
    const T* in_array,
    std::size_t num_data_transfer) const override
  {
    void* buffer_data = in_array->contents();
    memcpy(buffer_data, in_array, num_data_transfer * sizeof(T));
  }
};

#endif //MEMORY_MANAGEMENT_METAL_H_
