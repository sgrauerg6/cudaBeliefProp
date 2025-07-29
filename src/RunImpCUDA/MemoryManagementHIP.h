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
 * @file MemoryManagementHIP.h
 * @author Scott Grauer-Gray
 * @brief Declares and defines child class of MemoryManagement with overriden
 * member functions to manage memory on HIP device including transferring data
 * between host and HIP device.
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef MEMORY_MANAGEMENT_HIP_H_
#define MEMORY_MANAGEMENT_HIP_H_

#include "RunEval/RunTypeConstraints.h"
#include "RunImp/MemoryManagement.h"

/**
 * @brief Child class of MemoryManagement with overriden member functions to
 * manage memory on HIP device including transferring data between host and
 * HIP device.
 * 
 * @tparam T 
 */
template <RunData_t T>
class MemoryManagementHIP final : public MemoryManagement<T>
{
public:
  /**
   * @brief Allocate specified amount of data of type T on HIP device
   * 
   * @param numData 
   * @return Pointer to allocated memory on HIP device
   */
  T* AllocateMemoryOnDevice(std::size_t numData) const override
  {
    T* array_to_allocate;
    hipMalloc((void **) &array_to_allocate, numData * sizeof(T));
    return array_to_allocate;
  }

  /**
   * @brief Free memory on HIP device
   * 
   * @param array_to_free 
   */
  void FreeMemoryOnDevice(T* array_to_free) const override
  {
    hipFree(array_to_free);
  }

  /**
   * @brief Allocate aligned memory on HIP device (same as default allocating
   * of memory for HIP)
   * 
   * @param numData 
   * @param acc_setting 
   * @return Pointer to allocated memory on HIP device
   */
  T* AllocateAlignedMemoryOnDevice(
    std::size_t numData,
    run_environment::AccSetting acc_setting) const override
  {
    return AllocateMemoryOnDevice(numData);
  }

  /**
   * @brief Free aligned memory on HIP device (same as default free memory for
   * HIP)
   * 
   * @param memory_to_free 
   */
  void FreeAlignedMemoryOnDevice(T* memory_to_free) const override
  {
    FreeMemoryOnDevice(memory_to_free);
  }

  void TransferDataFromDeviceToHost(
    T* dest_array,
    const T* in_array,
    std::size_t num_data_transfer) const override
  {
    hipMemcpy(
      dest_array, in_array, num_data_transfer * sizeof(T), hipMemcpyDeviceToHost);
  }

  void TransferDataFromHostToDevice(
    T* dest_array,
    const T* in_array,
    std::size_t num_data_transfer) const override
  {
    hipMemcpy(
      dest_array, in_array, num_data_transfer * sizeof(T), hipMemcpyHostToDevice);
  }
};

#endif //MEMORY_MANAGEMENT_HIP_H_
