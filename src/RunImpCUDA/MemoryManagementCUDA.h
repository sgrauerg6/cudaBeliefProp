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
 * @file MemoryManagementCUDA.h
 * @author Scott Grauer-Gray
 * @brief Declares and defines child class of MemoryManagement with overriden
 * member functions to manage memory on CUDA device including transferring data
 * between host and CUDA device.
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef MEMORY_MANAGEMENT_CUDA_H_
#define MEMORY_MANAGEMENT_CUDA_H_

#include <cuda_runtime.h>
#include "RunEval/RunTypeConstraints.h"
#include "RunImp/MemoryManagement.h"

/**
 * @brief Child class of MemoryManagement with overriden member functions to
 * manage memory on CUDA device including transferring data between host and
 * CUDA device.
 * 
 * @tparam T 
 */
template <RunData_t T>
class MemoryManagementCUDA final : public MemoryManagement<T>
{
public:
  /**
   * @brief Allocate specified amount of data of type T on CUDA device
   * 
   * @param numData 
   * @return Pointer to allocated memory on CUDA device
   */
  T* AllocateMemoryOnDevice(std::size_t numData) const override
  {
    T* array_to_allocate;
    cudaMalloc((void **) &array_to_allocate, numData * sizeof(T));
    return array_to_allocate;
  }

  /**
   * @brief Free memory on CUDA device
   * 
   * @param array_to_free 
   */
  void FreeMemoryOnDevice(T* array_to_free) const override
  {
    cudaFree(array_to_free);
  }

  /**
   * @brief Allocate aligned memory on CUDA device (same as default allocating
   * of memory for CUDA)
   * 
   * @param numData 
   * @param acc_setting 
   * @return Pointer to allocated memory on CUDA device
   */
  T* AllocateAlignedMemoryOnDevice(
    std::size_t numData,
    run_environment::AccSetting acc_setting) const override
  {
    return AllocateMemoryOnDevice(numData);
  }

  /**
   * @brief Free aligned memory on CUDA device (same as default free memory for
   * CUDA)
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
    cudaMemcpy(
      dest_array, in_array, num_data_transfer * sizeof(T), cudaMemcpyDeviceToHost);
  }

  void TransferDataFromHostToDevice(
    T* dest_array,
    const T* in_array,
    std::size_t num_data_transfer) const override
  {
    cudaMemcpy(
      dest_array, in_array, num_data_transfer * sizeof(T), cudaMemcpyHostToDevice);
  }
};

#endif //MEMORY_MANAGEMENT_CUDA_H_
