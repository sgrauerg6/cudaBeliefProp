#ifndef MEMORY_MANAGEMENT_CUDA_H_
#define MEMORY_MANAGEMENT_CUDA_H_

#include <cuda_runtime.h>
#include "RunEval/RunTypeConstraints.h"
#include "RunImp/MemoryManagement.h"

/**
 * @brief Functions to manage memory on CUDA device including transferring data between host and CUDA device.
 * Functions defined in class override functions in parent class MemoryManagement
 * which is used for CPU only processing.
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
   * @return T* 
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
   * @brief Allocate aligned memory on CUDA device (same as default allocating of memory for CUDA)
   * 
   * @param numData 
   * @param acc_setting 
   * @return T* 
   */
  T* AllocateAlignedMemoryOnDevice(
    std::size_t numData,
    run_environment::AccSetting acc_setting) const override
  {
    return AllocateMemoryOnDevice(numData);
  }

  /**
   * @brief Free aligned memory on CUDA device (same as default free memory for CUDA)
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
