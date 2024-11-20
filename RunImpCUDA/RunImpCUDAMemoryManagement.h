#ifndef RUN_IMP_CUDA_MEMORY_MANAGEMENT_H_
#define RUN_IMP_CUDA_MEMORY_MANAGEMENT_H_

#include <cuda_runtime.h>
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunImp/RunImpMemoryManagement.h"

//functions to manage memory on CUDA device including transferring data between host and CUDA device
//functions defined in class override functions in parent class RunImpMemoryManagement which is used for CPU only processing
template <RunData_t T>
class RunImpCUDAMemoryManagement final : public RunImpMemoryManagement<T>
{
public:
  //allocate specified amount of data of type T on CUDA device
  T* AllocateMemoryOnDevice(std::size_t numData) const override {
    T* array_to_allocate;
    cudaMalloc((void **) &array_to_allocate, numData * sizeof(T));
    return array_to_allocate;
  }

  //free memory on CUDA device
  void FreeMemoryOnDevice(T* array_to_free) const override {
    cudaFree(array_to_free);
  }

  //allocate aligned memory on CUDA device (same as default allocating of memory for CUDA)
  T* AllocateAlignedMemoryOnDevice(std::size_t numData, run_environment::AccSetting acc_setting) const override
  {
    return AllocateMemoryOnDevice(numData);
  }

  //free aligned memory on CUDA device (same as default free memory for CUDA)
  void FreeAlignedMemoryOnDevice(T* memory_to_free) const override {
    FreeMemoryOnDevice(memory_to_free);
  }

  void TransferDataFromDeviceToHost(T* dest_array, const T* in_array, std::size_t num_data_transfer) const override {
    cudaMemcpy(dest_array, in_array, num_data_transfer * sizeof(T), cudaMemcpyDeviceToHost);
  }

  void TransferDataFromHostToDevice(T* dest_array, const T* in_array, std::size_t num_data_transfer) const override {
    cudaMemcpy(dest_array, in_array, num_data_transfer * sizeof(T), cudaMemcpyHostToDevice);
  }
};

#endif //RUN_IMP_CUDA_MEMORY_MANAGEMENT_H_
