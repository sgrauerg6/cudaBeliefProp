#ifndef RUN_IMP_CUDA_MEMORY_MANAGEMENT_H_
#define RUN_IMP_CUDA_MEMORY_MANAGEMENT_H_

#include <cuda_runtime.h>
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunImp/RunImpMemoryManagement.h"

//functions to manage memory on CUDA device including transferring data between host and CUDA device
template <RunData_t T>
class RunImpCUDAMemoryManagement final : public RunImpMemoryManagement<T>
{
public:
  T* AllocateMemoryOnDevice(unsigned int numData) const override {
    //allocate the space for the disparity map estimation
    T* array_to_allocate;
    cudaMalloc((void **) &array_to_allocate, numData * sizeof(T));
    return array_to_allocate;
  }

  void FreeMemoryOnDevice(T* array_to_free) const override {
    cudaFree(array_to_free);
  }

  T* AllocateAlignedMemoryOnDevice(const unsigned long numData, run_environment::AccSetting acc_setting) const override
  {
    T* array_to_allocate;
    cudaMalloc((void**)&array_to_allocate, numData*sizeof(T));
    return array_to_allocate;
  }

  void FreeAlignedMemoryOnDevice(T* memory_to_free) const override {
    cudaFree(memory_to_free);
  }

  void TransferDataFromDeviceToHost(T* dest_array, const T* in_array, unsigned int num_data_transfer) const override {
    cudaMemcpy(dest_array, in_array, num_data_transfer * sizeof(T), cudaMemcpyDeviceToHost);
  }

  void TransferDataFromHostToDevice(T* dest_array, const T* in_array, unsigned int num_data_transfer) const override {
    cudaMemcpy(dest_array, in_array, num_data_transfer * sizeof(T), cudaMemcpyHostToDevice);
  }
};

#endif //RUN_IMP_CUDA_MEMORY_MANAGEMENT_H_
