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
  T* allocateMemoryOnDevice(unsigned int numData) override {
    //allocate the space for the disparity map estimation
    T* arrayToAllocate;
    cudaMalloc((void **) &arrayToAllocate, numData * sizeof(T));
    return arrayToAllocate;
  }

  void freeMemoryOnDevice(T* arrayToFree) override {
    cudaFree(arrayToFree);
  }

  T* allocateAlignedMemoryOnDevice(const unsigned long numData, run_environment::AccSetting accSetting) override
  {
    T* arrayToAllocate;
    cudaMalloc((void**)&arrayToAllocate, numData*sizeof(T));
    return arrayToAllocate;
  }

  void freeAlignedMemoryOnDevice(T* memoryToFree) override {
    cudaFree(memoryToFree);
  }

  void transferDataFromDeviceToHost(T* destArray, const T* inArray, unsigned int numDataTransfer) override {
    cudaMemcpy(destArray, inArray, numDataTransfer * sizeof(T), cudaMemcpyDeviceToHost);
  }

  void transferDataFromHostToDevice(T* destArray, const T* inArray, unsigned int numDataTransfer) override {
    cudaMemcpy(destArray, inArray, numDataTransfer * sizeof(T), cudaMemcpyHostToDevice);
  }
};

#endif //RUN_IMP_CUDA_MEMORY_MANAGEMENT_H_
