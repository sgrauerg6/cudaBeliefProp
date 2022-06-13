#ifndef RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_
#define RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_

#include "../BpAndSmoothProcessing/RunBpStereoSetMemoryManagement.h"
#include <cuda_runtime.h>

//only processing that uses RunBpStereoSetCUDAMemoryManagement is the input stereo
//images and output disparity map that always uses float data type
template <typename T=float>
class RunBpStereoSetCUDAMemoryManagement : public RunBpStereoSetMemoryManagement<T>
{
public:
	T* allocateMemoryOnDevice(const unsigned int numData) override {
		//allocate the space for the disparity map estimation
		T* arrayToAllocate;
		cudaMalloc((void **) &arrayToAllocate, numData * sizeof(T));
		return arrayToAllocate;
	}

	void freeMemoryOnDevice(T* arrayToFree) override {
		cudaFree(arrayToFree);
	}

	T* allocateAlignedMemoryOnDevice(const unsigned long numData) override
	{
		T* arrayToAllocate;
		cudaMalloc((void**)&arrayToAllocate, numData*sizeof(T));
		return arrayToAllocate;
	}

	void freeAlignedMemoryOnDevice(T* memoryToFree) override {
		cudaFree(memoryToFree);
	}

	void transferDataFromDeviceToHost(T* destArray, const T* inArray, const unsigned int numDataTransfer) override {
		cudaMemcpy(destArray, inArray, numDataTransfer * sizeof(T), cudaMemcpyDeviceToHost);
	}

	void transferDataFromHostToDevice(T* destArray, const T* inArray, const unsigned int numDataTransfer) override {
		cudaMemcpy(destArray, inArray, numDataTransfer * sizeof(T), cudaMemcpyHostToDevice);
	}
};

#endif //RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_
