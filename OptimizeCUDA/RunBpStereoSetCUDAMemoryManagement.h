#ifndef RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_
#define RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_

#include "../BpAndSmoothProcessing/RunBpStereoSetMemoryManagement.h"
#include <cuda_runtime.h>

//only processing that uses RunBpStereoSetCUDAMemoryManagement is the input stereo
//images and output disparity map that always uses float data type
class RunBpStereoSetCUDAMemoryManagement : public RunBpStereoSetMemoryManagement
{
public:
	float* allocateDataOnCompDevice(const unsigned int numData) {
		//allocate the space for the disparity map estimation
		float* arrayToAllocate;
		cudaMalloc((void **) &arrayToAllocate, numData * sizeof(float));
		return arrayToAllocate;
	}

	void freeDataOnCompDevice(float* arrayToFree) {
		cudaFree(arrayToFree);
	}

	void transferDataFromCompDeviceToHost(float* destArray, const float* inArray, const unsigned int numDataTransfer) {
		cudaMemcpy(destArray, inArray, numDataTransfer * sizeof(float), cudaMemcpyDeviceToHost);
	}

	void transferDataFromCompHostToDevice(float* destArray, const float* inArray, const unsigned int numDataTransfer) {
		cudaMemcpy(destArray, inArray, numDataTransfer * sizeof(float), cudaMemcpyHostToDevice);
	}
};

#endif //RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_
