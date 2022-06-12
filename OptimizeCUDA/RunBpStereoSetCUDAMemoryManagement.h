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
		//std::cout << "ALLOC_GPU\n";
		//allocate the space for the disparity map estimation
		float* arrayToAllocate;
		cudaMalloc((void **) &arrayToAllocate, numData * sizeof(float));
		return arrayToAllocate;
	}

	void freeDataOnCompDevice(U arrayToFree) {
		//std::cout << "FREE_GPU\n";
		cudaFree(arrayToFree);
	}

	void transferDataFromCompDeviceToHost(float* destArray, const float* inArray, const unsigned int numDataTransfer) {
		//std::cout << "TRANSFER_GPU\n";
		cudaMemcpy(destArray, inArray, numDataTransfer * sizeof(float), cudaMemcpyDeviceToHost);
	}

	void transferDataFromCompHostToDevice(float* destArray, const float* inArray, const unsigned int numDataTransfer) {
		//std::cout << "TRANSFER_GPU\n";
		cudaMemcpy(destArray, inArray, numDataTransfer * sizeof(float), cudaMemcpyHostToDevice);
	}
};

#endif //RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_
