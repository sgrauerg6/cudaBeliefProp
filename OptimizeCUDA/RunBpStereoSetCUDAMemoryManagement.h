#ifndef RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_
#define RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_

#include "../BpAndSmoothProcessing/RunBpStereoSetMemoryManagement.h"
#include <cuda_runtime.h>

class RunBpStereoSetCUDAMemoryManagement : public RunBpStereoSetMemoryManagement
{
public:

	void allocateDataOnCompDevice(void** arrayToAllocate, int numBytes)
	{
		//std::cout << "ALLOC_GPU\n";
		//allocate the space for the disparity map estimation
		cudaMalloc((void **) arrayToAllocate, numBytes);
	}

	void freeDataOnCompDevice(void** arrayToFree)
	{
		//std::cout << "FREE_GPU\n";
		cudaFree(*arrayToFree);
	}

	void transferDataFromCompDeviceToHost(void* destArray, const void* inArray, int numBytesTransfer)
	{
		//std::cout << "TRANSFER_GPU\n";
		cudaMemcpy(destArray,
				inArray,
				numBytesTransfer,
				cudaMemcpyDeviceToHost);
	}

	void transferDataFromCompHostToDevice(void* destArray, const void* inArray, int numBytesTransfer)
	{
		//std::cout << "TRANSFER_GPU\n";
		cudaMemcpy(destArray,
				inArray,
				numBytesTransfer,
				cudaMemcpyHostToDevice);
	}
};

#endif //RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_
