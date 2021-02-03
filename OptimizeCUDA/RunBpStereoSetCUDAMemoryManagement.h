#ifndef RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_
#define RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_

#include "../BpAndSmoothProcessing/RunBpStereoSetMemoryManagement.h"
#include <cuda_runtime.h>

template <typename T = float, typename U = float*>
class RunBpStereoSetCUDAMemoryManagement : public RunBpStereoSetMemoryManagement<T, U>
{
public:

	U allocateDataOnCompDevice(const unsigned int numData) {
		//std::cout << "ALLOC_GPU\n";
		//allocate the space for the disparity map estimation
		U arrayToAllocate;
		cudaMalloc((void **) &arrayToAllocate, numData * sizeof(T));
		return arrayToAllocate;
	}

	void freeDataOnCompDevice(U arrayToFree) {
		//std::cout << "FREE_GPU\n";
		cudaFree(arrayToFree);
	}

	void transferDataFromCompDeviceToHost(T* destArray, const U inArray, const unsigned int numDataTransfer) {
		//std::cout << "TRANSFER_GPU\n";
		cudaMemcpy(destArray, inArray, numDataTransfer * sizeof(T), cudaMemcpyDeviceToHost);
	}

	void transferDataFromCompHostToDevice(U destArray, const T* inArray, const unsigned int numDataTransfer) {
		//std::cout << "TRANSFER_GPU\n";
		cudaMemcpy(destArray, inArray, numDataTransfer * sizeof(T), cudaMemcpyHostToDevice);
	}
};

#endif //RUN_BP_STEREO_SET_CUDA_MEMORY_MANAGEMENT_H_
