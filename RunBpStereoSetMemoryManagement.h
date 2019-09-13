#ifndef RUN_BP_STEREO_SET_MEMORY_MANAGEMENT_H_
#define RUN_BP_STEREO_SET_MEMORY_MANAGEMENT_H_

#include <stdlib.h>
#include <cstring>

class RunBpStereoSetMemoryManagement
{
public:
	RunBpStereoSetMemoryManagement() {
	}

	virtual ~RunBpStereoSetMemoryManagement() {
	}

	virtual void allocateDataOnCompDevice(void** arrayToAllocate, int numBytes)
	{
		//allocate the space for the disparity map estimation
		*arrayToAllocate = malloc(numBytes);
	}

	virtual void freeDataOnCompDevice(void** arrayToFree)
	{
		free(*arrayToFree);
	}

	virtual void transferDataFromCompDeviceToHost(void* destArray, void* inArray, int numBytesTransfer)
	{
		memcpy(destArray, inArray, numBytesTransfer);
	}

	virtual void transferDataFromCompHostToDevice(void* destArray, void* inArray, int numBytesTransfer)
	{
		memcpy(destArray, inArray, numBytesTransfer);
	}
};

#endif //RUN_BP_STEREO_SET_MEMORY_MANAGEMENT_H_
