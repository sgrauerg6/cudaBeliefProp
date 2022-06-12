/*
 * RunBpStereoSetMemoryManagement.h
 *
 * Class for memory management with functions defined
 * for standard memory allocation using CPU and can be
 * overridden to support other computation devices
 */

#ifndef RUN_BP_STEREO_SET_MEMORY_MANAGEMENT_H_
#define RUN_BP_STEREO_SET_MEMORY_MANAGEMENT_H_

#include <new>
#include <algorithm>

//Class for memory management with functions defined for standard memory allocation using CPU
//Class functions can be overridden to support other computation devices such as GPU
//only processing that uses RunBpStereoSetMemoryManagement is the input stereo
//images and output disparity map that always uses float data type
class RunBpStereoSetMemoryManagement
{
public:
	virtual float* allocateDataOnCompDevice(const unsigned int numData) {
		return (new float[numData]);
	}

	virtual void freeDataOnCompDevice(float* arrayToFree) {
		delete [] arrayToFree;
	}

	virtual void transferDataFromCompDeviceToHost(float* destArray, const float* inArray, const unsigned int numDataTransfer) {
		std::copy(inArray, inArray + numDataTransfer, destArray);
	}

	virtual void transferDataFromCompHostToDevice(float* destArray, const float* inArray, const unsigned int numDataTransfer) {
		std::copy(inArray, inArray + numDataTransfer, destArray);
	}
};

#endif //RUN_BP_STEREO_SET_MEMORY_MANAGEMENT_H_
