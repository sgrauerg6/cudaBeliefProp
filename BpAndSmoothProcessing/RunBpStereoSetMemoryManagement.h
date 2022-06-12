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
template <typename T = float, typename U = float*>
class RunBpStereoSetMemoryManagement
{
public:
	virtual U allocateDataOnCompDevice(const unsigned int numData) {
		return (new T[numData]);
	}

	virtual void freeDataOnCompDevice(U arrayToFree) {
		delete [] arrayToFree;
	}

	virtual void transferDataFromCompDeviceToHost(T* destArray, const U inArray, const unsigned int numDataTransfer) {
		std::copy(inArray, inArray + numDataTransfer, destArray);
	}

	virtual void transferDataFromCompHostToDevice(U destArray, const T* inArray, const unsigned int numDataTransfer) {
		std::copy(inArray, inArray + numDataTransfer, destArray);
	}
};

#endif //RUN_BP_STEREO_SET_MEMORY_MANAGEMENT_H_
