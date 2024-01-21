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
#include "BpConstsAndParams/bpTypeConstraints.h"
#include "RunSettingsEval/RunTypeConstraints.h"

//Class for memory management with functions defined for standard memory allocation using CPU
//Class functions can be overridden to support other computation devices such as GPU
template <RunData_t T>
class RunBpStereoSetMemoryManagement
{
public:
  virtual T* allocateMemoryOnDevice(const unsigned int numData) {
    return (new T[numData]);
  }

  virtual void freeMemoryOnDevice(T* arrayToFree) {
    delete [] arrayToFree;
  }

  virtual T* allocateAlignedMemoryOnDevice(const unsigned long numData, run_environment::AccSetting accSetting)
  {
#ifdef _WIN32
    T* memoryData = static_cast<T*>(_aligned_malloc(numData * sizeof(T), run_environment::getNumDataAlignWidth(accSetting) * sizeof(T)));
    return memoryData;
#else
    T* memoryData = static_cast<T*>(std::aligned_alloc(run_environment::getNumDataAlignWidth(accSetting) * sizeof(T), numData * sizeof(T)));
    return memoryData;
#endif
  }

  virtual void freeAlignedMemoryOnDevice(T* memoryToFree)
  {
#ifdef _WIN32
    _aligned_free(memoryToFree);
#else
    free(memoryToFree);
#endif
  }

  virtual void transferDataFromDeviceToHost(T* destArray, const T* inArray, const unsigned int numDataTransfer) {
    std::copy(inArray, inArray + numDataTransfer, destArray);
  }

  virtual void transferDataFromHostToDevice(T* destArray, const T* inArray, const unsigned int numDataTransfer) {
    std::copy(inArray, inArray + numDataTransfer, destArray);
  }
};

#endif //RUN_BP_STEREO_SET_MEMORY_MANAGEMENT_H_
