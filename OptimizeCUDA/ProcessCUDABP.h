/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

//This function declares the host functions to run the CUDA implementation of Stereo estimation using BP

#ifndef RUN_BP_STEREO_HOST_HEADER_CUH
#define RUN_BP_STEREO_HOST_HEADER_CUH

#include "../ParameterFiles/bpStereoCudaParameters.h"

//include for the kernel functions to be run on the GPU
#include <cuda_runtime.h>
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"
#include <cuda_fp16.h>

//define concepts of allowed data types for belief propagation data storage and processing
template <typename T>
concept BpData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, half>;

template <typename T>
concept BpDataProcess_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, half>;

template<BpData_t T, unsigned int DISP_VALS>
class ProcessCUDABP : public ProcessBPOnTargetDevice<T, DISP_VALS, beliefprop::AccSetting::CUDA>
{
public:
  ProcessCUDABP(const beliefprop::ParallelParameters& cudaParams) : cudaParams_(cudaParams) {}

  //initialize the data cost at each pixel for each disparity value
  beliefprop::Status initializeDataCosts(
    const beliefprop::BPsettings& algSettings,
    const beliefprop::levelProperties& currentLevelProperties,
    const std::array<float*, 2>& imagesOnTargetDevice,
    const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard) override;

  beliefprop::Status initializeDataCurrentLevel(
    const beliefprop::levelProperties& currentLevelProperties,
    const beliefprop::levelProperties& prevLevelProperties,
    const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
    const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboardWriteTo,
    const unsigned int bpSettingsNumDispVals) override;

  //initialize the message values for every pixel at every disparity to DEFAULT_INITIAL_MESSAGE_VAL (value is 0.0f unless changed)
  beliefprop::Status initializeMessageValsToDefault(
    const beliefprop::levelProperties& currentLevelProperties,
    const beliefprop::checkerboardMessages<T*>& messagesDevice,
    const unsigned int bpSettingsNumDispVals) override;

  //run the given number of iterations of BP at the current level using the given message values in global device memory
  beliefprop::Status runBPAtCurrentLevel(
    const beliefprop::BPsettings& algSettings,
    const beliefprop::levelProperties& currentLevelProperties,
    const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
    const beliefprop::checkerboardMessages<T*>& messagesDevice,
    T* allocatedMemForProcessing) override;

  //copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
  //pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
  //in the next level down
  //need two different "sets" of message values to avoid read-write conflicts
  beliefprop::Status copyMessageValuesToNextLevelDown(
    const beliefprop::levelProperties& currentLevelProperties,
    const beliefprop::levelProperties& nextlevelProperties,
    const beliefprop::checkerboardMessages<T*>& messagesDeviceCopyFrom,
    const beliefprop::checkerboardMessages<T*>& messagesDeviceCopyTo,
    const unsigned int bpSettingsNumDispVals) override;

  float* retrieveOutputDisparity(
    const beliefprop::levelProperties& currentLevelProperties,
    const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
    const beliefprop::checkerboardMessages<T*>& messagesDevice,
    const unsigned int bpSettingsNumDispVals) override;
  
  beliefprop::Status errorCheck(const char *file = "", int line = 0, bool abort = false) const override;

private:
  const beliefprop::ParallelParameters& cudaParams_;
};

#endif //RUN_BP_STEREO_HOST_HEADER_CUH
