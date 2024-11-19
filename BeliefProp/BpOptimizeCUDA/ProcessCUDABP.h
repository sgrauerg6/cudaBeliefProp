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

#ifndef RUN_BP_STEREO_HOST_HEADER_H
#define RUN_BP_STEREO_HOST_HEADER_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "BpConstsAndParams/BpTypeConstraints.h"
#include "BpRunProcessing/ProcessBPOnTargetDevice.h"
#include "BpRunImp/BpParallelParams.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class ProcessCUDABP final : public ProcessBPOnTargetDevice<T, DISP_VALS, ACCELERATION>
{
public:
  ProcessCUDABP(const ParallelParams& cudaParams) : 
    ProcessBPOnTargetDevice<T, DISP_VALS, ACCELERATION>(cudaParams) {}

private:
  //initialize the data cost at each pixel for each disparity value
  run_eval::Status initializeDataCosts(
    const beliefprop::BpSettings& algSettings,
    const beliefprop::BpLevel& currentBpLevel,
    const std::array<float*, 2>& imagesOnTargetDevice,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard) override;

  run_eval::Status initializeDataCurrentLevel(
    const beliefprop::BpLevel& currentBpLevel,
    const beliefprop::BpLevel& prevBpLevel,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboardWriteTo,
    unsigned int bp_settings_num_disp_vals) override;

  //initialize the message values for every pixel at every disparity to 0
  run_eval::Status initializeMessageValsToDefault(
    const beliefprop::BpLevel& currentBpLevel,
    const beliefprop::CheckerboardMessages<T*>& messagesDevice,
    unsigned int bp_settings_num_disp_vals) override;

  //run the given number of iterations of BP at the current level using the given message values in global device memory
  run_eval::Status runBPAtCurrentLevel(
    const beliefprop::BpSettings& algSettings,
    const beliefprop::BpLevel& currentBpLevel,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
    const beliefprop::CheckerboardMessages<T*>& messagesDevice,
    T* allocatedMemForProcessing) override;

  //copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
  //pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
  //in the next level down
  //need two different "sets" of message values to avoid read-write conflicts
  run_eval::Status copyMessageValuesToNextLevelDown(
    const beliefprop::BpLevel& currentBpLevel,
    const beliefprop::BpLevel& nextBpLevel,
    const beliefprop::CheckerboardMessages<T*>& messagesDeviceCopyFrom,
    const beliefprop::CheckerboardMessages<T*>& messagesDeviceCopyTo,
    unsigned int bp_settings_num_disp_vals) override;

  float* retrieveOutputDisparity(
    const beliefprop::BpLevel& currentBpLevel,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
    const beliefprop::CheckerboardMessages<T*>& messagesDevice,
    unsigned int bp_settings_num_disp_vals) override;
  
  run_eval::Status errorCheck(const char *file = "", int line = 0, bool abort = false) const override;
};

#endif //RUN_BP_STEREO_HOST_HEADER_H
