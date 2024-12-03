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
#include "BpRunProcessing/ProcessBPOnTargetDevice.h"
#include "BpRunProcessing/BpParallelParams.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunEval/RunEvalConstsEnums.h"

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class ProcessCUDABP final : public ProcessBPOnTargetDevice<T, DISP_VALS, ACCELERATION>
{
public:
  ProcessCUDABP(const ParallelParams& cuda_params) : 
    ProcessBPOnTargetDevice<T, DISP_VALS, ACCELERATION>(cuda_params) {}

private:
  //initialize the data cost at each pixel for each disparity value
  run_eval::Status InitializeDataCosts(
    const beliefprop::BpSettings& alg_settings,
    const beliefprop::BpLevel& current_bp_level,
    const std::array<float*, 2>& images_target_device,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device) const override;

  run_eval::Status InitializeDataCurrentLevel(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::BpLevel& prev_bp_level,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device_write,
    unsigned int bp_settings_num_disp_vals) const override;

  //initialize the message values for every pixel at every disparity to 0
  run_eval::Status InitializeMessageValsToDefault(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    unsigned int bp_settings_num_disp_vals) const override;

  //run the given number of iterations of BP at the current level using the given message values in global device memory
  run_eval::Status RunBPAtCurrentLevel(
    const beliefprop::BpSettings& alg_settings,
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    T* allocated_memory) const override;

  //copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
  //pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
  //in the next level down
  //need two different "sets" of message values to avoid read-write conflicts
  run_eval::Status CopyMessageValuesToNextLevelDown(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::BpLevel& next_bp_level,
    const beliefprop::CheckerboardMessages<T*>& messages_device_copy_from,
    const beliefprop::CheckerboardMessages<T*>& messages_device_copy_to,
    unsigned int bp_settings_num_disp_vals) const override;

  float* RetrieveOutputDisparity(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    unsigned int bp_settings_num_disp_vals) const override;
  
  run_eval::Status ErrorCheck(const char *file = "", int line = 0, bool abort = false) const override;
};

#endif //RUN_BP_STEREO_HOST_HEADER_H
