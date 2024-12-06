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
#include "BpRunProcessing/ProcessBp.h"
#include "BpRunProcessing/BpParallelParams.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunEval/RunEvalConstsEnums.h"

/**
 * @brief Class that define functions used in processing bp in the
 * CUDA implementation
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @tparam ACCELERATION 
 */
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class ProcessBpCUDA final : public ProcessBp<T, DISP_VALS, ACCELERATION>
{
public:
  ProcessBpCUDA(const ParallelParams& cuda_params) : 
    ProcessBp<T, DISP_VALS, ACCELERATION>(cuda_params) {}

private:
  /**
   * @brief Initialize the data cost at each pixel for each disparity value
   * 
   * @param alg_settings 
   * @param current_bp_level 
   * @param images_target_device 
   * @param data_costs_device 
   * @return run_eval::Status 
   */
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

  /**
   * @brief Initialize the message values for every pixel at every disparity to 0
   * 
   * @param current_bp_level 
   * @param messages_device 
   * @param bp_settings_num_disp_vals 
   * @return run_eval::Status 
   */
  run_eval::Status InitializeMessageValsToDefault(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    unsigned int bp_settings_num_disp_vals) const override;

  /**
   * @brief Run the given number of iterations of BP at the current level
   * using the given message values in global device memory
   * 
   * @param alg_settings 
   * @param current_bp_level 
   * @param data_costs_device 
   * @param messages_device 
   * @param allocated_memory 
   * @return run_eval::Status 
   */
  run_eval::Status RunBPAtCurrentLevel(
    const beliefprop::BpSettings& alg_settings,
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    T* allocated_memory) const override;

  
  /**
   * @brief Copy the computed BP message values from the current now-completed level
   * to the corresponding slots in the next level "down" in the computation
   * pyramid; the next level down is double the width and height of the current level
   * so each message in the current level is copied into four "slots"
   * in the next level down
   * Need two different "sets" of message values to avoid read-write conflicts
   * 
   * @param current_bp_level 
   * @param next_bp_level 
   * @param messages_device_copy_from 
   * @param messages_device_copy_to 
   * @param bp_settings_num_disp_vals 
   * @return run_eval::Status 
   */
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
