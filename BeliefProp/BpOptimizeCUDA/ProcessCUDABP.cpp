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

//Defines the functions to run the CUDA implementation of 2-D Stereo estimation using BP

#include <iostream>
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunImp/RunImpGenFuncts.h"
#include "ProcessCUDABP.h"
#include "kernelBpStereo.cu"

//return whether or not there was an error in CUDA processing
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::ErrorCheck(const char *file, int line, bool abort) const {
  const auto code = cudaPeekAtLastError();
  if (code != cudaSuccess) {
    std::cout << "CUDA ERROR: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    cudaGetLastError();
    cudaDeviceReset();
    cudaDeviceSynchronize();
    cudaSetDevice(0);
    if (abort) { 
      exit(code);
    }
    return run_eval::Status::kError;
   }
   return run_eval::Status::kNoError;
}

//functions for processing BP to retrieve the disparity between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::RunBPAtCurrentLevel(
  const beliefprop::BpSettings& alg_settings,
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
  const beliefprop::CheckerboardMessages<T*>& messages_device,
  T* allocated_memory)
{
  //set to prefer L1 cache since shared memory is not used in this implementation
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  const auto kernel_thread_block_dims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kBpAtLevel), current_bp_level.LevelProperties().level_num_});
  const dim3 threads{kernel_thread_block_dims[0], kernel_thread_block_dims[1]};
  const dim3 grid{(unsigned int)ceil((float)(current_bp_level.LevelProperties().width_checkerboard_level_) / (float)threads.x), //only updating half at a time
                  (unsigned int)ceil((float)current_bp_level.LevelProperties().height_level_ / (float)threads.y)};

  //in cuda kernel storing data one at a time (though it is coalesced), so num_data_SIMD_vect not relevant here and set to 1
  //still is a check if start of row is aligned
  const bool data_aligned{run_imp_util::MemoryAlignedAtDataStart(0, 1, current_bp_level.LevelProperties().num_data_align_width_,
    current_bp_level.LevelProperties().div_padded_checkerboard_w_align_)};

  //at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
  for (unsigned int iteration_num = 0; iteration_num < alg_settings.num_iterations; iteration_num++)
  {
    beliefprop::Checkerboard_Part checkerboard_part_update =
      ((iteration_num % 2) == 0) ?
      beliefprop::Checkerboard_Part::kCheckerboardPart1 :
      beliefprop::Checkerboard_Part::kCheckerboardPart0;
    cudaDeviceSynchronize();

    if constexpr (DISP_VALS > 0) {
      beliefpropCUDA::RunBPIterationUsingCheckerboardUpdates<T, DISP_VALS> <<<grid, threads>>> (checkerboard_part_update, current_bp_level.LevelProperties(),
        data_costs_device[0],
        data_costs_device[1],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
        alg_settings.disc_k_bp, data_aligned, alg_settings.num_disp_vals);
    }
    else {
      beliefpropCUDA::RunBPIterationUsingCheckerboardUpdates<T, DISP_VALS> <<<grid, threads>>> (checkerboard_part_update, current_bp_level.LevelProperties(),
        data_costs_device[0],
        data_costs_device[1],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
        messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
        alg_settings.disc_k_bp, data_aligned, alg_settings.num_disp_vals, allocated_memory);
    }

    cudaDeviceSynchronize();
    if (ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
      return run_eval::Status::kError;
    }
  }
  return run_eval::Status::kNoError;
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::CopyMessageValuesToNextLevelDown(
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::BpLevel& next_bp_level,
  const beliefprop::CheckerboardMessages<T*>& messages_device_copy_from,
  const beliefprop::CheckerboardMessages<T*>& messages_device_copy_to,
  unsigned int bp_settings_num_disp_vals)
{
  const auto kernel_thread_block_dims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kCopyAtLevel), current_bp_level.LevelProperties().level_num_});
  const dim3 threads{kernel_thread_block_dims[0], kernel_thread_block_dims[1]};
  const dim3 grid{(unsigned int)ceil((float)(current_bp_level.LevelProperties().width_checkerboard_level_) / (float)threads.x),
                  (unsigned int)ceil((float)(current_bp_level.LevelProperties().height_level_) / (float)threads.y)};

  cudaDeviceSynchronize();
  if (ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
    return run_eval::Status::kError;
  }

  for (const auto checkerboard_part : {beliefprop::Checkerboard_Part::kCheckerboardPart0, beliefprop::Checkerboard_Part::kCheckerboardPart1})
  {
    //call the kernel to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
    //storing the current message values
    beliefpropCUDA::CopyMsgDataToNextLevel<T, DISP_VALS> <<< grid, threads >>> (checkerboard_part,
      current_bp_level.LevelProperties(), next_bp_level.LevelProperties(),
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
      bp_settings_num_disp_vals);

    cudaDeviceSynchronize();
    if (ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
      return run_eval::Status::kError;
    }
  }
  return run_eval::Status::kNoError;
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::InitializeDataCosts(
  const beliefprop::BpSettings& alg_settings,
  const beliefprop::BpLevel& current_bp_level,
  const std::array<float*, 2>& images_target_device,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device)
{
  if (ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
    return run_eval::Status::kError;
  }

  //since this is first kernel run in BP, set to prefer L1 cache for now since no shared memory is used by default
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  //setup execution parameters
  //the thread size remains constant throughout but the grid size is adjusted based on the current level/kernel to run
  const auto kernel_thread_block_dims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kDataCostsAtLevel), 0});
  const dim3 threads{kernel_thread_block_dims[0], kernel_thread_block_dims[1]};
  //kernel run on full-sized image to retrieve data costs at the "bottom" level of the pyramid
  const dim3 grid{(unsigned int)ceil((float)current_bp_level.LevelProperties().width_level_ / (float)threads.x),
                  (unsigned int)ceil((float)current_bp_level.LevelProperties().height_level_ / (float)threads.y)};

  //initialize the data the the "bottom" of the image pyramid
  beliefpropCUDA::InitializeBottomLevelData<T, DISP_VALS> <<<grid, threads>>> (current_bp_level.LevelProperties(),
    images_target_device[0], images_target_device[1],
    data_costs_device[0], data_costs_device[1],
    alg_settings.lambda_bp, alg_settings.data_k_bp, alg_settings.num_disp_vals);
  cudaDeviceSynchronize();
  
  if (ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
    return run_eval::Status::kError;
  }

  return run_eval::Status::kNoError;
}

//initialize the message values with no previous message values...all message values are set to 0
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::InitializeMessageValsToDefault(
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::CheckerboardMessages<T*>& messages_device,
  unsigned int bp_settings_num_disp_vals)
{
  const auto kernel_thread_block_dims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kInitMessageVals), 0});
  const dim3 threads{kernel_thread_block_dims[0], kernel_thread_block_dims[1]};
  const dim3 grid{(unsigned int)ceil((float)current_bp_level.LevelProperties().width_checkerboard_level_ / (float)threads.x),
                  (unsigned int)ceil((float)current_bp_level.LevelProperties().height_level_ / (float)threads.y)};

  //initialize all the message values for each pixel at each possible movement to the default value in the kernel
  beliefpropCUDA::InitializeMessageValsToDefaultKernel<T, DISP_VALS> <<< grid, threads >>> (
    current_bp_level.LevelProperties(),
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
    bp_settings_num_disp_vals);
  cudaDeviceSynchronize();
  
  if (ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
    return run_eval::Status::kError;
  }

  return run_eval::Status::kNoError;
}

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::InitializeDataCurrentLevel(const beliefprop::BpLevel& current_bp_level,
  const beliefprop::BpLevel& prev_bp_level,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device_write,
  unsigned int bp_settings_num_disp_vals)
{
  const auto kernel_thread_block_dims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kDataCostsAtLevel), current_bp_level.LevelProperties().level_num_});
  const dim3 threads{kernel_thread_block_dims[0], kernel_thread_block_dims[1]};
  //each pixel "checkerboard" is half the width of the level and there are two of them; each "pixel/point" at the level belongs to one checkerboard and
  //the four-connected neighbors are in the other checkerboard
  const dim3 grid{(unsigned int)ceil(((float)current_bp_level.LevelProperties().width_checkerboard_level_) / (float)threads.x),
                  (unsigned int)ceil((float)current_bp_level.LevelProperties().height_level_ / (float)threads.y)};

  if (ErrorCheck(__FILE__, __LINE__ ) != run_eval::Status::kNoError) {
    return run_eval::Status::kError;
  }

  const size_t offset_num{0};
  for (const auto& checkerboard_data_cost : {
    std::make_pair(beliefprop::Checkerboard_Part::kCheckerboardPart0, data_costs_device_write[0]),
    std::make_pair(beliefprop::Checkerboard_Part::kCheckerboardPart1, data_costs_device_write[1])})
  {
    beliefpropCUDA::InitializeCurrentLevelData<T, DISP_VALS> <<<grid, threads>>> (checkerboard_data_cost.first,
      current_bp_level.LevelProperties(), prev_bp_level.LevelProperties(),
      data_costs_device[0],
      data_costs_device[1],
      checkerboard_data_cost.second, ((unsigned int) offset_num / sizeof(float)),
      bp_settings_num_disp_vals);

    cudaDeviceSynchronize();
    if (ErrorCheck(__FILE__, __LINE__ ) != run_eval::Status::kNoError) {
      return run_eval::Status::kError;
    }
  }
  return run_eval::Status::kNoError;
}

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
float* ProcessCUDABP<T, DISP_VALS, ACCELERATION>::RetrieveOutputDisparity(
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
  const beliefprop::CheckerboardMessages<T*>& messages_device,
  unsigned int bp_settings_num_disp_vals)
{
  float* result_disp_map_device;
  cudaMalloc((void**)&result_disp_map_device, current_bp_level.LevelProperties().width_level_ * current_bp_level.LevelProperties().height_level_ * sizeof(float));

  const auto kernel_thread_block_dims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kOutputDisp), 0});
  const dim3 threads{kernel_thread_block_dims[0], kernel_thread_block_dims[1]};
  const dim3 grid{(unsigned int)ceil((float)current_bp_level.LevelProperties().width_checkerboard_level_ / (float)threads.x),
                  (unsigned int)ceil((float)current_bp_level.LevelProperties().height_level_ / (float)threads.y)};

  beliefpropCUDA::RetrieveOutputDisparity<T, DISP_VALS> <<<grid, threads>>> (current_bp_level.LevelProperties(),
    data_costs_device[0], data_costs_device[1],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
    result_disp_map_device, bp_settings_num_disp_vals);
  cudaDeviceSynchronize();
  if (ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
    return nullptr;
  }

  return result_disp_map_device;
}

template class ProcessCUDABP<float, 0, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[0].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[1].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[2].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[3].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[4].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[5].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[6].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, 0, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[0].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[1].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[2].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[3].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[4].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[5].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[6].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, 0, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[0].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[1].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[2].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[3].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[4].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[5].num_disp_vals, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[6].num_disp_vals, run_environment::AccSetting::kCUDA>;
