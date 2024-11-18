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
inline run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::errorCheck(const char *file, int line, bool abort) const {
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
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::runBPAtCurrentLevel(
  const beliefprop::BPsettings& algSettings,
  const beliefprop::LevelProperties& currentLevelProperties,
  const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
  const beliefprop::CheckerboardMessages<T*>& messagesDevice,
  T* allocatedMemForProcessing)
{
  //set to prefer L1 cache since shared memory is not used in this implementation
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  const auto kernelTBlockDims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kBpAtLevel), currentLevelProperties.level_num_});
  const dim3 threads{kernelTBlockDims[0], kernelTBlockDims[1]};
  const dim3 grid{(unsigned int)ceil((float)(currentLevelProperties.width_checkerboard_level_) / (float)threads.x), //only updating half at a time
                  (unsigned int)ceil((float)currentLevelProperties.height_level_ / (float)threads.y)};

  //in cuda kernel storing data one at a time (though it is coalesced), so numDataInSIMDVector not relevant here and set to 1
  //still is a check if start of row is aligned
  const bool dataAligned{run_imp_util::MemoryAlignedAtDataStart(0, 1, currentLevelProperties.num_data_align_width_, currentLevelProperties.div_padded_checkerboard_w_align_)};

  //at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
  for (unsigned int iterationNum = 0; iterationNum < algSettings.num_iterations_; iterationNum++)
  {
    beliefprop::Checkerboard_Part checkboardPartUpdate =
      ((iterationNum % 2) == 0) ?
      beliefprop::Checkerboard_Part::kCheckerboardPart1 :
      beliefprop::Checkerboard_Part::kCheckerboardPart0;
    cudaDeviceSynchronize();

#if (((USE_SHARED_MEMORY == 3) || (USE_SHARED_MEMORY == 4))  && (DISP_INDEX_START_REG_LOCAL_MEM > 0))
    int numDataSharedMemory = beliefprop::DEFAULT_CUDA_TB_WIDTH * beliefprop::DEFAULT_CUDA_TB_HEIGHT * (DISP_INDEX_START_REG_LOCAL_MEM);
    int numBytesSharedMemory = numDataSharedMemory * sizeof(T);

#if (USE_SHARED_MEMORY == 4)

    numBytesSharedMemory *= 5;

#endif //(USE_SHARED_MEMORY == 4)

    int maxbytes = numBytesSharedMemory; // 96 KB
    cudaFuncSetAttribute(runBPIterationUsingCheckerboardUpdates<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

    //std::cout << "numDataSharedMemory: " << numDataSharedMemory << std::endl;
    beliefpropCUDA::runBPIterationUsingCheckerboardUpdates<T, DISP_VALS> <<<grid, threads, maxbytes>>> (checkboardPartUpdate, currentLevelProperties,
      dataCostDeviceCheckerboard[0],
      dataCostDeviceCheckerboard[1],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
      algSettings.disc_k_bp, dataAligned);

#else
    if constexpr (DISP_VALS > 0) {
      beliefpropCUDA::runBPIterationUsingCheckerboardUpdates<T, DISP_VALS> <<<grid, threads>>> (checkboardPartUpdate, currentLevelProperties,
        dataCostDeviceCheckerboard[0],
        dataCostDeviceCheckerboard[1],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
        algSettings.disc_k_bp_, dataAligned, algSettings.num_disp_vals_);
    }
    else {
      beliefpropCUDA::runBPIterationUsingCheckerboardUpdates<T, DISP_VALS> <<<grid, threads>>> (checkboardPartUpdate, currentLevelProperties,
        dataCostDeviceCheckerboard[0],
        dataCostDeviceCheckerboard[1],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
        messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
        algSettings.disc_k_bp_, dataAligned, algSettings.num_disp_vals_, allocatedMemForProcessing);
    }
#endif

    cudaDeviceSynchronize();
    if (errorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
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
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::copyMessageValuesToNextLevelDown(
  const beliefprop::LevelProperties& currentLevelProperties,
  const beliefprop::LevelProperties& nextLevelProperties,
  const beliefprop::CheckerboardMessages<T*>& messagesDeviceCopyFrom,
  const beliefprop::CheckerboardMessages<T*>& messagesDeviceCopyTo,
  unsigned int bpSettingsNumDispVals)
{
  const auto kernelTBlockDims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kCopyAtLevel), currentLevelProperties.level_num_});
  const dim3 threads{kernelTBlockDims[0], kernelTBlockDims[1]};
  const dim3 grid{(unsigned int)ceil((float)(currentLevelProperties.width_checkerboard_level_) / (float)threads.x),
                  (unsigned int)ceil((float)(currentLevelProperties.height_level_) / (float)threads.y)};

  cudaDeviceSynchronize();
  if (errorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
    return run_eval::Status::kError;
  }

  for (const auto checkerboard_part : {beliefprop::Checkerboard_Part::kCheckerboardPart0, beliefprop::Checkerboard_Part::kCheckerboardPart1})
  {
    //call the kernel to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
    //storing the current message values
    beliefpropCUDA::copyMsgDataToNextLevel<T, DISP_VALS> <<< grid, threads >>> (checkerboard_part, currentLevelProperties, nextLevelProperties,
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
      bpSettingsNumDispVals);

    cudaDeviceSynchronize();
    if (errorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
      return run_eval::Status::kError;
    }
  }
  return run_eval::Status::kNoError;
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::initializeDataCosts(
  const beliefprop::BPsettings& algSettings,
  const beliefprop::LevelProperties& currentLevelProperties,
  const std::array<float*, 2>& imagesOnTargetDevice,
  const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard)
{
  if (errorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
    return run_eval::Status::kError;
  }

  //since this is first kernel run in BP, set to prefer L1 cache for now since no shared memory is used by default
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  //setup execution parameters
  //the thread size remains constant throughout but the grid size is adjusted based on the current level/kernel to run
  const auto kernelTBlockDims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kDataCostsAtLevel), 0});
  const dim3 threads{kernelTBlockDims[0], kernelTBlockDims[1]};
  //kernel run on full-sized image to retrieve data costs at the "bottom" level of the pyramid
  const dim3 grid{(unsigned int)ceil((float)currentLevelProperties.width_level_ / (float)threads.x),
                  (unsigned int)ceil((float)currentLevelProperties.height_level_ / (float)threads.y)};

  //initialize the data the the "bottom" of the image pyramid
  beliefpropCUDA::initializeBottomLevelData<T, DISP_VALS> <<<grid, threads>>> (currentLevelProperties, imagesOnTargetDevice[0],
    imagesOnTargetDevice[1], dataCostDeviceCheckerboard[0],
    dataCostDeviceCheckerboard[1], algSettings.lambda_bp_, algSettings.data_k_bp_,
    algSettings.num_disp_vals_);
  cudaDeviceSynchronize();
  
  if (errorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
    return run_eval::Status::kError;
  }

  return run_eval::Status::kNoError;
}

//initialize the message values with no previous message values...all message values are set to 0
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::initializeMessageValsToDefault(
  const beliefprop::LevelProperties& currentLevelProperties,
  const beliefprop::CheckerboardMessages<T*>& messagesDevice,
  unsigned int bpSettingsNumDispVals)
{
  const auto kernelTBlockDims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kInitMessageVals), 0});
  const dim3 threads{kernelTBlockDims[0], kernelTBlockDims[1]};
  const dim3 grid{(unsigned int)ceil((float)currentLevelProperties.width_checkerboard_level_ / (float)threads.x),
                  (unsigned int)ceil((float)currentLevelProperties.height_level_ / (float)threads.y)};

  //initialize all the message values for each pixel at each possible movement to the default value in the kernel
  beliefpropCUDA::initializeMessageValsToDefaultKernel<T, DISP_VALS> <<< grid, threads >>> (currentLevelProperties,
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
    bpSettingsNumDispVals);
  cudaDeviceSynchronize();
  
  if (errorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
    return run_eval::Status::kError;
  }

  return run_eval::Status::kNoError;
}

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::initializeDataCurrentLevel(const beliefprop::LevelProperties& currentLevelProperties,
  const beliefprop::LevelProperties& prevLevelProperties,
  const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
  const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboardWriteTo,
  unsigned int bpSettingsNumDispVals)
{
  const auto kernelTBlockDims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kDataCostsAtLevel), currentLevelProperties.level_num_});
  const dim3 threads{kernelTBlockDims[0], kernelTBlockDims[1]};
  //each pixel "checkerboard" is half the width of the level and there are two of them; each "pixel/point" at the level belongs to one checkerboard and
  //the four-connected neighbors are in the other checkerboard
  const dim3 grid{(unsigned int)ceil(((float)currentLevelProperties.width_checkerboard_level_) / (float)threads.x),
                  (unsigned int)ceil((float)currentLevelProperties.height_level_ / (float)threads.y)};

  if (errorCheck(__FILE__, __LINE__ ) != run_eval::Status::kNoError) {
    return run_eval::Status::kError;
  }

  const size_t offsetNum{0};
  for (const auto& checkerboardAndDataCost : {
    std::make_pair(beliefprop::Checkerboard_Part::kCheckerboardPart0, dataCostDeviceCheckerboardWriteTo[0]),
    std::make_pair(beliefprop::Checkerboard_Part::kCheckerboardPart1,  dataCostDeviceCheckerboardWriteTo[1])})
  {
    beliefpropCUDA::initializeCurrentLevelData<T, DISP_VALS> <<<grid, threads>>> (checkerboardAndDataCost.first,
      currentLevelProperties, prevLevelProperties,
      dataCostDeviceCheckerboard[0],
      dataCostDeviceCheckerboard[1],
      checkerboardAndDataCost.second, ((unsigned int) offsetNum / sizeof(float)),
      bpSettingsNumDispVals);

    cudaDeviceSynchronize();
    if (errorCheck(__FILE__, __LINE__ ) != run_eval::Status::kNoError) {
      return run_eval::Status::kError;
    }
  }
  return run_eval::Status::kNoError;
}

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
float* ProcessCUDABP<T, DISP_VALS, ACCELERATION>::retrieveOutputDisparity(
  const beliefprop::LevelProperties& currentLevelProperties,
  const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
  const beliefprop::CheckerboardMessages<T*>& messagesDevice,
  unsigned int bpSettingsNumDispVals)
{
  float* resultingDisparityMapCompDevice;
  cudaMalloc((void**)&resultingDisparityMapCompDevice, currentLevelProperties.width_level_ * currentLevelProperties.height_level_ * sizeof(float));

  const auto kernelTBlockDims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kOutputDisp), 0});
  const dim3 threads{kernelTBlockDims[0], kernelTBlockDims[1]};
  const dim3 grid{(unsigned int)ceil((float)currentLevelProperties.width_checkerboard_level_ / (float)threads.x),
                  (unsigned int)ceil((float)currentLevelProperties.height_level_ / (float)threads.y)};

  beliefpropCUDA::retrieveOutputDisparity<T, DISP_VALS> <<<grid, threads>>> (currentLevelProperties,
    dataCostDeviceCheckerboard[0], dataCostDeviceCheckerboard[1],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
    resultingDisparityMapCompDevice, bpSettingsNumDispVals);
  cudaDeviceSynchronize();
  if (errorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
    return nullptr;
  }

  return resultingDisparityMapCompDevice;
}

template class ProcessCUDABP<float, 0, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[0].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[1].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[2].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[3].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[4].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[5].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<float, bp_params::kStereoSetsToProcess[6].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, 0, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[0].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[1].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[2].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[3].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[4].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[5].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<double, bp_params::kStereoSetsToProcess[6].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, 0, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[0].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[1].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[2].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[3].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[4].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[5].num_disp_vals_, run_environment::AccSetting::kCUDA>;
template class ProcessCUDABP<halftype, bp_params::kStereoSetsToProcess[6].num_disp_vals_, run_environment::AccSetting::kCUDA>;
