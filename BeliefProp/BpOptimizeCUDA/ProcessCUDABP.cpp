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

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::errorCheck(const char *file, int line, bool abort) const {
  const auto code = cudaPeekAtLastError();
  if (code != cudaSuccess) {
    std::cout << "CUDA ERROR: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    cudaGetLastError();
    cudaDeviceReset();
    cudaDeviceSynchronize();
    cudaSetDevice(0);
    if (abort) { exit(code); }
    return run_eval::Status::ERROR;
   }
   return run_eval::Status::NO_ERROR;
}

//functions directed related to running BP to retrieve the movement between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::runBPAtCurrentLevel(
  const beliefprop::BPsettings& algSettings,
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
  const beliefprop::checkerboardMessages<T*>& messagesDevice,
  T* allocatedMemForProcessing)
{
  //set to prefer L1 cache since shared memory is not used in this implementation
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  const dim3 threads(this->parallelParams_.getOptParamsForKernel({beliefprop::BpKernel::BP_AT_LEVEL, currentLevelProperties.levelNum_})[0],
                     this->parallelParams_.getOptParamsForKernel({beliefprop::BpKernel::BP_AT_LEVEL, currentLevelProperties.levelNum_})[1]);
  const dim3 grid{(unsigned int)ceil((float)(currentLevelProperties.widthCheckerboardLevel_) / (float)threads.x), //only updating half at a time
                  (unsigned int)ceil((float)currentLevelProperties.heightLevel_ / (float)threads.y)};

  //in cuda kernel storing data one at a time (though it is coalesced), so numDataInSIMDVector not relevant here and set to 1
  //still is a check if start of row is aligned
  const bool dataAligned{GenProcessingFuncts::MemoryAlignedAtDataStart(0, 1, currentLevelProperties.numDataAlignWidth_, currentLevelProperties.divPaddedChBoardWAlign_)};

  //at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
  for (unsigned int iterationNum = 0; iterationNum < algSettings.numIterations_; iterationNum++)
  {
    beliefprop::Checkerboard_Parts checkboardPartUpdate =
      ((iterationNum % 2) == 0) ?
      beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1 :
      beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0;
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
      dataCostDeviceCheckerboard.dataCostCheckerboard0_,
      dataCostDeviceCheckerboard.dataCostCheckerboard1_,
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
      algSettings.disc_k_bp, dataAligned);

#else
    if constexpr (DISP_VALS > 0) {
      beliefpropCUDA::runBPIterationUsingCheckerboardUpdates<T, DISP_VALS> <<<grid, threads>>> (checkboardPartUpdate, currentLevelProperties,
        dataCostDeviceCheckerboard.dataCostCheckerboard0_,
        dataCostDeviceCheckerboard.dataCostCheckerboard1_,
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
        algSettings.disc_k_bp_, dataAligned, algSettings.numDispVals_);
    }
    else {
      beliefpropCUDA::runBPIterationUsingCheckerboardUpdates<T, DISP_VALS> <<<grid, threads>>> (checkboardPartUpdate, currentLevelProperties,
        dataCostDeviceCheckerboard.dataCostCheckerboard0_,
        dataCostDeviceCheckerboard.dataCostCheckerboard1_,
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1],
        messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
        algSettings.disc_k_bp_, dataAligned, algSettings.numDispVals_, allocatedMemForProcessing);
    }
#endif

    cudaDeviceSynchronize();
    if (errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) {
      return run_eval::Status::ERROR;
    }
  }
  return run_eval::Status::NO_ERROR;
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::copyMessageValuesToNextLevelDown(
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::levelProperties& nextlevelProperties,
  const beliefprop::checkerboardMessages<T*>& messagesDeviceCopyFrom,
  const beliefprop::checkerboardMessages<T*>& messagesDeviceCopyTo,
  const unsigned int bpSettingsNumDispVals)
{
  const dim3 threads(this->parallelParams_.getOptParamsForKernel({beliefprop::BpKernel::COPY_AT_LEVEL, currentLevelProperties.levelNum_})[0],
                     this->parallelParams_.getOptParamsForKernel({beliefprop::BpKernel::COPY_AT_LEVEL, currentLevelProperties.levelNum_})[1]);
  const dim3 grid{(unsigned int)ceil((float)(currentLevelProperties.widthCheckerboardLevel_) / (float)threads.x),
                  (unsigned int)ceil((float)(currentLevelProperties.heightLevel_) / (float)threads.y)};

  cudaDeviceSynchronize();
  if (errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) {
    return run_eval::Status::ERROR;
  }

  for (const auto& checkerboard_part : {beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0, beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1})
  {
    //call the kernel to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
    //storing the current message values
    beliefpropCUDA::copyMsgDataToNextLevel<T, DISP_VALS> <<< grid, threads >>> (checkerboard_part, currentLevelProperties, nextlevelProperties,
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
      bpSettingsNumDispVals);

    cudaDeviceSynchronize();
    if (errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) {
      return run_eval::Status::ERROR;
    }
  }
  return run_eval::Status::NO_ERROR;
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::initializeDataCosts(
  const beliefprop::BPsettings& algSettings,
  const beliefprop::levelProperties& currentLevelProperties,
  const std::array<float*, 2>& imagesOnTargetDevice,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard)
{
  if (errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) {
    return run_eval::Status::ERROR;
  }

  //since this is first kernel run in BP, set to prefer L1 cache for now since no shared memory is used by default
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  //setup execution parameters
  //the thread size remains constant throughout but the grid size is adjusted based on the current level/kernel to run
  const dim3 threads(this->parallelParams_.getOptParamsForKernel({beliefprop::BpKernel::DATA_COSTS_AT_LEVEL, 0})[0],
                     this->parallelParams_.getOptParamsForKernel({beliefprop::BpKernel::DATA_COSTS_AT_LEVEL, 0})[1]);
  //kernel run on full-sized image to retrieve data costs at the "bottom" level of the pyramid
  const dim3 grid{(unsigned int)ceil((float)currentLevelProperties.widthLevel_ / (float)threads.x),
                  (unsigned int)ceil((float)currentLevelProperties.heightLevel_ / (float)threads.y)};

  //initialize the data the the "bottom" of the image pyramid
  beliefpropCUDA::initializeBottomLevelData<T, DISP_VALS> <<<grid, threads>>> (currentLevelProperties, imagesOnTargetDevice[0],
    imagesOnTargetDevice[1], dataCostDeviceCheckerboard.dataCostCheckerboard0_,
    dataCostDeviceCheckerboard.dataCostCheckerboard1_, algSettings.lambda_bp_, algSettings.data_k_bp_,
    algSettings.numDispVals_);
  cudaDeviceSynchronize();
  
  if (errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) {
    return run_eval::Status::ERROR;
  }

  return run_eval::Status::NO_ERROR;
}

//initialize the message values with no previous message values...all message values are set to 0
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::initializeMessageValsToDefault(
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::checkerboardMessages<T*>& messagesDevice,
  const unsigned int bpSettingsNumDispVals)
{
  const dim3 threads(this->parallelParams_.getOptParamsForKernel({beliefprop::BpKernel::INIT_MESSAGE_VALS, 0})[0],
                     this->parallelParams_.getOptParamsForKernel({beliefprop::BpKernel::INIT_MESSAGE_VALS, 0})[1]);
  const dim3 grid{(unsigned int)ceil((float)currentLevelProperties.widthCheckerboardLevel_ / (float)threads.x),
                  (unsigned int)ceil((float)currentLevelProperties.heightLevel_ / (float)threads.y)};

  //initialize all the message values for each pixel at each possible movement to the default value in the kernel
  beliefpropCUDA::initializeMessageValsToDefaultKernel<T, DISP_VALS> <<< grid, threads >>> (currentLevelProperties,
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
    bpSettingsNumDispVals);
  cudaDeviceSynchronize();
  
  if (errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) {
    return run_eval::Status::ERROR;
  }

  return run_eval::Status::NO_ERROR;
}

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
run_eval::Status ProcessCUDABP<T, DISP_VALS, ACCELERATION>::initializeDataCurrentLevel(const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::levelProperties& prevLevelProperties,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboardWriteTo,
  const unsigned int bpSettingsNumDispVals)
{
  const dim3 threads(this->parallelParams_.getOptParamsForKernel({beliefprop::BpKernel::DATA_COSTS_AT_LEVEL, currentLevelProperties.levelNum_})[0],
                     this->parallelParams_.getOptParamsForKernel({beliefprop::BpKernel::DATA_COSTS_AT_LEVEL, currentLevelProperties.levelNum_})[1]);
  //each pixel "checkerboard" is half the width of the level and there are two of them; each "pixel/point" at the level belongs to one checkerboard and
  //the four-connected neighbors are in the other checkerboard
  const dim3 grid{(unsigned int)ceil(((float)currentLevelProperties.widthCheckerboardLevel_) / (float)threads.x),
                  (unsigned int)ceil((float)currentLevelProperties.heightLevel_ / (float)threads.y)};

  if (errorCheck(__FILE__, __LINE__ ) != run_eval::Status::NO_ERROR) {
    return run_eval::Status::ERROR;
  }

  const size_t offsetNum{0};
  for (const auto& checkerboardAndDataCost : {
         std::make_pair(beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0, dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard0_),
         std::make_pair(beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1,  dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard1_)})
  {
    beliefpropCUDA::initializeCurrentLevelData<T, DISP_VALS> <<<grid, threads>>> (checkerboardAndDataCost.first,
      currentLevelProperties, prevLevelProperties,
      dataCostDeviceCheckerboard.dataCostCheckerboard0_,
      dataCostDeviceCheckerboard.dataCostCheckerboard1_,
      checkerboardAndDataCost.second, ((unsigned int) offsetNum / sizeof(float)),
      bpSettingsNumDispVals);

    cudaDeviceSynchronize();
    if (errorCheck(__FILE__, __LINE__ ) != run_eval::Status::NO_ERROR) {
      return run_eval::Status::ERROR;
    }
  }
  return run_eval::Status::NO_ERROR;
}

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
float* ProcessCUDABP<T, DISP_VALS, ACCELERATION>::retrieveOutputDisparity(
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
  const beliefprop::checkerboardMessages<T*>& messagesDevice,
  const unsigned int bpSettingsNumDispVals)
{
  float* resultingDisparityMapCompDevice;
  cudaMalloc((void**)&resultingDisparityMapCompDevice, currentLevelProperties.widthLevel_ * currentLevelProperties.heightLevel_ * sizeof(float));

  const dim3 threads(this->parallelParams_.getOptParamsForKernel({beliefprop::BpKernel::OUTPUT_DISP, 0})[0],
                     this->parallelParams_.getOptParamsForKernel({beliefprop::BpKernel::OUTPUT_DISP, 0})[1]);
  const dim3 grid{(unsigned int)ceil((float)currentLevelProperties.widthCheckerboardLevel_ / (float)threads.x),
                  (unsigned int)ceil((float)currentLevelProperties.heightLevel_ / (float)threads.y)};

  beliefpropCUDA::retrieveOutputDisparity<T, DISP_VALS> <<<grid, threads>>> (currentLevelProperties,
    dataCostDeviceCheckerboard.dataCostCheckerboard0_, dataCostDeviceCheckerboard.dataCostCheckerboard1_,
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
    resultingDisparityMapCompDevice, bpSettingsNumDispVals);
  cudaDeviceSynchronize();
  if (errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) {
    return nullptr;
  }

  return resultingDisparityMapCompDevice;
}

template class ProcessCUDABP<float, 0, run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<double, 0, run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<halftype, 0, run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5], run_environment::AccSetting::CUDA>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6], run_environment::AccSetting::CUDA>;
