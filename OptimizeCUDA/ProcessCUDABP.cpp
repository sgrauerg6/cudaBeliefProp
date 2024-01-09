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

#include "ProcessCUDABP.h"
#include "kernelBpStereo.cu"
#include "../ParameterFiles/bpStereoCudaParameters.h"
#include <iostream>

template<BpData_t T, unsigned int DISP_VALS>
inline beliefprop::Status ProcessCUDABP<T, DISP_VALS>::errorCheck(const char *file, int line, bool abort) const {
  const auto code = cudaPeekAtLastError();
  if (code != cudaSuccess) {
    //fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    std::cout << "CUDA ERROR: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    cudaGetLastError();
    cudaDeviceReset();
    cudaDeviceSynchronize();
    cudaSetDevice(0);
    if (abort) { exit(code); }
    return beliefprop::Status::ERROR;
   }
   return beliefprop::Status::NO_ERROR;
}

/* May be needed if using half2
#if ((CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF) || (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO))

template<>
int ProcessCUDABP<half2>::getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize) {
      return (int)ceil(((ceil(((float)widthLevelActualIntegerSize) / 2.0)) / 2.0));
}

template<>
int ProcessCUDABP<half>::getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize) {
  ProcessCUDABP<half2> processCUDABPHalf;
  return processCUDABPHalf.getCheckerboardWidthTargetDevice(widthLevelActualIntegerSize) * 2;
}

#endif
*/

//functions directed related to running BP to retrieve the movement between the images

//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
//run the given number of iterations of BP at the current level using the given message values in global device memory
template<BpData_t T, unsigned int DISP_VALS>
beliefprop::Status ProcessCUDABP<T, DISP_VALS>::runBPAtCurrentLevel(
  const beliefprop::BPsettings& algSettings,
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
  const beliefprop::checkerboardMessages<T*>& messagesDevice,
  T* allocatedMemForProcessing)
{
  //cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  const dim3 threads(cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::BP_AT_LEVEL][currentLevelProperties.levelNum_][0],
                     cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::BP_AT_LEVEL][currentLevelProperties.levelNum_][1]);
  const dim3 grid{(unsigned int)ceil((float)(currentLevelProperties.widthCheckerboardLevel_) / (float)threads.x), //only updating half at a time
                  (unsigned int)ceil((float)currentLevelProperties.heightLevel_ / (float)threads.y)};

  //in cuda kernel storing data one at a time (though it is coalesced), so numDataInSIMDVector not relevant here and set to 1
  //still is a check if start of row is aligned
  const bool dataAligned{MemoryAlignedAtDataStart(0, 1, currentLevelProperties.numDataAlignWidth_, currentLevelProperties.divPaddedChBoardWAlign_)};

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
    runBPIterationUsingCheckerboardUpdates<T, DISP_VALS><<<grid, threads, maxbytes>>>(checkboardPartUpdate, currentLevelProperties,
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
      runBPIterationUsingCheckerboardUpdates<T, DISP_VALS><<<grid, threads>>>(checkboardPartUpdate, currentLevelProperties,
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
      runBPIterationUsingCheckerboardUpdates<T, DISP_VALS><<<grid, threads>>>(checkboardPartUpdate, currentLevelProperties,
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
    if (errorCheck(__FILE__, __LINE__) != beliefprop::Status::NO_ERROR) {
      return beliefprop::Status::ERROR;
    }
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  }
  return beliefprop::Status::NO_ERROR;
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<BpData_t T, unsigned int DISP_VALS>
beliefprop::Status ProcessCUDABP<T, DISP_VALS>::copyMessageValuesToNextLevelDown(
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::levelProperties& nextlevelProperties,
  const beliefprop::checkerboardMessages<T*>& messagesDeviceCopyFrom,
  const beliefprop::checkerboardMessages<T*>& messagesDeviceCopyTo,
  const unsigned int bpSettingsNumDispVals)
{
  const dim3 threads(cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::COPY_AT_LEVEL][currentLevelProperties.levelNum_][0],
                     cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::COPY_AT_LEVEL][currentLevelProperties.levelNum_][1]);
  const dim3 grid{(unsigned int)ceil((float)(currentLevelProperties.widthCheckerboardLevel_) / (float)threads.x),
                  (unsigned int)ceil((float)(currentLevelProperties.heightLevel_) / (float)threads.y)};

  cudaDeviceSynchronize();
  if (errorCheck(__FILE__, __LINE__) != beliefprop::Status::NO_ERROR) {
    return beliefprop::Status::ERROR;
  }

  for (const auto& checkerboard_part : {beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0, beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1})
  {
    //call the kernel to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
    //storing the current message values
    copyPrevLevelToNextLevelBPCheckerboardStereo<T, DISP_VALS> <<< grid, threads >>> (checkerboard_part, currentLevelProperties, nextlevelProperties,
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
    if (errorCheck(__FILE__, __LINE__) != beliefprop::Status::NO_ERROR) {
      return beliefprop::Status::ERROR;
    }
  }
  return beliefprop::Status::NO_ERROR;
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<BpData_t T, unsigned int DISP_VALS>
beliefprop::Status ProcessCUDABP<T, DISP_VALS>::initializeDataCosts(
  const beliefprop::BPsettings& algSettings,
  const beliefprop::levelProperties& currentLevelProperties,
  const std::array<float*, 2>& imagesOnTargetDevice,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard)
{
  if (errorCheck(__FILE__, __LINE__) != beliefprop::Status::NO_ERROR) {
    return beliefprop::Status::ERROR;
  }

  //since this is first kernel run in BP, set to prefer L1 cache for now since no shared memory is used by default
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  //setup execution parameters
  //the thread size remains constant throughout but the grid size is adjusted based on the current level/kernel to run
  const dim3 threads(cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][0][0],
                     cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][0][1]);
  //kernel run on full-sized image to retrieve data costs at the "bottom" level of the pyramid
  const dim3 grid{(unsigned int)ceil((float)currentLevelProperties.widthLevel_ / (float)threads.x),
                  (unsigned int)ceil((float)currentLevelProperties.heightLevel_ / (float)threads.y)};

  //initialize the data the the "bottom" of the image pyramid
  initializeBottomLevelDataStereo<T, DISP_VALS><<<grid, threads>>>(currentLevelProperties, imagesOnTargetDevice[0],
    imagesOnTargetDevice[1], dataCostDeviceCheckerboard.dataCostCheckerboard0_,
    dataCostDeviceCheckerboard.dataCostCheckerboard1_, algSettings.lambda_bp_, algSettings.data_k_bp_,
    algSettings.numDispVals_);
  cudaDeviceSynchronize();
  
  if (errorCheck(__FILE__, __LINE__) != beliefprop::Status::NO_ERROR) {
    return beliefprop::Status::ERROR;
  }

  return beliefprop::Status::NO_ERROR;
}

//initialize the message values with no previous message values...all message values are set to 0
template<BpData_t T, unsigned int DISP_VALS>
beliefprop::Status ProcessCUDABP<T, DISP_VALS>::initializeMessageValsToDefault(
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::checkerboardMessages<T*>& messagesDevice,
  const unsigned int bpSettingsNumDispVals)
{
  const dim3 threads(cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::INIT_MESSAGE_VALS][0][0],
                     cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::INIT_MESSAGE_VALS][0][1]);
  const dim3 grid{(unsigned int)ceil((float)currentLevelProperties.widthCheckerboardLevel_ / (float)threads.x),
                  (unsigned int)ceil((float)currentLevelProperties.heightLevel_ / (float)threads.y)};

  //initialize all the message values for each pixel at each possible movement to the default value in the kernel
  initializeMessageValsToDefaultKernel<T, DISP_VALS> <<< grid, threads >>> (currentLevelProperties,
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
  
  if (errorCheck(__FILE__, __LINE__) != beliefprop::Status::NO_ERROR) {
    return beliefprop::Status::ERROR;
  }

  return beliefprop::Status::NO_ERROR;
}

template<BpData_t T, unsigned int DISP_VALS>
beliefprop::Status ProcessCUDABP<T, DISP_VALS>::initializeDataCurrentLevel(const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::levelProperties& prevLevelProperties,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboardWriteTo,
  const unsigned int bpSettingsNumDispVals)
{
  const dim3 threads(cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][currentLevelProperties.levelNum_][0],
                     cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][currentLevelProperties.levelNum_][1]);
  //each pixel "checkerboard" is half the width of the level and there are two of them; each "pixel/point" at the level belongs to one checkerboard and
  //the four-connected neighbors are in the other checkerboard
  const dim3 grid{(unsigned int)ceil(((float)currentLevelProperties.widthCheckerboardLevel_) / (float)threads.x),
                  (unsigned int)ceil((float)currentLevelProperties.heightLevel_ / (float)threads.y)};

  if (errorCheck(__FILE__, __LINE__ ) != beliefprop::Status::NO_ERROR) {
    return beliefprop::Status::ERROR;
  }

  const size_t offsetNum{0};
  for (const auto& checkerboardAndDataCost : {
         std::make_pair(beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0, dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard0_),
         std::make_pair(beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1,  dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard1_)})
  {
    initializeCurrentLevelDataStereo<T, DISP_VALS> <<<grid, threads>>>(checkerboardAndDataCost.first,
      currentLevelProperties, prevLevelProperties,
      dataCostDeviceCheckerboard.dataCostCheckerboard0_,
      dataCostDeviceCheckerboard.dataCostCheckerboard1_,
      checkerboardAndDataCost.second, ((unsigned int) offsetNum / sizeof(float)),
      bpSettingsNumDispVals);

    cudaDeviceSynchronize();
    if (errorCheck(__FILE__, __LINE__ ) != beliefprop::Status::NO_ERROR) {
      return beliefprop::Status::ERROR;
    }
  }
  return beliefprop::Status::NO_ERROR;
}

#if (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO)

//due to the checkerboard indexing, half2 must be converted to half with the half function used for copying to the next level
template<>
beliefprop::Status ProcessCUDABP<half2, half2*>::copyMessageValuesToNextLevelDown(
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::levelProperties& nextlevelProperties,
  const beliefprop::checkerboardMessages<half2*>& messagesDeviceCopyFrom,
  const beliefprop::checkerboardMessages<half2*>& messagesDeviceCopyTo)
{
  /*ProcessCUDABP<half> processCUDABPHalf;
  processCUDABPHalf.copyMessageValuesToNextLevelDown(
      prevlevelProperties,
      currentLevelProperties,
      (half*)messagesDeviceCopyFrom.messagesU_Checkerboard0,
      (half*)messagesDeviceCopyFrom.messagesD_Checkerboard0,
      (half*)messagesDeviceCopyFrom.messagesL_Checkerboard0,
      (half*)messagesDeviceCopyFrom.messagesR_Checkerboard0,
      (half*)messagesDeviceCopyFrom.messagesU_Checkerboard1,
      (half*)messagesDeviceCopyFrom.messagesD_Checkerboard1,
      (half*)messagesDeviceCopyFrom.messagesL_Checkerboard1,
      (half*)messagesDeviceCopyFrom.messagesR_Checkerboard1,
      (half*)messagesDeviceCopyTo.messagesU_Checkerboard0,
      (half*)messagesDeviceCopyTo.messagesD_Checkerboard0,
      (half*)messagesDeviceCopyTo.messagesL_Checkerboard0,
      (half*)messagesDeviceCopyTo.messagesR_Checkerboard0,
      (half*)messagesDeviceCopyTo.messagesU_Checkerboard1,
      (half*)messagesDeviceCopyTo.messagesD_Checkerboard1,
      (half*)messagesDeviceCopyTo.messagesL_Checkerboard1,
      (half*)messagesDeviceCopyTo.messagesR_Checkerboard1);*/
  return beliefprop::Status::NO_ERROR;
}

//due to indexing, need to convert to half* and use half arrays for this function
template<>
beliefprop::Status ProcessCUDABP<half2, half2*>::initializeDataCurrentLevel(const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::levelProperties& prevLevelProperties,
  const beliefprop::dataCostData<half2*>& dataCostDeviceCheckerboard,
  const beliefprop::dataCostData<half2*>& dataCostDeviceCheckerboardWriteTo)
{
  /*ProcessCUDABP<half> processCUDABPHalf;
  processCUDABPHalf.initializeDataCurrentLevel(currentLevelProperties,
      prevLevelProperties,
      (half*)dataCostStereoCheckerboard1,
      (half*)dataCostStereoCheckerboard2,
      (half*)dataCostDeviceToWriteToCheckerboard1,
      (half*)dataCostDeviceToWriteToCheckerboard2);*/
  return beliefprop::Status::NO_ERROR;
}

#endif

template<BpData_t T, unsigned int DISP_VALS>
float* ProcessCUDABP<T, DISP_VALS>::retrieveOutputDisparity(
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
  const beliefprop::checkerboardMessages<T*>& messagesDevice,
  const unsigned int bpSettingsNumDispVals)
{
  float* resultingDisparityMapCompDevice;
  cudaMalloc((void**)&resultingDisparityMapCompDevice, currentLevelProperties.widthLevel_ * currentLevelProperties.heightLevel_ * sizeof(float));

  const dim3 threads(cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::OUTPUT_DISP][0][0],
                     cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::OUTPUT_DISP][0][1]);
  const dim3 grid{(unsigned int)ceil((float)currentLevelProperties.widthCheckerboardLevel_ / (float)threads.x),
                  (unsigned int)ceil((float)currentLevelProperties.heightLevel_ / (float)threads.y)};

  retrieveOutputDisparityCheckerboardStereoOptimized<T, DISP_VALS> <<<grid, threads>>>(currentLevelProperties,
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
  if (errorCheck(__FILE__, __LINE__) != beliefprop::Status::NO_ERROR) {
    return nullptr;
  }

  return resultingDisparityMapCompDevice;
}

template class ProcessCUDABP<float, 0>;
template class ProcessCUDABP<double, 0>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>;
template class ProcessCUDABP<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>;
template class ProcessCUDABP<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>;
//half precision only supported with compute capability 5.3 and higher
//TODO: not sure if using CUDA_ARCH works as intended here since it's host code
//may need to define whether or not to process half-precision elsewhere
#ifdef CUDA_HALF_SUPPORT
template class ProcessCUDABP<halftype, 0>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>;
template class ProcessCUDABP<halftype, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>;
#endif //CUDA_HALF_SUPPORT
//not currently supporting half2 data type
//template class ProcessCUDABP<half2>;
