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

//This file defines the template specialization to perform belief propagation using half precision for
//disparity map estimation from stereo images on CUDA

#include "BpConstsAndParams/BpStereoCudaParameters.h"

//only need template specialization for half precision if CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF is set
//in BpConstsAndParams/BpStereoCudaParameters.h
#ifdef CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF

//set constexpr unsigned int values for number of disparity values for each stereo set used
constexpr unsigned int DISP_VALS_0{bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_};
constexpr unsigned int DISP_VALS_1{bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_};
constexpr unsigned int DISP_VALS_2{bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_};
constexpr unsigned int DISP_VALS_3{bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_};
constexpr unsigned int DISP_VALS_4{bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_};
constexpr unsigned int DISP_VALS_5{bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_};
constexpr unsigned int DISP_VALS_6{bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_};

//device function to process messages using half precision with number of disparity values
//given in template parameter
template <unsigned int DISP_VALS>
__device__ inline void msgStereoHalf(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS],
  half messageValsNeighbor2[DISP_VALS], half messageValsNeighbor3[DISP_VALS],
  half dataCosts[DISP_VALS], half* dstMessageArray, half disc_k_bp, bool dataAligned)
{
  // aggregate and find min
  half minimum = bp_consts::INF_BP;
  half dst[DISP_VALS];

  for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
    dst[currentDisparity] = messageValsNeighbor1[currentDisparity] +
                            messageValsNeighbor2[currentDisparity] +
                            messageValsNeighbor3[currentDisparity] +
                            dataCosts[currentDisparity];
    if (dst[currentDisparity] < minimum) {
      minimum = dst[currentDisparity];
    }
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
  dtStereo<half, DISP_VALS>(dst);

  // truncate
  minimum += disc_k_bp;

  // normalize
  half valToNormalize = 0;

  for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
  {
    if (minimum < dst[currentDisparity]) {
      dst[currentDisparity] = minimum;
    }
    valToNormalize += dst[currentDisparity];
  }

  //if valToNormalize is infinite or NaN (observed when using more than 5 computation levels with half-precision),
  //set destination vector to 0 for all disparities
  //note that may cause results to differ a little from ideal
  if (__hisnan(valToNormalize) || ((__hisinf(valToNormalize)) != 0)) {
    unsigned int destMessageArrayIndex = beliefprop::retrieveIndexInDataAndMessage(xVal, yVal,
      currentLevelProperties.paddedWidthCheckerboardLevel_,
      currentLevelProperties.heightLevel_, 0,
      DISP_VALS);

    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      dstMessageArray[destMessageArrayIndex] = (half) 0.0;
      if constexpr (bp_params::OPTIMIZED_INDEXING_SETTING) {
        destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
      }
      else {
        destMessageArrayIndex++;
      }
    }
  }
  else
  {
    valToNormalize /= DISP_VALS;

    unsigned int destMessageArrayIndex = beliefprop::retrieveIndexInDataAndMessage(xVal, yVal,
      currentLevelProperties.paddedWidthCheckerboardLevel_,
      currentLevelProperties.heightLevel_, 0,
      DISP_VALS);

    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
    {
      dst[currentDisparity] -= valToNormalize;
      dstMessageArray[destMessageArrayIndex] = dst[currentDisparity];
      if constexpr (bp_params::OPTIMIZED_INDEXING_SETTING) {
        destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
      }
      else {
        destMessageArrayIndex++;
      }
    }
  }
}

//template BP message processing when number of disparity values is given
//as an input parameter and not as a template
template <beliefprop::MessageComp M>
__device__ inline void msgStereoHalf(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  half* prevUMessageArray, half* prevDMessageArray,
  half* prevLMessageArray, half* prevRMessageArray,
  half* dataMessageArray, half* dstMessageArray,
  half disc_k_bp, bool dataAligned, unsigned int bpSettingsDispVals,
  half* dstProcessing, unsigned int checkerboardAdjustment,
  unsigned int offsetData)
{
  // aggregate and find min
  half minimum{(half)bp_consts::INF_BP};
  unsigned int processingArrIndexDisp0 = beliefprop::retrieveIndexInDataAndMessage(xVal, yVal,
    currentLevelProperties.paddedWidthCheckerboardLevel_,
    currentLevelProperties.heightLevel_, 0,
    bpSettingsDispVals);
  unsigned int procArrIdx{processingArrIndexDisp0};

  for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
  {
    //set initial dst processing array value corresponding to disparity for M message type
    beliefprop::setInitDstProcessing<half, half, M>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray,
      prevLMessageArray, prevRMessageArray, dataMessageArray, dstMessageArray,
      disc_k_bp, dataAligned, bpSettingsDispVals, dstProcessing, checkerboardAdjustment,
      offsetData, currentDisparity, procArrIdx);

    if (dstProcessing[procArrIdx] < minimum)
      minimum = dstProcessing[procArrIdx];

    if constexpr (bp_params::OPTIMIZED_INDEXING_SETTING) {
      procArrIdx += currentLevelProperties.paddedWidthCheckerboardLevel_;
    }
    else {
      procArrIdx++;
    }
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
  dtStereo<half>(dstProcessing, bpSettingsDispVals, xVal, yVal, currentLevelProperties);

  // truncate
  minimum += disc_k_bp;

  // normalize
  half valToNormalize{(half)0.0};

  procArrIdx = processingArrIndexDisp0;
  for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
    if (minimum < dstProcessing[procArrIdx]) {
      dstProcessing[procArrIdx] = minimum;
    }

    valToNormalize += dstProcessing[procArrIdx];

    if constexpr (bp_params::OPTIMIZED_INDEXING_SETTING) {
      procArrIdx += currentLevelProperties.paddedWidthCheckerboardLevel_;
    }
    else {
      procArrIdx++;
    }
  }

  //if valToNormalize is infinite or NaN (observed when using more than 5 computation levels with half-precision),
  //set destination vector to 0 for all disparities
  //note that may cause results to differ a little from ideal
  if (__hisnan(valToNormalize) || ((__hisinf(valToNormalize)) != 0)) {
    //dst processing index and message array index are the same for each disparity value in this processing
    procArrIdx = processingArrIndexDisp0;

    for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
      dstMessageArray[procArrIdx] = (half)0.0;
      if constexpr (bp_params::OPTIMIZED_INDEXING_SETTING) {
        procArrIdx += currentLevelProperties.paddedWidthCheckerboardLevel_;
      }
      else {
        procArrIdx++;
      }
    }
  }
  else
  {
    valToNormalize /= ((half)bpSettingsDispVals);

    //dst processing index and message array index are the same for each disparity value in this processing
    procArrIdx = processingArrIndexDisp0;

    for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
      dstProcessing[procArrIdx] -= valToNormalize;
      dstMessageArray[procArrIdx] = convertValToDifferentDataTypeIfNeeded<half, half>(dstProcessing[procArrIdx]);
      if constexpr (bp_params::OPTIMIZED_INDEXING_SETTING) {
        procArrIdx += currentLevelProperties.paddedWidthCheckerboardLevel_;
      }
      else {
        procArrIdx++;
      }
    }
  }
}

template<>
__device__ inline void msgStereo<half, half, beliefprop::MessageComp::kUMessage>(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  half* prevUMessageArray, half* prevDMessageArray,
  half* prevLMessageArray, half* prevRMessageArray,
  half* dataMessageArray, half* dstMessageArray,
  half disc_k_bp, bool dataAligned, unsigned int bpSettingsDispVals,
  half* dstProcessing, unsigned int checkerboardAdjustment,
  unsigned int offsetData)
{
  msgStereoHalf<beliefprop::MessageComp::kUMessage>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray,
    prevLMessageArray, prevRMessageArray, dataMessageArray, dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals,
    dstProcessing, checkerboardAdjustment, offsetData);
}

template<>
__device__ inline void msgStereo<half, half, beliefprop::MessageComp::kDMessage>(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  half* prevUMessageArray, half* prevDMessageArray,
  half* prevLMessageArray, half* prevRMessageArray,
  half* dataMessageArray, half* dstMessageArray,
  half disc_k_bp, bool dataAligned, unsigned int bpSettingsDispVals,
  half* dstProcessing, unsigned int checkerboardAdjustment,
  unsigned int offsetData)
{
  msgStereoHalf<beliefprop::MessageComp::kDMessage>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray,
    prevLMessageArray, prevRMessageArray, dataMessageArray, dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals,
    dstProcessing, checkerboardAdjustment, offsetData);
}

template<>
__device__ inline void msgStereo<half, half, beliefprop::MessageComp::kLMessage>(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  half* prevUMessageArray, half* prevDMessageArray,
  half* prevLMessageArray, half* prevRMessageArray,
  half* dataMessageArray, half* dstMessageArray,
  half disc_k_bp, bool dataAligned, unsigned int bpSettingsDispVals,
  half* dstProcessing, unsigned int checkerboardAdjustment,
  unsigned int offsetData)
{
  msgStereoHalf<beliefprop::MessageComp::kLMessage>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray,
    prevLMessageArray, prevRMessageArray, dataMessageArray, dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals,
    dstProcessing, checkerboardAdjustment, offsetData);
}

template<>
__device__ inline void msgStereo<half, half, beliefprop::MessageComp::kRMessage>(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  half* prevUMessageArray, half* prevDMessageArray,
  half* prevLMessageArray, half* prevRMessageArray,
  half* dataMessageArray, half* dstMessageArray,
  half disc_k_bp, bool dataAligned, unsigned int bpSettingsDispVals,
  half* dstProcessing, unsigned int checkerboardAdjustment,
  unsigned int offsetData)
{
  msgStereoHalf<beliefprop::MessageComp::kRMessage>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray,
    prevLMessageArray, prevRMessageArray, dataMessageArray, dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals,
    dstProcessing, checkerboardAdjustment, offsetData);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_0>(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_0],
  half messageValsNeighbor2[DISP_VALS_0], half messageValsNeighbor3[DISP_VALS_0],
  half dataCosts[DISP_VALS_0], half* dstMessageArray, half disc_k_bp, bool dataAligned)
{
  msgStereoHalf<DISP_VALS_0>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_1>(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_1],
  half messageValsNeighbor2[DISP_VALS_1], half messageValsNeighbor3[DISP_VALS_1],
  half dataCosts[DISP_VALS_1], half* dstMessageArray, half disc_k_bp, bool dataAligned)
{
  msgStereoHalf<DISP_VALS_1>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_2>(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_2],
  half messageValsNeighbor2[DISP_VALS_2], half messageValsNeighbor3[DISP_VALS_2],
  half dataCosts[DISP_VALS_2], half* dstMessageArray, half disc_k_bp, bool dataAligned)
{
  msgStereoHalf<DISP_VALS_2>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_3>(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_3],
  half messageValsNeighbor2[DISP_VALS_3], half messageValsNeighbor3[DISP_VALS_3],
  half dataCosts[DISP_VALS_3], half* dstMessageArray, half disc_k_bp, bool dataAligned)
{
  msgStereoHalf<DISP_VALS_3>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_4>(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_4],
  half messageValsNeighbor2[DISP_VALS_4], half messageValsNeighbor3[DISP_VALS_4],
  half dataCosts[DISP_VALS_4], half* dstMessageArray, half disc_k_bp, bool dataAligned)
{
  msgStereoHalf<DISP_VALS_4>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_5>(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_5],
  half messageValsNeighbor2[DISP_VALS_5], half messageValsNeighbor3[DISP_VALS_5],
  half dataCosts[DISP_VALS_5], half* dstMessageArray, half disc_k_bp, bool dataAligned)
{
  msgStereoHalf<DISP_VALS_5>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_6>(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_6],
  half messageValsNeighbor2[DISP_VALS_6], half messageValsNeighbor3[DISP_VALS_6],
  half dataCosts[DISP_VALS_6], half* dstMessageArray, half disc_k_bp, bool dataAligned)
{
  msgStereoHalf<DISP_VALS_6>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

#endif //CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF
