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

#include "BpConstsAndParams/bpStereoCudaParameters.h"

//only need template specialization for half precision if CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF is set
//in BpConstsAndParams/bpStereoCudaParameters.h
#ifdef CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF

//set constexpr unsigned int values for number of disparity values for each stereo set used
constexpr unsigned int DISP_VALS_0{bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]};
constexpr unsigned int DISP_VALS_1{bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]};
constexpr unsigned int DISP_VALS_2{bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]};
constexpr unsigned int DISP_VALS_3{bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]};
constexpr unsigned int DISP_VALS_4{bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]};
constexpr unsigned int DISP_VALS_5{bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]};
constexpr unsigned int DISP_VALS_6{bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]};

//device function to process messages using half precision with number of disparity values
//given in template parameter
template <unsigned int DISP_VALS>
__device__ inline void msgStereoHalf(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS],
  half messageValsNeighbor2[DISP_VALS], half messageValsNeighbor3[DISP_VALS],
  half dataCosts[DISP_VALS], half* dstMessageArray, const half disc_k_bp, const bool dataAligned)
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
    unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
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

    unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
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
template <beliefprop::messageComp M>
__device__ inline void msgStereoHalf(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  half* prevUMessageArray, half* prevDMessageArray,
  half* prevLMessageArray, half* prevRMessageArray,
  half* dataMessageArray, half* dstMessageArray,
  half disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals,
  half* dstProcessing, const unsigned int checkerboardAdjustment,
  const unsigned int offsetData)
{
  // aggregate and find min
  half minimum{(half)bp_consts::INF_BP};
  unsigned int processingArrIndexDisp0 = retrieveIndexInDataAndMessage(xVal, yVal,
    currentLevelProperties.paddedWidthCheckerboardLevel_,
    currentLevelProperties.heightLevel_, 0,
    bpSettingsDispVals);
  unsigned int procArrIdx{processingArrIndexDisp0};

  for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
  {
    //set initial dst processing array value corresponding to disparity for M message type
    setInitDstProcessing<half, half, M>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray,
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
__device__ inline void msgStereo<half, half, beliefprop::messageComp::U_MESSAGE>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  half* prevUMessageArray, half* prevDMessageArray,
  half* prevLMessageArray, half* prevRMessageArray,
  half* dataMessageArray, half* dstMessageArray,
  half disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals,
  half* dstProcessing, const unsigned int checkerboardAdjustment,
  const unsigned int offsetData)
{
  msgStereoHalf<beliefprop::messageComp::U_MESSAGE>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray,
    prevLMessageArray, prevRMessageArray, dataMessageArray, dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals,
    dstProcessing, checkerboardAdjustment, offsetData);
}

template<>
__device__ inline void msgStereo<half, half, beliefprop::messageComp::D_MESSAGE>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  half* prevUMessageArray, half* prevDMessageArray,
  half* prevLMessageArray, half* prevRMessageArray,
  half* dataMessageArray, half* dstMessageArray,
  half disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals,
  half* dstProcessing, const unsigned int checkerboardAdjustment,
  const unsigned int offsetData)
{
  msgStereoHalf<beliefprop::messageComp::D_MESSAGE>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray,
    prevLMessageArray, prevRMessageArray, dataMessageArray, dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals,
    dstProcessing, checkerboardAdjustment, offsetData);
}

template<>
__device__ inline void msgStereo<half, half, beliefprop::messageComp::L_MESSAGE>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  half* prevUMessageArray, half* prevDMessageArray,
  half* prevLMessageArray, half* prevRMessageArray,
  half* dataMessageArray, half* dstMessageArray,
  half disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals,
  half* dstProcessing, const unsigned int checkerboardAdjustment,
  const unsigned int offsetData)
{
  msgStereoHalf<beliefprop::messageComp::L_MESSAGE>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray,
    prevLMessageArray, prevRMessageArray, dataMessageArray, dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals,
    dstProcessing, checkerboardAdjustment, offsetData);
}

template<>
__device__ inline void msgStereo<half, half, beliefprop::messageComp::R_MESSAGE>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  half* prevUMessageArray, half* prevDMessageArray,
  half* prevLMessageArray, half* prevRMessageArray,
  half* dataMessageArray, half* dstMessageArray,
  half disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals,
  half* dstProcessing, const unsigned int checkerboardAdjustment,
  const unsigned int offsetData)
{
  msgStereoHalf<beliefprop::messageComp::R_MESSAGE>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray,
    prevLMessageArray, prevRMessageArray, dataMessageArray, dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals,
    dstProcessing, checkerboardAdjustment, offsetData);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_0>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_0],
  half messageValsNeighbor2[DISP_VALS_0], half messageValsNeighbor3[DISP_VALS_0],
  half dataCosts[DISP_VALS_0], half* dstMessageArray, const half disc_k_bp, const bool dataAligned)
{
  msgStereoHalf<DISP_VALS_0>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_1>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_1],
  half messageValsNeighbor2[DISP_VALS_1], half messageValsNeighbor3[DISP_VALS_1],
  half dataCosts[DISP_VALS_1], half* dstMessageArray, const half disc_k_bp, const bool dataAligned)
{
  msgStereoHalf<DISP_VALS_1>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_2>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_2],
  half messageValsNeighbor2[DISP_VALS_2], half messageValsNeighbor3[DISP_VALS_2],
  half dataCosts[DISP_VALS_2], half* dstMessageArray, const half disc_k_bp, const bool dataAligned)
{
  msgStereoHalf<DISP_VALS_2>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_3>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_3],
  half messageValsNeighbor2[DISP_VALS_3], half messageValsNeighbor3[DISP_VALS_3],
  half dataCosts[DISP_VALS_3], half* dstMessageArray, const half disc_k_bp, const bool dataAligned)
{
  msgStereoHalf<DISP_VALS_3>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_4>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_4],
  half messageValsNeighbor2[DISP_VALS_4], half messageValsNeighbor3[DISP_VALS_4],
  half dataCosts[DISP_VALS_4], half* dstMessageArray, const half disc_k_bp, const bool dataAligned)
{
  msgStereoHalf<DISP_VALS_4>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_5>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_5],
  half messageValsNeighbor2[DISP_VALS_5], half messageValsNeighbor3[DISP_VALS_5],
  half dataCosts[DISP_VALS_5], half* dstMessageArray, const half disc_k_bp, const bool dataAligned)
{
  msgStereoHalf<DISP_VALS_5>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<>
__device__ inline void msgStereo<half, half, DISP_VALS_6>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, half messageValsNeighbor1[DISP_VALS_6],
  half messageValsNeighbor2[DISP_VALS_6], half messageValsNeighbor3[DISP_VALS_6],
  half dataCosts[DISP_VALS_6], half* dstMessageArray, const half disc_k_bp, const bool dataAligned)
{
  msgStereoHalf<DISP_VALS_6>(xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3,
    dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

#endif //CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF
