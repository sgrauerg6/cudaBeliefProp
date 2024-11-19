/*
 * SharedBPProcessingFuncts.h
 *
 *  Created on: Jun 24, 2019
 *      Author: scott
 */

#ifndef SHAREDBPPROCESSINGFUNCTS_H_
#define SHAREDBPPROCESSINGFUNCTS_H_

#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"
#include "BpConstsAndParams/BpTypeConstraints.h"
#include "BpConstsAndParams/BpStereoParameters.h"
#include "BpConstsAndParams/BpStructsAndEnums.h"
#include "RunImp/RunImpGenFuncts.h"

namespace beliefprop {

//retrieve the current 1-D index value of the given point at the given disparity in the data cost and message data
ARCHITECTURE_ADDITION inline unsigned int retrieveIndexInDataAndMessage(unsigned int xVal, unsigned int yVal,
  unsigned int width, unsigned int height, unsigned int currentDisparity, unsigned int totalNumDispVals,
  unsigned int offsetData = 0u)
{
  if constexpr (bp_params::kOptimizedIndexingSetting) {
    //indexing is performed in such a way so that the memory accesses as coalesced as much as possible
    return (yVal * width * totalNumDispVals + width * currentDisparity + xVal) + offsetData;
  }
  else {
    return ((yVal * width + xVal) * totalNumDispVals + currentDisparity);
  }
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void dtStereo(T f[DISP_VALS])
{
  T prev;
  for (unsigned int currentDisparity = 1; currentDisparity < DISP_VALS; currentDisparity++)
  {
    prev = f[currentDisparity-1] + (T)1.0;
    if (prev < f[currentDisparity])
      f[currentDisparity] = prev;
  }

  for (int currentDisparity = (int)DISP_VALS - 2; currentDisparity >= 0; currentDisparity--)
  {
    prev = f[currentDisparity + 1] + (T)1.0;
    if (prev < f[currentDisparity])
      f[currentDisparity] = prev;
  }
}

template<RunData_t T>
ARCHITECTURE_ADDITION inline void dtStereo(T* f, unsigned int bp_settings_disp_vals)
{
  T prev;
  for (unsigned int currentDisparity = 1; currentDisparity < bp_settings_disp_vals; currentDisparity++)
  {
    prev = f[currentDisparity-1] + (T)1.0;
    if (prev < f[currentDisparity])
      f[currentDisparity] = prev;
  }

  for (int currentDisparity = (int)bp_settings_disp_vals - 2; currentDisparity >= 0; currentDisparity--)
  {
    prev = f[currentDisparity + 1] + (T)1.0;
    if (prev < f[currentDisparity])
      f[currentDisparity] = prev;
  }
}

template<RunData_t T>
ARCHITECTURE_ADDITION inline void dtStereo(T* f, unsigned int bp_settings_disp_vals,
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel)
{
  unsigned int fArrayIndexDisp0 = retrieveIndexInDataAndMessage(xVal, yVal,
    currentBpLevel.padded_width_checkerboard_level_,
    currentBpLevel.height_level_, 0,
    bp_settings_disp_vals);
  unsigned int currFArrayIndexLast = fArrayIndexDisp0;
  unsigned int currFArrayIndex = fArrayIndexDisp0;
  T prev;

  for (unsigned int currentDisparity = 1; currentDisparity < bp_settings_disp_vals; currentDisparity++)
  {
    if constexpr (bp_params::kOptimizedIndexingSetting) {
      currFArrayIndex += currentBpLevel.padded_width_checkerboard_level_;
    }
    else {
      currFArrayIndex++;
    }

    prev = f[currFArrayIndexLast] + (T)1.0;
    if (prev < f[currFArrayIndex])
      f[currFArrayIndex] = prev;
    currFArrayIndexLast = currFArrayIndex;
  }

  for (int currentDisparity = (int)bp_settings_disp_vals - 2; currentDisparity >= 0; currentDisparity--)
  {
    if constexpr (bp_params::kOptimizedIndexingSetting) {
      currFArrayIndex -= currentBpLevel.padded_width_checkerboard_level_;
    }
    else {
      currFArrayIndex--;
    }

    prev = f[currFArrayIndexLast] + (T)1.0;
    if (prev < f[currFArrayIndex])
      f[currFArrayIndex] = prev;
    currFArrayIndexLast = currFArrayIndex;
  }
}

template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void msgStereo(unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  U messageValsNeighbor1[DISP_VALS], U messageValsNeighbor2[DISP_VALS],
  U messageValsNeighbor3[DISP_VALS], U dataCosts[DISP_VALS],
  T* dstMessageArray, U disc_k_bp, bool dataAligned)
{
  // aggregate and find min
  U minimum{(U)bp_consts::kInfBp};
  U dst[DISP_VALS];

  for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
  {
    dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
    if (dst[currentDisparity] < minimum)
      minimum = dst[currentDisparity];
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
  dtStereo<U, DISP_VALS>(dst);

  // truncate
  minimum += disc_k_bp;

  // normalize
  U valToNormalize{(U)0.0};

  for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
    if (minimum < dst[currentDisparity]) {
      dst[currentDisparity] = minimum;
    }

    valToNormalize += dst[currentDisparity];
  }

  valToNormalize /= ((U)DISP_VALS);

  int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
    currentBpLevel.padded_width_checkerboard_level_,
    currentBpLevel.height_level_, 0,
    DISP_VALS);

  for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
    dst[currentDisparity] -= valToNormalize;
    dstMessageArray[destMessageArrayIndex] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(dst[currentDisparity]);
    if constexpr (bp_params::kOptimizedIndexingSetting) {
      destMessageArrayIndex += currentBpLevel.padded_width_checkerboard_level_;
    }
    else {
      destMessageArrayIndex++;
    }
  }
}

template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline void msgStereo(unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  U* messageValsNeighbor1, U* messageValsNeighbor2,
  U* messageValsNeighbor3, U* dataCosts,
  T* dstMessageArray, U disc_k_bp, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  // aggregate and find min
  U minimum{(U)bp_consts::kInfBp};
  U* dst = new U[bp_settings_disp_vals];

  for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++)
  {
    dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
    if (dst[currentDisparity] < minimum)
      minimum = dst[currentDisparity];
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
  dtStereo<U>(dst, bp_settings_disp_vals);

  // truncate
  minimum += disc_k_bp;

  // normalize
  U valToNormalize{(U)0.0};

  for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
    if (minimum < dst[currentDisparity]) {
      dst[currentDisparity] = minimum;
    }

    valToNormalize += dst[currentDisparity];
  }

  valToNormalize /= ((U)bp_settings_disp_vals);

  int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
    currentBpLevel.padded_width_checkerboard_level_,
    currentBpLevel.height_level_, 0,
    bp_settings_disp_vals);

  for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
    dst[currentDisparity] -= valToNormalize;
    dstMessageArray[destMessageArrayIndex] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(dst[currentDisparity]);
    if constexpr (bp_params::kOptimizedIndexingSetting) {
      destMessageArrayIndex += currentBpLevel.padded_width_checkerboard_level_;
    }
    else {
      destMessageArrayIndex++;
    }
  }

  delete [] dst;
}

template<RunData_t T, RunData_t U, beliefprop::MessageComp M>
ARCHITECTURE_ADDITION void inline setInitDstProcessing(unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel,
  T* prevUMessageArray, T* prevDMessageArray,
  T* prevLMessageArray, T* prevRMessageArray,
  T* dataMessageArray, T* dstMessageArray,
  U disc_k_bp, bool dataAligned, unsigned int bp_settings_disp_vals,
  U* dstProcessing, unsigned int checkerboardAdjustment,
  unsigned int offsetData, unsigned int currentDisparity,
  unsigned int procArrIdx)
{
  const U dataVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataMessageArray[retrieveIndexInDataAndMessage(xVal, yVal,
    currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
    currentDisparity, bp_settings_disp_vals, offsetData)]);

  if constexpr (M == beliefprop::MessageComp::kUMessage) {
    const U prevUVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prevUMessageArray[retrieveIndexInDataAndMessage(xVal, (yVal+1),
      currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
      currentDisparity, bp_settings_disp_vals)]);
    const U prevLVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prevLMessageArray[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
      currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
      currentDisparity, bp_settings_disp_vals)]);
    const U prevRVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prevRMessageArray[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
      currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
      currentDisparity, bp_settings_disp_vals)]);

    dstProcessing[procArrIdx] = prevUVal + prevLVal + prevRVal + dataVal;
  }
  else if constexpr (M == beliefprop::MessageComp::kDMessage) {
    const U prevDVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prevDMessageArray[retrieveIndexInDataAndMessage(xVal, (yVal-1),
      currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
      currentDisparity, bp_settings_disp_vals)]);
    const U prevLVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prevLMessageArray[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
      currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
      currentDisparity, bp_settings_disp_vals)]);
    const U prevRVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prevRMessageArray[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
      currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
      currentDisparity, bp_settings_disp_vals)]);

    dstProcessing[procArrIdx] = prevDVal + prevLVal + prevRVal + dataVal;
  }
  else if constexpr (M == beliefprop::MessageComp::kLMessage) {
    const U prevUVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prevUMessageArray[retrieveIndexInDataAndMessage(xVal, (yVal+1),
      currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
      currentDisparity, bp_settings_disp_vals)]);
    const U prevDVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prevDMessageArray[retrieveIndexInDataAndMessage(xVal, (yVal-1),
      currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
      currentDisparity, bp_settings_disp_vals)]);
    const U prevLVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prevLMessageArray[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
      currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
      currentDisparity, bp_settings_disp_vals)]);

    dstProcessing[procArrIdx] = prevUVal + prevDVal + prevLVal + dataVal;
  }
  else if constexpr (M == beliefprop::MessageComp::kRMessage) {
    const U prevUVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prevUMessageArray[retrieveIndexInDataAndMessage(xVal, (yVal+1),
      currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
      currentDisparity, bp_settings_disp_vals)]);
    const U prevDVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prevDMessageArray[retrieveIndexInDataAndMessage(xVal, (yVal-1),
      currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
      currentDisparity, bp_settings_disp_vals)]);
    const U prevRVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prevRMessageArray[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
      currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
      currentDisparity, bp_settings_disp_vals)]);

    dstProcessing[procArrIdx] = prevUVal + prevDVal + prevRVal + dataVal;
  }
}

//TODO: may need to specialize for half-precision to account for possible NaN/inf vals
template<RunData_t T, RunData_t U, beliefprop::MessageComp M>
ARCHITECTURE_ADDITION inline void msgStereo(unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel,
  T* prevUMessageArray, T* prevDMessageArray,
  T* prevLMessageArray, T* prevRMessageArray,
  T* dataMessageArray, T* dstMessageArray,
  U disc_k_bp, bool dataAligned, unsigned int bp_settings_disp_vals,
  U* dstProcessing, unsigned int checkerboardAdjustment,
  unsigned int offsetData)
{
  // aggregate and find min
  U minimum{(U)bp_consts::kInfBp};
  unsigned int processingArrIndexDisp0 = retrieveIndexInDataAndMessage(xVal, yVal,
    currentBpLevel.padded_width_checkerboard_level_,
    currentBpLevel.height_level_, 0,
    bp_settings_disp_vals);
  unsigned int procArrIdx{processingArrIndexDisp0};

  for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++)
  {
    //set initial dst processing array value corresponding to disparity for M message type
    setInitDstProcessing<T, U, M>(xVal, yVal, currentBpLevel, prevUMessageArray, prevDMessageArray,
      prevLMessageArray, prevRMessageArray, dataMessageArray, dstMessageArray,
      disc_k_bp, dataAligned, bp_settings_disp_vals, dstProcessing, checkerboardAdjustment,
      offsetData, currentDisparity, procArrIdx);

    if (dstProcessing[procArrIdx] < minimum)
      minimum = dstProcessing[procArrIdx];

    if constexpr (bp_params::kOptimizedIndexingSetting) {
      procArrIdx += currentBpLevel.padded_width_checkerboard_level_;
    }
    else {
      procArrIdx++;
    }
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
  dtStereo<U>(dstProcessing, bp_settings_disp_vals, xVal, yVal, currentBpLevel);

  // truncate
  minimum += disc_k_bp;

  // normalize
  U valToNormalize{(U)0.0};

  procArrIdx = processingArrIndexDisp0;
  for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
    if (minimum < dstProcessing[procArrIdx]) {
      dstProcessing[procArrIdx] = minimum;
    }

    valToNormalize += dstProcessing[procArrIdx];

    if constexpr (bp_params::kOptimizedIndexingSetting) {
      procArrIdx += currentBpLevel.padded_width_checkerboard_level_;
    }
    else {
      procArrIdx++;
    }
  }

  valToNormalize /= ((U)bp_settings_disp_vals);

  //dst processing index and message array index are the same for each disparity value in this processing
  procArrIdx = processingArrIndexDisp0;

  for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
    dstProcessing[procArrIdx] -= valToNormalize;
    dstMessageArray[procArrIdx] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(dstProcessing[procArrIdx]);
    if constexpr (bp_params::kOptimizedIndexingSetting) {
      procArrIdx += currentBpLevel.padded_width_checkerboard_level_;
    }
    else {
      procArrIdx++;
    }
  }
}

//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void initializeBottomLevelDataPixel(unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel, float* image1PixelsDevice,
  float* image2PixelsDevice, T* dataCostDeviceStereoCheckerboard0,
  T* dataCostDeviceStereoCheckerboard1, float lambda_bp,
  float data_k_bp, unsigned int bp_settings_disp_vals)
{
  if constexpr (DISP_VALS > 0) {
    unsigned int indexVal;
    const unsigned int xInCheckerboard = xVal / 2;

    if (run_imp_util::WithinImageBounds(xInCheckerboard, yVal, currentBpLevel.width_checkerboard_level_, currentBpLevel.height_level_)) {
      //make sure that it is possible to check every disparity value
      //need to cast DISP_VALS from unsigned int to int
      //for conditional to work as expected
      if (((int)xVal - ((int)DISP_VALS - 1)) >= 0) {
        for (unsigned int currentDisparity = 0u; currentDisparity < DISP_VALS; currentDisparity++) {
          float currentPixelImage1 = 0.0f, currentPixelImage2 = 0.0f;

          if (run_imp_util::WithinImageBounds(xVal, yVal, currentBpLevel.width_level_, currentBpLevel.height_level_)) {
            currentPixelImage1 = image1PixelsDevice[yVal * currentBpLevel.width_level_ + xVal];
            currentPixelImage2 = image2PixelsDevice[yVal * currentBpLevel.width_level_ + (xVal - currentDisparity)];
          }

          indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal,
            currentBpLevel.padded_width_checkerboard_level_,
            currentBpLevel.height_level_, currentDisparity,
            DISP_VALS);

          //data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
          if (((xVal + yVal) % 2) == 0) {
            dataCostDeviceStereoCheckerboard0[indexVal] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * run_imp_util::GetMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
          }
          else {
            dataCostDeviceStereoCheckerboard1[indexVal] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * run_imp_util::GetMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
          }
        }
      } else {
        for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
          indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal,
            currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
            currentDisparity, DISP_VALS);

          //data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
          if (((xVal + yVal) % 2) == 0) {
            dataCostDeviceStereoCheckerboard0[indexVal] = run_imp_util::ZeroVal<T>();
          }
          else {
            dataCostDeviceStereoCheckerboard1[indexVal] = run_imp_util::ZeroVal<T>();
          }
        }
      }
    }
  }
  else {
    unsigned int indexVal;
    const unsigned int xInCheckerboard = xVal / 2;

    if (run_imp_util::WithinImageBounds(xInCheckerboard, yVal, currentBpLevel.width_checkerboard_level_, currentBpLevel.height_level_)) {
      //make sure that it is possible to check every disparity value
      //need to cast bp_settings_disp_vals from unsigned int to int
      //for conditional to work as expected
      if (((int)xVal - ((int)bp_settings_disp_vals - 1)) >= 0) {
        for (unsigned int currentDisparity = 0u; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
          float currentPixelImage1 = 0.0f, currentPixelImage2 = 0.0f;

          if (run_imp_util::WithinImageBounds(xVal, yVal, currentBpLevel.width_level_, currentBpLevel.height_level_)) {
            currentPixelImage1 = image1PixelsDevice[yVal * currentBpLevel.width_level_ + xVal];
            currentPixelImage2 = image2PixelsDevice[yVal * currentBpLevel.width_level_ + (xVal - currentDisparity)];
          }

          indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal,
            currentBpLevel.padded_width_checkerboard_level_,
            currentBpLevel.height_level_, currentDisparity,
            bp_settings_disp_vals);

          //data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
          if (((xVal + yVal) % 2) == 0) {
            dataCostDeviceStereoCheckerboard0[indexVal] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * run_imp_util::GetMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
          }
          else {
            dataCostDeviceStereoCheckerboard1[indexVal] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * run_imp_util::GetMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
          }
        }
      } else {
        for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
          indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal,
            currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
            currentDisparity, bp_settings_disp_vals);

          //data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
          if (((xVal + yVal) % 2) == 0) {
            dataCostDeviceStereoCheckerboard0[indexVal] = run_imp_util::ZeroVal<T>();
          }
          else {
            dataCostDeviceStereoCheckerboard1[indexVal] = run_imp_util::ZeroVal<T>();
          }
        }
      }
    }
  }
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void initializeCurrentLevelDataPixel(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* dataCostDeviceToWriteTo, unsigned int offsetNum,
  unsigned int bp_settings_disp_vals)
{
  //add 1 or 0 to the x-value depending on checkerboard part and row adding to; beliefprop::Checkerboard_Part::kCheckerboardPart0 with slot at (0, 0) has adjustment of 0 in row 0,
  //while beliefprop::Checkerboard_Part::kCheckerboardPart1 with slot at (0, 1) has adjustment of 1 in row 0
  const unsigned int checkerboardPartAdjustment = (checkerboardPart == beliefprop::Checkerboard_Part::kCheckerboardPart0) ? (yVal % 2) : ((yVal + 1) % 2);

  //the corresponding x-values at the "lower" level depends on which checkerboard the pixel is in
  const unsigned int xValPrev = xVal*2 + checkerboardPartAdjustment;

  if (run_imp_util::WithinImageBounds(xValPrev, (yVal * 2 + 1), prevBpLevel.width_checkerboard_level_, prevBpLevel.height_level_)) {
    if constexpr (DISP_VALS > 0) {
      for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
        const U dataCostVal =
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2, prevBpLevel.padded_width_checkerboard_level_, prevBpLevel.height_level_,
            currentDisparity, DISP_VALS, offsetNum)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2, prevBpLevel.padded_width_checkerboard_level_, prevBpLevel.height_level_,
            currentDisparity, DISP_VALS, offsetNum)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2 + 1, prevBpLevel.padded_width_checkerboard_level_, prevBpLevel.height_level_,
            currentDisparity, DISP_VALS, offsetNum)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2 + 1, prevBpLevel.padded_width_checkerboard_level_, prevBpLevel.height_level_,
            currentDisparity, DISP_VALS, offsetNum)]);

        dataCostDeviceToWriteTo[retrieveIndexInDataAndMessage(xVal, yVal,
          currentBpLevel.padded_width_checkerboard_level_,
          currentBpLevel.height_level_, currentDisparity,
          DISP_VALS)] =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(dataCostVal);
      }
    }
    else {
      for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
        const U dataCostVal =
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2, prevBpLevel.padded_width_checkerboard_level_, prevBpLevel.height_level_,
            currentDisparity, bp_settings_disp_vals, offsetNum)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2, prevBpLevel.padded_width_checkerboard_level_, prevBpLevel.height_level_,
            currentDisparity, bp_settings_disp_vals, offsetNum)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2 + 1, prevBpLevel.padded_width_checkerboard_level_, prevBpLevel.height_level_,
            currentDisparity, bp_settings_disp_vals, offsetNum)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2 + 1, prevBpLevel.padded_width_checkerboard_level_, prevBpLevel.height_level_,
            currentDisparity, bp_settings_disp_vals, offsetNum)]);

        dataCostDeviceToWriteTo[retrieveIndexInDataAndMessage(xVal, yVal,
          currentBpLevel.padded_width_checkerboard_level_,
          currentBpLevel.height_level_, currentDisparity,
          bp_settings_disp_vals)] =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(dataCostVal);
      }
    }
  }
}

//initialize the message values at each pixel of the current level to the default value
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void initializeMessageValsToDefaultKernelPixel(
  unsigned int xValInCheckerboard, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int bp_settings_disp_vals)
{
  //initialize message values in both checkerboards

  if constexpr (DISP_VALS > 0) {
    //set the message value at each pixel for each disparity to 0
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
    }

    //retrieve the previous message value at each movement at each pixel
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
    }
  }
  else {
    //set the message value at each pixel for each disparity to 0
    for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
      messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
    }

    //retrieve the previous message value at each movement at each pixel
    for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
      messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
    }
  }
}

//device portion of the kernel function to run the current iteration of belief propagation where the input messages and data costs come in as array in local memory
//and the output message values are stored in local memory
template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void runBPIterationUpdateMsgVals(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  U prevUMessage[DISP_VALS], U prevDMessage[DISP_VALS],
  U prevLMessage[DISP_VALS], U prevRMessage[DISP_VALS],
  U dataMessage[DISP_VALS],
  T* currentUMessageArray, T* currentDMessageArray,
  T* currentLMessageArray, T* currentRMessageArray,
  const U disc_k_bp, bool dataAligned)
{
  msgStereo<T, U, DISP_VALS>(xVal, yVal, currentBpLevel, prevUMessage, prevLMessage, prevRMessage, dataMessage,
    currentUMessageArray, disc_k_bp, dataAligned);

  msgStereo<T, U, DISP_VALS>(xVal, yVal, currentBpLevel, prevDMessage, prevLMessage, prevRMessage, dataMessage,
    currentDMessageArray, disc_k_bp, dataAligned);

  msgStereo<T, U, DISP_VALS>(xVal, yVal, currentBpLevel, prevUMessage, prevDMessage, prevRMessage, dataMessage,
    currentRMessageArray, disc_k_bp, dataAligned);

  msgStereo<T, U, DISP_VALS>(xVal, yVal, currentBpLevel, prevUMessage, prevDMessage, prevLMessage, dataMessage,
    currentLMessageArray, disc_k_bp, dataAligned);
}

template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline void runBPIterationUpdateMsgVals(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  U* prevUMessage, U* prevDMessage,
  U* prevLMessage, U* prevRMessage,
  U* dataMessage,
  T* currentUMessageArray, T* currentDMessageArray,
  T* currentLMessageArray, T* currentRMessageArray,
  const U disc_k_bp, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  msgStereo<T, U>(xVal, yVal, currentBpLevel, prevUMessage, prevLMessage, prevRMessage, dataMessage,
    currentUMessageArray, disc_k_bp, dataAligned, bp_settings_disp_vals);

  msgStereo<T, U>(xVal, yVal, currentBpLevel, prevDMessage, prevLMessage, prevRMessage, dataMessage,
    currentDMessageArray, disc_k_bp, dataAligned, bp_settings_disp_vals);

  msgStereo<T, U>(xVal, yVal, currentBpLevel, prevUMessage, prevDMessage, prevRMessage, dataMessage,
    currentRMessageArray, disc_k_bp, dataAligned, bp_settings_disp_vals);

  msgStereo<T, U>(xVal, yVal, currentBpLevel, prevUMessage, prevDMessage, prevLMessage, dataMessage,
    currentLMessageArray, disc_k_bp, dataAligned, bp_settings_disp_vals);
}

template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline void runBPIterationUpdateMsgVals(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  T* prevUMessageArray, T* prevDMessageArray,
  T* prevLMessageArray, T* prevRMessageArray,
  T* dataMessageArray,
  T* currentUMessageArray, T* currentDMessageArray,
  T* currentLMessageArray, T* currentRMessageArray,
  const U disc_k_bp, bool dataAligned, unsigned int bp_settings_disp_vals,
  U* dstProcessing, unsigned int checkerboardAdjustment,
  unsigned int offsetData)
{
  msgStereo<T, U, beliefprop::MessageComp::kUMessage>(xVal, yVal, currentBpLevel, prevUMessageArray, prevDMessageArray, prevLMessageArray, prevRMessageArray, dataMessageArray,
    currentUMessageArray, disc_k_bp, dataAligned, bp_settings_disp_vals, dstProcessing, checkerboardAdjustment, offsetData);

  msgStereo<T, U, beliefprop::MessageComp::kDMessage>(xVal, yVal, currentBpLevel, prevUMessageArray, prevDMessageArray, prevLMessageArray, prevRMessageArray, dataMessageArray,
    currentDMessageArray, disc_k_bp, dataAligned, bp_settings_disp_vals, dstProcessing, checkerboardAdjustment, offsetData);

  msgStereo<T, U, beliefprop::MessageComp::kLMessage>(xVal, yVal, currentBpLevel, prevUMessageArray, prevDMessageArray, prevLMessageArray, prevRMessageArray, dataMessageArray,
    currentLMessageArray, disc_k_bp, dataAligned, bp_settings_disp_vals, dstProcessing, checkerboardAdjustment, offsetData);

  msgStereo<T, U, beliefprop::MessageComp::kRMessage>(xVal, yVal, currentBpLevel, prevUMessageArray, prevDMessageArray, prevLMessageArray, prevRMessageArray, dataMessageArray,
    currentRMessageArray, disc_k_bp, dataAligned, bp_settings_disp_vals, dstProcessing, checkerboardAdjustment, offsetData);
}

//device portion of the kernel function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void runBPIterationUsingCheckerboardUpdatesKernel(
  unsigned int xVal, unsigned int yVal,
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned,
  unsigned int bp_settings_disp_vals)
{
  //checkerboardAdjustment used for indexing into current checkerboard to update
  const unsigned int checkerboardAdjustment = (checkerboardToUpdate == beliefprop::Checkerboard_Part::kCheckerboardPart0) ? ((yVal)%2) : ((yVal+1)%2);

  //may want to look into (xVal < (width_level_checkerboard_part - 1) since it may affect the edges
  //make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
  if ((xVal >= (1u - checkerboardAdjustment)) && (xVal < (currentBpLevel.width_checkerboard_level_ - checkerboardAdjustment)) &&
      (yVal > 0) && (yVal < (currentBpLevel.height_level_ - 1u)))
  {
    if constexpr (DISP_VALS > 0) {
      U dataMessage[DISP_VALS], prevUMessage[DISP_VALS], prevDMessage[DISP_VALS], prevLMessage[DISP_VALS], prevRMessage[DISP_VALS];

      for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
        if (checkerboardToUpdate == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
          dataMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(xVal, yVal,
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, DISP_VALS, offsetData)]);
          prevUMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal+1),
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, DISP_VALS)]);
          prevDMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal-1),
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, DISP_VALS)]);
          prevLMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, DISP_VALS)]);
          prevRMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, DISP_VALS)]);
        }
        else { //checkerboardToUpdate == beliefprop::Checkerboard_Part::kCheckerboardPart1
          dataMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xVal, yVal,
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, DISP_VALS, offsetData)]);
          prevUMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal+1),
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, DISP_VALS)]);
          prevDMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal-1),
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, DISP_VALS)]);
          prevLMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, DISP_VALS)]);
          prevRMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, DISP_VALS)]);
        }
      }

      //uses the previous message values and data cost to calculate the current message values and store the results
      if (checkerboardToUpdate == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
        runBPIterationUpdateMsgVals<T, U, DISP_VALS>(xVal, yVal, currentBpLevel,
          prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
          messageUDeviceCurrentCheckerboard0,  messageDDeviceCurrentCheckerboard0,
          messageLDeviceCurrentCheckerboard0,  messageRDeviceCurrentCheckerboard0,
          (U)disc_k_bp, dataAligned);
      }
      else { //checkerboardToUpdate == beliefprop::Checkerboard_Part::kCheckerboardPart1
        runBPIterationUpdateMsgVals<T, U, DISP_VALS>(xVal, yVal, currentBpLevel,
          prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
          messageUDeviceCurrentCheckerboard1,  messageDDeviceCurrentCheckerboard1,
          messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
          (U)disc_k_bp, dataAligned);
      }
    }
    else {
      U* dataMessage = new U[bp_settings_disp_vals];
      U* prevUMessage = new U[bp_settings_disp_vals];
      U* prevDMessage = new U[bp_settings_disp_vals];
      U* prevLMessage = new U[bp_settings_disp_vals];
      U* prevRMessage = new U[bp_settings_disp_vals];

      for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
        if (checkerboardToUpdate == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
          dataMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(xVal, yVal,
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, bp_settings_disp_vals, offsetData)]);
          prevUMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal+1),
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, bp_settings_disp_vals)]);
          prevDMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal-1),
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, bp_settings_disp_vals)]);
          prevLMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, bp_settings_disp_vals)]);
          prevRMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, bp_settings_disp_vals)]);
        }
        else { //checkerboardToUpdate == beliefprop::Checkerboard_Part::kCheckerboardPart1
          dataMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xVal, yVal,
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, bp_settings_disp_vals, offsetData)]);
          prevUMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal+1),
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, bp_settings_disp_vals)]);
          prevDMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal-1),
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, bp_settings_disp_vals)]);
          prevLMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, bp_settings_disp_vals)]);
          prevRMessage[currentDisparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
              currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
              currentDisparity, bp_settings_disp_vals)]);
        }
      }

      //uses the previous message values and data cost to calculate the current message values and store the results
      if (checkerboardToUpdate == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
        runBPIterationUpdateMsgVals<T, U>(xVal, yVal, currentBpLevel,
          prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
          messageUDeviceCurrentCheckerboard0,  messageDDeviceCurrentCheckerboard0,
          messageLDeviceCurrentCheckerboard0,  messageRDeviceCurrentCheckerboard0,
          (U)disc_k_bp, dataAligned, bp_settings_disp_vals);
      }
      else { //checkerboardToUpdate == beliefprop::Checkerboard_Part::kCheckerboardPart1
        runBPIterationUpdateMsgVals<T, U>(xVal, yVal, currentBpLevel,
          prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
          messageUDeviceCurrentCheckerboard1,  messageDDeviceCurrentCheckerboard1,
          messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
          (U)disc_k_bp, dataAligned, bp_settings_disp_vals);
      }

      delete [] dataMessage;
      delete [] prevUMessage;
      delete [] prevDMessage;
      delete [] prevLMessage;
      delete [] prevRMessage;
    }
  }
}

template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void runBPIterationUsingCheckerboardUpdatesKernel(
  unsigned int xVal, unsigned int yVal,
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned,
  unsigned int bp_settings_disp_vals, void* dstProcessing)
{
  //checkerboardAdjustment used for indexing into current checkerboard to update
  const unsigned int checkerboardAdjustment = (checkerboardToUpdate == beliefprop::Checkerboard_Part::kCheckerboardPart0) ? ((yVal)%2) : ((yVal+1)%2);

  //may want to look into (xVal < (width_level_checkerboard_part - 1) since it may affect the edges
  //make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
  if ((xVal >= (1u - checkerboardAdjustment)) && (xVal < (currentBpLevel.width_checkerboard_level_ - checkerboardAdjustment)) &&
      (yVal > 0) && (yVal < (currentBpLevel.height_level_ - 1u)))
  {
    //uses the previous message values and data cost to calculate the current message values and store the results
    if (checkerboardToUpdate == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
      runBPIterationUpdateMsgVals<T, U>(xVal, yVal, currentBpLevel,
        messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
        messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
        dataCostStereoCheckerboard0,
        messageUDeviceCurrentCheckerboard0,  messageDDeviceCurrentCheckerboard0,
        messageLDeviceCurrentCheckerboard0,  messageRDeviceCurrentCheckerboard0,
        (U)disc_k_bp, dataAligned, bp_settings_disp_vals, (U*)dstProcessing,
        checkerboardAdjustment, offsetData);
    }
    else { //checkerboardToUpdate == beliefprop::Checkerboard_Part::kCheckerboardPart1
      runBPIterationUpdateMsgVals<T, U>(xVal, yVal, currentBpLevel,
        messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
        messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
        dataCostStereoCheckerboard1,
        messageUDeviceCurrentCheckerboard1,  messageDDeviceCurrentCheckerboard1,
        messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
        (U)disc_k_bp, dataAligned, bp_settings_disp_vals, (U*)dstProcessing,
        checkerboardAdjustment, offsetData);
    }
  }
}

//kernel to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernel works from the point of view of the pixel at the prev level that is being copied to four different places
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void copyMsgDataToNextLevelPixel(
  unsigned int xVal, unsigned int yVal,
  beliefprop::Checkerboard_Part checkerboardPart, const beliefprop::BpLevelProperties& currentBpLevel,
  const beliefprop::BpLevelProperties& nextBpLevel,
  T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
  T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
  T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
  T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int bp_settings_disp_vals)
{
  //only need to copy checkerboard 1 around "edge" since updating checkerboard 1 in first belief propagation iteration
  //(and checkerboard 0 message values are used in the iteration to update message values in checkerboard 1)
  const bool copyCheckerboard1 = (((xVal == 0) || (yVal == 0)) || (((xVal >= (currentBpLevel.width_checkerboard_level_ - 2)) || (yVal >= (currentBpLevel.height_level_ - 2)))));
  
  unsigned int indexCopyTo, indexCopyFrom;
  T prevValU, prevValD, prevValL, prevValR;
  const unsigned int checkerboardPartAdjustment = (checkerboardPart == beliefprop::Checkerboard_Part::kCheckerboardPart0) ? (yVal % 2) : ((yVal + 1) % 2);
  
  if constexpr (DISP_VALS > 0) {
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      indexCopyFrom = retrieveIndexInDataAndMessage(xVal, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, DISP_VALS);

      if (checkerboardPart == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
        prevValU = messageUPrevStereoCheckerboard0[indexCopyFrom];
        prevValD = messageDPrevStereoCheckerboard0[indexCopyFrom];
        prevValL = messageLPrevStereoCheckerboard0[indexCopyFrom];
        prevValR = messageRPrevStereoCheckerboard0[indexCopyFrom];
      } else /*(checkerboardPart == beliefprop::Checkerboard_Part::kCheckerboardPart1)*/ {
        prevValU = messageUPrevStereoCheckerboard1[indexCopyFrom];
        prevValD = messageDPrevStereoCheckerboard1[indexCopyFrom];
        prevValL = messageLPrevStereoCheckerboard1[indexCopyFrom];
        prevValR = messageRPrevStereoCheckerboard1[indexCopyFrom];
      }

      if (run_imp_util::WithinImageBounds(xVal*2 + checkerboardPartAdjustment, yVal*2, nextBpLevel.width_checkerboard_level_, nextBpLevel.height_level_)) {
        indexCopyTo = retrieveIndexInDataAndMessage(xVal*2 + checkerboardPartAdjustment, yVal*2,
          nextBpLevel.padded_width_checkerboard_level_, nextBpLevel.height_level_,
          currentDisparity, DISP_VALS);

        messageUDeviceCurrentCheckerboard0[indexCopyTo] = prevValU;
        messageDDeviceCurrentCheckerboard0[indexCopyTo] = prevValD;
        messageLDeviceCurrentCheckerboard0[indexCopyTo] = prevValL;
        messageRDeviceCurrentCheckerboard0[indexCopyTo] = prevValR;

        if (copyCheckerboard1) {
          messageUDeviceCurrentCheckerboard1[indexCopyTo] = prevValU;
          messageDDeviceCurrentCheckerboard1[indexCopyTo] = prevValD;
          messageLDeviceCurrentCheckerboard1[indexCopyTo] = prevValL;
          messageRDeviceCurrentCheckerboard1[indexCopyTo] = prevValR;
        }
      }

      if (run_imp_util::WithinImageBounds(xVal*2 + checkerboardPartAdjustment, yVal*2 + 1, nextBpLevel.width_checkerboard_level_, nextBpLevel.height_level_)) {
        indexCopyTo = retrieveIndexInDataAndMessage(xVal*2 + checkerboardPartAdjustment, yVal*2 + 1,
          nextBpLevel.padded_width_checkerboard_level_, nextBpLevel.height_level_,
          currentDisparity, DISP_VALS);

        messageUDeviceCurrentCheckerboard0[indexCopyTo] = prevValU;
        messageDDeviceCurrentCheckerboard0[indexCopyTo] = prevValD;
        messageLDeviceCurrentCheckerboard0[indexCopyTo] = prevValL;
        messageRDeviceCurrentCheckerboard0[indexCopyTo] = prevValR;

        if (copyCheckerboard1) {
          messageUDeviceCurrentCheckerboard1[indexCopyTo] = prevValU;
          messageDDeviceCurrentCheckerboard1[indexCopyTo] = prevValD;
          messageLDeviceCurrentCheckerboard1[indexCopyTo] = prevValL;
          messageRDeviceCurrentCheckerboard1[indexCopyTo] = prevValR;
        }
      }
    }
  }
  else {
    for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
      indexCopyFrom = retrieveIndexInDataAndMessage(xVal, yVal,
        currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
        currentDisparity, bp_settings_disp_vals);

      if (checkerboardPart == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
        prevValU = messageUPrevStereoCheckerboard0[indexCopyFrom];
        prevValD = messageDPrevStereoCheckerboard0[indexCopyFrom];
        prevValL = messageLPrevStereoCheckerboard0[indexCopyFrom];
        prevValR = messageRPrevStereoCheckerboard0[indexCopyFrom];
      } else /*(checkerboardPart == beliefprop::Checkerboard_Part::kCheckerboardPart1)*/ {
        prevValU = messageUPrevStereoCheckerboard1[indexCopyFrom];
        prevValD = messageDPrevStereoCheckerboard1[indexCopyFrom];
        prevValL = messageLPrevStereoCheckerboard1[indexCopyFrom];
        prevValR = messageRPrevStereoCheckerboard1[indexCopyFrom];
      }

      if (run_imp_util::WithinImageBounds(xVal*2 + checkerboardPartAdjustment, yVal*2, nextBpLevel.width_checkerboard_level_, nextBpLevel.height_level_)) {
        indexCopyTo = retrieveIndexInDataAndMessage(xVal*2 + checkerboardPartAdjustment, yVal*2,
          nextBpLevel.padded_width_checkerboard_level_, nextBpLevel.height_level_,
          currentDisparity, bp_settings_disp_vals);

        messageUDeviceCurrentCheckerboard0[indexCopyTo] = prevValU;
        messageDDeviceCurrentCheckerboard0[indexCopyTo] = prevValD;
        messageLDeviceCurrentCheckerboard0[indexCopyTo] = prevValL;
        messageRDeviceCurrentCheckerboard0[indexCopyTo] = prevValR;

        if (copyCheckerboard1) {
          messageUDeviceCurrentCheckerboard1[indexCopyTo] = prevValU;
          messageDDeviceCurrentCheckerboard1[indexCopyTo] = prevValD;
          messageLDeviceCurrentCheckerboard1[indexCopyTo] = prevValL;
          messageRDeviceCurrentCheckerboard1[indexCopyTo] = prevValR;
        }
      }

      if (run_imp_util::WithinImageBounds(xVal*2 + checkerboardPartAdjustment, yVal*2 + 1, nextBpLevel.width_checkerboard_level_, nextBpLevel.height_level_)) {
        indexCopyTo = retrieveIndexInDataAndMessage(xVal*2 + checkerboardPartAdjustment, yVal*2 + 1,
          nextBpLevel.padded_width_checkerboard_level_, nextBpLevel.height_level_,
          currentDisparity, bp_settings_disp_vals);

        messageUDeviceCurrentCheckerboard0[indexCopyTo] = prevValU;
        messageDDeviceCurrentCheckerboard0[indexCopyTo] = prevValD;
        messageLDeviceCurrentCheckerboard0[indexCopyTo] = prevValL;
        messageRDeviceCurrentCheckerboard0[indexCopyTo] = prevValR;

        if (copyCheckerboard1) {
          messageUDeviceCurrentCheckerboard1[indexCopyTo] = prevValU;
          messageDDeviceCurrentCheckerboard1[indexCopyTo] = prevValD;
          messageLDeviceCurrentCheckerboard1[indexCopyTo] = prevValL;
          messageRDeviceCurrentCheckerboard1[indexCopyTo] = prevValR;
        }
      }
    }
  }
}

template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void retrieveOutputDisparityPixel(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUPrevStereoCheckerboard0,  T* messageDPrevStereoCheckerboard0,
  T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
  T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
  T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  const unsigned int xValInCheckerboardPart = xVal;

  //first processing from first part of checkerboard

  //adjustment based on checkerboard; need to add 1 to x for odd-numbered rows for final index mapping into disparity images for checkerboard 1
  unsigned int checkerboardPartAdjustment = (yVal % 2);

  if (run_imp_util::WithinImageBounds(xValInCheckerboardPart*2 + checkerboardPartAdjustment, yVal, currentBpLevel.width_level_, currentBpLevel.height_level_)) {
    if ((xValInCheckerboardPart >= (1 - checkerboardPartAdjustment)) &&
        (xValInCheckerboardPart < (currentBpLevel.width_checkerboard_level_ - checkerboardPartAdjustment)) &&
        (yVal > 0u) && (yVal < (currentBpLevel.height_level_ - 1u)))
    {
      // keep track of "best" disparity for current pixel
      unsigned int bestDisparity{0u};
      U best_val{(U)bp_consts::kInfBp};
      if constexpr (DISP_VALS > 0) {
        for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
          const U val =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal + 1u),
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal - 1u),
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart  + checkerboardPartAdjustment), yVal,
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart + checkerboardPartAdjustment) - 1u, yVal,
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, yVal,
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              DISP_VALS)]);
          if (val < best_val) {
            best_val = val;
            bestDisparity = currentDisparity;
          }
        }
      }
      else {
        for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
          const U val =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal + 1u),
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal - 1u),
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart  + checkerboardPartAdjustment), yVal,
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart + checkerboardPartAdjustment) - 1u, yVal,
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, yVal,
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              bp_settings_disp_vals)]);
          if (val < best_val) {
            best_val = val;
            bestDisparity = currentDisparity;
          }
        }
      }

      disparityBetweenImagesDevice[yVal*currentBpLevel.width_level_ +
        (xValInCheckerboardPart * 2 + checkerboardPartAdjustment)] = bestDisparity;
    } else {
      disparityBetweenImagesDevice[yVal* currentBpLevel.width_level_ +
        (xValInCheckerboardPart * 2 + checkerboardPartAdjustment)] = 0;
    }
  }

  //process from part 2 of checkerboard
  //adjustment based on checkerboard; need to add 1 to x for even-numbered rows for final index mapping into disparity images for checkerboard 2
  checkerboardPartAdjustment = ((yVal + 1u) % 2);

  if (run_imp_util::WithinImageBounds(xValInCheckerboardPart*2 + checkerboardPartAdjustment, yVal, currentBpLevel.width_level_, currentBpLevel.height_level_)) {
    if ((xValInCheckerboardPart >= (1 - checkerboardPartAdjustment)) &&
        (xValInCheckerboardPart < (currentBpLevel.width_checkerboard_level_ - checkerboardPartAdjustment)) &&
        (yVal > 0) && (yVal < (currentBpLevel.height_level_ - 1)))
    {
      // keep track of "best" disparity for current pixel
      unsigned int bestDisparity{0u};
      U best_val{(U)bp_consts::kInfBp};
      if constexpr (DISP_VALS > 0) {
        for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
          const U val = 
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal + 1u),
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal - 1u),
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart  + checkerboardPartAdjustment), yVal,
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart + checkerboardPartAdjustment) - 1u,  yVal,
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, yVal,
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              DISP_VALS)]);
          if (val < best_val) {
            best_val = val;
            bestDisparity = currentDisparity;
          }
        }
      }
      else {
        for (unsigned int currentDisparity = 0; currentDisparity < bp_settings_disp_vals; currentDisparity++) {
          const U val =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal + 1u),
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal - 1u),
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart  + checkerboardPartAdjustment), yVal,
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart + checkerboardPartAdjustment) - 1u,  yVal,
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, yVal,
              currentBpLevel.padded_width_checkerboard_level_,
              currentBpLevel.height_level_,
              currentDisparity,
              bp_settings_disp_vals)]);
          if (val < best_val) {
            best_val = val;
            bestDisparity = currentDisparity;
          }
        }
      }

      disparityBetweenImagesDevice[yVal * currentBpLevel.width_level_ +
        (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = bestDisparity;
    } else {
      disparityBetweenImagesDevice[yVal * currentBpLevel.width_level_ +
        (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = 0;
    }
  }
}

template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void printDataAndMessageValsAtPointKernel(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int bp_settings_disp_vals = 0)
{
  if (((xVal + yVal) % 2) == 0) {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
    }
  } else {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
    }
  }
}

template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void printDataAndMessageValsToPointKernel(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int bp_settings_disp_vals = 0)
{
  const unsigned int checkerboardAdjustment = (((xVal + yVal) % 2) == 0) ? ((yVal)%2) : ((yVal+1)%2);
  if (((xVal + yVal) % 2) == 0) {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal + 1, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal - 1, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2 + checkerboardAdjustment, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          (xVal / 2 - 1) + checkerboardAdjustment, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
           currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
    }
  } else {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal + 1, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal - 1, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2 + checkerboardAdjustment, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          (xVal / 2 - 1) + checkerboardAdjustment, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentBpLevel.padded_width_checkerboard_level_, currentBpLevel.height_level_,
          currentDisparity, DISP_VALS)]);
    }
  }
}

}

#endif /* SHAREDBPPROCESSINGFUNCTS_H_ */
