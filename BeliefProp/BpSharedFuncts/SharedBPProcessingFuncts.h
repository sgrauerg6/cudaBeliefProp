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
ARCHITECTURE_ADDITION inline unsigned int retrieveIndexInDataAndMessage(const unsigned int xVal, const unsigned int yVal,
  const unsigned int width, const unsigned int height, const unsigned int currentDisparity, const unsigned int totalNumDispVals,
  const unsigned int offsetData = 0u)
{
  if constexpr (bp_params::OPTIMIZED_INDEXING_SETTING) {
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
ARCHITECTURE_ADDITION inline void dtStereo(T* f, const unsigned int bpSettingsDispVals)
{
  T prev;
  for (unsigned int currentDisparity = 1; currentDisparity < bpSettingsDispVals; currentDisparity++)
  {
    prev = f[currentDisparity-1] + (T)1.0;
    if (prev < f[currentDisparity])
      f[currentDisparity] = prev;
  }

  for (int currentDisparity = (int)bpSettingsDispVals - 2; currentDisparity >= 0; currentDisparity--)
  {
    prev = f[currentDisparity + 1] + (T)1.0;
    if (prev < f[currentDisparity])
      f[currentDisparity] = prev;
  }
}

template<RunData_t T>
ARCHITECTURE_ADDITION inline void dtStereo(T* f, const unsigned int bpSettingsDispVals,
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties)
{
  unsigned int fArrayIndexDisp0 = retrieveIndexInDataAndMessage(xVal, yVal,
    currentLevelProperties.paddedWidthCheckerboardLevel_,
    currentLevelProperties.heightLevel_, 0,
    bpSettingsDispVals);
  unsigned int currFArrayIndexLast = fArrayIndexDisp0;
  unsigned int currFArrayIndex = fArrayIndexDisp0;
  T prev;

  for (unsigned int currentDisparity = 1; currentDisparity < bpSettingsDispVals; currentDisparity++)
  {
    if constexpr (bp_params::OPTIMIZED_INDEXING_SETTING) {
      currFArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
    }
    else {
      currFArrayIndex++;
    }

    prev = f[currFArrayIndexLast] + (T)1.0;
    if (prev < f[currFArrayIndex])
      f[currFArrayIndex] = prev;
    currFArrayIndexLast = currFArrayIndex;
  }

  for (int currentDisparity = (int)bpSettingsDispVals - 2; currentDisparity >= 0; currentDisparity--)
  {
    if constexpr (bp_params::OPTIMIZED_INDEXING_SETTING) {
      currFArrayIndex -= currentLevelProperties.paddedWidthCheckerboardLevel_;
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
ARCHITECTURE_ADDITION inline void msgStereo(const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  U messageValsNeighbor1[DISP_VALS], U messageValsNeighbor2[DISP_VALS],
  U messageValsNeighbor3[DISP_VALS], U dataCosts[DISP_VALS],
  T* dstMessageArray, U disc_k_bp, const bool dataAligned)
{
  // aggregate and find min
  U minimum{(U)bp_consts::INF_BP};
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
    currentLevelProperties.paddedWidthCheckerboardLevel_,
    currentLevelProperties.heightLevel_, 0,
    DISP_VALS);

  for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
    dst[currentDisparity] -= valToNormalize;
    dstMessageArray[destMessageArrayIndex] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<U, T>(dst[currentDisparity]);
    if constexpr (bp_params::OPTIMIZED_INDEXING_SETTING) {
      destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
    }
    else {
      destMessageArrayIndex++;
    }
  }
}

template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline void msgStereo(const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  U* messageValsNeighbor1, U* messageValsNeighbor2,
  U* messageValsNeighbor3, U* dataCosts,
  T* dstMessageArray, U disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  // aggregate and find min
  U minimum{(U)bp_consts::INF_BP};
  U* dst = new U[bpSettingsDispVals];

  for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
  {
    dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
    if (dst[currentDisparity] < minimum)
      minimum = dst[currentDisparity];
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
  dtStereo<U>(dst, bpSettingsDispVals);

  // truncate
  minimum += disc_k_bp;

  // normalize
  U valToNormalize{(U)0.0};

  for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
    if (minimum < dst[currentDisparity]) {
      dst[currentDisparity] = minimum;
    }

    valToNormalize += dst[currentDisparity];
  }

  valToNormalize /= ((U)bpSettingsDispVals);

  int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
    currentLevelProperties.paddedWidthCheckerboardLevel_,
    currentLevelProperties.heightLevel_, 0,
    bpSettingsDispVals);

  for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
    dst[currentDisparity] -= valToNormalize;
    dstMessageArray[destMessageArrayIndex] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<U, T>(dst[currentDisparity]);
    if constexpr (bp_params::OPTIMIZED_INDEXING_SETTING) {
      destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
    }
    else {
      destMessageArrayIndex++;
    }
  }

  delete [] dst;
}

template<RunData_t T, RunData_t U, beliefprop::messageComp M>
ARCHITECTURE_ADDITION void inline setInitDstProcessing(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  T* prevUMessageArray, T* prevDMessageArray,
  T* prevLMessageArray, T* prevRMessageArray,
  T* dataMessageArray, T* dstMessageArray,
  U disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals,
  U* dstProcessing, const unsigned int checkerboardAdjustment,
  const unsigned int offsetData, const unsigned int currentDisparity,
  const unsigned int procArrIdx)
{
  const U dataVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataMessageArray[retrieveIndexInDataAndMessage(xVal, yVal,
    currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
    currentDisparity, bpSettingsDispVals, offsetData)]);

  if constexpr (M == beliefprop::messageComp::U_MESSAGE) {
    const U prevUVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(prevUMessageArray[retrieveIndexInDataAndMessage(xVal, (yVal+1),
      currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
      currentDisparity, bpSettingsDispVals)]);
    const U prevLVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(prevLMessageArray[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
      currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
      currentDisparity, bpSettingsDispVals)]);
    const U prevRVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(prevRMessageArray[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
      currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
      currentDisparity, bpSettingsDispVals)]);

    dstProcessing[procArrIdx] = prevUVal + prevLVal + prevRVal + dataVal;
  }
  else if constexpr (M == beliefprop::messageComp::D_MESSAGE) {
    const U prevDVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(prevDMessageArray[retrieveIndexInDataAndMessage(xVal, (yVal-1),
      currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
      currentDisparity, bpSettingsDispVals)]);
    const U prevLVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(prevLMessageArray[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
      currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
      currentDisparity, bpSettingsDispVals)]);
    const U prevRVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(prevRMessageArray[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
      currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
      currentDisparity, bpSettingsDispVals)]);

    dstProcessing[procArrIdx] = prevDVal + prevLVal + prevRVal + dataVal;
  }
  else if constexpr (M == beliefprop::messageComp::L_MESSAGE) {
    const U prevUVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(prevUMessageArray[retrieveIndexInDataAndMessage(xVal, (yVal+1),
      currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
      currentDisparity, bpSettingsDispVals)]);
    const U prevDVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(prevDMessageArray[retrieveIndexInDataAndMessage(xVal, (yVal-1),
      currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
      currentDisparity, bpSettingsDispVals)]);
    const U prevLVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(prevLMessageArray[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
      currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
      currentDisparity, bpSettingsDispVals)]);

    dstProcessing[procArrIdx] = prevUVal + prevDVal + prevLVal + dataVal;
  }
  else if constexpr (M == beliefprop::messageComp::R_MESSAGE) {
    const U prevUVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(prevUMessageArray[retrieveIndexInDataAndMessage(xVal, (yVal+1),
      currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
      currentDisparity, bpSettingsDispVals)]);
    const U prevDVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(prevDMessageArray[retrieveIndexInDataAndMessage(xVal, (yVal-1),
      currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
      currentDisparity, bpSettingsDispVals)]);
    const U prevRVal = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(prevRMessageArray[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
      currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
      currentDisparity, bpSettingsDispVals)]);

    dstProcessing[procArrIdx] = prevUVal + prevDVal + prevRVal + dataVal;
  }
}

//TODO: may need to specialize for half-precision to account for possible NaN/inf vals
template<RunData_t T, RunData_t U, beliefprop::messageComp M>
ARCHITECTURE_ADDITION inline void msgStereo(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  T* prevUMessageArray, T* prevDMessageArray,
  T* prevLMessageArray, T* prevRMessageArray,
  T* dataMessageArray, T* dstMessageArray,
  U disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals,
  U* dstProcessing, const unsigned int checkerboardAdjustment,
  const unsigned int offsetData)
{
  // aggregate and find min
  U minimum{(U)bp_consts::INF_BP};
  unsigned int processingArrIndexDisp0 = retrieveIndexInDataAndMessage(xVal, yVal,
    currentLevelProperties.paddedWidthCheckerboardLevel_,
    currentLevelProperties.heightLevel_, 0,
    bpSettingsDispVals);
  unsigned int procArrIdx{processingArrIndexDisp0};

  for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
  {
    //set initial dst processing array value corresponding to disparity for M message type
    setInitDstProcessing<T, U, M>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray,
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
  dtStereo<U>(dstProcessing, bpSettingsDispVals, xVal, yVal, currentLevelProperties);

  // truncate
  minimum += disc_k_bp;

  // normalize
  U valToNormalize{(U)0.0};

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

  valToNormalize /= ((U)bpSettingsDispVals);

  //dst processing index and message array index are the same for each disparity value in this processing
  procArrIdx = processingArrIndexDisp0;

  for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
    dstProcessing[procArrIdx] -= valToNormalize;
    dstMessageArray[procArrIdx] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<U, T>(dstProcessing[procArrIdx]);
    if constexpr (bp_params::OPTIMIZED_INDEXING_SETTING) {
      procArrIdx += currentLevelProperties.paddedWidthCheckerboardLevel_;
    }
    else {
      procArrIdx++;
    }
  }
}

//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void initializeBottomLevelDataPixel(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, float* image1PixelsDevice,
  float* image2PixelsDevice, T* dataCostDeviceStereoCheckerboard0,
  T* dataCostDeviceStereoCheckerboard1, const float lambda_bp,
  const float data_k_bp, const unsigned int bpSettingsDispVals)
{
  if constexpr (DISP_VALS > 0) {
    unsigned int indexVal;
    const unsigned int xInCheckerboard = xVal / 2;

    if (GenProcessingFuncts::withinImageBounds(xInCheckerboard, yVal, currentLevelProperties.widthCheckerboardLevel_, currentLevelProperties.heightLevel_)) {
      //make sure that it is possible to check every disparity value
      //need to cast DISP_VALS from unsigned int to int
      //for conditional to work as expected
      if (((int)xVal - ((int)DISP_VALS - 1)) >= 0) {
        for (unsigned int currentDisparity = 0u; currentDisparity < DISP_VALS; currentDisparity++) {
          float currentPixelImage1 = 0.0f, currentPixelImage2 = 0.0f;

          if (GenProcessingFuncts::withinImageBounds(xVal, yVal, currentLevelProperties.widthLevel_, currentLevelProperties.heightLevel_)) {
            currentPixelImage1 = image1PixelsDevice[yVal * currentLevelProperties.widthLevel_ + xVal];
            currentPixelImage2 = image2PixelsDevice[yVal * currentLevelProperties.widthLevel_ + (xVal - currentDisparity)];
          }

          indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal,
            currentLevelProperties.paddedWidthCheckerboardLevel_,
            currentLevelProperties.heightLevel_, currentDisparity,
            DISP_VALS);

          //data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
          if (((xVal + yVal) % 2) == 0) {
            dataCostDeviceStereoCheckerboard0[indexVal] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * GenProcessingFuncts::getMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
          }
          else {
            dataCostDeviceStereoCheckerboard1[indexVal] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * GenProcessingFuncts::getMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
          }
        }
      } else {
        for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
          indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal,
            currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
            currentDisparity, DISP_VALS);

          //data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
          if (((xVal + yVal) % 2) == 0) {
            dataCostDeviceStereoCheckerboard0[indexVal] = GenProcessingFuncts::getZeroVal<T>();
          }
          else {
            dataCostDeviceStereoCheckerboard1[indexVal] = GenProcessingFuncts::getZeroVal<T>();
          }
        }
      }
    }
  }
  else {
    unsigned int indexVal;
    const unsigned int xInCheckerboard = xVal / 2;

    if (GenProcessingFuncts::withinImageBounds(xInCheckerboard, yVal, currentLevelProperties.widthCheckerboardLevel_, currentLevelProperties.heightLevel_)) {
      //make sure that it is possible to check every disparity value
      //need to cast bpSettingsDispVals from unsigned int to int
      //for conditional to work as expected
      if (((int)xVal - ((int)bpSettingsDispVals - 1)) >= 0) {
        for (unsigned int currentDisparity = 0u; currentDisparity < bpSettingsDispVals; currentDisparity++) {
          float currentPixelImage1 = 0.0f, currentPixelImage2 = 0.0f;

          if (GenProcessingFuncts::withinImageBounds(xVal, yVal, currentLevelProperties.widthLevel_, currentLevelProperties.heightLevel_)) {
            currentPixelImage1 = image1PixelsDevice[yVal * currentLevelProperties.widthLevel_ + xVal];
            currentPixelImage2 = image2PixelsDevice[yVal * currentLevelProperties.widthLevel_ + (xVal - currentDisparity)];
          }

          indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal,
            currentLevelProperties.paddedWidthCheckerboardLevel_,
            currentLevelProperties.heightLevel_, currentDisparity,
            bpSettingsDispVals);

          //data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
          if (((xVal + yVal) % 2) == 0) {
            dataCostDeviceStereoCheckerboard0[indexVal] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * GenProcessingFuncts::getMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
          }
          else {
            dataCostDeviceStereoCheckerboard1[indexVal] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * GenProcessingFuncts::getMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
          }
        }
      } else {
        for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
          indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal,
            currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
            currentDisparity, bpSettingsDispVals);

          //data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
          if (((xVal + yVal) % 2) == 0) {
            dataCostDeviceStereoCheckerboard0[indexVal] = GenProcessingFuncts::getZeroVal<T>();
          }
          else {
            dataCostDeviceStereoCheckerboard1[indexVal] = GenProcessingFuncts::getZeroVal<T>();
          }
        }
      }
    }
  }
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void initializeCurrentLevelDataPixel(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* dataCostDeviceToWriteTo, const unsigned int offsetNum,
  const unsigned int bpSettingsDispVals)
{
  //add 1 or 0 to the x-value depending on checkerboard part and row adding to; beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0 with slot at (0, 0) has adjustment of 0 in row 0,
  //while beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1 with slot at (0, 1) has adjustment of 1 in row 0
  const unsigned int checkerboardPartAdjustment = (checkerboardPart == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0) ? (yVal % 2) : ((yVal + 1) % 2);

  //the corresponding x-values at the "lower" level depends on which checkerboard the pixel is in
  const unsigned int xValPrev = xVal*2 + checkerboardPartAdjustment;

  if (GenProcessingFuncts::withinImageBounds(xValPrev, (yVal * 2 + 1), prevLevelProperties.widthCheckerboardLevel_, prevLevelProperties.heightLevel_)) {
    if constexpr (DISP_VALS > 0) {
      for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
        const U dataCostVal =
          GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
            currentDisparity, DISP_VALS, offsetNum)]) +
          GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
            currentDisparity, DISP_VALS, offsetNum)]) +
          GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2 + 1, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
            currentDisparity, DISP_VALS, offsetNum)]) +
          GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2 + 1, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
            currentDisparity, DISP_VALS, offsetNum)]);

        dataCostDeviceToWriteTo[retrieveIndexInDataAndMessage(xVal, yVal,
          currentLevelProperties.paddedWidthCheckerboardLevel_,
          currentLevelProperties.heightLevel_, currentDisparity,
          DISP_VALS)] =
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<U, T>(dataCostVal);
      }
    }
    else {
      for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
        const U dataCostVal =
          GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
            currentDisparity, bpSettingsDispVals, offsetNum)]) +
          GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
            currentDisparity, bpSettingsDispVals, offsetNum)]) +
          GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2 + 1, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
            currentDisparity, bpSettingsDispVals, offsetNum)]) +
          GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
            xValPrev, yVal*2 + 1, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
            currentDisparity, bpSettingsDispVals, offsetNum)]);

        dataCostDeviceToWriteTo[retrieveIndexInDataAndMessage(xVal, yVal,
          currentLevelProperties.paddedWidthCheckerboardLevel_,
          currentLevelProperties.heightLevel_, currentDisparity,
          bpSettingsDispVals)] =
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<U, T>(dataCostVal);
      }
    }
  }
}

//initialize the message values at each pixel of the current level to the default value
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void initializeMessageValsToDefaultKernelPixel(
  const unsigned int xValInCheckerboard, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  const unsigned int bpSettingsDispVals)
{
  //initialize message values in both checkerboards

  if constexpr (DISP_VALS > 0) {
    //set the message value at each pixel for each disparity to 0
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, DISP_VALS)] =
          GenProcessingFuncts::getZeroVal<T>();
      messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, DISP_VALS)] =
          GenProcessingFuncts::getZeroVal<T>();
      messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, DISP_VALS)] =
          GenProcessingFuncts::getZeroVal<T>();
      messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, DISP_VALS)] =
          GenProcessingFuncts::getZeroVal<T>();
    }

    //retrieve the previous message value at each movement at each pixel
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, DISP_VALS)] =
          GenProcessingFuncts::getZeroVal<T>();
      messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, DISP_VALS)] =
          GenProcessingFuncts::getZeroVal<T>();
      messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, DISP_VALS)] =
          GenProcessingFuncts::getZeroVal<T>();
      messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, DISP_VALS)] =
          GenProcessingFuncts::getZeroVal<T>();
    }
  }
  else {
    //set the message value at each pixel for each disparity to 0
    for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
      messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, bpSettingsDispVals)] =
          GenProcessingFuncts::getZeroVal<T>();
      messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, bpSettingsDispVals)] =
          GenProcessingFuncts::getZeroVal<T>();
      messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, bpSettingsDispVals)] =
          GenProcessingFuncts::getZeroVal<T>();
      messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, bpSettingsDispVals)] =
          GenProcessingFuncts::getZeroVal<T>();
    }

    //retrieve the previous message value at each movement at each pixel
    for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
      messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, bpSettingsDispVals)] =
          GenProcessingFuncts::getZeroVal<T>();
      messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, bpSettingsDispVals)] =
          GenProcessingFuncts::getZeroVal<T>();
      messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, bpSettingsDispVals)] =
          GenProcessingFuncts::getZeroVal<T>();
      messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, bpSettingsDispVals)] =
          GenProcessingFuncts::getZeroVal<T>();
    }
  }
}

//device portion of the kernel function to run the current iteration of belief propagation where the input messages and data costs come in as array in local memory
//and the output message values are stored in local memory
template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void runBPIterationUpdateMsgVals(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  U prevUMessage[DISP_VALS], U prevDMessage[DISP_VALS],
  U prevLMessage[DISP_VALS], U prevRMessage[DISP_VALS],
  U dataMessage[DISP_VALS],
  T* currentUMessageArray, T* currentDMessageArray,
  T* currentLMessageArray, T* currentRMessageArray,
  const U disc_k_bp, const bool dataAligned)
{
  msgStereo<T, U, DISP_VALS>(xVal, yVal, currentLevelProperties, prevUMessage, prevLMessage, prevRMessage, dataMessage,
    currentUMessageArray, disc_k_bp, dataAligned);

  msgStereo<T, U, DISP_VALS>(xVal, yVal, currentLevelProperties, prevDMessage, prevLMessage, prevRMessage, dataMessage,
    currentDMessageArray, disc_k_bp, dataAligned);

  msgStereo<T, U, DISP_VALS>(xVal, yVal, currentLevelProperties, prevUMessage, prevDMessage, prevRMessage, dataMessage,
    currentRMessageArray, disc_k_bp, dataAligned);

  msgStereo<T, U, DISP_VALS>(xVal, yVal, currentLevelProperties, prevUMessage, prevDMessage, prevLMessage, dataMessage,
    currentLMessageArray, disc_k_bp, dataAligned);
}

template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline void runBPIterationUpdateMsgVals(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  U* prevUMessage, U* prevDMessage,
  U* prevLMessage, U* prevRMessage,
  U* dataMessage,
  T* currentUMessageArray, T* currentDMessageArray,
  T* currentLMessageArray, T* currentRMessageArray,
  const U disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevUMessage, prevLMessage, prevRMessage, dataMessage,
    currentUMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);

  msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevDMessage, prevLMessage, prevRMessage, dataMessage,
    currentDMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);

  msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevUMessage, prevDMessage, prevRMessage, dataMessage,
    currentRMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);

  msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevUMessage, prevDMessage, prevLMessage, dataMessage,
    currentLMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);
}

template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline void runBPIterationUpdateMsgVals(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  T* prevUMessageArray, T* prevDMessageArray,
  T* prevLMessageArray, T* prevRMessageArray,
  T* dataMessageArray,
  T* currentUMessageArray, T* currentDMessageArray,
  T* currentLMessageArray, T* currentRMessageArray,
  const U disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals,
  U* dstProcessing, const unsigned int checkerboardAdjustment,
  const unsigned int offsetData)
{
  msgStereo<T, U, beliefprop::messageComp::U_MESSAGE>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray, prevLMessageArray, prevRMessageArray, dataMessageArray,
    currentUMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals, dstProcessing, checkerboardAdjustment, offsetData);

  msgStereo<T, U, beliefprop::messageComp::D_MESSAGE>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray, prevLMessageArray, prevRMessageArray, dataMessageArray,
    currentDMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals, dstProcessing, checkerboardAdjustment, offsetData);

  msgStereo<T, U, beliefprop::messageComp::L_MESSAGE>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray, prevLMessageArray, prevRMessageArray, dataMessageArray,
    currentLMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals, dstProcessing, checkerboardAdjustment, offsetData);

  msgStereo<T, U, beliefprop::messageComp::R_MESSAGE>(xVal, yVal, currentLevelProperties, prevUMessageArray, prevDMessageArray, prevLMessageArray, prevRMessageArray, dataMessageArray,
    currentRMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals, dstProcessing, checkerboardAdjustment, offsetData);
}

//device portion of the kernel function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void runBPIterationUsingCheckerboardUpdatesKernel(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned,
  const unsigned int bpSettingsDispVals)
{
  //checkerboardAdjustment used for indexing into current checkerboard to update
  const unsigned int checkerboardAdjustment = (checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0) ? ((yVal)%2) : ((yVal+1)%2);

  //may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
  //make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
  if ((xVal >= (1u - checkerboardAdjustment)) && (xVal < (currentLevelProperties.widthCheckerboardLevel_ - checkerboardAdjustment)) &&
      (yVal > 0) && (yVal < (currentLevelProperties.heightLevel_ - 1u)))
  {
    if constexpr (DISP_VALS > 0) {
      U dataMessage[DISP_VALS], prevUMessage[DISP_VALS], prevDMessage[DISP_VALS], prevLMessage[DISP_VALS], prevRMessage[DISP_VALS];

      for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
        if (checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0) {
          dataMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(xVal, yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, DISP_VALS, offsetData)]);
          prevUMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal+1),
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, DISP_VALS)]);
          prevDMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal-1),
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, DISP_VALS)]);
          prevLMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, DISP_VALS)]);
          prevRMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, DISP_VALS)]);
        }
        else { //checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1
          dataMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xVal, yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, DISP_VALS, offsetData)]);
          prevUMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal+1),
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, DISP_VALS)]);
          prevDMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal-1),
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, DISP_VALS)]);
          prevLMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, DISP_VALS)]);
          prevRMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, DISP_VALS)]);
        }
      }

      //uses the previous message values and data cost to calculate the current message values and store the results
      if (checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0) {
        runBPIterationUpdateMsgVals<T, U, DISP_VALS>(xVal, yVal, currentLevelProperties,
          prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
          messageUDeviceCurrentCheckerboard0,  messageDDeviceCurrentCheckerboard0,
          messageLDeviceCurrentCheckerboard0,  messageRDeviceCurrentCheckerboard0,
          (U)disc_k_bp, dataAligned);
      }
      else { //checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1
        runBPIterationUpdateMsgVals<T, U, DISP_VALS>(xVal, yVal, currentLevelProperties,
          prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
          messageUDeviceCurrentCheckerboard1,  messageDDeviceCurrentCheckerboard1,
          messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
          (U)disc_k_bp, dataAligned);
      }
    }
    else {
      U* dataMessage = new U[bpSettingsDispVals];
      U* prevUMessage = new U[bpSettingsDispVals];
      U* prevDMessage = new U[bpSettingsDispVals];
      U* prevLMessage = new U[bpSettingsDispVals];
      U* prevRMessage = new U[bpSettingsDispVals];

      for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
        if (checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0) {
          dataMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(xVal, yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, bpSettingsDispVals, offsetData)]);
          prevUMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal+1),
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, bpSettingsDispVals)]);
          prevDMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal-1),
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, bpSettingsDispVals)]);
          prevLMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, bpSettingsDispVals)]);
          prevRMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, bpSettingsDispVals)]);
        }
        else { //checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1
          dataMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xVal, yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, bpSettingsDispVals, offsetData)]);
          prevUMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal+1),
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, bpSettingsDispVals)]);
          prevDMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal-1),
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, bpSettingsDispVals)]);
          prevLMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, bpSettingsDispVals)]);
          prevRMessage[currentDisparity] = GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(
            messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
              currentDisparity, bpSettingsDispVals)]);
        }
      }

      //uses the previous message values and data cost to calculate the current message values and store the results
      if (checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0) {
        runBPIterationUpdateMsgVals<T, U>(xVal, yVal, currentLevelProperties,
          prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
          messageUDeviceCurrentCheckerboard0,  messageDDeviceCurrentCheckerboard0,
          messageLDeviceCurrentCheckerboard0,  messageRDeviceCurrentCheckerboard0,
          (U)disc_k_bp, dataAligned, bpSettingsDispVals);
      }
      else { //checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1
        runBPIterationUpdateMsgVals<T, U>(xVal, yVal, currentLevelProperties,
          prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
          messageUDeviceCurrentCheckerboard1,  messageDDeviceCurrentCheckerboard1,
          messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
          (U)disc_k_bp, dataAligned, bpSettingsDispVals);
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
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned,
  const unsigned int bpSettingsDispVals, void* dstProcessing)
{
  //checkerboardAdjustment used for indexing into current checkerboard to update
  const unsigned int checkerboardAdjustment = (checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0) ? ((yVal)%2) : ((yVal+1)%2);

  //may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
  //make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
  if ((xVal >= (1u - checkerboardAdjustment)) && (xVal < (currentLevelProperties.widthCheckerboardLevel_ - checkerboardAdjustment)) &&
      (yVal > 0) && (yVal < (currentLevelProperties.heightLevel_ - 1u)))
  {
    //uses the previous message values and data cost to calculate the current message values and store the results
    if (checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0) {
      runBPIterationUpdateMsgVals<T, U>(xVal, yVal, currentLevelProperties,
        messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
        messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
        dataCostStereoCheckerboard0,
        messageUDeviceCurrentCheckerboard0,  messageDDeviceCurrentCheckerboard0,
        messageLDeviceCurrentCheckerboard0,  messageRDeviceCurrentCheckerboard0,
        (U)disc_k_bp, dataAligned, bpSettingsDispVals, (U*)dstProcessing,
        checkerboardAdjustment, offsetData);
    }
    else { //checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1
      runBPIterationUpdateMsgVals<T, U>(xVal, yVal, currentLevelProperties,
        messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
        messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
        dataCostStereoCheckerboard1,
        messageUDeviceCurrentCheckerboard1,  messageDDeviceCurrentCheckerboard1,
        messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
        (U)disc_k_bp, dataAligned, bpSettingsDispVals, (U*)dstProcessing,
        checkerboardAdjustment, offsetData);
    }
  }
}

//kernel to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernel works from the point of view of the pixel at the prev level that is being copied to four different places
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void copyMsgDataToNextLevelPixel(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::Checkerboard_Parts checkerboardPart, const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::levelProperties& nextLevelProperties,
  T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
  T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
  T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
  T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  const unsigned int bpSettingsDispVals)
{
  //only need to copy checkerboard 1 around "edge" since updating checkerboard 1 in first belief propagation iteration
  //(and checkerboard 0 message values are used in the iteration to update message values in checkerboard 1)
  const bool copyCheckerboard1 = (((xVal == 0) || (yVal == 0)) || (((xVal >= (currentLevelProperties.widthCheckerboardLevel_ - 2)) || (yVal >= (currentLevelProperties.heightLevel_ - 2)))));
  
  unsigned int indexCopyTo, indexCopyFrom;
  T prevValU, prevValD, prevValL, prevValR;
  const unsigned int checkerboardPartAdjustment = (checkerboardPart == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0) ? (yVal % 2) : ((yVal + 1) % 2);
  
  if constexpr (DISP_VALS > 0) {
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      indexCopyFrom = retrieveIndexInDataAndMessage(xVal, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, DISP_VALS);

      if (checkerboardPart == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0) {
        prevValU = messageUPrevStereoCheckerboard0[indexCopyFrom];
        prevValD = messageDPrevStereoCheckerboard0[indexCopyFrom];
        prevValL = messageLPrevStereoCheckerboard0[indexCopyFrom];
        prevValR = messageRPrevStereoCheckerboard0[indexCopyFrom];
      } else /*(checkerboardPart == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1)*/ {
        prevValU = messageUPrevStereoCheckerboard1[indexCopyFrom];
        prevValD = messageDPrevStereoCheckerboard1[indexCopyFrom];
        prevValL = messageLPrevStereoCheckerboard1[indexCopyFrom];
        prevValR = messageRPrevStereoCheckerboard1[indexCopyFrom];
      }

      if (GenProcessingFuncts::withinImageBounds(xVal*2 + checkerboardPartAdjustment, yVal*2, nextLevelProperties.widthCheckerboardLevel_, nextLevelProperties.heightLevel_)) {
        indexCopyTo = retrieveIndexInDataAndMessage(xVal*2 + checkerboardPartAdjustment, yVal*2,
          nextLevelProperties.paddedWidthCheckerboardLevel_, nextLevelProperties.heightLevel_,
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

      if (GenProcessingFuncts::withinImageBounds(xVal*2 + checkerboardPartAdjustment, yVal*2 + 1, nextLevelProperties.widthCheckerboardLevel_, nextLevelProperties.heightLevel_)) {
        indexCopyTo = retrieveIndexInDataAndMessage(xVal*2 + checkerboardPartAdjustment, yVal*2 + 1,
          nextLevelProperties.paddedWidthCheckerboardLevel_, nextLevelProperties.heightLevel_,
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
    for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
      indexCopyFrom = retrieveIndexInDataAndMessage(xVal, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
        currentDisparity, bpSettingsDispVals);

      if (checkerboardPart == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0) {
        prevValU = messageUPrevStereoCheckerboard0[indexCopyFrom];
        prevValD = messageDPrevStereoCheckerboard0[indexCopyFrom];
        prevValL = messageLPrevStereoCheckerboard0[indexCopyFrom];
        prevValR = messageRPrevStereoCheckerboard0[indexCopyFrom];
      } else /*(checkerboardPart == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1)*/ {
        prevValU = messageUPrevStereoCheckerboard1[indexCopyFrom];
        prevValD = messageDPrevStereoCheckerboard1[indexCopyFrom];
        prevValL = messageLPrevStereoCheckerboard1[indexCopyFrom];
        prevValR = messageRPrevStereoCheckerboard1[indexCopyFrom];
      }

      if (GenProcessingFuncts::withinImageBounds(xVal*2 + checkerboardPartAdjustment, yVal*2, nextLevelProperties.widthCheckerboardLevel_, nextLevelProperties.heightLevel_)) {
        indexCopyTo = retrieveIndexInDataAndMessage(xVal*2 + checkerboardPartAdjustment, yVal*2,
          nextLevelProperties.paddedWidthCheckerboardLevel_, nextLevelProperties.heightLevel_,
          currentDisparity, bpSettingsDispVals);

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

      if (GenProcessingFuncts::withinImageBounds(xVal*2 + checkerboardPartAdjustment, yVal*2 + 1, nextLevelProperties.widthCheckerboardLevel_, nextLevelProperties.heightLevel_)) {
        indexCopyTo = retrieveIndexInDataAndMessage(xVal*2 + checkerboardPartAdjustment, yVal*2 + 1,
          nextLevelProperties.paddedWidthCheckerboardLevel_, nextLevelProperties.heightLevel_,
          currentDisparity, bpSettingsDispVals);

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
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUPrevStereoCheckerboard0,  T* messageDPrevStereoCheckerboard0,
  T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
  T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
  T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  const unsigned int xValInCheckerboardPart = xVal;

  //first processing from first part of checkerboard

  //adjustment based on checkerboard; need to add 1 to x for odd-numbered rows for final index mapping into disparity images for checkerboard 1
  unsigned int checkerboardPartAdjustment = (yVal % 2);

  if (GenProcessingFuncts::withinImageBounds(xValInCheckerboardPart*2 + checkerboardPartAdjustment, yVal, currentLevelProperties.widthLevel_, currentLevelProperties.heightLevel_)) {
    if ((xValInCheckerboardPart >= (1 - checkerboardPartAdjustment)) &&
        (xValInCheckerboardPart < (currentLevelProperties.widthCheckerboardLevel_ - checkerboardPartAdjustment)) &&
        (yVal > 0u) && (yVal < (currentLevelProperties.heightLevel_ - 1u)))
    {
      // keep track of "best" disparity for current pixel
      unsigned int bestDisparity{0u};
      U best_val{(U)bp_consts::INF_BP};
      if constexpr (DISP_VALS > 0) {
        for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
          const U val =
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal + 1u),
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              DISP_VALS)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal - 1u),
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              DISP_VALS)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart  + checkerboardPartAdjustment), yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              DISP_VALS)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart + checkerboardPartAdjustment) - 1u, yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              DISP_VALS)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              DISP_VALS)]);
          if (val < best_val) {
            best_val = val;
            bestDisparity = currentDisparity;
          }
        }
      }
      else {
        for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
          const U val =
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal + 1u),
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              bpSettingsDispVals)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal - 1u),
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              bpSettingsDispVals)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart  + checkerboardPartAdjustment), yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              bpSettingsDispVals)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart + checkerboardPartAdjustment) - 1u, yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              bpSettingsDispVals)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              bpSettingsDispVals)]);
          if (val < best_val) {
            best_val = val;
            bestDisparity = currentDisparity;
          }
        }
      }

      disparityBetweenImagesDevice[yVal*currentLevelProperties.widthLevel_ +
        (xValInCheckerboardPart * 2 + checkerboardPartAdjustment)] = bestDisparity;
    } else {
      disparityBetweenImagesDevice[yVal* currentLevelProperties.widthLevel_ +
        (xValInCheckerboardPart * 2 + checkerboardPartAdjustment)] = 0;
    }
  }

  //process from part 2 of checkerboard
  //adjustment based on checkerboard; need to add 1 to x for even-numbered rows for final index mapping into disparity images for checkerboard 2
  checkerboardPartAdjustment = ((yVal + 1u) % 2);

  if (GenProcessingFuncts::withinImageBounds(xValInCheckerboardPart*2 + checkerboardPartAdjustment, yVal, currentLevelProperties.widthLevel_, currentLevelProperties.heightLevel_)) {
    if ((xValInCheckerboardPart >= (1 - checkerboardPartAdjustment)) &&
        (xValInCheckerboardPart < (currentLevelProperties.widthCheckerboardLevel_ - checkerboardPartAdjustment)) &&
        (yVal > 0) && (yVal < (currentLevelProperties.heightLevel_ - 1)))
    {
      // keep track of "best" disparity for current pixel
      unsigned int bestDisparity{0u};
      U best_val{(U)bp_consts::INF_BP};
      if constexpr (DISP_VALS > 0) {
        for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
          const U val = 
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal + 1u),
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              DISP_VALS)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal - 1u),
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              DISP_VALS)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart  + checkerboardPartAdjustment), yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              DISP_VALS)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart + checkerboardPartAdjustment) - 1u,  yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              DISP_VALS)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              DISP_VALS)]);
          if (val < best_val) {
            best_val = val;
            bestDisparity = currentDisparity;
          }
        }
      }
      else {
        for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
          const U val =
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal + 1u),
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              bpSettingsDispVals)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, (yVal - 1u),
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              bpSettingsDispVals)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart  + checkerboardPartAdjustment), yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              bpSettingsDispVals)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
              (xValInCheckerboardPart + checkerboardPartAdjustment) - 1u,  yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              bpSettingsDispVals)]) +
            GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
              xValInCheckerboardPart, yVal,
              currentLevelProperties.paddedWidthCheckerboardLevel_,
              currentLevelProperties.heightLevel_,
              currentDisparity,
              bpSettingsDispVals)]);
          if (val < best_val) {
            best_val = val;
            bestDisparity = currentDisparity;
          }
        }
      }

      disparityBetweenImagesDevice[yVal * currentLevelProperties.widthLevel_ +
        (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = bestDisparity;
    } else {
      disparityBetweenImagesDevice[yVal * currentLevelProperties.widthLevel_ +
        (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = 0;
    }
  }
}

template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void printDataAndMessageValsAtPointKernel(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  const unsigned int bpSettingsDispVals = 0)
{
  if (((xVal + yVal) % 2) == 0) {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
    }
  } else {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
    }
  }
}

template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void printDataAndMessageValsToPointKernel(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  const unsigned int bpSettingsDispVals = 0)
{
  const unsigned int checkerboardAdjustment = (((xVal + yVal) % 2) == 0) ? ((yVal)%2) : ((yVal+1)%2);
  if (((xVal + yVal) % 2) == 0) {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal + 1, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal - 1, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2 + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
          (xVal / 2 - 1) + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
           currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
    }
  } else {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal + 1, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2, yVal - 1, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          xVal / 2 + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
          (xVal / 2 - 1) + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
          xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
          currentDisparity, DISP_VALS)]);
    }
  }
}

}

#endif /* SHAREDBPPROCESSINGFUNCTS_H_ */
