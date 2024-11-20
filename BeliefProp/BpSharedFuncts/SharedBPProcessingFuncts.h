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
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include "BpRunProcessing/BpStructsAndEnums.h"
#include "RunImp/RunImpGenFuncts.h"

namespace beliefprop {

//retrieve the current 1-D index value of the given point at the given disparity in the data cost and message data
ARCHITECTURE_ADDITION inline unsigned int RetrieveIndexInDataAndMessage(unsigned int x_val, unsigned int y_val,
  unsigned int width, unsigned int height, unsigned int current_disparity, unsigned int total_num_disp_vals,
  unsigned int offsetData = 0u)
{
  if constexpr (beliefprop::kOptimizedIndexingSetting) {
    //indexing is performed in such a way so that the memory accesses as coalesced as much as possible
    return (y_val * width * total_num_disp_vals + width * current_disparity + x_val) + offsetData;
  }
  else {
    return ((y_val * width + x_val) * total_num_disp_vals + current_disparity);
  }
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method
//(see "Efficient Belief Propagation for Early Vision")
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void DtStereo(T f[DISP_VALS])
{
  T prev;
  for (unsigned int current_disparity = 1; current_disparity < DISP_VALS; current_disparity++)
  {
    prev = f[current_disparity-1] + (T)1.0;
    if (prev < f[current_disparity])
      f[current_disparity] = prev;
  }

  for (int current_disparity = (int)DISP_VALS - 2; current_disparity >= 0; current_disparity--)
  {
    prev = f[current_disparity + 1] + (T)1.0;
    if (prev < f[current_disparity])
      f[current_disparity] = prev;
  }
}

template<RunData_t T>
ARCHITECTURE_ADDITION inline void DtStereo(T* f, unsigned int bp_settings_disp_vals)
{
  T prev;
  for (unsigned int current_disparity = 1; current_disparity < bp_settings_disp_vals; current_disparity++)
  {
    prev = f[current_disparity-1] + (T)1.0;
    if (prev < f[current_disparity])
      f[current_disparity] = prev;
  }

  for (int current_disparity = (int)bp_settings_disp_vals - 2; current_disparity >= 0; current_disparity--)
  {
    prev = f[current_disparity + 1] + (T)1.0;
    if (prev < f[current_disparity])
      f[current_disparity] = prev;
  }
}

template<RunData_t T>
ARCHITECTURE_ADDITION inline void DtStereo(T* f, unsigned int bp_settings_disp_vals,
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level)
{
  unsigned int fArrayIndexDisp0 = RetrieveIndexInDataAndMessage(x_val, y_val,
    current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, 0,
    bp_settings_disp_vals);
  unsigned int currFArrayIndexLast = fArrayIndexDisp0;
  unsigned int currFArrayIndex = fArrayIndexDisp0;
  T prev;

  for (unsigned int current_disparity = 1; current_disparity < bp_settings_disp_vals; current_disparity++)
  {
    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      currFArrayIndex += current_bp_level.padded_width_checkerboard_level_;
    }
    else {
      currFArrayIndex++;
    }

    prev = f[currFArrayIndexLast] + (T)1.0;
    if (prev < f[currFArrayIndex])
      f[currFArrayIndex] = prev;
    currFArrayIndexLast = currFArrayIndex;
  }

  for (int current_disparity = (int)bp_settings_disp_vals - 2; current_disparity >= 0; current_disparity--)
  {
    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      currFArrayIndex -= current_bp_level.padded_width_checkerboard_level_;
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
ARCHITECTURE_ADDITION inline void MsgStereo(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const U messageValsNeighbor1[DISP_VALS], const U messageValsNeighbor2[DISP_VALS],
  const U messageValsNeighbor3[DISP_VALS], const U data_costs[DISP_VALS],
  T* dst_message_array, U disc_k_bp, bool data_aligned)
{
  // aggregate and find min
  U minimum{(U)bp_consts::kInfBp};
  U dst[DISP_VALS];

  for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++)
  {
    dst[current_disparity] = messageValsNeighbor1[current_disparity] + messageValsNeighbor2[current_disparity] +
                             messageValsNeighbor3[current_disparity] + data_costs[current_disparity];
    if (dst[current_disparity] < minimum)
      minimum = dst[current_disparity];
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method
  //(see "Efficient Belief Propagation for Early Vision")
  DtStereo<U, DISP_VALS>(dst);

  // truncate
  minimum += disc_k_bp;

  // normalize
  U val_to_normalize{(U)0.0};

  for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
    if (minimum < dst[current_disparity]) {
      dst[current_disparity] = minimum;
    }

    val_to_normalize += dst[current_disparity];
  }

  val_to_normalize /= ((U)DISP_VALS);

  int dest_message_array_index = RetrieveIndexInDataAndMessage(x_val, y_val,
    current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, 0,
    DISP_VALS);

  for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
    dst[current_disparity] -= val_to_normalize;
    dst_message_array[dest_message_array_index] =
      run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(dst[current_disparity]);
    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      dest_message_array_index += current_bp_level.padded_width_checkerboard_level_;
    }
    else {
      dest_message_array_index++;
    }
  }
}

template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline void MsgStereo(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const U* messageValsNeighbor1, const U* messageValsNeighbor2,
  const U* messageValsNeighbor3, const U* data_costs,
  T* dst_message_array, U disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  // aggregate and find min
  U minimum{(U)bp_consts::kInfBp};
  U* dst = new U[bp_settings_disp_vals];

  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++)
  {
    dst[current_disparity] = messageValsNeighbor1[current_disparity] + messageValsNeighbor2[current_disparity] +
                             messageValsNeighbor3[current_disparity] + data_costs[current_disparity];
    if (dst[current_disparity] < minimum)
      minimum = dst[current_disparity];
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method
  //(see "Efficient Belief Propagation for Early Vision")
  DtStereo<U>(dst, bp_settings_disp_vals);

  // truncate
  minimum += disc_k_bp;

  // normalize
  U val_to_normalize{(U)0.0};

  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
    if (minimum < dst[current_disparity]) {
      dst[current_disparity] = minimum;
    }

    val_to_normalize += dst[current_disparity];
  }

  val_to_normalize /= ((U)bp_settings_disp_vals);

  int dest_message_array_index = RetrieveIndexInDataAndMessage(x_val, y_val,
    current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, 0,
    bp_settings_disp_vals);

  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
    dst[current_disparity] -= val_to_normalize;
    dst_message_array[dest_message_array_index] =
      run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(dst[current_disparity]);
    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      dest_message_array_index += current_bp_level.padded_width_checkerboard_level_;
    }
    else {
      dest_message_array_index++;
    }
  }

  delete [] dst;
}

template<RunData_t T, RunData_t U, beliefprop::MessageComp M>
ARCHITECTURE_ADDITION void inline SetInitDstProcessing(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const T* prev_u_message_array, const T* prev_d_message_array,
  const T* prev_l_message_array, const T* prev_r_message_array,
  const T* data_message_array, T* dst_message_array,
  U disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals,
  U* dstProcessing, unsigned int checkerboard_adjustment,
  unsigned int offsetData, unsigned int current_disparity,
  unsigned int procArrIdx)
{
  const U dataVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_message_array[
    RetrieveIndexInDataAndMessage(x_val, y_val,
      current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
      current_disparity, bp_settings_disp_vals, offsetData)]);

  if constexpr (M == beliefprop::MessageComp::kUMessage) {
    const U prevUVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prev_u_message_array[
      RetrieveIndexInDataAndMessage(x_val, (y_val+1),
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)]);
    const U prevLVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prev_l_message_array[
      RetrieveIndexInDataAndMessage((x_val + checkerboard_adjustment), y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)]);
    const U prevRVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prev_r_message_array[
      RetrieveIndexInDataAndMessage(((x_val + checkerboard_adjustment) - 1), y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)]);

    dstProcessing[procArrIdx] = prevUVal + prevLVal + prevRVal + dataVal;
  }
  else if constexpr (M == beliefprop::MessageComp::kDMessage) {
    const U prevDVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prev_d_message_array[
      RetrieveIndexInDataAndMessage(x_val, (y_val-1),
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)]);
    const U prevLVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prev_l_message_array[
      RetrieveIndexInDataAndMessage((x_val + checkerboard_adjustment), y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)]);
    const U prevRVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prev_r_message_array[
      RetrieveIndexInDataAndMessage(((x_val + checkerboard_adjustment) - 1), y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)]);

    dstProcessing[procArrIdx] = prevDVal + prevLVal + prevRVal + dataVal;
  }
  else if constexpr (M == beliefprop::MessageComp::kLMessage) {
    const U prevUVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prev_u_message_array[
      RetrieveIndexInDataAndMessage(x_val, (y_val+1),
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)]);
    const U prevDVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prev_d_message_array[
      RetrieveIndexInDataAndMessage(x_val, (y_val-1),
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)]);
    const U prevLVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prev_l_message_array[
      RetrieveIndexInDataAndMessage((x_val + checkerboard_adjustment), y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)]);

    dstProcessing[procArrIdx] = prevUVal + prevDVal + prevLVal + dataVal;
  }
  else if constexpr (M == beliefprop::MessageComp::kRMessage) {
    const U prevUVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prev_u_message_array[
      RetrieveIndexInDataAndMessage(x_val, (y_val+1),
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)]);
    const U prevDVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prev_d_message_array[
      RetrieveIndexInDataAndMessage(x_val, (y_val-1),
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)]);
    const U prevRVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(prev_r_message_array[
      RetrieveIndexInDataAndMessage(((x_val + checkerboard_adjustment) - 1), y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)]);

    dstProcessing[procArrIdx] = prevUVal + prevDVal + prevRVal + dataVal;
  }
}

//TODO: may need to specialize for half-precision to account for possible NaN/inf vals
template<RunData_t T, RunData_t U, beliefprop::MessageComp M>
ARCHITECTURE_ADDITION inline void MsgStereo(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const T* prev_u_message_array, const T* prev_d_message_array,
  const T* prev_l_message_array, const T* prev_r_message_array,
  const T* data_message_array, T* dst_message_array,
  U disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals,
  U* dstProcessing, unsigned int checkerboard_adjustment,
  unsigned int offsetData)
{
  // aggregate and find min
  U minimum{(U)bp_consts::kInfBp};
  unsigned int processingArrIndexDisp0 = RetrieveIndexInDataAndMessage(x_val, y_val,
    current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, 0,
    bp_settings_disp_vals);
  unsigned int procArrIdx{processingArrIndexDisp0};

  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++)
  {
    //set initial dst processing array value corresponding to disparity for M message type
    SetInitDstProcessing<T, U, M>(x_val, y_val, current_bp_level, prev_u_message_array, prev_d_message_array,
      prev_l_message_array, prev_r_message_array, data_message_array, dst_message_array,
      disc_k_bp, data_aligned, bp_settings_disp_vals, dstProcessing, checkerboard_adjustment,
      offsetData, current_disparity, procArrIdx);

    if (dstProcessing[procArrIdx] < minimum)
      minimum = dstProcessing[procArrIdx];

    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      procArrIdx += current_bp_level.padded_width_checkerboard_level_;
    }
    else {
      procArrIdx++;
    }
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method
  //(see "Efficient Belief Propagation for Early Vision")
  DtStereo<U>(dstProcessing, bp_settings_disp_vals, x_val, y_val, current_bp_level);

  // truncate
  minimum += disc_k_bp;

  // normalize
  U val_to_normalize{(U)0.0};

  procArrIdx = processingArrIndexDisp0;
  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
    if (minimum < dstProcessing[procArrIdx]) {
      dstProcessing[procArrIdx] = minimum;
    }

    val_to_normalize += dstProcessing[procArrIdx];

    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      procArrIdx += current_bp_level.padded_width_checkerboard_level_;
    }
    else {
      procArrIdx++;
    }
  }

  val_to_normalize /= ((U)bp_settings_disp_vals);

  //dst processing index and message array index are the same for each disparity value in this processing
  procArrIdx = processingArrIndexDisp0;

  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
    dstProcessing[procArrIdx] -= val_to_normalize;
    dst_message_array[procArrIdx] =
      run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(dstProcessing[procArrIdx]);
    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      procArrIdx += current_bp_level.padded_width_checkerboard_level_;
    }
    else {
      procArrIdx++;
    }
  }
}

//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void InitializeBottomLevelDataPixel(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float* image_1_pixels_device, const float* image_2_pixels_device,
  T* dataCostDeviceStereoCheckerboard0,
  T* dataCostDeviceStereoCheckerboard1, 
  float lambda_bp, float data_k_bp, unsigned int bp_settings_disp_vals)
{
  if constexpr (DISP_VALS > 0) {
    unsigned int index_val;
    const unsigned int xInCheckerboard = x_val / 2;

    if (run_imp_util::WithinImageBounds(
      xInCheckerboard, y_val, current_bp_level.width_checkerboard_level_, current_bp_level.height_level_)) {
      //make sure that it is possible to check every disparity value
      //need to cast DISP_VALS from unsigned int to int
      //for conditional to work as expected
      if (((int)x_val - ((int)DISP_VALS - 1)) >= 0) {
        for (unsigned int current_disparity = 0u; current_disparity < DISP_VALS; current_disparity++) {
          float currentPixelImage1 = 0.0f, currentPixelImage2 = 0.0f;

          if (run_imp_util::WithinImageBounds(x_val, y_val, current_bp_level.width_level_, current_bp_level.height_level_)) {
            currentPixelImage1 = image_1_pixels_device[y_val * current_bp_level.width_level_ + x_val];
            currentPixelImage2 = image_2_pixels_device[y_val * current_bp_level.width_level_ + (x_val - current_disparity)];
          }

          index_val = RetrieveIndexInDataAndMessage(xInCheckerboard, y_val,
            current_bp_level.padded_width_checkerboard_level_,
            current_bp_level.height_level_, current_disparity,
            DISP_VALS);

          //data cost is equal to dataWeight value for weighting times the absolute difference
          //in corresponding pixel intensity values capped at dataCostCap
          if (((x_val + y_val) % 2) == 0) {
            dataCostDeviceStereoCheckerboard0[index_val] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * run_imp_util::GetMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
          }
          else {
            dataCostDeviceStereoCheckerboard1[index_val] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * run_imp_util::GetMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
          }
        }
      } else {
        for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
          index_val = RetrieveIndexInDataAndMessage(xInCheckerboard, y_val,
            current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
            current_disparity, DISP_VALS);

          //set data cost to zero if not possible to determine cost at disparity for pixel
          if (((x_val + y_val) % 2) == 0) {
            dataCostDeviceStereoCheckerboard0[index_val] = run_imp_util::ZeroVal<T>();
          }
          else {
            dataCostDeviceStereoCheckerboard1[index_val] = run_imp_util::ZeroVal<T>();
          }
        }
      }
    }
  }
  else {
    unsigned int index_val;
    const unsigned int xInCheckerboard = x_val / 2;

    if (run_imp_util::WithinImageBounds(
      xInCheckerboard, y_val, current_bp_level.width_checkerboard_level_, current_bp_level.height_level_)) {
      //make sure that it is possible to check every disparity value
      //need to cast bp_settings_disp_vals from unsigned int to int
      //for conditional to work as expected
      if (((int)x_val - ((int)bp_settings_disp_vals - 1)) >= 0) {
        for (unsigned int current_disparity = 0u; current_disparity < bp_settings_disp_vals; current_disparity++) {
          float currentPixelImage1 = 0.0f, currentPixelImage2 = 0.0f;

          if (run_imp_util::WithinImageBounds(x_val, y_val, current_bp_level.width_level_, current_bp_level.height_level_)) {
            currentPixelImage1 = image_1_pixels_device[y_val * current_bp_level.width_level_ + x_val];
            currentPixelImage2 = image_2_pixels_device[y_val * current_bp_level.width_level_ + (x_val - current_disparity)];
          }

          index_val = RetrieveIndexInDataAndMessage(xInCheckerboard, y_val,
            current_bp_level.padded_width_checkerboard_level_,
            current_bp_level.height_level_, current_disparity,
            bp_settings_disp_vals);

          //data cost is equal to dataWeight value for weighting times the absolute difference
          //in corresponding pixel intensity values capped at dataCostCap
          if (((x_val + y_val) % 2) == 0) {
            dataCostDeviceStereoCheckerboard0[index_val] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * run_imp_util::GetMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
          }
          else {
            dataCostDeviceStereoCheckerboard1[index_val] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * run_imp_util::GetMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
          }
        }
      } else {
        for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
          index_val = RetrieveIndexInDataAndMessage(xInCheckerboard, y_val,
            current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
            current_disparity, bp_settings_disp_vals);

          //set data cost to zero if not possible to determine cost at disparity for pixel
          if (((x_val + y_val) % 2) == 0) {
            dataCostDeviceStereoCheckerboard0[index_val] = run_imp_util::ZeroVal<T>();
          }
          else {
            dataCostDeviceStereoCheckerboard1[index_val] = run_imp_util::ZeroVal<T>();
          }
        }
      }
    }
  }
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void InitializeCurrentLevelDataPixel(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const T* dataCostStereoCheckerboard0, const T* dataCostStereoCheckerboard1,
  T* dataCostDeviceToWriteTo, unsigned int offset_num,
  unsigned int bp_settings_disp_vals)
{
  //add 1 or 0 to the x-value depending on checkerboard part and row
  //beliefprop::Checkerboard_Part::kCheckerboardPart0 with slot at (0, 0) has adjustment of 0 in row 0,
  //beliefprop::Checkerboard_Part::kCheckerboardPart1 with slot at (0, 1) has adjustment of 1 in row 0
  const unsigned int checkerboard_part_adjustment =
    (checkerboard_part == beliefprop::Checkerboard_Part::kCheckerboardPart0) ? (y_val % 2) : ((y_val + 1) % 2);

  //the corresponding x-values at the "lower" level depends on which checkerboard the pixel is in
  const unsigned int x_valPrev = x_val*2 + checkerboard_part_adjustment;

  if (run_imp_util::WithinImageBounds(
    x_valPrev, (y_val * 2 + 1), prev_bp_level.width_checkerboard_level_, prev_bp_level.height_level_))
  {
    if constexpr (DISP_VALS > 0) {
      for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
        const U dataCostVal =
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[RetrieveIndexInDataAndMessage(
            x_valPrev, y_val*2, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, DISP_VALS, offset_num)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[RetrieveIndexInDataAndMessage(
            x_valPrev, y_val*2, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, DISP_VALS, offset_num)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[RetrieveIndexInDataAndMessage(
            x_valPrev, y_val*2 + 1, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, DISP_VALS, offset_num)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[RetrieveIndexInDataAndMessage(
            x_valPrev, y_val*2 + 1, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, DISP_VALS, offset_num)]);

        dataCostDeviceToWriteTo[RetrieveIndexInDataAndMessage(x_val, y_val,
          current_bp_level.padded_width_checkerboard_level_,
          current_bp_level.height_level_, current_disparity,
          DISP_VALS)] =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(dataCostVal);
      }
    }
    else {
      for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
        const U dataCostVal =
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[RetrieveIndexInDataAndMessage(
            x_valPrev, y_val*2, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, bp_settings_disp_vals, offset_num)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[RetrieveIndexInDataAndMessage(
            x_valPrev, y_val*2, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, bp_settings_disp_vals, offset_num)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[RetrieveIndexInDataAndMessage(
            x_valPrev, y_val*2 + 1, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, bp_settings_disp_vals, offset_num)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[RetrieveIndexInDataAndMessage(
            x_valPrev, y_val*2 + 1, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, bp_settings_disp_vals, offset_num)]);

        dataCostDeviceToWriteTo[RetrieveIndexInDataAndMessage(x_val, y_val,
          current_bp_level.padded_width_checkerboard_level_,
          current_bp_level.height_level_, current_disparity,
          bp_settings_disp_vals)] =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(dataCostVal);
      }
    }
  }
}

//initialize the message values at each pixel of the current level to the default value
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void InitializeMessageValsToDefaultKernelPixel(
  unsigned int x_val_in_checkerboard, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int bp_settings_disp_vals)
{
  //initialize message values in both checkerboards

  if constexpr (DISP_VALS > 0) {
    //set the message value at each pixel for each disparity to 0
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      messageUDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      messageDDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      messageLDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      messageRDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
    }

    //retrieve the previous message value at each movement at each pixel
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      messageUDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      messageDDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      messageLDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      messageRDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
    }
  }
  else {
    //set the message value at each pixel for each disparity to 0
    for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
      messageUDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      messageDDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      messageLDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      messageRDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
    }

    //retrieve the previous message value at each movement at each pixel
    for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
      messageUDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      messageDDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      messageLDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      messageRDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
    }
  }
}

//device portion of the kernel function to run the current iteration of belief propagation
//where the input messages and data costs come in as array in local memory
//and the output message values are stored in local memory
template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void RunBPIterationUpdateMsgVals(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const U prev_u_message[DISP_VALS], const U prev_d_message[DISP_VALS],
  const U prev_l_message[DISP_VALS], const U prev_r_message[DISP_VALS],
  const U data_message[DISP_VALS],
  T* currentUMessageArray, T* currentDMessageArray,
  T* currentLMessageArray, T* currentRMessageArray,
  const U disc_k_bp, bool data_aligned)
{
  MsgStereo<T, U, DISP_VALS>(x_val, y_val, current_bp_level,
    prev_u_message, prev_l_message, prev_r_message, data_message,
    currentUMessageArray, disc_k_bp, data_aligned);

  MsgStereo<T, U, DISP_VALS>(x_val, y_val, current_bp_level,
    prev_d_message, prev_l_message, prev_r_message, data_message,
    currentDMessageArray, disc_k_bp, data_aligned);

  MsgStereo<T, U, DISP_VALS>(x_val, y_val, current_bp_level,
    prev_u_message, prev_d_message, prev_r_message, data_message,
    currentRMessageArray, disc_k_bp, data_aligned);

  MsgStereo<T, U, DISP_VALS>(x_val, y_val, current_bp_level,
    prev_u_message, prev_d_message, prev_l_message, data_message,
    currentLMessageArray, disc_k_bp, data_aligned);
}

template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline void RunBPIterationUpdateMsgVals(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const U* prev_u_message, const U* prev_d_message,
  const U* prev_l_message, const U* prev_r_message,
  const U* data_message,
  T* currentUMessageArray, T* currentDMessageArray,
  T* currentLMessageArray, T* currentRMessageArray,
  const U disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  MsgStereo<T, U>(x_val, y_val, current_bp_level,
    prev_u_message, prev_l_message, prev_r_message, data_message,
    currentUMessageArray, disc_k_bp, data_aligned, bp_settings_disp_vals);

  MsgStereo<T, U>(x_val, y_val, current_bp_level,
    prev_d_message, prev_l_message, prev_r_message, data_message,
    currentDMessageArray, disc_k_bp, data_aligned, bp_settings_disp_vals);

  MsgStereo<T, U>(x_val, y_val, current_bp_level,
    prev_u_message, prev_d_message, prev_r_message, data_message,
    currentRMessageArray, disc_k_bp, data_aligned, bp_settings_disp_vals);

  MsgStereo<T, U>(x_val, y_val, current_bp_level,
    prev_u_message, prev_d_message, prev_l_message, data_message,
    currentLMessageArray, disc_k_bp, data_aligned, bp_settings_disp_vals);
}

template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline void RunBPIterationUpdateMsgVals(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const T* prev_u_message_array, const T* prev_d_message_array,
  const T* prev_l_message_array, const T* prev_r_message_array,
  const T* data_message_array,
  T* currentUMessageArray, T* currentDMessageArray,
  T* currentLMessageArray, T* currentRMessageArray,
  const U disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals,
  U* dstProcessing, unsigned int checkerboard_adjustment,
  unsigned int offsetData)
{
  MsgStereo<T, U, beliefprop::MessageComp::kUMessage>(x_val, y_val, current_bp_level,
    prev_u_message_array, prev_d_message_array, prev_l_message_array, prev_r_message_array, data_message_array,
    currentUMessageArray, disc_k_bp, data_aligned, bp_settings_disp_vals,
    dstProcessing, checkerboard_adjustment, offsetData);

  MsgStereo<T, U, beliefprop::MessageComp::kDMessage>(x_val, y_val, current_bp_level,
    prev_u_message_array, prev_d_message_array, prev_l_message_array, prev_r_message_array, data_message_array,
    currentDMessageArray, disc_k_bp, data_aligned, bp_settings_disp_vals,
    dstProcessing, checkerboard_adjustment, offsetData);

  MsgStereo<T, U, beliefprop::MessageComp::kLMessage>(x_val, y_val, current_bp_level,
    prev_u_message_array, prev_d_message_array, prev_l_message_array, prev_r_message_array, data_message_array,
    currentLMessageArray, disc_k_bp, data_aligned, bp_settings_disp_vals,
    dstProcessing, checkerboard_adjustment, offsetData);

  MsgStereo<T, U, beliefprop::MessageComp::kRMessage>(x_val, y_val, current_bp_level,
    prev_u_message_array, prev_d_message_array, prev_l_message_array, prev_r_message_array, data_message_array,
    currentRMessageArray, disc_k_bp, data_aligned, bp_settings_disp_vals,
    dstProcessing, checkerboard_adjustment, offsetData);
}

//device portion of the kernel function to run the current iteration of belief propagation in parallel
//using the checkerboard update method where half the pixels in the "checkerboard" scheme retrieve messages
//from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void RunBPIterationUsingCheckerboardUpdatesKernel(
  unsigned int x_val, unsigned int y_val,
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const T* dataCostStereoCheckerboard0, const T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned,
  unsigned int bp_settings_disp_vals)
{
  //checkerboard_adjustment used for indexing into current checkerboard to update
  const unsigned int checkerboard_adjustment =
    (checkerboard_to_update == beliefprop::Checkerboard_Part::kCheckerboardPart0) ? ((y_val)%2) : ((y_val+1)%2);

  //may want to look into (x_val < (width_level_checkerboard_part - 1) since it may affect the edges
  //make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
  if ((x_val >= (1u - checkerboard_adjustment)) && 
      (x_val < (current_bp_level.width_checkerboard_level_ - checkerboard_adjustment)) &&
      (y_val > 0) && (y_val < (current_bp_level.height_level_ - 1u)))
  {
    if constexpr (DISP_VALS > 0) {
      U data_message[DISP_VALS], prev_u_message[DISP_VALS], prev_d_message[DISP_VALS], 
        prev_l_message[DISP_VALS], prev_r_message[DISP_VALS];

      for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
        if (checkerboard_to_update == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
          data_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            dataCostStereoCheckerboard0[RetrieveIndexInDataAndMessage(x_val, y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS, offsetData)]);
          prev_u_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageUDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(x_val, (y_val+1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
          prev_d_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageDDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(x_val, (y_val-1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
          prev_l_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageLDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage((x_val + checkerboard_adjustment), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
          prev_r_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageRDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(((x_val + checkerboard_adjustment) - 1), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
        }
        else { //checkerboard_to_update == beliefprop::Checkerboard_Part::kCheckerboardPart1
          data_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            dataCostStereoCheckerboard1[RetrieveIndexInDataAndMessage(x_val, y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS, offsetData)]);
          prev_u_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageUDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(x_val, (y_val+1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
          prev_d_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageDDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(x_val, (y_val-1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
          prev_l_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageLDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage((x_val + checkerboard_adjustment), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
          prev_r_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageRDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(((x_val + checkerboard_adjustment) - 1), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
        }
      }

      //uses the previous message values and data cost to calculate the current message values and store the results
      if (checkerboard_to_update == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
        RunBPIterationUpdateMsgVals<T, U, DISP_VALS>(x_val, y_val, current_bp_level,
          prev_u_message, prev_d_message, prev_l_message, prev_r_message, data_message,
          messageUDeviceCurrentCheckerboard0,  messageDDeviceCurrentCheckerboard0,
          messageLDeviceCurrentCheckerboard0,  messageRDeviceCurrentCheckerboard0,
          (U)disc_k_bp, data_aligned);
      }
      else { //checkerboard_to_update == beliefprop::Checkerboard_Part::kCheckerboardPart1
        RunBPIterationUpdateMsgVals<T, U, DISP_VALS>(x_val, y_val, current_bp_level,
          prev_u_message, prev_d_message, prev_l_message, prev_r_message, data_message,
          messageUDeviceCurrentCheckerboard1,  messageDDeviceCurrentCheckerboard1,
          messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
          (U)disc_k_bp, data_aligned);
      }
    }
    else {
      U* data_message = new U[bp_settings_disp_vals];
      U* prev_u_message = new U[bp_settings_disp_vals];
      U* prev_d_message = new U[bp_settings_disp_vals];
      U* prev_l_message = new U[bp_settings_disp_vals];
      U* prev_r_message = new U[bp_settings_disp_vals];

      for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
        if (checkerboard_to_update == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
          data_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            dataCostStereoCheckerboard0[RetrieveIndexInDataAndMessage(x_val, y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals, offsetData)]);
          prev_u_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageUDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(x_val, (y_val+1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
          prev_d_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageDDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(x_val, (y_val-1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
          prev_l_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageLDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage((x_val + checkerboard_adjustment), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
          prev_r_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageRDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(((x_val + checkerboard_adjustment) - 1), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
        }
        else { //checkerboard_to_update == beliefprop::Checkerboard_Part::kCheckerboardPart1
          data_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            dataCostStereoCheckerboard1[RetrieveIndexInDataAndMessage(x_val, y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals, offsetData)]);
          prev_u_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageUDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(x_val, (y_val+1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
          prev_d_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageDDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(x_val, (y_val-1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
          prev_l_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageLDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage((x_val + checkerboard_adjustment), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
          prev_r_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            messageRDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(((x_val + checkerboard_adjustment) - 1), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
        }
      }

      //uses the previous message values and data cost to calculate the current message values and store the results
      if (checkerboard_to_update == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
        RunBPIterationUpdateMsgVals<T, U>(x_val, y_val, current_bp_level,
          prev_u_message, prev_d_message, prev_l_message, prev_r_message, data_message,
          messageUDeviceCurrentCheckerboard0,  messageDDeviceCurrentCheckerboard0,
          messageLDeviceCurrentCheckerboard0,  messageRDeviceCurrentCheckerboard0,
          (U)disc_k_bp, data_aligned, bp_settings_disp_vals);
      }
      else { //checkerboard_to_update == beliefprop::Checkerboard_Part::kCheckerboardPart1
        RunBPIterationUpdateMsgVals<T, U>(x_val, y_val, current_bp_level,
          prev_u_message, prev_d_message, prev_l_message, prev_r_message, data_message,
          messageUDeviceCurrentCheckerboard1,  messageDDeviceCurrentCheckerboard1,
          messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
          (U)disc_k_bp, data_aligned, bp_settings_disp_vals);
      }

      delete [] data_message;
      delete [] prev_u_message;
      delete [] prev_d_message;
      delete [] prev_l_message;
      delete [] prev_r_message;
    }
  }
}

template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void RunBPIterationUsingCheckerboardUpdatesKernel(
  unsigned int x_val, unsigned int y_val,
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const T* dataCostStereoCheckerboard0, const T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned,
  unsigned int bp_settings_disp_vals, void* dstProcessing)
{
  //checkerboard_adjustment used for indexing into current checkerboard to update
  const unsigned int checkerboard_adjustment = 
    (checkerboard_to_update == beliefprop::Checkerboard_Part::kCheckerboardPart0) ? ((y_val)%2) : ((y_val+1)%2);

  //may want to look into (x_val < (width_level_checkerboard_part - 1) since it may affect the edges
  //make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
  if ((x_val >= (1u - checkerboard_adjustment)) &&
      (x_val < (current_bp_level.width_checkerboard_level_ - checkerboard_adjustment)) &&
      (y_val > 0) && (y_val < (current_bp_level.height_level_ - 1u)))
  {
    //uses the previous message values and data cost to calculate the current message values and store the results
    if (checkerboard_to_update == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
      RunBPIterationUpdateMsgVals<T, U>(x_val, y_val, current_bp_level,
        messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
        messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
        dataCostStereoCheckerboard0,
        messageUDeviceCurrentCheckerboard0,  messageDDeviceCurrentCheckerboard0,
        messageLDeviceCurrentCheckerboard0,  messageRDeviceCurrentCheckerboard0,
        (U)disc_k_bp, data_aligned, bp_settings_disp_vals, (U*)dstProcessing,
        checkerboard_adjustment, offsetData);
    }
    else { //checkerboard_to_update == beliefprop::Checkerboard_Part::kCheckerboardPart1
      RunBPIterationUpdateMsgVals<T, U>(x_val, y_val, current_bp_level,
        messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
        messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
        dataCostStereoCheckerboard1,
        messageUDeviceCurrentCheckerboard1,  messageDDeviceCurrentCheckerboard1,
        messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
        (U)disc_k_bp, data_aligned, bp_settings_disp_vals, (U*)dstProcessing,
        checkerboard_adjustment, offsetData);
    }
  }
}

//kernel to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernel works from the point of view of the pixel at the prev level that is being copied to four different places
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void CopyMsgDataToNextLevelPixel(
  unsigned int x_val, unsigned int y_val,
  beliefprop::Checkerboard_Part checkerboard_part, const beliefprop::BpLevelProperties& current_bp_level,
  const beliefprop::BpLevelProperties& next_bp_level,
  const T* messageUPrevStereoCheckerboard0, const T* messageDPrevStereoCheckerboard0,
  const T* messageLPrevStereoCheckerboard0, const T* messageRPrevStereoCheckerboard0,
  const T* messageUPrevStereoCheckerboard1, const T* messageDPrevStereoCheckerboard1,
  const T* messageLPrevStereoCheckerboard1, const T* messageRPrevStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int bp_settings_disp_vals)
{
  //only need to copy checkerboard 1 around "edge" since updating checkerboard 1 in first belief propagation iteration
  //(and checkerboard 0 message values are used in the iteration to update message values in checkerboard 1)
  const bool copyCheckerboard1 = (((x_val == 0) || (y_val == 0)) || 
    (((x_val >= (current_bp_level.width_checkerboard_level_ - 2)) ||
      (y_val >= (current_bp_level.height_level_ - 2)))));
  
  unsigned int indexCopyTo, indexCopyFrom;
  T prevValU, prevValD, prevValL, prevValR;
  const unsigned int checkerboard_part_adjustment =
    (checkerboard_part == beliefprop::Checkerboard_Part::kCheckerboardPart0) ? (y_val % 2) : ((y_val + 1) % 2);
  
  if constexpr (DISP_VALS > 0) {
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      indexCopyFrom = RetrieveIndexInDataAndMessage(x_val, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS);

      if (checkerboard_part == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
        prevValU = messageUPrevStereoCheckerboard0[indexCopyFrom];
        prevValD = messageDPrevStereoCheckerboard0[indexCopyFrom];
        prevValL = messageLPrevStereoCheckerboard0[indexCopyFrom];
        prevValR = messageRPrevStereoCheckerboard0[indexCopyFrom];
      } else /*(checkerboard_part == beliefprop::Checkerboard_Part::kCheckerboardPart1)*/ {
        prevValU = messageUPrevStereoCheckerboard1[indexCopyFrom];
        prevValD = messageDPrevStereoCheckerboard1[indexCopyFrom];
        prevValL = messageLPrevStereoCheckerboard1[indexCopyFrom];
        prevValR = messageRPrevStereoCheckerboard1[indexCopyFrom];
      }

      if (run_imp_util::WithinImageBounds(
        x_val*2 + checkerboard_part_adjustment, y_val*2,
        next_bp_level.width_checkerboard_level_,
        next_bp_level.height_level_))
      {
        indexCopyTo = RetrieveIndexInDataAndMessage(x_val*2 + checkerboard_part_adjustment, y_val*2,
          next_bp_level.padded_width_checkerboard_level_, next_bp_level.height_level_,
          current_disparity, DISP_VALS);

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

      if (run_imp_util::WithinImageBounds(
        x_val*2 + checkerboard_part_adjustment, y_val*2 + 1,
        next_bp_level.width_checkerboard_level_, next_bp_level.height_level_))
      {
        indexCopyTo = RetrieveIndexInDataAndMessage(x_val*2 + checkerboard_part_adjustment, y_val*2 + 1,
          next_bp_level.padded_width_checkerboard_level_, next_bp_level.height_level_,
          current_disparity, DISP_VALS);

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
    for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
      indexCopyFrom = RetrieveIndexInDataAndMessage(x_val, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals);

      if (checkerboard_part == beliefprop::Checkerboard_Part::kCheckerboardPart0) {
        prevValU = messageUPrevStereoCheckerboard0[indexCopyFrom];
        prevValD = messageDPrevStereoCheckerboard0[indexCopyFrom];
        prevValL = messageLPrevStereoCheckerboard0[indexCopyFrom];
        prevValR = messageRPrevStereoCheckerboard0[indexCopyFrom];
      } else /*(checkerboard_part == beliefprop::Checkerboard_Part::kCheckerboardPart1)*/ {
        prevValU = messageUPrevStereoCheckerboard1[indexCopyFrom];
        prevValD = messageDPrevStereoCheckerboard1[indexCopyFrom];
        prevValL = messageLPrevStereoCheckerboard1[indexCopyFrom];
        prevValR = messageRPrevStereoCheckerboard1[indexCopyFrom];
      }

      if (run_imp_util::WithinImageBounds(
        x_val*2 + checkerboard_part_adjustment, y_val*2,
        next_bp_level.width_checkerboard_level_, next_bp_level.height_level_))
      {
        indexCopyTo = RetrieveIndexInDataAndMessage(x_val*2 + checkerboard_part_adjustment, y_val*2,
          next_bp_level.padded_width_checkerboard_level_, next_bp_level.height_level_,
          current_disparity, bp_settings_disp_vals);

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

      if (run_imp_util::WithinImageBounds(
        x_val*2 + checkerboard_part_adjustment, y_val*2 + 1,
        next_bp_level.width_checkerboard_level_, next_bp_level.height_level_))
      {
        indexCopyTo = RetrieveIndexInDataAndMessage(x_val*2 + checkerboard_part_adjustment, y_val*2 + 1,
          next_bp_level.padded_width_checkerboard_level_, next_bp_level.height_level_,
          current_disparity, bp_settings_disp_vals);

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
ARCHITECTURE_ADDITION inline void RetrieveOutputDisparityPixel(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const T* dataCostStereoCheckerboard0, const T* dataCostStereoCheckerboard1,
  const T* messageUPrevStereoCheckerboard0, const T* messageDPrevStereoCheckerboard0,
  const T* messageLPrevStereoCheckerboard0, const T* messageRPrevStereoCheckerboard0,
  const T* messageUPrevStereoCheckerboard1, const T* messageDPrevStereoCheckerboard1,
  const T* messageLPrevStereoCheckerboard1, const T* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  const unsigned int x_val_in_checkerboard_part = x_val;

  //first processing from first part of checkerboard

  //adjustment based on checkerboard; need to add 1 to x for odd-numbered rows
  //for final index mapping into disparity images for checkerboard 1
  unsigned int checkerboard_part_adjustment = (y_val % 2);

  if (run_imp_util::WithinImageBounds(
    x_val_in_checkerboard_part*2 + checkerboard_part_adjustment, y_val,
    current_bp_level.width_level_, current_bp_level.height_level_))
  {
    if ((x_val_in_checkerboard_part >= (1 - checkerboard_part_adjustment)) &&
        (x_val_in_checkerboard_part < (current_bp_level.width_checkerboard_level_ - checkerboard_part_adjustment)) &&
        (y_val > 0u) && (y_val < (current_bp_level.height_level_ - 1u)))
    {
      // keep track of "best" disparity for current pixel
      unsigned int bestDisparity{0u};
      U best_val{(U)bp_consts::kInfBp};
      if constexpr (DISP_VALS > 0) {
        for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
          const U val =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard1[
              RetrieveIndexInDataAndMessage(
                x_val_in_checkerboard_part, (y_val + 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard1[
              RetrieveIndexInDataAndMessage(
                x_val_in_checkerboard_part, (y_val - 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard1[
              RetrieveIndexInDataAndMessage(
                (x_val_in_checkerboard_part  + checkerboard_part_adjustment), y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard1[
              RetrieveIndexInDataAndMessage(
                (x_val_in_checkerboard_part + checkerboard_part_adjustment) - 1u, y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[
              RetrieveIndexInDataAndMessage(
                x_val_in_checkerboard_part, y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]);
          if (val < best_val) {
            best_val = val;
            bestDisparity = current_disparity;
          }
        }
      }
      else {
        for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
          const U val =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard1[
              RetrieveIndexInDataAndMessage(
                x_val_in_checkerboard_part, (y_val + 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard1[
              RetrieveIndexInDataAndMessage(
                x_val_in_checkerboard_part, (y_val - 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard1[
              RetrieveIndexInDataAndMessage(
                (x_val_in_checkerboard_part  + checkerboard_part_adjustment), y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard1[
              RetrieveIndexInDataAndMessage(
                (x_val_in_checkerboard_part + checkerboard_part_adjustment) - 1u, y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[
              RetrieveIndexInDataAndMessage(
                x_val_in_checkerboard_part, y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]);
          if (val < best_val) {
            best_val = val;
            bestDisparity = current_disparity;
          }
        }
      }

      disparity_between_images_device[y_val*current_bp_level.width_level_ +
        (x_val_in_checkerboard_part * 2 + checkerboard_part_adjustment)] = bestDisparity;
    } else {
      disparity_between_images_device[y_val* current_bp_level.width_level_ +
        (x_val_in_checkerboard_part * 2 + checkerboard_part_adjustment)] = 0;
    }
  }

  //process from part 2 of checkerboard
  //adjustment based on checkerboard; need to add 1 to x for even-numbered rows for final index mapping into disparity images for checkerboard 2
  checkerboard_part_adjustment = ((y_val + 1u) % 2);

  if (run_imp_util::WithinImageBounds(
    x_val_in_checkerboard_part*2 + checkerboard_part_adjustment, y_val,
    current_bp_level.width_level_, current_bp_level.height_level_))
  {
    if ((x_val_in_checkerboard_part >= (1 - checkerboard_part_adjustment)) &&
        (x_val_in_checkerboard_part < (current_bp_level.width_checkerboard_level_ - checkerboard_part_adjustment)) &&
        (y_val > 0) && (y_val < (current_bp_level.height_level_ - 1)))
    {
      // keep track of "best" disparity for current pixel
      unsigned int bestDisparity{0u};
      U best_val{(U)bp_consts::kInfBp};
      if constexpr (DISP_VALS > 0) {
        for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
          const U val = 
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard0[
              RetrieveIndexInDataAndMessage(
                x_val_in_checkerboard_part, (y_val + 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard0[
              RetrieveIndexInDataAndMessage(
                x_val_in_checkerboard_part, (y_val - 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard0[
              RetrieveIndexInDataAndMessage(
                (x_val_in_checkerboard_part  + checkerboard_part_adjustment), y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard0[
              RetrieveIndexInDataAndMessage(
                (x_val_in_checkerboard_part + checkerboard_part_adjustment) - 1u,  y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[
              RetrieveIndexInDataAndMessage(
                x_val_in_checkerboard_part, y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]);
          if (val < best_val) {
            best_val = val;
            bestDisparity = current_disparity;
          }
        }
      }
      else {
        for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
          const U val =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard0[
              RetrieveIndexInDataAndMessage(
                x_val_in_checkerboard_part, (y_val + 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard0[
              RetrieveIndexInDataAndMessage(
                x_val_in_checkerboard_part, (y_val - 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard0[
              RetrieveIndexInDataAndMessage(
                (x_val_in_checkerboard_part  + checkerboard_part_adjustment), y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard0[
              RetrieveIndexInDataAndMessage(
                (x_val_in_checkerboard_part + checkerboard_part_adjustment) - 1u,  y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[
              RetrieveIndexInDataAndMessage(
                x_val_in_checkerboard_part, y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]);
          if (val < best_val) {
            best_val = val;
            bestDisparity = current_disparity;
          }
        }
      }

      disparity_between_images_device[y_val * current_bp_level.width_level_ +
        (x_val_in_checkerboard_part*2 + checkerboard_part_adjustment)] = bestDisparity;
    } else {
      disparity_between_images_device[y_val * current_bp_level.width_level_ +
        (x_val_in_checkerboard_part*2 + checkerboard_part_adjustment)] = 0;
    }
  }
}

template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void PrintDataAndMessageValsAtPointKernel(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const T* dataCostStereoCheckerboard0, const T* dataCostStereoCheckerboard1,
  const T* messageUDeviceCurrentCheckerboard0, const T* messageDDeviceCurrentCheckerboard0,
  const T* messageLDeviceCurrentCheckerboard0, const T* messageRDeviceCurrentCheckerboard0,
  const T* messageUDeviceCurrentCheckerboard1, const T* messageDDeviceCurrentCheckerboard1,
  const T* messageLDeviceCurrentCheckerboard1, const T* messageRDeviceCurrentCheckerboard1,
  unsigned int bp_settings_disp_vals = 0)
{
  if (((x_val + y_val) % 2) == 0) {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)messageUDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)messageDDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)messageLDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)messageRDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)dataCostStereoCheckerboard0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
    }
  } else {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)messageUDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)messageDDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)messageLDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)messageRDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)dataCostStereoCheckerboard1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
    }
  }
}

template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void PrintDataAndMessageValsToPointKernel(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const T* dataCostStereoCheckerboard0, const T* dataCostStereoCheckerboard1,
  const T* messageUDeviceCurrentCheckerboard0, const T* messageDDeviceCurrentCheckerboard0,
  const T* messageLDeviceCurrentCheckerboard0, const T* messageRDeviceCurrentCheckerboard0,
  const T* messageUDeviceCurrentCheckerboard1, const T* messageDDeviceCurrentCheckerboard1,
  const T* messageLDeviceCurrentCheckerboard1, const T* messageRDeviceCurrentCheckerboard1,
  unsigned int bp_settings_disp_vals = 0)
{
  const unsigned int checkerboard_adjustment = (((x_val + y_val) % 2) == 0) ? ((y_val)%2) : ((y_val+1)%2);
  if (((x_val + y_val) % 2) == 0) {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)messageUDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val + 1, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)messageDDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val - 1, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)messageLDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(
          x_val / 2 + checkerboard_adjustment, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)messageRDeviceCurrentCheckerboard1[RetrieveIndexInDataAndMessage(
          (x_val / 2 - 1) + checkerboard_adjustment, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
           current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)dataCostStereoCheckerboard0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
    }
  } else {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)messageUDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val + 1, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)messageDDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val - 1, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)messageLDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(
          x_val / 2 + checkerboard_adjustment, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)messageRDeviceCurrentCheckerboard0[RetrieveIndexInDataAndMessage(
          (x_val / 2 - 1) + checkerboard_adjustment, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)dataCostStereoCheckerboard1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
    }
  }
}

}

#endif /* SHAREDBPPROCESSINGFUNCTS_H_ */
