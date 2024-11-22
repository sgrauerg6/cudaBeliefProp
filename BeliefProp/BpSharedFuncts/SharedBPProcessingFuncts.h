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
#include "BpRunProcessing/BpConstsEnumsAliases.h"
#include "RunImp/RunImpGenFuncts.h"

namespace beliefprop {

//retrieve the current 1-D index value of the given point at the given disparity in the data cost and message data
ARCHITECTURE_ADDITION inline unsigned int RetrieveIndexInDataAndMessage(unsigned int x_val, unsigned int y_val,
  unsigned int width, unsigned int height, unsigned int current_disparity, unsigned int total_num_disp_vals,
  unsigned int offset_data = 0u)
{
  if constexpr (beliefprop::kOptimizedIndexingSetting) {
    //indexing is performed in such a way so that the memory accesses as coalesced as much as possible
    return (y_val * width * total_num_disp_vals + width * current_disparity + x_val) + offset_data;
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
  const U messages_neighbor_1[DISP_VALS], const U messages_neighbor_2[DISP_VALS],
  const U messages_neighbor_3[DISP_VALS], const U data_costs[DISP_VALS],
  T* dst_message_array, U disc_k_bp, bool data_aligned)
{
  // aggregate and find min
  U minimum{(U)beliefprop::kInfBp};
  U dst[DISP_VALS];

  for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++)
  {
    dst[current_disparity] = messages_neighbor_1[current_disparity] + messages_neighbor_2[current_disparity] +
                             messages_neighbor_3[current_disparity] + data_costs[current_disparity];
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
  const U* messages_neighbor_1, const U* messages_neighbor_2,
  const U* messages_neighbor_3, const U* data_costs,
  T* dst_message_array, U disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  // aggregate and find min
  U minimum{(U)beliefprop::kInfBp};
  U* dst = new U[bp_settings_disp_vals];

  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++)
  {
    dst[current_disparity] = messages_neighbor_1[current_disparity] + messages_neighbor_2[current_disparity] +
                             messages_neighbor_3[current_disparity] + data_costs[current_disparity];
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
  U* dst_processing, unsigned int checkerboard_adjustment,
  unsigned int offset_data, unsigned int current_disparity,
  unsigned int proc_array_idx)
{
  const U dataVal = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_message_array[
    RetrieveIndexInDataAndMessage(x_val, y_val,
      current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
      current_disparity, bp_settings_disp_vals, offset_data)]);

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

    dst_processing[proc_array_idx] = prevUVal + prevLVal + prevRVal + dataVal;
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

    dst_processing[proc_array_idx] = prevDVal + prevLVal + prevRVal + dataVal;
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

    dst_processing[proc_array_idx] = prevUVal + prevDVal + prevLVal + dataVal;
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

    dst_processing[proc_array_idx] = prevUVal + prevDVal + prevRVal + dataVal;
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
  U* dst_processing, unsigned int checkerboard_adjustment,
  unsigned int offset_data)
{
  // aggregate and find min
  U minimum{(U)beliefprop::kInfBp};
  unsigned int proc_array_idx_disp_0 = RetrieveIndexInDataAndMessage(x_val, y_val,
    current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, 0,
    bp_settings_disp_vals);
  unsigned int proc_array_idx{proc_array_idx_disp_0};

  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++)
  {
    //set initial dst processing array value corresponding to disparity for M message type
    SetInitDstProcessing<T, U, M>(x_val, y_val, current_bp_level, prev_u_message_array, prev_d_message_array,
      prev_l_message_array, prev_r_message_array, data_message_array, dst_message_array,
      disc_k_bp, data_aligned, bp_settings_disp_vals, dst_processing, checkerboard_adjustment,
      offset_data, current_disparity, proc_array_idx);

    if (dst_processing[proc_array_idx] < minimum)
      minimum = dst_processing[proc_array_idx];

    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      proc_array_idx += current_bp_level.padded_width_checkerboard_level_;
    }
    else {
      proc_array_idx++;
    }
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method
  //(see "Efficient Belief Propagation for Early Vision")
  DtStereo<U>(dst_processing, bp_settings_disp_vals, x_val, y_val, current_bp_level);

  // truncate
  minimum += disc_k_bp;

  // normalize
  U val_to_normalize{(U)0.0};

  proc_array_idx = proc_array_idx_disp_0;
  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
    if (minimum < dst_processing[proc_array_idx]) {
      dst_processing[proc_array_idx] = minimum;
    }

    val_to_normalize += dst_processing[proc_array_idx];

    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      proc_array_idx += current_bp_level.padded_width_checkerboard_level_;
    }
    else {
      proc_array_idx++;
    }
  }

  val_to_normalize /= ((U)bp_settings_disp_vals);

  //dst processing index and message array index are the same for each disparity value in this processing
  proc_array_idx = proc_array_idx_disp_0;

  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
    dst_processing[proc_array_idx] -= val_to_normalize;
    dst_message_array[proc_array_idx] =
      run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(dst_processing[proc_array_idx]);
    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      proc_array_idx += current_bp_level.padded_width_checkerboard_level_;
    }
    else {
      proc_array_idx++;
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
  T* data_cost_stereo_checkerboard_0,
  T* data_cost_stereo_checkerboard_1, 
  float lambda_bp, float data_k_bp, unsigned int bp_settings_disp_vals)
{
  if constexpr (DISP_VALS > 0) {
    unsigned int index_val;
    const unsigned int x_checkerboard = x_val / 2;

    if (run_imp_util::WithinImageBounds(
      x_checkerboard, y_val, current_bp_level.width_checkerboard_level_, current_bp_level.height_level_)) {
      //make sure that it is possible to check every disparity value
      //need to cast DISP_VALS from unsigned int to int
      //for conditional to work as expected
      if (((int)x_val - ((int)DISP_VALS - 1)) >= 0) {
        for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
          float current_pixel_image_1{0}, current_pixel_image_2{0};

          if (run_imp_util::WithinImageBounds(x_val, y_val, current_bp_level.width_level_, current_bp_level.height_level_)) {
            current_pixel_image_1 = image_1_pixels_device[y_val * current_bp_level.width_level_ + x_val];
            current_pixel_image_2 = image_2_pixels_device[y_val * current_bp_level.width_level_ + (x_val - current_disparity)];
          }

          index_val = RetrieveIndexInDataAndMessage(x_checkerboard, y_val,
            current_bp_level.padded_width_checkerboard_level_,
            current_bp_level.height_level_, current_disparity,
            DISP_VALS);

          //data cost is equal to dataWeight value for weighting times the absolute difference
          //in corresponding pixel intensity values capped at dataCostCap
          if (((x_val + y_val) % 2) == 0) {
            data_cost_stereo_checkerboard_0[index_val] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * run_imp_util::GetMin<float>((fabs(current_pixel_image_1 - current_pixel_image_2)), data_k_bp)));
          }
          else {
            data_cost_stereo_checkerboard_1[index_val] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * run_imp_util::GetMin<float>((fabs(current_pixel_image_1 - current_pixel_image_2)), data_k_bp)));
          }
        }
      } else {
        for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
          index_val = RetrieveIndexInDataAndMessage(x_checkerboard, y_val,
            current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
            current_disparity, DISP_VALS);

          //set data cost to zero if not possible to determine cost at disparity for pixel
          if (((x_val + y_val) % 2) == 0) {
            data_cost_stereo_checkerboard_0[index_val] = run_imp_util::ZeroVal<T>();
          }
          else {
            data_cost_stereo_checkerboard_1[index_val] = run_imp_util::ZeroVal<T>();
          }
        }
      }
    }
  }
  else {
    unsigned int index_val;
    const unsigned int x_checkerboard = x_val / 2;

    if (run_imp_util::WithinImageBounds(
      x_checkerboard, y_val, current_bp_level.width_checkerboard_level_, current_bp_level.height_level_)) {
      //make sure that it is possible to check every disparity value
      //need to cast bp_settings_disp_vals from unsigned int to int
      //for conditional to work as expected
      if (((int)x_val - ((int)bp_settings_disp_vals - 1)) >= 0) {
        for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
          float current_pixel_image_1{0}, current_pixel_image_2{0};

          if (run_imp_util::WithinImageBounds(x_val, y_val, current_bp_level.width_level_, current_bp_level.height_level_)) {
            current_pixel_image_1 = image_1_pixels_device[y_val * current_bp_level.width_level_ + x_val];
            current_pixel_image_2 = image_2_pixels_device[y_val * current_bp_level.width_level_ + (x_val - current_disparity)];
          }

          index_val = RetrieveIndexInDataAndMessage(x_checkerboard, y_val,
            current_bp_level.padded_width_checkerboard_level_,
            current_bp_level.height_level_, current_disparity,
            bp_settings_disp_vals);

          //data cost is equal to dataWeight value for weighting times the absolute difference
          //in corresponding pixel intensity values capped at dataCostCap
          if (((x_val + y_val) % 2) == 0) {
            data_cost_stereo_checkerboard_0[index_val] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * run_imp_util::GetMin<float>((fabs(current_pixel_image_1 - current_pixel_image_2)), data_k_bp)));
          }
          else {
            data_cost_stereo_checkerboard_1[index_val] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<float, T>(
              (float)(lambda_bp * run_imp_util::GetMin<float>((fabs(current_pixel_image_1 - current_pixel_image_2)), data_k_bp)));
          }
        }
      } else {
        for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
          index_val = RetrieveIndexInDataAndMessage(x_checkerboard, y_val,
            current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
            current_disparity, bp_settings_disp_vals);

          //set data cost to zero if not possible to determine cost at disparity for pixel
          if (((x_val + y_val) % 2) == 0) {
            data_cost_stereo_checkerboard_0[index_val] = run_imp_util::ZeroVal<T>();
          }
          else {
            data_cost_stereo_checkerboard_1[index_val] = run_imp_util::ZeroVal<T>();
          }
        }
      }
    }
  }
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void InitializeCurrentLevelDataPixel(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  T* data_cost_current_level, unsigned int offset_num,
  unsigned int bp_settings_disp_vals)
{
  //add 1 or 0 to the x-value depending on checkerboard part and row
  //beliefprop::CheckerboardPart::kCheckerboardPart0 with slot at (0, 0) has adjustment of 0 in row 0,
  //beliefprop::CheckerboardPart::kCheckerboardPart1 with slot at (0, 1) has adjustment of 1 in row 0
  const unsigned int checkerboard_part_adjustment =
    (checkerboard_part == beliefprop::CheckerboardPart::kCheckerboardPart0) ? (y_val % 2) : ((y_val + 1) % 2);

  //the corresponding x-values at the "lower" level depends on which checkerboard the pixel is in
  const unsigned int x_val_prev = x_val*2 + checkerboard_part_adjustment;

  if (run_imp_util::WithinImageBounds(
    x_val_prev, (y_val * 2 + 1), prev_bp_level.width_checkerboard_level_, prev_bp_level.height_level_))
  {
    if constexpr (DISP_VALS > 0) {
      for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
        const U data_cost =
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_cost_checkerboard_0[RetrieveIndexInDataAndMessage(
            x_val_prev, y_val*2, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, DISP_VALS, offset_num)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_cost_checkerboard_1[RetrieveIndexInDataAndMessage(
            x_val_prev, y_val*2, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, DISP_VALS, offset_num)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_cost_checkerboard_1[RetrieveIndexInDataAndMessage(
            x_val_prev, y_val*2 + 1, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, DISP_VALS, offset_num)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_cost_checkerboard_0[RetrieveIndexInDataAndMessage(
            x_val_prev, y_val*2 + 1, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, DISP_VALS, offset_num)]);

        data_cost_current_level[RetrieveIndexInDataAndMessage(x_val, y_val,
          current_bp_level.padded_width_checkerboard_level_,
          current_bp_level.height_level_, current_disparity,
          DISP_VALS)] =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(data_cost);
      }
    }
    else {
      for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
        const U data_cost =
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_cost_checkerboard_0[RetrieveIndexInDataAndMessage(
            x_val_prev, y_val*2, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, bp_settings_disp_vals, offset_num)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_cost_checkerboard_1[RetrieveIndexInDataAndMessage(
            x_val_prev, y_val*2, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, bp_settings_disp_vals, offset_num)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_cost_checkerboard_1[RetrieveIndexInDataAndMessage(
            x_val_prev, y_val*2 + 1, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, bp_settings_disp_vals, offset_num)]) +
          run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_cost_checkerboard_0[RetrieveIndexInDataAndMessage(
            x_val_prev, y_val*2 + 1, prev_bp_level.padded_width_checkerboard_level_, prev_bp_level.height_level_,
            current_disparity, bp_settings_disp_vals, offset_num)]);

        data_cost_current_level[RetrieveIndexInDataAndMessage(x_val, y_val,
          current_bp_level.padded_width_checkerboard_level_,
          current_bp_level.height_level_, current_disparity,
          bp_settings_disp_vals)] =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<U, T>(data_cost);
      }
    }
  }
}

//initialize the message values at each pixel of the current level to the default value
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void InitializeMessageValsToDefaultKernelPixel(
  unsigned int x_val_in_checkerboard, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  unsigned int bp_settings_disp_vals)
{
  //initialize message values in both checkerboards

  if constexpr (DISP_VALS > 0) {
    //set the message value at each pixel for each disparity to 0
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      message_u_checkerboard_0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      message_d_checkerboard_0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      message_l_checkerboard_0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      message_r_checkerboard_0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
    }

    //retrieve the previous message value at each movement at each pixel
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      message_u_checkerboard_1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      message_d_checkerboard_1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      message_l_checkerboard_1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
      message_r_checkerboard_1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS)] =
          run_imp_util::ZeroVal<T>();
    }
  }
  else {
    //set the message value at each pixel for each disparity to 0
    for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
      message_u_checkerboard_0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      message_d_checkerboard_0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      message_l_checkerboard_0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      message_r_checkerboard_0[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
    }

    //retrieve the previous message value at each movement at each pixel
    for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
      message_u_checkerboard_1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      message_d_checkerboard_1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      message_l_checkerboard_1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals)] =
          run_imp_util::ZeroVal<T>();
      message_r_checkerboard_1[RetrieveIndexInDataAndMessage(x_val_in_checkerboard, y_val,
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
  T* current_u_message, T* current_d_message,
  T* current_l_message, T* current_r_message,
  const U disc_k_bp, bool data_aligned)
{
  MsgStereo<T, U, DISP_VALS>(x_val, y_val, current_bp_level,
    prev_u_message, prev_l_message, prev_r_message, data_message,
    current_u_message, disc_k_bp, data_aligned);

  MsgStereo<T, U, DISP_VALS>(x_val, y_val, current_bp_level,
    prev_d_message, prev_l_message, prev_r_message, data_message,
    current_d_message, disc_k_bp, data_aligned);

  MsgStereo<T, U, DISP_VALS>(x_val, y_val, current_bp_level,
    prev_u_message, prev_d_message, prev_r_message, data_message,
    current_r_message, disc_k_bp, data_aligned);

  MsgStereo<T, U, DISP_VALS>(x_val, y_val, current_bp_level,
    prev_u_message, prev_d_message, prev_l_message, data_message,
    current_l_message, disc_k_bp, data_aligned);
}

template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline void RunBPIterationUpdateMsgVals(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const U* prev_u_message, const U* prev_d_message,
  const U* prev_l_message, const U* prev_r_message,
  const U* data_message,
  T* current_u_message, T* current_d_message,
  T* current_l_message, T* current_r_message,
  const U disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  MsgStereo<T, U>(x_val, y_val, current_bp_level,
    prev_u_message, prev_l_message, prev_r_message, data_message,
    current_u_message, disc_k_bp, data_aligned, bp_settings_disp_vals);

  MsgStereo<T, U>(x_val, y_val, current_bp_level,
    prev_d_message, prev_l_message, prev_r_message, data_message,
    current_d_message, disc_k_bp, data_aligned, bp_settings_disp_vals);

  MsgStereo<T, U>(x_val, y_val, current_bp_level,
    prev_u_message, prev_d_message, prev_r_message, data_message,
    current_r_message, disc_k_bp, data_aligned, bp_settings_disp_vals);

  MsgStereo<T, U>(x_val, y_val, current_bp_level,
    prev_u_message, prev_d_message, prev_l_message, data_message,
    current_l_message, disc_k_bp, data_aligned, bp_settings_disp_vals);
}

template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline void RunBPIterationUpdateMsgVals(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const T* prev_u_message_array, const T* prev_d_message_array,
  const T* prev_l_message_array, const T* prev_r_message_array,
  const T* data_message_array,
  T* current_u_message, T* current_d_message,
  T* current_l_message, T* current_r_message,
  const U disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals,
  U* dst_processing, unsigned int checkerboard_adjustment,
  unsigned int offset_data)
{
  MsgStereo<T, U, beliefprop::MessageComp::kUMessage>(x_val, y_val, current_bp_level,
    prev_u_message_array, prev_d_message_array, prev_l_message_array, prev_r_message_array, data_message_array,
    current_u_message, disc_k_bp, data_aligned, bp_settings_disp_vals,
    dst_processing, checkerboard_adjustment, offset_data);

  MsgStereo<T, U, beliefprop::MessageComp::kDMessage>(x_val, y_val, current_bp_level,
    prev_u_message_array, prev_d_message_array, prev_l_message_array, prev_r_message_array, data_message_array,
    current_d_message, disc_k_bp, data_aligned, bp_settings_disp_vals,
    dst_processing, checkerboard_adjustment, offset_data);

  MsgStereo<T, U, beliefprop::MessageComp::kLMessage>(x_val, y_val, current_bp_level,
    prev_u_message_array, prev_d_message_array, prev_l_message_array, prev_r_message_array, data_message_array,
    current_l_message, disc_k_bp, data_aligned, bp_settings_disp_vals,
    dst_processing, checkerboard_adjustment, offset_data);

  MsgStereo<T, U, beliefprop::MessageComp::kRMessage>(x_val, y_val, current_bp_level,
    prev_u_message_array, prev_d_message_array, prev_l_message_array, prev_r_message_array, data_message_array,
    current_r_message, disc_k_bp, data_aligned, bp_settings_disp_vals,
    dst_processing, checkerboard_adjustment, offset_data);
}

//device portion of the kernel function to run the current iteration of belief propagation in parallel
//using the checkerboard update method where half the pixels in the "checkerboard" scheme retrieve messages
//from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void RunBPIterationUsingCheckerboardUpdatesKernel(
  unsigned int x_val, unsigned int y_val,
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned,
  unsigned int bp_settings_disp_vals)
{
  //checkerboard_adjustment used for indexing into current checkerboard to update
  const unsigned int checkerboard_adjustment =
    (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) ? ((y_val)%2) : ((y_val+1)%2);

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
        if (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) {
          data_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            data_cost_checkerboard_0[RetrieveIndexInDataAndMessage(x_val, y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS, offset_data)]);
          prev_u_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_u_checkerboard_1[RetrieveIndexInDataAndMessage(x_val, (y_val+1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
          prev_d_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_d_checkerboard_1[RetrieveIndexInDataAndMessage(x_val, (y_val-1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
          prev_l_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_l_checkerboard_1[RetrieveIndexInDataAndMessage((x_val + checkerboard_adjustment), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
          prev_r_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_r_checkerboard_1[RetrieveIndexInDataAndMessage(((x_val + checkerboard_adjustment) - 1), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
        }
        else { //checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart1
          data_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            data_cost_checkerboard_1[RetrieveIndexInDataAndMessage(x_val, y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS, offset_data)]);
          prev_u_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_u_checkerboard_0[RetrieveIndexInDataAndMessage(x_val, (y_val+1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
          prev_d_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_d_checkerboard_0[RetrieveIndexInDataAndMessage(x_val, (y_val-1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
          prev_l_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_l_checkerboard_0[RetrieveIndexInDataAndMessage((x_val + checkerboard_adjustment), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
          prev_r_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_r_checkerboard_0[RetrieveIndexInDataAndMessage(((x_val + checkerboard_adjustment) - 1), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, DISP_VALS)]);
        }
      }

      //uses the previous message values and data cost to calculate the current message values and store the results
      if (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) {
        RunBPIterationUpdateMsgVals<T, U, DISP_VALS>(x_val, y_val, current_bp_level,
          prev_u_message, prev_d_message, prev_l_message, prev_r_message, data_message,
          message_u_checkerboard_0,  message_d_checkerboard_0,
          message_l_checkerboard_0,  message_r_checkerboard_0,
          (U)disc_k_bp, data_aligned);
      }
      else { //checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart1
        RunBPIterationUpdateMsgVals<T, U, DISP_VALS>(x_val, y_val, current_bp_level,
          prev_u_message, prev_d_message, prev_l_message, prev_r_message, data_message,
          message_u_checkerboard_1,  message_d_checkerboard_1,
          message_l_checkerboard_1, message_r_checkerboard_1,
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
        if (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) {
          data_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            data_cost_checkerboard_0[RetrieveIndexInDataAndMessage(x_val, y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals, offset_data)]);
          prev_u_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_u_checkerboard_1[RetrieveIndexInDataAndMessage(x_val, (y_val+1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
          prev_d_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_d_checkerboard_1[RetrieveIndexInDataAndMessage(x_val, (y_val-1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
          prev_l_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_l_checkerboard_1[RetrieveIndexInDataAndMessage((x_val + checkerboard_adjustment), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
          prev_r_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_r_checkerboard_1[RetrieveIndexInDataAndMessage(((x_val + checkerboard_adjustment) - 1), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
        }
        else { //checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart1
          data_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            data_cost_checkerboard_1[RetrieveIndexInDataAndMessage(x_val, y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals, offset_data)]);
          prev_u_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_u_checkerboard_0[RetrieveIndexInDataAndMessage(x_val, (y_val+1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
          prev_d_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_d_checkerboard_0[RetrieveIndexInDataAndMessage(x_val, (y_val-1),
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
          prev_l_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_l_checkerboard_0[RetrieveIndexInDataAndMessage((x_val + checkerboard_adjustment), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
          prev_r_message[current_disparity] = run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(
            message_r_checkerboard_0[RetrieveIndexInDataAndMessage(((x_val + checkerboard_adjustment) - 1), y_val,
              current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
              current_disparity, bp_settings_disp_vals)]);
        }
      }

      //uses the previous message values and data cost to calculate the current message values and store the results
      if (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) {
        RunBPIterationUpdateMsgVals<T, U>(x_val, y_val, current_bp_level,
          prev_u_message, prev_d_message, prev_l_message, prev_r_message, data_message,
          message_u_checkerboard_0,  message_d_checkerboard_0,
          message_l_checkerboard_0,  message_r_checkerboard_0,
          (U)disc_k_bp, data_aligned, bp_settings_disp_vals);
      }
      else { //checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart1
        RunBPIterationUpdateMsgVals<T, U>(x_val, y_val, current_bp_level,
          prev_u_message, prev_d_message, prev_l_message, prev_r_message, data_message,
          message_u_checkerboard_1,  message_d_checkerboard_1,
          message_l_checkerboard_1, message_r_checkerboard_1,
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
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned,
  unsigned int bp_settings_disp_vals, void* dst_processing)
{
  //checkerboard_adjustment used for indexing into current checkerboard to update
  const unsigned int checkerboard_adjustment = 
    (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) ? ((y_val)%2) : ((y_val+1)%2);

  //may want to look into (x_val < (width_level_checkerboard_part - 1) since it may affect the edges
  //make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
  if ((x_val >= (1u - checkerboard_adjustment)) &&
      (x_val < (current_bp_level.width_checkerboard_level_ - checkerboard_adjustment)) &&
      (y_val > 0) && (y_val < (current_bp_level.height_level_ - 1u)))
  {
    //uses the previous message values and data cost to calculate the current message values and store the results
    if (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) {
      RunBPIterationUpdateMsgVals<T, U>(x_val, y_val, current_bp_level,
        message_u_checkerboard_1, message_d_checkerboard_1,
        message_l_checkerboard_1, message_r_checkerboard_1,
        data_cost_checkerboard_0,
        message_u_checkerboard_0,  message_d_checkerboard_0,
        message_l_checkerboard_0,  message_r_checkerboard_0,
        (U)disc_k_bp, data_aligned, bp_settings_disp_vals, (U*)dst_processing,
        checkerboard_adjustment, offset_data);
    }
    else { //checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart1
      RunBPIterationUpdateMsgVals<T, U>(x_val, y_val, current_bp_level,
        message_u_checkerboard_0, message_d_checkerboard_0,
        message_l_checkerboard_0, message_r_checkerboard_0,
        data_cost_checkerboard_1,
        message_u_checkerboard_1,  message_d_checkerboard_1,
        message_l_checkerboard_1, message_r_checkerboard_1,
        (U)disc_k_bp, data_aligned, bp_settings_disp_vals, (U*)dst_processing,
        checkerboard_adjustment, offset_data);
    }
  }
}

//kernel to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernel works from the point of view of the pixel at the prev level that is being copied to four different places
template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void CopyMsgDataToNextLevelPixel(
  unsigned int x_val, unsigned int y_val,
  beliefprop::CheckerboardPart checkerboard_part, const beliefprop::BpLevelProperties& current_bp_level,
  const beliefprop::BpLevelProperties& next_bp_level,
  const T* message_u_prev_checkerboard_0, const T* message_d_prev_checkerboard_0,
  const T* message_l_prev_checkerboard_0, const T* message_r_prev_checkerboard_0,
  const T* message_u_prev_checkerboard_1, const T* message_d_prev_checkerboard_1,
  const T* message_l_prev_checkerboard_1, const T* message_r_prev_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  unsigned int bp_settings_disp_vals)
{
  //only need to copy checkerboard 1 around "edge" since updating checkerboard 1 in first belief propagation iteration
  //(and checkerboard 0 message values are used in the iteration to update message values in checkerboard 1)
  const bool copyCheckerboard1 = (((x_val == 0) || (y_val == 0)) || 
    (((x_val >= (current_bp_level.width_checkerboard_level_ - 2)) ||
      (y_val >= (current_bp_level.height_level_ - 2)))));
  
  unsigned int index_copy_to, index_copy_from;
  T prev_val_u, prev_val_d, prev_val_l, prev_val_r;
  const unsigned int checkerboard_part_adjustment =
    (checkerboard_part == beliefprop::CheckerboardPart::kCheckerboardPart0) ? (y_val % 2) : ((y_val + 1) % 2);
  
  if constexpr (DISP_VALS > 0) {
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      index_copy_from = RetrieveIndexInDataAndMessage(x_val, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, DISP_VALS);

      if (checkerboard_part == beliefprop::CheckerboardPart::kCheckerboardPart0) {
        prev_val_u = message_u_prev_checkerboard_0[index_copy_from];
        prev_val_d = message_d_prev_checkerboard_0[index_copy_from];
        prev_val_l = message_l_prev_checkerboard_0[index_copy_from];
        prev_val_r = message_r_prev_checkerboard_0[index_copy_from];
      } else /*(checkerboard_part == beliefprop::CheckerboardPart::kCheckerboardPart1)*/ {
        prev_val_u = message_u_prev_checkerboard_1[index_copy_from];
        prev_val_d = message_d_prev_checkerboard_1[index_copy_from];
        prev_val_l = message_l_prev_checkerboard_1[index_copy_from];
        prev_val_r = message_r_prev_checkerboard_1[index_copy_from];
      }

      if (run_imp_util::WithinImageBounds(
        x_val*2 + checkerboard_part_adjustment, y_val*2,
        next_bp_level.width_checkerboard_level_,
        next_bp_level.height_level_))
      {
        index_copy_to = RetrieveIndexInDataAndMessage(x_val*2 + checkerboard_part_adjustment, y_val*2,
          next_bp_level.padded_width_checkerboard_level_, next_bp_level.height_level_,
          current_disparity, DISP_VALS);

        message_u_checkerboard_0[index_copy_to] = prev_val_u;
        message_d_checkerboard_0[index_copy_to] = prev_val_d;
        message_l_checkerboard_0[index_copy_to] = prev_val_l;
        message_r_checkerboard_0[index_copy_to] = prev_val_r;

        if (copyCheckerboard1) {
          message_u_checkerboard_1[index_copy_to] = prev_val_u;
          message_d_checkerboard_1[index_copy_to] = prev_val_d;
          message_l_checkerboard_1[index_copy_to] = prev_val_l;
          message_r_checkerboard_1[index_copy_to] = prev_val_r;
        }
      }

      if (run_imp_util::WithinImageBounds(
        x_val*2 + checkerboard_part_adjustment, y_val*2 + 1,
        next_bp_level.width_checkerboard_level_, next_bp_level.height_level_))
      {
        index_copy_to = RetrieveIndexInDataAndMessage(x_val*2 + checkerboard_part_adjustment, y_val*2 + 1,
          next_bp_level.padded_width_checkerboard_level_, next_bp_level.height_level_,
          current_disparity, DISP_VALS);

        message_u_checkerboard_0[index_copy_to] = prev_val_u;
        message_d_checkerboard_0[index_copy_to] = prev_val_d;
        message_l_checkerboard_0[index_copy_to] = prev_val_l;
        message_r_checkerboard_0[index_copy_to] = prev_val_r;

        if (copyCheckerboard1) {
          message_u_checkerboard_1[index_copy_to] = prev_val_u;
          message_d_checkerboard_1[index_copy_to] = prev_val_d;
          message_l_checkerboard_1[index_copy_to] = prev_val_l;
          message_r_checkerboard_1[index_copy_to] = prev_val_r;
        }
      }
    }
  }
  else {
    for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
      index_copy_from = RetrieveIndexInDataAndMessage(x_val, y_val,
        current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
        current_disparity, bp_settings_disp_vals);

      if (checkerboard_part == beliefprop::CheckerboardPart::kCheckerboardPart0) {
        prev_val_u = message_u_prev_checkerboard_0[index_copy_from];
        prev_val_d = message_d_prev_checkerboard_0[index_copy_from];
        prev_val_l = message_l_prev_checkerboard_0[index_copy_from];
        prev_val_r = message_r_prev_checkerboard_0[index_copy_from];
      } else /*(checkerboard_part == beliefprop::CheckerboardPart::kCheckerboardPart1)*/ {
        prev_val_u = message_u_prev_checkerboard_1[index_copy_from];
        prev_val_d = message_d_prev_checkerboard_1[index_copy_from];
        prev_val_l = message_l_prev_checkerboard_1[index_copy_from];
        prev_val_r = message_r_prev_checkerboard_1[index_copy_from];
      }

      if (run_imp_util::WithinImageBounds(
        x_val*2 + checkerboard_part_adjustment, y_val*2,
        next_bp_level.width_checkerboard_level_, next_bp_level.height_level_))
      {
        index_copy_to = RetrieveIndexInDataAndMessage(x_val*2 + checkerboard_part_adjustment, y_val*2,
          next_bp_level.padded_width_checkerboard_level_, next_bp_level.height_level_,
          current_disparity, bp_settings_disp_vals);

        message_u_checkerboard_0[index_copy_to] = prev_val_u;
        message_d_checkerboard_0[index_copy_to] = prev_val_d;
        message_l_checkerboard_0[index_copy_to] = prev_val_l;
        message_r_checkerboard_0[index_copy_to] = prev_val_r;

        if (copyCheckerboard1) {
          message_u_checkerboard_1[index_copy_to] = prev_val_u;
          message_d_checkerboard_1[index_copy_to] = prev_val_d;
          message_l_checkerboard_1[index_copy_to] = prev_val_l;
          message_r_checkerboard_1[index_copy_to] = prev_val_r;
        }
      }

      if (run_imp_util::WithinImageBounds(
        x_val*2 + checkerboard_part_adjustment, y_val*2 + 1,
        next_bp_level.width_checkerboard_level_, next_bp_level.height_level_))
      {
        index_copy_to = RetrieveIndexInDataAndMessage(x_val*2 + checkerboard_part_adjustment, y_val*2 + 1,
          next_bp_level.padded_width_checkerboard_level_, next_bp_level.height_level_,
          current_disparity, bp_settings_disp_vals);

        message_u_checkerboard_0[index_copy_to] = prev_val_u;
        message_d_checkerboard_0[index_copy_to] = prev_val_d;
        message_l_checkerboard_0[index_copy_to] = prev_val_l;
        message_r_checkerboard_0[index_copy_to] = prev_val_r;

        if (copyCheckerboard1) {
          message_u_checkerboard_1[index_copy_to] = prev_val_u;
          message_d_checkerboard_1[index_copy_to] = prev_val_d;
          message_l_checkerboard_1[index_copy_to] = prev_val_l;
          message_r_checkerboard_1[index_copy_to] = prev_val_r;
        }
      }
    }
  }
}

template<RunData_t T, RunData_t U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void RetrieveOutputDisparityPixel(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  const T* message_u_checkerboard_0, const T* message_d_checkerboard_0,
  const T* message_l_checkerboard_0, const T* message_r_checkerboard_0,
  const T* message_u_checkerboard_1, const T* message_d_checkerboard_1,
  const T* message_l_checkerboard_1, const T* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  const unsigned int x_val_checkerboard = x_val;

  //first processing from first part of checkerboard

  //adjustment based on checkerboard; need to add 1 to x for odd-numbered rows
  //for final index mapping into disparity images for checkerboard 1
  unsigned int checkerboard_part_adjustment = (y_val % 2);

  if (run_imp_util::WithinImageBounds(
    x_val_checkerboard*2 + checkerboard_part_adjustment, y_val,
    current_bp_level.width_level_, current_bp_level.height_level_))
  {
    if ((x_val_checkerboard >= (1 - checkerboard_part_adjustment)) &&
        (x_val_checkerboard < (current_bp_level.width_checkerboard_level_ - checkerboard_part_adjustment)) &&
        (y_val > 0u) && (y_val < (current_bp_level.height_level_ - 1u)))
    {
      // keep track of "best" disparity for current pixel
      unsigned int bestDisparity{0u};
      U best_val{(U)beliefprop::kInfBp};
      if constexpr (DISP_VALS > 0) {
        for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
          const U val =
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_u_checkerboard_1[
              RetrieveIndexInDataAndMessage(
                x_val_checkerboard, (y_val + 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_d_checkerboard_1[
              RetrieveIndexInDataAndMessage(
                x_val_checkerboard, (y_val - 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_l_checkerboard_1[
              RetrieveIndexInDataAndMessage(
                (x_val_checkerboard  + checkerboard_part_adjustment), y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_r_checkerboard_1[
              RetrieveIndexInDataAndMessage(
                (x_val_checkerboard + checkerboard_part_adjustment) - 1u, y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_cost_checkerboard_0[
              RetrieveIndexInDataAndMessage(
                x_val_checkerboard, y_val,
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
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_u_checkerboard_1[
              RetrieveIndexInDataAndMessage(
                x_val_checkerboard, (y_val + 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_d_checkerboard_1[
              RetrieveIndexInDataAndMessage(
                x_val_checkerboard, (y_val - 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_l_checkerboard_1[
              RetrieveIndexInDataAndMessage(
                (x_val_checkerboard  + checkerboard_part_adjustment), y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_r_checkerboard_1[
              RetrieveIndexInDataAndMessage(
                (x_val_checkerboard + checkerboard_part_adjustment) - 1u, y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_cost_checkerboard_0[
              RetrieveIndexInDataAndMessage(
                x_val_checkerboard, y_val,
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
        (x_val_checkerboard * 2 + checkerboard_part_adjustment)] = bestDisparity;
    } else {
      disparity_between_images_device[y_val* current_bp_level.width_level_ +
        (x_val_checkerboard * 2 + checkerboard_part_adjustment)] = 0;
    }
  }

  //process from part 2 of checkerboard
  //adjustment based on checkerboard; need to add 1 to x for even-numbered rows for final index mapping into disparity images for checkerboard 2
  checkerboard_part_adjustment = ((y_val + 1u) % 2);

  if (run_imp_util::WithinImageBounds(
    x_val_checkerboard*2 + checkerboard_part_adjustment, y_val,
    current_bp_level.width_level_, current_bp_level.height_level_))
  {
    if ((x_val_checkerboard >= (1 - checkerboard_part_adjustment)) &&
        (x_val_checkerboard < (current_bp_level.width_checkerboard_level_ - checkerboard_part_adjustment)) &&
        (y_val > 0) && (y_val < (current_bp_level.height_level_ - 1)))
    {
      // keep track of "best" disparity for current pixel
      unsigned int bestDisparity{0u};
      U best_val{(U)beliefprop::kInfBp};
      if constexpr (DISP_VALS > 0) {
        for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
          const U val = 
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_u_checkerboard_0[
              RetrieveIndexInDataAndMessage(
                x_val_checkerboard, (y_val + 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_d_checkerboard_0[
              RetrieveIndexInDataAndMessage(
                x_val_checkerboard, (y_val - 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_l_checkerboard_0[
              RetrieveIndexInDataAndMessage(
                (x_val_checkerboard  + checkerboard_part_adjustment), y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_r_checkerboard_0[
              RetrieveIndexInDataAndMessage(
                (x_val_checkerboard + checkerboard_part_adjustment) - 1u,  y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                DISP_VALS)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_cost_checkerboard_1[
              RetrieveIndexInDataAndMessage(
                x_val_checkerboard, y_val,
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
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_u_checkerboard_0[
              RetrieveIndexInDataAndMessage(
                x_val_checkerboard, (y_val + 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_d_checkerboard_0[
              RetrieveIndexInDataAndMessage(
                x_val_checkerboard, (y_val - 1u),
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_l_checkerboard_0[
              RetrieveIndexInDataAndMessage(
                (x_val_checkerboard  + checkerboard_part_adjustment), y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(message_r_checkerboard_0[
              RetrieveIndexInDataAndMessage(
                (x_val_checkerboard + checkerboard_part_adjustment) - 1u,  y_val,
                current_bp_level.padded_width_checkerboard_level_,
                current_bp_level.height_level_,
                current_disparity,
                bp_settings_disp_vals)]) +
            run_imp_util::ConvertValToDifferentDataTypeIfNeeded<T, U>(data_cost_checkerboard_1[
              RetrieveIndexInDataAndMessage(
                x_val_checkerboard, y_val,
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
        (x_val_checkerboard*2 + checkerboard_part_adjustment)] = bestDisparity;
    } else {
      disparity_between_images_device[y_val * current_bp_level.width_level_ +
        (x_val_checkerboard*2 + checkerboard_part_adjustment)] = 0;
    }
  }
}

template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void PrintDataAndMessageValsAtPointKernel(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  const T* message_u_checkerboard_0, const T* message_d_checkerboard_0,
  const T* message_l_checkerboard_0, const T* message_r_checkerboard_0,
  const T* message_u_checkerboard_1, const T* message_d_checkerboard_1,
  const T* message_l_checkerboard_1, const T* message_r_checkerboard_1,
  unsigned int bp_settings_disp_vals = 0)
{
  if (((x_val + y_val) % 2) == 0) {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)message_u_checkerboard_0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)message_d_checkerboard_0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)message_l_checkerboard_0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)message_r_checkerboard_0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)data_cost_checkerboard_0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
    }
  } else {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)message_u_checkerboard_1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)message_d_checkerboard_1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)message_l_checkerboard_1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)message_r_checkerboard_1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)data_cost_checkerboard_1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
    }
  }
}

template<RunData_t T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void PrintDataAndMessageValsToPointKernel(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  const T* message_u_checkerboard_0, const T* message_d_checkerboard_0,
  const T* message_l_checkerboard_0, const T* message_r_checkerboard_0,
  const T* message_u_checkerboard_1, const T* message_d_checkerboard_1,
  const T* message_l_checkerboard_1, const T* message_r_checkerboard_1,
  unsigned int bp_settings_disp_vals = 0)
{
  const unsigned int checkerboard_adjustment = (((x_val + y_val) % 2) == 0) ? ((y_val)%2) : ((y_val+1)%2);
  if (((x_val + y_val) % 2) == 0) {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)message_u_checkerboard_1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val + 1, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)message_d_checkerboard_1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val - 1, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)message_l_checkerboard_1[RetrieveIndexInDataAndMessage(
          x_val / 2 + checkerboard_adjustment, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)message_r_checkerboard_1[RetrieveIndexInDataAndMessage(
          (x_val / 2 - 1) + checkerboard_adjustment, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
           current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)data_cost_checkerboard_0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
    }
  } else {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)message_u_checkerboard_0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val + 1, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)message_d_checkerboard_0[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val - 1, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)message_l_checkerboard_0[RetrieveIndexInDataAndMessage(
          x_val / 2 + checkerboard_adjustment, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)message_r_checkerboard_0[RetrieveIndexInDataAndMessage(
          (x_val / 2 - 1) + checkerboard_adjustment, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)data_cost_checkerboard_1[RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
    }
  }
}

}

#endif /* SHAREDBPPROCESSINGFUNCTS_H_ */
