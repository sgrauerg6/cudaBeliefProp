/*
Copyright (C) 2024 Scott Grauer-Gray

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

/**
 * @file KernelBpCUDAHalf.cu
 * @author Scott Grauer-Gray
 * @brief This file defines the template specialization to perform belief propagation using half precision for
 * disparity map estimation from stereo images on CUDA to prevent overflow in val_to_normalize
 * message value computation
 * shouldn't be an issue if using bfloat instead of half so recommend using that instead of using
 * these template specialization functions
 * 
 * @copyright Copyright (c) 2024
 */

//set constexpr unsigned int values for number of disparity values for each stereo set used
constexpr unsigned int kDispVals0{beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSetHalfSize)].num_disp_vals};
constexpr unsigned int kDispVals1{beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSet)].num_disp_vals};
constexpr unsigned int kDispVals2{beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kVenus)].num_disp_vals};
constexpr unsigned int kDispVals3{beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kBarn1)].num_disp_vals};
constexpr unsigned int kDispVals4{beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesQuarterSize)].num_disp_vals};
constexpr unsigned int kDispVals5{beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesHalfSize)].num_disp_vals};
constexpr unsigned int kDispVals6{beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesFullSizeCropped)].num_disp_vals};

//device function to process messages using half precision with number of disparity values
//given in template parameter
template <unsigned int DISP_VALS>
__device__ inline void MsgStereoHalf(unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level, half messages_neighbor_1[DISP_VALS],
  half messages_neighbor_2[DISP_VALS], half messages_neighbor_3[DISP_VALS],
  half data_costs[DISP_VALS], half* dst_message_array, half disc_k_bp, bool data_aligned)
{
  // aggregate and find min
  half minimum = beliefprop::kHighValBp<half>;
  half dst[DISP_VALS];

  for (size_t current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
    dst[current_disparity] = messages_neighbor_1[current_disparity] +
                            messages_neighbor_2[current_disparity] +
                            messages_neighbor_3[current_disparity] +
                            data_costs[current_disparity];
    if (dst[current_disparity] < minimum) {
      minimum = dst[current_disparity];
    }
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
  DtStereo<half, DISP_VALS>(dst);

  // truncate
  minimum += disc_k_bp;

  // normalize
  half val_to_normalize = 0;

  for (size_t current_disparity = 0; current_disparity < DISP_VALS; current_disparity++)
  {
    if (minimum < dst[current_disparity]) {
      dst[current_disparity] = minimum;
    }
    val_to_normalize += dst[current_disparity];
  }

  //if val_to_normalize is infinite or NaN (observed when using more than 5 computation levels with half-precision),
  //set destination vector to 0 for all disparities
  //note that may cause results to differ a little from ideal
  if (__hisnan(val_to_normalize) || ((__hisinf(val_to_normalize)) != 0)) {
    unsigned int dest_message_array_index = beliefprop::RetrieveIndexInDataAndMessage(x_val, y_val,
      current_bp_level.LevelProperties().padded_width_checkerboard_level_,
      current_bp_level.LevelProperties().height_level_, 0,
      DISP_VALS);

    for (size_t current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      dst_message_array[dest_message_array_index] = (half) 0.0;
      if constexpr (beliefprop::kOptimizedIndexingSetting) {
        dest_message_array_index += current_bp_level.LevelProperties().padded_width_checkerboard_level_;
      }
      else {
        dest_message_array_index++;
      }
    }
  }
  else
  {
    val_to_normalize /= DISP_VALS;

    unsigned int dest_message_array_index = beliefprop::RetrieveIndexInDataAndMessage(x_val, y_val,
      current_bp_level.LevelProperties().padded_width_checkerboard_level_,
      current_bp_level.LevelProperties().height_level_, 0,
      DISP_VALS);

    for (size_t current_disparity = 0; current_disparity < DISP_VALS; current_disparity++)
    {
      dst[current_disparity] -= val_to_normalize;
      dst_message_array[dest_message_array_index] = dst[current_disparity];
      if constexpr (beliefprop::kOptimizedIndexingSetting) {
        dest_message_array_index += current_bp_level.LevelProperties().padded_width_checkerboard_level_;
      }
      else {
        dest_message_array_index++;
      }
    }
  }
}

//template BP message processing when number of disparity values is given
//as an input parameter and not as a template
template <beliefprop::MessageComp M>
__device__ inline void MsgStereoHalf(unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level,
  half* prev_u_messageArray, half* prev_d_messageArray,
  half* prev_l_messageArray, half* prev_r_messageArray,
  half* data_message_array, half* dst_message_array,
  half disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals,
  half* dst_processing, unsigned int checkerboard_adjustment,
  unsigned int offset_data)
{
  // aggregate and find min
  half minimum{beliefprop::kHighValBp<half>};
  unsigned int proc_array_idx_disp_0 = beliefprop::RetrieveIndexInDataAndMessage(x_val, y_val,
    current_bp_level.LevelProperties().padded_width_checkerboard_level_,
    current_bp_level.LevelProperties().height_level_, 0,
    bp_settings_disp_vals);
  unsigned int proc_array_idx{proc_array_idx_disp_0};

  for (size_t current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++)
  {
    //set initial dst processing array value corresponding to disparity for M message type
    beliefprop::SetInitDstProcessing<half, half, M>(x_val, y_val, current_bp_level, prev_u_messageArray, prev_d_messageArray,
      prev_l_messageArray, prev_r_messageArray, data_message_array, dst_message_array,
      disc_k_bp, data_aligned, bp_settings_disp_vals, dst_processing, checkerboard_adjustment,
      offset_data, current_disparity, proc_array_idx);

    if (dst_processing[proc_array_idx] < minimum)
      minimum = dst_processing[proc_array_idx];

    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      proc_array_idx += current_bp_level.LevelProperties().padded_width_checkerboard_level_;
    }
    else {
      proc_array_idx++;
    }
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method
  //(see "Efficient Belief Propagation for Early Vision")
  DtStereo<half>(dst_processing, bp_settings_disp_vals, x_val, y_val, current_bp_level);

  // truncate
  minimum += disc_k_bp;

  // normalize
  half val_to_normalize{(half)0.0};

  proc_array_idx = proc_array_idx_disp_0;
  for (size_t current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
    if (minimum < dst_processing[proc_array_idx]) {
      dst_processing[proc_array_idx] = minimum;
    }

    val_to_normalize += dst_processing[proc_array_idx];

    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      proc_array_idx += current_bp_level.LevelProperties().padded_width_checkerboard_level_;
    }
    else {
      proc_array_idx++;
    }
  }

  //if val_to_normalize is infinite or NaN (observed when using more than 5 computation levels with half-precision),
  //set destination vector to 0 for all disparities
  //note that may cause results to differ a little from ideal
  if (__hisnan(val_to_normalize) || ((__hisinf(val_to_normalize)) != 0)) {
    //dst processing index and message array index are the same for each disparity value in this processing
    proc_array_idx = proc_array_idx_disp_0;

    for (size_t current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
      dst_message_array[proc_array_idx] = (half)0.0;
      if constexpr (beliefprop::kOptimizedIndexingSetting) {
        proc_array_idx += current_bp_level.LevelProperties().padded_width_checkerboard_level_;
      }
      else {
        proc_array_idx++;
      }
    }
  }
  else
  {
    val_to_normalize /= ((half)bp_settings_disp_vals);

    //dst processing index and message array index are the same for each disparity value in this processing
    proc_array_idx = proc_array_idx_disp_0;

    for (size_t current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
      dst_processing[proc_array_idx] -= val_to_normalize;
      dst_message_array[proc_array_idx] = ConvertValToDifferentDataTypeIfNeeded<half, half>(dst_processing[proc_array_idx]);
      if constexpr (beliefprop::kOptimizedIndexingSetting) {
        proc_array_idx += current_bp_level.LevelProperties().padded_width_checkerboard_level_;
      }
      else {
        proc_array_idx++;
      }
    }
  }
}

template<>
__device__ inline void MsgStereo<half, half, beliefprop::MessageComp::kUMessage>(
  unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level,
  half* prev_u_messageArray, half* prev_d_messageArray,
  half* prev_l_messageArray, half* prev_r_messageArray,
  half* data_message_array, half* dst_message_array,
  half disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals,
  half* dst_processing, unsigned int checkerboard_adjustment,
  unsigned int offset_data)
{
  MsgStereoHalf<beliefprop::MessageComp::kUMessage>(x_val, y_val, current_bp_level, prev_u_messageArray, prev_d_messageArray,
    prev_l_messageArray, prev_r_messageArray, data_message_array, dst_message_array, disc_k_bp, data_aligned, bp_settings_disp_vals,
    dst_processing, checkerboard_adjustment, offset_data);
}

template<>
__device__ inline void MsgStereo<half, half, beliefprop::MessageComp::kDMessage>(
  unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level,
  half* prev_u_messageArray, half* prev_d_messageArray,
  half* prev_l_messageArray, half* prev_r_messageArray,
  half* data_message_array, half* dst_message_array,
  half disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals,
  half* dst_processing, unsigned int checkerboard_adjustment,
  unsigned int offset_data)
{
  MsgStereoHalf<beliefprop::MessageComp::kDMessage>(x_val, y_val, current_bp_level, prev_u_messageArray, prev_d_messageArray,
    prev_l_messageArray, prev_r_messageArray, data_message_array, dst_message_array, disc_k_bp, data_aligned, bp_settings_disp_vals,
    dst_processing, checkerboard_adjustment, offset_data);
}

template<>
__device__ inline void MsgStereo<half, half, beliefprop::MessageComp::kLMessage>(
  unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level,
  half* prev_u_messageArray, half* prev_d_messageArray,
  half* prev_l_messageArray, half* prev_r_messageArray,
  half* data_message_array, half* dst_message_array,
  half disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals,
  half* dst_processing, unsigned int checkerboard_adjustment,
  unsigned int offset_data)
{
  MsgStereoHalf<beliefprop::MessageComp::kLMessage>(x_val, y_val, current_bp_level, prev_u_messageArray, prev_d_messageArray,
    prev_l_messageArray, prev_r_messageArray, data_message_array, dst_message_array, disc_k_bp, data_aligned, bp_settings_disp_vals,
    dst_processing, checkerboard_adjustment, offset_data);
}

template<>
__device__ inline void MsgStereo<half, half, beliefprop::MessageComp::kRMessage>(
  unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level,
  half* prev_u_messageArray, half* prev_d_messageArray,
  half* prev_l_messageArray, half* prev_r_messageArray,
  half* data_message_array, half* dst_message_array,
  half disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals,
  half* dst_processing, unsigned int checkerboard_adjustment,
  unsigned int offset_data)
{
  MsgStereoHalf<beliefprop::MessageComp::kRMessage>(x_val, y_val, current_bp_level, prev_u_messageArray, prev_d_messageArray,
    prev_l_messageArray, prev_r_messageArray, data_message_array, dst_message_array, disc_k_bp, data_aligned, bp_settings_disp_vals,
    dst_processing, checkerboard_adjustment, offset_data);
}

template<>
__device__ inline void MsgStereo<half, half, kDispVals0>(
  unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level, half messages_neighbor_1[kDispVals0],
  half messages_neighbor_2[kDispVals0], half messages_neighbor_3[kDispVals0],
  half data_costs[kDispVals0], half* dst_message_array, half disc_k_bp, bool data_aligned)
{
  MsgStereoHalf<kDispVals0>(x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2, messages_neighbor_3,
    data_costs, dst_message_array, disc_k_bp, data_aligned);
}

template<>
__device__ inline void MsgStereo<half, half, kDispVals1>(
  unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level, half messages_neighbor_1[kDispVals1],
  half messages_neighbor_2[kDispVals1], half messages_neighbor_3[kDispVals1],
  half data_costs[kDispVals1], half* dst_message_array, half disc_k_bp, bool data_aligned)
{
  MsgStereoHalf<kDispVals1>(x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2, messages_neighbor_3,
    data_costs, dst_message_array, disc_k_bp, data_aligned);
}

template<>
__device__ inline void MsgStereo<half, half, kDispVals2>(
  unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level, half messages_neighbor_1[kDispVals2],
  half messages_neighbor_2[kDispVals2], half messages_neighbor_3[kDispVals2],
  half data_costs[kDispVals2], half* dst_message_array, half disc_k_bp, bool data_aligned)
{
  MsgStereoHalf<kDispVals2>(x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2, messages_neighbor_3,
    data_costs, dst_message_array, disc_k_bp, data_aligned);
}

template<>
__device__ inline void MsgStereo<half, half, kDispVals3>(
  unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level, half messages_neighbor_1[kDispVals3],
  half messages_neighbor_2[kDispVals3], half messages_neighbor_3[kDispVals3],
  half data_costs[kDispVals3], half* dst_message_array, half disc_k_bp, bool data_aligned)
{
  MsgStereoHalf<kDispVals3>(x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2, messages_neighbor_3,
    data_costs, dst_message_array, disc_k_bp, data_aligned);
}

template<>
__device__ inline void MsgStereo<half, half, kDispVals4>(
  unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level, half messages_neighbor_1[kDispVals4],
  half messages_neighbor_2[kDispVals4], half messages_neighbor_3[kDispVals4],
  half data_costs[kDispVals4], half* dst_message_array, half disc_k_bp, bool data_aligned)
{
  MsgStereoHalf<kDispVals4>(x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2, messages_neighbor_3,
    data_costs, dst_message_array, disc_k_bp, data_aligned);
}

template<>
__device__ inline void MsgStereo<half, half, kDispVals5>(
  unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level, half messages_neighbor_1[kDispVals5],
  half messages_neighbor_2[kDispVals5], half messages_neighbor_3[kDispVals5],
  half data_costs[kDispVals5], half* dst_message_array, half disc_k_bp, bool data_aligned)
{
  MsgStereoHalf<kDispVals5>(x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2, messages_neighbor_3,
    data_costs, dst_message_array, disc_k_bp, data_aligned);
}

template<>
__device__ inline void MsgStereo<half, half, kDispVals6>(
  unsigned int x_val, unsigned int y_val,
  const BpLevel<T>& current_bp_level, half messages_neighbor_1[kDispVals6],
  half messages_neighbor_2[kDispVals6], half messages_neighbor_3[kDispVals6],
  half data_costs[kDispVals6], half* dst_message_array, half disc_k_bp, bool data_aligned)
{
  MsgStereoHalf<kDispVals6>(x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2, messages_neighbor_3,
    data_costs, dst_message_array, disc_k_bp, data_aligned);
}
