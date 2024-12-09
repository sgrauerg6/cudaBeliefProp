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
 * @file KernelBpStereoCPU_AVX512TemplateSpFuncts.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "BpSharedFuncts/SharedBpProcessingFuncts.h"
#include "RunImpCPU/AVX512TemplateSpFuncts.h"

template<unsigned int DISP_VALS>
void beliefprop_cpu::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512(
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const float* data_cost_checkerboard_0, const float* data_cost_checkerboard_1,
  float* message_u_checkerboard_0, float* message_d_checkerboard_0,
  float* message_l_checkerboard_0, float* message_r_checkerboard_0,
  float* message_u_checkerboard_1, float* message_d_checkerboard_1,
  float* message_l_checkerboard_1, float* message_r_checkerboard_1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  constexpr unsigned int simd_data_size{16};
  RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<float, __m512, DISP_VALS>(
    checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, simd_data_size, bp_settings_disp_vals, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefprop_cpu::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512(
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* message_u_checkerboard_0, short* message_d_checkerboard_0,
  short* message_l_checkerboard_0, short* message_r_checkerboard_0,
  short* message_u_checkerboard_1, short* message_d_checkerboard_1,
  short* message_l_checkerboard_1, short* message_r_checkerboard_1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  constexpr unsigned int simd_data_size{16};
  RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<short, __m256i, DISP_VALS>(
    checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, simd_data_size, bp_settings_disp_vals, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefprop_cpu::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512(
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const double* data_cost_checkerboard_0, const double* data_cost_checkerboard_1,
  double* message_u_checkerboard_0, double* message_d_checkerboard_0,
  double* message_l_checkerboard_0, double* message_r_checkerboard_0,
  double* message_u_checkerboard_1, double* message_d_checkerboard_1,
  double* message_l_checkerboard_1, double* message_r_checkerboard_1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  constexpr unsigned int simd_data_size{8};
  RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<double, __m512d, DISP_VALS>(
    checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, simd_data_size, bp_settings_disp_vals, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefprop_cpu::RetrieveOutputDisparityUseSIMDVectorsAVX512(
  const beliefprop::BpLevelProperties& current_bp_level,
  const float* data_cost_checkerboard_0, const float* data_cost_checkerboard_1,
  const float* message_u_prev_checkerboard_0, const float* message_d_prev_checkerboard_0,
  const float* message_l_prev_checkerboard_0, const float* message_r_prev_checkerboard_0,
  const float* message_u_prev_checkerboard_1, const float* message_d_prev_checkerboard_1,
  const float* message_l_prev_checkerboard_1, const float* message_r_prev_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  constexpr unsigned int simd_data_size{16};
  RetrieveOutputDisparityUseSIMDVectors<float, __m512, float, __m512, DISP_VALS>(current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_prev_checkerboard_0, message_d_prev_checkerboard_0,
    message_l_prev_checkerboard_0, message_r_prev_checkerboard_0,
    message_u_prev_checkerboard_1, message_d_prev_checkerboard_1,
    message_l_prev_checkerboard_1, message_r_prev_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals,
    simd_data_size, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefprop_cpu::RetrieveOutputDisparityUseSIMDVectorsAVX512(
  const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  const short* message_u_prev_checkerboard_0, const short* message_d_prev_checkerboard_0,
  const short* message_l_prev_checkerboard_0, const short* message_r_prev_checkerboard_0,
  const short* message_u_prev_checkerboard_1, const short* message_d_prev_checkerboard_1,
  const short* message_l_prev_checkerboard_1, const short* message_r_prev_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  constexpr unsigned int simd_data_size{16};
  RetrieveOutputDisparityUseSIMDVectors<short, __m256i, float, __m512, DISP_VALS>(current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_prev_checkerboard_0, message_d_prev_checkerboard_0,
    message_l_prev_checkerboard_0, message_r_prev_checkerboard_0,
    message_u_prev_checkerboard_1, message_d_prev_checkerboard_1,
    message_l_prev_checkerboard_1, message_r_prev_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals,
    simd_data_size, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefprop_cpu::RetrieveOutputDisparityUseSIMDVectorsAVX512(
  const beliefprop::BpLevelProperties& current_bp_level,
  const double* data_cost_checkerboard_0, const double* data_cost_checkerboard_1,
  const double* message_u_prev_checkerboard_0, const double* message_d_prev_checkerboard_0,
  const double* message_l_prev_checkerboard_0, const double* message_r_prev_checkerboard_0,
  const double* message_u_prev_checkerboard_1, const double* message_d_prev_checkerboard_1,
  const double* message_l_prev_checkerboard_1, const double* message_r_prev_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  constexpr unsigned int simd_data_size{8};
  RetrieveOutputDisparityUseSIMDVectors<double, __m512d, double, __m512d, DISP_VALS>(current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_prev_checkerboard_0, message_d_prev_checkerboard_0,
    message_l_prev_checkerboard_0, message_r_prev_checkerboard_0,
    message_u_prev_checkerboard_1, message_d_prev_checkerboard_1,
    message_l_prev_checkerboard_1, message_r_prev_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals,
    simd_data_size, opt_cpu_params);
}

template<> inline void beliefprop_cpu::UpdateBestDispBestVals<__m512>(__m512& best_disparities, __m512& best_vals,
  const __m512& current_disparity, const __m512& val_at_disp)
{
  __mmask16 maskNeedUpdate =  _mm512_cmp_ps_mask(val_at_disp, best_vals, _CMP_LT_OS);
  best_vals = _mm512_mask_blend_ps(maskNeedUpdate, best_vals, val_at_disp);
  best_disparities = _mm512_mask_blend_ps(maskNeedUpdate, best_disparities, current_disparity);
}

template<> inline void beliefprop_cpu::UpdateBestDispBestVals<__m512d>(__m512d& best_disparities, __m512d& best_vals,
  const __m512d& current_disparity, const __m512d& val_at_disp)
{
  __mmask16 maskNeedUpdate =  _mm512_cmp_pd_mask(val_at_disp, best_vals, _CMP_LT_OS);
  best_vals = _mm512_mask_blend_pd(maskNeedUpdate, best_vals, val_at_disp);
  best_disparities = _mm512_mask_blend_pd(maskNeedUpdate, best_disparities, current_disparity);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messages_neighbor_1[beliefprop::kStereoSetsToProcess[0].num_disp_vals],
  const __m256i messages_neighbor_2[beliefprop::kStereoSetsToProcess[0].num_disp_vals],
  const __m256i messages_neighbor_3[beliefprop::kStereoSetsToProcess[0].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[0].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messages_neighbor_1[beliefprop::kStereoSetsToProcess[1].num_disp_vals],
  const __m256i messages_neighbor_2[beliefprop::kStereoSetsToProcess[1].num_disp_vals],
  const __m256i messages_neighbor_3[beliefprop::kStereoSetsToProcess[1].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[1].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messages_neighbor_1[beliefprop::kStereoSetsToProcess[2].num_disp_vals],
  const __m256i messages_neighbor_2[beliefprop::kStereoSetsToProcess[2].num_disp_vals],
  const __m256i messages_neighbor_3[beliefprop::kStereoSetsToProcess[2].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[2].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messages_neighbor_1[beliefprop::kStereoSetsToProcess[3].num_disp_vals],
  const __m256i messages_neighbor_2[beliefprop::kStereoSetsToProcess[3].num_disp_vals],
  const __m256i messages_neighbor_3[beliefprop::kStereoSetsToProcess[3].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[3].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messages_neighbor_1[beliefprop::kStereoSetsToProcess[4].num_disp_vals],
  const __m256i messages_neighbor_2[beliefprop::kStereoSetsToProcess[4].num_disp_vals],
  const __m256i messages_neighbor_3[beliefprop::kStereoSetsToProcess[4].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[4].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messages_neighbor_1[beliefprop::kStereoSetsToProcess[5].num_disp_vals],
  const __m256i messages_neighbor_2[beliefprop::kStereoSetsToProcess[5].num_disp_vals],
  const __m256i messages_neighbor_3[beliefprop::kStereoSetsToProcess[5].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[5].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messages_neighbor_1[beliefprop::kStereoSetsToProcess[6].num_disp_vals],
  const __m256i messages_neighbor_2[beliefprop::kStereoSetsToProcess[6].num_disp_vals],
  const __m256i messages_neighbor_3[beliefprop::kStereoSetsToProcess[6].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[6].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m256i>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i* messages_neighbor_1, const __m256i* messages_neighbor_2,
  const __m256i* messages_neighbor_3, const __m256i* data_costs,
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned,
  unsigned int bp_settings_disp_vals)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512>(x_val, y_val, current_bp_level,
    messages_neighbor_1, messages_neighbor_2, messages_neighbor_3, data_costs,
    dst_message_array, disc_k_bp, data_aligned, bp_settings_disp_vals);
}

#endif /* KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_ */
