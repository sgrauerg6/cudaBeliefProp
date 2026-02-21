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
 * @file KernelBpOptCPU_AVX256.h
 * @author Scott Grauer-Gray
 * @brief Defines functions used in processing belief propagation that are
 * specific to implementation with AVX256 vectorization
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef KERNEL_BP_OPT_CPU_AVX256_H_
#define KERNEL_BP_OPT_CPU_AVX256_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <immintrin.h>
#include "BpSharedFuncts/SharedBpProcessingFuncts.h"
#include "RunImpCPU/AVX256TemplateSpFuncts.h"
#include "RunImpCPU/RunCPUSettings.h"

template<unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
void beliefprop_cpu::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX256(
  beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float* data_cost_checkerboard_0, const float* data_cost_checkerboard_1,
  float* message_u_checkerboard_0, float* message_d_checkerboard_0,
  float* message_l_checkerboard_0, float* message_r_checkerboard_0,
  float* message_u_checkerboard_1, float* message_d_checkerboard_1,
  float* message_l_checkerboard_1, float* message_r_checkerboard_1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<float, __m256, DISP_VALS>(
    checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, bp_settings_disp_vals, opt_cpu_params);
}

template<unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
void beliefprop_cpu::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX256(
  beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const double* data_cost_checkerboard_0, const double* data_cost_checkerboard_1,
  double* message_u_checkerboard_0, double* message_d_checkerboard_0,
  double* message_l_checkerboard_0, double* message_r_checkerboard_0,
  double* message_u_checkerboard_1, double* message_d_checkerboard_1,
  double* message_l_checkerboard_1, double* message_r_checkerboard_1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<double, __m256d, DISP_VALS>(
    checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, bp_settings_disp_vals, opt_cpu_params);
}

template<unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
void beliefprop_cpu::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX256(
  beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* message_u_checkerboard_0, short* message_d_checkerboard_0,
  short* message_l_checkerboard_0, short* message_r_checkerboard_0,
  short* message_u_checkerboard_1, short* message_d_checkerboard_1,
  short* message_l_checkerboard_1, short* message_r_checkerboard_1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<short, __m128i, DISP_VALS>(
    checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, bp_settings_disp_vals, opt_cpu_params);
}

#if defined(FLOAT16_VECTORIZATION)
template<unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
void beliefprop_cpu::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX256(
  beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const _Float16* data_cost_checkerboard_0, const _Float16* data_cost_checkerboard_1,
  _Float16* message_u_checkerboard_0, _Float16* message_d_checkerboard_0,
  _Float16* message_l_checkerboard_0, _Float16* message_r_checkerboard_0,
  _Float16* message_u_checkerboard_1, _Float16* message_d_checkerboard_1,
  _Float16* message_l_checkerboard_1, _Float16* message_r_checkerboard_1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
    RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<_Float16, __m256h, DISP_VALS>(
      checkerboard_to_update, current_bp_level,
      data_cost_checkerboard_0, data_cost_checkerboard_1,
      message_u_checkerboard_0, message_d_checkerboard_0,
      message_l_checkerboard_0, message_r_checkerboard_0,
      message_u_checkerboard_1, message_d_checkerboard_1,
      message_l_checkerboard_1, message_r_checkerboard_1,
      disc_k_bp, bp_settings_disp_vals, opt_cpu_params);
}

#endif //FLOAT16_VECTORIZATION

template<unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
void beliefprop_cpu::RetrieveOutputDisparityUseSIMDVectorsAVX256(
  const beliefprop::BpLevelProperties& current_bp_level,
  const float* data_cost_checkerboard_0, const float* data_cost_checkerboard_1,
  const float* message_u_prev_checkerboard_0, const float* message_d_prev_checkerboard_0,
  const float* message_l_prev_checkerboard_0, const float* message_r_prev_checkerboard_0,
  const float* message_u_prev_checkerboard_1, const float* message_d_prev_checkerboard_1,
  const float* message_l_prev_checkerboard_1, const float* message_r_prev_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  RetrieveOutputDisparityUseSIMDVectors<float, __m256, float, __m256, DISP_VALS>(
    current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_prev_checkerboard_0, message_d_prev_checkerboard_0,
    message_l_prev_checkerboard_0, message_r_prev_checkerboard_0,
    message_u_prev_checkerboard_1, message_d_prev_checkerboard_1,
    message_l_prev_checkerboard_1, message_r_prev_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals,
    opt_cpu_params);
}

template<unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
void beliefprop_cpu::RetrieveOutputDisparityUseSIMDVectorsAVX256(
  const beliefprop::BpLevelProperties& current_bp_level,
  const double* data_cost_checkerboard_0, const double* data_cost_checkerboard_1,
  const double* message_u_prev_checkerboard_0, const double* message_d_prev_checkerboard_0,
  const double* message_l_prev_checkerboard_0, const double* message_r_prev_checkerboard_0,
  const double* message_u_prev_checkerboard_1, const double* message_d_prev_checkerboard_1,
  const double* message_l_prev_checkerboard_1, const double* message_r_prev_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  RetrieveOutputDisparityUseSIMDVectors<double, __m256d, double, __m256d, DISP_VALS>(
    current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_prev_checkerboard_0, message_d_prev_checkerboard_0,
    message_l_prev_checkerboard_0, message_r_prev_checkerboard_0,
    message_u_prev_checkerboard_1, message_d_prev_checkerboard_1,
    message_l_prev_checkerboard_1, message_r_prev_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals,
    opt_cpu_params);
}

template<unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
void beliefprop_cpu::RetrieveOutputDisparityUseSIMDVectorsAVX256(
  const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  const short* message_u_prev_checkerboard_0, const short* message_d_prev_checkerboard_0,
  const short* message_l_prev_checkerboard_0, const short* message_r_prev_checkerboard_0,
  const short* message_u_prev_checkerboard_1, const short* message_d_prev_checkerboard_1,
  const short* message_l_prev_checkerboard_1, const short* message_r_prev_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  RetrieveOutputDisparityUseSIMDVectors<short, __m128i, float, __m256, DISP_VALS>(
    current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_prev_checkerboard_0, message_d_prev_checkerboard_0,
    message_l_prev_checkerboard_0, message_r_prev_checkerboard_0,
    message_u_prev_checkerboard_1, message_d_prev_checkerboard_1,
    message_l_prev_checkerboard_1, message_r_prev_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals,
    opt_cpu_params);
}

#if defined(FLOAT16_VECTORIZATION)

template<unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
void beliefprop_cpu::RetrieveOutputDisparityUseSIMDVectorsAVX256(
  const beliefprop::BpLevelProperties& current_bp_level,
  const _Float16* data_cost_checkerboard_0, const _Float16* data_cost_checkerboard_1,
  const _Float16* message_u_prev_checkerboard_0, const _Float16* message_d_prev_checkerboard_0,
  const _Float16* message_l_prev_checkerboard_0, const _Float16* message_r_prev_checkerboard_0,
  const _Float16* message_u_prev_checkerboard_1, const _Float16* message_d_prev_checkerboard_1,
  const _Float16* message_l_prev_checkerboard_1, const _Float16* message_r_prev_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  RetrieveOutputDisparityUseSIMDVectors<_Float16, __m256h, _Float16, __m256h, DISP_VALS>(
    current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_prev_checkerboard_0, message_d_prev_checkerboard_0,
    message_l_prev_checkerboard_0, message_r_prev_checkerboard_0,
    message_u_prev_checkerboard_1, message_d_prev_checkerboard_1,
    message_l_prev_checkerboard_1, message_r_prev_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals,
    opt_cpu_params);
}

#endif //FLOAT16_VECTORIZATION

template<> inline void beliefprop_cpu::UpdateBestDispBestVals<__m256>(
  __m256& best_disparities, __m256& best_vals,
  const __m256& current_disparity, const __m256& vals_at_disp)
{
  __m256 maskNeedUpdate = _mm256_cmp_ps(vals_at_disp, best_vals, _CMP_LT_OS);
  best_vals = _mm256_blendv_ps(best_vals, vals_at_disp, maskNeedUpdate);
  best_disparities = _mm256_blendv_ps(best_disparities, current_disparity, maskNeedUpdate);
  /*__mmask8 maskNeedUpdate = _mm256_cmp_ps_mask(vals_at_disp, best_vals, _CMP_LT_OS);
  best_vals = _mm256_mask_blend_ps(maskNeedUpdate, best_vals, vals_at_disp);
  best_disparities = _mm256_mask_blend_ps(maskNeedUpdate, best_disparities, current_disparity);*/
}

template<> inline void beliefprop_cpu::UpdateBestDispBestVals<__m256d>(
  __m256d& best_disparities, __m256d& best_vals,
  const __m256d& current_disparity, const __m256d& vals_at_disp)
{
  __m256d maskNeedUpdate = _mm256_cmp_pd(vals_at_disp, best_vals, _CMP_LT_OS);
  best_vals = _mm256_blendv_pd(best_vals, vals_at_disp, maskNeedUpdate);
  best_disparities = _mm256_blendv_pd(best_disparities, current_disparity, maskNeedUpdate);
  /*__mmask8 maskNeedUpdate = _mm256_cmp_pd_mask(vals_at_disp, best_vals, _CMP_LT_OS);
  best_vals = _mm256_mask_blend_pd(maskNeedUpdate, best_vals, vals_at_disp);
  best_disparities = _mm256_mask_blend_pd(maskNeedUpdate, best_disparities, current_disparity);*/
}

#if defined(FLOAT16_VECTORIZATION)

template<> inline void beliefprop_cpu::UpdateBestDispBestVals<__m256h>(
  __m256h& best_disparities, __m256h& best_vals,
  const __m256h& current_disparity, const __m256h& vals_at_disp)
{
  /*__m256h maskNeedUpdate = _mm256_cmp_ph(vals_at_disp, best_vals, _CMP_LT_OS);
  best_vals = _mm256_blendv_ph(best_vals, vals_at_disp, maskNeedUpdate);
  best_disparities = _mm256_blendv_ph(best_disparities, current_disparity, maskNeedUpdate);*/
  __mmask16 maskNeedUpdate =  _mm256_cmp_ph_mask(vals_at_disp, best_vals, _CMP_LT_OS);
  best_vals = _mm256_mask_blend_ph(maskNeedUpdate, best_vals, vals_at_disp);
  best_disparities = _mm256_mask_blend_ph(maskNeedUpdate, best_disparities, current_disparity);
}

#endif //FLOAT16_VECTORIZATION

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m128i, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSetHalfSize)].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m128i messages_neighbor_1[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSetHalfSize)].num_disp_vals],
  const __m128i messages_neighbor_2[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSetHalfSize)].num_disp_vals],
  const __m128i messages_neighbor_3[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSetHalfSize)].num_disp_vals],
  const __m128i data_costs[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSetHalfSize)].num_disp_vals],
  short* dst_message_array, const __m128i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m128i, float, __m256, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSetHalfSize)].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m128i, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSet)].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m128i messages_neighbor_1[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSet)].num_disp_vals],
  const __m128i messages_neighbor_2[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSet)].num_disp_vals],
  const __m128i messages_neighbor_3[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSet)].num_disp_vals],
  const __m128i data_costs[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSet)].num_disp_vals],
  short* dst_message_array, const __m128i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m128i, float, __m256, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kTsukubaSet)].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m128i, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kVenus)].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m128i messages_neighbor_1[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kVenus)].num_disp_vals],
  const __m128i messages_neighbor_2[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kVenus)].num_disp_vals],
  const __m128i messages_neighbor_3[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kVenus)].num_disp_vals],
  const __m128i data_costs[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kVenus)].num_disp_vals],
  short* dst_message_array, const __m128i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m128i, float, __m256, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kVenus)].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m128i, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kBarn1)].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m128i messages_neighbor_1[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kBarn1)].num_disp_vals],
  const __m128i messages_neighbor_2[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kBarn1)].num_disp_vals],
  const __m128i messages_neighbor_3[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kBarn1)].num_disp_vals],
  const __m128i data_costs[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kBarn1)].num_disp_vals],
  short* dst_message_array, const __m128i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m128i, float, __m256, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kBarn1)].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m128i, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesQuarterSize)].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m128i messages_neighbor_1[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesQuarterSize)].num_disp_vals],
  const __m128i messages_neighbor_2[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesQuarterSize)].num_disp_vals],
  const __m128i messages_neighbor_3[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesQuarterSize)].num_disp_vals],
  const __m128i data_costs[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesQuarterSize)].num_disp_vals],
  short* dst_message_array, const __m128i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m128i, float, __m256, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesQuarterSize)].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m128i, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesHalfSize)].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m128i messages_neighbor_1[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesHalfSize)].num_disp_vals],
  const __m128i messages_neighbor_2[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesHalfSize)].num_disp_vals],
  const __m128i messages_neighbor_3[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesHalfSize)].num_disp_vals],
  const __m128i data_costs[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesHalfSize)].num_disp_vals],
  short* dst_message_array, const __m128i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m128i, float, __m256, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesHalfSize)].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m128i, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesFullSizeCropped)].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m128i messages_neighbor_1[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesFullSizeCropped)].num_disp_vals],
  const __m128i messages_neighbor_2[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesFullSizeCropped)].num_disp_vals],
  const __m128i messages_neighbor_3[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesFullSizeCropped)].num_disp_vals],
  const __m128i data_costs[beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesFullSizeCropped)].num_disp_vals],
  short* dst_message_array, const __m128i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m128i, float, __m256, beliefprop::kStereoSetsToProcess[static_cast<size_t>(beliefprop::Stereo_Set::kConesFullSizeCropped)].num_disp_vals>(
    x_val, y_val, current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

template<> inline void beliefprop_cpu::MsgStereoSIMD<short, __m128i>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m128i* messages_neighbor_1, const __m128i* messages_neighbor_2,
  const __m128i* messages_neighbor_3, const __m128i* data_costs,
  short* dst_message_array, const __m128i& disc_k_bp, bool data_aligned,
  unsigned int bp_settings_disp_vals)
{
  MsgStereoSIMDProcessing<short, __m128i, float, __m256>(x_val, y_val, current_bp_level,
    messages_neighbor_1, messages_neighbor_2, messages_neighbor_3, data_costs,
    dst_message_array, disc_k_bp, data_aligned, bp_settings_disp_vals);
}

#endif /* KERNEL_BP_OPT_CPU_AVX256_H_ */
