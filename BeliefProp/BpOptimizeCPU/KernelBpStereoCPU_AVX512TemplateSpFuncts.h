/*
 * KernelBpStereoCPU_AVX512TemplateSpFuncts.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "BpSharedFuncts/SharedBPProcessingFuncts.h"
#include "RunImpCPU/AVX512TemplateSpFuncts.h"

template<unsigned int DISP_VALS>
void beliefpropCPU::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512(
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const float* dataCostStereoCheckerboard0, const float* dataCostStereoCheckerboard1,
  float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
  float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
  float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
  float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  constexpr unsigned int num_data_SIMD_vect{16u};
  RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<float, __m512, DISP_VALS>(
    checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, num_data_SIMD_vect, bp_settings_disp_vals, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512(
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  constexpr unsigned int num_data_SIMD_vect{16u};
  RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<short, __m256i, DISP_VALS>(
    checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, num_data_SIMD_vect, bp_settings_disp_vals, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512(
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const double* dataCostStereoCheckerboard0, const double* dataCostStereoCheckerboard1,
  double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
  double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
  double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
  double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  constexpr unsigned int num_data_SIMD_vect{8u};
  RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<double, __m512d, DISP_VALS>(
    checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, num_data_SIMD_vect, bp_settings_disp_vals, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::RetrieveOutputDisparityUseSIMDVectorsAVX512(
  const beliefprop::BpLevelProperties& current_bp_level,
  const float* dataCostStereoCheckerboard0, const float* dataCostStereoCheckerboard1,
  const float* messageUPrevStereoCheckerboard0, const float* messageDPrevStereoCheckerboard0,
  const float* messageLPrevStereoCheckerboard0, const float* messageRPrevStereoCheckerboard0,
  const float* messageUPrevStereoCheckerboard1, const float* messageDPrevStereoCheckerboard1,
  const float* messageLPrevStereoCheckerboard1, const float* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  constexpr unsigned int num_data_SIMD_vect{16u};
  RetrieveOutputDisparityUseSIMDVectors<float, __m512, float, __m512, DISP_VALS>(current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparity_between_images_device, bp_settings_disp_vals,
    num_data_SIMD_vect, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::RetrieveOutputDisparityUseSIMDVectorsAVX512(
  const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  const short* messageUPrevStereoCheckerboard0, const short* messageDPrevStereoCheckerboard0,
  const short* messageLPrevStereoCheckerboard0, const short* messageRPrevStereoCheckerboard0,
  const short* messageUPrevStereoCheckerboard1, const short* messageDPrevStereoCheckerboard1,
  const short* messageLPrevStereoCheckerboard1, const short* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  constexpr unsigned int num_data_SIMD_vect{16u};
  RetrieveOutputDisparityUseSIMDVectors<short, __m256i, float, __m512, DISP_VALS>(current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparity_between_images_device, bp_settings_disp_vals,
    num_data_SIMD_vect, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::RetrieveOutputDisparityUseSIMDVectorsAVX512(
  const beliefprop::BpLevelProperties& current_bp_level,
  const double* dataCostStereoCheckerboard0, const double* dataCostStereoCheckerboard1,
  const double* messageUPrevStereoCheckerboard0, const double* messageDPrevStereoCheckerboard0,
  const double* messageLPrevStereoCheckerboard0, const double* messageRPrevStereoCheckerboard0,
  const double* messageUPrevStereoCheckerboard1, const double* messageDPrevStereoCheckerboard1,
  const double* messageLPrevStereoCheckerboard1, const double* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  constexpr unsigned int num_data_SIMD_vect{8u};
  RetrieveOutputDisparityUseSIMDVectors<double, __m512d, double, __m512d, DISP_VALS>(current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparity_between_images_device, bp_settings_disp_vals,
    num_data_SIMD_vect, opt_cpu_params);
}

template<> inline void beliefpropCPU::UpdateBestDispBestVals<__m512>(__m512& best_disparities, __m512& best_vals,
  const __m512& current_disparity, const __m512& val_at_disp)
{
  __mmask16 maskNeedUpdate =  _mm512_cmp_ps_mask(val_at_disp, best_vals, _CMP_LT_OS);
  best_vals = _mm512_mask_blend_ps(maskNeedUpdate, best_vals, val_at_disp);
  best_disparities = _mm512_mask_blend_ps(maskNeedUpdate, best_disparities, current_disparity);
}

template<> inline void beliefpropCPU::UpdateBestDispBestVals<__m512d>(__m512d& best_disparities, __m512d& best_vals,
  const __m512d& current_disparity, const __m512d& val_at_disp)
{
  __mmask16 maskNeedUpdate =  _mm512_cmp_pd_mask(val_at_disp, best_vals, _CMP_LT_OS);
  best_vals = _mm512_mask_blend_pd(maskNeedUpdate, best_vals, val_at_disp);
  best_disparities = _mm512_mask_blend_pd(maskNeedUpdate, best_disparities, current_disparity);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messageValsNeighbor1[beliefprop::kStereoSetsToProcess[0].num_disp_vals],
  const __m256i messageValsNeighbor2[beliefprop::kStereoSetsToProcess[0].num_disp_vals],
  const __m256i messageValsNeighbor3[beliefprop::kStereoSetsToProcess[0].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[0].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messageValsNeighbor1[beliefprop::kStereoSetsToProcess[1].num_disp_vals],
  const __m256i messageValsNeighbor2[beliefprop::kStereoSetsToProcess[1].num_disp_vals],
  const __m256i messageValsNeighbor3[beliefprop::kStereoSetsToProcess[1].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[1].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messageValsNeighbor1[beliefprop::kStereoSetsToProcess[2].num_disp_vals],
  const __m256i messageValsNeighbor2[beliefprop::kStereoSetsToProcess[2].num_disp_vals],
  const __m256i messageValsNeighbor3[beliefprop::kStereoSetsToProcess[2].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[2].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messageValsNeighbor1[beliefprop::kStereoSetsToProcess[3].num_disp_vals],
  const __m256i messageValsNeighbor2[beliefprop::kStereoSetsToProcess[3].num_disp_vals],
  const __m256i messageValsNeighbor3[beliefprop::kStereoSetsToProcess[3].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[3].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messageValsNeighbor1[beliefprop::kStereoSetsToProcess[4].num_disp_vals],
  const __m256i messageValsNeighbor2[beliefprop::kStereoSetsToProcess[4].num_disp_vals],
  const __m256i messageValsNeighbor3[beliefprop::kStereoSetsToProcess[4].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[4].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messageValsNeighbor1[beliefprop::kStereoSetsToProcess[5].num_disp_vals],
  const __m256i messageValsNeighbor2[beliefprop::kStereoSetsToProcess[5].num_disp_vals],
  const __m256i messageValsNeighbor3[beliefprop::kStereoSetsToProcess[5].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[5].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i messageValsNeighbor1[beliefprop::kStereoSetsToProcess[6].num_disp_vals],
  const __m256i messageValsNeighbor2[beliefprop::kStereoSetsToProcess[6].num_disp_vals],
  const __m256i messageValsNeighbor3[beliefprop::kStereoSetsToProcess[6].num_disp_vals],
  const __m256i data_costs[beliefprop::kStereoSetsToProcess[6].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const __m256i* messageValsNeighbor1, const __m256i* messageValsNeighbor2,
  const __m256i* messageValsNeighbor3, const __m256i* data_costs,
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned,
  unsigned int bp_settings_disp_vals)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512>(x_val, y_val, current_bp_level,
    messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3, data_costs,
    dst_message_array, disc_k_bp, data_aligned, bp_settings_disp_vals);
}

#endif /* KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_ */
