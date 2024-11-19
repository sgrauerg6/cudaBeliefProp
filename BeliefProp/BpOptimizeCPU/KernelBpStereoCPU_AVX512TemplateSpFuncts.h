/*
 * KernelBpStereoCPU_kAVX512TemplateSpFuncts.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_kAVX512TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_kAVX512TEMPLATESPFUNCTS_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "BpSharedFuncts/SharedBPProcessingFuncts.h"
#include "RunImpCPU/AVX512TemplateSpFuncts.h"

template<unsigned int DISP_VALS>
void beliefpropCPU::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorskAVX512(
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
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
void beliefpropCPU::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorskAVX512(
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
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
void beliefpropCPU::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorskAVX512(
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
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
void beliefpropCPU::RetrieveOutputDisparityUseSIMDVectorskAVX512(
  const beliefprop::BpLevelProperties& current_bp_level,
  float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
  float* messageUPrevStereoCheckerboard0, float* messageDPrevStereoCheckerboard0,
  float* messageLPrevStereoCheckerboard0, float* messageRPrevStereoCheckerboard0,
  float* messageUPrevStereoCheckerboard1, float* messageDPrevStereoCheckerboard1,
  float* messageLPrevStereoCheckerboard1, float* messageRPrevStereoCheckerboard1,
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
void beliefpropCPU::RetrieveOutputDisparityUseSIMDVectorskAVX512(
  const beliefprop::BpLevelProperties& current_bp_level,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0,
  short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1,
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
void beliefpropCPU::RetrieveOutputDisparityUseSIMDVectorskAVX512(
  const beliefprop::BpLevelProperties& current_bp_level,
  double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
  double* messageUPrevStereoCheckerboard0, double* messageDPrevStereoCheckerboard0,
  double* messageLPrevStereoCheckerboard0, double* messageRPrevStereoCheckerboard0,
  double* messageUPrevStereoCheckerboard1, double* messageDPrevStereoCheckerboard1,
  double* messageLPrevStereoCheckerboard1, double* messageRPrevStereoCheckerboard1,
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
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[0].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[0].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[0].num_disp_vals],
  __m256i data_costs[bp_params::kStereoSetsToProcess[0].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[1].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[1].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[1].num_disp_vals],
  __m256i data_costs[bp_params::kStereoSetsToProcess[1].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[2].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[2].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[2].num_disp_vals],
  __m256i data_costs[bp_params::kStereoSetsToProcess[2].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[3].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[3].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[3].num_disp_vals],
  __m256i data_costs[bp_params::kStereoSetsToProcess[3].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[4].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[4].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[4].num_disp_vals],
  __m256i data_costs[bp_params::kStereoSetsToProcess[4].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[5].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[5].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[5].num_disp_vals],
  __m256i data_costs[bp_params::kStereoSetsToProcess[5].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[6].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[6].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[6].num_disp_vals],
  __m256i data_costs[bp_params::kStereoSetsToProcess[6].num_disp_vals],
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

template<> inline void beliefpropCPU::MsgStereoSIMD<short, __m256i>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  __m256i* messageValsNeighbor1, __m256i* messageValsNeighbor2,
  __m256i* messageValsNeighbor3, __m256i* data_costs,
  short* dst_message_array, const __m256i& disc_k_bp, bool data_aligned,
  unsigned int bp_settings_disp_vals)
{
  MsgStereoSIMDProcessing<short, __m256i, float, __m512>(x_val, y_val, current_bp_level,
    messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3, data_costs,
    dst_message_array, disc_k_bp, data_aligned, bp_settings_disp_vals);
}

#endif /* KERNELBPSTEREOCPU_kAVX512TEMPLATESPFUNCTS_H_ */
