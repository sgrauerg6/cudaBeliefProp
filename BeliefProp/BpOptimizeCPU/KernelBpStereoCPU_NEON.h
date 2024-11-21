/*
 * KernelBpStereoCPU_NEON.h
 *
 *  Created on: Jun 23, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_NEON_H_
#define KERNELBPSTEREOCPU_NEON_H_

//this is only used when processing using an ARM CPU with NEON instructions
#include <arm_neon.h>
#include "RunImpCPU/NEONTemplateSpFuncts.h"

template<unsigned int DISP_VALS>
void beliefpropCPU::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsNEON(
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  float* data_cost_checkerboard_0, float* data_cost_checkerboard_1,
  float* message_u_checkerboard_0, float* message_d_checkerboard_0,
  float* message_l_checkerboard_0, float* message_r_checkerboard_0,
  float* message_u_checkerboard_1, float* message_d_checkerboard_1,
  float* message_l_checkerboard_1, float* message_r_checkerboard_1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  constexpr unsigned int num_data_SIMD_vect{4};
  RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<float, float32x4_t, DISP_VALS>(
    checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, num_data_SIMD_vect, bp_settings_disp_vals, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsNEON(
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  float16_t* data_cost_checkerboard_0, float16_t* data_cost_checkerboard_1,
  float16_t* message_u_checkerboard_0, float16_t* message_d_checkerboard_0,
  float16_t* message_l_checkerboard_0, float16_t* message_r_checkerboard_0,
  float16_t* message_u_checkerboard_1, float16_t* message_d_checkerboard_1,
  float16_t* message_l_checkerboard_1, float16_t* message_r_checkerboard_1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  constexpr unsigned int num_data_SIMD_vect{4};
  RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<float16_t, float16x4_t, DISP_VALS>(
    checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, num_data_SIMD_vect, bp_settings_disp_vals, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsNEON(
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  double* data_cost_checkerboard_0, double* data_cost_checkerboard_1,
  double* message_u_checkerboard_0, double* message_d_checkerboard_0,
  double* message_l_checkerboard_0, double* message_r_checkerboard_0,
  double* message_u_checkerboard_1, double* message_d_checkerboard_1,
  double* message_l_checkerboard_1, double* message_r_checkerboard_1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  constexpr unsigned int num_data_SIMD_vect{2};
  RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<double, float64x2_t, DISP_VALS>(
    checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, num_data_SIMD_vect, bp_settings_disp_vals, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::RetrieveOutputDisparityUseSIMDVectorsNEON(
  const beliefprop::BpLevelProperties& current_bp_level,
  float* data_cost_checkerboard_0, float* data_cost_checkerboard_1,
  float* message_u_prev_checkerboard_0, float* message_d_prev_checkerboard_0,
  float* message_l_prev_checkerboard_0, float* message_r_prev_checkerboard_0,
  float* message_u_prev_checkerboard_1, float* message_d_prev_checkerboard_1,
  float* message_l_prev_checkerboard_1, float* message_r_prev_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  constexpr unsigned int num_data_SIMD_vect{4};
  RetrieveOutputDisparityUseSIMDVectors<float, float32x4_t, float, float32x4_t, DISP_VALS>(current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_prev_checkerboard_0, message_d_prev_checkerboard_0,
    message_l_prev_checkerboard_0, message_r_prev_checkerboard_0,
    message_u_prev_checkerboard_1, message_d_prev_checkerboard_1,
    message_l_prev_checkerboard_1, message_r_prev_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals,
    num_data_SIMD_vect, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::RetrieveOutputDisparityUseSIMDVectorsNEON(
  const beliefprop::BpLevelProperties& current_bp_level,
  float16_t* data_cost_checkerboard_0, float16_t* data_cost_checkerboard_1,
  float16_t* message_u_prev_checkerboard_0, float16_t* message_d_prev_checkerboard_0,
  float16_t* message_l_prev_checkerboard_0, float16_t* message_r_prev_checkerboard_0,
  float16_t* message_u_prev_checkerboard_1, float16_t* message_d_prev_checkerboard_1,
  float16_t* message_l_prev_checkerboard_1, float16_t* message_r_prev_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  constexpr unsigned int num_data_SIMD_vect{4};
  RetrieveOutputDisparityUseSIMDVectors<float16_t, float16x4_t, float, float32x4_t, DISP_VALS>(current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_prev_checkerboard_0, message_d_prev_checkerboard_0,
    message_l_prev_checkerboard_0, message_r_prev_checkerboard_0,
    message_u_prev_checkerboard_1, message_d_prev_checkerboard_1,
    message_l_prev_checkerboard_1, message_r_prev_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals,
    num_data_SIMD_vect, opt_cpu_params);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::RetrieveOutputDisparityUseSIMDVectorsNEON(
  const beliefprop::BpLevelProperties& current_bp_level,
  double* data_cost_checkerboard_0, double* data_cost_checkerboard_1,
  double* message_u_prev_checkerboard_0, double* message_d_prev_checkerboard_0,
  double* message_l_prev_checkerboard_0, double* message_r_prev_checkerboard_0,
  double* message_u_prev_checkerboard_1, double* message_d_prev_checkerboard_1,
  double* message_l_prev_checkerboard_1, double* message_r_prev_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{      
  constexpr unsigned int num_data_SIMD_vect{2};
  RetrieveOutputDisparityUseSIMDVectors<double, float64x2_t, double, float64x2_t, DISP_VALS>(current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_prev_checkerboard_0, message_d_prev_checkerboard_0,
    message_l_prev_checkerboard_0, message_r_prev_checkerboard_0,
    message_u_prev_checkerboard_1, message_d_prev_checkerboard_1,
    message_l_prev_checkerboard_1, message_r_prev_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals,
    num_data_SIMD_vect, opt_cpu_params);
}

template<> inline void beliefpropCPU::UpdateBestDispBestVals<float32x4_t>(float32x4_t& best_disparities, float32x4_t& best_vals,
  const float32x4_t& current_disparity, const float32x4_t& val_at_disp)
{
  //get mask with value 1 where current value less then current best 1, 0 otherwise
  uint32x4_t maskUpdateVals = vcltq_f32(val_at_disp, best_vals);
  //update best values and best disparities using mask
  //vbslq_f32 operation uses first float32x4_t argument if mask value is 1 and seconde float32x4_t argument if mask value is 0
  best_vals = vbslq_f32(maskUpdateVals, val_at_disp, best_vals);
  best_disparities = vbslq_f32(maskUpdateVals, current_disparity, best_disparities);
}

template<> inline void beliefpropCPU::UpdateBestDispBestVals<float64x2_t>(float64x2_t& best_disparities, float64x2_t& best_vals,
  const float64x2_t& current_disparity, const float64x2_t& val_at_disp)
{
  uint64x2_t maskUpdateVals = vcltq_f64(val_at_disp, best_vals);
  best_vals = vbslq_f64(maskUpdateVals, val_at_disp, best_vals);
  best_disparities = vbslq_f64(maskUpdateVals, current_disparity, best_disparities);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<float16_t, float16x4_t, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16x4_t messageValsNeighbor1[beliefprop::kStereoSetsToProcess[0].num_disp_vals],
  float16x4_t messageValsNeighbor2[beliefprop::kStereoSetsToProcess[0].num_disp_vals],
  float16x4_t messageValsNeighbor3[beliefprop::kStereoSetsToProcess[0].num_disp_vals],
  float16x4_t data_costs[beliefprop::kStereoSetsToProcess[0].num_disp_vals],
  float16_t* dst_message_array, const float16x4_t& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<float16_t, float16x4_t, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16x4_t messageValsNeighbor1[beliefprop::kStereoSetsToProcess[1].num_disp_vals],
  float16x4_t messageValsNeighbor2[beliefprop::kStereoSetsToProcess[1].num_disp_vals],
  float16x4_t messageValsNeighbor3[beliefprop::kStereoSetsToProcess[1].num_disp_vals],
  float16x4_t data_costs[beliefprop::kStereoSetsToProcess[1].num_disp_vals],
  float16_t* dst_message_array, const float16x4_t& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<float16_t, float16x4_t, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16x4_t messageValsNeighbor1[beliefprop::kStereoSetsToProcess[2].num_disp_vals],
  float16x4_t messageValsNeighbor2[beliefprop::kStereoSetsToProcess[2].num_disp_vals],
  float16x4_t messageValsNeighbor3[beliefprop::kStereoSetsToProcess[2].num_disp_vals],
  float16x4_t data_costs[beliefprop::kStereoSetsToProcess[2].num_disp_vals],
  float16_t* dst_message_array, const float16x4_t& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<float16_t, float16x4_t, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16x4_t messageValsNeighbor1[beliefprop::kStereoSetsToProcess[3].num_disp_vals],
  float16x4_t messageValsNeighbor2[beliefprop::kStereoSetsToProcess[3].num_disp_vals],
  float16x4_t messageValsNeighbor3[beliefprop::kStereoSetsToProcess[3].num_disp_vals],
  float16x4_t data_costs[beliefprop::kStereoSetsToProcess[3].num_disp_vals],
  float16_t* dst_message_array, const float16x4_t& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<float16_t, float16x4_t, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16x4_t messageValsNeighbor1[beliefprop::kStereoSetsToProcess[4].num_disp_vals],
  float16x4_t messageValsNeighbor2[beliefprop::kStereoSetsToProcess[4].num_disp_vals],
  float16x4_t messageValsNeighbor3[beliefprop::kStereoSetsToProcess[4].num_disp_vals],
  float16x4_t data_costs[beliefprop::kStereoSetsToProcess[4].num_disp_vals],
  float16_t* dst_message_array, const float16x4_t& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<float16_t, float16x4_t, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16x4_t messageValsNeighbor1[beliefprop::kStereoSetsToProcess[5].num_disp_vals],
  float16x4_t messageValsNeighbor2[beliefprop::kStereoSetsToProcess[5].num_disp_vals],
  float16x4_t messageValsNeighbor3[beliefprop::kStereoSetsToProcess[5].num_disp_vals],
  float16x4_t data_costs[beliefprop::kStereoSetsToProcess[5].num_disp_vals],
  float16_t* dst_message_array, const float16x4_t& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<> inline void beliefpropCPU::MsgStereoSIMD<float16_t, float16x4_t, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16x4_t messageValsNeighbor1[beliefprop::kStereoSetsToProcess[6].num_disp_vals],
  float16x4_t messageValsNeighbor2[beliefprop::kStereoSetsToProcess[6].num_disp_vals],
  float16x4_t messageValsNeighbor3[beliefprop::kStereoSetsToProcess[6].num_disp_vals],
  float16x4_t data_costs[beliefprop::kStereoSetsToProcess[6].num_disp_vals],
  float16_t* dst_message_array, const float16x4_t& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
    x_val, y_val, current_bp_level, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

template<> inline void beliefpropCPU::MsgStereoSIMD<float16_t, float16x4_t>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16x4_t* messageValsNeighbor1, float16x4_t* messageValsNeighbor2,
  float16x4_t* messageValsNeighbor3, float16x4_t* data_costs,
  float16_t* dst_message_array, const float16x4_t& disc_k_bp, bool data_aligned,
  unsigned int bp_settings_disp_vals)
{
  MsgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t>(x_val, y_val, current_bp_level,
    messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3, data_costs,
    dst_message_array, disc_k_bp, data_aligned, bp_settings_disp_vals);
}

#endif /* KERNELBPSTEREOCPU_NEON_H_ */
