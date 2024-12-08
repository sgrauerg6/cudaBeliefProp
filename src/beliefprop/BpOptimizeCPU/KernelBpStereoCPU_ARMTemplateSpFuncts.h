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
 * @file KernelBpStereoCPU_ARMTemplateSpFuncts.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_

#include "BpSharedFuncts/SharedBpProcessingFuncts.h"
#include "RunImpCPU/ARMTemplateSpFuncts.h"
#include "KernelBpStereoCPU.h"

#ifdef COMPILING_FOR_ARM

#include <arm_neon.h>

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, 0>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* message_u_checkerboard_0, float16_t* message_d_checkerboard_0,
  float16_t* message_l_checkerboard_0, float16_t* message_r_checkerboard_0,
  float16_t* message_u_checkerboard_1, float16_t* message_d_checkerboard_1,
  float16_t* message_l_checkerboard_1, float16_t* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, 0>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* message_u_checkerboard_0, float16_t* message_d_checkerboard_0,
  float16_t* message_l_checkerboard_0, float16_t* message_r_checkerboard_0,
  float16_t* message_u_checkerboard_1, float16_t* message_d_checkerboard_1,
  float16_t* message_l_checkerboard_1, float16_t* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* message_u_checkerboard_0, float16_t* message_d_checkerboard_0,
  float16_t* message_l_checkerboard_0, float16_t* message_r_checkerboard_0,
  float16_t* message_u_checkerboard_1, float16_t* message_d_checkerboard_1,
  float16_t* message_l_checkerboard_1, float16_t* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* message_u_checkerboard_0, float16_t* message_d_checkerboard_0,
  float16_t* message_l_checkerboard_0, float16_t* message_r_checkerboard_0,
  float16_t* message_u_checkerboard_1, float16_t* message_d_checkerboard_1,
  float16_t* message_l_checkerboard_1, float16_t* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* message_u_checkerboard_0, float16_t* message_d_checkerboard_0,
  float16_t* message_l_checkerboard_0, float16_t* message_r_checkerboard_0,
  float16_t* message_u_checkerboard_1, float16_t* message_d_checkerboard_1,
  float16_t* message_l_checkerboard_1, float16_t* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* message_u_checkerboard_0, float16_t* message_d_checkerboard_0,
  float16_t* message_l_checkerboard_0, float16_t* message_r_checkerboard_0,
  float16_t* message_u_checkerboard_1, float16_t* message_d_checkerboard_1,
  float16_t* message_l_checkerboard_1, float16_t* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* message_u_checkerboard_0, float16_t* message_d_checkerboard_0,
  float16_t* message_l_checkerboard_0, float16_t* message_r_checkerboard_0,
  float16_t* message_u_checkerboard_1, float16_t* message_d_checkerboard_1,
  float16_t* message_l_checkerboard_1, float16_t* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* message_u_checkerboard_0, float16_t* message_d_checkerboard_0,
  float16_t* message_l_checkerboard_0, float16_t* message_r_checkerboard_0,
  float16_t* message_u_checkerboard_1, float16_t* message_d_checkerboard_1,
  float16_t* message_l_checkerboard_1, float16_t* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, 0>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, 0>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  float16_t* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level, offset_num, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, 0>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  const float16_t* message_u_checkerboard_0, const float16_t* message_d_checkerboard_0,
  const float16_t* message_l_checkerboard_0, const float16_t* message_r_checkerboard_0,
  const float16_t* message_u_checkerboard_1, const float16_t* message_d_checkerboard_1,
  const float16_t* message_l_checkerboard_1, const float16_t* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, 0>(x_val, y_val, current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1, message_u_checkerboard_0,
    message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0, message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1,
    message_r_checkerboard_1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  const float16_t* message_u_checkerboard_0, const float16_t* message_d_checkerboard_0,
  const float16_t* message_l_checkerboard_0, const float16_t* message_r_checkerboard_0,
  const float16_t* message_u_checkerboard_1, const float16_t* message_d_checkerboard_1,
  const float16_t* message_l_checkerboard_1, const float16_t* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(x_val, y_val, current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1, message_u_checkerboard_0,
    message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0, message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1,
    message_r_checkerboard_1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  const float16_t* message_u_checkerboard_0, const float16_t* message_d_checkerboard_0,
  const float16_t* message_l_checkerboard_0, const float16_t* message_r_checkerboard_0,
  const float16_t* message_u_checkerboard_1, const float16_t* message_d_checkerboard_1,
  const float16_t* message_l_checkerboard_1, const float16_t* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(x_val, y_val, current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1, message_u_checkerboard_0,
    message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0, message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1,
    message_r_checkerboard_1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  const float16_t* message_u_checkerboard_0, const float16_t* message_d_checkerboard_0,
  const float16_t* message_l_checkerboard_0, const float16_t* message_r_checkerboard_0,
  const float16_t* message_u_checkerboard_1, const float16_t* message_d_checkerboard_1,
  const float16_t* message_l_checkerboard_1, const float16_t* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(x_val, y_val, current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1, message_u_checkerboard_0,
    message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0, message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1,
    message_r_checkerboard_1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  const float16_t* message_u_checkerboard_0, const float16_t* message_d_checkerboard_0,
  const float16_t* message_l_checkerboard_0, const float16_t* message_r_checkerboard_0,
  const float16_t* message_u_checkerboard_1, const float16_t* message_d_checkerboard_1,
  const float16_t* message_l_checkerboard_1, const float16_t* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(x_val, y_val, current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1, message_u_checkerboard_0,
    message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0, message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1,
    message_r_checkerboard_1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  const float16_t* message_u_checkerboard_0, const float16_t* message_d_checkerboard_0,
  const float16_t* message_l_checkerboard_0, const float16_t* message_r_checkerboard_0,
  const float16_t* message_u_checkerboard_1, const float16_t* message_d_checkerboard_1,
  const float16_t* message_l_checkerboard_1, const float16_t* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(x_val, y_val, current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1, message_u_checkerboard_0,
    message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0, message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1,
    message_r_checkerboard_1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  const float16_t* message_u_checkerboard_0, const float16_t* message_d_checkerboard_0,
  const float16_t* message_l_checkerboard_0, const float16_t* message_r_checkerboard_0,
  const float16_t* message_u_checkerboard_1, const float16_t* message_d_checkerboard_1,
  const float16_t* message_l_checkerboard_1, const float16_t* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(x_val, y_val, current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1, message_u_checkerboard_0,
    message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0, message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1,
    message_r_checkerboard_1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
  const float16_t* message_u_checkerboard_0, const float16_t* message_d_checkerboard_0,
  const float16_t* message_l_checkerboard_0, const float16_t* message_r_checkerboard_0,
  const float16_t* message_u_checkerboard_1, const float16_t* message_d_checkerboard_1,
  const float16_t* message_l_checkerboard_1, const float16_t* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(x_val, y_val, current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1, message_u_checkerboard_0,
    message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0, message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1,
    message_r_checkerboard_1, disparity_between_images_device, bp_settings_disp_vals);
}

#endif //COMPILING_FOR_ARM

#endif /* KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_ */
