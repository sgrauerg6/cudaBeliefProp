/*
 * KernelBpStereoCPU_ARMTemplateSpFuncts.h
 *
 *  Created on: Jun 23, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_

#include "BpSharedFuncts/SharedBPProcessingFuncts.h"
#include "RunImpCPU/ARMTemplateSpFuncts.h"
#include "KernelBpStereoCPU.h"

#ifdef COMPILING_FOR_ARM

#include <arm_neon.h>

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, 0>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, 0>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, 0>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, 0>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[0].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[1].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[2].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[3].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[4].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[5].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[6].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offset_num, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, 0>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, 0>(x_val, y_val, current_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[0].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[0].num_disp_vals>(x_val, y_val, current_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[1].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[1].num_disp_vals>(x_val, y_val, current_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[2].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[2].num_disp_vals>(x_val, y_val, current_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[3].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[3].num_disp_vals>(x_val, y_val, current_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[4].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[4].num_disp_vals>(x_val, y_val, current_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[5].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[5].num_disp_vals>(x_val, y_val, current_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[6].num_disp_vals>(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[6].num_disp_vals>(x_val, y_val, current_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

#endif //COMPILING_FOR_ARM

#endif /* KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_ */
