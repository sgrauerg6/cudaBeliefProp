#ifndef KERNELBPSTEREOCPU_TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_TEMPLATESPFUNCTS_H_

//this is only processed when on x86
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "KernelBpStereoCPU.h"
#include "BpSharedFuncts/SharedBpProcessingFuncts.h"

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, 0>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* message_u_checkerboard_0, short* message_d_checkerboard_0,
  short* message_l_checkerboard_0, short* message_r_checkerboard_0,
  short* message_u_checkerboard_1, short* message_d_checkerboard_1,
  short* message_l_checkerboard_1, short* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, 0>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, 0>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* message_u_checkerboard_0, short* message_d_checkerboard_0,
  short* message_l_checkerboard_0, short* message_r_checkerboard_0,
  short* message_u_checkerboard_1, short* message_d_checkerboard_1,
  short* message_l_checkerboard_1, short* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals,
  void* dst_processing)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, 0>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals, dst_processing);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* message_u_checkerboard_0, short* message_d_checkerboard_0,
  short* message_l_checkerboard_0, short* message_r_checkerboard_0,
  short* message_u_checkerboard_1, short* message_d_checkerboard_1,
  short* message_l_checkerboard_1, short* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* message_u_checkerboard_0, short* message_d_checkerboard_0,
  short* message_l_checkerboard_0, short* message_r_checkerboard_0,
  short* message_u_checkerboard_1, short* message_d_checkerboard_1,
  short* message_l_checkerboard_1, short* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* message_u_checkerboard_0, short* message_d_checkerboard_0,
  short* message_l_checkerboard_0, short* message_r_checkerboard_0,
  short* message_u_checkerboard_1, short* message_d_checkerboard_1,
  short* message_l_checkerboard_1, short* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* message_u_checkerboard_0, short* message_d_checkerboard_0,
  short* message_l_checkerboard_0, short* message_r_checkerboard_0,
  short* message_u_checkerboard_1, short* message_d_checkerboard_1,
  short* message_l_checkerboard_1, short* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* message_u_checkerboard_0, short* message_d_checkerboard_0,
  short* message_l_checkerboard_0, short* message_r_checkerboard_0,
  short* message_u_checkerboard_1, short* message_d_checkerboard_1,
  short* message_l_checkerboard_1, short* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* message_u_checkerboard_0, short* message_d_checkerboard_0,
  short* message_l_checkerboard_0, short* message_r_checkerboard_0,
  short* message_u_checkerboard_1, short* message_d_checkerboard_1,
  short* message_l_checkerboard_1, short* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0,
    message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1,
    message_l_checkerboard_1, message_r_checkerboard_1,
    disc_k_bp, offset_data, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* message_u_checkerboard_0, short* message_d_checkerboard_0,
  short* message_l_checkerboard_0, short* message_r_checkerboard_0,
  short* message_u_checkerboard_1, short* message_d_checkerboard_1,
  short* message_l_checkerboard_1, short* message_r_checkerboard_1,
  float disc_k_bp, unsigned int offset_data, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
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
void beliefprop::InitializeCurrentLevelDataPixel<short, short, 0>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, 0>(
    x_val, y_val, checkerboard_part, current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level,
    offset_num, bp_settings_disp_vals);
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
    x_val, y_val, checkerboard_part, current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level,
    offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
    x_val, y_val, checkerboard_part, current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level,
    offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
    x_val, y_val, checkerboard_part, current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level,
    offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
    x_val, y_val, checkerboard_part, current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level,
    offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
    x_val, y_val, checkerboard_part, current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level,
    offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
    x_val, y_val, checkerboard_part, current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level,
    offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  short* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
    x_val, y_val, checkerboard_part, current_bp_level, prev_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level,
    offset_num, bp_settings_disp_vals);
}


template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, 0>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  const short* message_u_checkerboard_0, const short* message_d_checkerboard_0,
  const short* message_l_checkerboard_0, const short* message_r_checkerboard_0,
  const short* message_u_checkerboard_1, const short* message_d_checkerboard_1,
  const short* message_l_checkerboard_1, const short* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, 0>(
    x_val, y_val, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1, message_r_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  const short* message_u_checkerboard_0, const short* message_d_checkerboard_0,
  const short* message_l_checkerboard_0, const short* message_r_checkerboard_0,
  const short* message_u_checkerboard_1, const short* message_d_checkerboard_1,
  const short* message_l_checkerboard_1, const short* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
    x_val, y_val, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1, message_r_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1, 
  const short* message_u_checkerboard_0, const short* message_d_checkerboard_0,
  const short* message_l_checkerboard_0, const short* message_r_checkerboard_0,
  const short* message_u_checkerboard_1, const short* message_d_checkerboard_1,
  const short* message_l_checkerboard_1, const short* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
    x_val, y_val, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1, message_r_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  const short* message_u_checkerboard_0, const short* message_d_checkerboard_0,
  const short* message_l_checkerboard_0, const short* message_r_checkerboard_0,
  const short* message_u_checkerboard_1, const short* message_d_checkerboard_1,
  const short* message_l_checkerboard_1, const short* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
    x_val, y_val, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1, message_r_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  const short* message_u_checkerboard_0, const short* message_d_checkerboard_0,
  const short* message_l_checkerboard_0, const short* message_r_checkerboard_0,
  const short* message_u_checkerboard_1, const short* message_d_checkerboard_1,
  const short* message_l_checkerboard_1, const short* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
    x_val, y_val, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1, message_r_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  const short* message_u_checkerboard_0, const short* message_d_checkerboard_0,
  const short* message_l_checkerboard_0, const short* message_r_checkerboard_0,
  const short* message_u_checkerboard_1, const short* message_d_checkerboard_1,
  const short* message_l_checkerboard_1, const short* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
    x_val, y_val, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1, message_r_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  const short* message_u_checkerboard_0, const short* message_d_checkerboard_0,
  const short* message_l_checkerboard_0, const short* message_r_checkerboard_0,
  const short* message_u_checkerboard_1, const short* message_d_checkerboard_1,
  const short* message_l_checkerboard_1, const short* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
    x_val, y_val, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1, message_r_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
  const short* message_u_checkerboard_0, const short* message_d_checkerboard_0,
  const short* message_l_checkerboard_0, const short* message_r_checkerboard_0,
  const short* message_u_checkerboard_1, const short* message_d_checkerboard_1,
  const short* message_l_checkerboard_1, const short* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
    x_val, y_val, current_bp_level,
    data_cost_checkerboard_0, data_cost_checkerboard_1,
    message_u_checkerboard_0, message_d_checkerboard_0, message_l_checkerboard_0, message_r_checkerboard_0,
    message_u_checkerboard_1, message_d_checkerboard_1, message_l_checkerboard_1, message_r_checkerboard_1,
    disparity_between_images_device, bp_settings_disp_vals);
}

#endif //KERNELBPSTEREOCPU_TEMPLATESPFUNCTS_H_
