#ifndef KERNELBPSTEREOCPU_TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_TEMPLATESPFUNCTS_H_

//this is only processed when on x86
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "KernelBpStereoCPU.h"
#include "BpSharedFuncts/SharedBPProcessingFuncts.h"

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, 0>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, 0>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, 0>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals,
  void* dstProcessing)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, 0>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals, dstProcessing);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val,
  beliefprop::Checkerboard_Part checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
    x_val, y_val, checkerboard_to_update, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, data_aligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, short, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_to_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<short, float, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
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
void beliefprop::InitializeCurrentLevelDataPixel<short, short, 0>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, 0>(x_val, y_val, checkerboard_part, current_bp_level, prev_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offset_num, bp_settings_disp_vals);
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offset_num, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::InitializeCurrentLevelDataPixel<short, short, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, beliefprop::Checkerboard_Part checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  short* dataCostDeviceToWriteTo, unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  beliefprop::InitializeCurrentLevelDataPixel<short, float, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(x_val, y_val, checkerboard_part,
    current_bp_level, prev_bp_level, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offset_num, bp_settings_disp_vals);
}


template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, 0>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  const short* messageUPrevStereoCheckerboard0, const short* messageDPrevStereoCheckerboard0,
  const short* messageLPrevStereoCheckerboard0, const short* messageRPrevStereoCheckerboard0,
  const short* messageUPrevStereoCheckerboard1, const short* messageDPrevStereoCheckerboard1,
  const short* messageLPrevStereoCheckerboard1, const short* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, 0>(x_val, y_val, current_bp_level, dataCostStereoCheckerboard0,
    dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  const short* messageUPrevStereoCheckerboard0, const short* messageDPrevStereoCheckerboard0,
  const short* messageLPrevStereoCheckerboard0, const short* messageRPrevStereoCheckerboard0,
  const short* messageUPrevStereoCheckerboard1, const short* messageDPrevStereoCheckerboard1,
  const short* messageLPrevStereoCheckerboard1, const short* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[0].num_disp_vals>(x_val, y_val, current_bp_level, dataCostStereoCheckerboard0,
    dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1, 
  const short* messageUPrevStereoCheckerboard0, const short* messageDPrevStereoCheckerboard0,
  const short* messageLPrevStereoCheckerboard0, const short* messageRPrevStereoCheckerboard0,
  const short* messageUPrevStereoCheckerboard1, const short* messageDPrevStereoCheckerboard1,
  const short* messageLPrevStereoCheckerboard1, const short* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[1].num_disp_vals>(x_val, y_val, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  const short* messageUPrevStereoCheckerboard0, const short* messageDPrevStereoCheckerboard0,
  const short* messageLPrevStereoCheckerboard0, const short* messageRPrevStereoCheckerboard0,
  const short* messageUPrevStereoCheckerboard1, const short* messageDPrevStereoCheckerboard1,
  const short* messageLPrevStereoCheckerboard1, const short* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[2].num_disp_vals>(x_val, y_val, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  const short* messageUPrevStereoCheckerboard0, const short* messageDPrevStereoCheckerboard0,
  const short* messageLPrevStereoCheckerboard0, const short* messageRPrevStereoCheckerboard0,
  const short* messageUPrevStereoCheckerboard1, const short* messageDPrevStereoCheckerboard1,
  const short* messageLPrevStereoCheckerboard1, const short* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[3].num_disp_vals>(x_val, y_val, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  const short* messageUPrevStereoCheckerboard0, const short* messageDPrevStereoCheckerboard0,
  const short* messageLPrevStereoCheckerboard0, const short* messageRPrevStereoCheckerboard0,
  const short* messageUPrevStereoCheckerboard1, const short* messageDPrevStereoCheckerboard1,
  const short* messageLPrevStereoCheckerboard1, const short* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[4].num_disp_vals>(x_val, y_val, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  const short* messageUPrevStereoCheckerboard0, const short* messageDPrevStereoCheckerboard0,
  const short* messageLPrevStereoCheckerboard0, const short* messageRPrevStereoCheckerboard0,
  const short* messageUPrevStereoCheckerboard1, const short* messageDPrevStereoCheckerboard1,
  const short* messageLPrevStereoCheckerboard1, const short* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[5].num_disp_vals>(x_val, y_val, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

template<> inline
void beliefprop::RetrieveOutputDisparityPixel<short, short, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int x_val, unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
  const short* dataCostStereoCheckerboard0, const short* dataCostStereoCheckerboard1,
  const short* messageUPrevStereoCheckerboard0, const short* messageDPrevStereoCheckerboard0,
  const short* messageLPrevStereoCheckerboard0, const short* messageRPrevStereoCheckerboard0,
  const short* messageUPrevStereoCheckerboard1, const short* messageDPrevStereoCheckerboard1,
  const short* messageLPrevStereoCheckerboard1, const short* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  beliefprop::RetrieveOutputDisparityPixel<short, float, beliefprop::kStereoSetsToProcess[6].num_disp_vals>(x_val, y_val, current_bp_level,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparity_between_images_device, bp_settings_disp_vals);
}

#endif //KERNELBPSTEREOCPU_TEMPLATESPFUNCTS_H_
