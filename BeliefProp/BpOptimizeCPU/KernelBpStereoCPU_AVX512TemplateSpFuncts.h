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
void beliefpropCPU::runBPIterationUsingCheckerboardUpdatesUseSIMDVectorskAVX512(
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
  float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
  float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
  float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
  float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& optCPUParams)
{
  constexpr unsigned int numDataInSIMDVector{16u};
  runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<float, __m512, DISP_VALS>(
    checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, numDataInSIMDVector, bp_settings_disp_vals, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::runBPIterationUsingCheckerboardUpdatesUseSIMDVectorskAVX512(
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& optCPUParams)
{
  constexpr unsigned int numDataInSIMDVector{16u};
  runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<short, __m256i, DISP_VALS>(
    checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, numDataInSIMDVector, bp_settings_disp_vals, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::runBPIterationUsingCheckerboardUpdatesUseSIMDVectorskAVX512(
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
  double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
  double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
  double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
  double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& optCPUParams)
{
  constexpr unsigned int numDataInSIMDVector{8u};
  runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<double, __m512d, DISP_VALS>(
    checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, numDataInSIMDVector, bp_settings_disp_vals, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::retrieveOutputDisparityUseSIMDVectorskAVX512(
  const beliefprop::BpLevelProperties& currentBpLevel,
  float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
  float* messageUPrevStereoCheckerboard0, float* messageDPrevStereoCheckerboard0,
  float* messageLPrevStereoCheckerboard0, float* messageRPrevStereoCheckerboard0,
  float* messageUPrevStereoCheckerboard1, float* messageDPrevStereoCheckerboard1,
  float* messageLPrevStereoCheckerboard1, float* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals,
  const ParallelParams& optCPUParams)
{      
  constexpr unsigned int numDataInSIMDVector{16u};
  retrieveOutputDisparityUseSIMDVectors<float, __m512, float, __m512, DISP_VALS>(currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparityBetweenImagesDevice, bp_settings_disp_vals,
    numDataInSIMDVector, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::retrieveOutputDisparityUseSIMDVectorskAVX512(
  const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0,
  short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals,
  const ParallelParams& optCPUParams)
{      
  constexpr unsigned int numDataInSIMDVector{16u};
  retrieveOutputDisparityUseSIMDVectors<short, __m256i, float, __m512, DISP_VALS>(currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparityBetweenImagesDevice, bp_settings_disp_vals,
    numDataInSIMDVector, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::retrieveOutputDisparityUseSIMDVectorskAVX512(
  const beliefprop::BpLevelProperties& currentBpLevel,
  double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
  double* messageUPrevStereoCheckerboard0, double* messageDPrevStereoCheckerboard0,
  double* messageLPrevStereoCheckerboard0, double* messageRPrevStereoCheckerboard0,
  double* messageUPrevStereoCheckerboard1, double* messageDPrevStereoCheckerboard1,
  double* messageLPrevStereoCheckerboard1, double* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals,
  const ParallelParams& optCPUParams)
{      
  constexpr unsigned int numDataInSIMDVector{8u};
  retrieveOutputDisparityUseSIMDVectors<double, __m512d, double, __m512d, DISP_VALS>(currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparityBetweenImagesDevice, bp_settings_disp_vals,
    numDataInSIMDVector, optCPUParams);
}

template<> inline void beliefpropCPU::updateBestDispBestVals<__m512>(__m512& bestDisparities, __m512& bestVals,
  const __m512& currentDisparity, const __m512& valAtDisp)
{
  __mmask16 maskNeedUpdate =  _mm512_cmp_ps_mask(valAtDisp, bestVals, _CMP_LT_OS);
  bestVals = _mm512_mask_blend_ps(maskNeedUpdate, bestVals, valAtDisp);
  bestDisparities = _mm512_mask_blend_ps(maskNeedUpdate, bestDisparities, currentDisparity);
}

template<> inline void beliefpropCPU::updateBestDispBestVals<__m512d>(__m512d& bestDisparities, __m512d& bestVals,
  const __m512d& currentDisparity, const __m512d& valAtDisp)
{
  __mmask16 maskNeedUpdate =  _mm512_cmp_pd_mask(valAtDisp, bestVals, _CMP_LT_OS);
  bestVals = _mm512_mask_blend_pd(maskNeedUpdate, bestVals, valAtDisp);
  bestDisparities = _mm512_mask_blend_pd(maskNeedUpdate, bestDisparities, currentDisparity);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[0].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[0].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[0].num_disp_vals],
  __m256i dataCosts[bp_params::kStereoSetsToProcess[0].num_disp_vals],
  short* dstMessageArray, const __m256i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
    xVal, yVal, currentBpLevel, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[1].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[1].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[1].num_disp_vals],
  __m256i dataCosts[bp_params::kStereoSetsToProcess[1].num_disp_vals],
  short* dstMessageArray, const __m256i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
    xVal, yVal, currentBpLevel, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[2].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[2].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[2].num_disp_vals],
  __m256i dataCosts[bp_params::kStereoSetsToProcess[2].num_disp_vals],
  short* dstMessageArray, const __m256i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
    xVal, yVal, currentBpLevel, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[3].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[3].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[3].num_disp_vals],
  __m256i dataCosts[bp_params::kStereoSetsToProcess[3].num_disp_vals],
  short* dstMessageArray, const __m256i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
    xVal, yVal, currentBpLevel, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[4].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[4].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[4].num_disp_vals],
  __m256i dataCosts[bp_params::kStereoSetsToProcess[4].num_disp_vals],
  short* dstMessageArray, const __m256i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
    xVal, yVal, currentBpLevel, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[5].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[5].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[5].num_disp_vals],
  __m256i dataCosts[bp_params::kStereoSetsToProcess[5].num_disp_vals],
  short* dstMessageArray, const __m256i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
    xVal, yVal, currentBpLevel, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel,
  __m256i messageValsNeighbor1[bp_params::kStereoSetsToProcess[6].num_disp_vals],
  __m256i messageValsNeighbor2[bp_params::kStereoSetsToProcess[6].num_disp_vals],
  __m256i messageValsNeighbor3[bp_params::kStereoSetsToProcess[6].num_disp_vals],
  __m256i dataCosts[bp_params::kStereoSetsToProcess[6].num_disp_vals],
  short* dstMessageArray, const __m256i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
    xVal, yVal, currentBpLevel, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i>(unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel,
  __m256i* messageValsNeighbor1, __m256i* messageValsNeighbor2,
  __m256i* messageValsNeighbor3, __m256i* dataCosts,
  short* dstMessageArray, const __m256i& disc_k_bp, bool dataAligned,
  unsigned int bp_settings_disp_vals)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512>(xVal, yVal, currentBpLevel,
    messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3, dataCosts,
    dstMessageArray, disc_k_bp, dataAligned, bp_settings_disp_vals);
}

#endif /* KERNELBPSTEREOCPU_kAVX512TEMPLATESPFUNCTS_H_ */
