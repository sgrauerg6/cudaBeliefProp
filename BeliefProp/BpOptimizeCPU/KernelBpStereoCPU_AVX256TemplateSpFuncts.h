/*
 * KernelBpStereoCPU_AVX256TemplateSpFuncts.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "BpSharedFuncts/SharedBPProcessingFuncts.h"
#include "RunImpCPU/AVX256TemplateSpFuncts.h"

template<unsigned int DISP_VALS>
void beliefpropCPU::runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX256(
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::LevelProperties& currentLevelProperties,
  float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
  float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
  float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
  float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
  float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int bpSettingsDispVals,
  const ParallelParams& optCPUParams)
{
  constexpr unsigned int numDataInSIMDVector{8u};
  runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<float, __m256, DISP_VALS>(
    checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, numDataInSIMDVector, bpSettingsDispVals, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX256(
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::LevelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int bpSettingsDispVals,
  const ParallelParams& optCPUParams)
{
  constexpr unsigned int numDataInSIMDVector{8u};
  runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<short, __m128i, DISP_VALS>(
    checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, numDataInSIMDVector, bpSettingsDispVals, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX256(
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::LevelProperties& currentLevelProperties,
  double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
  double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
  double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
  double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
  double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int bpSettingsDispVals,
  const ParallelParams& optCPUParams)
{
  constexpr unsigned int numDataInSIMDVector{4u};
  runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<double, __m256d, DISP_VALS>(
    checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, numDataInSIMDVector, bpSettingsDispVals, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::retrieveOutputDisparityUseSIMDVectorsAVX256(
  const beliefprop::LevelProperties& currentLevelProperties,
  float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
  float* messageUPrevStereoCheckerboard0, float* messageDPrevStereoCheckerboard0,
  float* messageLPrevStereoCheckerboard0, float* messageRPrevStereoCheckerboard0,
  float* messageUPrevStereoCheckerboard1, float* messageDPrevStereoCheckerboard1,
  float* messageLPrevStereoCheckerboard1, float* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bpSettingsDispVals,
  const ParallelParams& optCPUParams)
{      
  constexpr unsigned int numDataInSIMDVector{8u};
  retrieveOutputDisparityUseSIMDVectors<float, __m256, float, __m256, DISP_VALS>(currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparityBetweenImagesDevice, bpSettingsDispVals,
    numDataInSIMDVector, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::retrieveOutputDisparityUseSIMDVectorsAVX256(
  const beliefprop::LevelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0,
  short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bpSettingsDispVals,
  const ParallelParams& optCPUParams)
{      
  constexpr unsigned int numDataInSIMDVector{8u};
  retrieveOutputDisparityUseSIMDVectors<short, __m128i, float, __m256, DISP_VALS>(currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparityBetweenImagesDevice, bpSettingsDispVals,
    numDataInSIMDVector, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::retrieveOutputDisparityUseSIMDVectorsAVX256(
  const beliefprop::LevelProperties& currentLevelProperties,
  double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
  double* messageUPrevStereoCheckerboard0, double* messageDPrevStereoCheckerboard0,
  double* messageLPrevStereoCheckerboard0, double* messageRPrevStereoCheckerboard0,
  double* messageUPrevStereoCheckerboard1, double* messageDPrevStereoCheckerboard1,
  double* messageLPrevStereoCheckerboard1, double* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bpSettingsDispVals,
  const ParallelParams& optCPUParams)
{      
  constexpr unsigned int numDataInSIMDVector{4u};
  retrieveOutputDisparityUseSIMDVectors<double, __m256d, double, __m256d, DISP_VALS>(currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparityBetweenImagesDevice, bpSettingsDispVals,
    numDataInSIMDVector, optCPUParams);
}

template<> inline void beliefpropCPU::updateBestDispBestVals<__m256>(__m256& bestDisparities, __m256& bestVals,
  const __m256& currentDisparity, const __m256& valAtDisp)
{
  __m256 maskNeedUpdate = _mm256_cmp_ps(valAtDisp, bestVals, _CMP_LT_OS);
  bestVals = _mm256_blendv_ps(bestVals, valAtDisp, maskNeedUpdate);
  bestDisparities = _mm256_blendv_ps(bestDisparities, currentDisparity, maskNeedUpdate);
   /* __mmask8 maskNeedUpdate =  _mm256_cmp_ps_mask(valAtDisp, bestVals, _CMP_LT_OS);
  bestVals = _mm256_mask_blend_ps(maskNeedUpdate, bestVals, valAtDisp);
  bestDisparities = _mm256_mask_blend_ps(maskNeedUpdate, bestDisparities, currentDisparity);*/
}

template<> inline void beliefpropCPU::updateBestDispBestVals<__m256d>(__m256d& bestDisparities, __m256d& bestVals,
  const __m256d& currentDisparity, const __m256d& valAtDisp)
{
  __m256d maskNeedUpdate = _mm256_cmp_pd(valAtDisp, bestVals, _CMP_LT_OS);
  bestVals = _mm256_blendv_pd(bestVals, valAtDisp, maskNeedUpdate);
  bestDisparities = _mm256_blendv_pd(bestDisparities, currentDisparity, maskNeedUpdate);
  /*__mmask8 maskNeedUpdate =  _mm256_cmp_pd_mask(valAtDisp, bestVals, _CMP_LT_OS);
  bestVals = _mm256_mask_blend_pd(maskNeedUpdate, bestVals, valAtDisp);
  bestDisparities = _mm256_mask_blend_pd(maskNeedUpdate, bestDisparities, currentDisparity);*/
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m128i, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  __m128i messageValsNeighbor1[bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_],
  __m128i messageValsNeighbor2[bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_],
  __m128i messageValsNeighbor3[bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_],
  __m128i dataCosts[bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_],
  short* dstMessageArray, const __m128i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m128i, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  __m128i messageValsNeighbor1[bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_],
  __m128i messageValsNeighbor2[bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_],
  __m128i messageValsNeighbor3[bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_],
  __m128i dataCosts[bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_],
  short* dstMessageArray, const __m128i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m128i, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  __m128i messageValsNeighbor1[bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_],
  __m128i messageValsNeighbor2[bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_],
  __m128i messageValsNeighbor3[bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_],
  __m128i dataCosts[bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_],
  short* dstMessageArray, const __m128i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m128i, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  __m128i messageValsNeighbor1[bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_],
  __m128i messageValsNeighbor2[bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_],
  __m128i messageValsNeighbor3[bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_],
  __m128i dataCosts[bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_],
  short* dstMessageArray, const __m128i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m128i, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  __m128i messageValsNeighbor1[bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_],
  __m128i messageValsNeighbor2[bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_],
  __m128i messageValsNeighbor3[bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_],
  __m128i dataCosts[bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_],
  short* dstMessageArray, const __m128i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m128i, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  __m128i messageValsNeighbor1[bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_],
  __m128i messageValsNeighbor2[bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_],
  __m128i messageValsNeighbor3[bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_],
  __m128i dataCosts[bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_],
  short* dstMessageArray, const __m128i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m128i, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>(
  unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  __m128i messageValsNeighbor1[bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_],
  __m128i messageValsNeighbor2[bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_],
  __m128i messageValsNeighbor3[bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_],
  __m128i dataCosts[bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_],
  short* dstMessageArray, const __m128i& disc_k_bp, bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<> inline void beliefpropCPU::msgStereoSIMD<short, __m128i>(unsigned int xVal, unsigned int yVal,
  const beliefprop::LevelProperties& currentLevelProperties,
  __m128i* messageValsNeighbor1, __m128i* messageValsNeighbor2,
  __m128i* messageValsNeighbor3, __m128i* dataCosts,
  short* dstMessageArray, const __m128i& disc_k_bp, bool dataAligned,
  unsigned int bpSettingsDispVals)
{
  msgStereoSIMDProcessing<short, __m128i, float, __m256>(xVal, yVal, currentLevelProperties,
    messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3, dataCosts,
    dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);
}

#endif /* KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_ */
