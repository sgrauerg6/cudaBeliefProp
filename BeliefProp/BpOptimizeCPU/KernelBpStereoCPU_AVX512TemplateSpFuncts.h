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
void beliefpropCPU::runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512(
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
  float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
  float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
  float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
  float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int bpSettingsDispVals,
  const BpParallelParams& optCPUParams)
{
  constexpr unsigned int numDataInSIMDVector{16u};
  runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<float, __m512, DISP_VALS>(
    checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, numDataInSIMDVector, bpSettingsDispVals, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512(
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int bpSettingsDispVals,
  const BpParallelParams& optCPUParams)
{
  constexpr unsigned int numDataInSIMDVector{16u};
  runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<short, __m256i, DISP_VALS>(
    checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, numDataInSIMDVector, bpSettingsDispVals, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512(
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
  double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
  double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
  double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
  double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int bpSettingsDispVals,
  const BpParallelParams& optCPUParams)
{
  constexpr unsigned int numDataInSIMDVector{8u};
  runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<double, __m512d, DISP_VALS>(
    checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, numDataInSIMDVector, bpSettingsDispVals, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::retrieveOutputDisparityUseSIMDVectorsAVX512(
  const beliefprop::levelProperties& currentLevelProperties,
  float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
  float* messageUPrevStereoCheckerboard0, float* messageDPrevStereoCheckerboard0,
  float* messageLPrevStereoCheckerboard0, float* messageRPrevStereoCheckerboard0,
  float* messageUPrevStereoCheckerboard1, float* messageDPrevStereoCheckerboard1,
  float* messageLPrevStereoCheckerboard1, float* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
  const BpParallelParams& optCPUParams)
{      
  constexpr unsigned int numDataInSIMDVector{16u};
  retrieveOutputDisparityUseSIMDVectors<float, __m512, float, __m512, DISP_VALS>(currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparityBetweenImagesDevice, bpSettingsDispVals,
    numDataInSIMDVector, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::retrieveOutputDisparityUseSIMDVectorsAVX512(
  const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0,
  short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
  const BpParallelParams& optCPUParams)
{      
  constexpr unsigned int numDataInSIMDVector{16u};
  retrieveOutputDisparityUseSIMDVectors<short, __m256i, float, __m512, DISP_VALS>(currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparityBetweenImagesDevice, bpSettingsDispVals,
    numDataInSIMDVector, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::retrieveOutputDisparityUseSIMDVectorsAVX512(
  const beliefprop::levelProperties& currentLevelProperties,
  double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
  double* messageUPrevStereoCheckerboard0, double* messageDPrevStereoCheckerboard0,
  double* messageLPrevStereoCheckerboard0, double* messageRPrevStereoCheckerboard0,
  double* messageUPrevStereoCheckerboard1, double* messageDPrevStereoCheckerboard1,
  double* messageLPrevStereoCheckerboard1, double* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
  const BpParallelParams& optCPUParams)
{      
  constexpr unsigned int numDataInSIMDVector{8u};
  retrieveOutputDisparityUseSIMDVectors<double, __m512d, double, __m512d, DISP_VALS>(currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparityBetweenImagesDevice, bpSettingsDispVals,
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
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  __m256i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
  __m256i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
  __m256i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
  __m256i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
  short* dstMessageArray, const __m256i& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  __m256i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
  __m256i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
  __m256i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
  __m256i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
  short* dstMessageArray, const __m256i& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  __m256i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
  __m256i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
  __m256i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
  __m256i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
  short* dstMessageArray, const __m256i& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  __m256i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
  __m256i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
  __m256i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
  __m256i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
  short* dstMessageArray, const __m256i& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  __m256i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
  __m256i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
  __m256i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
  __m256i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
  short* dstMessageArray, const __m256i& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  __m256i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
  __m256i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
  __m256i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
  __m256i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
  short* dstMessageArray, const __m256i& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  __m256i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]],
  __m256i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]],
  __m256i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]],
  __m256i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]],
  short* dstMessageArray, const __m256i& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<> inline void beliefpropCPU::msgStereoSIMD<short, __m256i>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  __m256i* messageValsNeighbor1, __m256i* messageValsNeighbor2,
  __m256i* messageValsNeighbor3, __m256i* dataCosts,
  short* dstMessageArray, const __m256i& disc_k_bp, const bool dataAligned,
  const unsigned int bpSettingsDispVals)
{
  msgStereoSIMDProcessing<short, __m256i, float, __m512>(xVal, yVal, currentLevelProperties,
    messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3, dataCosts,
    dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);
}

#endif /* KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_ */
