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
void beliefpropCPU::runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsNEON(
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
  float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
  float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
  float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
  float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int bpSettingsDispVals,
  const ParallelParams& optCPUParams)
{
  constexpr unsigned int numDataInSIMDVector{4u};
  runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<float, float32x4_t, DISP_VALS>(
    checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, numDataInSIMDVector, bpSettingsDispVals, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsNEON(
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int bpSettingsDispVals,
  const ParallelParams& optCPUParams)
{
  constexpr unsigned int numDataInSIMDVector{4u};
  runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<float16_t, float16x4_t, DISP_VALS>(
    checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, numDataInSIMDVector, bpSettingsDispVals, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsNEON(
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
  double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
  double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
  double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
  double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int bpSettingsDispVals,
  const ParallelParams& optCPUParams)
{
  constexpr unsigned int numDataInSIMDVector{2u};
  runBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess<double, float64x2_t, DISP_VALS>(
    checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, numDataInSIMDVector, bpSettingsDispVals, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::retrieveOutputDisparityUseSIMDVectorsNEON(
  const beliefprop::levelProperties& currentLevelProperties,
  float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
  float* messageUPrevStereoCheckerboard0, float* messageDPrevStereoCheckerboard0,
  float* messageLPrevStereoCheckerboard0, float* messageRPrevStereoCheckerboard0,
  float* messageUPrevStereoCheckerboard1, float* messageDPrevStereoCheckerboard1,
  float* messageLPrevStereoCheckerboard1, float* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
  const ParallelParams& optCPUParams)
{      
  constexpr unsigned int numDataInSIMDVector{4u};
  retrieveOutputDisparityUseSIMDVectors<float, float32x4_t, float, float32x4_t, DISP_VALS>(currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparityBetweenImagesDevice, bpSettingsDispVals,
    numDataInSIMDVector, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::retrieveOutputDisparityUseSIMDVectorsNEON(
  const beliefprop::levelProperties& currentLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
  const ParallelParams& optCPUParams)
{      
  constexpr unsigned int numDataInSIMDVector{4u};
  retrieveOutputDisparityUseSIMDVectors<float16_t, float16x4_t, float, float32x4_t, DISP_VALS>(currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparityBetweenImagesDevice, bpSettingsDispVals,
    numDataInSIMDVector, optCPUParams);
}

template<unsigned int DISP_VALS>
void beliefpropCPU::retrieveOutputDisparityUseSIMDVectorsNEON(
  const beliefprop::levelProperties& currentLevelProperties,
  double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
  double* messageUPrevStereoCheckerboard0, double* messageDPrevStereoCheckerboard0,
  double* messageLPrevStereoCheckerboard0, double* messageRPrevStereoCheckerboard0,
  double* messageUPrevStereoCheckerboard1, double* messageDPrevStereoCheckerboard1,
  double* messageLPrevStereoCheckerboard1, double* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
  const ParallelParams& optCPUParams)
{      
  constexpr unsigned int numDataInSIMDVector{2u};
  retrieveOutputDisparityUseSIMDVectors<double, float64x2_t, double, float64x2_t, DISP_VALS>(currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
    messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
    messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
    messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
    disparityBetweenImagesDevice, bpSettingsDispVals,
    numDataInSIMDVector, optCPUParams);
}

template<> inline void beliefpropCPU::updateBestDispBestVals<float32x4_t>(float32x4_t& bestDisparities, float32x4_t& bestVals,
  const float32x4_t& currentDisparity, const float32x4_t& valAtDisp)
{
  //get mask with value 1 where current value less then current best 1, 0 otherwise
  uint32x4_t maskUpdateVals = vcltq_f32(valAtDisp, bestVals);
  //update best values and best disparities using mask
  //vbslq_f32 operation uses first float32x4_t argument if mask value is 1 and seconde float32x4_t argument if mask value is 0
  bestVals = vbslq_f32(maskUpdateVals, valAtDisp, bestVals);
  bestDisparities = vbslq_f32(maskUpdateVals, currentDisparity, bestDisparities);
}

template<> inline void beliefpropCPU::updateBestDispBestVals<float64x2_t>(float64x2_t& bestDisparities, float64x2_t& bestVals,
  const float64x2_t& currentDisparity, const float64x2_t& valAtDisp)
{
  uint64x2_t maskUpdateVals = vcltq_f64(valAtDisp, bestVals);
  bestVals = vbslq_f64(maskUpdateVals, valAtDisp, bestVals);
  bestDisparities = vbslq_f64(maskUpdateVals, currentDisparity, bestDisparities);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
  float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
  float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
  float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
  float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
  float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
  float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
  float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
  float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
  float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
  float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
  float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
  float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
  float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
  float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
  float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
  float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
  float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
  float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
  float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
  float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
  float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
  float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
  float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
  float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void beliefpropCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]],
  float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]],
  float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]],
  float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]],
  float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
  msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(
    xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
    messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<> inline void beliefpropCPU::msgStereoSIMD<float16_t, float16x4_t>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties,
  float16x4_t* messageValsNeighbor1, float16x4_t* messageValsNeighbor2,
  float16x4_t* messageValsNeighbor3, float16x4_t* dataCosts,
  float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned,
  const unsigned int bpSettingsDispVals)
{
  msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t>(xVal, yVal, currentLevelProperties,
    messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3, dataCosts,
    dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);
}

#endif /* KERNELBPSTEREOCPU_NEON_H_ */
