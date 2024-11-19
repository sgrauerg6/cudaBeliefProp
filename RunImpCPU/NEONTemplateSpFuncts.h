/*
 * NEONTemplateSpFuncts.h
 *
 *  Created on: Jun 26, 2024
 *      Author: scott
 */

#ifndef NEONTEMPLATESPFUNCTS_H_
#define NEONTEMPLATESPFUNCTS_H_

//this is only used when processing using an ARM CPU with NEON instructions
#include <arm_neon.h>

template<> inline float64x2_t VectProcessingFuncts::loadPackedDataAligned<double, float64x2_t>(
  unsigned int x, unsigned int y, unsigned int currentDisparity,
  const beliefprop::BpLevelProperties& currentBpLevel, unsigned int numDispVals, double* inData)
{
  return vld1q_f64(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentBpLevel.padded_width_checkerboard_level_,
    currentBpLevel.height_level_, currentDisparity, numDispVals)]);
}

template<> inline float32x4_t VectProcessingFuncts::loadPackedDataAligned<float, float32x4_t>(
  unsigned int x, unsigned int y, unsigned int currentDisparity,
  const beliefprop::BpLevelProperties& currentBpLevel, unsigned int numDispVals, float* inData)
{
  return vld1q_f32(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentBpLevel.padded_width_checkerboard_level_,
    currentBpLevel.height_level_, currentDisparity, numDispVals)]);
}

template<> inline float16x4_t VectProcessingFuncts::loadPackedDataAligned<float16_t, float16x4_t>(
  unsigned int x, unsigned int y, unsigned int currentDisparity,
  const beliefprop::BpLevelProperties& currentBpLevel, unsigned int numDispVals, float16_t* inData)
{
  return vld1_f16(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentBpLevel.padded_width_checkerboard_level_,
    currentBpLevel.height_level_, currentDisparity,
    numDispVals)]);
}

template<> inline float32x4_t VectProcessingFuncts::loadPackedDataUnaligned<float, float32x4_t>(
  unsigned int x, unsigned int y, unsigned int currentDisparity,
  const beliefprop::BpLevelProperties& currentBpLevel, unsigned int numDispVals, float* inData)
{
  return vld1q_f32(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentBpLevel.padded_width_checkerboard_level_,
    currentBpLevel.height_level_, currentDisparity, numDispVals)]);
}

template<> inline float16x4_t VectProcessingFuncts::loadPackedDataUnaligned<float16_t, float16x4_t>(
  unsigned int x, unsigned int y, unsigned int currentDisparity,
  const beliefprop::BpLevelProperties& currentBpLevel, unsigned int numDispVals, float16_t* inData)
{
  return vld1_f16(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentBpLevel.padded_width_checkerboard_level_,
    currentBpLevel.height_level_, currentDisparity, numDispVals)]);
}

template<> inline float64x2_t VectProcessingFuncts::loadPackedDataUnaligned<double, float64x2_t>(
  unsigned int x, unsigned int y, unsigned int currentDisparity,
  const beliefprop::BpLevelProperties& currentBpLevel, unsigned int numDispVals, double* inData)
{
  return vld1q_f64(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentBpLevel.padded_width_checkerboard_level_,
    currentBpLevel.height_level_, currentDisparity, numDispVals)]);
}

template<> inline float32x4_t VectProcessingFuncts::createSIMDVectorSameData<float32x4_t>(float data) {
  return vdupq_n_f32(data);
}

template<> inline float16x4_t VectProcessingFuncts::createSIMDVectorSameData<float16x4_t>(float data) {
  return vcvt_f16_f32(createSIMDVectorSameData<float32x4_t>(data));
}

template<> inline float64x2_t VectProcessingFuncts::createSIMDVectorSameData<float64x2_t>(float data) {
  return vdupq_n_f64((double)data);
}

template<> inline float32x4_t VectProcessingFuncts::addVals<float32x4_t, float32x4_t, float32x4_t>(const float32x4_t& val1, const float32x4_t& val2) {
  return vaddq_f32(val1, val2);
}

template<> inline float64x2_t VectProcessingFuncts::addVals<float64x2_t, float64x2_t, float64x2_t>(const float64x2_t& val1, const float64x2_t& val2) {
  return vaddq_f64(val1, val2);
}

template<> inline float32x4_t VectProcessingFuncts::addVals<float32x4_t, float16x4_t, float32x4_t>(const float32x4_t& val1, const float16x4_t& val2) {
  return vaddq_f32(val1, vcvt_f32_f16(val2));
}

template<> inline float32x4_t VectProcessingFuncts::addVals<float16x4_t, float32x4_t, float32x4_t>(const float16x4_t& val1, const float32x4_t& val2) {
  return vaddq_f32(vcvt_f32_f16(val1), val2);
}

template<> inline float32x4_t VectProcessingFuncts::addVals<float16x4_t, float16x4_t, float32x4_t>(const float16x4_t& val1, const float16x4_t& val2) {
  return vaddq_f32(vcvt_f32_f16(val1), vcvt_f32_f16(val2));
}

template<> inline float32x4_t VectProcessingFuncts::subtractVals<float32x4_t, float32x4_t, float32x4_t>(const float32x4_t& val1, const float32x4_t& val2) {
  return vsubq_f32(val1, val2);
}

template<> inline float64x2_t VectProcessingFuncts::subtractVals<float64x2_t, float64x2_t, float64x2_t>(const float64x2_t& val1, const float64x2_t& val2) {
  return vsubq_f64(val1, val2);
}

template<> inline float32x4_t VectProcessingFuncts::divideVals<float32x4_t, float32x4_t, float32x4_t>(const float32x4_t& val1, const float32x4_t& val2) {
  return vdivq_f32(val1, val2);
}

template<> inline float64x2_t VectProcessingFuncts::divideVals<float64x2_t, float64x2_t, float64x2_t>(const float64x2_t& val1, const float64x2_t& val2) {
  return vdivq_f64(val1, val2);
}

template<> inline float32x4_t VectProcessingFuncts::convertValToDatatype<float32x4_t, float>(float val) {
  return vdupq_n_f32(val);
}

template<> inline float64x2_t VectProcessingFuncts::convertValToDatatype<float64x2_t, double>(double val) {
  return vdupq_n_f64(val);
}

template<> inline float32x4_t VectProcessingFuncts::GetMinByElement<float32x4_t>(const float32x4_t& val1, const float32x4_t& val2) {
  return vminnmq_f32(val1, val2);
}

template<> inline float64x2_t VectProcessingFuncts::GetMinByElement<float64x2_t>(const float64x2_t& val1, const float64x2_t& val2) {
  return vminnmq_f64(val1, val2);
}

template<> inline void VectProcessingFuncts::storePackedDataAligned<float, float32x4_t>(
  unsigned int indexDataStore, float* locationDataStore, const float32x4_t& dataToStore)
{
  vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::storePackedDataAligned<float16_t, float32x4_t>(
  unsigned int indexDataStore, float16_t* locationDataStore, const float32x4_t& dataToStore)
{
  vst1_f16(&locationDataStore[indexDataStore], vcvt_f16_f32(dataToStore));
}

template<> inline void VectProcessingFuncts::storePackedDataAligned<double, float64x2_t>(
  unsigned int indexDataStore, double* locationDataStore, const float64x2_t& dataToStore)
{
  vst1q_f64(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::storePackedDataUnaligned<float, float32x4_t>(
  unsigned int indexDataStore, float* locationDataStore, const float32x4_t& dataToStore)
{
  vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::storePackedDataUnaligned<float16_t, float32x4_t>(
  unsigned int indexDataStore, float16_t* locationDataStore, const float32x4_t& dataToStore)
{
  vst1_f16(&locationDataStore[indexDataStore], vcvt_f16_f32(dataToStore));
}

template<> inline void VectProcessingFuncts::storePackedDataUnaligned<double, float64x2_t>(
  unsigned int indexDataStore, double* locationDataStore, const float64x2_t& dataToStore)
{
  vst1q_f64(&locationDataStore[indexDataStore], dataToStore);
}

#endif /* NEONTEMPLATESPFUNCTS_H_ */
