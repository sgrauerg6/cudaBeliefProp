/*
 * NEONTemplateSpFuncts.h
 *
 * Template specializations for processing on SIMD vector data types supported by NEON
 * on ARM CPUs.
 *
 *  Created on: Jun 26, 2024
 *      Author: scott
 */

#ifndef NEONTEMPLATESPFUNCTS_H_
#define NEONTEMPLATESPFUNCTS_H_

//NEON only used when processing on an ARM CPU that supports NEON instructions
#include <arm_neon.h>

template<> inline float64x2_t VectProcessingFuncts::LoadPackedDataAligned<double, float64x2_t>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, double* inData)
{
  return vld1q_f64(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline float32x4_t VectProcessingFuncts::LoadPackedDataAligned<float, float32x4_t>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, float* inData)
{
  return vld1q_f32(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline float16x4_t VectProcessingFuncts::LoadPackedDataAligned<float16_t, float16x4_t>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, float16_t* inData)
{
  return vld1_f16(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity,
    numDispVals)]);
}

template<> inline float32x4_t VectProcessingFuncts::LoadPackedDataUnaligned<float, float32x4_t>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, float* inData)
{
  return vld1q_f32(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline float16x4_t VectProcessingFuncts::LoadPackedDataUnaligned<float16_t, float16x4_t>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, float16_t* inData)
{
  return vld1_f16(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline float64x2_t VectProcessingFuncts::LoadPackedDataUnaligned<double, float64x2_t>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, double* inData)
{
  return vld1q_f64(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
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

template<> inline float32x4_t VectProcessingFuncts::AddVals<float32x4_t, float32x4_t, float32x4_t>(const float32x4_t& val1, const float32x4_t& val2) {
  return vaddq_f32(val1, val2);
}

template<> inline float64x2_t VectProcessingFuncts::AddVals<float64x2_t, float64x2_t, float64x2_t>(const float64x2_t& val1, const float64x2_t& val2) {
  return vaddq_f64(val1, val2);
}

template<> inline float32x4_t VectProcessingFuncts::AddVals<float32x4_t, float16x4_t, float32x4_t>(const float32x4_t& val1, const float16x4_t& val2) {
  return vaddq_f32(val1, vcvt_f32_f16(val2));
}

template<> inline float32x4_t VectProcessingFuncts::AddVals<float16x4_t, float32x4_t, float32x4_t>(const float16x4_t& val1, const float32x4_t& val2) {
  return vaddq_f32(vcvt_f32_f16(val1), val2);
}

template<> inline float32x4_t VectProcessingFuncts::AddVals<float16x4_t, float16x4_t, float32x4_t>(const float16x4_t& val1, const float16x4_t& val2) {
  return vaddq_f32(vcvt_f32_f16(val1), vcvt_f32_f16(val2));
}

template<> inline float32x4_t VectProcessingFuncts::SubtractVals<float32x4_t, float32x4_t, float32x4_t>(const float32x4_t& val1, const float32x4_t& val2) {
  return vsubq_f32(val1, val2);
}

template<> inline float64x2_t VectProcessingFuncts::SubtractVals<float64x2_t, float64x2_t, float64x2_t>(const float64x2_t& val1, const float64x2_t& val2) {
  return vsubq_f64(val1, val2);
}

template<> inline float32x4_t VectProcessingFuncts::divideVals<float32x4_t, float32x4_t, float32x4_t>(const float32x4_t& val1, const float32x4_t& val2) {
  return vdivq_f32(val1, val2);
}

template<> inline float64x2_t VectProcessingFuncts::divideVals<float64x2_t, float64x2_t, float64x2_t>(const float64x2_t& val1, const float64x2_t& val2) {
  return vdivq_f64(val1, val2);
}

template<> inline float32x4_t VectProcessingFuncts::ConvertValToDatatype<float32x4_t, float>(float val) {
  return vdupq_n_f32(val);
}

template<> inline float64x2_t VectProcessingFuncts::ConvertValToDatatype<float64x2_t, double>(double val) {
  return vdupq_n_f64(val);
}

template<> inline float32x4_t VectProcessingFuncts::GetMinByElement<float32x4_t>(const float32x4_t& val1, const float32x4_t& val2) {
  return vminnmq_f32(val1, val2);
}

template<> inline float64x2_t VectProcessingFuncts::GetMinByElement<float64x2_t>(const float64x2_t& val1, const float64x2_t& val2) {
  return vminnmq_f64(val1, val2);
}

template<> inline void VectProcessingFuncts::StorePackedDataAligned<float, float32x4_t>(
  unsigned int indexDataStore, float* locationDataStore, const float32x4_t& dataToStore)
{
  vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::StorePackedDataAligned<float16_t, float32x4_t>(
  unsigned int indexDataStore, float16_t* locationDataStore, const float32x4_t& dataToStore)
{
  vst1_f16(&locationDataStore[indexDataStore], vcvt_f16_f32(dataToStore));
}

template<> inline void VectProcessingFuncts::StorePackedDataAligned<double, float64x2_t>(
  unsigned int indexDataStore, double* locationDataStore, const float64x2_t& dataToStore)
{
  vst1q_f64(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::StorePackedDataUnaligned<float, float32x4_t>(
  unsigned int indexDataStore, float* locationDataStore, const float32x4_t& dataToStore)
{
  vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::StorePackedDataUnaligned<float16_t, float32x4_t>(
  unsigned int indexDataStore, float16_t* locationDataStore, const float32x4_t& dataToStore)
{
  vst1_f16(&locationDataStore[indexDataStore], vcvt_f16_f32(dataToStore));
}

template<> inline void VectProcessingFuncts::StorePackedDataUnaligned<double, float64x2_t>(
  unsigned int indexDataStore, double* locationDataStore, const float64x2_t& dataToStore)
{
  vst1q_f64(&locationDataStore[indexDataStore], dataToStore);
}

#endif /* NEONTEMPLATESPFUNCTS_H_ */
