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
 * @file NEONTemplateSpFuncts.h
 * @author Scott Grauer-Gray
 * @brief Template specializations for processing on SIMD vector data types supported by NEON
 * on ARM CPUs.
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef NEON_TEMPLATE_SP_FUNCTS_H_
#define NEON_TEMPLATE_SP_FUNCTS_H_

//NEON only used when processing on an ARM CPU that supports NEON instructions
#include <arm_neon.h>

template<> inline float64x2_t simd_processing::LoadPackedDataAligned<double, float64x2_t>(
  size_t indexDataLoad, const double* inData)
{
  return vld1q_f64(&inData[indexDataLoad]);
}

template<> inline float32x4_t simd_processing::LoadPackedDataAligned<float, float32x4_t>(
  size_t indexDataLoad, const float* inData)
{
  return vld1q_f32(&inData[indexDataLoad]);
}

template<> inline float16x4_t simd_processing::LoadPackedDataAligned<float16_t, float16x4_t>(
  size_t indexDataLoad, const float16_t* inData)
{
  return vld1_f16(&inData[indexDataLoad]);
}

template<> inline float32x4_t simd_processing::LoadPackedDataUnaligned<float, float32x4_t>(
  size_t indexDataLoad, const float* inData)
{
  return vld1q_f32(&inData[indexDataLoad]);
}

template<> inline float16x4_t simd_processing::LoadPackedDataUnaligned<float16_t, float16x4_t>(
  size_t indexDataLoad, const float16_t* inData)
{
  return vld1_f16(&inData[indexDataLoad]);
}

template<> inline float64x2_t simd_processing::LoadPackedDataUnaligned<double, float64x2_t>(
  size_t indexDataLoad, const double* inData)
{
  return vld1q_f64(&inData[indexDataLoad]);
}

template<> inline float32x4_t simd_processing::createSIMDVectorSameData<float32x4_t>(float data) {
  return vdupq_n_f32(data);
}

template<> inline float16x4_t simd_processing::createSIMDVectorSameData<float16x4_t>(float data) {
  return vcvt_f16_f32(createSIMDVectorSameData<float32x4_t>(data));
}

template<> inline float64x2_t simd_processing::createSIMDVectorSameData<float64x2_t>(float data) {
  return vdupq_n_f64((double)data);
}

template<> inline float32x4_t simd_processing::AddVals<float32x4_t, float32x4_t, float32x4_t>(
  const float32x4_t& val1, const float32x4_t& val2)
{
  return vaddq_f32(val1, val2);
}

template<> inline float64x2_t simd_processing::AddVals<float64x2_t, float64x2_t, float64x2_t>(
  const float64x2_t& val1, const float64x2_t& val2)
{
  return vaddq_f64(val1, val2);
}

template<> inline float32x4_t simd_processing::AddVals<float32x4_t, float16x4_t, float32x4_t>(
  const float32x4_t& val1, const float16x4_t& val2)
{
  return vaddq_f32(val1, vcvt_f32_f16(val2));
}

template<> inline float32x4_t simd_processing::AddVals<float16x4_t, float32x4_t, float32x4_t>(
  const float16x4_t& val1, const float32x4_t& val2)
{
  return vaddq_f32(vcvt_f32_f16(val1), val2);
}

template<> inline float32x4_t simd_processing::AddVals<float16x4_t, float16x4_t, float32x4_t>(
  const float16x4_t& val1, const float16x4_t& val2)
{
  return vaddq_f32(vcvt_f32_f16(val1), vcvt_f32_f16(val2));
}

template<> inline float32x4_t simd_processing::SubtractVals<float32x4_t, float32x4_t, float32x4_t>(
  const float32x4_t& val1, const float32x4_t& val2)
{
  return vsubq_f32(val1, val2);
}

template<> inline float64x2_t simd_processing::SubtractVals<float64x2_t, float64x2_t, float64x2_t>(
  const float64x2_t& val1, const float64x2_t& val2)
{
  return vsubq_f64(val1, val2);
}

template<> inline float32x4_t simd_processing::divideVals<float32x4_t, float32x4_t, float32x4_t>(
  const float32x4_t& val1, const float32x4_t& val2)
{
  return vdivq_f32(val1, val2);
}

template<> inline float64x2_t simd_processing::divideVals<float64x2_t, float64x2_t, float64x2_t>(
  const float64x2_t& val1, const float64x2_t& val2)
{
  return vdivq_f64(val1, val2);
}

template<> inline float32x4_t simd_processing::ConvertValToDatatype<float32x4_t, float>(float val) {
  return vdupq_n_f32(val);
}

template<> inline float64x2_t simd_processing::ConvertValToDatatype<float64x2_t, double>(double val) {
  return vdupq_n_f64(val);
}

template<> inline float32x4_t simd_processing::GetMinByElement<float32x4_t>(
  const float32x4_t& val1, const float32x4_t& val2)
{
  return vminnmq_f32(val1, val2);
}

template<> inline float64x2_t simd_processing::GetMinByElement<float64x2_t>(
  const float64x2_t& val1, const float64x2_t& val2)
{
  return vminnmq_f64(val1, val2);
}

template<> inline void simd_processing::StorePackedDataAligned<float, float32x4_t>(
  size_t indexDataStore, float* locationDataStore, const float32x4_t& dataToStore)
{
  vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void simd_processing::StorePackedDataAligned<float16_t, float32x4_t>(
  size_t indexDataStore, float16_t* locationDataStore, const float32x4_t& dataToStore)
{
  vst1_f16(&locationDataStore[indexDataStore], vcvt_f16_f32(dataToStore));
}

template<> inline void simd_processing::StorePackedDataAligned<double, float64x2_t>(
  size_t indexDataStore, double* locationDataStore, const float64x2_t& dataToStore)
{
  vst1q_f64(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void simd_processing::StorePackedDataUnaligned<float, float32x4_t>(
  size_t indexDataStore, float* locationDataStore, const float32x4_t& dataToStore)
{
  vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void simd_processing::StorePackedDataUnaligned<float16_t, float32x4_t>(
  size_t indexDataStore, float16_t* locationDataStore, const float32x4_t& dataToStore)
{
  vst1_f16(&locationDataStore[indexDataStore], vcvt_f16_f32(dataToStore));
}

template<> inline void simd_processing::StorePackedDataUnaligned<double, float64x2_t>(
  size_t indexDataStore, double* locationDataStore, const float64x2_t& dataToStore)
{
  vst1q_f64(&locationDataStore[indexDataStore], dataToStore);
}

#endif /* NEON_TEMPLATE_SP_FUNCTS_H_ */
