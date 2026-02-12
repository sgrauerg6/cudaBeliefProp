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
 * @file AVX512TemplateSpFuncts.h
 * @author Scott Grauer-Gray
 * @brief Template specializations for processing on SIMD vector data types supported by AVX512.
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef AVX512TEMPLATESPFUNCTS_H_
#define AVX512TEMPLATESPFUNCTS_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "RunImpCPU/RunCPUSettings.h"
#include "SIMDProcessing.h"
#include <immintrin.h>

template<> inline __m512d simd_processing::LoadPackedDataAligned<double, __m512d>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level,
  unsigned int numDispVals, const double* inData)
{
  return _mm512_load_pd(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline __m512 simd_processing::LoadPackedDataAligned<float, __m512>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level,
  unsigned int numDispVals, const float* inData)
{
  return _mm512_load_ps(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline __m256i simd_processing::LoadPackedDataAligned<short, __m256i>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level,
  unsigned int numDispVals, const short* inData)
{
  return _mm256_load_si256((__m256i*)(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity,
    numDispVals)]));
}

#if defined(FLOAT16_VECTORIZATION)

template<> inline __m512h simd_processing::LoadPackedDataAligned<_Float16, __m512h>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level,
  unsigned int numDispVals, const _Float16* inData)
{
  return _mm512_load_ph((__m512h*)(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity,
    numDispVals)]));
}

#endif //FLOAT16_VECTORIZATION

template<> inline __m512 simd_processing::LoadPackedDataUnaligned<float, __m512>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level,
  unsigned int numDispVals, const float* inData)
{
  return _mm512_loadu_ps(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline __m256i simd_processing::LoadPackedDataUnaligned<short, __m256i>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level,
  unsigned int numDispVals, const short* inData)
{
  return _mm256_loadu_si256((__m256i*)(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]));
}

template<> inline __m512d simd_processing::LoadPackedDataUnaligned<double, __m512d>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level,
  unsigned int numDispVals, const double* inData)
{
  return _mm512_loadu_pd(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

#if defined(FLOAT16_VECTORIZATION)

template<> inline __m512h simd_processing::LoadPackedDataUnaligned<_Float16, __m512h>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level,
  unsigned int numDispVals, const _Float16* inData)
{
  return _mm512_loadu_ph(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

#endif //FLOAT16_VECTORIZATION

template<> inline __m512 simd_processing::createSIMDVectorSameData<__m512>(float data) {
  return _mm512_set1_ps(data);
}

template<> inline __m256i simd_processing::createSIMDVectorSameData<__m256i>(float data) {
  return _mm512_cvtps_ph(_mm512_set1_ps(data), 0);
}

template<> inline __m512d simd_processing::createSIMDVectorSameData<__m512d>(float data) {
  return _mm512_set1_pd((double)data);
}

#if defined(FLOAT16_VECTORIZATION)

template<> inline __m512h simd_processing::createSIMDVectorSameData<__m512h>(float data) {
  return _mm512_set1_ph((_Float16)data);
}

#endif //FLOAT16_VECTORIZATION

template<> inline __m512 simd_processing::AddVals<__m512, __m512, __m512>(
  const __m512& val1, const __m512& val2)
{
  return _mm512_add_ps(val1, val2);
}

template<> inline __m512d simd_processing::AddVals<__m512d, __m512d, __m512d>(
  const __m512d& val1, const __m512d& val2)
{
  return _mm512_add_pd(val1, val2);
}

#if defined(FLOAT16_VECTORIZATION)

template<> inline __m512h simd_processing::AddVals<__m512h, __m512h, __m512h>(
  const __m512h& val1, const __m512h& val2)
{
  return _mm512_add_ph(val1, val2);
}

#endif //FLOAT16_VECTORIZATION

template<> inline __m512 simd_processing::AddVals<__m512, __m256i, __m512>(
  const __m512& val1, const __m256i& val2)
{
  return _mm512_add_ps(val1, _mm512_cvtph_ps(val2));
}

template<> inline __m512 simd_processing::AddVals<__m256i, __m512, __m512>(
  const __m256i& val1, const __m512& val2)
{
  return _mm512_add_ps(_mm512_cvtph_ps(val1), val2);
}

template<> inline __m512 simd_processing::AddVals<__m256i, __m256i, __m512>(
  const __m256i& val1, const __m256i& val2)
{
  return _mm512_add_ps(_mm512_cvtph_ps(val1), _mm512_cvtph_ps(val2));
}

template<> inline __m512 simd_processing::SubtractVals<__m512, __m512, __m512>(
  const __m512& val1, const __m512& val2)
{
  return _mm512_sub_ps(val1, val2);
}

template<> inline __m512d simd_processing::SubtractVals<__m512d, __m512d, __m512d>(
  const __m512d& val1, const __m512d& val2)
{
  return _mm512_sub_pd(val1, val2);
}

#if defined(FLOAT16_VECTORIZATION)

template<> inline __m512h simd_processing::SubtractVals<__m512h, __m512h, __m512h>(
  const __m512h& val1, const __m512h& val2)
{
  return _mm512_sub_ph(val1, val2);
}

#endif //FLOAT16_VECTORIZATION

template<> inline __m512 simd_processing::divideVals<__m512, __m512, __m512>(
  const __m512& val1, const __m512& val2)
{
  return _mm512_div_ps(val1, val2);
}

template<> inline __m512d simd_processing::divideVals<__m512d, __m512d, __m512d>(
  const __m512d& val1, const __m512d& val2)
{
  return _mm512_div_pd(val1, val2);
}

#if defined(FLOAT16_VECTORIZATION)

template<> inline __m512h simd_processing::divideVals<__m512h, __m512h, __m512h>(
  const __m512h& val1, const __m512h& val2)
{
  return _mm512_div_ph(val1, val2);
}

#endif //FLOAT16_VECTORIZATION

template<> inline __m512 simd_processing::ConvertValToDatatype<__m512, float>(float val) {
  return _mm512_set1_ps(val);
}

template<> inline __m512d simd_processing::ConvertValToDatatype<__m512d, double>(double val) {
  return _mm512_set1_pd(val);
}

#if defined(FLOAT16_VECTORIZATION)

template<> inline __m512h simd_processing::ConvertValToDatatype<__m512h, _Float16>(_Float16 val) {
  return _mm512_set1_ph((_Float16)val);
}

#endif //FLOAT16_VECTORIZATION

template<> inline __m512 simd_processing::GetMinByElement<__m512>(
  const __m512& val1, const __m512& val2)
{
  return _mm512_min_ps(val1, val2);
}

template<> inline __m512d simd_processing::GetMinByElement<__m512d>(
  const __m512d& val1, const __m512d& val2)
{
  return _mm512_min_pd(val1, val2);
}

#if defined(FLOAT16_VECTORIZATION)

template<> inline __m512h simd_processing::GetMinByElement<__m512h>(
  const __m512h& val1, const __m512h& val2)
{
  return _mm512_min_ph(val1, val2);
}

#endif //FLOAT16_VECTORIZATION

template<> inline void simd_processing::StorePackedDataAligned<float, __m512>(
  size_t indexDataStore, float* locationDataStore, const __m512& dataToStore)
{
  _mm512_store_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void simd_processing::StorePackedDataAligned<short, __m512>(
  size_t indexDataStore, short* locationDataStore, const __m512& dataToStore)
{
  _mm256_store_si256((__m256i*)(&locationDataStore[indexDataStore]), _mm512_cvtps_ph(dataToStore, 0));
}

template<> inline void simd_processing::StorePackedDataAligned<double, __m512d>(
  size_t indexDataStore, double* locationDataStore, const __m512d& dataToStore)
{
  _mm512_store_pd(&locationDataStore[indexDataStore], dataToStore);
}

#if defined(FLOAT16_VECTORIZATION)

template<> inline void simd_processing::StorePackedDataAligned<_Float16, __m512h>(
  size_t indexDataStore, _Float16* locationDataStore, const __m512h& dataToStore)
{
  _mm512_store_ph(&locationDataStore[indexDataStore], dataToStore);
}

#endif //FLOAT16_VECTORIZATION

template<> inline void simd_processing::StorePackedDataUnaligned<float, __m512>(
  size_t indexDataStore, float* locationDataStore, const __m512& dataToStore)
{
  _mm512_storeu_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void simd_processing::StorePackedDataUnaligned<short, __m512>(
  size_t indexDataStore, short* locationDataStore, const __m512& dataToStore)
{
  _mm256_storeu_si256((__m256i*)(&locationDataStore[indexDataStore]), _mm512_cvtps_ph(dataToStore, 0));
}

template<> inline void simd_processing::StorePackedDataUnaligned<double, __m512d>(
  size_t indexDataStore, double* locationDataStore, const __m512d& dataToStore)
{
  _mm512_storeu_pd(&locationDataStore[indexDataStore], dataToStore);
}

#if defined(FLOAT16_VECTORIZATION)

template<> inline void simd_processing::StorePackedDataUnaligned<_Float16, __m512h>(
  size_t indexDataStore, _Float16* locationDataStore, const __m512h& dataToStore)
{
  _mm512_storeu_ph(&locationDataStore[indexDataStore], dataToStore);
}

#endif //FLOAT16_VECTORIZATION

#endif /* AVX512TEMPLATESPFUNCTS_H_ */
