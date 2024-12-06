/*
 * AVX256TemplateSpFuncts.h
 *
 * Template specializations for processing on SIMD vector data types supported by AVX256.
 *
 *  Created on: Jun 26, 2024
 *      Author: scott
 */

#ifndef AVX256TEMPLATESPFUNCTS_H_
#define AVX256TEMPLATESPFUNCTS_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "VectProcessingFuncts.h"

template<> inline __m256d VectProcessingFuncts::LoadPackedDataAligned<double, __m256d>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, const double* inData)
{
  return _mm256_load_pd(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline __m256 VectProcessingFuncts::LoadPackedDataAligned<float, __m256>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, const float* inData)
{
  return _mm256_load_ps(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline __m128i VectProcessingFuncts::LoadPackedDataAligned<short, __m128i>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, const short* inData)
{
  return _mm_load_si128((__m128i *)(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity,
    numDispVals)]));
}

template<> inline __m256 VectProcessingFuncts::LoadPackedDataUnaligned<float, __m256>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, const float* inData)
{
  return _mm256_loadu_ps(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline __m128i VectProcessingFuncts::LoadPackedDataUnaligned<short, __m128i>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, const short* inData)
{
  return _mm_loadu_si128((__m128i*)(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]));
}

template<> inline __m256d VectProcessingFuncts::LoadPackedDataUnaligned<double, __m256d>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, const double* inData)
{
  return _mm256_loadu_pd(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline __m256 VectProcessingFuncts::createSIMDVectorSameData<__m256>(float data) {
  return _mm256_set1_ps(data);
}

template<> inline __m128i VectProcessingFuncts::createSIMDVectorSameData<__m128i>(float data) {
  return _mm256_cvtps_ph(_mm256_set1_ps(data), 0);
}

template<> inline __m256d VectProcessingFuncts::createSIMDVectorSameData<__m256d>(float data) {
  return _mm256_set1_pd((double)data);
}

template<> inline __m256 VectProcessingFuncts::AddVals<__m256, __m256, __m256>(const __m256& val1, const __m256& val2) {
  return _mm256_add_ps(val1, val2);
}

template<> inline __m256d VectProcessingFuncts::AddVals<__m256d, __m256d, __m256d>(const __m256d& val1, const __m256d& val2) {
  return _mm256_add_pd(val1, val2);
}

template<> inline __m256 VectProcessingFuncts::AddVals<__m256, __m128i, __m256>(const __m256& val1, const __m128i& val2) {
  return _mm256_add_ps(val1, _mm256_cvtph_ps(val2));
}

template<> inline __m256 VectProcessingFuncts::AddVals<__m128i, __m256, __m256>(const __m128i& val1, const __m256& val2) {
  return _mm256_add_ps(_mm256_cvtph_ps(val1), val2);
}

template<> inline __m256 VectProcessingFuncts::AddVals<__m128i, __m128i, __m256>(const __m128i& val1, const __m128i& val2) {
  return _mm256_add_ps(_mm256_cvtph_ps(val1), _mm256_cvtph_ps(val2));
}

template<> inline __m256 VectProcessingFuncts::SubtractVals<__m256, __m256, __m256>(const __m256& val1, const __m256& val2) {
  return _mm256_sub_ps(val1, val2);
}

template<> inline __m256d VectProcessingFuncts::SubtractVals<__m256d, __m256d, __m256d>(const __m256d& val1, const __m256d& val2) {
  return _mm256_sub_pd(val1, val2);
}

template<> inline __m256 VectProcessingFuncts::divideVals<__m256, __m256, __m256>(const __m256& val1, const __m256& val2) {
  return _mm256_div_ps(val1, val2);
}

template<> inline __m256d VectProcessingFuncts::divideVals<__m256d, __m256d, __m256d>(const __m256d& val1, const __m256d& val2) {
  return _mm256_div_pd(val1, val2);
}

template<> inline __m256 VectProcessingFuncts::ConvertValToDatatype<__m256, float>(float val) {
  return _mm256_set1_ps(val);
}

template<> inline __m256d VectProcessingFuncts::ConvertValToDatatype<__m256d, double>(double val) {
  return _mm256_set1_pd(val);
}

template<> inline __m256 VectProcessingFuncts::GetMinByElement<__m256>(const __m256& val1, const __m256& val2) {
  return _mm256_min_ps(val1, val2);
}

template<> inline __m256d VectProcessingFuncts::GetMinByElement<__m256d>(const __m256d& val1, const __m256d& val2) {
  return _mm256_min_pd(val1, val2);
}

template<> inline void VectProcessingFuncts::StorePackedDataAligned<float, __m256>(
  unsigned int indexDataStore, float* locationDataStore, const __m256& dataToStore)
{
  _mm256_store_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::StorePackedDataAligned<short, __m256>(
  unsigned int indexDataStore, short* locationDataStore, const __m256& dataToStore)
{
  _mm_store_si128((__m128i*)(&locationDataStore[indexDataStore]), _mm256_cvtps_ph(dataToStore, 0));
}

template<> inline void VectProcessingFuncts::StorePackedDataAligned<double, __m256d>(
  unsigned int indexDataStore, double* locationDataStore, const __m256d& dataToStore)
{
  _mm256_store_pd(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::StorePackedDataUnaligned<float, __m256>(
  unsigned int indexDataStore, float* locationDataStore, const __m256& dataToStore)
{
  _mm256_storeu_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::StorePackedDataUnaligned<short, __m256>(
  unsigned int indexDataStore, short* locationDataStore, const __m256& dataToStore)
{
  _mm_storeu_si128((__m128i*)(&locationDataStore[indexDataStore]), _mm256_cvtps_ph(dataToStore, 0));
}

template<> inline void VectProcessingFuncts::StorePackedDataUnaligned<double, __m256d>(
  unsigned int indexDataStore, double* locationDataStore, const __m256d& dataToStore)
{
  _mm256_storeu_pd(&locationDataStore[indexDataStore], dataToStore);
}

#endif /* AVX256TEMPLATESPFUNCTS_H_ */