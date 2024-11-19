/*
 * kAVX512TemplateSpFuncts.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef kAVX512TEMPLATESPFUNCTS_H_
#define kAVX512TEMPLATESPFUNCTS_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "VectProcessingFuncts.h"

template<> inline __m512d VectProcessingFuncts::LoadPackedDataAligned<double, __m512d>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, double* inData)
{
  return _mm512_load_pd(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline __m512 VectProcessingFuncts::LoadPackedDataAligned<float, __m512>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, float* inData)
{
  return _mm512_load_ps(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline __m256i VectProcessingFuncts::LoadPackedDataAligned<short, __m256i>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, short* inData)
{
  return _mm256_load_si256((__m256i*)(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity,
    numDispVals)]));
}

template<> inline __m512 VectProcessingFuncts::LoadPackedDataUnaligned<float, __m512>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, float* inData)
{
  return _mm512_loadu_ps(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline __m256i VectProcessingFuncts::LoadPackedDataUnaligned<short, __m256i>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, short* inData)
{
  return _mm256_loadu_si256((__m256i*)(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]));
}

template<> inline __m512d VectProcessingFuncts::LoadPackedDataUnaligned<double, __m512d>(
  unsigned int x, unsigned int y, unsigned int current_disparity,
  const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, double* inData)
{
  return _mm512_loadu_pd(&inData[beliefprop::RetrieveIndexInDataAndMessage(
    x, y, current_bp_level.padded_width_checkerboard_level_,
    current_bp_level.height_level_, current_disparity, numDispVals)]);
}

template<> inline __m512 VectProcessingFuncts::createSIMDVectorSameData<__m512>(float data) {
  return _mm512_set1_ps(data);
}

template<> inline __m256i VectProcessingFuncts::createSIMDVectorSameData<__m256i>(float data) {
  return _mm512_cvtps_ph(_mm512_set1_ps(data), 0);
}

template<> inline __m512d VectProcessingFuncts::createSIMDVectorSameData<__m512d>(float data) {
  return _mm512_set1_pd((double)data);
}

template<> inline __m512 VectProcessingFuncts::AddVals<__m512, __m512, __m512>(const __m512& val1, const __m512& val2) {
  return _mm512_add_ps(val1, val2);
}

template<> inline __m512d VectProcessingFuncts::AddVals<__m512d, __m512d, __m512d>(const __m512d& val1, const __m512d& val2) {
  return _mm512_add_pd(val1, val2);
}

template<> inline __m512 VectProcessingFuncts::AddVals<__m512, __m256i, __m512>(const __m512& val1, const __m256i& val2) {
  return _mm512_add_ps(val1, _mm512_cvtph_ps(val2));
}

template<> inline __m512 VectProcessingFuncts::AddVals<__m256i, __m512, __m512>(const __m256i& val1, const __m512& val2) {
  return _mm512_add_ps(_mm512_cvtph_ps(val1), val2);
}

template<> inline __m512 VectProcessingFuncts::AddVals<__m256i, __m256i, __m512>(const __m256i& val1, const __m256i& val2) {
  return _mm512_add_ps(_mm512_cvtph_ps(val1), _mm512_cvtph_ps(val2));
}

template<> inline __m512 VectProcessingFuncts::SubtractVals<__m512, __m512, __m512>(const __m512& val1, const __m512& val2) {
  return _mm512_sub_ps(val1, val2);
}

template<> inline __m512d VectProcessingFuncts::SubtractVals<__m512d, __m512d, __m512d>(const __m512d& val1, const __m512d& val2) {
  return _mm512_sub_pd(val1, val2);
}

template<> inline __m512 VectProcessingFuncts::divideVals<__m512, __m512, __m512>(const __m512& val1, const __m512& val2) {
  return _mm512_div_ps(val1, val2);
}

template<> inline __m512d VectProcessingFuncts::divideVals<__m512d, __m512d, __m512d>(const __m512d& val1, const __m512d& val2) {
  return _mm512_div_pd(val1, val2);
}

template<> inline __m512 VectProcessingFuncts::ConvertValToDatatype<__m512, float>(float val) {
  return _mm512_set1_ps(val);
}

template<> inline __m512d VectProcessingFuncts::ConvertValToDatatype<__m512d, double>(double val) {
  return _mm512_set1_pd(val);
}

template<> inline __m512 VectProcessingFuncts::GetMinByElement<__m512>(const __m512& val1, const __m512& val2) {
  return _mm512_min_ps(val1, val2);
}

template<> inline __m512d VectProcessingFuncts::GetMinByElement<__m512d>(const __m512d& val1, const __m512d& val2) {
  return _mm512_min_pd(val1, val2);
}

template<> inline void VectProcessingFuncts::StorePackedDataAligned<float, __m512>(
  unsigned int indexDataStore, float* locationDataStore, const __m512& dataToStore)
{
  _mm512_store_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::StorePackedDataAligned<short, __m512>(
  unsigned int indexDataStore, short* locationDataStore, const __m512& dataToStore)
{
  _mm256_store_si256((__m256i*)(&locationDataStore[indexDataStore]), _mm512_cvtps_ph(dataToStore, 0));
}

template<> inline void VectProcessingFuncts::StorePackedDataAligned<double, __m512d>(
  unsigned int indexDataStore, double* locationDataStore, const __m512d& dataToStore)
{
  _mm512_store_pd(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::StorePackedDataUnaligned<float, __m512>(
  unsigned int indexDataStore, float* locationDataStore, const __m512& dataToStore)
{
  _mm512_storeu_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::StorePackedDataUnaligned<short, __m512>(
  unsigned int indexDataStore, short* locationDataStore, const __m512& dataToStore)
{
  _mm256_storeu_si256((__m256i*)(&locationDataStore[indexDataStore]), _mm512_cvtps_ph(dataToStore, 0));
}

template<> inline void VectProcessingFuncts::StorePackedDataUnaligned<double, __m512d>(
  unsigned int indexDataStore, double* locationDataStore, const __m512d& dataToStore)
{
  _mm512_storeu_pd(&locationDataStore[indexDataStore], dataToStore);
}

#endif /* kAVX512TEMPLATESPFUNCTS_H_ */
