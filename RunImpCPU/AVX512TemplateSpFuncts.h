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

template<> inline __m512d VectProcessingFuncts::loadPackedDataAligned<double, __m512d>(
  unsigned int x, unsigned int y, unsigned int currentDisparity,
  const beliefprop::LevelProperties& currentLevelProperties, unsigned int numDispVals, double* inData)
{
  return _mm512_load_pd(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentLevelProperties.padded_width_checkerboard_level_,
    currentLevelProperties.height_level_, currentDisparity, numDispVals)]);
}

template<> inline __m512 VectProcessingFuncts::loadPackedDataAligned<float, __m512>(
  unsigned int x, unsigned int y, unsigned int currentDisparity,
  const beliefprop::LevelProperties& currentLevelProperties, unsigned int numDispVals, float* inData)
{
  return _mm512_load_ps(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentLevelProperties.padded_width_checkerboard_level_,
    currentLevelProperties.height_level_, currentDisparity, numDispVals)]);
}

template<> inline __m256i VectProcessingFuncts::loadPackedDataAligned<short, __m256i>(
  unsigned int x, unsigned int y, unsigned int currentDisparity,
  const beliefprop::LevelProperties& currentLevelProperties, unsigned int numDispVals, short* inData)
{
  return _mm256_load_si256((__m256i*)(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentLevelProperties.padded_width_checkerboard_level_,
    currentLevelProperties.height_level_, currentDisparity,
    numDispVals)]));
}

template<> inline __m512 VectProcessingFuncts::loadPackedDataUnaligned<float, __m512>(
  unsigned int x, unsigned int y, unsigned int currentDisparity,
  const beliefprop::LevelProperties& currentLevelProperties, unsigned int numDispVals, float* inData)
{
  return _mm512_loadu_ps(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentLevelProperties.padded_width_checkerboard_level_,
    currentLevelProperties.height_level_, currentDisparity, numDispVals)]);
}

template<> inline __m256i VectProcessingFuncts::loadPackedDataUnaligned<short, __m256i>(
  unsigned int x, unsigned int y, unsigned int currentDisparity,
  const beliefprop::LevelProperties& currentLevelProperties, unsigned int numDispVals, short* inData)
{
  return _mm256_loadu_si256((__m256i*)(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentLevelProperties.padded_width_checkerboard_level_,
    currentLevelProperties.height_level_, currentDisparity, numDispVals)]));
}

template<> inline __m512d VectProcessingFuncts::loadPackedDataUnaligned<double, __m512d>(
  unsigned int x, unsigned int y, unsigned int currentDisparity,
  const beliefprop::LevelProperties& currentLevelProperties, unsigned int numDispVals, double* inData)
{
  return _mm512_loadu_pd(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentLevelProperties.padded_width_checkerboard_level_,
    currentLevelProperties.height_level_, currentDisparity, numDispVals)]);
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

template<> inline __m512 VectProcessingFuncts::addVals<__m512, __m512, __m512>(const __m512& val1, const __m512& val2) {
  return _mm512_add_ps(val1, val2);
}

template<> inline __m512d VectProcessingFuncts::addVals<__m512d, __m512d, __m512d>(const __m512d& val1, const __m512d& val2) {
  return _mm512_add_pd(val1, val2);
}

template<> inline __m512 VectProcessingFuncts::addVals<__m512, __m256i, __m512>(const __m512& val1, const __m256i& val2) {
  return _mm512_add_ps(val1, _mm512_cvtph_ps(val2));
}

template<> inline __m512 VectProcessingFuncts::addVals<__m256i, __m512, __m512>(const __m256i& val1, const __m512& val2) {
  return _mm512_add_ps(_mm512_cvtph_ps(val1), val2);
}

template<> inline __m512 VectProcessingFuncts::addVals<__m256i, __m256i, __m512>(const __m256i& val1, const __m256i& val2) {
  return _mm512_add_ps(_mm512_cvtph_ps(val1), _mm512_cvtph_ps(val2));
}

template<> inline __m512 VectProcessingFuncts::subtractVals<__m512, __m512, __m512>(const __m512& val1, const __m512& val2) {
  return _mm512_sub_ps(val1, val2);
}

template<> inline __m512d VectProcessingFuncts::subtractVals<__m512d, __m512d, __m512d>(const __m512d& val1, const __m512d& val2) {
  return _mm512_sub_pd(val1, val2);
}

template<> inline __m512 VectProcessingFuncts::divideVals<__m512, __m512, __m512>(const __m512& val1, const __m512& val2) {
  return _mm512_div_ps(val1, val2);
}

template<> inline __m512d VectProcessingFuncts::divideVals<__m512d, __m512d, __m512d>(const __m512d& val1, const __m512d& val2) {
  return _mm512_div_pd(val1, val2);
}

template<> inline __m512 VectProcessingFuncts::convertValToDatatype<__m512, float>(float val) {
  return _mm512_set1_ps(val);
}

template<> inline __m512d VectProcessingFuncts::convertValToDatatype<__m512d, double>(double val) {
  return _mm512_set1_pd(val);
}

template<> inline __m512 VectProcessingFuncts::getMinByElement<__m512>(const __m512& val1, const __m512& val2) {
  return _mm512_min_ps(val1, val2);
}

template<> inline __m512d VectProcessingFuncts::getMinByElement<__m512d>(const __m512d& val1, const __m512d& val2) {
  return _mm512_min_pd(val1, val2);
}

template<> inline void VectProcessingFuncts::storePackedDataAligned<float, __m512>(
  unsigned int indexDataStore, float* locationDataStore, const __m512& dataToStore)
{
  _mm512_store_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::storePackedDataAligned<short, __m512>(
  unsigned int indexDataStore, short* locationDataStore, const __m512& dataToStore)
{
  _mm256_store_si256((__m256i*)(&locationDataStore[indexDataStore]), _mm512_cvtps_ph(dataToStore, 0));
}

template<> inline void VectProcessingFuncts::storePackedDataAligned<double, __m512d>(
  unsigned int indexDataStore, double* locationDataStore, const __m512d& dataToStore)
{
  _mm512_store_pd(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::storePackedDataUnaligned<float, __m512>(
  unsigned int indexDataStore, float* locationDataStore, const __m512& dataToStore)
{
  _mm512_storeu_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::storePackedDataUnaligned<short, __m512>(
  unsigned int indexDataStore, short* locationDataStore, const __m512& dataToStore)
{
  _mm256_storeu_si256((__m256i*)(&locationDataStore[indexDataStore]), _mm512_cvtps_ph(dataToStore, 0));
}

template<> inline void VectProcessingFuncts::storePackedDataUnaligned<double, __m512d>(
  unsigned int indexDataStore, double* locationDataStore, const __m512d& dataToStore)
{
  _mm512_storeu_pd(&locationDataStore[indexDataStore], dataToStore);
}

#endif /* kAVX512TEMPLATESPFUNCTS_H_ */
