/*
 * AVX256TemplateSpFuncts.h
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

template<> inline __m256d VectProcessingFuncts::loadPackedDataAligned<double, __m256d>(
  const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
  const beliefprop::levelProperties& currentLevelProperties, const unsigned int numDispVals, double* inData)
{
  return _mm256_load_pd(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
    currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline __m256 VectProcessingFuncts::loadPackedDataAligned<float, __m256>(
  const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
  const beliefprop::levelProperties& currentLevelProperties, const unsigned int numDispVals, float* inData)
{
  return _mm256_load_ps(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
    currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline __m128i VectProcessingFuncts::loadPackedDataAligned<short, __m128i>(
  const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
  const beliefprop::levelProperties& currentLevelProperties, const unsigned int numDispVals, short* inData)
{
  return _mm_load_si128((__m128i *)(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
    currentLevelProperties.heightLevel_, currentDisparity,
    numDispVals)]));
}

template<> inline __m256 VectProcessingFuncts::loadPackedDataUnaligned<float, __m256>(
  const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
  const beliefprop::levelProperties& currentLevelProperties, const unsigned int numDispVals, float* inData)
{
  return _mm256_loadu_ps(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
    currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline __m128i VectProcessingFuncts::loadPackedDataUnaligned<short, __m128i>(
  const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
  const beliefprop::levelProperties& currentLevelProperties, const unsigned int numDispVals, short* inData)
{
  return _mm_loadu_si128((__m128i*)(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
    currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]));
}

template<> inline __m256d VectProcessingFuncts::loadPackedDataUnaligned<double, __m256d>(
  const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
  const beliefprop::levelProperties& currentLevelProperties, const unsigned int numDispVals, double* inData)
{
  return _mm256_loadu_pd(&inData[beliefprop::retrieveIndexInDataAndMessage(
    x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
    currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline __m256 VectProcessingFuncts::createSIMDVectorSameData<__m256>(const float data) {
  return _mm256_set1_ps(data);
}

template<> inline __m128i VectProcessingFuncts::createSIMDVectorSameData<__m128i>(const float data) {
  return _mm256_cvtps_ph(_mm256_set1_ps(data), 0);
}

template<> inline __m256d VectProcessingFuncts::createSIMDVectorSameData<__m256d>(const float data) {
  return _mm256_set1_pd((double)data);
}

template<> inline __m256 VectProcessingFuncts::addVals<__m256, __m256, __m256>(const __m256& val1, const __m256& val2) {
  return _mm256_add_ps(val1, val2);
}

template<> inline __m256d VectProcessingFuncts::addVals<__m256d, __m256d, __m256d>(const __m256d& val1, const __m256d& val2) {
  return _mm256_add_pd(val1, val2);
}

template<> inline __m256 VectProcessingFuncts::addVals<__m256, __m128i, __m256>(const __m256& val1, const __m128i& val2) {
  return _mm256_add_ps(val1, _mm256_cvtph_ps(val2));
}

template<> inline __m256 VectProcessingFuncts::addVals<__m128i, __m256, __m256>(const __m128i& val1, const __m256& val2) {
  return _mm256_add_ps(_mm256_cvtph_ps(val1), val2);
}

template<> inline __m256 VectProcessingFuncts::addVals<__m128i, __m128i, __m256>(const __m128i& val1, const __m128i& val2) {
  return _mm256_add_ps(_mm256_cvtph_ps(val1), _mm256_cvtph_ps(val2));
}

template<> inline __m256 VectProcessingFuncts::subtractVals<__m256, __m256, __m256>(const __m256& val1, const __m256& val2) {
  return _mm256_sub_ps(val1, val2);
}

template<> inline __m256d VectProcessingFuncts::subtractVals<__m256d, __m256d, __m256d>(const __m256d& val1, const __m256d& val2) {
  return _mm256_sub_pd(val1, val2);
}

template<> inline __m256 VectProcessingFuncts::divideVals<__m256, __m256, __m256>(const __m256& val1, const __m256& val2) {
  return _mm256_div_ps(val1, val2);
}

template<> inline __m256d VectProcessingFuncts::divideVals<__m256d, __m256d, __m256d>(const __m256d& val1, const __m256d& val2) {
  return _mm256_div_pd(val1, val2);
}

template<> inline __m256 VectProcessingFuncts::convertValToDatatype<__m256, float>(const float val) {
  return _mm256_set1_ps(val);
}

template<> inline __m256d VectProcessingFuncts::convertValToDatatype<__m256d, double>(const double val) {
  return _mm256_set1_pd(val);
}

template<> inline __m256 VectProcessingFuncts::getMinByElement<__m256>(const __m256& val1, const __m256& val2) {
  return _mm256_min_ps(val1, val2);
}

template<> inline __m256d VectProcessingFuncts::getMinByElement<__m256d>(const __m256d& val1, const __m256d& val2) {
  return _mm256_min_pd(val1, val2);
}

template<> inline void VectProcessingFuncts::storePackedDataAligned<float, __m256>(
  const unsigned int indexDataStore, float* locationDataStore, const __m256& dataToStore)
{
  _mm256_store_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::storePackedDataAligned<short, __m256>(
  const unsigned int indexDataStore, short* locationDataStore, const __m256& dataToStore)
{
  _mm_store_si128((__m128i*)(&locationDataStore[indexDataStore]), _mm256_cvtps_ph(dataToStore, 0));
}

template<> inline void VectProcessingFuncts::storePackedDataAligned<double, __m256d>(
  const unsigned int indexDataStore, double* locationDataStore, const __m256d& dataToStore)
{
  _mm256_store_pd(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::storePackedDataUnaligned<float, __m256>(
  const unsigned int indexDataStore, float* locationDataStore, const __m256& dataToStore)
{
  _mm256_storeu_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void VectProcessingFuncts::storePackedDataUnaligned<short, __m256>(
  const unsigned int indexDataStore, short* locationDataStore, const __m256& dataToStore)
{
  _mm_storeu_si128((__m128i*)(&locationDataStore[indexDataStore]), _mm256_cvtps_ph(dataToStore, 0));
}

template<> inline void VectProcessingFuncts::storePackedDataUnaligned<double, __m256d>(
  const unsigned int indexDataStore, double* locationDataStore, const __m256d& dataToStore)
{
  _mm256_storeu_pd(&locationDataStore[indexDataStore], dataToStore);
}

#endif /* AVX256TEMPLATESPFUNCTS_H_ */
