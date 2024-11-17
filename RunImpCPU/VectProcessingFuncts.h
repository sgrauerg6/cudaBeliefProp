/*
 * VectProcessingFuncts.h
 *
 *  Created on: Jun 26, 2024
 *      Author: scott
 */

#ifndef VECT_PROCESSING_FUNCTS_H_
#define VECT_PROCESSING_FUNCTS_H_

#include <math.h>
#include <omp.h>
#include <algorithm>
#include <iostream>
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunImpCPU/RunCPUSettings.h"

//functions for processing using vectorization on CPU
namespace VectProcessingFuncts
{
  template<RunData_t T, RunDataVect_t U>
  U loadPackedDataAligned(unsigned int x, unsigned int y, unsigned int currentDisparity,
    const beliefprop::levelProperties& currentLevelProperties, unsigned int numDispVals, T* inData)
  {
    std::cout << "Data type not supported for loading aligned data" << std::endl;
  }

  template<RunData_t T, RunDataVect_t U>
  U loadPackedDataUnaligned(unsigned int x, unsigned int y, unsigned int currentDisparity,
    const beliefprop::levelProperties& currentLevelProperties, unsigned int numDispVals, T* inData)
  {
    std::cout << "Data type not supported for loading unaligned data" << std::endl;
  }

  template<RunDataVect_t T>
  T createSIMDVectorSameData(float data) {
    std::cout << "Data type not supported for creating simd vector" << std::endl;
  }

  template<RunDataSingOrVect_t T, RunDataSingOrVect_t U, RunDataSingOrVect_t V>
  V addVals(const T& val1, const U& val2) { return (val1 + val2); }

  template<RunDataSingOrVect_t T, RunDataSingOrVect_t U, RunDataSingOrVect_t V>
  V subtractVals(const T& val1, const U& val2) { return (val1 - val2); }

  template<RunDataSingOrVect_t T, RunDataSingOrVect_t U, RunDataSingOrVect_t V>
  V divideVals(const T& val1, const U& val2) { return (val1 / val2); }

  template<RunDataSingOrVect_t T, RunDataSingOrVect_t V>
  T convertValToDatatype(V val) { return (T)val; }

  template<RunDataSingOrVect_t T>
  T getMinByElement(const T& val1, const T& val2) { return std::min(val1, val2); }

  template<RunData_t T, RunDataVectProcess_t U>
  void storePackedDataAligned(unsigned int indexDataStore, T* locationDataStore, const U& dataToStore) {
    locationDataStore[indexDataStore] = dataToStore;
  }

  template<RunData_t T, RunDataVectProcess_t U>
  void storePackedDataUnaligned(unsigned int indexDataStore, T* locationDataStore, const U& dataToStore) {
    locationDataStore[indexDataStore] = dataToStore;
  }
};

//headers to include differ depending on architecture and CPU vectorization setting
#ifdef COMPILING_FOR_ARM
#include "ARMTemplateSpFuncts.h"

#if (CPU_VECTORIZATION_DEFINE == NEON_DEFINE)
#include "NEONTemplateSpFuncts.h"
#endif //CPU_VECTORIZATION_DEFINE == NEON_DEFINE

#else
//needed so that template specializations are used when available
#include "AVXTemplateSpFuncts.h"

#if (CPU_VECTORIZATION_DEFINE == AVX_256_DEFINE)
#include "AVX256TemplateSpFuncts.h"
#elif (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
#include "AVX256TemplateSpFuncts.h"
#include "AVX512TemplateSpFuncts.h"
#endif

#endif //COMPILING_FOR_ARM

#endif //VECT_PROCESSING_FUNCTS_H_
