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

//general functions for processing using SIMD vector data types on CPU
//template specializations must be defined to support specific SIMD vector data types
namespace VectProcessingFuncts
{
  //load multiple values of primitive type data from inData array to SIMD vector data type
  //inData array must be aligned according to the rules of the specified SIMD type to
  //use this operation
  template<RunData_t T, RunDataVect_t U>
  U LoadPackedDataAligned(unsigned int x, unsigned int y, unsigned int current_disparity,
    const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, T* inData)
  {
    std::cout << "Data type not supported for loading aligned data" << std::endl;
  }

  //load multiple values of primitive type data from inData array to SIMD vector data type
  //inData array does not need to be aligned to use this operation
  template<RunData_t T, RunDataVect_t U>
  U LoadPackedDataUnaligned(unsigned int x, unsigned int y, unsigned int current_disparity,
    const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals, T* inData)
  {
    std::cout << "Data type not supported for loading unaligned data" << std::endl;
  }

  //create a SIMD vector of the specified type with all elements containing the same data
  template<RunDataVect_t T>
  T createSIMDVectorSameData(float data) {
    std::cout << "Data type not supported for creating simd vector" << std::endl;
  }

  //add values of specified types and return sum as specified type
  //define template specialization to support addition of specific SIMD vector types
  template<RunDataSingOrVect_t T, RunDataSingOrVect_t U, RunDataSingOrVect_t V>
  V AddVals(const T& val1, const U& val2) { return (val1 + val2); }

  //subtract values of specified types and return difference as specified type
  //define template specialization to support subtraction of specific SIMD vector types
  template<RunDataSingOrVect_t T, RunDataSingOrVect_t U, RunDataSingOrVect_t V>
  V SubtractVals(const T& val1, const U& val2) { return (val1 - val2); }

  //divide values of specified types and return quotient as specified type
  //define template specialization to support division of specific SIMD vector types
  template<RunDataSingOrVect_t T, RunDataSingOrVect_t U, RunDataSingOrVect_t V>
  V divideVals(const T& val1, const U& val2) { return (val1 / val2); }

  //convert value of specified type to value of another specified type
  //define template specialization to support conversion between specific types
  //for SIMD vector processing
  template<RunDataSingOrVect_t T, RunDataSingOrVect_t V>
  T ConvertValToDatatype(V val) { return (T)val; }

  //get element-wise minimum of two inputs which may be of a SIMD vector type where corresponding
  //values within the SIMD vector are compared
  //define template specialization to support function for specific vector type
  template<RunDataSingOrVect_t T>
  T GetMinByElement(const T& val1, const T& val2) { return std::min(val1, val2); }

  //write data in SIMD vector (or single element) to specified location in array
  //array that data is written to must be aligned according to the rules of the specified SIMD type to
  //use this operation
  template<RunData_t T, RunDataVectProcess_t U>
  void StorePackedDataAligned(unsigned int indexDataStore, T* locationDataStore, const U& dataToStore) {
    locationDataStore[indexDataStore] = dataToStore;
  }

  //write data in SIMD vector (or single element) to specified location in array
  //array that data is written does not need to be aligned to use this operation
  template<RunData_t T, RunDataVectProcess_t U>
  void StorePackedDataUnaligned(unsigned int indexDataStore, T* locationDataStore, const U& dataToStore) {
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
