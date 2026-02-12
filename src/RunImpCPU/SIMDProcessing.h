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
 * @file SIMDProcessing.h
 * @author Scott Grauer-Gray
 * @brief Contains general functions for processing using SIMD vector data
 * types on CPU.
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef SIMD_PROCESSING_H_
#define SIMD_PROCESSING_H_

#ifndef __APPLE__
#include <omp.h>
#endif //__APPLE__

#include <math.h>
#include <algorithm>
#include <iostream>
#include "RunEval/RunTypeConstraints.h"
#include "RunImpCPU/RunCPUSettings.h"

/**
 * @brief General functions for processing using SIMD vector data types on CPU.<br>
 * Template specializations must be defined to support specific SIMD vector data types.
 */
namespace simd_processing
{
  /**
   * @brief Load multiple values of primitive type data from inData array to
   * SIMD vector data type<br>
   * inData array must be aligned according to the rules of the specified SIMD type to
   * use this operation
   * 
   * @tparam T 
   * @tparam U 
   * @param x 
   * @param y 
   * @param current_disparity 
   * @param current_bp_level 
   * @param numDispVals 
   * @param inData 
   * @return SIMD structure with data loaded
   */
  template<RunData_t T, RunDataVect_t U>
  U LoadPackedDataAligned(unsigned int x, unsigned int y, unsigned int current_disparity,
    const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals,
    const T* inData)
  {
    std::cout << "Data type not supported for loading aligned data" << std::endl;
  }

  /**
   * @brief Load multiple values of primitive type data from inData array to
   * SIMD vector data type<br>
   * inData array does not need to be aligned to use this operation
   * 
   * @tparam T 
   * @tparam U 
   * @param x 
   * @param y 
   * @param current_disparity 
   * @param current_bp_level 
   * @param numDispVals 
   * @param inData 
   * @return SIMD structure with data loaded
   */
  template<RunData_t T, RunDataVect_t U>
  U LoadPackedDataUnaligned(unsigned int x, unsigned int y, unsigned int current_disparity,
    const beliefprop::BpLevelProperties& current_bp_level, unsigned int numDispVals,
    const T* inData)
  {
    std::cout << "Data type not supported for loading unaligned data" << std::endl;
  }

  /**
   * @brief Create a SIMD vector of the specified type with all elements containing
   * the same data
   * 
   * @tparam T 
   * @param data 
   * @return SIMD structure with all elements containing the same data
   */
  template<RunDataVect_t T>
  T createSIMDVectorSameData(float data) {
    std::cout << "Data type not supported for creating simd vector" << std::endl;
  }

  //
  /**
   * @brief Add values of specified types and return sum as specified type<br>
   * Define template specialization to support addition of specific SIMD vector types
   * 
   * @tparam T 
   * @tparam U 
   * @tparam V 
   * @param val1 
   * @param val2 
   * @return SIMD structure with sum of values
   */
  template<RunDataSingOrVect_t T, RunDataSingOrVect_t U, RunDataSingOrVect_t V>
  V AddVals(const T& val1, const U& val2) { return (val1 + val2); }

  /**
   * @brief Subtract values of specified types and return difference as specified type
   * Define template specialization to support subtraction of specific SIMD vector types
   * 
   * @tparam T 
   * @tparam U 
   * @tparam V 
   * @param val1 
   * @param val2 
   * @return SIMD structure with difference of values
   */
  template<RunDataSingOrVect_t T, RunDataSingOrVect_t U, RunDataSingOrVect_t V>
  V SubtractVals(const T& val1, const U& val2) { return (val1 - val2); }

  /**
   * @brief Divide values of specified types and return quotient as specified type<br>
   * Define template specialization to support division of specific SIMD vector types
   * 
   * @tparam T 
   * @tparam U 
   * @tparam V 
   * @param val1 
   * @param val2 
   * @return SIMD structure with quotient of values
   */
  template<RunDataSingOrVect_t T, RunDataSingOrVect_t U, RunDataSingOrVect_t V>
  V divideVals(const T& val1, const U& val2) { return (val1 / val2); }

  /**
   * @brief Convert value of specified type to value of another specified type<br>
   * Define template specialization to support conversion between specific types
   * for SIMD vector processing
   * 
   * @tparam T 
   * @tparam V 
   * @param val 
   * @return Value converted to specified type
   */
  template<RunDataSingOrVect_t T, RunDataSingOrVect_t V>
  T ConvertValToDatatype(V val) { return (T)val; }

  /**
   * @brief Get element-wise minimum of two inputs which may be of a SIMD
   * vector type where corresponding values within the SIMD vector are compared<br>
   * Define template specialization to support function for specific vector type
   * 
   * @tparam T 
   * @param val1 
   * @param val2 
   * @return Element-wise minimum of two inputs (may be SIMD structures)
   */
  template<RunDataSingOrVect_t T>
  T GetMinByElement(const T& val1, const T& val2) { return std::min(val1, val2); }

  /**
   * @brief Write data in SIMD vector (or single element) to specified location
   * in array<br>
   * Array that data is written to must be aligned according to the rules of
   * the specified SIMD type to use this operation
   * 
   * @tparam T 
   * @tparam U 
   * @param indexDataStore 
   * @param locationDataStore 
   * @param dataToStore 
   */
  template<RunData_t T, RunDataVectProcess_t U>
  void StorePackedDataAligned(size_t indexDataStore, T* locationDataStore, const U& dataToStore) {
    locationDataStore[indexDataStore] = dataToStore;
  }

  /**
   * @brief Write data in SIMD vector (or single element) to specified
   * location in array<br>
   * Array that data is written does not need to be aligned to use this
   * operation
   * 
   * @tparam T 
   * @tparam U 
   * @param indexDataStore 
   * @param locationDataStore 
   * @param dataToStore 
   */
  template<RunData_t T, RunDataVectProcess_t U>
  void StorePackedDataUnaligned(size_t indexDataStore, T* locationDataStore, const U& dataToStore) {
    locationDataStore[indexDataStore] = dataToStore;
  }
};

//headers to include differ depending on architecture and CPU vectorization setting
#if defined(COMPILING_FOR_ARM)
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
