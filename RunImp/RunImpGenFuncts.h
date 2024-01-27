/*
 * RunImpGenFuncts.h
 *
 *  Created on: Jan 26, 2024
 *      Author: scott
 */

#ifndef RUNIMPGENFUNCTS_H_
#define RUNIMPGENFUNCTS_H_

#include "RunSettingsEval/RunTypeConstraints.h"

#ifdef OPTIMIZED_CUDA_RUN
//added in front of function header to indicate that function is device function to be processed on GPU
#define ARCHITECTURE_ADDITION __device__
#else
#define ARCHITECTURE_ADDITION
#endif //OPTIMIZED_CUDA_RUN

//T is input type, U is output type
template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline U convertValToDifferentDataTypeIfNeeded(const T data) {
  return data; //by default assume same data type and just return data
}

template<RunData_t T>
ARCHITECTURE_ADDITION inline T getZeroVal() {
  return (T)0.0;
}

template<typename T>
requires std::is_arithmetic_v<T>
ARCHITECTURE_ADDITION inline T getMin(const T val1, const T val2) {
  return ((val1 < val2) ? val1 : val2);
}

template<typename T>
requires std::is_arithmetic_v<T>
ARCHITECTURE_ADDITION inline T getMax(const T val1, const T val2) {
  return ((val1 > val2) ? val1 : val2);
}

//checks if the current point is within the image bounds
//assumed that input x/y vals are above zero since their unsigned int so no need for >= 0 check
ARCHITECTURE_ADDITION inline bool withinImageBounds(const unsigned int xVal, const unsigned int yVal, const unsigned int width, const unsigned int height) {
  return ((xVal < width) && (yVal < height));
}

#endif //RUNIMPGENFUNCTS_H_