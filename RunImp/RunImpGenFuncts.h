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

namespace run_imp_util {

//T is input type, U is output type
template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline U ConvertValToDifferentDataTypeIfNeeded(T data) {
  return data; //by default assume same data type and just return data
}

template<RunData_t T>
ARCHITECTURE_ADDITION inline T ZeroVal() {
  return (T)0.0;
}

template<typename T>
requires std::is_arithmetic_v<T>
ARCHITECTURE_ADDITION inline T GetMin(T val1, T val2) {
  return ((val1 < val2) ? val1 : val2);
}

template<typename T>
requires std::is_arithmetic_v<T>
ARCHITECTURE_ADDITION inline T GetMax(T val1, T val2) {
  return ((val1 > val2) ? val1 : val2);
}

//checks if the current point is within the image bounds
//assumed that input x/y vals are above zero since their unsigned int so no need for >= 0 check
ARCHITECTURE_ADDITION inline bool WithinImageBounds(
  unsigned int x_val, unsigned int y_val,
  unsigned int width, unsigned int height)
{
  return ((x_val < width) && (y_val < height));
}

//inline function to check if data is aligned at x_val_data_start for SIMD loads/stores that require alignment
inline bool MemoryAlignedAtDataStart(
  unsigned int x_val_data_start,
  unsigned int num_data_SIMD_vect,
  unsigned int num_data_align_width,
  unsigned int divPaddedChBoardWidthForAlign)
{
  //assuming that the padded checkerboard width divides evenly by beliefprop::NUM_DATA_ALIGN_WIDTH (if that's not the case it's a bug)
  return (((x_val_data_start % num_data_SIMD_vect) == 0) && ((num_data_align_width % divPaddedChBoardWidthForAlign) == 0));
}

};

#endif //RUNIMPGENFUNCTS_H_