/*
 * SharedUtilFuncts.h
 *
 *  Created on: Jun 24, 2019
 *      Author: scott
 */

#ifndef SHAREDUTILFUNCTS_H_
#define SHAREDUTILFUNCTS_H_

#ifdef OPTIMIZED_CUDA_RUN
#include "../ParameterFiles/bpStereoCudaParameters.h"
//added in front of function header to indicate that function is device function to be processed on GPU
#define ARCHITECTURE_ADDITION __device__
//define concept of allowed data types for belief propagation kernel processing on GPU
template <typename T>
concept BpKernelData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, halftype>;
#else
#define ARCHITECTURE_ADDITION
//define concept of allowed data types for belief propagation kernel processing on CPU
#ifdef COMPILING_FOR_ARM
//float16_t is used for half data type in ARM processing
template <typename T>
concept BpKernelData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, float16_t>;
#else
//short is used for half data type in x86 processing
template <typename T>
concept BpKernelData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, short>;
#endif //COMPILING_FOR_ARM
#endif //OPTIMIZED_CUDA_RUN

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

#endif /* SHAREDUTILFUNCTS_H_ */
