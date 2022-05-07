/*
 * bpRunSettings.h
 *
 *  Created on: Sep 21, 2019
 *      Author: scott
 */

#ifndef BPRUNSETTINGS_H_
#define BPRUNSETTINGS_H_

#include "bpStereoParameters.h"
#include "bpParametersFromPython.h"
#include <fstream>
#include <typeinfo>
#include <typeindex>
#include <map>

//uncomment if compiling/running on ARM architecture
#define COMPILING_FOR_ARM
#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
#endif

//comment out corresponding define if half or double precision
//not supported (or to not to process half or double precision) 
#define HALF_PRECISION_SUPPORTED
#define DOUBLE_PRECISION_SUPPORTED

//uncomment to use C++ thread pool rather than OpenMP
#define USE_THREAD_POOL_CHUNKS 1
#define USE_THREAD_POOL_DISTRIBUTED 2
#define USE_OPENMP 3

//define cpu parallelization method
//OpenMP set by default since it is generally faster than
//current thread pool options
#define CPU_PARALLELIZATION_METHOD USE_OPENMP

//get string corresponding to CPU parallelization method
constexpr const char* cpuParallelizationString() {
  #if (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS)
    return "THREAD_POOL_CHUNKS";
  #elif (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED)
    return "THREAD_POOL_DISTRIBUTED";
  #else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
    return "OPEN_MP";
  #endif //CPU_PARALLELIZATION_METHOD
}

//mapping from data size to data type string
const std::map<std::size_t, std::string> DATA_SIZE_TO_NAME_MAP{
	{sizeof(float), "FLOAT"}, {sizeof(double), "DOUBLE"}, {sizeof(short), "HALF"}
};

#define DATA_TYPE_PROCESSING_FLOAT 0
#define DATA_TYPE_PROCESSING_DOUBLE 1
#define DATA_TYPE_PROCESSING_HALF 2
//not currently supporting half2 data type
#define DATA_TYPE_PROCESSING_HALF_TWO 3

//enum for cpu vectorization setting
enum class cpu_vectorization_setting {
	NO_CPU_VECTORIZATION, USE_AVX_256, USE_AVX_512, USE_NEON
};

//If image set parameters from python, then use optimization settings set in current iteration in python script
//These settings are written to file bpParametersFromPython.h as part of the python script

//by default, 32-bit float data is used with optimized GPU memory management and optimized indexing
//See http://scottgg.net/OptimizingGlobalStereoMatchingOnNVIDIAGPUs.pdf for more info on these optimizations (note that the optimized indexing was present in the initial implementation)
//Can remove optimized GPU memory management (making the processing more similar to the initial work) by setting USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT to false
//May be able to speed up processing by switching to using 16-bit half data by setting CURRENT_DATA_TYPE_PROCESSING to DATA_TYPE_PROCESSING_HALF
//Optimized indexing can be turned off by changing the OPTIMIZED_INDEXING_SETTING value to false (not recommended; this slows down processing)
#define CURRENT_DATA_TYPE_PROCESSING DATA_TYPE_PROCESSING_FLOAT
constexpr bool OPTIMIZED_INDEXING_SETTING{true};
constexpr bool USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT{true};
constexpr bool ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS{true};

//define and set CPU vectorization options using preprocessor (since needed to determine what code gets compiled to support vectorization)
#define AVX_256 0
#define AVX_512 1
#define NEON 2
#define NO_VECTORIZATION 3
#ifdef COMPILING_FOR_ARM //NEON supported on ARM but AVX is not
#define CPU_VECTORIZATION_SETTING NEON
#else
#define CPU_VECTORIZATION_SETTING AVX_512
#endif

#if (CPU_VECTORIZATION_SETTING == NEON)
constexpr cpu_vectorization_setting CPU_OPTIMIZATION_SETTING{cpu_vectorization_setting::USE_NEON};
#elif (CPU_VECTORIZATION_SETTING == AVX_256)
constexpr cpu_vectorization_setting CPU_OPTIMIZATION_SETTING{cpu_vectorization_setting::USE_AVX_256};
#elif (CPU_VECTORIZATION_SETTING == AVX_512)
constexpr cpu_vectorization_setting CPU_OPTIMIZATION_SETTING{cpu_vectorization_setting::USE_AVX_512};
#else
constexpr cpu_vectorization_setting CPU_OPTIMIZATION_SETTING{cpu_vectorization_setting::NO_CPU_VECTORIZATION};
#endif

//get string corresponding to CPU vectorization method
constexpr const char* cpuVectorizationString() {
  #if (CPU_VECTORIZATION_SETTING == NEON)
    return "NEON";
  #elif (CPU_VECTORIZATION_SETTING == AVX_256)
    return "AVX_256";
  #elif (CPU_VECTORIZATION_SETTING == AVX_512)
    return "AVX_512";
  #else //(CPU_VECTORIZATION_SETTING == NO_VECTORIZATION)
    return "NO_VECTORIZATION";
  #endif //CPU_VECTORIZATION_SETTING
}

constexpr unsigned int getBytesAlignMemory(cpu_vectorization_setting inVectSetting) {
	//avx512 requires data to be aligned on 64 bytes
	return (inVectSetting == cpu_vectorization_setting::USE_AVX_512) ? 64 : 16;
}

constexpr unsigned int getNumDataAlignWidth(cpu_vectorization_setting inVectSetting) {
	//align width with 16 data values in AVX512
	return (inVectSetting == cpu_vectorization_setting::USE_AVX_512) ? 16 : 8;
}

namespace bp_params
{
	constexpr unsigned int BYTES_ALIGN_MEMORY = getBytesAlignMemory(CPU_OPTIMIZATION_SETTING);
	constexpr unsigned int NUM_DATA_ALIGN_WIDTH = getNumDataAlignWidth(CPU_OPTIMIZATION_SETTING);
}

#endif /* BPRUNSETTINGS_H_ */
