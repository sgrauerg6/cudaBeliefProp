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

#define DATA_TYPE_PROCESSING_FLOAT 0
#define DATA_TYPE_PROCESSING_DOUBLE 1
#define DATA_TYPE_PROCESSING_HALF 2
//not currently supporting half2 data type
#define DATA_TYPE_PROCESSING_HALF_TWO 3

enum class cpu_vectorization_setting {
	NO_CPU_VECTORIZATION_CODE, USE_AVX_256, USE_AVX_512, USE_NEON
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
constexpr cpu_vectorization_setting CPU_OPTIMIZATION_SETTING{cpu_vectorization_setting::USE_AVX_256};

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

//#define COMPILING_FOR_ARM
#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
#endif

//remove (or don't use) capability for half precision if using GPU with compute capability under 5.3
#if CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT
typedef float beliefPropProcessingDataType;
const std::string BELIEF_PROP_PROCESSING_DATA_TYPE_STRING = "FLOAT";
#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE
typedef double beliefPropProcessingDataType;
const std::string BELIEF_PROP_PROCESSING_DATA_TYPE_STRING = "DOUBLE";
#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF
#ifdef COMPILING_FOR_ARM
typedef float16_t beliefPropProcessingDataType;
#else
typedef short beliefPropProcessingDataType;
#endif
const std::string BELIEF_PROP_PROCESSING_DATA_TYPE_STRING = "HALF";
//not currently supporting half2 data type
/*#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO
typedef short beliefPropProcessingDataType;*/
//const std::string BELIEF_PROP_PROCESSING_DATA_TYPE_STRING = "HALF2";
#endif

#endif /* BPRUNSETTINGS_H_ */
