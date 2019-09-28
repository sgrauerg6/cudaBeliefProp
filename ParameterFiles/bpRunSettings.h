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

#define DATA_TYPE_PROCESSING_FLOAT 0
#define DATA_TYPE_PROCESSING_DOUBLE 1
#define DATA_TYPE_PROCESSING_HALF 2
//not currently supporting half2 data type
#define DATA_TYPE_PROCESSING_HALF_TWO 3

#define USE_DEFAULT 0
#define USE_AVX_256 1
#define USE_AVX_512 2
#define USE_NEON 3

//If image set parameters from python, then use optimization settings set in current iteration in python script
//These settings are written to file bpParametersFromPython.h as part of the python script

//by default, 32-bit float data is used with optimized GPU memory management and optimized indexing
//See http://scottgg.net/OptimizingGlobalStereoMatchingOnNVIDIAGPUs.pdf for more info on these optimizations (note that the optimized indexing was present in the initial implementation)
//Can remove optimized GPU memory management (making the processing more similar to the initial work) by commenting out the "#define USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT" line
//May be able to speed up processing by switching to using 16-bit half data by setting CURRENT_DATA_TYPE_PROCESSING to DATA_TYPE_PROCESSING_HALF
//Optimized indexing can be turned off by changing the OPTIMIZED_INDEXING_SETTING value to 0 (not recommended; this slows down processing)
#define CURRENT_DATA_TYPE_PROCESSING DATA_TYPE_PROCESSING_FLOAT
#define OPTIMIZED_INDEXING_SETTING 1
#define USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT
#define CPU_OPTIMIZATION_SETTING USE_AVX_256

//#define COMPILING_FOR_ARM
#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
#endif

//remove (or don't use) capability for half precision if using GPU with compute capability under 5.3
#if CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT
typedef float beliefPropProcessingDataType;
#define BELIEF_PROP_PROCESSING_DATA_TYPE_STRING "FLOAT"
#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE
typedef double beliefPropProcessingDataType;
#define BELIEF_PROP_PROCESSING_DATA_TYPE_STRING "DOUBLE"
#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF
#ifdef COMPILING_FOR_ARM
typedef float16_t beliefPropProcessingDataType;
#else
typedef short beliefPropProcessingDataType;
#endif
#define BELIEF_PROP_PROCESSING_DATA_TYPE_STRING "HALF"
//not currently supporting half2 data type
/*#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO
typedef short beliefPropProcessingDataType;*/
//#define BELIEF_PROP_PROCESSING_DATA_TYPE_STRING "HALF2"
#endif

#if CPU_OPTIMIZATION_SETTING == USE_AVX_512
//avx512 requires data to be aligned on 64 bytes
#define BYTES_ALIGN_MEMORY 64
#define NUM_DATA_ALIGN_WIDTH 16
#else
#define BYTES_ALIGN_MEMORY 32
#define NUM_DATA_ALIGN_WIDTH 8
#endif //CPU_OPTIMIZATION_SETTING == USE_AVX_512

#endif /* BPRUNSETTINGS_H_ */
