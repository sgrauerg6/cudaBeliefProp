/*
 * bpRunSettings.h
 *
 *  Created on: Sep 21, 2019
 *      Author: scott
 */

#ifndef BPRUNSETTINGS_H_
#define BPRUNSETTINGS_H_

#include <iostream>
#include <typeinfo>
#include <typeindex>
#include <map>
#include "../OutputEvaluation/RunData.h"

//uncomment if compiling/running on ARM architecture
//#define COMPILING_FOR_ARM
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

#define DATA_TYPE_PROCESSING_FLOAT 0
#define DATA_TYPE_PROCESSING_DOUBLE 1
#define DATA_TYPE_PROCESSING_HALF 2
//not currently supporting half2 data type
#define DATA_TYPE_PROCESSING_HALF_TWO 3

//by default, 32-bit float data is used with optimized GPU memory management and optimized indexing
//See http://scottgg.net/OptimizingGlobalStereoMatchingOnNVIDIAGPUs.pdf for more info on these optimizations (note that the optimized indexing was present in the initial implementation)
//Can remove optimized GPU memory management (making the processing more similar to the initial work) by setting beliefprop::USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT to false
//May be able to speed up processing by switching to using 16-bit half data by setting CURRENT_DATA_TYPE_PROCESSING to DATA_TYPE_PROCESSING_HALF
//Optimized indexing can be turned off by changing the beliefprop::OPTIMIZED_INDEXING_SETTING value to false (not recommended; this slows down processing)
#define CURRENT_DATA_TYPE_PROCESSING DATA_TYPE_PROCESSING_FLOAT

//define and set CPU vectorization options using preprocessor (since needed to determine what code gets compiled to support vectorization)
#define AVX_256_DEFINE 0
#define AVX_512_DEFINE 1
#define NEON_DEFINE 2
#define NO_VECTORIZATION 3
#ifdef COMPILING_FOR_ARM //NEON supported on ARM but AVX is not
#define CPU_VECTORIZATION_DEFINE NEON_DEFINE
#else
#define CPU_VECTORIZATION_DEFINE AVX_512_DEFINE
#endif

namespace beliefprop {

//mapping from data size to data type string
const std::map<std::size_t, std::string> DATA_SIZE_TO_NAME_MAP{
	{sizeof(float), "FLOAT"}, {sizeof(double), "DOUBLE"}, {sizeof(short), "HALF"}
};

//enum for cpu vectorization setting
enum class CPUVectorization {
	NONE, AVX256, AVX512, NEON
};

#if (CPU_VECTORIZATION_DEFINE == NEON_DEFINE)
constexpr CPUVectorization CPU_VECTORIZATION{CPUVectorization::NEON};
#elif (CPU_VECTORIZATION_DEFINE == AVX_256_DEFINE)
constexpr CPUVectorization CPU_VECTORIZATION{CPUVectorization::AVX256};
#elif (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
constexpr CPUVectorization CPU_VECTORIZATION{CPUVectorization::AVX512};
#else
constexpr CPUVectorization CPU_VECTORIZATION{CPUVectorization::NONE};
#endif

constexpr bool OPTIMIZED_INDEXING_SETTING{true};
constexpr bool USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT{true};
constexpr bool ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS{true};

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

//get string corresponding to CPU vectorization method
constexpr const char* cpuVectorizationString() {
  #if (CPU_VECTORIZATION_DEFINE == NEON_DEFINE)
    return "NEON";
  #elif (CPU_VECTORIZATION_DEFINE == AVX_256_DEFINE)
    return "AVX_256";
  #elif (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
    return "AVX_512";
  #else //(CPU_VECTORIZATION_DEFINE == NO_VECTORIZATION)
    return "NO_VECTORIZATION";
  #endif //CPU_VECTORIZATION_DEFINE
}

constexpr unsigned int getBytesAlignMemory(const CPUVectorization inVectSetting) {
	//avx512 requires data to be aligned on 64 bytes
	return (inVectSetting == CPUVectorization::AVX512) ? 64 : 16;
}

constexpr unsigned int getNumDataAlignWidth(const CPUVectorization inVectSetting) {
	//align width with 16 data values in AVX512
	return (inVectSetting == CPUVectorization::AVX512) ? 16 : 8;
}

constexpr unsigned int BYTES_ALIGN_MEMORY = getBytesAlignMemory(CPU_VECTORIZATION);
constexpr unsigned int NUM_DATA_ALIGN_WIDTH = getNumDataAlignWidth(CPU_VECTORIZATION);

inline void writeRunSettingsToStream(std::ostream& resultsStream)
{
	resultsStream << "Memory Optimization Level: " << beliefprop::USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT << "\n";
	resultsStream << "Indexing Optimization Level: " << beliefprop::OPTIMIZED_INDEXING_SETTING << "\n";
	resultsStream << "BYTES_ALIGN_MEMORY: " << beliefprop::BYTES_ALIGN_MEMORY << "\n";
	resultsStream << "NUM_DATA_ALIGN_WIDTH: " << beliefprop::NUM_DATA_ALIGN_WIDTH << "\n";
}

inline RunData runSettings()  {
    RunData currRunData;
		currRunData.addDataWHeader("Memory Optimization Level", std::to_string(beliefprop::USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT));
		currRunData.addDataWHeader("Indexing Optimization Level", std::to_string(beliefprop::OPTIMIZED_INDEXING_SETTING));
		currRunData.addDataWHeader("BYTES_ALIGN_MEMORY", std::to_string(beliefprop::BYTES_ALIGN_MEMORY));
		currRunData.addDataWHeader("NUM_DATA_ALIGN_WIDTH", std::to_string(beliefprop::NUM_DATA_ALIGN_WIDTH));

    return currRunData;
}

};

#endif /* BPRUNSETTINGS_H_ */
