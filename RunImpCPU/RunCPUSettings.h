/*
 * RunCPUSettings.h
 *
 *  Created on: Jan 27, 2024
 *      Author: scott
 */

#ifndef RUNCPUSETTINGS_H_
#define RUNCPUSETTINGS_H_

#include <vector>
#include <array>
#include <thread>
#include "RunSettingsEval/RunSettings.h"

//check if running on ARM architecture
#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
#endif

//define and set CPU vectorization options using preprocessor (since needed to determine what code gets compiled to support vectorization)
#define AVX_256_DEFINE 0
#define AVX_512_DEFINE 1
#define NEON_DEFINE 2
#define NO_VECTORIZATION 3
#ifdef COMPILING_FOR_ARM //NEON supported on ARM but AVX is not
#define CPU_VECTORIZATION_DEFINE NEON_DEFINE
#else
//by default CPU vectorization during compilation via Makefile
//use AVX 512 if not set during compilation
#if defined(AVX_512_VECTORIZATION)
#define CPU_VECTORIZATION_DEFINE AVX_512_DEFINE
#elif defined(AVX_256_VECTORIZATION)
#define CPU_VECTORIZATION_DEFINE AVX_256_DEFINE
#else
#define CPU_VECTORIZATION_DEFINE AVX_512_DEFINE
#endif //defined(AVX_512_VECTORIZATION)
#endif //COMPILING_FOR_ARM

namespace run_cpu {

//constant to specify to run simulate single CPU on a dual-CPU system
constexpr std::string_view SIMULATE_SINGLE_CPU{"SimulateSingleCPU"};

//set the CPU vectorization
#if (CPU_VECTORIZATION_DEFINE == NEON_DEFINE)
constexpr run_environment::AccSetting CPU_VECTORIZATION{run_environment::AccSetting::NEON};
#elif (CPU_VECTORIZATION_DEFINE == AVX_256_DEFINE)
constexpr run_environment::AccSetting CPU_VECTORIZATION{run_environment::AccSetting::AVX256};
#elif (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
constexpr run_environment::AccSetting CPU_VECTORIZATION{run_environment::AccSetting::AVX512};
#else
constexpr run_environment::AccSetting CPU_VECTORIZATION{run_environment::AccSetting::NONE};
#endif

//parallel parameter options to run to retrieve optimized parallel parameters in optimized CPU implementation
//parallel parameter corresponds to number of OpenMP threads in optimized CPU implementation
const unsigned int NUM_THREADS_CPU{std::thread::hardware_concurrency()};
#ifdef LIMITED_TEST_PARAMS_FEWER_RUNS
const std::vector<std::array<unsigned int, 2>> PARALLEL_PARAMETERS_OPTIONS{
  { NUM_THREADS_CPU, 1},{ NUM_THREADS_CPU / 2, 1}};
#else
const std::vector<std::array<unsigned int, 2>> PARALLEL_PARAMETERS_OPTIONS{
  { NUM_THREADS_CPU, 1}, { (3 * NUM_THREADS_CPU) / 4 , 1}, { NUM_THREADS_CPU / 2, 1},
  { NUM_THREADS_CPU / 4, 1}, { NUM_THREADS_CPU / 8, 1}};
#endif //LIMITED_TEST_PARAMS_FEWER_RUNS
const unsigned int MIN_NUM_THREADS_RUN{std::min(NUM_THREADS_CPU, 4u)};
const std::array<unsigned int, 2> PARALLEL_PARAMS_DEFAULT{{NUM_THREADS_CPU, 1}};

};

#endif /* RUNCPUSETTINGS_H_ */
