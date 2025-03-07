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
 * @file RunCPUSettings.h
 * @author Scott Grauer-Gray
 * @brief Contains namespace with CPU run defaults and constants.
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RUNCPUSETTINGS_H_
#define RUNCPUSETTINGS_H_

#include <vector>
#include <array>
#include <thread>
#include <set>

//check if running on ARM architecture
#if defined(COMPILING_FOR_ARM)
#include <arm_neon.h> //needed for float16_t type
#endif

//define and set CPU vectorization options using preprocessor
//needed to determine what code gets compiled to support vectorization
#define AVX_256_DEFINE 0
#define AVX_256_F16_DEFINE 1
#define AVX_512_DEFINE 2
#define AVX_512_F16_DEFINE 3
#define NEON_DEFINE 4
#define NO_VECTORIZATION 5
#if defined(COMPILING_FOR_ARM) //NEON supported on ARM but AVX is not
#define CPU_VECTORIZATION_DEFINE NEON_DEFINE
#else
//by default CPU vectorization during compilation via Makefile
//use AVX 512 if not set during compilation
#if defined(AVX_512_VECTORIZATION)
#define CPU_VECTORIZATION_DEFINE AVX_512_DEFINE
#elif defined(AVX_512_F16_VECTORIZATION)
#define CPU_VECTORIZATION_DEFINE AVX_512_F16_DEFINE
#elif defined(AVX_256_VECTORIZATION)
#define CPU_VECTORIZATION_DEFINE AVX_256_DEFINE
#elif defined(AVX_256_F16_VECTORIZATION)
#define CPU_VECTORIZATION_DEFINE AVX_256_F16_DEFINE
#else
#define CPU_VECTORIZATION_DEFINE AVX_512_DEFINE
#endif //defined(AVX_512_VECTORIZATION)
#endif //COMPILING_FOR_ARM

/**
 * @brief Namespace with CPU run defaults and constants.
 */
namespace run_cpu {

/** @brief Constant that specifies that run is simulating single CPU on a
 *  dual-CPU system */
constexpr std::string_view kSimulateSingleCPU{"SimulateSingleCPU"};

/** @brief Constant corresponding to number of threads on CPU. */
const unsigned int kNumThreadsCPU{std::thread::hardware_concurrency()};

/** @brief Minimum number of threads to allow for any parallel parameters
 *  setting on CPU. */
const unsigned int kMinNumThreadsRun{std::min(kNumThreadsCPU, 4u)};

/** @brief Default parallel parameters setting on CPU. */
const std::array<unsigned int, 2> kParallelParamsDefault{
  {kNumThreadsCPU, 1}};

#if defined(DEFAULT_PARALLEL_PARAMS_ONLY)

/** @brief Empty parallel parameter alternative options since setting is to
  *  only use default parallel parameters. */
const std::set<std::array<unsigned int, 2>> kParallelParameterAltOptions{};

#elif defined(LIMITED_ALT_PARALLEL_PARAMS)

/** @brief Parallel parameters options that are tested in order to find optimized
 *  configuration in run. OK to include default option but not required.*/
const std::set<std::array<unsigned int, 2>> kParallelParameterAltOptions{
  { kNumThreadsCPU, 1},
  { kNumThreadsCPU / 2, 1}};

#else

/** @brief Parallel parameters options that are tested in order to find
 *  optimized configuration in run. OK to include default option but not
 *  required.*/
const std::set<std::array<unsigned int, 2>> kParallelParameterAltOptions{
  { kNumThreadsCPU, 1}, 
  { (3 * kNumThreadsCPU) / 4 , 1}, 
  { kNumThreadsCPU / 2, 1},
  { kNumThreadsCPU / 4, 1},
  { kNumThreadsCPU / 8, 1}};

#endif

};

#endif /* RUNCPUSETTINGS_H_ */
