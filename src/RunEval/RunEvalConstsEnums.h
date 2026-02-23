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
 * @file RunEvalConstsEnums.h
 * @author Scott Grauer-Gray
 * @brief Contains namespace with enums and constants for implementation run
 * evaluation
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RUN_EVAL_CONSTS_ENUMS_H_
#define RUN_EVAL_CONSTS_ENUMS_H_

#include <array>
#include <string_view>
#include <map>
#include <filesystem>
#include "RunSettingsParams/RunSettingsConstsEnums.h"

/** 
 * @brief Namespace with enums and constants for implementation run evaluation
 */
namespace run_eval {

/** @brief Enum for status to indicate if error or no error */
enum class Status { kNoError, kError };

/** @brief Enum to specify average or median for "middle" value in data */
enum class MiddleValData { kAverage, kMedian };

//set data types to use in evaluation
//by default evaluate using float, double, and half data types
//can set to only evaluate specific datatype via preprocessor define
#if defined(EVAL_FLOAT_DATATYPE_ONLY)
  constexpr std::array<size_t, 1> kDataTypesEvalSizes{sizeof(float)};
#elif defined(EVAL_DOUBLE_DATATYPE_ONLY)
  constexpr std::array<size_t, 1> kDataTypesEvalSizes{sizeof(double)};
#elif defined(EVAL_HALF_DATATYPE_ONLY)
  constexpr std::array<size_t, 1> kDataTypesEvalSizes{sizeof(halftype)};
#else
  constexpr std::array<size_t, 3> kDataTypesEvalSizes{
    sizeof(float), sizeof(double), sizeof(halftype)};
#endif //EVAL_FLOAT_DATATYPE_ONLY

//set whether or not to run and evaluate alternative optimized implementations
//other than the expected fastest implementation available based on
//acceleration type
#if defined(NO_ALT_OPTIMIZED_IMPS)
  constexpr bool kRunAltOptimizedImps{false};
#else
  constexpr bool kRunAltOptimizedImps{true};
#endif //NO_ALT_OPTIMIZED_IMPS

//set templated iterations setting to use in evaluation
#if defined(EVAL_NOT_TEMPLATED_ITERS_ONLY)
  constexpr run_environment::TemplatedItersSetting kTemplatedItersEvalSettings =
    run_environment::TemplatedItersSetting::kRunOnlyNonTemplated;
#else
  constexpr run_environment::TemplatedItersSetting kTemplatedItersEvalSettings =
    run_environment::TemplatedItersSetting::kRunTemplatedAndNotTemplated;
#endif //EVAL_NOT_TEMPLATED_ITERS_ONLY

  /** @brief Define string for display of "true" and "false" values of bool value */
  constexpr std::array<std::string_view, 2> kBoolValFalseTrueDispStr{"NO", "YES"};

  //constants for headers corresponding to input, input settings, results, and run success
  constexpr std::string_view kInputIdxHeader{"Input Index"};
  constexpr std::string_view kDatatypeHeader{"Data Type"};
  constexpr std::string_view kLoopItersTemplatedHeader{"Loop Iters Templated"};

  /** @brief Constant to describing timing as median across evaluation runs */
  constexpr std::string_view kMedianOfTestRunsDesc{"(median timing across evaluation runs)"};

  /** @brief Constant for "all runs" string */
  constexpr std::string_view kAllRunsStr{"All Runs"};

  /** @brief Header for number of evaluation runs (can differ across inputs) */
  constexpr std::string_view kNumEvalRuns{"Number of evaluation runs"};

  //constants for output results for individual and sets of runs
  constexpr std::string_view kRunResultsDescFileName{"RunResults"};
  constexpr std::string_view kRunResultsDefaultPParamsDescFileName{"RunResultsDefaultPParams"};
  constexpr std::string_view kRunResultsWSpeedupsDescFileName{"RunResultsWSpeedups"};
  constexpr std::string_view kRunResultsDescDefaultPParamsFileName{"ResultsDefaultParallelParams"};
  constexpr std::string_view kSpeedupsDescFileName{"Speedups"};
  constexpr std::string_view kEvalAcrossRunsFileName{"EvaluationAcrossRuns"};
  constexpr std::string_view kCsvFileExtension{".csv"};
  constexpr std::string_view kOptimizedRuntimeHeader{"Median Optimized Runtime"};
  //TODO: may be able to delete previous runtime header if previous results adjusted to use new
  //header or older results no longer need to be supported
  constexpr std::string_view kOptimizedRuntimeHeader_Prev{"Median Optimized Runtime (including transfer time)"};
  constexpr std::string_view kSingleThreadRuntimeHeader{"Single-Thread CPU run time"};

  //constant for speedup description for optimized parallel parameters
  constexpr std::string_view kSpeedupOptParParamsHeader{"Speedup Over Default OMP Thread Count / CUDA Thread Block Dimensions"};

  //constants for speedups descriptions for alternate data types compared to flot
  constexpr std::string_view kSpeedupDoubleHeader{"Speedup using double-precision relative to float (actually slowdown)"};
  constexpr std::string_view kSpeedupHalfHeader{"Speedup using half-precision relative to float"};

  //constant for speedup description for templated loop iteration count where number
  //of loop iterations in at least some loops in benchmarks is known at compile time
  //via template parameter
  constexpr std::string_view kSpeedupLoopItersCountTemplate{"Speedup w/ templated loop iteration count (known at compile-time)"};

  //constants for speedup descriptions compared to alternate accelerations
  constexpr std::string_view kSpeedupCPUVectorization{"Speedup over no CPU vectorization"};
  constexpr std::string_view kSpeedupVsAvx256Vectorization{"Speedup over AVX256 CPU vectorization"};
  constexpr std::string_view kSpeedupVsAvx256F16Vectorization{"Speedup over AVX256 (w/ float16) CPU vectorization"};
  constexpr std::string_view kSpeedupVsAvx512Vectorization{"Speedup over AVX512 CPU vectorization"};
  constexpr std::string_view kSpeedupVsAvx512F16Vectorization{"Speedup over AVX512 (w/ float16) CPU vectorization"};
  constexpr std::string_view kSpeedupVsNEONVectorization{"Speedup over NEON CPU vectorization"};
  constexpr std::string_view kSpeedupVsCUDAAcceleration{"Speedup over CUDA acceleration"};
  
  //mapping of alternate acceleration setting to speedup description
  const std::map<run_environment::AccSetting, const std::string_view> kAltAccToSpeedupDesc{
    {run_environment::AccSetting::kAVX512_F16, kSpeedupVsAvx512F16Vectorization},
    {run_environment::AccSetting::kAVX512, kSpeedupVsAvx512Vectorization},
    {run_environment::AccSetting::kAVX256_F16, kSpeedupVsAvx256F16Vectorization},
    {run_environment::AccSetting::kAVX256, kSpeedupVsAvx256Vectorization},
    {run_environment::AccSetting::kNEON, kSpeedupVsNEONVectorization},
    {run_environment::AccSetting::kCUDA, kSpeedupVsCUDAAcceleration},
    {run_environment::AccSetting::kNone, kSpeedupCPUVectorization}
  };

  //constants for implementation result
  constexpr std::string_view kImpResultsFolderName{"ImpResults"};
  constexpr std::string_view kImpResultsRunDataFolderName{"RunResults"};
  constexpr std::string_view kImpResultsRunDataDefaultPParamsFolderName{"RunResultsDefaultPParams"};
  constexpr std::string_view kImpResultsRunDataWSpeedupsFolderName{"RunResultsWSpeedups"};
  constexpr std::string_view kImpResultsSpeedupsFolderName{"Speedups"};
  constexpr std::string_view kImpResultsAcrossArchsFolderName{"ResultsAcrossArchitectures"};
  constexpr std::string_view kImpResultsRunDataAccFolderName{"RunResultsAcc"};
  
  //headers that correspond to unique "signature" set of inputs in run
  constexpr std::array<std::string_view, 3> kRunInputSigHeaders{
    kInputIdxHeader, kDatatypeHeader, kLoopItersTemplatedHeader};
  constexpr std::size_t kRunInputNumInputIdx{0};
  constexpr std::size_t kRunInputDatatypeIdx{1};
  constexpr std::size_t kRunInputLoopItersTemplatedIdx{2};

  //declare output results type and array containing all output results types
   enum class OutResults{
    kDefaultPParams, kOptPParams, kSpeedups, kOptWSpeedups
  };
    
  //structure containing directory path and description in file name for
  //each output result file
  struct OutFileInfo{
    std::filesystem::path dir_path;
    std::string_view desc_file_name;
  };

  //mapping from output result type to full directory path where results for
  //type stored and description part of file name 
  const std::map<OutResults, const OutFileInfo> kOutResultsFileInfo{
    {OutResults::kDefaultPParams,
     {kImpResultsRunDataDefaultPParamsFolderName,
      kRunResultsDefaultPParamsDescFileName}},
    {OutResults::kOptPParams,
     {kImpResultsRunDataFolderName,
      kRunResultsDescFileName}},
    {OutResults::kSpeedups,
     {kImpResultsSpeedupsFolderName,
      kSpeedupsDescFileName}},
    {OutResults::kOptWSpeedups,
     {kImpResultsRunDataWSpeedupsFolderName,
      kRunResultsWSpeedupsDescFileName}}
  };

  //mapping from output type to description shown to user
  const std::map<OutResults, const std::string_view> kOutResultsDesc{
    {OutResults::kDefaultPParams,
     "Run inputs and results using default parallel parameters"},
    {OutResults::kOptPParams,
     "Run inputs and optimized results"},
    {OutResults::kSpeedups,
     "Speedup results"},
    {OutResults::kOptWSpeedups, 
     "Input/settings/parameters info, detailed timings, and evaluation for "
     "each run including speedup results"}
  };
};

#endif //RUN_EVAL_CONSTS_ENUMS_H_