/*
 * RunEvalConstsEnums.h
 *
 *  Created on: Jan 19, 2024
 *      Author: scott
 */

#ifndef RUN_EVAL_CONSTS_ENUMS_H_
#define RUN_EVAL_CONSTS_ENUMS_H_

#include <array>
#include <string_view>
#include "RunSettingsParams/RunSettingsConstsEnums.h"

/** @brief Namespace for general program run evaluation */
namespace run_eval {

/** @brief Enum for status to indicate if error or no error */
enum class Status { kNoError, kError };

/** @brief Enum to specify average or median for "middle" value
 * in data */
enum class MiddleValData { kAverage, kMedian };

//set data types to use in evaluation
//by default evaluate using float, double, and half data types
//but can set to only evaluate using float datatype
#ifdef EVAL_FLOAT_DATATYPE_ONLY
  constexpr std::array<size_t, 1> kDataTypesEvalSizes{sizeof(float)};
#else
  constexpr std::array<size_t, 3> kDataTypesEvalSizes{sizeof(float), sizeof(double), sizeof(halftype)};
#endif //EVAL_FLOAT_DATATYPE_ONLY

  /** @brief Define string for display of "true" and "false" values of bool value */
  constexpr std::array<std::string_view, 2> kBoolValFalseTrueDispStr{"NO", "YES"};

  //constants for headers corresponding to input, input settings, results, and run success
  constexpr std::string_view kInputIdxHeader{"Input Index"};
  constexpr std::string_view kDatatypeHeader{"Data Type"};
  constexpr std::string_view kLoopItersTemplatedHeader{"Loop Iters Templated"};
  constexpr std::string_view kRunSuccessHeader{"Run Success"};

  /** @brief Constant for "all runs" string */
  constexpr std::string_view kAllRunsStr{"All Runs"};

  //constants for output results for individual and sets of runs
  constexpr std::string_view kRunResultsDescFileName{"RunResults"};
  constexpr std::string_view kRunResultsDefaultPParamsDescFileName{"RunResultsDefaultPParams"};
  constexpr std::string_view kRunResultsWSpeedupsDescFileName{"RunResultsWSpeedups"};
  constexpr std::string_view kRunResultsDescDefaultPParamsFileName{"ResultsDefaultParallelParams"};
  constexpr std::string_view kSpeedupsDescFileName{"Speedups"};
  constexpr std::string_view kEvalAcrossRunsFileName{"EvaluationAcrossRuns"};
  constexpr std::string_view kCsvFileExtension{".csv"};
  constexpr std::string_view kOptimizedRuntimeHeader{"Median Optimized Runtime (including transfer time)"};
  constexpr std::string_view kSingleThreadRuntimeHeader{"Average Single-Thread CPU run time"};
  constexpr std::string_view kSpeedupOptParParamsHeader{"Speedup Over Default OMP Thread Count / CUDA Thread Block Dimensions"};
  constexpr std::string_view kSpeedupDouble{"Speedup using double-precision relative to float (actually slowdown)"};
  constexpr std::string_view kSpeedupHalf{"Speedup using half-precision relative to float"};
  constexpr std::string_view kSpeedupLoopItersCountTemplate{"Speedup w/ templated disparity count (known at compile-time)"};
  constexpr std::string_view kSpeedupCPUVectorization{"Speedup using CPU vectorization"};
  constexpr std::string_view kSpeedupVsAvx256Vectorization{"Speedup over AVX256 CPU vectorization"};

  //constants for implementation result
  constexpr std::string_view kImpResultsFolderName{"ImpResults"};
  constexpr std::string_view kImpResultsRunDataFolderName{"RunResults"};
  constexpr std::string_view kImpResultsRunDataDefaultPParamsFolderName{"RunResultsDefaultPParams"};
  constexpr std::string_view kImpResultsRunDataWSpeedupsFolderName{"RunResultsWSpeedups"};
  constexpr std::string_view kImpResultsSpeedupsFolderName{"Speedups"};
  constexpr std::string_view kImpResultsAcrossArchsFolderName{"ResultsAcrossArchitectures"};
  
  //headers that correspond to unique "signature" set of inputs in run
  constexpr std::array<std::string_view, 3> kRunInputSigHeaders{
    kInputIdxHeader, kDatatypeHeader, kLoopItersTemplatedHeader};
  constexpr std::size_t kRunInputNumInputIdx{0};
  constexpr std::size_t kRunInputDatatypeIdx{1};
  constexpr std::size_t kRunInputLoopItersTemplatedIdx{2};
};

#endif //RUN_EVAL_CONSTS_ENUMS_H_