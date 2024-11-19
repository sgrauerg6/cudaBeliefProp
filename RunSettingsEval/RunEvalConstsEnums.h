/*
 * RunEvalConstsEnums.h
 *
 *  Created on: Jan 19, 2024
 *      Author: scott
 */

#ifndef RUN_EVAL_CONSTS_ENUMS_H
#define RUN_EVAL_CONSTS_ENUMS_H

#include <string_view>
#include <array>
#include <vector>
#include <optional>
#include "RunData.h"
#include "RunSettings.h"

//namespace for general program run evaluation
namespace run_eval {
  //enum for status to indicate if error or no error
  enum class Status { kNoError, kError };

  //constants for headers corresponding to input, input settings, results, and run success
  constexpr std::string_view kInputIdxHeader{"Input Index"};
  constexpr std::string_view kDatatypeHeader{"Data Type"};
  constexpr std::string_view kLoopItersTemplatedHeader{"Loop Iters Templated"};
  constexpr std::string_view kRunSuccessHeader{"Run Success"};

  //constants for output results for individual and sets of runs
  constexpr std::string_view kRunResultsDescFileName{"RunResults"};
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
  constexpr std::string_view kSpeedupDispCountTemplate{"Speedup w/ templated disparity count (known at compile-time)"};
  constexpr std::string_view kSpeedupCPUVectorization{"Speedup using CPU vectorization"};
  constexpr std::string_view kSpeedupVsAvx256Vectorization{"Speedup over AVX256 CPU vectorization"};

  //constants for implementation result
  constexpr std::string_view kImpResultsFolderName{"ImpResults"};
  constexpr std::string_view kImpResultsRunDataFolderName{"RunResults"};
  constexpr std::string_view kImpResultsRunDataWSpeedupsFolderName{"RunResultsWSpeedups"};
  constexpr std::string_view kImpResultsSpeedupsFolderName{"Speedups"};
  constexpr std::string_view kImpResultsAcrossArchsFolderName{"ResultsAcrossArchitectures"};
  
  //headers that correspond to unique "signature" for input mapping in run across multiple inputs on an architecture
  constexpr std::array<std::string_view, 3> kRunInputSigHeaders{kInputIdxHeader, kDatatypeHeader, kLoopItersTemplatedHeader};
  constexpr size_t kRunInputNumInputIdx{0};
  constexpr size_t kRunInputDatatypeIdx{1};
  constexpr size_t kRunInputLoopItersTemplatedIdx{2};

  //comparison for sorting of input signature for each run (datatype, then input number, then whether or not using templated iter count)
  struct LessThanRunSigHdrs {
    inline bool operator()(const std::array<std::string, 3>& a, const std::array<std::string, 3>& b) const {
      //sort by datatype followed by input number followed by templated iters setting
      if (a[1] != b[1]) {
        if (a[1] == "FLOAT") { return true; /* a < b is true */ }
        else if (a[1] == "HALF") { return false; /* a < b is false */ }
        else if (b[1] == "FLOAT") { return false; /* a < b is false */ }
        else if (b[1] == "HALF") { return true; /* a < b is true */ }
      }
      else if (a[0] != b[0]) {
        return std::stoi(a[0]) < std::stoi(b[0]);
      }
      else if (a[2] != b[2]) {
        if (a[2] == std::string(kBoolValFalseTrueDispStr[1])) { return true; /* a < b is true */ }
      }
      return false; /* a <= b is false */
    }

    inline bool operator()(const std::array<std::string_view, 3>& a, const std::array<std::string_view, 3>& b) const {
      return operator()(std::array<std::string, 3>{std::string(a[0]), std::string(a[1]), std::string(a[2])},
                        std::array<std::string, 3>{std::string(b[0]), std::string(b[1]), std::string(b[2])});
    }
  };
};

using MultRunData = std::vector<std::optional<std::vector<RunData>>>;
using MultRunSpeedup = std::pair<std::string, std::array<double, 2>>;
using MultRunDataWSpeedupByAcc = std::unordered_map<run_environment::AccSetting, std::pair<MultRunData, std::vector<MultRunSpeedup>>>;

#endif //RUN_EVAL_CONSTS_ENUMS_H