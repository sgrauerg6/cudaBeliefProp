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
#include <charconv>
#include "RunData.h"
#include "RunSettings.h"

//namespace for general program run evaluation
namespace run_eval {
  //enum for status to indicate if error or no error
  enum class Status { kNoError, kError };

  //define string for display of "true" and "false" values of bool value
  constexpr std::array<std::string_view, 2> kBoolValFalseTrueDispStr{"NO", "YES"};

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
  
  //headers that correspond to unique "signature" set of inputs in run
  constexpr std::array<std::string_view, 3> kRunInputSigHeaders{
    kInputIdxHeader, kDatatypeHeader, kLoopItersTemplatedHeader};
  constexpr std::size_t kRunInputNumInputIdx{0};
  constexpr std::size_t kRunInputDatatypeIdx{1};
  constexpr std::size_t kRunInputLoopItersTemplatedIdx{2};

  //comparison for sorting of input signature for each run
  //(datatype, evaluation data number, whether or not using templated iter count)
  struct LessThanRunSigHdrs {
    inline bool operator()(const std::array<std::string_view, 3>& a, const std::array<std::string_view, 3>& b) const {
      //sort by datatype followed by evaluation data number followed by templated iters setting
      if (a[1] != b[1]) {
        //compare datatype
        //order is float, double, half
        //define mapping of datatype string to value for comparison
        const std::map<std::string_view, unsigned int> datatype_str_to_val{
          {run_environment::kDataSizeToNameMap.at(sizeof(float)), 0},
          {run_environment::kDataSizeToNameMap.at(sizeof(double)), 1},
          {run_environment::kDataSizeToNameMap.at(sizeof(short)), 2}};
        return (datatype_str_to_val.at(a[1]) < datatype_str_to_val.at(b[1]));
      }
      else if (a[0] != b[0]) {
        //compare evaluation data number
        //ordering is as expected by numeric value (such as 0 < 1)
        int a_val, b_val;
        std::from_chars(a[0].begin(), a[0].begin() + a[0].size(), a_val);
        std::from_chars(b[0].begin(), b[0].begin() + b[0].size(), b_val);
        return (a_val < b_val);
      }
      else if (a[2] != b[2]) {
        //compare whether or not using templated iter count
        //order is using templated iter count followed by not using templated iter count
        if (a[2] == kBoolValFalseTrueDispStr[1]) { return true; /* a < b is true */ }
      }
      return false; /* a <= b is false */
    }

    /*inline bool operator()(const std::array<std::string, 3>& a, const std::array<std::string, 3>& b) const {
      return operator()(std::array<std::string_view, 3>{a[0], a[1], a[2]},
                        std::array<std::string_view, 3>{b[0], b[1], b[2]});
    }*/
  };
};

using MultRunData = std::vector<std::optional<std::vector<RunData>>>;
using MultRunSpeedup = std::pair<std::string, std::array<double, 2>>;
using MultRunDataWSpeedupByAcc =
  std::unordered_map<run_environment::AccSetting, std::pair<MultRunData, std::vector<MultRunSpeedup>>>;

#endif //RUN_EVAL_CONSTS_ENUMS_H