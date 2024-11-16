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
  enum class Status { NO_ERROR, ERROR };

  //constants for headers corresponding to input, input settings, results, and run success
  constexpr std::string_view INPUT_IDX_HEADER{"Input Index"};
  constexpr std::string_view DATATYPE_HEADER{"Data Type"};
  constexpr std::string_view LOOP_ITERS_TEMPLATED_HEADER{"Loop Iters Templated"};
  constexpr std::string_view RUN_SUCCESS_HEADER{"Run Success"};

  //constants for output results for individual and sets of runs
  constexpr std::string_view RUN_RESULTS_DESCRIPTION_FILE_NAME{"RunResults"};
  constexpr std::string_view RUN_RESULTS_W_SPEEDUPS_DESCRIPTION_FILE_NAME{"RunResultsWSpeedups"};
  constexpr std::string_view RUN_RESULTS_DESCRIPTION_DEFAULT_P_PARAMS_FILE_NAME{"ResultsDefaultParallelParams"};
  constexpr std::string_view SPEEDUPS_DESCRIPTION_FILE_NAME{"Speedups"};
  constexpr std::string_view EVALUATION_ACROSS_RUNS_FILE_NAME{"EvaluationAcrossRuns"};
  constexpr std::string_view CSV_FILE_EXTENSION{".csv"};
  constexpr std::string_view OPTIMIZED_RUNTIME_HEADER{"Median Optimized Runtime (including transfer time)"};
  constexpr std::string_view SINGLE_THREAD_RUNTIME_HEADER{"Average Single-Thread CPU run time"};
  constexpr std::string_view SPEEDUP_OPT_PAR_PARAMS_HEADER{"Speedup Over Default OMP Thread Count / CUDA Thread Block Dimensions"};
  constexpr std::string_view SPEEDUP_DOUBLE{"Speedup using double-precision relative to float (actually slowdown)"};
  constexpr std::string_view SPEEDUP_HALF{"Speedup using half-precision relative to float"};
  constexpr std::string_view SPEEDUP_DISP_COUNT_TEMPLATE{"Speedup w/ templated disparity count (known at compile-time)"};
  constexpr std::string_view SPEEDUP_VECTORIZATION{"Speedup using CPU vectorization"};
  constexpr std::string_view SPEEDUP_VS_AVX256_VECTORIZATION{"Speedup over AVX256 CPU vectorization"};

  //constants for implementation result
  constexpr std::string_view IMP_RESULTS_FOLDER_NAME{"ImpResults"};
  constexpr std::string_view IMP_RESULTS_RUN_DATA_FOLDER_NAME{"RunResults"};
  constexpr std::string_view IMP_RESULTS_RUN_DATA_W_SPEEDUPS_FOLDER_NAME{"RunResultsWSpeedups"};
  constexpr std::string_view IMP_RESULTS_SPEEDUPS_FOLDER_NAME{"Speedups"};
  constexpr std::string_view IMP_RESULTS_RESULTS_ACROSS_ARCHS_FOLDER_NAME{"ResultsAcrossArchitectures"};
  
  //headers that correspond to unique "signature" for input mapping in run across multiple inputs on an architecture
  constexpr std::array<std::string_view, 3> RUN_INPUT_SIG_HDRS{INPUT_IDX_HEADER, DATATYPE_HEADER, LOOP_ITERS_TEMPLATED_HEADER};
  constexpr size_t RUN_INPUT_NUM_INPUT_IDX{0};
  constexpr size_t RUN_INPUT_DATATYPE_IDX{1};
  constexpr size_t RUN_INPUT_LOOP_ITERS_TEMPLATED_IDX{2};

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
        if (a[2] == std::string(BOOL_VAL_FALSE_TRUE_DISP_STR[1])) { return true; /* a < b is true */ }
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