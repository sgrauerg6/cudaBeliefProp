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

//namespace for general program run evaluation
namespace run_eval {
  //enum for status to indicate if error or no error
  enum class Status { NO_ERROR, ERROR };

  //constants for output results for individual and sets of runs
  constexpr std::string_view ALL_RUNS_OUTPUT_CSV_FILE_NAME_START{"outputResults"};
  constexpr std::string_view ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE_START{"outputResultsDefaultParallelParams"};
  constexpr std::string_view CSV_FILE_EXTENSION{".csv"};
  constexpr std::string_view OPTIMIZED_RUNTIME_HEADER{"Median Optimized Runtime (including transfer time)"};
  constexpr std::string_view SINGLE_THREAD_RUNTIME_HEADER{"AVERAGE CPU RUN TIME"};
  constexpr std::string_view SPEEDUP_OPT_PAR_PARAMS_HEADER{"Speedup Over Default OMP Thread Count / CUDA Thread Block Dimensions"};
  constexpr std::string_view SPEEDUP_DOUBLE{"Speedup using double-precision relative to float (actually slowdown)"};
  constexpr std::string_view SPEEDUP_HALF{"Speedup using half-precision relative to float"};
  constexpr std::string_view SPEEDUP_DISP_COUNT_TEMPLATE{"Speedup w/ templated disparity count (known at compile-time)"};
  constexpr std::string_view SPEEDUP_VECTORIZATION{"Speedup using CPU vectorization"};
  constexpr std::string_view SPEEDUP_VS_AVX256_VECTORIZATION{"Speedup over AVX256 CPU vectorization"};
}

using MultRunData = std::vector<std::optional<std::vector<RunData>>>;
using MultRunSpeedup = std::pair<std::string, std::array<double, 2>>;

#endif //RUN_EVAL_CONSTS_ENUMS_H