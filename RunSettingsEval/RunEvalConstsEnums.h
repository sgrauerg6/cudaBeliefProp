/*
 * RunEvalConstsEnums.h
 *
 *  Created on: Jan 19, 2024
 *      Author: scott
 */

#ifndef RUN_EVAL_CONSTS_ENUMS_H
#define RUN_EVAL_CONSTS_ENUMS_H

#include <string_view>

//remove comment to only process on smaller stereo sets (reduces runtime)
#define SMALLER_SETS_ONLY

namespace run_eval {

  enum class BaselineData { OPTIMIZED, SINGLE_THREAD };
  enum class Status { NO_ERROR, ERROR };

  //constants for output results for individual and sets of runs
  constexpr std::string_view BP_ALL_RUNS_OUTPUT_CSV_FILE_NAME_START{"outputResults"};
  constexpr std::string_view BP_ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE_START{"outputResultsDefaultParallelParams"};
  constexpr std::string_view CSV_FILE_EXTENSION{".csv"};
  constexpr std::string_view OPTIMIZED_RUNTIME_HEADER{"Median Optimized Runtime (including transfer time)"};
  constexpr std::string_view SINGLE_THREAD_RUNTIME_HEADER{"AVERAGE CPU RUN TIME"};
  constexpr std::string_view SPEEDUP_OPT_PAR_PARAMS_HEADER{"Speedup Over Default OMP Thread Count / CUDA Thread Block Dimensions"};
  constexpr std::string_view SPEEDUP_DOUBLE{"Speedup using double-precision relative to float (actually slowdown)"};
  constexpr std::string_view SPEEDUP_HALF{"Speedup using half-precision relative to float"};
  constexpr std::string_view SPEEDUP_DISP_COUNT_TEMPLATE{"Speedup w/ templated disparity count (known at compile-time)"};
  constexpr std::string_view SPEEDUP_VECTORIZATION{"Speedup using CPU vectorization"};
  constexpr std::string_view SPEEDUP_VS_AVX256_VECTORIZATION{"Speedup over AVX256 CPU vectorization"};
#ifdef SMALLER_SETS_ONLY
  constexpr std::string_view BASELINE_RUNTIMES_FILE_PATH{"../BpBaselineRuntimes/baselineRuntimesSmallerSetsOnly.txt"};
  constexpr std::string_view SINGLE_THREAD_BASELINE_RUNTIMES_FILE_PATH{"../BpBaselineRuntimes/singleThreadBaselineRuntimesSmallerSetsOnly.txt"};
#else
  constexpr std::string_view BASELINE_RUNTIMES_FILE_PATH{"../BpBaselineRuntimes/baselineRuntimes.txt"};
  constexpr std::string_view SINGLE_THREAD_BASELINE_RUNTIMES_FILE_PATH{"../BpBaselineRuntimes/singleThreadBaselineRuntimes.txt"};
#endif //SMALLER_SETS_ONLY

}

#endif //RUN_EVAL_CONSTS_ENUMS_H