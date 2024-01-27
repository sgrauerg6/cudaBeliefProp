/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

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

//This file contains the "main" function that drives the optimized CPU BP implementation

#include <array>
#include <vector>
#include <thread>
#include <string_view>
#include "BpConstsAndParams/bpStructsAndEnums.h"
#include "BpRunImp/RunEvalBpImp.h"
#include "BpFileProcessing/BpFileHandlingConsts.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunEvalImp.h"

#if (CPU_VECTORIZATION_DEFINE == NEON_DEFINE)
constexpr run_environment::AccSetting CPU_VECTORIZATION{run_environment::AccSetting::NEON};
#elif (CPU_VECTORIZATION_DEFINE == AVX_256_DEFINE)
constexpr run_environment::AccSetting CPU_VECTORIZATION{run_environment::AccSetting::AVX256};
#elif (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
constexpr run_environment::AccSetting CPU_VECTORIZATION{run_environment::AccSetting::AVX512};
#else
constexpr run_environment::AccSetting CPU_VECTORIZATION{run_environment::AccSetting::NONE};
#endif

int main(int argc, char** argv)
{
  std::unique_ptr<RunEvalBpImp> runBpImp = std::make_unique<RunEvalBpImp>();
  run_environment::RunImpSettings runImpSettings;
  #ifdef PROCESSOR_NAME
    runImpSettings.processorName_ = PROCESSOR_NAME;
  #endif //PROCESSOR_NAME
  //enable optimization of parallel parameters with setting to use the same parallel parameters for all kernels in run
  //testing on i7-11800H has found that using different parallel parameters (corresponding to OpenMP thread counts)
  //in different kernels in the optimized CPU implementation can increase runtime (may want to test on additional processors)
  runImpSettings.optParallelParmsOptionSetting_ = {true, run_environment::OptParallelParamsSetting::SAME_PARALLEL_PARAMS_ALL_KERNELS_IN_RUN};
  runImpSettings.pParamsDefaultOptOptions_ = {run_environment::PARALLEL_PARAMS_DEFAULT, run_environment::PARALLEL_PARAMETERS_OPTIONS};
  runImpSettings.templatedItersSetting_ = run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED;
  runImpSettings.baselineRunDataPathsOptSingThread_ = bp_file_handling::BASELINE_RUN_DATA_PATHS_OPT_SINGLE_THREAD;
  RunAndEvaluateImp::runBpOnStereoSets<CPU_VECTORIZATION>(runBpImp, runImpSettings);
  return 0;
}
