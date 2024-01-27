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

//This file contains the "main" function that drives the CUDA BP implementation

#include <array>
#include <vector>
#include <thread>
#include <string_view>
#include "BpConstsAndParams/bpStructsAndEnums.h"
#include "BpFileProcessing/BpFileHandlingConsts.h"
#include "RunImpCUDA/RunCUDASettings.h"
#include "RunEvalImp.h"

int main(int argc, char** argv)
{
  std::unique_ptr<RunEvalBpImp> runBpImp = std::make_unique<RunEvalBpImp>();
  run_environment::RunImpSettings runImpSettings;
  #ifdef PROCESSOR_NAME
    runImpSettings.processorName_ = PROCESSOR_NAME;
  #endif //PROCESSOR_NAME
  //enable optimization of parallel parameters with setting to use the allow different thread block dimensions
  //on kernels in same run
  //testing on has found that using different parallel parameters (corresponding to thread block dimensions)
  //in different kernels in the optimized CUDA implementation can decrease runtime
  runImpSettings.optParallelParmsOptionSetting_ = {true, run_environment::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN};
  runImpSettings.pParamsDefaultOptOptions_ = {run_cuda::PARALLEL_PARAMS_DEFAULT, run_cuda::PARALLEL_PARAMETERS_OPTIONS};
  runImpSettings.templatedItersSetting_ = run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED;
  runImpSettings.baselineRunDataPathsOptSingThread_ = bp_file_handling::BASELINE_RUN_DATA_PATHS_OPT_SINGLE_THREAD;
  RunAndEvaluateImp::runBpOnStereoSets<run_environment::AccSetting::CUDA>(runBpImp, runImpSettings);
  return 0;
}
