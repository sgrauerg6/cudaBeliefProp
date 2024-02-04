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

#include "BpFileProcessing/BpFileHandlingConsts.h"
#include "BpRunImp/RunEvalBpImp.h"
#include "RunImpCPU/RunCPUSettings.h"
#include "RunEvalImp.h"

int main(int argc, char** argv)
{
  //initialize runs using AVX512, AVX256, and no vectorization
  std::shared_ptr<RunBenchmarkImp> runBpImpAVX512 = std::make_shared<RunEvalBpImp<run_environment::AccSetting::AVX512>>();
  std::shared_ptr<RunBenchmarkImp> runBpImpAVX256 = std::make_shared<RunEvalBpImp<run_environment::AccSetting::AVX256>>();
  std::shared_ptr<RunBenchmarkImp> runBpImpNoVect = std::make_shared<RunEvalBpImp<run_environment::AccSetting::NONE>>();
  #ifdef PROCESSOR_NAME
    runImpSettings.processorName_ = PROCESSOR_NAME;
  #endif //PROCESSOR_NAME
  //initialize settings to run implementation and evaluation
  run_environment::RunImpSettings runImpSettings;
  //enable optimization of parallel parameters with setting to use the same parallel parameters for all kernels in run
  //testing on i7-11800H has found that using different parallel parameters (corresponding to OpenMP thread counts)
  //in different kernels in the optimized CPU implementation can increase runtime (may want to test on additional processors)
  runImpSettings.optParallelParmsOptionSetting_ = {true, run_environment::OptParallelParamsSetting::SAME_PARALLEL_PARAMS_ALL_KERNELS_IN_RUN};
  runImpSettings.pParamsDefaultOptOptions_ = {run_cpu::PARALLEL_PARAMS_DEFAULT, run_cpu::PARALLEL_PARAMETERS_OPTIONS};
  runImpSettings.templatedItersSetting_ = run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED;
  runImpSettings.baseOptSingThreadRTimeForTSetting_ = 
        {bp_file_handling::BASELINE_RUN_DATA_PATHS_OPT_SINGLE_THREAD, run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED};
  
  //run belief propagation with all three vectorization implementations, with the AVX512 implementation given first as the expected fastest implementation
  RunAndEvaluateImp::runBenchmark({runBpImpAVX512, runBpImpAVX256, runBpImpNoVect}, runImpSettings);
  return 0;
}
