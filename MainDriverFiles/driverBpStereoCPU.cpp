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
#include "RunImp/RunEvalImpMultSettings.h"

//enum to define setting to run implementation
enum class RunImpSetting {
  RUN_IMP_DEFAULT,
  RUN_IMP_THREADS_PINNED_TO_SOCKET,
  RUN_IMP_SIM_SINGLE_CPU_TWO_CPU_SYSTEM
};

//run implementation using input parameters from command line with specified setting
void runImp(int argc, char** argv, RunImpSetting impSetting) {
  //initialize settings to run implementation and evaluation
  run_environment::RunImpSettings runImpSettings;
  //enable optimization of parallel parameters with setting to use the same parallel parameters for all kernels in run
  //testing on i7-11800H has found that using different parallel parameters (corresponding to OpenMP thread counts)
  //in different kernels in the optimized CPU implementation can increase runtime (may want to test on additional processors)
  runImpSettings.optParallelParamsOptionSetting_ = {true, run_environment::OptParallelParamsSetting::SAME_PARALLEL_PARAMS_ALL_KERNELS_IN_RUN};
  runImpSettings.pParamsDefaultOptOptions_ = {run_cpu::PARALLEL_PARAMS_DEFAULT, run_cpu::PARALLEL_PARAMETERS_OPTIONS};
  //set run name to first argument if it exists
  //otherwise set to "CurrentRun"
  runImpSettings.runName_ = (argc > 1) ? argv[1] : "CurrentRun";

  //adjust thread count to simulate single CPU on dual-CPU system
  //currently only works as expected if environment variables are set before run such that threads are pinned to socket via
  //"export OMP_PLACES="sockets"" and "export OMP_PROC_BIND=true" commands on command line
  if (impSetting == RunImpSetting::RUN_IMP_SIM_SINGLE_CPU_TWO_CPU_SYSTEM) {
    //adjust settings to simulate run on single CPU in two-CPU system, specifically set parallel thread count options so that
    //maximum number of parallel threads is thread count of single CPU and set environment variables so that CPU threads
    //are pinned to socket
    //set default parallel threads count to be number of threads on a single CPU in the two-CPU system
    runImpSettings.pParamsDefaultOptOptions_.first = {std::thread::hardware_concurrency() / 2, 1};

    //erase parallel thread count options with more than the number of threads on a single CPU in the two-CPU system
    runImpSettings.removeParallelParamAboveMaxThreads(std::thread::hardware_concurrency() / 2);

    //adjust settings so that CPU threads pinned to socket to simulate run on single CPU
    //TODO: commented out since currently has no effect; needs to be set before run by
    //calling "export OMP_PLACES="sockets"" and "export OMP_PROC_BIND=true" commands on
    //command line before run
    //run_environment::CPUThreadsPinnedToSocket().operator()(true);

    //append run name to specify that simulating single CPU on dual-CPU system
    if (runImpSettings.runName_) {
      *(runImpSettings.runName_) += "_SimSingleCPUOnDualCPUSystem";
    }
  }
  //check if running implementation with CPU threads pinned to socket (for cases with multiple CPUs)
  //TODO: currently not supported due to code for setting CPU threads to be pinned to socket not working as expected
  //can still run implementation with CPU threads pinned to socket by calling "export OMP_PLACES="sockets"" and
  //"export OMP_PROC_BIND=true" commands on command line before run
  /*else if (impSetting == RunImpSetting::RUN_IMP_THREADS_PINNED_TO_SOCKET) {
    //adjust settings so that CPU threads pinned to socket
    //TODO: commented out since currently has no effect
    //run_environment::CPUThreadsPinnedToSocket().operator()(true);

    //append run name to specify that CPU threads pinned to socket
    if (runImpSettings.runName_) {
      *(runImpSettings.runName_) += "_ThreadsPinnedToSocket";
    }
  }*/

  //remove any parallel processing below given minimum number of threads
  runImpSettings.removeParallelParamBelowMinThreads(run_cpu::MIN_NUM_THREADS_RUN);
  runImpSettings.templatedItersSetting_ = run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED;
  runImpSettings.baseOptSingThreadRTimeForTSetting_ = 
    {bp_file_handling::BASELINE_RUN_DATA_PATHS_OPT_SINGLE_THREAD, run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED};
  runImpSettings.subsetStrIndices_ = {{"smallest 3 stereo sets", {0, 1, 2, 3, 4, 5}},
  #ifndef SMALLER_SETS_ONLY
                                      {"largest 3 stereo sets", {8, 9, 10, 11, 12, 13}}};
  #else
                                      {"largest stereo set", {8, 9}}};
  #endif //SMALLER_SETS_ONLY
    
  //run belief propagation with all AVX512, AVX256, and no vectorization implementations, with the AVX512 implementation
  //given first as the expected fastest implementation
  RunEvalImpMultSettings().operator()({{run_environment::AccSetting::AVX512, std::make_shared<RunEvalBpImp>(run_environment::AccSetting::AVX512)},
    {run_environment::AccSetting::AVX256, std::make_shared<RunEvalBpImp>(run_environment::AccSetting::AVX256)},
    {run_environment::AccSetting::NONE, std::make_shared<RunEvalBpImp>(run_environment::AccSetting::NONE)}},
    runImpSettings);
}

int main(int argc, char** argv)
{
  //if running on a system with two cpus, run implementation with settings adjusted
  //to simulate run on single CPU if specified in define
#ifdef SIM_SINGLE_CPU_ON_DUAL_CPU_SYSTEM
  runImp(argc, argv, RunImpSetting::RUN_IMP_SIM_SINGLE_CPU_TWO_CPU_SYSTEM);
#else //SIM_SINGLE_CPU_ON_DUAL_CPU_SYSTEM
  //run default implementation
  runImp(argc, argv, RunImpSetting::RUN_IMP_DEFAULT);
#endif //SIM_SINGLE_CPU_ON_DUAL_CPU_SYSTEM

  return 0;
}
