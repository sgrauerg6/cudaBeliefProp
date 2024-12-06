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

#include <iostream>
#include <array>
#include "RunImp/RunEvalImpMultSettings.h"
#include "RunImpCPU/RunCPUSettings.h"
#include "BpFileProcessing/BpFileHandlingConsts.h"
#include "BpRunProcessing/BpSettings.h"
#include "BpRunEvalImp/RunEvalBpImp.h"
#include "BpResultsEvaluation/EvaluateBPImpResults.h"

//enum to define setting to run implementation
enum class RunImpSetting {
  kRunImpDefault,
  kRunImpThreadsPinnedToSocket,
  kRunImpSimSingleCPUTwoCPUSystem
};

//run implementation using input parameters from command line with specified setting
void runImp(int argc, char** argv, RunImpSetting impSetting)
{
  //initialize settings to run implementation and evaluation
  run_environment::RunImpSettings run_imp_settings;

  //enable optimization of parallel parameters with setting to use the same parallel parameters for all kernels in run
  //testing on i7-11800H has found that using different parallel parameters (corresponding to OpenMP thread counts)
  //in different kernels in the optimized CPU implementation can increase runtime (may want to test on additional processors)
  run_imp_settings.opt_parallel_params_setting =
    {true,
     run_environment::OptParallelParamsSetting::kSameParallelParamsAllKernels};

  //set default parallel parameters and parallel parameters to benchmark when searching for optimal
  //parallel parameters
  run_imp_settings.p_params_default_opt_settings =
    {run_cpu::kParallelParamsDefault,
     run_cpu::kParallelParameterOptions};

  //set run name to first argument if it exists
  //otherwise set to "CurrentRun"
  run_imp_settings.run_name = (argc > 1) ? argv[1] : "CurrentRun";

  //adjust thread count to simulate single CPU on dual-CPU system
  //currently only works as expected if environment variables are set before run such that threads are pinned to socket via
  //"export OMP_PLACES="sockets"" and "export OMP_PROC_BIND=true" commands on command line
  if (impSetting == RunImpSetting::kRunImpSimSingleCPUTwoCPUSystem)
  {
    //adjust settings to simulate run on single CPU in two-CPU system, specifically set parallel thread count options so that
    //maximum number of parallel threads is thread count of single CPU and set environment variables so that CPU threads
    //are pinned to socket
    //set default parallel threads count to be number of threads on a single CPU in the two-CPU system
    run_imp_settings.p_params_default_opt_settings.first =
      {std::thread::hardware_concurrency() / 2, 1};

    //erase parallel thread count options with more than the number of threads on a single CPU in the two-CPU system
    run_imp_settings.RemoveParallelParamAboveMaxThreads(
      std::thread::hardware_concurrency() / 2);

    //adjust settings so that CPU threads pinned to socket to simulate run on single CPU
    //TODO: commented out since currently has no effect; needs to be set before run by
    //calling "export OMP_PLACES="sockets"" and "export OMP_PROC_BIND=true" commands on
    //command line before run
    //run_environment::CPUThreadsPinnedToSocket().operator()(true);

    //append run name to specify that simulating single CPU on dual-CPU system
    if (run_imp_settings.run_name) {
      *(run_imp_settings.run_name) += "_SimSingleCPUOnDualCPUSystem";
    }
  }
  //check if running implementation with CPU threads pinned to socket (for cases with multiple CPUs)
  //TODO: currently not supported due to code for setting CPU threads to be pinned to socket not working as expected
  //can still run implementation with CPU threads pinned to socket by calling "export OMP_PLACES="sockets"" and
  //"export OMP_PROC_BIND=true" commands on command line before run
  /*else if (impSetting == RunImpSetting::kRunImpThreadsPinnedToSocket) {
    //adjust settings so that CPU threads pinned to socket
    //TODO: commented out since currently has no effect
    //run_environment::CPUThreadsPinnedToSocket().operator()(true);

    //append run name to specify that CPU threads pinned to socket
    if (run_imp_settings.run_name) {
      *(run_imp_settings.run_name) += "_ThreadsPinnedToSocket";
    }
  }*/

  //set datatype(s) to use in run processing in evaluation
  run_imp_settings.datatypes_eval_sizes =
    {run_eval::kDataTypesEvalSizes.begin(),
     run_eval::kDataTypesEvalSizes.end()};

  //remove any parallel processing below given minimum number of threads
  run_imp_settings.RemoveParallelParamBelowMinThreads(
    run_cpu::kMinNumThreadsRun);

  //set setting of whether or not to use templated loop iterations in implementation
  //in evaluation runs
  run_imp_settings.templated_iters_setting =
    run_environment::TemplatedItersSetting::kRunOnlyNonTemplated;

  //set path of baseline runtimes and baseline description
  run_imp_settings.baseline_runtimes_path_desc =
    {bp_file_handling::kBaselineRunDataPath,
     bp_file_handling::kBaselineRunDesc};

  //set data subsets to evaluate separate from all data
  run_imp_settings.subset_desc_input_sig = beliefprop::kEvalDataSubsets;

  //run belief propagation with AVX512, AVX256, and no vectorization implementations,
  //with the AVX512 implementation given first as the expected fastest implementation
  RunEvalImpMultSettings().operator()({
    {run_environment::AccSetting::kAVX512,
     std::make_shared<RunEvalBpImp>(run_environment::AccSetting::kAVX512)},
    {run_environment::AccSetting::kAVX256,
     std::make_shared<RunEvalBpImp>(run_environment::AccSetting::kAVX256)},
    {run_environment::AccSetting::kNone,
     std::make_shared<RunEvalBpImp>(run_environment::AccSetting::kNone)}},
    run_imp_settings,
    std::make_unique<EvaluateBPImpResults>());
}

int main(int argc, char** argv)
{
  //if running on a system with two cpus, run implementation with settings adjusted
  //to simulate run on single CPU if specified in second command line argument
  if ((argc > 2) && (std::string(argv[2]) == std::string(run_cpu::kSimulateSingleCPU))) {
    std::cout << "Running optimized CPU implementation with settings adjusted such that "
                 "a single CPU is simulated on a dual-CPU system." << std::endl;
    std::cout << "Results only as expected if running on dual-CPU system and "
                 "environment variables set so that threads pinned to CPU socket." << std::endl;
    runImp(argc, argv, RunImpSetting::kRunImpSimSingleCPUTwoCPUSystem);
  }
  else {
    //run default implementation
    std::cout << "Running optimized CPU implementation" << std::endl;
    runImp(argc, argv, RunImpSetting::kRunImpDefault);
  }

  return 0;
}
