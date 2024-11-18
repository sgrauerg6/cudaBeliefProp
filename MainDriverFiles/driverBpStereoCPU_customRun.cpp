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
#include "BpFileProcessing/BpFileHandlingConsts.h"
#include "BpRunImp/RunEvalBpImp.h"
#include "BpRunImp/RunEvalBPImpOnInput.h"
#include "RunImpCPU/RunCPUSettings.h"
#include "RunImp/RunEvalImpMultSettings.h"
#include "BpRunProcessing/RunBpStereoSet.h"
#include "BpOptimizeCPU/RunBpStereoOptimizedCPU.h"
#include "BpSingleThreadCPU/stereo.h"

int main(int argc, char** argv)
{
  std::array<std::string, 2> refTestImPath{argv[1], argv[2]};
  beliefprop::BPsettings algSettings;
  algSettings.num_disp_vals_ = std::stoi(argv[3]);
  algSettings.disc_k_bp_ = (float)algSettings.num_disp_vals_ / 7.5f;
  unsigned int dispMapScale = 256 / algSettings.num_disp_vals_;

  auto numThreads = std::thread::hardware_concurrency();
  BpParallelParams parallelParams{run_environment::OptParallelParamsSetting::kSameParallelParamsAllKernels, algSettings.num_levels_, {numThreads, 1}};

  std::unique_ptr<RunBpStereoSet<float, 0, run_environment::AccSetting::kAVX512>> runOptBpNumItersNoTemplate =
    std::make_unique<RunBpStereoOptimizedCPU<float, 0, run_environment::AccSetting::kAVX512>>();
  auto run_output = runOptBpNumItersNoTemplate->operator()({refTestImPath[0], refTestImPath[1]}, algSettings, parallelParams);
  std::cout << "BP processing runtime (optimized w/ OpenMP + SIMD on CPU): " << run_output->run_time.count() << std::endl;
  if ((argc > 5) && (std::string(argv[5]) == "comp")) {
    std::unique_ptr<RunBpStereoSet<float, 64, run_environment::AccSetting::kNone>> runBpStereoSingleThread = 
      std::make_unique<RunBpStereoCPUSingleThread<float, 64>>();
    auto run_output_single_thread = runBpStereoSingleThread->operator()({refTestImPath[0], refTestImPath[1]}, algSettings, parallelParams);
    std::cout << "BP processing runtime (single threaded imp): " << run_output_single_thread->run_time.count() << std::endl;
  }
  std::cout << "Output disparity map saved to " << argv[4] << std::endl;
  run_output->out_disparity_map.SaveDisparityMap(argv[4], dispMapScale);

  return 0;
}
