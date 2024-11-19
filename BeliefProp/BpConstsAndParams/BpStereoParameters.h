/*
 * BpStereoParameters.h
 *
 *  Created on: Jun 18, 2019
 *      Author: scott
 */

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

//This class defines parameters for the cuda implementation for disparity map estimation for a pair of stereo images

#ifndef BPSTEREOPARAMETERS_H_
#define BPSTEREOPARAMETERS_H_

#include <string>
#include <string_view>
#include <array>
#include "RunSettingsEval/RunData.h"

namespace bp_consts
{
  //float value of "infinity" that works with half-precision
  constexpr float kInfBp = 65504.0f;
}

namespace bp_params
{
  struct BpStereoSet {
    const std::string_view name;
    const unsigned int num_disp_vals;
    const unsigned int scale_factor;
  };

  //declare stereo sets to process with name, num disparity values, and scale factor
  constexpr std::array<BpStereoSet, 8> kStereoSetsToProcess{
    BpStereoSet{"tsukubaSetHalfSize", 8, 32},
    BpStereoSet{"tsukubaSet", 16, 16},
    BpStereoSet{"venus", 21, 8},
    BpStereoSet{"barn1", 32, 8},
    BpStereoSet{"conesQuarterSize", 64, 4},
    BpStereoSet{"conesHalfSize", 128, 2},
    BpStereoSet{"conesFullSizeCropped", 256, 1},
    BpStereoSet{"conesFullSize", 256, 1}
  };

  //number of belief propagation stereo runs of same image set
  //fewer runs if using limited parameters for faster processing
#ifdef LIMITED_TEST_PARAMS_FEWER_RUNS
  constexpr unsigned int kNumBpStereoRuns = 3;
#else
  constexpr unsigned int kNumBpStereoRuns = 15;
#endif //LIMITED_TEST_PARAMS_FEWER_RUNS

  // number of BP iterations at each scale/level
  constexpr unsigned int kItersBp = 7;

  // number of scales/levels in the pyramid to run BP
  constexpr unsigned int kLevelsBp = 5;

  // truncation of data cost
  constexpr float kDataKBp = 15.0f;

  // weighing of data cost
  constexpr float kLambdaBp = 0.1f;

  // amount to smooth the input images
  constexpr float kSigmaBp = 0.0f;

  //by default, optimized GPU memory management and optimized indexing used
  //See http://scottgg.net/OptimizingGlobalStereoMatchingOnNVIDIAGPUs.pdf for more info on these
  //optimizations (note that the optimized indexing was present in the initial implementation)
  //Can remove optimized GPU memory management (making the processing more similar to the initial work)
  //by setting kUseOptGPUMemManagement to false
  //Optimized indexing can be turned off by changing the kOptimizedIndexingSetting value to false
  //(not recommended; this slows down processing)
  constexpr bool kUseOptGPUMemManagement{true};
  constexpr bool kOptimizedIndexingSetting{true};
  constexpr bool kAllocateFreeBpMemoryOutsideRuns{true};

  //retrieve run settings as a RunData object for output
  inline RunData RunSettings()  {
    RunData curr_run_data;
    curr_run_data.AddDataWHeader("Memory Optimization Level", std::to_string(kUseOptGPUMemManagement));
    curr_run_data.AddDataWHeader("Indexing Optimization Level", std::to_string(kOptimizedIndexingSetting));
    return curr_run_data;
  }
};

#endif /* BPSTEREOPARAMETERS_H_ */
