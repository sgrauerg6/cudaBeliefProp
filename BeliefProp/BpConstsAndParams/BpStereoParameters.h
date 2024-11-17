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
  constexpr float INF_BP = 65504.0f;
}

namespace bp_params
{
  struct BpStereoSet {
    const std::string_view name_;
    const unsigned int numDispVals_;
    const unsigned int scaleFactor_;
  };

  //declare stereo sets to process with name, num disparity values, and scale factor
  constexpr std::array<BpStereoSet, 8> STEREO_SETS_TO_PROCESS{
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
  constexpr unsigned int NUM_BP_STEREO_RUNS = 3;
#else
  constexpr unsigned int NUM_BP_STEREO_RUNS = 15;
#endif //LIMITED_TEST_PARAMS_FEWER_RUNS

  // number of BP iterations at each scale/level
  constexpr unsigned int ITER_BP = 7;

  // number of scales/levels in the pyramid to run BP
  constexpr unsigned int LEVELS_BP = 5;

  // truncation of data cost
  constexpr float DATA_K_BP = 15.0f;

  // weighing of data cost
  constexpr float LAMBDA_BP = 0.1f;

  // amount to smooth the input images
  constexpr float SIGMA_BP = 0.0f;

  //by default, optimized GPU memory management and optimized indexing used
  //See http://scottgg.net/OptimizingGlobalStereoMatchingOnNVIDIAGPUs.pdf for more info on these
  //optimizations (note that the optimized indexing was present in the initial implementation)
  //Can remove optimized GPU memory management (making the processing more similar to the initial work)
  //by setting USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT to false
  //Optimized indexing can be turned off by changing the OPTIMIZED_INDEXING_SETTING value to false
  //(not recommended; this slows down processing)
  constexpr bool USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT{true};
  constexpr bool OPTIMIZED_INDEXING_SETTING{true};
  constexpr bool ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS{true};

  //retrieve run settings as a RunData object for output
  inline RunData runSettings()  {
    RunData currRunData;
    currRunData.addDataWHeader("Memory Optimization Level", std::to_string(USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT));
    currRunData.addDataWHeader("Indexing Optimization Level", std::to_string(OPTIMIZED_INDEXING_SETTING));
    return currRunData;
  }
};

#endif /* BPSTEREOPARAMETERS_H_ */
