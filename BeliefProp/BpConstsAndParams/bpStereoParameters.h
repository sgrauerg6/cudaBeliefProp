/*
 * bpStereoParameters.h
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
#include <array>
#include <fstream>
#include <ostream>
#include "RunSettingsEval/RunData.h"

namespace bp_consts
{
  constexpr float INF_BP = 65504.0f;     // large cost (used for "infinity"), value set to support half type
}

namespace bp_params
{
  enum image_set_options
  {
    TSUKUBA_IMAGES_HALF_SIZE_E = 0,
    TSUKUBA_IMAGES_E = 1,
    VENUS_IMAGES_E = 2,
    BARN_1_IMAGES_E = 3,
    CONES_IMAGES_QUARTER_SIZE_E = 4,
    CONES_IMAGES_HALF_SIZE_E = 5,
    CONES_IMAGES_FULL_SIZE_CROPPED_E = 6,
    CONES_IMAGES_FULL_SIZE_E = 7
  };

  struct BpStereoSet {
    const char* name;
    unsigned int numDispVals;
    unsigned int scaleFactor;
  };

  constexpr std::array<BpStereoSet, 8> STEREO_SETS_TO_PROCESS{
    //declare stereo sets to process with name, num disparity values, and scale factor
    //order is the same as in image_set_options enum
    BpStereoSet{"tsukubaSetHalfSize", 8, 32},
    BpStereoSet{"tsukubaSet", 16, 16},
    BpStereoSet{"venus", 21, 8},
    BpStereoSet{"barn1", 32, 8},
    BpStereoSet{"conesQuarterSize", 64, 4},
    BpStereoSet{"conesHalfSize", 128, 2},
    BpStereoSet{"conesFullSizeCropped", 256, 1},
    BpStereoSet{"conesFullSize", 256, 1}
  };

  constexpr std::array<unsigned int, 8> NUM_POSSIBLE_DISPARITY_VALUES{
    STEREO_SETS_TO_PROCESS[TSUKUBA_IMAGES_HALF_SIZE_E].numDispVals,
    STEREO_SETS_TO_PROCESS[TSUKUBA_IMAGES_E].numDispVals,
    STEREO_SETS_TO_PROCESS[VENUS_IMAGES_E].numDispVals,
    STEREO_SETS_TO_PROCESS[BARN_1_IMAGES_E].numDispVals,
    STEREO_SETS_TO_PROCESS[CONES_IMAGES_QUARTER_SIZE_E].numDispVals,
    STEREO_SETS_TO_PROCESS[CONES_IMAGES_HALF_SIZE_E].numDispVals,
    STEREO_SETS_TO_PROCESS[CONES_IMAGES_FULL_SIZE_CROPPED_E].numDispVals,
    STEREO_SETS_TO_PROCESS[CONES_IMAGES_FULL_SIZE_E].numDispVals};
  constexpr std::array<unsigned int, 8> SCALE_BP{
    STEREO_SETS_TO_PROCESS[TSUKUBA_IMAGES_HALF_SIZE_E].scaleFactor,
    STEREO_SETS_TO_PROCESS[TSUKUBA_IMAGES_E].scaleFactor,
    STEREO_SETS_TO_PROCESS[VENUS_IMAGES_E].scaleFactor,
    STEREO_SETS_TO_PROCESS[BARN_1_IMAGES_E].scaleFactor,
    STEREO_SETS_TO_PROCESS[CONES_IMAGES_QUARTER_SIZE_E].scaleFactor,
    STEREO_SETS_TO_PROCESS[CONES_IMAGES_HALF_SIZE_E].scaleFactor,
    STEREO_SETS_TO_PROCESS[CONES_IMAGES_FULL_SIZE_CROPPED_E].scaleFactor,
    STEREO_SETS_TO_PROCESS[CONES_IMAGES_FULL_SIZE_E].scaleFactor};
  constexpr std::array<const char*, 8> STEREO_SET{
    STEREO_SETS_TO_PROCESS[TSUKUBA_IMAGES_HALF_SIZE_E].name,
    STEREO_SETS_TO_PROCESS[TSUKUBA_IMAGES_E].name,
    STEREO_SETS_TO_PROCESS[VENUS_IMAGES_E].name,
    STEREO_SETS_TO_PROCESS[BARN_1_IMAGES_E].name,
    STEREO_SETS_TO_PROCESS[CONES_IMAGES_QUARTER_SIZE_E].name,
    STEREO_SETS_TO_PROCESS[CONES_IMAGES_HALF_SIZE_E].name,
    STEREO_SETS_TO_PROCESS[CONES_IMAGES_FULL_SIZE_CROPPED_E].name,
    STEREO_SETS_TO_PROCESS[CONES_IMAGES_FULL_SIZE_E].name};

  //number of belief propagation stereo runs of same image set
  constexpr unsigned int NUM_BP_STEREO_RUNS = 15;

  //define the default message value...
  constexpr float DEFAULT_INITIAL_MESSAGE_VAL = 0.0f;

  // number of BP iterations at each scale/level
  constexpr unsigned int ITER_BP = 7;

  // number of scales/levels in the pyramid to run BP
  constexpr unsigned int LEVELS_BP = 5;

  //truncation of discontinuity cost
  constexpr std::array<float, 8> DISC_K_BP{
    (float)STEREO_SETS_TO_PROCESS[TSUKUBA_IMAGES_HALF_SIZE_E].numDispVals / 7.5f,
    (float)STEREO_SETS_TO_PROCESS[TSUKUBA_IMAGES_E].numDispVals / 7.5f,
    (float)STEREO_SETS_TO_PROCESS[VENUS_IMAGES_E].numDispVals / 7.5f,
    (float)STEREO_SETS_TO_PROCESS[BARN_1_IMAGES_E].numDispVals / 7.5f,
    (float)STEREO_SETS_TO_PROCESS[CONES_IMAGES_QUARTER_SIZE_E].numDispVals / 7.5f,
    (float)STEREO_SETS_TO_PROCESS[CONES_IMAGES_HALF_SIZE_E].numDispVals / 7.5f,
    (float)STEREO_SETS_TO_PROCESS[CONES_IMAGES_FULL_SIZE_CROPPED_E].numDispVals / 7.5f,
    (float)STEREO_SETS_TO_PROCESS[CONES_IMAGES_FULL_SIZE_E].numDispVals / 7.5f};

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
