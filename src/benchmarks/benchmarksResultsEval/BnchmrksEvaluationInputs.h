/*
Copyright (C) 2026 Scott Grauer-Gray

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

/**
 * @file BnchmrksEvaluationInputs.h
 * @author Scott Grauer-Gray
 * @brief Header file that contains information about the inputs used for
 * evaluation of the benchmark(s).
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef BNCHMRKS_EVALUATION_INPUTS_H_
#define BNCHMRKS_EVALUATION_INPUTS_H_

#include <string_view>
#include <array>
#include <unordered_map>
#include "RunSettingsParams/InputSignature.h"

namespace benchmarks
{
  //headers for input matrix width and height
  constexpr std::string_view kMatWidthHeader{"Matrix Width"};
  constexpr std::string_view kMatHeightHeader{"Matrix Height"};

  /**
   * @brief Enum type where each named constant corresponds to a
   * matrix width and height that can be processed and evaluated
   * Underlying size_t value for each constant must match index of
   * stereo set in kStereoSetsToProcess array
   */
  enum class MtrxWH : size_t {
    kMtrxWH_32 = 0,
    kMtrxWH_64,
    kMtrxWH_128,
    kMtrxWH_256,
    kMtrxWH_512,
    kMtrxWH_1024,
    kMtrxWH_2048,
    kMtrxWH_4096,
    kMtrxWH_6144,
    kMtrxWH_8192,
    kMtrxWH_12288,
    kMtrxWH_16384
  };
  
  /**
   * @brief Structure with matrix size name and corresponding width/height
   */
  struct MtrxNameWH {
    const std::string_view name;
    const unsigned int mtrx_wh;
  };

  /** 
   * @brief Declare matrix to process with name and width/height
   * 
   * Index for each matrix must match corresponding enum value in MtrxWH enum
   * 
   * Reason array is used is that the data structure must be constexpr due to height/width
   * potentially being used as a template parameter where the value must be known
   * at compile time...otherwise would be unordered_map with MtrxWH enum as the key
   * (in C++20 array can be constexpr but unordered_map can't be)
   */
  constexpr std::array<MtrxNameWH, 12> kMtrxsToProcess{
    MtrxNameWH{"kMtrxWH_32", 32},
    MtrxNameWH{"kMtrxWH_64", 64},
    MtrxNameWH{"kMtrxWH_128", 128},
    MtrxNameWH{"kMtrxWH_256", 256},
    MtrxNameWH{"kMtrxWH_512", 512},
    MtrxNameWH{"kMtrxWH_1024", 1024},
    MtrxNameWH{"kMtrxWH_2048", 2048},
    MtrxNameWH{"kMtrxWH_4096", 4096},
    MtrxNameWH{"kMtrxWH_6144", 6144},
    MtrxNameWH{"kMtrxWH_8192", 8192},
    MtrxNameWH{"kMtrxWH_12288", 12288},
    MtrxNameWH{"kMtrxWH_16384", 16384}
  };

  //constants for evaluation
  constexpr std::string_view kBenchmarksDirectoryName{"Benchmarks"};
  /*constexpr std::string_view kBaselineRunDataPath{
    "../../Benchmarks/ImpResults/RunResults/AMDRome48Cores_RunResults.csv"};
  constexpr std::string_view kBaselineRunDesc{"AMD Rome (48 Cores)"};*/

  /** @brief Define subsets for evaluating run results on specified inputs */
  const std::vector<std::pair<std::string, std::vector<InputSignature>>> kEvalDataSubsets{};
};

#endif /* BNCHMRKS_EVALUATION_INPUTS_H_ */
