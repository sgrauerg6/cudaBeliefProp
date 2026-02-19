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

  //constants for evaluation
  constexpr std::string_view kBenchmarksDirectoryName{"Benchmarks"};
  /*constexpr std::string_view kBaselineRunDataPath{
    "../../Benchmarks/ImpResults/RunResults/AMDRome48Cores_RunResults.csv"};
  constexpr std::string_view kBaselineRunDesc{"AMD Rome (48 Cores)"};*/

  /** @brief Define subsets for evaluating run results on specified inputs */
  const std::vector<std::pair<std::string, std::vector<InputSignature>>> kEvalDataSubsets{};
};

#endif /* BNCHMRKS_EVALUATION_INPUTS_H_ */
