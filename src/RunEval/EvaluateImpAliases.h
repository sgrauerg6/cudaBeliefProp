/*
Copyright (C) 2024 Scott Grauer-Gray

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
 * @file EvaluateImpAliases.h
 * @author Scott Grauer-Gray
 * @brief Header for defining aliases for storing run results and evaluating implementations.
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef EVALUATE_IMP_ALIASES_H_
#define EVALUATE_IMP_ALIASES_H_

#include <map>
#include <unordered_map>
#include <array>
#include <vector>
#include <utility>
#include <string>
#include <optional>
#include "RunData.h"
#include "RunSettingsParams/InputSignature.h"
#include "RunEval/RunEvalConstsEnums.h"

using MultRunData = std::map<InputSignature, std::optional<std::map<run_environment::ParallelParamsSetting, RunData>>>;
using RunSpeedupAvgMedian = std::pair<std::string, std::map<run_eval::MiddleValData, double>>;
using MultRunDataWSpeedupByAcc =
  std::unordered_map<run_environment::AccSetting, std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>>>;

#endif //EVALUATE_IMP_ALIASES_H_