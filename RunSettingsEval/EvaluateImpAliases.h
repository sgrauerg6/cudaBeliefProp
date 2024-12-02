/*
 * EvaluateImpAliases.h
 *
 *  Created on: December 2, 2024
 *      Author: scott
 * 
 *  Header for defining aliases for storing run results and evaluating implementations.
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
#include "RunSettings.h"
#include "RunData.h"
#include "EvalInputSignature.h"

using MultRunData = std::map<EvalInputSignature, std::optional<std::map<run_environment::ParallelParamsSetting, RunData>>>;
using RunSpeedupAvgMedian = std::pair<std::string, std::array<double, 2>>;
using MultRunDataWSpeedupByAcc =
  std::unordered_map<run_environment::AccSetting, std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>>>;

#endif //EVALUATE_IMP_ALIASES_H_