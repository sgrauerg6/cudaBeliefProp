/*
 * RunEvalUtils.h
 *
 *  Created on: Jan 19, 2024
 *      Author: scott
 */

#ifndef RUN_EVAL_UTILS_H
#define RUN_EVAL_UTILS_H

#include <memory>
#include <array>
#include <fstream>
#include <vector>
#include <algorithm>
#include <optional>
#include <ranges>
#include "RunTypeConstraints.h"
#include "RunEvalConstsEnums.h"
#include "RunSettings.h"
#include "RunData.h"

//parameters type requires runData() function to return the parameters as a
//RunData object
template <typename T>
concept Params_t =
  requires(T t) {
    { t.runData() } -> std::same_as<RunData>;
  };

namespace run_eval {

//get current run inputs and parameters in RunData structure
template<RunData_t T, Params_t U, unsigned int LOOP_ITERS_TEMPLATE_OPTIMIZED, run_environment::AccSetting ACCELERATION>
RunData inputAndParamsRunData(const U& algSettings);
};

//get current run inputs and parameters in RunData structure
template<RunData_t T, Params_t U, unsigned int LOOP_ITERS_TEMPLATE_OPTIMIZED, run_environment::AccSetting ACCELERATION>
inline RunData run_eval::inputAndParamsRunData(const U& algSettings) {
  RunData currRunData;
  currRunData.addDataWHeader("DataType", run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T)));
  currRunData.appendData(algSettings.runData());
  currRunData.appendData(run_environment::runSettings<ACCELERATION>());
  currRunData.addDataWHeader("LOOP_ITERS_TEMPLATED",
                            (LOOP_ITERS_TEMPLATE_OPTIMIZED == 0) ? "NO" : "YES");
  return currRunData;
}

#endif //RUN_EVAL_UTILS_H