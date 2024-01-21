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
#include "BpConstsAndParams/bpStereoParameters.h"
#include "RunTypeConstraints.h"
#include "RunEvalConstsEnums.h"
#include "RunSettings.h"
#include "RunData.h"

using MultRunData = std::vector<std::pair<run_eval::Status, std::vector<RunData>>>;
using MultRunSpeedup = std::pair<std::string, std::array<double, 2>>;

namespace run_eval {

//get current run inputs and parameters in RunData structure
template<RunData_t T, unsigned int NUM_SET, unsigned int DISP_VALS_TEMPLATE_OPTIMIZED, run_environment::AccSetting ACC_SETTING>
RunData inputAndParamsRunData(const beliefprop::BPsettings& algSettings) {
  RunData currRunData;
  currRunData.addDataWHeader("DataType", run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T)));
  currRunData.addDataWHeader("Stereo Set", bp_params::STEREO_SET[NUM_SET]);
  currRunData.appendData(algSettings.runData());
  currRunData.appendData(run_environment::runSettings<ACC_SETTING>());
  currRunData.addDataWHeader("DISP_VALS_TEMPLATED",
                            (DISP_VALS_TEMPLATE_OPTIMIZED == 0) ? "NO" : "YES");
  return currRunData;
}

std::pair<std::string, std::vector<double>> getBaselineRuntimeData(BaselineData baselineDataType) {
  std::ifstream baselineData((baselineDataType == BaselineData::SINGLE_THREAD) ?
                             std::string(SINGLE_THREAD_BASELINE_RUNTIMES_FILE_PATH) :
                             std::string(BASELINE_RUNTIMES_FILE_PATH));
  std::string line;
  //first line of data is string with baseline processor description and all subsequent data is runtimes
  //on that processor in same order as runtimes from runBpOnStereoSets() function
  std::pair<std::string, std::vector<double>> baselineNameData;
  bool firstLine{true};
  while (std::getline(baselineData, line))
  {
    if (firstLine) {
      baselineNameData.first = line;
      firstLine = false;
    }
    else {
      baselineNameData.second.push_back(std::stod(line));
    }
  }

  return baselineNameData;
}

//get average and median speedup from vector of speedup values
std::array<double, 2> getAvgMedSpeedup(std::vector<double>& speedupsVect) {
  const double averageSpeedup = (std::accumulate(speedupsVect.begin(), speedupsVect.end(), 0.0) / (double)speedupsVect.size());
  std::sort(speedupsVect.begin(), speedupsVect.end());
  const double medianSpeedup = ((speedupsVect.size() % 2) == 0) ? 
    (speedupsVect[(speedupsVect.size() / 2) - 1] + speedupsVect[(speedupsVect.size() / 2)]) / 2.0 : 
    speedupsVect[(speedupsVect.size() / 2)];
  return {averageSpeedup, medianSpeedup};
}

//get average and median speedup of current runs compared to baseline data from file
std::vector<MultRunSpeedup> getAvgMedSpeedupOverBaseline(MultRunData& runOutput,
  const std::string& dataTypeStr) {
  std::vector<double> speedupsVect;
  const auto baselineRunData = getBaselineRuntimeData(BaselineData::OPTIMIZED);
  std::string speedupHeader = "Speedup relative to " + baselineRunData.first + " - " + dataTypeStr;
  const auto baselineRuntimes = baselineRunData.second;
  std::vector<MultRunSpeedup> speedupData;
  for (unsigned int i=0; i < runOutput.size(); i++) {
    if (runOutput[i].first == run_eval::Status::NO_ERROR) {
      speedupsVect.push_back(baselineRuntimes[i] / std::stod(runOutput[i].second[1].getData(std::string(OPTIMIZED_RUNTIME_HEADER))));
      runOutput[i].second[0].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
      runOutput[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
    }
  }
  if (speedupsVect.size() > 0) {
    speedupData.push_back({speedupHeader, getAvgMedSpeedup(speedupsVect)});
    speedupsVect.clear();
  }
  else {
    return {MultRunSpeedup()};
  }
  speedupHeader = "Single-Thread (Orig Imp) speedup relative to " + baselineRunData.first + " - " + dataTypeStr;
  const auto baselineRunDataSThread = getBaselineRuntimeData(BaselineData::SINGLE_THREAD);
  const auto baselineRuntimesSThread = baselineRunDataSThread.second;
  for (unsigned int i=0; i < runOutput.size(); i++) {
    if (runOutput[i].first == run_eval::Status::NO_ERROR) {
      speedupsVect.push_back(baselineRuntimesSThread[i] / std::stod(runOutput[i].second[1].getData(std::string(SINGLE_THREAD_RUNTIME_HEADER))));
      runOutput[i].second[0].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
      runOutput[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
    }
  }
  if (speedupsVect.size() > 0) {
    speedupData.push_back({speedupHeader, getAvgMedSpeedup(speedupsVect)});
    speedupsVect.clear();
  }
  //if processing floats, also get results for 3 smallest and 3 largest stereo sets
  if (dataTypeStr == run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(float))) {
    speedupHeader = "Speedup relative to " + baselineRunData.first + " on 3 smallest stereo sets - " + dataTypeStr;
    for (unsigned int i=0; i < 6; i++) {
      if (runOutput[i].first == run_eval::Status::NO_ERROR) {
        speedupsVect.push_back(baselineRuntimes[i] / std::stod(runOutput[i].second[1].getData(std::string(OPTIMIZED_RUNTIME_HEADER))));
        runOutput[i].second[0].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
        runOutput[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
      }
    }
    if (speedupsVect.size() > 0) {
      speedupData.push_back({speedupHeader, getAvgMedSpeedup(speedupsVect)});
      speedupsVect.clear();
    }
#ifndef SMALLER_SETS_ONLY
    speedupHeader = "Speedup relative to " + baselineRunData.first + " on 3 largest stereo sets - " + dataTypeStr;
    for (unsigned int i=9; i < 15; i++) {
#else
    speedupHeader = "Speedup relative to " + baselineRunData.first + " on largest stereo set - " + dataTypeStr;
    for (unsigned int i=9; i < 11; i++) {
#endif //SMALLER_SETS_ONLY
      if (runOutput[i].first == run_eval::Status::NO_ERROR) {
        speedupsVect.push_back(baselineRuntimes[i] / std::stod(runOutput[i].second[1].getData(std::string(OPTIMIZED_RUNTIME_HEADER))));
        runOutput[i].second[0].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
        runOutput[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
      }
    }
    if (speedupsVect.size() > 0) {
      speedupData.push_back({speedupHeader, getAvgMedSpeedup(speedupsVect)});
    }
  }
  return speedupData;
}

//get average and median speedup using optimized parallel parameters compared to default parallel parameters
MultRunSpeedup getAvgMedSpeedupOptPParams(MultRunData& runOutput,
  const std::string& speedupHeader) {
  std::vector<double> speedupsVect;
  for (unsigned int i=0; i < runOutput.size(); i++) {
    if (runOutput[i].first == run_eval::Status::NO_ERROR) {
      speedupsVect.push_back(std::stod(runOutput[i].second[0].getData(std::string(OPTIMIZED_RUNTIME_HEADER))) / 
                             std::stod(runOutput[i].second[1].getData(std::string(OPTIMIZED_RUNTIME_HEADER))));
      runOutput[i].second[0].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
      runOutput[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
    }
  }
  if (speedupsVect.size() > 0) {
    return {speedupHeader, getAvgMedSpeedup(speedupsVect)};
  }
  return {speedupHeader, {0.0, 0.0}};
}

MultRunSpeedup getAvgMedSpeedup(MultRunData& runOutputBase, MultRunData& runOutputTarget,
  const std::string& speedupHeader) {
  std::vector<double> speedupsVect;
  for (unsigned int i=0; i < runOutputBase.size(); i++) {
    if ((runOutputBase[i].first == run_eval::Status::NO_ERROR) && (runOutputTarget[i].first == run_eval::Status::NO_ERROR))  {
      speedupsVect.push_back(std::stod(runOutputBase[i].second.back().getData(std::string(OPTIMIZED_RUNTIME_HEADER))) / 
                             std::stod(runOutputTarget[i].second.back().getData(std::string(OPTIMIZED_RUNTIME_HEADER))));
      runOutputBase[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
      runOutputTarget[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
    }
  }
  if (speedupsVect.size() > 0) {
    return {speedupHeader, getAvgMedSpeedup(speedupsVect)};
  }
  return {speedupHeader, {0.0, 0.0}};
}

MultRunSpeedup getAvgMedSpeedupDispValsInTemplate(MultRunData& runOutput,
  const std::string& speedupHeader) {
  std::vector<double> speedupsVect;
  //assumine that runs with and without disparity count given in template parameter are consectutive with the run with the
  //disparity count given in template being first
  for (unsigned int i=0; (i+1) < runOutput.size(); i+=2) {
    if ((runOutput[i].first == run_eval::Status::NO_ERROR) && (runOutput[i+1].first == run_eval::Status::NO_ERROR))  {
      speedupsVect.push_back(std::stod(runOutput[i+1].second.back().getData(std::string(OPTIMIZED_RUNTIME_HEADER))) / 
                             std::stod(runOutput[i].second.back().getData(std::string(OPTIMIZED_RUNTIME_HEADER))));
      runOutput[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
    }
  }
  if (speedupsVect.size() > 0) {
    return {speedupHeader, getAvgMedSpeedup(speedupsVect)};
  }
  return {speedupHeader, {0.0, 0.0}};
}

//write data for file corresponding to runs for a specified data type or across all data type
//includes results for each run as well as average and median speedup data across multiple runs
template <run_environment::AccSetting OPT_IMP_ACCEL, bool MULT_DATA_TYPES, RunData_t T = float>
void writeRunOutput(const std::pair<MultRunData, std::vector<MultRunSpeedup>>& runOutput) {
  //get iterator to first run with success
  const auto firstSuccessRun = std::find_if(runOutput.first.begin(), runOutput.first.end(), [](const auto& runResult)
    { return (runResult.first == run_eval::Status::NO_ERROR); } );
  
  //check if there was at least one successful run
  if (firstSuccessRun != runOutput.first.end()) {
    //write results from default and optimized parallel parameters runs to csv file
    const std::string dataTypeStr = MULT_DATA_TYPES ? "MULT_DATA_TYPES" : run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T));
    const auto accelStr = run_environment::accelerationString<OPT_IMP_ACCEL>();
    const std::string optResultsFileName{std::string(BP_ALL_RUNS_OUTPUT_CSV_FILE_NAME_START) + "_" + 
      (PROCESSOR_NAME.size() > 0 ? std::string(PROCESSOR_NAME) + "_" : "") + dataTypeStr + "_" + accelStr + std::string(CSV_FILE_EXTENSION)};
    const std::string defaultParamsResultsFileName{std::string(BP_ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE_START) + "_" +
      (PROCESSOR_NAME.size() > 0 ? std::string(PROCESSOR_NAME) + "_" : "") + dataTypeStr + "_" + accelStr + std::string(CSV_FILE_EXTENSION)};
    std::array<std::ofstream, 2> resultsStreamDefaultTBFinal{
      std::ofstream(OPTIMIZE_PARALLEL_PARAMS ? defaultParamsResultsFileName : optResultsFileName),
      OPTIMIZE_PARALLEL_PARAMS ? std::ofstream(optResultsFileName) : std::ofstream()};
    //get headers from first successful run
    const auto headersInOrder = firstSuccessRun->second.back().getHeadersInOrder();
    for (const auto& currHeader : headersInOrder) {
      resultsStreamDefaultTBFinal[0] << currHeader << ",";
    }
    resultsStreamDefaultTBFinal[0] << std::endl;

    if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
      for (const auto& currHeader : headersInOrder) {
        resultsStreamDefaultTBFinal[1] << currHeader << ",";
      }
    }
    resultsStreamDefaultTBFinal[1] << std::endl;

    for (unsigned int i=0; i < (OPTIMIZE_PARALLEL_PARAMS ? resultsStreamDefaultTBFinal.size() : 1); i++) {
      for (unsigned int runNum=0; runNum < runOutput.first.size(); runNum++) {
        //if run not successful only have single set of output data from run
        const unsigned int runResultIdx = (runOutput.first[runNum].first == run_eval::Status::NO_ERROR) ? i : 0;
        for (const auto& currHeader : headersInOrder) {
          if (!(runOutput.first[runNum].second[runResultIdx].isData(currHeader))) {
            resultsStreamDefaultTBFinal[i] << "No Data" << ",";
          }
          else {
            resultsStreamDefaultTBFinal[i] << runOutput.first[runNum].second[runResultIdx].getData(currHeader) << ",";
          }
        }
        resultsStreamDefaultTBFinal[i] << std::endl;
      }
    }

    //write speedup results
    const unsigned int indexBestResults{(OPTIMIZE_PARALLEL_PARAMS ? resultsStreamDefaultTBFinal.size() - 1 : 0)};
    resultsStreamDefaultTBFinal[indexBestResults] << std::endl << ",Average Speedup,Median Speedup" << std::endl;
    for (const auto& speedup : runOutput.second) {
      resultsStreamDefaultTBFinal[indexBestResults] << speedup.first;
      if (speedup.second[0] > 0) {
        resultsStreamDefaultTBFinal[indexBestResults] << "," << speedup.second[0] << "," << speedup.second[1];
      }
      resultsStreamDefaultTBFinal[indexBestResults] << std::endl;      
    }

    resultsStreamDefaultTBFinal[0].close();
    if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
      resultsStreamDefaultTBFinal[1].close();
    }

    if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
      std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run (optimized parallel parameters) in "
                << optResultsFileName << std::endl;
      std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run (default parallel parameters) in "
                << defaultParamsResultsFileName << std::endl;
    }
    else {
      std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run using default parallel parameters in "
                << optResultsFileName << std::endl;
    }
  }
  else {
    std::cout << "Error, no runs completed successfully" << std::endl;
  }
}
    
}

#endif //RUN_EVAL_UTILS_H