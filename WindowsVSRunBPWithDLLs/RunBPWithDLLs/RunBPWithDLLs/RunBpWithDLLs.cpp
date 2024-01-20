// BeliefPropVSUseDLLs.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define NOMINMAX
#include <memory>
#include <fstream>
#include <windows.h>
#include <iostream>
#include "BpConstsAndParams/bpStereoParameters.h"

//needed for consts and functions for running bp using DLLs
#include "GetDllFuncts/RunBpWithDLLsHelpers.h"

//needed to run the implementation a stereo set using CUDA
#include "MainDriverFiles/RunAndEvaluateBpResults.h"

const std::string BP_RUN_OUTPUT_FILE{ "output.txt" };
const std::string BP_ALL_RUNS_OUTPUT_CSV_FILE{ "outputResults.csv" };
enum class Implementation { OPTIMIZED_CPU, CUDA};
constexpr Implementation IMP_TO_RUN{Implementation::CUDA};

template<typename T, unsigned int NUM_SET>
void runBpOnSetAndUpdateResultsCUDA(const std::string& dataTypeName, std::map<std::string, std::vector<std::string>>& resultsAcrossRuns,
  const bool isTemplatedDispVals)
{
  //get mapping of device config to factory function to retrieve run stereo set object for device config
  auto runBpFactoryFuncts = RunBpWithDLLsHelpers::getRunBpFactoryFuncts<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>(NUM_SET);

  std::ofstream resultsStream(BP_RUN_OUTPUT_FILE, std::ofstream::out);

  std::array<std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>, 2> runBpStereo = {
      std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>(runBpFactoryFuncts[run_bp_dlls::device_run::CUDA]()),
      std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>(runBpFactoryFuncts[run_bp_dlls::device_run::SINGLE_THREAD_CPU]())
  };

  //load all the BP default settings as set in bpStereoCudaParameters.cuh
  BPsettings algSettings;
  algSettings.numDispVals_ = bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET];

  resultsStream << "DataType:" << dataTypeName << std::endl;
  if (isTemplatedDispVals) {
    RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>(
      resultsStream, runBpStereo, NUM_SET, algSettings);
  }
  else {
    auto runBpFactoryFuncts_dispValNotTemplated = RunBpWithDLLsHelpers::getRunBpFactoryFuncts<T, 0>(-1);
    std::unique_ptr<RunBpStereoSet<T, 0>> optCpuDispValsNoTemplate =
      std::unique_ptr<RunBpStereoSet<T, 0>>(runBpFactoryFuncts_dispValNotTemplated[run_bp_dlls::device_run::CUDA]());
    RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>(
      resultsStream, optCpuDispValsNoTemplate, runBpStereo[1], NUM_SET, algSettings);
  }
  resultsStream.close();

  auto resultsCurrentRun = RunAndEvaluateBpResults::getResultsMappingFromFile(BP_RUN_OUTPUT_FILE).first;
  for (auto& currRunResult : resultsCurrentRun) {
    if (resultsAcrossRuns.count(currRunResult.first)) {
      resultsAcrossRuns[currRunResult.first].push_back(currRunResult.second);
    }
    else {
      resultsAcrossRuns[currRunResult.first] = std::vector{ currRunResult.second };
    }
  }
}

template<typename T, unsigned int NUM_SET>
void runBpOnSetAndUpdateResults(const std::string& dataTypeName, std::map<std::string, std::vector<std::string>>& resultsAcrossRuns,
  const bool isTemplatedDispVals)
{
  //get mapping of device config to factory function to retrieve run stereo set object for device config
  auto runBpFactoryFuncts = RunBpWithDLLsHelpers::getRunBpFactoryFuncts<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>(NUM_SET);

  std::ofstream resultsStream(BP_RUN_OUTPUT_FILE, std::ofstream::out);

  std::array<std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>, 2> runBpStereo = {
      std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>(runBpFactoryFuncts[run_bp_dlls::device_run::OPTIMIZED_CPU]()),
      std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>(runBpFactoryFuncts[run_bp_dlls::device_run::SINGLE_THREAD_CPU]())
  };

  //load all the BP default settings as set in bpStereoCudaParameters.cuh
  BPsettings algSettings;
  algSettings.numDispVals_ = bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET];

  resultsStream << "DataType:" << dataTypeName << std::endl;
  if (isTemplatedDispVals) {
    RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>(
      resultsStream, runBpStereo, NUM_SET, algSettings);
  }
  else {
    auto runBpFactoryFuncts_dispValNotTemplated = RunBpWithDLLsHelpers::getRunBpFactoryFuncts<T, 0>(-1);
    std::unique_ptr<RunBpStereoSet<T, 0>> optCpuDispValsNoTemplate = 
      std::unique_ptr<RunBpStereoSet<T, 0>>(runBpFactoryFuncts_dispValNotTemplated[run_bp_dlls::device_run::OPTIMIZED_CPU]());
    RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>(
      resultsStream, optCpuDispValsNoTemplate, runBpStereo[1], NUM_SET, algSettings);
  }
  resultsStream.close();

  auto resultsCurrentRun = RunAndEvaluateBpResults::getResultsMappingFromFile(BP_RUN_OUTPUT_FILE).first;
  for (auto& currRunResult : resultsCurrentRun) {
    if (resultsAcrossRuns.count(currRunResult.first)) {
      resultsAcrossRuns[currRunResult.first].push_back(currRunResult.second);
    }
    else {
      resultsAcrossRuns[currRunResult.first] = std::vector{ currRunResult.second };
    }
  }
}

int main(int argc, char** argv)
{
  std::map<std::string, std::vector<std::string>> resultsAcrossRuns;
  if constexpr (IMP_TO_RUN == Implementation::OPTIMIZED_CPU) {
    runBpOnSetAndUpdateResults<float, 0>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float, 0>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<float, 1>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float, 1>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<float, 2>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float, 2>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<float, 3>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float, 3>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<float, 4>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float, 4>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<float, 5>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float, 5>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<float, 6>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float, 6>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<double, 0>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<double, 0>("DOUBLE", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<double, 1>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<double, 1>("DOUBLE", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<double, 2>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<double, 2>("DOUBLE", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<double, 3>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<double, 3>("DOUBLE", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<double, 4>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<double, 4>("DOUBLE", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<double, 5>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<double, 5>("DOUBLE", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<double, 6>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<double, 6>("DOUBLE", resultsAcrossRuns, false);
#ifdef COMPILING_FOR_ARM
    runBpOnSetAndUpdateResults<float16_t, 0>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float16_t, 0>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<float16_t, 1>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float16_t, 1>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<float16_t, 2>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float16_t, 2>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<float16_t, 3>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float16_t, 3>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<float16_t, 4>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float16_t, 4>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<float16_t, 5>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float16_t, 5>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<float16_t, 6>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<float16_t, 6>("HALF", resultsAcrossRuns, false);
#else
    runBpOnSetAndUpdateResults<short, 0>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<short, 0>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<short, 1>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<short, 1>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<short, 2>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<short, 2>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<short, 3>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<short, 3>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<short, 4>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<short, 4>("HALF", resultsAcrossRuns, false);
     runBpOnSetAndUpdateResults<short, 5>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<short, 5>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResults<short, 6>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResults<short, 6>("HALF", resultsAcrossRuns, false);
#endif //COMPILING_FOR_ARM
  }
  else if constexpr (IMP_TO_RUN == Implementation::CUDA) {
    runBpOnSetAndUpdateResultsCUDA<float, 0>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float, 0>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<float, 1>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float, 1>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<float, 2>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float, 2>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<float, 3>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float, 3>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<float, 4>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float, 4>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<float, 5>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float, 5>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<float, 6>("FLOAT", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float, 6>("FLOAT", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<double, 0>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<double, 0>("DOUBLE", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<double, 1>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<double, 1>("DOUBLE", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<double, 2>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<double, 2>("DOUBLE", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<double, 3>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<double, 3>("DOUBLE", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<double, 4>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<double, 4>("DOUBLE", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<double, 5>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<double, 5>("DOUBLE", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<double, 6>("DOUBLE", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<double, 6>("DOUBLE", resultsAcrossRuns, false);
#ifdef COMPILING_FOR_ARM
    runBpOnSetAndUpdateResultsCUDA<float16_t, 0>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<, 0>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<float16_t, 1>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float16_t, 1>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<float16_t, 2>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float16_t, 2>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<float16_t, 3>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float16_t, 3>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<float16_t, 4>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float16_t, 4>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<float16_t, 5>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float16_t, 5>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<float16_t, 6>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<float16_t, 6>("HALF", resultsAcrossRuns, false);
#else
    runBpOnSetAndUpdateResultsCUDA<short, 0>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<short, 0>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<short, 1>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<short, 1>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<short, 2>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<short, 2>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<short, 3>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<short, 3>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<short, 4>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<short, 4>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<short, 5>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<short, 5>("HALF", resultsAcrossRuns, false);
    runBpOnSetAndUpdateResultsCUDA<short, 6>("HALF", resultsAcrossRuns, true);
    runBpOnSetAndUpdateResultsCUDA<short, 6>("HALF", resultsAcrossRuns, false);
  }
#endif //COMPILING_FOR_ARM

  const auto headersInOrder = RunAndEvaluateBpResults::getResultsMappingFromFile(BP_RUN_OUTPUT_FILE).second;

  std::ofstream resultsStream(BP_ALL_RUNS_OUTPUT_CSV_FILE);
  for (const auto& currHeader : headersInOrder) {
    resultsStream << currHeader << ",";
  }
  resultsStream << std::endl;

  for (unsigned int i = 0; i < resultsAcrossRuns.begin()->second.size(); i++) {
    for (auto& currHeader : headersInOrder) {
      resultsStream << resultsAcrossRuns[currHeader][i] << ",";
    }
    resultsStream << std::endl;
  }
  resultsStream.close();

  std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run in "
    << BP_ALL_RUNS_OUTPUT_CSV_FILE << std::endl;
}
