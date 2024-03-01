/*
 * EvaluateImpResults.cpp
 *
 *  Created on: Feb 20, 2024
 *      Author: scott
 * 
 *  Function definitions for class to evaluate implementation results.
 */

#include "EvaluateImpResults.h"
#include "CombineMultResultSets.h"
#include <fstream>
#include <numeric>
#include <sstream>

//evaluate results for implementation runs on multiple inputs with all the runs having the same data type and acceleration method
void EvaluateImpResults::operator()(const MultRunData& runResults, const run_environment::RunImpSettings runImpSettings, run_environment::AccSetting optImpAcc, size_t dataSize) {
  runImpOrigResults_ = runResults;
  runImpSettings_ = runImpSettings;
  optImpAccel_ = optImpAcc;
  dataSize_ = dataSize;
  evalResultsSingDTypeAccRun();
}

//evaluate results for implementation runs on multiple inputs with the runs having different data type and acceleration methods
void EvaluateImpResults::operator()(const std::unordered_map<size_t, std::unordered_map<run_environment::AccSetting, std::pair<MultRunData, std::vector<MultRunSpeedup>>>>& runResultsMultRuns,
  const run_environment::RunImpSettings runImpSettings, run_environment::AccSetting optImpAcc)
{
  runImpResultsMultRuns_ = runResultsMultRuns;
  runImpSettings_ = runImpSettings;
  optImpAccel_ = optImpAcc;
  evalResultsMultDTypeAccRuns();
}

std::pair<MultRunData, std::vector<MultRunSpeedup>> EvaluateImpResults::getRunDataWSpeedups() const {
  return {runImpOptResults_, runImpSpeedups_};
}

//write data for file corresponding to runs for a specified data type or across all data type
//includes results for each run as well as average and median speedup data across multiple runs
template <bool MULT_DATA_TYPES>
void EvaluateImpResults::writeRunOutput(const std::pair<MultRunData, std::vector<MultRunSpeedup>>& runOutput, const run_environment::RunImpSettings& runImpSettings,
  run_environment::AccSetting accelerationSetting, const unsigned int dataTypeSize)
{
  //get iterator to first run with success
  const auto firstSuccessRun = std::find_if(runOutput.first.begin(), runOutput.first.end(), [](const auto& runResult)
    { return runResult; } );

  //check if there was at least one successful run
  if (firstSuccessRun != runOutput.first.end()) {
    //write results from default and optimized parallel parameters runs to csv file
    //file name contains info about data type, parameter settings, and processor name if available
    //only show data type string and acceleration string for runs using a single data type that are used for debugging (not multidata type results) 
    const std::string dataTypeStr = MULT_DATA_TYPES ? "" : '_' + std::string(run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize));
    const auto accelStr = MULT_DATA_TYPES ? "" : '_' + run_environment::accelerationString(accelerationSetting);
    const auto impResultsFp = getImpResultsPath();
    const std::filesystem::path optResultsFilePath{impResultsFp / run_eval::IMP_RESULTS_RUN_DATA_FOLDER_NAME / std::filesystem::path(((runImpSettings.runName_) ? std::string(runImpSettings.runName_.value()) + "_" : "") + 
      std::string(run_eval::RUN_RESULTS_DESCRIPTION_FILE_NAME) + dataTypeStr + accelStr + std::string(run_eval::CSV_FILE_EXTENSION))};
    const std::filesystem::path optResultsWSpeedupFilePath{impResultsFp / run_eval::IMP_RESULTS_RUN_DATA_W_SPEEDUPS_FOLDER_NAME / std::filesystem::path(((runImpSettings.runName_) ? std::string(runImpSettings.runName_.value()) + "_" : "") + 
      std::string(run_eval::RUN_RESULTS_W_SPEEDUPS_DESCRIPTION_FILE_NAME) + dataTypeStr + accelStr + std::string(run_eval::CSV_FILE_EXTENSION))};
    const std::filesystem::path defaultParamsResultsFilePath{impResultsFp / run_eval::IMP_RESULTS_RUN_DATA_FOLDER_NAME / std::filesystem::path(((runImpSettings.runName_) ? std::string(runImpSettings.runName_.value()) + "_" : "") + 
      std::string(run_eval::RUN_RESULTS_DESCRIPTION_DEFAULT_P_PARAMS_FILE_NAME) + dataTypeStr + accelStr + std::string(run_eval::CSV_FILE_EXTENSION))};
    const std::filesystem::path speedupResultsFilePath{impResultsFp / run_eval::IMP_RESULTS_SPEEDUPS_FOLDER_NAME / std::filesystem::path(((runImpSettings.runName_) ? std::string(runImpSettings.runName_.value()) + "_" : "") + 
      std::string(run_eval::SPEEDUPS_DESCRIPTION_FILE_NAME) + dataTypeStr + accelStr + std::string(run_eval::CSV_FILE_EXTENSION))};
    std::array<std::ostringstream, 2> runDataOptDefaultSStr;
    std::array<std::ostringstream, 2> speedupsHeadersLeftTopSStr;
        
    //get headers from first successful run and write headers to top of output files
    const auto headersInOrder = (*firstSuccessRun)->back().getHeadersInOrder();
    for (unsigned int i=0; i < (runImpSettings.optParallelParamsOptionSetting_.first ? runDataOptDefaultSStr.size() : 1); i++) {
      for (const auto& currHeader : headersInOrder) {
        runDataOptDefaultSStr[i] << currHeader << ',';
      }
      runDataOptDefaultSStr[i] << std::endl;
    }

    //write output for run on each input with each data type
    for (unsigned int i=0; i < (runImpSettings.optParallelParamsOptionSetting_.first ? runDataOptDefaultSStr.size() : 1); i++) {
      for (unsigned int runNum=0; runNum < runOutput.first.size(); runNum++) {
        //if run not successful only have single set of output data from run
        const unsigned int runResultIdx = runOutput.first[runNum] ? i : 0;
        for (const auto& currHeader : headersInOrder) {
          if (!(runOutput.first[runNum]->at(runResultIdx).isData(currHeader))) {
            runDataOptDefaultSStr[i] << "No Data" << ',';
          }
          else {
            runDataOptDefaultSStr[i] << runOutput.first[runNum]->at(runResultIdx).getData(currHeader) << ',';
          }
        }
        runDataOptDefaultSStr[i] << std::endl;
      }
    }

    //generate speedup results with headers on left side and with headers on top row
    speedupsHeadersLeftTopSStr[0] << std::endl << ",Average Speedup,Median Speedup" << std::endl;
    for (const auto& speedup : runOutput.second) {
      speedupsHeadersLeftTopSStr[0] << speedup.first;
      if (speedup.second[0] > 0) {
        speedupsHeadersLeftTopSStr[0] << ',' << speedup.second[0] << ',' << speedup.second[1];
      }
      speedupsHeadersLeftTopSStr[0] << std::endl;
    }
    speedupsHeadersLeftTopSStr[1] << ',';
    for (const auto& speedup : runOutput.second) {
      speedupsHeadersLeftTopSStr[1] << speedup.first << ',';
    }
    for (const auto& speedupDescWIndex : {std::pair<std::string, size_t>{"Average Speedup", 0}, std::pair<std::string, size_t>{"Median Speedup", 1}}) {
      speedupsHeadersLeftTopSStr[1] << std::endl << speedupDescWIndex.first << ',';
      for (const auto& speedup : runOutput.second) {
        speedupsHeadersLeftTopSStr[1] << speedup.second[speedupDescWIndex.second] << ',';
      }
    }
    
    //write run results strings to output streams
    //one results file contains only speedup results, another contains only run results, and a third contains run results followed by speedups
    std::ofstream speedupResultsStr{speedupResultsFilePath};
    speedupResultsStr << speedupsHeadersLeftTopSStr[1].str();
    std::array<std::ofstream, 2> resultsStreamDefaultTBFinal{
      runImpSettings.optParallelParamsOptionSetting_.first ? (writeDebugOutputFiles_ ? std::ofstream(defaultParamsResultsFilePath) : std::ofstream()) : std::ofstream(optResultsFilePath),
      runImpSettings.optParallelParamsOptionSetting_.first ? std::ofstream(optResultsFilePath) : std::ofstream()};
    std::ofstream runResultWSpeedupsStr(optResultsWSpeedupFilePath);
    if (writeDebugOutputFiles_ || (!runImpSettings.optParallelParamsOptionSetting_.first)) {
      //only write file with default params when there are results with optimized parameters when writing debug files
      resultsStreamDefaultTBFinal[0] << runDataOptDefaultSStr[0].str();
    }
    if (runImpSettings.optParallelParamsOptionSetting_.first) {
      resultsStreamDefaultTBFinal[1] << runDataOptDefaultSStr[1].str();
      runResultWSpeedupsStr << runDataOptDefaultSStr[1].str() << std::endl;
    }
    else {
      runResultWSpeedupsStr << runDataOptDefaultSStr[0].str() << std::endl;        
    }
    //add speedups with headers on left to file containing run results and speedups
    runResultWSpeedupsStr << "Speedup Results" << std::endl << speedupsHeadersLeftTopSStr[0].str();

    std::cout << "Input/settings/parameters info, detailed timings, and evaluation for each run and across runs in " << optResultsWSpeedupFilePath << std::endl;
    std::cout << "Run inputs and results in " << optResultsFilePath << std::endl;
    std::cout << "Speedup results in " << speedupResultsFilePath << std::endl;
    CombineMultResultSets().operator()(impResultsFp);
  }
  else {
    std::cout << "Error, no runs completed successfully" << std::endl;
  }
}

void EvaluateImpResults::evalResultsSingDTypeAccRun() {
  //initialize and add speedup results over baseline data if available for current input
  runImpOptResults_ = runImpOrigResults_;
  const auto speedupOverBaseline = getSpeedupOverBaseline(runImpSettings_, runImpOptResults_, dataSize_);
  const auto speedupOverBaselineSubsets = getSpeedupOverBaselineSubsets(runImpSettings_, runImpOptResults_, dataSize_);
  runImpSpeedups_.insert(runImpSpeedups_.end(), speedupOverBaseline.begin(), speedupOverBaseline.end());
  runImpSpeedups_.insert(runImpSpeedups_.end(), speedupOverBaselineSubsets.begin(), speedupOverBaselineSubsets.end());

  //compute and add speedup info for using optimized parallel parameters and disparity count as template parameter to speedup results
  if (runImpSettings_.optParallelParamsOptionSetting_.first) {
    runImpSpeedups_.push_back(getAvgMedSpeedupOptPParams(runImpOptResults_, std::string(run_eval::SPEEDUP_OPT_PAR_PARAMS_HEADER) + " - " +
      std::string(run_environment::DATA_SIZE_TO_NAME_MAP.at(dataSize_))));
  }
  if (runImpSettings_.templatedItersSetting_ == run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED) {
    runImpSpeedups_.push_back(getAvgMedSpeedupLoopItersInTemplate(runImpOptResults_, std::string(run_eval::SPEEDUP_DISP_COUNT_TEMPLATE) + " - " +
      std::string(run_environment::DATA_SIZE_TO_NAME_MAP.at(dataSize_))));
  }

  //write output corresponding to results for current data type if writing debug output
  if (writeDebugOutputFiles_) {
    constexpr bool MULT_DATA_TYPES{false};
    writeRunOutput<MULT_DATA_TYPES>({runImpOptResults_, runImpSpeedups_}, runImpSettings_, optImpAccel_, dataSize_);
  }
}

//perform runs without CPU vectorization and get speedup for each run and overall when using vectorization
//CPU vectorization does not apply to CUDA acceleration so "NO_DATA" output is returned in that case
std::vector<MultRunSpeedup> EvaluateImpResults::getAltAccelSpeedups(
  std::unordered_map<run_environment::AccSetting, std::pair<MultRunData, std::vector<MultRunSpeedup>>>& runImpResultsByAccSetting,
  const run_environment::RunImpSettings& runImpSettings, size_t dataTypeSize, run_environment::AccSetting fastestAcc) const
{
  //set up mapping from acceleration type to description
  const std::map<run_environment::AccSetting, std::string> accToSpeedupStr{
    {run_environment::AccSetting::NONE, 
     std::string(run_eval::SPEEDUP_VECTORIZATION) + " - " + std::string(run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize))},
    {run_environment::AccSetting::AVX256, 
     std::string(run_eval::SPEEDUP_VS_AVX256_VECTORIZATION) + " - " + std::string(run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize))}};

  if (runImpResultsByAccSetting.size() == 1) {
    //no alternate run results
    return {{accToSpeedupStr.at(run_environment::AccSetting::NONE), {0.0, 0.0}}, {accToSpeedupStr.at(run_environment::AccSetting::AVX256), {0.0, 0.0}}};
  }
  else {
    //initialize speedup/slowdown using alternate acceleration
    std::vector<MultRunSpeedup> altAccSpeedups;
    for (auto& altAccImpResults : runImpResultsByAccSetting) {
      if ((altAccImpResults.first != fastestAcc) && (accToSpeedupStr.contains(altAccImpResults.first))) {
        //process results using alternate acceleration
        //go through each result and replace initial run data with alternate implementation run data if alternate implementation run is faster
        for (unsigned int i = 0; i < runImpResultsByAccSetting[fastestAcc].first.size(); i++) {
          if (runImpResultsByAccSetting[fastestAcc].first[i] && altAccImpResults.second.first[i]) {
            const double initResultTime = std::stod(std::string(runImpResultsByAccSetting[fastestAcc].first[i]->back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER))));
            const double altAccResultTime = std::stod(std::string(altAccImpResults.second.first[i]->back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER))));
            if (altAccResultTime < initResultTime) {
              runImpResultsByAccSetting[fastestAcc].first[i] = altAccImpResults.second.first[i];
            }
          }
        }
        //get speedup/slowdown using alternate acceleration compared to fastest implementation and store in speedup results
        altAccSpeedups.push_back(getAvgMedSpeedup(altAccImpResults.second.first, runImpResultsByAccSetting[fastestAcc].first,
          accToSpeedupStr.at(altAccImpResults.first)));
      }
    }
    return altAccSpeedups;
  }
}

void EvaluateImpResults::evalResultsMultDTypeAccRuns() {
  //get speedup/slowdown using alternate accelerations
  std::unordered_map<size_t, std::vector<MultRunSpeedup>> altImpSpeedup;
  std::unordered_map<size_t, MultRunSpeedup> altDataTypeSpeedup;
  for (const size_t dataSize : {sizeof(float), sizeof(double), sizeof(halftype)}) {
    altImpSpeedup[dataSize] = getAltAccelSpeedups(runImpResultsMultRuns_[dataSize], runImpSettings_, dataSize, optImpAccel_);
    if (dataSize != sizeof(float)) {
      //get speedup or slowdown using alternate data type (double or half) compared with float
      altDataTypeSpeedup[dataSize] = getAvgMedSpeedup(runImpResultsMultRuns_[sizeof(float)][optImpAccel_].first,
        runImpResultsMultRuns_[dataSize][optImpAccel_].first, (dataSize > sizeof(float)) ? std::string(run_eval::SPEEDUP_DOUBLE) : std::string(run_eval::SPEEDUP_HALF));
    }
  }

  //initialize overall results to float results using fastest acceleration and add double and half-type results to it
  auto resultsWSpeedups = runImpResultsMultRuns_[sizeof(float)][optImpAccel_];
  resultsWSpeedups.first.insert(resultsWSpeedups.first.end(),
    runImpResultsMultRuns_[sizeof(double)][optImpAccel_].first.begin(), runImpResultsMultRuns_[sizeof(double)][optImpAccel_].first.end());
  resultsWSpeedups.first.insert(resultsWSpeedups.first.end(),
    runImpResultsMultRuns_[sizeof(halftype)][optImpAccel_].first.begin(), runImpResultsMultRuns_[sizeof(halftype)][optImpAccel_].first.end());

  //add speedup data from double and half precision runs to speedup results
  resultsWSpeedups.second.insert(resultsWSpeedups.second.end(), altImpSpeedup[sizeof(float)].begin(), altImpSpeedup[sizeof(float)].end());
  resultsWSpeedups.second.insert(resultsWSpeedups.second.end(),
    runImpResultsMultRuns_[sizeof(double)][optImpAccel_].second.begin(), runImpResultsMultRuns_[sizeof(double)][optImpAccel_].second.end());
  resultsWSpeedups.second.insert(resultsWSpeedups.second.end(), altImpSpeedup[sizeof(double)].begin(), altImpSpeedup[sizeof(double)].end());
  resultsWSpeedups.second.insert(resultsWSpeedups.second.end(),
    runImpResultsMultRuns_[sizeof(halftype)][optImpAccel_].second.begin(), runImpResultsMultRuns_[sizeof(halftype)][optImpAccel_].second.end());
  resultsWSpeedups.second.insert(resultsWSpeedups.second.end(), altImpSpeedup[sizeof(halftype)].begin(), altImpSpeedup[sizeof(halftype)].end());

  //get speedup over baseline runtimes...can only compare with baseline runtimes that are
  //generated using same templated iterations setting as current run
  if ((runImpSettings_.baseOptSingThreadRTimeForTSetting_) &&
      (runImpSettings_.baseOptSingThreadRTimeForTSetting_.value().second == runImpSettings_.templatedItersSetting_)) {
      const auto speedupOverBaseline = getAvgMedSpeedupOverBaseline(resultsWSpeedups.first, "All Runs",
        runImpSettings_.baseOptSingThreadRTimeForTSetting_.value().first);
      resultsWSpeedups.second.insert(resultsWSpeedups.second.end(), speedupOverBaseline.begin(), speedupOverBaseline.end());
  }

  //get speedup info for using optimized parallel parameters
  if (runImpSettings_.optParallelParamsOptionSetting_.first) {
    resultsWSpeedups.second.push_back(getAvgMedSpeedupOptPParams(
    resultsWSpeedups.first, std::string(run_eval::SPEEDUP_OPT_PAR_PARAMS_HEADER) + " - All Runs"));
  }

  //get speedup when using template for loop iteration count
  if (runImpSettings_.templatedItersSetting_ == run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED) {
    resultsWSpeedups.second.push_back(getAvgMedSpeedupLoopItersInTemplate(
    resultsWSpeedups.first, std::string(run_eval::SPEEDUP_DISP_COUNT_TEMPLATE) + " - All Runs"));
  }

  //add speedups when using doubles and half precision compared to float to end of speedup data
  resultsWSpeedups.second.insert(resultsWSpeedups.second.end(), {altDataTypeSpeedup[sizeof(double)], altDataTypeSpeedup[sizeof(halftype)]});

  //write output corresponding to results and speedups for all data types
  constexpr bool MULT_DATA_TYPES{true};
  writeRunOutput<MULT_DATA_TYPES>(resultsWSpeedups, runImpSettings_, optImpAccel_);
}

//get speedup over baseline data if data available
std::vector<MultRunSpeedup> EvaluateImpResults::getSpeedupOverBaseline(const run_environment::RunImpSettings& runImpSettings,
  MultRunData& runDataAllRuns, const size_t dataTypeSize) const
{
  //initialize speedup results
  std::vector<MultRunSpeedup> speedupResults;

  //only get speedup over baseline when processing float data type since that is run first and corresponds to the data at the top
  //of the baseline data
  if (dataTypeSize == sizeof(float)) {
    //get speedup over baseline runtimes...can only compare with baseline runtimes that are
    //generated using same templated iterations setting as current run
    if ((runImpSettings.baseOptSingThreadRTimeForTSetting_) &&
        (runImpSettings.baseOptSingThreadRTimeForTSetting_.value().second == runImpSettings.templatedItersSetting_))
    {
      const auto speedupOverBaselineSubsets = getAvgMedSpeedupOverBaseline(
        runDataAllRuns, run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize),
        runImpSettings.baseOptSingThreadRTimeForTSetting_.value().first);
      speedupResults.insert(speedupResults.end(), speedupOverBaselineSubsets.begin(), speedupOverBaselineSubsets.end());
    }
  }
  return speedupResults;
}

//get speedup over baseline data for belief propagation run for subsets of smallest and largest sets if data available
std::vector<MultRunSpeedup> EvaluateImpResults::getSpeedupOverBaselineSubsets(const run_environment::RunImpSettings& runImpSettings,
  MultRunData& runDataAllRuns, const size_t dataTypeSize) const
{
  if ((dataTypeSize == sizeof(float)) &&
    (runImpSettings.baseOptSingThreadRTimeForTSetting_ && (runImpSettings.baseOptSingThreadRTimeForTSetting_->second == runImpSettings.templatedItersSetting_)))
  {
    return getAvgMedSpeedupOverBaselineSubsets(runDataAllRuns, run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize),
      runImpSettings.baseOptSingThreadRTimeForTSetting_->first, runImpSettings.subsetStrIndices_);
  }
  //return empty vector if doesn't match settings to get speedup over baseline for subsets
  return std::vector<MultRunSpeedup>();
}

//get baseline runtime data if available...return null if baseline data not available
std::optional<std::pair<std::string, std::vector<double>>> EvaluateImpResults::getBaselineRuntimeData(const std::string& baselineDataPath) const {
  std::ifstream baselineData(std::string{baselineDataPath});
  if (!(baselineData.is_open())) {
    return {};
  }
  std::string line;
  //first line of data is string with baseline processor description and all subsequent data is runtimes
  //on that processor in same order as runtimes from runBenchmark() function
  std::pair<std::string, std::vector<double>> baselineNameData;
  bool firstLine{true};
  while (std::getline(baselineData, line)) {
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
std::array<double, 2> EvaluateImpResults::getAvgMedSpeedup(const std::vector<double>& speedupsVect) const {
  const double averageSpeedup = (std::accumulate(speedupsVect.begin(), speedupsVect.end(), 0.0) / (double)speedupsVect.size());
  auto speedupsVectSorted = speedupsVect;
  std::ranges::sort(speedupsVectSorted);
  const double medianSpeedup = ((speedupsVectSorted.size() % 2) == 0) ? 
    (speedupsVectSorted[(speedupsVectSorted.size() / 2) - 1] + speedupsVectSorted[(speedupsVectSorted.size() / 2)]) / 2.0 : 
    speedupsVectSorted[(speedupsVectSorted.size() / 2)];
  return {averageSpeedup, medianSpeedup};
}

//get average and median speedup of specified subset(s) of runs compared to baseline data from file
std::vector<MultRunSpeedup> EvaluateImpResults::getAvgMedSpeedupOverBaselineSubsets(MultRunData& runOutput,
  const std::string& dataTypeStr, const std::array<std::string, 2>& baseDataPathOptSingThrd,
  const std::vector<std::pair<std::string, std::vector<unsigned int>>>& subsetStrIndices) const
{
  //get speedup over baseline for optimized runs
  std::vector<MultRunSpeedup> speedupData;
  const auto baselineRunData = getBaselineRuntimeData(std::string(baseDataPathOptSingThrd[0]));
  if (baselineRunData) {
    std::string speedupHeader = "Speedup relative to " + std::string((*baselineRunData).first) + " - " + std::string(dataTypeStr);
    const auto baselineRuntimes = (*baselineRunData).second;
    //retrieve speedup data for any subsets of optimized runs
    for (const auto& currSubsetStrIndices : subsetStrIndices) {
      std::vector<double> speedupsVect;
      speedupHeader = "Speedup relative to " + std::string((*baselineRunData).first) + " on " + std::string(currSubsetStrIndices.first) + " - " + std::string(dataTypeStr);
      for (unsigned int i : currSubsetStrIndices.second) {
        if (runOutput[i]) {
          speedupsVect.push_back(baselineRuntimes[i] / std::stod(std::string(runOutput[i]->at(1).getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)))));
          for (auto& runData : runOutput[i].value()) {
            runData.addDataWHeader(std::string(speedupHeader), std::to_string(speedupsVect.back()));
          }
        }
      }
      if (!(speedupsVect.empty())) {
        speedupData.push_back({speedupHeader, getAvgMedSpeedup(speedupsVect)});
      }
    }
  }

  return speedupData;
}

//get average and median speedup of current runs compared to baseline data from file
std::vector<MultRunSpeedup> EvaluateImpResults::getAvgMedSpeedupOverBaseline(MultRunData& runOutput,
  const std::string& dataTypeStr, const std::array<std::string, 2>& baselinePathOptSingThread) const
{
  //get speedup over baseline for optimized runs
  std::vector<MultRunSpeedup> speedupData;
  const auto baselineRunData = getBaselineRuntimeData(std::string(baselinePathOptSingThread[0]));
  if (baselineRunData) {
    std::vector<double> speedupsVect;
    const std::string speedupHeader = "Speedup relative to " + std::string((*baselineRunData).first) + " - " + std::string(dataTypeStr);
    const auto baselineRuntimes = (*baselineRunData).second;
    for (unsigned int i=0; i < runOutput.size(); i++) {
      if (runOutput[i]) {
        speedupsVect.push_back(baselineRuntimes[i] / std::stod(std::string(runOutput[i]->at(1).getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)))));
        for (auto& runData : runOutput[i].value()) {
          runData.addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
        }
      }
    }
    if (!(speedupsVect.empty())) {
      speedupData.push_back({speedupHeader, getAvgMedSpeedup(speedupsVect)});
    }
  }

  //get speedup over baseline for single thread runs
  const auto baselineRunDataSThread = getBaselineRuntimeData(std::string(baselinePathOptSingThread[1]));
  if (baselineRunDataSThread) {
    std::vector<double> speedupsVect;
    const std::string speedupHeader = "Single-Thread (Orig Imp) speedup relative to " + std::string((*baselineRunDataSThread).first) + " - " + std::string(dataTypeStr);
    const auto baselineRuntimesSThread = (*baselineRunDataSThread).second;
    for (unsigned int i=0; i < runOutput.size(); i++) {
      if (runOutput[i]) {
        speedupsVect.push_back(baselineRuntimesSThread[i] / std::stod(std::string(runOutput[i]->at(1).getData(std::string(run_eval::SINGLE_THREAD_RUNTIME_HEADER)))));
        for (auto& runData : runOutput[i].value()) {
          runData.addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
        }
      }
    }
    if (!(speedupsVect.empty())) {
      speedupData.push_back({speedupHeader, getAvgMedSpeedup(speedupsVect)});
    }
  }

  return speedupData;
}

//get average and median speedup using optimized parallel parameters compared to default parallel parameters
MultRunSpeedup EvaluateImpResults::getAvgMedSpeedupOptPParams(MultRunData& runOutput, const std::string& speedupHeader) const {
  std::vector<double> speedupsVect;
  for (unsigned int i=0; i < runOutput.size(); i++) {
    if (runOutput[i]) {
      speedupsVect.push_back(std::stod(std::string(runOutput[i]->at(0).getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)))) / 
                             std::stod(std::string(runOutput[i]->at(1).getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)))));
      runOutput[i]->at(0).addDataWHeader(std::string(speedupHeader), std::to_string(speedupsVect.back()));
      runOutput[i]->at(1).addDataWHeader(std::string(speedupHeader), std::to_string(speedupsVect.back()));
    }
  }
  if (!(speedupsVect.empty())) {
    return {speedupHeader, getAvgMedSpeedup(speedupsVect)};
  }
  return {speedupHeader, {0.0, 0.0}};
}

//get average and median speedup between base and target runtime data
MultRunSpeedup EvaluateImpResults::getAvgMedSpeedup(MultRunData& runOutputBase, MultRunData& runOutputTarget,
  const std::string& speedupHeader) const {
  std::vector<double> speedupsVect;
  for (unsigned int i=0; i < runOutputBase.size(); i++) {
    if (runOutputBase[i] && runOutputTarget[i])  {
      speedupsVect.push_back(std::stod(std::string(runOutputBase[i]->back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)))) / 
                             std::stod(std::string(runOutputTarget[i]->back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)))));
      runOutputBase[i]->at(1).addDataWHeader(std::string(speedupHeader), std::to_string(speedupsVect.back()));
      runOutputTarget[i]->at(1).addDataWHeader(std::string(speedupHeader), std::to_string(speedupsVect.back()));
    }
  }
  if (!(speedupsVect.empty())) {
    return {speedupHeader, getAvgMedSpeedup(speedupsVect)};
  }
  return {speedupHeader, {0.0, 0.0}};
}

//get average and median speedup when loop iterations are given at compile time as template value
MultRunSpeedup EvaluateImpResults::getAvgMedSpeedupLoopItersInTemplate(MultRunData& runOutput,
  const std::string& speedupHeader)
{
  //get mapping of run inputs to runtime with index value in run output
  std::map<std::vector<std::string>, std::pair<std::string, size_t>> runInputSettingsToTimeWIdx;
  for (unsigned int i=0; i < runOutput.size(); i++) {
    if (runOutput[i]) {
      const auto inputSettingsToTime = runOutput[i]->back().getParamsToParamRunData({"DataType", "Input Index", "LOOP_ITERS_TEMPLATED"}, std::string(run_eval::OPTIMIZED_RUNTIME_HEADER));
      if (inputSettingsToTime) {
        runInputSettingsToTimeWIdx.insert({inputSettingsToTime->first, {inputSettingsToTime->second, i}});
      }
    }
  }
  //go through all run input settings to time and get each pair that is the same in datatype and input and differs in disp values templated
  //and get speedup for each of templated compared to non-templated
  std::vector<double> speedupsVect;
  auto runDataIter = runInputSettingsToTimeWIdx.begin();
  while (runDataIter != runInputSettingsToTimeWIdx.end()) {
    auto dataTypeRun = runDataIter->first[0];
    auto inputIdxRun = runDataIter->first[1];
    auto runComp1 = runDataIter;
    auto runComp2 = runDataIter;
    //find run with same datatype and input index
    while (++runDataIter != runInputSettingsToTimeWIdx.end()) {
      if ((runDataIter->first[0] == dataTypeRun) && (runDataIter->first[1] == inputIdxRun)) {
        runComp2 = runDataIter;
        break;
      }
    }
    //if don't have two separate runs with same data type and input, erase current run from mapping and continue
    if (runComp1 == runComp2) {
      runInputSettingsToTimeWIdx.erase(runComp1);
      runDataIter = runInputSettingsToTimeWIdx.begin();
      continue;
    }
    //retrieve which run data uses templated iteration count and which one doesn't and get speedup
    //add speedup to speedup vector and also to run data of run with templated iteration count
    if ((runComp1->first[2] == "YES") && (runComp2->first[2] == "NO"))  {
      speedupsVect.push_back(std::stod(std::string(runComp2->second.first)) / std::stod(std::string(runComp1->second.first)));
      runOutput[runComp1->second.second]->back().addDataWHeader(std::string(speedupHeader), std::to_string(speedupsVect.back()));
    }
    else if ((runComp1->first[2] == "NO") && (runComp2->first[2] == "YES"))  {
      speedupsVect.push_back(std::stod(std::string(runComp1->second.first)) / std::stod(std::string(runComp2->second.first)));
      runOutput[runComp2->second.second]->back().addDataWHeader(std::string(speedupHeader), std::to_string(speedupsVect.back()));
    }
    //remove runs that have been processed from mapping
    runInputSettingsToTimeWIdx.erase(runComp1);
    runInputSettingsToTimeWIdx.erase(runComp2);
    runDataIter = runInputSettingsToTimeWIdx.begin();
  }
  if (!(speedupsVect.empty())) {
    return {speedupHeader, getAvgMedSpeedup(speedupsVect)};
  }
  return {speedupHeader, {0.0, 0.0}};
}

//retrieve path of implementation results
std::filesystem::path EvaluateImpResults::getImpResultsPath() const
{
  std::filesystem::path currentPath = std::filesystem::current_path();
  while (true) {
    //create directory iterator corresponding to current path
    std::filesystem::directory_iterator dirIt = std::filesystem::directory_iterator(currentPath);

    //check if any of the directories in the current path correspond to the implementation results directory;
    //if so return iterator to directory; otherwise return iterator to end indicating that directory not
    //found in current path
    std::filesystem::directory_iterator it = std::find_if(std::filesystem::begin(dirIt), std::filesystem::end(dirIt), 
      [](const auto &p) { return p.path().stem() == run_eval::IMP_RESULTS_FOLDER_NAME; });
    //check if return from find_if at iterator end and therefore didn't find stereo sets directory;
    //if that's the case continue to outer directory
    //for now assuming stereo sets directory exists in some outer directory and program won't work without it
    if (it == std::filesystem::end(dirIt))
    {
      //if current path same as parent path, throw error
      if (currentPath == currentPath.parent_path()) {
        throw std::filesystem::filesystem_error("Implementation results directory not found", std::error_code());
      }
      //continue to next outer directory
      currentPath = currentPath.parent_path();
    }
    
    //return path for implementation results
    if (it != std::filesystem::end(dirIt)) {
      return it->path();
    }
  }
  return std::filesystem::path();
}
