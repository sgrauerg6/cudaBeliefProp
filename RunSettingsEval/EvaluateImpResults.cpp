/*
 * EvaluateImpResults.cpp
 *
 *  Created on: Feb 20, 2024
 *      Author: scott
 * 
 *  Function definitions for class to evaluate implementation results.
 */

#include "EvaluateImpResults.h"
#include "EvaluateAcrossRuns.h"
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
void EvaluateImpResults::operator()(const std::unordered_map<size_t, MultRunDataWSpeedupByAcc>& runResultsMultRuns,
  const run_environment::RunImpSettings runImpSettings, run_environment::AccSetting optImpAcc)
{
  runImpResultsMultRuns_ = runResultsMultRuns;
  runImpSettings_ = runImpSettings;
  optImpAccel_ = optImpAcc;
  evalResultsMultDTypeAccRuns();
}

//get run data with speedup from evaluation of implementation runs using multiple inputs with runs having the same data type and acceleration method
std::pair<MultRunData, std::vector<MultRunSpeedup>> EvaluateImpResults::getRunDataWSpeedups() const {
  return {runImpOptResults_, runImpSpeedups_};
}

//write data for file corresponding to runs for a specified data type or across all data type
//includes results for each run as well as average and median speedup data across multiple runs
template <bool kMultDataTypes>
void EvaluateImpResults::writeRunOutput(const std::pair<MultRunData, std::vector<MultRunSpeedup>>& runOutput, const run_environment::RunImpSettings& runImpSettings,
  run_environment::AccSetting accelerationSetting, unsigned int dataTypeSize) const
{
  //get iterator to first run with success
  const auto firstSuccessRun = std::find_if(runOutput.first.cbegin(), runOutput.first.cend(), [](const auto& runResult)
    { return runResult; } );

  //check if there was at least one successful run
  if (firstSuccessRun != runOutput.first.cend()) {
    //write results from default and optimized parallel parameters runs to csv file
    //file name contains info about data type, parameter settings, and processor name if available
    //only show data type string and acceleration string for runs using a single data type that are used for debugging (not multidata type results) 
    const std::string dataTypeStr = kMultDataTypes ? "" : '_' + std::string(run_environment::kDataSizeToNameMap.at(dataTypeSize));
    const auto accelStr = kMultDataTypes ? "" : '_' + std::string(run_environment::accelerationString(accelerationSetting));
    //get directory to store implementation results for specific implementation
    const auto impResultsFp = getImpResultsPath();
    //create any results directories if needed
    if (!(std::filesystem::is_directory(impResultsFp / run_eval::kImpResultsRunDataFolderName))) {
      std::filesystem::create_directory(impResultsFp / run_eval::kImpResultsRunDataFolderName);
    }
    if (!(std::filesystem::is_directory(impResultsFp / run_eval::kImpResultsRunDataWSpeedupsFolderName))) {
      std::filesystem::create_directory(impResultsFp / run_eval::kImpResultsRunDataWSpeedupsFolderName);
    }
    if (!(std::filesystem::is_directory(impResultsFp / run_eval::kImpResultsSpeedupsFolderName))) {
      std::filesystem::create_directory(impResultsFp / run_eval::kImpResultsSpeedupsFolderName);
    }
    //get file paths for run result and speedup files for implementation run
    const std::filesystem::path optResultsFilePath{impResultsFp / run_eval::kImpResultsRunDataFolderName / std::filesystem::path(((runImpSettings.runName_) ? std::string(runImpSettings.runName_.value()) + "_" : "") + 
      std::string(run_eval::kRunResultsDescFileName) + dataTypeStr + accelStr + std::string(run_eval::kCsvFileExtension))};
    const std::filesystem::path optResultsWSpeedupFilePath{impResultsFp / run_eval::kImpResultsRunDataWSpeedupsFolderName / std::filesystem::path(((runImpSettings.runName_) ? std::string(runImpSettings.runName_.value()) + "_" : "") + 
      std::string(run_eval::kRunResultsWSpeedupsDescFileName) + dataTypeStr + accelStr + std::string(run_eval::kCsvFileExtension))};
    const std::filesystem::path defaultParamsResultsFilePath{impResultsFp / run_eval::kImpResultsRunDataFolderName / std::filesystem::path(((runImpSettings.runName_) ? std::string(runImpSettings.runName_.value()) + "_" : "") + 
      std::string(run_eval::kRunResultsDescDefaultPParamsFileName) + dataTypeStr + accelStr + std::string(run_eval::kCsvFileExtension))};
    const std::filesystem::path speedupResultsFilePath{impResultsFp / run_eval::kImpResultsSpeedupsFolderName / std::filesystem::path(((runImpSettings.runName_) ? std::string(runImpSettings.runName_.value()) + "_" : "") + 
      std::string(run_eval::kSpeedupsDescFileName) + dataTypeStr + accelStr + std::string(run_eval::kCsvFileExtension))};
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
            runDataOptDefaultSStr[i] << runOutput.first[runNum]->at(runResultIdx).getDataAsStr(currHeader) << ',';
          }
        }
        runDataOptDefaultSStr[i] << std::endl;
      }
    }

    //generate speedup results with headers on left side and with headers on top row
    speedupsHeadersLeftTopSStr[0] << "Speedup Results,Average Speedup,Median Speedup" << std::endl;
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
    runResultWSpeedupsStr << speedupsHeadersLeftTopSStr[0].str();

    std::cout << "Input/settings/parameters info, detailed timings, and evaluation for each run and across runs in " << optResultsWSpeedupFilePath << std::endl;
    std::cout << "Run inputs and results in " << optResultsFilePath << std::endl;
    std::cout << "Speedup results in " << speedupResultsFilePath << std::endl;
    EvaluateAcrossRuns().operator()(impResultsFp, getCombResultsTopText(), getInputParamsShow());
  }
  else {
    std::cout << "Error, no runs completed successfully" << std::endl;
  }
}

//evaluate results across runs using the same data type
void EvaluateImpResults::evalResultsSingDTypeAccRun() {
  //initialize and add speedup results over baseline data if available for current input
  runImpOptResults_ = runImpOrigResults_;
  const auto speedupOverBaseline = getSpeedupOverBaseline(runImpSettings_, runImpOptResults_, dataSize_);
  const auto speedupOverBaselineSubsets = getSpeedupOverBaselineSubsets(runImpSettings_, runImpOptResults_, dataSize_);
  runImpSpeedups_.insert(runImpSpeedups_.cend(), speedupOverBaseline.cbegin(), speedupOverBaseline.cend());
  runImpSpeedups_.insert(runImpSpeedups_.cend(), speedupOverBaselineSubsets.cbegin(), speedupOverBaselineSubsets.cend());

  //compute and add speedup info for using optimized parallel parameters and disparity count as template parameter to speedup results
  if (runImpSettings_.optParallelParamsOptionSetting_.first) {
    runImpSpeedups_.push_back(getAvgMedSpeedupOptPParams(runImpOptResults_, std::string(run_eval::kSpeedupOptParParamsHeader) + " - " +
      std::string(run_environment::kDataSizeToNameMap.at(dataSize_))));
  }
  if (runImpSettings_.templatedItersSetting_ == run_environment::TemplatedItersSetting::kRunTemplatedAndNotTemplated) {
    runImpSpeedups_.push_back(getAvgMedSpeedupLoopItersInTemplate(runImpOptResults_, std::string(run_eval::kSpeedupDispCountTemplate) + " - " +
      std::string(run_environment::kDataSizeToNameMap.at(dataSize_))));
  }

  //write output corresponding to results for current data type if writing debug output
  if (writeDebugOutputFiles_) {
    constexpr bool kMultDataTypes{false};
    writeRunOutput<kMultDataTypes>({runImpOptResults_, runImpSpeedups_}, runImpSettings_, optImpAccel_, dataSize_);
  }
}

//perform runs without CPU vectorization and get speedup for each run and overall when using vectorization
//CPU vectorization does not apply to CUDA acceleration so "NO_DATA" output is returned in that case
std::vector<MultRunSpeedup> EvaluateImpResults::getAltAccelSpeedups(
  MultRunDataWSpeedupByAcc& runImpResultsByAccSetting,
  const run_environment::RunImpSettings& runImpSettings, size_t dataTypeSize, run_environment::AccSetting fastestAcc) const
{
  //set up mapping from acceleration type to description
  const std::map<run_environment::AccSetting, std::string> accToSpeedupStr{
    {run_environment::AccSetting::NONE, 
     std::string(run_eval::kSpeedupCPUVectorization) + " - " + std::string(run_environment::kDataSizeToNameMap.at(dataTypeSize))},
    {run_environment::AccSetting::AVX256, 
     std::string(run_eval::kSpeedupVsAvx256Vectorization) + " - " + std::string(run_environment::kDataSizeToNameMap.at(dataTypeSize))}};

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
            const double initResultTime = runImpResultsByAccSetting[fastestAcc].first[i]->back().getDataAsDouble(run_eval::kOptimizedRuntimeHeader).value();
            const double altAccResultTime = altAccImpResults.second.first[i]->back().getDataAsDouble(run_eval::kOptimizedRuntimeHeader).value();
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
        runImpResultsMultRuns_[dataSize][optImpAccel_].first, (dataSize > sizeof(float)) ? run_eval::kSpeedupDouble : run_eval::kSpeedupHalf);
    }
  }

  //initialize overall results to float results using fastest acceleration and add double and half-type results to it
  auto resultsWSpeedups = runImpResultsMultRuns_[sizeof(float)][optImpAccel_];
  resultsWSpeedups.first.insert(resultsWSpeedups.first.cend(),
    runImpResultsMultRuns_[sizeof(double)][optImpAccel_].first.cbegin(), runImpResultsMultRuns_[sizeof(double)][optImpAccel_].first.cend());
  resultsWSpeedups.first.insert(resultsWSpeedups.first.cend(),
    runImpResultsMultRuns_[sizeof(halftype)][optImpAccel_].first.cbegin(), runImpResultsMultRuns_[sizeof(halftype)][optImpAccel_].first.cend());

  //add speedup data from double and half precision runs to speedup results
  resultsWSpeedups.second.insert(resultsWSpeedups.second.cend(), altImpSpeedup[sizeof(float)].cbegin(), altImpSpeedup[sizeof(float)].cend());
  resultsWSpeedups.second.insert(resultsWSpeedups.second.cend(),
    runImpResultsMultRuns_[sizeof(double)][optImpAccel_].second.cbegin(), runImpResultsMultRuns_[sizeof(double)][optImpAccel_].second.cend());
  resultsWSpeedups.second.insert(resultsWSpeedups.second.cend(), altImpSpeedup[sizeof(double)].cbegin(), altImpSpeedup[sizeof(double)].cend());
  resultsWSpeedups.second.insert(resultsWSpeedups.second.cend(),
    runImpResultsMultRuns_[sizeof(halftype)][optImpAccel_].second.cbegin(), runImpResultsMultRuns_[sizeof(halftype)][optImpAccel_].second.cend());
  resultsWSpeedups.second.insert(resultsWSpeedups.second.cend(), altImpSpeedup[sizeof(halftype)].cbegin(), altImpSpeedup[sizeof(halftype)].cend());

  //get speedup over baseline runtimes...can only compare with baseline runtimes that are
  //generated using same templated iterations setting as current run
  if ((runImpSettings_.baseOptSingThreadRTimeForTSetting_) &&
      (runImpSettings_.baseOptSingThreadRTimeForTSetting_.value().second == runImpSettings_.templatedItersSetting_)) {
      const auto speedupOverBaseline = getAvgMedSpeedupOverBaseline(resultsWSpeedups.first, "All Runs", runImpSettings_.baseOptSingThreadRTimeForTSetting_.value().first);
      resultsWSpeedups.second.insert(resultsWSpeedups.second.cend(), speedupOverBaseline.cbegin(), speedupOverBaseline.cend());
  }

  //get speedup info for using optimized parallel parameters
  if (runImpSettings_.optParallelParamsOptionSetting_.first) {
    resultsWSpeedups.second.push_back(getAvgMedSpeedupOptPParams(resultsWSpeedups.first, std::string(run_eval::kSpeedupOptParParamsHeader) + " - All Runs"));
  }

  //get speedup when using template for loop iteration count
  if (runImpSettings_.templatedItersSetting_ == run_environment::TemplatedItersSetting::kRunTemplatedAndNotTemplated) {
    resultsWSpeedups.second.push_back(getAvgMedSpeedupLoopItersInTemplate(resultsWSpeedups.first, std::string(run_eval::kSpeedupDispCountTemplate) + " - All Runs"));
  }

  //add speedups when using doubles and half precision compared to float to end of speedup data
  resultsWSpeedups.second.insert(resultsWSpeedups.second.cend(), {altDataTypeSpeedup[sizeof(double)], altDataTypeSpeedup[sizeof(halftype)]});

  //write output corresponding to results and speedups for all data types
  constexpr bool kMultDataTypes{true};
  writeRunOutput<kMultDataTypes>(resultsWSpeedups, runImpSettings_, optImpAccel_);
}

//get speedup over baseline data if data available
std::vector<MultRunSpeedup> EvaluateImpResults::getSpeedupOverBaseline(const run_environment::RunImpSettings& runImpSettings,
  MultRunData& runDataAllRuns, size_t dataTypeSize) const
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
        runDataAllRuns, run_environment::kDataSizeToNameMap.at(dataTypeSize),
        runImpSettings.baseOptSingThreadRTimeForTSetting_.value().first);
      speedupResults.insert(speedupResults.cend(), speedupOverBaselineSubsets.cbegin(), speedupOverBaselineSubsets.cend());
    }
  }
  return speedupResults;
}

//get speedup over baseline run for subsets of smallest and largest sets if data available
std::vector<MultRunSpeedup> EvaluateImpResults::getSpeedupOverBaselineSubsets(const run_environment::RunImpSettings& runImpSettings,
  MultRunData& runDataAllRuns, const size_t dataTypeSize) const
{
  if ((dataTypeSize == sizeof(float)) &&
    (runImpSettings.baseOptSingThreadRTimeForTSetting_ && (runImpSettings.baseOptSingThreadRTimeForTSetting_->second == runImpSettings.templatedItersSetting_)))
  {
    return getAvgMedSpeedupOverBaselineSubsets(runDataAllRuns, run_environment::kDataSizeToNameMap.at(dataTypeSize),
      runImpSettings.baseOptSingThreadRTimeForTSetting_->first, runImpSettings.subsetStrIndices_);
  }
  //return empty vector if doesn't match settings to get speedup over baseline for subsets
  return std::vector<MultRunSpeedup>();
}

//get baseline runtime data if available...return null if baseline data not available
std::optional<std::pair<std::string, std::vector<double>>> EvaluateImpResults::getBaselineRuntimeData(std::string_view baselineDataPath) const {
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
  const double averageSpeedup = (std::accumulate(speedupsVect.cbegin(), speedupsVect.cend(), 0.0) / (double)speedupsVect.size());
  auto speedupsVectSorted = speedupsVect;
  std::ranges::sort(speedupsVectSorted);
  const double medianSpeedup = ((speedupsVectSorted.size() % 2) == 0) ? 
    (speedupsVectSorted[(speedupsVectSorted.size() / 2) - 1] + speedupsVectSorted[(speedupsVectSorted.size() / 2)]) / 2.0 : 
    speedupsVectSorted[(speedupsVectSorted.size() / 2)];
  return {averageSpeedup, medianSpeedup};
}

//get average and median speedup of specified subset(s) of runs compared to baseline data from file
std::vector<MultRunSpeedup> EvaluateImpResults::getAvgMedSpeedupOverBaselineSubsets(MultRunData& runOutput,
  std::string_view dataTypeStr, const std::array<std::string_view, 2>& baseDataPathOptSingThrd,
  const std::vector<std::pair<std::string, std::vector<unsigned int>>>& subsetStrIndices) const
{
  //get speedup over baseline for optimized runs
  std::vector<MultRunSpeedup> speedupData;
  const auto baselineRunData = getBaselineRuntimeData(baseDataPathOptSingThrd[0]);
  if (baselineRunData) {
    std::string speedupHeader = "Speedup relative to " + std::string((*baselineRunData).first) + " - " + std::string(dataTypeStr);
    const auto baselineRuntimes = (*baselineRunData).second;
    //retrieve speedup data for any subsets of optimized runs
    for (const auto& currSubsetStrIndices : subsetStrIndices) {
      std::vector<double> speedupsVect;
      speedupHeader = "Speedup relative to " + std::string((*baselineRunData).first) + " on " + std::string(currSubsetStrIndices.first) + " - " + std::string(dataTypeStr);
      for (unsigned int i : currSubsetStrIndices.second) {
        if (runOutput[i]) {
          speedupsVect.push_back(baselineRuntimes[i] / runOutput[i]->at(1).getDataAsDouble(run_eval::kOptimizedRuntimeHeader).value());
          for (auto& runData : runOutput[i].value()) {
            runData.addDataWHeader(std::string(speedupHeader), speedupsVect.back());
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
std::vector<MultRunSpeedup> EvaluateImpResults:: getAvgMedSpeedupOverBaseline(MultRunData& runOutput,
  std::string_view dataTypeStr, const std::array<std::string_view, 2>& baselinePathOptSingThread) const
{
  //get speedup over baseline for optimized runs
  std::vector<MultRunSpeedup> speedupData;
  const auto baselineRunData = getBaselineRuntimeData(baselinePathOptSingThread[0]);
  if (baselineRunData) {
    std::vector<double> speedupsVect;
    const std::string speedupHeader = "Speedup relative to " + baselineRunData->first + " - " + std::string(dataTypeStr);
    const auto baselineRuntimes = (*baselineRunData).second;
    for (unsigned int i=0; i < runOutput.size(); i++) {
      if (runOutput[i]) {
       speedupsVect.push_back(baselineRuntimes[i] / runOutput[i]->at(1).getDataAsDouble(run_eval::kOptimizedRuntimeHeader).value());
        for (auto& runData : runOutput[i].value()) {
          runData.addDataWHeader(speedupHeader, speedupsVect.back());
        }
      }
    }
    if (!(speedupsVect.empty())) {
      speedupData.push_back({speedupHeader, getAvgMedSpeedup(speedupsVect)});
    }
  }

  //get speedup over baseline for single thread runs
  const auto baselineRunDataSThread = getBaselineRuntimeData(baselinePathOptSingThread[1]);
  if (baselineRunDataSThread) {
    std::vector<double> speedupsVect;
    const std::string speedupHeader = "Single-Thread (Orig Imp) speedup relative to " + std::string((*baselineRunDataSThread).first) + " - " + std::string(dataTypeStr);
    const auto baselineRuntimesSThread = (*baselineRunDataSThread).second;
    for (unsigned int i=0; i < runOutput.size(); i++) {
      if (runOutput[i]) {
        speedupsVect.push_back(baselineRuntimesSThread[i] / runOutput[i]->at(1).getDataAsDouble(run_eval::kSingleThreadRuntimeHeader).value());
        for (auto& runData : runOutput[i].value()) {
          runData.addDataWHeader(speedupHeader, speedupsVect.back());
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
MultRunSpeedup EvaluateImpResults::getAvgMedSpeedupOptPParams(MultRunData& runOutput, std::string_view speedupHeader) const {
  std::vector<double> speedupsVect;
  for (unsigned int i=0; i < runOutput.size(); i++) {
    if (runOutput[i]) {
      speedupsVect.push_back(runOutput[i]->at(0).getDataAsDouble(run_eval::kOptimizedRuntimeHeader).value() / 
                             runOutput[i]->at(1).getDataAsDouble(run_eval::kOptimizedRuntimeHeader).value());
      runOutput[i]->at(0).addDataWHeader(std::string(speedupHeader), speedupsVect.back());
      runOutput[i]->at(1).addDataWHeader(std::string(speedupHeader), speedupsVect.back());
    }
  }
  if (!(speedupsVect.empty())) {
    return {std::string(speedupHeader), getAvgMedSpeedup(speedupsVect)};
  }
  return {std::string(speedupHeader), {0.0, 0.0}};
}

//get average and median speedup between base and target runtime data
MultRunSpeedup EvaluateImpResults::getAvgMedSpeedup(MultRunData& runOutputBase, MultRunData& runOutputTarget,
  std::string_view speedupHeader) const
{
  std::vector<double> speedupsVect;
  for (unsigned int i=0; i < runOutputBase.size(); i++) {
    if (runOutputBase[i] && runOutputTarget[i]) {
      speedupsVect.push_back(runOutputBase[i]->back().getDataAsDouble(run_eval::kOptimizedRuntimeHeader).value() / 
                             runOutputTarget[i]->back().getDataAsDouble(run_eval::kOptimizedRuntimeHeader).value());
      runOutputBase[i]->at(1).addDataWHeader(std::string(speedupHeader), speedupsVect.back());
      runOutputTarget[i]->at(1).addDataWHeader(std::string(speedupHeader), speedupsVect.back());
    }
  }
  if (!(speedupsVect.empty())) {
    return {std::string(speedupHeader), getAvgMedSpeedup(speedupsVect)};
  }
  return {std::string(speedupHeader), {0.0, 0.0}};
}

//get average and median speedup when loop iterations are given at compile time as template value
MultRunSpeedup EvaluateImpResults::getAvgMedSpeedupLoopItersInTemplate(MultRunData& runOutput,
  std::string_view speedupHeader) const
{
  //get mapping of run inputs to runtime with index value in run output
  std::map<std::vector<std::string>, std::pair<double, size_t>> runInputSettingsToTimeWIdx;
  for (unsigned int i=0; i < runOutput.size(); i++) {
    if (runOutput[i]) {
      const auto inputSettingsToTime = runOutput[i]->back().getParamsToRuntime(
        std::vector<std::string_view>(run_eval::kRunInputSigHeaders.cbegin(), run_eval::kRunInputSigHeaders.cend()),
        run_eval::kOptimizedRuntimeHeader);
      if (inputSettingsToTime) {
        runInputSettingsToTimeWIdx.insert({inputSettingsToTime->first, {inputSettingsToTime->second, i}});
      }
    }
  }
  //go through all run input settings to time and get each pair that is the same in datatype and input and differs in disp values templated
  //and get speedup for each of templated compared to non-templated
  std::vector<double> speedupsVect;
  auto runDataIter = runInputSettingsToTimeWIdx.cbegin();
  while (runDataIter != runInputSettingsToTimeWIdx.cend()) {
    auto dataTypeRun = runDataIter->first[run_eval::kRunInputDatatypeIdx];
    auto inputIdxRun = runDataIter->first[run_eval::kRunInputNumInputIdx];
    auto runComp1 = runDataIter;
    auto runComp2 = runDataIter;
    //find run with same datatype and input index
    while (++runDataIter != runInputSettingsToTimeWIdx.cend()) {
      if ((runDataIter->first[run_eval::kRunInputDatatypeIdx] == dataTypeRun) &&
          (runDataIter->first[run_eval::kRunInputNumInputIdx] == inputIdxRun))
      {
        runComp2 = runDataIter;
        break;
      }
    }
    //if don't have two separate runs with same data type and input, erase current run from mapping and continue
    if (runComp1 == runComp2) {
      runInputSettingsToTimeWIdx.erase(runComp1);
      runDataIter = runInputSettingsToTimeWIdx.cbegin();
      continue;
    }
    //retrieve which run data uses templated iteration count and which one doesn't and get speedup
    //add speedup to speedup vector and also to run data of run with templated iteration count
    if ((runComp1->first[run_eval::kRunInputLoopItersTemplatedIdx] == run_eval::kBoolValFalseTrueDispStr[1]) &&
        (runComp2->first[run_eval::kRunInputLoopItersTemplatedIdx] == run_eval::kBoolValFalseTrueDispStr[0]))
    {
      speedupsVect.push_back(runComp2->second.first / runComp1->second.first);
      runOutput[runComp1->second.second]->back().addDataWHeader(std::string(speedupHeader), speedupsVect.back());
    }
    else if ((runComp1->first[run_eval::kRunInputLoopItersTemplatedIdx] == run_eval::kBoolValFalseTrueDispStr[0]) &&
             (runComp2->first[run_eval::kRunInputLoopItersTemplatedIdx] == run_eval::kBoolValFalseTrueDispStr[1]))
    {
      speedupsVect.push_back(runComp1->second.first / runComp2->second.first);
      runOutput[runComp2->second.second]->back().addDataWHeader(std::string(speedupHeader), speedupsVect.back());
    }
    //remove runs that have been processed from mapping
    runInputSettingsToTimeWIdx.erase(runComp1);
    runInputSettingsToTimeWIdx.erase(runComp2);
    runDataIter = runInputSettingsToTimeWIdx.cbegin();
  }
  if (!(speedupsVect.empty())) {
    return {std::string(speedupHeader), getAvgMedSpeedup(speedupsVect)};
  }
  return {std::string(speedupHeader), {0.0, 0.0}};
}