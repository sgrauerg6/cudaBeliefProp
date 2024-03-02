/*
 * CombineMultResultSets.cpp
 *
 *  Created on: Feb 25, 2024
 *      Author: scott
 * 
 *  Class for combining of multiple result sets across multiple architectures.
 */

#ifndef COMBINE_MULT_RESULT_SETS_H_
#define COMBINE_MULT_RESULT_SETS_H_

#include <map>
#include <set>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <ranges>
#include <algorithm>
#include "RunEvalConstsEnums.h"

class CombineMultResultSets {
public:
  void operator()(const std::filesystem::path& impResultsFilePath) const {
    //get header to data of each set of run results
    //iterate through all run results files and get run name to results
    //create directory iterator with all results files
    std::filesystem::directory_iterator resultsFilesIt = std::filesystem::directory_iterator(impResultsFilePath / run_eval::IMP_RESULTS_RUN_DATA_FOLDER_NAME);
    std::vector<std::string> runNames;
    std::map<std::string, std::pair<std::vector<std::string>, std::map<std::string, std::vector<std::string>>>> runResultsNameToData;
    for (const auto& resultsFp : resultsFilesIt) {
      std::string fileNameNoExt = resultsFp.path().stem();
      if (fileNameNoExt.ends_with("_" + std::string(run_eval::RUN_RESULTS_DESCRIPTION_FILE_NAME))) {
        std::string runName = fileNameNoExt.substr(0, fileNameNoExt.find("_" + std::string(run_eval::RUN_RESULTS_DESCRIPTION_FILE_NAME)));
        runNames.push_back(runName);
        runResultsNameToData[runName] = getHeaderToDataInCsvFile(resultsFp);
      }
    }

    std::map<std::string, std::map<std::array<std::string, 3>, std::string, run_eval::LessThanRunSigHdrs>> inputToRuntimeAcrossArchs;
    std::set<std::array<std::string, 3>, run_eval::LessThanRunSigHdrs> inputSet;
    const std::vector<std::string> inputParamsDisp{getInputParamsShow()};
    std::map<std::array<std::string, 3>, std::vector<std::string>, run_eval::LessThanRunSigHdrs> inputSetToInputDisp;
    for (const auto& runResult : runResultsNameToData) {
      inputToRuntimeAcrossArchs[runResult.first] = std::map<std::array<std::string, 3>, std::string, run_eval::LessThanRunSigHdrs>();
      const auto& resultKeysToResVect = runResult.second.second;
      const unsigned int totNumRuns = resultKeysToResVect.at(std::string(run_eval::RUN_INPUT_SIG_HDRS[0])).size();
      for (size_t numRun = 0; numRun < totNumRuns; numRun++) {
        const std::array<std::string, 3> runInput{
          resultKeysToResVect.at(std::string(run_eval::RUN_INPUT_SIG_HDRS[0]))[numRun],
          resultKeysToResVect.at(std::string(run_eval::RUN_INPUT_SIG_HDRS[1]))[numRun],
          resultKeysToResVect.at(std::string(run_eval::RUN_INPUT_SIG_HDRS[2]))[numRun]};
        inputToRuntimeAcrossArchs[runResult.first][runInput] = resultKeysToResVect.at(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER))[numRun];
        inputSet.insert(runInput);
        //add mapping from run input signature to run input to be displayed
        if (!(inputSetToInputDisp.contains(runInput))) {
          inputSetToInputDisp[runInput] = std::vector<std::string>();
          for (const auto& dispParam : inputParamsDisp) {
            inputSetToInputDisp[runInput].push_back(resultKeysToResVect.at(dispParam)[numRun]);
          }
        }
      }
    }

    //get header to data of each set of speedups
    //iterate through all speedup data files and get run name to results
    std::map<std::string, std::pair<std::vector<std::string>, std::map<std::string, std::vector<std::string>>>> speedupResultsNameToData;
    std::vector<std::string> speedupHeadersInOrder;
    for (const auto& runName : runNames) {
      std::filesystem::path runSpeedupFp = impResultsFilePath / run_eval::IMP_RESULTS_SPEEDUPS_FOLDER_NAME /
        (std::string(runName) + '_' + std::string(run_eval::SPEEDUPS_DESCRIPTION_FILE_NAME) + std::string(run_eval::CSV_FILE_EXTENSION));
      if (std::filesystem::is_regular_file(runSpeedupFp)) {
        speedupResultsNameToData[runName] = getHeaderToDataInCsvFile(runSpeedupFp);
      }
    }

    //generate results across architectures
    std::ostringstream resultAcrossArchsSStr;
    //add text to display on top of results across architecture comparison file
    for (const auto& compFileTopTextLine : getCombResultsTopText()) {
      resultAcrossArchsSStr << compFileTopTextLine << std::endl;
    }
    resultAcrossArchsSStr << std::endl;

    //write out the name of each input parameter to be displayed
    for (const auto& inputParamDispHeader : inputParamsDisp) {
      resultAcrossArchsSStr << inputParamDispHeader << ',';
    }

    //write each architecture name and save order of architectures with speedup corresponding to first
    //speedup header
    //order of architectures from left to right is in speedup from largest to smallest
    std::set<std::pair<float, std::string>, std::greater<std::pair<float, std::string>>> runNamesInOrderWSpeedup;
    std::string firstSpeedupHeader;
    for (const auto& archWSpeedupData : speedupResultsNameToData.cbegin()->second.first) {
      if (!(archWSpeedupData.empty())) {
        firstSpeedupHeader = archWSpeedupData;
        break;
      }
    }
    for (const auto& archWSpeedupData : speedupResultsNameToData) {
      const float avgSpeedupVsBase = std::stof(std::string(archWSpeedupData.second.second.at(firstSpeedupHeader).at(0)));
      runNamesInOrderWSpeedup.insert({avgSpeedupVsBase, archWSpeedupData.first});
      resultAcrossArchsSStr << archWSpeedupData.first << ',';
    }
    resultAcrossArchsSStr << std::endl;

    //write input data and runtime for each run for each architecture
    for (const auto& currRunInput : inputSetToInputDisp) {
      for (const auto& runInputVal : currRunInput.second) {
        resultAcrossArchsSStr << runInputVal << ',';
      }
      for (const auto& runName : runNamesInOrderWSpeedup) {
        if (inputToRuntimeAcrossArchs.at(runName.second).contains(currRunInput.first)) {
          resultAcrossArchsSStr << inputToRuntimeAcrossArchs.at(runName.second).at(currRunInput.first) << ',';
        }
      }
      resultAcrossArchsSStr << std::endl;
    }
    resultAcrossArchsSStr << std::endl;

    //write each average speedup with results for each architecture
    resultAcrossArchsSStr << "Average Speedups" << std::endl;
    std::string firstRunName = speedupResultsNameToData.cbegin()->first;
    for (const auto& speedupHeader : speedupResultsNameToData.at(firstRunName).first) {
      //don't process if header is empty
      if (!(speedupHeader.empty())) {
        resultAcrossArchsSStr << speedupHeader << ',';
        //add empty cell for each input parameter after the first that's displayed so speedup shown on same line as runtime for architecture
        for (size_t i = 1; i < inputParamsDisp.size(); i++) {
          resultAcrossArchsSStr << ',';
        }
        //write speedup for each architecture in separate cells in horizontal direction
        for (const auto& runName : runNamesInOrderWSpeedup) {
          resultAcrossArchsSStr << speedupResultsNameToData.at(runName.second).second.at(speedupHeader).at(0) << ',';
        }
        //continue to next row of table
        resultAcrossArchsSStr << std::endl;
      }
    }
    std::ofstream combResultsStr("CombResults.csv");
    combResultsStr << resultAcrossArchsSStr.str();
  }

private:
  //get text at top of results summary file with each string_view in the vector corresponding to a separate line
  std::vector<std::string> getCombResultsTopText() const {
    return {{"Stereo Processing using optimized CUDA and optimized CPU belief propagation implementations"},
            {"Code available at https://github.com/sgrauerg6/cudaBeliefProp"},
            {"All stereo sets used in evaluation are from (or adapted from) Middlebury stereo datasets at https://vision.middlebury.edu/stereo/data/"},
            {"\"tsukubaSetHalfSize: tsukubaSet with half the height, width, and disparity count of tsukubaSet\""},
            {"conesFullSizeCropped: 900 x 750 region in center of the reference and test cones stereo set images"},
            {"Results shown in this comparison for each run are for total runtime including any time for data transfer between device and host"}};
  }

  //input parameters that are showed in results summary with runtimes
  std::vector<std::string> getInputParamsShow() const {
    return {"Stereo Set", "DataType", "Image Width", "Image Height", "Num Possible Disparity Values", "LOOP_ITERS_TEMPLATED"};
  }

  //get mapping of headers to data in csv file for run results and speedups
  //assumed that there are no commas in data since it is used as delimiter between data
  //first output is headers in order, second output is mapping of headers to results
  std::pair<std::vector<std::string>, std::map<std::string, std::vector<std::string>>> getHeaderToDataInCsvFile(const std::filesystem::path& csvFilePath) const {
    std::ifstream csvFileStr(csvFilePath);
    //retrieve data headers from top row
    std::string headersLine;
    std::getline(csvFileStr, headersLine);
    std::stringstream headersStr(headersLine);
    std::vector<std::string> dataHeaders;
    std::string header;
    std::map<std::string, std::vector<std::string>> headerToData;
    while (std::getline(headersStr, header, ',')) {
      dataHeaders.push_back(header);
      headerToData[header] = std::vector<std::string>();
    }
    //go through each data line and add to mapping from headers to data
    std::string dataLine;
    while (std::getline(csvFileStr, dataLine)) {
      std::stringstream dataLineStr(dataLine);
      std::string data;
      unsigned int numData{0};
      while (std::getline(dataLineStr, data, ',')) {
        headerToData[dataHeaders[numData++]].push_back(data);
      }
    }
    return {dataHeaders, headerToData};
  }
};

#endif //COMBINE_MULT_RESULT_SETS_H_