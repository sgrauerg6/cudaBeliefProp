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

    //get input mapping to runtime
    const std::array<std::string, 3> inputSigKeys{"Input Index", "DataType", "LOOP_ITERS_TEMPLATED"};
    //comparison lambda for sorting of input signature for each run (datatype, then input number, then whether or not using templated iter count)
    auto inputCmp = [] (const std::array<std::string, 3>& a, const std::array<std::string, 3>& b)
    {
      //sort by datatype followed by input number followed by templated iters setting
      if (a[1] != b[1]) {
        if (a[1] == "FLOAT") { return true; /* a < b is true*/ }
        else if (a[1] == "HALF") { return false; /* a < b is false */ }
        else if (b[1] == "FLOAT") { return false; /* a < b is false */ }
        else if (b[1] == "HALF") { return true; /* a < b is true */ }
      }
      else if (a[0] != b[0]) {
        return std::stoi(a[0]) < std::stoi(b[0]);
      }
      else if (a[2] != b[2]) {
        if (a[2] == "YES") { return true; /* a < b is true */ }
      }
      return false; /* a <= b is false*/
    };
    std::map<std::string, std::map<std::array<std::string, 3>, std::string, decltype(inputCmp)>> inputToRuntimeAcrossArchs;
    std::set<std::array<std::string, 3>, decltype(inputCmp)> inputSet;
    std::vector<std::string> inputParamsDisp{"Stereo Set", "DataType", "Image Width", "Image Height", "Num Possible Disparity Values", "LOOP_ITERS_TEMPLATED"};
    std::map<std::array<std::string, 3>, std::vector<std::string>, decltype(inputCmp)> inputSetToInputDisp;
    for (const auto& runResult : runResultsNameToData) {
      inputToRuntimeAcrossArchs[runResult.first] = std::map<std::array<std::string, 3>, std::string, decltype(inputCmp)>();
      const auto& resultKeysToResVect = runResult.second.second;
      const unsigned int totNumRuns = resultKeysToResVect.at(inputSigKeys[0]).size();
      for (size_t numRun = 0; numRun < totNumRuns; numRun++) {
        std::array<std::string, 3> runInput{resultKeysToResVect.at(inputSigKeys[0])[numRun], resultKeysToResVect.at(inputSigKeys[1])[numRun],
          resultKeysToResVect.at(inputSigKeys[2])[numRun]};
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
        (runName + '_' + std::string(run_eval::SPEEDUPS_DESCRIPTION_FILE_NAME) + std::string(run_eval::CSV_FILE_EXTENSION));
      if (std::filesystem::is_regular_file(runSpeedupFp)) {
        speedupResultsNameToData[runName] = getHeaderToDataInCsvFile(runSpeedupFp);
      }
    }

    //generate results across architectures
    std::ostringstream resultAcrossArchsSStr;
    //write out the name of each input parameter to be displayed
    for (const auto& inputParamDispHeader : inputParamsDisp) {
      resultAcrossArchsSStr << inputParamDispHeader << ',';
    }
    //write each architecture name and save order of architectures
    std::vector<std::string> runNamesInOrder;
    std::set<std::pair<float, std::string>, std::greater<std::pair<float, std::string>>> runNamesInOrderWSpeedup;
    std::string firstSpeedupHeader;
    for (const auto& archWSpeedupData : speedupResultsNameToData.begin()->second.first) {
      if (!(archWSpeedupData.empty())) {
        firstSpeedupHeader = archWSpeedupData;
        break;
      }
    }
    std::cout << "firstSpeedupHeader: " << firstSpeedupHeader << std::endl;
    for (const auto& archWSpeedupData : speedupResultsNameToData) {
      const float avgSpeedupVsBase = std::stof(archWSpeedupData.second.second.at(firstSpeedupHeader).at(0));
      runNamesInOrder.push_back(archWSpeedupData.first);
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
    std::string firstRunName = speedupResultsNameToData.begin()->first;
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