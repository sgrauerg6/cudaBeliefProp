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
        std::cout << "RUN NAME: " << runName << std::endl;
        runNames.push_back(runName);
        runResultsNameToData[runName] = getHeaderToDataInCsvFile(resultsFp);
        std::cout << "HEADERS w data" << std::endl;
        for (const auto& header : runResultsNameToData[runName].first) {
          std::cout << header << " " << runResultsNameToData[runName].second[header][0] << std::endl;
        }
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
    for (const auto& runResult : runResultsNameToData) {
      inputToRuntimeAcrossArchs[runResult.first] = std::map<std::array<std::string, 3>, std::string, decltype(inputCmp)>();
      const auto& resultKeysToResVect = runResult.second.second;
      const unsigned int totNumRuns = resultKeysToResVect.at(inputSigKeys[0]).size();
      std::cout << "totNumRuns: " << totNumRuns << std::endl;
      for (size_t numRun = 0; numRun < totNumRuns; numRun++) {
        std::cout << "numRun: " << numRun << std::endl;
        std::array<std::string, 3> runInput{resultKeysToResVect.at(inputSigKeys[0])[numRun], resultKeysToResVect.at(inputSigKeys[1])[numRun],
          resultKeysToResVect.at(inputSigKeys[2])[numRun]};
        std::cout << "runInput: " << runInput[0] << " " << runInput[1] << " " << runInput[2] << std::endl;
        inputToRuntimeAcrossArchs[runResult.first][runInput] = resultKeysToResVect.at(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER))[numRun];
        inputSet.insert(runInput);
      }
    }
    
    for (const auto& inputToRuntimeArch : inputToRuntimeAcrossArchs) {
      std::cout << inputToRuntimeArch.first << std::endl;
      std::cout << "Input Set" << " " << "DataType" << " " << "LOOP_ITERS_TEMPLATED" << std::endl;
      for (const auto& runInput : inputSet) {
        std::cout << runInput[0] << " " << runInput[1] << " " << runInput[2] << " ";
        if (inputToRuntimeArch.second.contains(runInput)) {
          std::cout << inputToRuntimeArch.second.at(runInput);
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
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
        std::cout << "SPEEDUP HEADERS w data" << std::endl;
        for (const auto& header : speedupResultsNameToData[runName].second) {
          std::cout << header.first << " " << header.second[0] << std::endl;
        }
      }
    }

    //generate results across architectures
    std::ostringstream resultAcrossArchsSStr;
    //write each architecture name
    std::cout << "1" << std::endl;
    std::vector<std::string> runNamesInOrder;
    resultAcrossArchsSStr << ',';
    for (const auto& archWSpeedupData : speedupResultsNameToData) {
      runNamesInOrder.push_back(archWSpeedupData.first);
      resultAcrossArchsSStr << archWSpeedupData.first << ',';
    }
    std::cout << "2" << std::endl;
    resultAcrossArchsSStr << std::endl;
    //write input data and runtime for each run for each architecture
    std::map<std::string, std::vector<std::string>> inDataRunTimesEachArch;
    std::cout << "3" << std::endl;
    for (const auto& archWResultsData : runResultsNameToData) {
      //if (!(inDataRunTimesEachArch.contains(archWResultsData.first))) {
      //  inDataRunTimesEachArch[archWResultsData.first] = std::vector<std::string>();
      //}
      inDataRunTimesEachArch[archWResultsData.first] = archWResultsData.second.second.at(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER));
    }
    std::cout << "4" << std::endl;
    size_t numRuntimes = inDataRunTimesEachArch.begin()->second.size();
    for (unsigned int i = 0; i < numRuntimes; i++) {
      std::cout << "4a" << std::endl;
      resultAcrossArchsSStr << ',';
      for (const auto& runName : runNamesInOrder) {
        resultAcrossArchsSStr << inDataRunTimesEachArch.at(runName).at(i) << ',';
      }
      resultAcrossArchsSStr << std::endl;
    }
    std::cout << "5" << std::endl;
    resultAcrossArchsSStr << std::endl;
    //write each speedup with results for each architecture
    std::string firstRunName = speedupResultsNameToData.begin()->first;
    for (const auto& speedupHeader : speedupResultsNameToData.at(firstRunName).first) {
      resultAcrossArchsSStr << speedupHeader << ',';
      for (const auto& speedupData : speedupResultsNameToData) {
        resultAcrossArchsSStr << speedupData.second.second.at(speedupHeader).at(0) << ',';
      }
      resultAcrossArchsSStr << std::endl;
    }
    std::cout << "6" << std::endl;
    std::ofstream combResultsStr("CombResults.csv");
    combResultsStr << resultAcrossArchsSStr.str();
    std::cout << "7" << std::endl;
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