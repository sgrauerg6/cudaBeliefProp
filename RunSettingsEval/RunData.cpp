/*
 * RunData.cpp
 *
 *  Created on: Feb 13, 2024
 *      Author: scott
 * 
 * Implementation of functions in RunData class
 */

#include "RunData.h"
#include "RunEvalConstsEnums.h"

//get header to add...use input header if not yet used
//user original header with number appended if original header is already used
std::string RunData::getHeaderToAdd(const std::string& inHeader) const {
  auto headerToAdd = inHeader;
  unsigned int num{0};
  while (std::find(headersInOrder_.begin(), headersInOrder_.end(), headerToAdd) != headersInOrder_.end()) {
    //add "_{num}" to header if header already in data
    num++;
    headerToAdd = inHeader + "_" + std::to_string(num);
  }
  return headerToAdd;
}

//add data with header describing added data
void RunData::addDataWHeader(const std::string& header, const std::string& data) {
  const auto headerToAdd{getHeaderToAdd(header)};
  headersInOrder_.push_back(headerToAdd);
  headersWData_[headerToAdd] = data;
}

//add data with header describing added data
void RunData::addDataWHeader(const std::string& header, double data) {
  const auto headerToAdd{getHeaderToAdd(header)};
  headersInOrder_.push_back(headerToAdd);
  headersWData_[headerToAdd] = data;
}

//add data with header describing added data
void RunData::addDataWHeader(const std::string& header, bool data) {
  const auto headerToAdd{getHeaderToAdd(header)};
  headersInOrder_.push_back(headerToAdd);
  headersWData_[headerToAdd] = data;
}

//add data with header describing added data
void RunData::addDataWHeader(const std::string& header, unsigned int data) {
  const auto headerToAdd{getHeaderToAdd(header)};
  headersInOrder_.push_back(headerToAdd);
  headersWData_[headerToAdd] = data;
}

//append current RunData with input RunData
void RunData::appendData(const RunData& inRunData) {
  //const auto& inRunDataMapping = inRunData.getAllData();
  for (const auto& header : inRunData.getHeadersInOrder()) {
    if (inRunData.getDataAsDouble(header)) {
      addDataWHeader(header, *(inRunData.getDataAsDouble(header)));
    }
    else if (inRunData.getDataAsUInt(header)) {
      addDataWHeader(header, *(inRunData.getDataAsUInt(header)));
    }
    else if (inRunData.getDataAsBool(header)) {
      addDataWHeader(header, *(inRunData.getDataAsBool(header)));
    }
    else {
      addDataWHeader(header, inRunData.getDataAsStr(header));
    }
  }
}

//get data corresponding to header
std::string RunData::getDataAsStr(const std::string_view header) const {
  const auto variantVal = headersWData_.at(std::string(header));
  if (std::holds_alternative<std::string>(variantVal)) {
    return std::get<std::string>(variantVal);
  }
  if (std::holds_alternative<double>(variantVal)) {
    return std::to_string(std::get<double>(variantVal));
  }
  if (std::holds_alternative<unsigned int>(variantVal)) {
    return std::to_string(std::get<unsigned int>(variantVal));
  }
  if (std::holds_alternative<bool>(variantVal)) {
    return (std::get<bool>(variantVal)) ?
      std::string(run_eval::BOOL_VAL_FALSE_TRUE_DISP_STR[1]) :
      std::string(run_eval::BOOL_VAL_FALSE_TRUE_DISP_STR[0]);
  }
  return "";
}

std::optional<double> RunData::getDataAsDouble(const std::string_view header) const {
  const auto variantVal = headersWData_.at(std::string(header));
  if (std::holds_alternative<double>(variantVal)) {
    return std::get<double>(variantVal);
  }
  return {};
}

std::optional<unsigned int> RunData::getDataAsUInt(const std::string_view header) const {
  const auto variantVal = headersWData_.at(std::string(header));
  if (std::holds_alternative<unsigned int>(variantVal)) {
    return std::get<unsigned int>(variantVal);
  }
  return {};
}

std::optional<bool> RunData::getDataAsBool(const std::string_view header) const {
  const auto variantVal = headersWData_.at(std::string(header));
  if (std::holds_alternative<bool>(variantVal)) {
    return std::get<bool>(variantVal);
  }
  return {};
}

//retrieve pair between a set of parameters and a single parameter
std::optional<std::pair<std::vector<std::string>, double>> RunData::getParamsToRuntime(
  const std::vector<std::string_view>& keyParams, std::string_view valParam) const
{
  std::vector<std::string> keyParamVals;
  for (const auto& keyParam : keyParams) {
    //check if current key params exists as header; return null if not
    if (!(headersWData_.contains(std::string(keyParam)))) {
      return {};
    }
    //add value of key param for first part of pair to return
    keyParamVals.push_back(getDataAsStr(std::string(keyParam)));
  }
    //check if value params exists as header; return null if not
  if (!(headersWData_.contains(std::string(valParam)))) {
    return {};
  }

  //return pair of vector key parameters values with value parameter value for run data
  return std::pair<std::vector<std::string>, double>{keyParamVals, getDataAsDouble(std::string(valParam)).value()};
}
