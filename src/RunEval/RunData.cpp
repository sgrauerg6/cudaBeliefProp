/*
 * RunData.cpp
 *
 *  Created on: Feb 13, 2024
 *      Author: scott
 * 
 * Implementation of functions in RunData class
 */

#include "RunEvalConstsEnums.h"
#include "RunData.h"

//get header to add...use input header if not yet used
//user original header with number appended if original header is already used
std::string RunData::GetHeaderToAdd(const std::string& in_header) const {
  auto headerToAdd = in_header;
  unsigned int num{0};
  while (std::find(headers_in_order_.cbegin(), headers_in_order_.cend(), headerToAdd) != headers_in_order_.cend()) {
    //add "_{num}" to header if header already in data
    num++;
    headerToAdd = in_header + "_" + std::to_string(num);
  }
  return headerToAdd;
}

//add data with header describing added data
void RunData::AddDataWHeader(const std::string& header, const std::string& data) {
  const auto headerToAdd{GetHeaderToAdd(header)};
  headers_in_order_.push_back(headerToAdd);
  headers_w_data_[headerToAdd] = data;
}

//add data with header describing added data
void RunData::AddDataWHeader(const std::string& header, const char* data) {
  const auto headerToAdd{GetHeaderToAdd(header)};
  headers_in_order_.push_back(headerToAdd);
  headers_w_data_[headerToAdd] = std::string(data);
}

//add data with header describing added data
void RunData::AddDataWHeader(const std::string& header, double data) {
  const auto headerToAdd{GetHeaderToAdd(header)};
  headers_in_order_.push_back(headerToAdd);
  headers_w_data_[headerToAdd] = data;
}

//add data with header describing added data
void RunData::AddDataWHeader(const std::string& header, bool data) {
  const auto headerToAdd{GetHeaderToAdd(header)};
  headers_in_order_.push_back(headerToAdd);
  headers_w_data_[headerToAdd] = data;
}

//add data with header describing added data
void RunData::AddDataWHeader(const std::string& header, unsigned int data) {
  const auto headerToAdd{GetHeaderToAdd(header)};
  headers_in_order_.push_back(headerToAdd);
  headers_w_data_[headerToAdd] = data;
}

//append current RunData with input RunData
void RunData::AppendData(const RunData& rundata) {
  //const auto& rundataMapping = rundata.GetAllData();
  for (const auto& header : rundata.HeadersInOrder()) {
    if (rundata.GetDataAsDouble(header)) {
      AddDataWHeader(header, *(rundata.GetDataAsDouble(header)));
    }
    else if (rundata.GetDataAsUInt(header)) {
      AddDataWHeader(header, *(rundata.GetDataAsUInt(header)));
    }
    else if (rundata.GetDataAsBool(header)) {
      AddDataWHeader(header, *(rundata.GetDataAsBool(header)));
    }
    else {
      AddDataWHeader(header, rundata.GetDataAsStr(header));
    }
  }
}

//get data corresponding to header
std::string RunData::GetDataAsStr(const std::string_view header) const {
  const auto variantVal = headers_w_data_.at(std::string(header));
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
      std::string(run_eval::kBoolValFalseTrueDispStr[1]) :
      std::string(run_eval::kBoolValFalseTrueDispStr[0]);
  }
  return "";
}

//get data as double if variant corresponding to header is double type
//return null if data corresponds to a different data type
std::optional<double> RunData::GetDataAsDouble(const std::string_view header) const {
  const auto variantVal = headers_w_data_.at(std::string(header));
  if (std::holds_alternative<double>(variantVal)) {
    return std::get<double>(variantVal);
  }
  return {};
}

//get data as unsigned integer if variant corresponding to header is unsigned integer type
//return null if data corresponds to a different data type
std::optional<unsigned int> RunData::GetDataAsUInt(const std::string_view header) const {
  const auto variantVal = headers_w_data_.at(std::string(header));
  if (std::holds_alternative<unsigned int>(variantVal)) {
    return std::get<unsigned int>(variantVal);
  }
  return {};
}

//get data as boolean if variant corresponding to header is boolean type
//return null if data corresponds to a different data type
std::optional<bool> RunData::GetDataAsBool(const std::string_view header) const {
  const auto variantVal = headers_w_data_.at(std::string(header));
  if (std::holds_alternative<bool>(variantVal)) {
    return std::get<bool>(variantVal);
  }
  return {};
}
