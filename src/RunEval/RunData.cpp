/*
Copyright (C) 2024 Scott Grauer-Gray

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/**
 * @file RunData.cpp
 * @author Scott Grauer-Gray
 * @brief Implementation of functions in RunData class
 * 
 * @copyright Copyright (c) 2024
 */

#include "RunEvalConstsEnums.h"
#include "RunData.h"

//get header to add...use input header if not yet used
//use original header with number appended if original header is already used
std::string RunData::GetHeaderToAdd(const std::string& in_header) const {
  auto header_to_add = in_header;
  unsigned int num{0};
  while (std::any_of(headers_in_order_.cbegin(),
                     headers_in_order_.cend(),
                     [&header_to_add](const auto& ordered_header){
                       return (ordered_header == header_to_add);
                     }))
  {
    //add "_{num}" to header if header already in data
    num++;
    header_to_add = in_header + "_" + std::to_string(num);
  }
  return header_to_add;
}

//add data with header describing added data
void RunData::AddDataWHeader(const std::string& header, const std::string& data) {
  const auto header_to_add{GetHeaderToAdd(header)};
  headers_in_order_.push_back(header_to_add);
  headers_w_data_[header_to_add] = data;
}

//add data with header describing added data
void RunData::AddDataWHeader(const std::string& header, const char* data) {
  const auto header_to_add{GetHeaderToAdd(header)};
  headers_in_order_.push_back(header_to_add);
  headers_w_data_[header_to_add] = std::string(data);
}

//add data with header describing added data
void RunData::AddDataWHeader(const std::string& header, double data) {
  const auto header_to_add{GetHeaderToAdd(header)};
  headers_in_order_.push_back(header_to_add);
  headers_w_data_[header_to_add] = data;
}

//add data with header describing added data
void RunData::AddDataWHeader(const std::string& header, bool data) {
  const auto header_to_add{GetHeaderToAdd(header)};
  headers_in_order_.push_back(header_to_add);
  headers_w_data_[header_to_add] = data;
}

//add data with header describing added data
void RunData::AddDataWHeader(const std::string& header, unsigned int data) {
  const auto header_to_add{GetHeaderToAdd(header)};
  headers_in_order_.push_back(header_to_add);
  headers_w_data_[header_to_add] = data;
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
