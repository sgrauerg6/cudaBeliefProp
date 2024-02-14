/*
 * RunData.cpp
 *
 *  Created on: Feb 13, 2024
 *      Author: scott
 * 
 * Implementation of functions in RunData class
 */

#include "RunData.h"

//add data with header describing added data
void RunData::addDataWHeader(const std::string& header, const std::string& data) {
  const auto origHeader = header;
  auto headerToAdd = origHeader;
  unsigned int num{0};
  while (std::find(headersInOrder_.begin(), headersInOrder_.end(), headerToAdd) != headersInOrder_.end()) {
    //add "_{num}" to header if header already in data
    num++;
    headerToAdd = origHeader + "_" + std::to_string(num);
  }
  headersInOrder_.push_back(headerToAdd);
  headersWData_[headerToAdd] = data;
}

//append current RunData with input RunData
void RunData::appendData(const RunData& inRunData) {
  const auto& inRunDataMapping = inRunData.getAllData();
  for (const auto& header : inRunData.getHeadersInOrder()) {
    addDataWHeader(header, inRunDataMapping.at(header));
  }
}