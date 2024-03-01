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
  std::cout << "addDataWHeader: " << header << " " << data << std::endl;
  const auto origHeader = std::string(header);
  auto headerToAdd = origHeader;
  unsigned int num{0};
  while (std::find(headersInOrder_.begin(), headersInOrder_.end(), headerToAdd) != headersInOrder_.end()) {
    //add "_{num}" to header if header already in data
    num++;
    headerToAdd = origHeader + "_" + std::to_string(num);
  }
  headersInOrder_.push_back(headerToAdd);
  std::cout << "headerToAdd: " << headerToAdd << std::endl;
  headersWData_[headerToAdd] = data;
  std::cout << "data: " << headersWData_[headerToAdd] << std::endl;
}

//append current RunData with input RunData
void RunData::appendData(const RunData& inRunData) {
  std::cout << "5a" << std::endl;
  const auto& inRunDataMapping = inRunData.getAllData();
  for (const auto& header : inRunData.getHeadersInOrder()) {
  std::cout << "5b " << header << std::endl;
    addDataWHeader(header, inRunDataMapping.at(header));
  std::cout << "5c" << std::endl;
  }
}