/*
 * RunData.h
 *
 *  Created on: May 16, 2023
 *      Author: scott
 * 
 *  Class to store headers with data corresponding to current program run.
 */

#ifndef RUN_DATA_H
#define RUN_DATA_H

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <iostream>

class RunData {
public:
  void addDataWHeader(const std::string& header, const std::string& data) {
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

  const std::vector<std::string>& getHeadersInOrder() const { return headersInOrder_; }
  const std::map<std::string, std::string>& getAllData() const { return headersWData_; }
  bool isData(const std::string& header) const { 
    return (std::find(headersInOrder_.begin(), headersInOrder_.end(), header) != headersInOrder_.end()); }
  const std::string getData(const std::string& header) const { return headersWData_.at(header); }
  void appendData(const RunData& inRunData) {
    const auto& inRunDataMapping = inRunData.getAllData();
    for (const auto& header : inRunData.getHeadersInOrder()) {
      addDataWHeader(header, inRunDataMapping.at(header));
    }
  }

private:
  //data stored as mapping between header and data value corresponding to headers
  std::map<std::string, std::string> headersWData_;
  
  //headers ordered from first header added to last header added
  std::vector<std::string> headersInOrder_;
};

#endif //RUN_DATA_H