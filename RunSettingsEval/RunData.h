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
#include <optional>

class RunData {
public:
  //add data with header describing added data
  void addDataWHeader(const std::string& header, const std::string& data);

  //return data headers in order
  const std::vector<std::string>& getHeadersInOrder() const { return headersInOrder_; }

  //return data mapped to corresponding headers
  const std::map<std::string, std::string>& getAllData() const { return headersWData_; }

  //return whether or not there is data corresponding to a specific header
  bool isData(const std::string& header) const { 
    return (std::find(headersInOrder_.begin(), headersInOrder_.end(), header) != headersInOrder_.end()); }

  //get data corresponding to ,
  const std::string getData(const std::string& header) const { return headersWData_.at(header); }
  
  //append current RunData with input RunData
  void appendData(const RunData& inRunData);

  //retrieve pair between a set of parameters and a single parameter
  std::optional<std::pair<std::vector<std::string>, std::string>> getParamsToParamRunData(
    const std::vector<std::string>& keyParams, const std::string& valParam) const
  {
    std::vector<std::string> keyParamVals;
    for (const auto& keyParam : keyParams) {
      //check if current key params exists as header; return null if not
      if (!(headersWData_.contains(keyParam))) {
        return {};
      }
      //add value of key param for first part of pair to return
      keyParamVals.push_back(headersWData_.at(keyParam));
    }
      //check if value params exists as header; return null if not
    if (!(headersWData_.contains(valParam))) {
      return {};
    }

    //return pair of vector key parameters values with value parameter value for run data
    return std::pair<std::vector<std::string>, std::string>{keyParamVals, headersWData_.at(valParam)};
  }

private:
  //data stored as mapping between header and data value corresponding to headers
  std::map<std::string, std::string> headersWData_;
  
  //headers ordered from first header added to last header added
  std::vector<std::string> headersInOrder_;
};

#endif //RUN_DATA_H