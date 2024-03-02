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
#include <variant>

namespace run_eval {
  //define string for display of "true" and "false" values of bool value
  constexpr std::array<std::string_view, 2> BOOL_VAL_FALSE_TRUE_DISP_STR{"NO", "YES"};
};

class RunData {
public:
  //add data with header describing added data
  void addDataWHeader(const std::string& header, const std::string& data);

  //add data with header describing added data
  void addDataWHeader(const std::string& header, double data);

  //add data with header describing added data
  void addDataWHeader(const std::string& header, bool data);

  //add data with header describing added data
  void addDataWHeader(const std::string& header, unsigned int data);

  //return data headers in order
  const std::vector<std::string>& getHeadersInOrder() const { return headersInOrder_; }

  //return data mapped to corresponding headers
  const std::map<std::string, std::variant<unsigned int, double, bool, std::string>>& getAllData() const { return headersWData_; }

  //return whether or not there is data corresponding to a specific header
  bool isData(const std::string_view header) const { 
    return (std::find(headersInOrder_.cbegin(), headersInOrder_.cend(), std::string(header)) != headersInOrder_.cend()); }

  //get data corresponding to header as a string
  //returns data as string regardless of underlying data type
  std::string getDataAsStr(const std::string_view header) const;

  //get data as specified type if variant corresponding to header is specified type
  //return null if data corresponds to a different data type
  std::optional<double> getDataAsDouble(const std::string_view header) const;
  std::optional<unsigned int> getDataAsUInt(const std::string_view header) const;
  std::optional<bool> getDataAsBool(const std::string_view header) const;

  //append current RunData with input RunData
  void appendData(const RunData& inRunData);

  //retrieve pair between a set of parameters and a single parameter
  std::optional<std::pair<std::vector<std::string>, double>> getParamsToRuntime(
    const std::vector<std::string_view>& keyParams, std::string_view valParam) const;

private:
  //get header to add...use input header if not yet used
  //user original header with number appended if original header is already used
  std::string getHeaderToAdd(const std::string& inHeader) const;

  //data stored as mapping between header and data value corresponding to headers
  std::map<std::string, std::variant<unsigned int, double, bool, std::string>> headersWData_;
  
  //headers ordered from first header added to last header added
  std::vector<std::string> headersInOrder_;
};

#endif //RUN_DATA_H