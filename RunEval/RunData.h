/*
 * RunData.h
 *
 *  Created on: May 16, 2023
 *      Author: scott
 * 
 *  Class to store headers with data corresponding to current program run and evaluation.
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

//class to store headers with data corresponding to current program run and evaluation
class RunData {
public:
  //add data with header describing added data
  void AddDataWHeader(const std::string& header, const std::string& data);

  //add data with header describing added data
  void AddDataWHeader(const std::string& header, const char* data);

  //add data with header describing added data
  void AddDataWHeader(const std::string& header, double data);

  //add data with header describing added data
  void AddDataWHeader(const std::string& header, bool data);

  //add data with header describing added data
  void AddDataWHeader(const std::string& header, unsigned int data);

  //return data headers in order
  const std::vector<std::string>& HeadersInOrder() const { return headers_in_order_; }

  //return data mapped to corresponding headers
  const std::map<std::string, std::variant<unsigned int, double, bool, std::string>>& GetAllData() const {
    return headers_w_data_; }

  //return whether or not there is data corresponding to a specific header
  bool IsData(const std::string_view header) const { 
    return (std::find(headers_in_order_.cbegin(), headers_in_order_.cend(),
                      std::string(header)) != headers_in_order_.cend()); }

  //get data corresponding to header as a string
  //returns data as string regardless of underlying data type
  std::string GetDataAsStr(const std::string_view header) const;

  //get data as specified type if variant corresponding to header is specified type
  //return null if data corresponds to a different data type
  std::optional<double> GetDataAsDouble(const std::string_view header) const;
  std::optional<unsigned int> GetDataAsUInt(const std::string_view header) const;
  std::optional<bool> GetDataAsBool(const std::string_view header) const;

  //append current RunData with input RunData
  void AppendData(const RunData& rundata);

  //retrieve pair between a set of parameters and a single parameter
  std::optional<std::pair<std::vector<std::string>, double>> GetParamsToRuntime(
    const std::vector<std::string_view>& key_params, std::string_view val_param) const;
  
  //overloaded << operator for output to stream
  friend std::ostream& operator<<(std::ostream& os, const RunData& run_data);

private:
  //get header to add...returns input header if not yet used
  //generates and returns input header with number appended if header already used
  std::string GetHeaderToAdd(const std::string& in_header) const;

  //data stored as map between header and value corresponding to header that can
  //be of type unsigned int, double, bool, or string
  std::map<std::string, std::variant<unsigned int, double, bool, std::string>> headers_w_data_;
  
  //headers in order
  std::vector<std::string> headers_in_order_;
};

//overloaded << operator for output to stream
inline std::ostream& operator<<(std::ostream& os, const RunData& run_data) {
  //add each header with corresponding data separated by ':' to stream in a separate line
  for (const auto& header : run_data.headers_in_order_) {
    os << header << ": " << run_data.GetDataAsStr(header) << std::endl;
  }
  return os;
}

#endif //RUN_DATA_H