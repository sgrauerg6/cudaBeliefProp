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

/**
 * @brief Class to store headers with data corresponding to current program run and evaluation
 * 
 */
class RunData {
public:
  /**
   * @brief Add string data with header describing added data
   * 
   * @param header 
   * @param data 
   */
  void AddDataWHeader(const std::string& header, const std::string& data);

  /**
   * @brief Add const char* data with header describing added data
   * 
   * @param header 
   * @param data 
   */
  void AddDataWHeader(const std::string& header, const char* data);

  /**
   * @brief Add double data with header describing added data
   * 
   * @param header 
   * @param data 
   */
  void AddDataWHeader(const std::string& header, double data);

 /**
   * @brief Add boolean data with header describing added data
   * 
   * @param header 
   * @param data 
   */
  void AddDataWHeader(const std::string& header, bool data);

  /**
   * @brief Add unsigned int data with header describing added data
   * 
   * @param header 
   * @param data 
   */
  void AddDataWHeader(const std::string& header, unsigned int data);

  /**
   * @brief Return data headers in order
   * 
   * @return const std::vector<std::string>& 
   */
  const std::vector<std::string>& HeadersInOrder() const { return headers_in_order_; }

  /**
   * @brief Return data mapped to corresponding headers
   * 
   * @return const std::map<std::string, std::variant<unsigned int, double, bool, std::string>>& 
   */
  const std::map<std::string, std::variant<unsigned int, double, bool, std::string>>& GetAllData() const {
    return headers_w_data_; }

  /**
   * @brief Return whether or not there is data corresponding to a specific header
   * 
   * @param header 
   * @return true 
   * @return false 
   */
  bool IsData(const std::string_view header) const { 
    return (std::find(headers_in_order_.cbegin(), headers_in_order_.cend(),
                      std::string(header)) != headers_in_order_.cend()); }

  /**
   * @brief Get data corresponding to header as a string
   * Returns data as string regardless of underlying data type
   * 
   * @param header 
   * @return std::string 
   */
  std::string GetDataAsStr(const std::string_view header) const;
  
  /**
   * @brief Get data corresponding to header as double
   * Return null if data corresponds to a different data type
   * 
   * @param header 
   * @return std::optional<double> 
   */
  std::optional<double> GetDataAsDouble(const std::string_view header) const;

  /**
   * @brief Get data corresponding to header as unsigned int
   * Return null if data corresponds to a different data type
   * 
   * @param header 
   * @return std::optional<double> 
   */
  std::optional<unsigned int> GetDataAsUInt(const std::string_view header) const;

  /**
   * @brief Get data corresponding to header as boolean
   * Return null if data corresponds to a different data type
   * 
   * @param header 
   * @return std::optional<double> 
   */
  std::optional<bool> GetDataAsBool(const std::string_view header) const;

  /**
   * @brief Append current RunData with input RunData
   * 
   * @param rundata 
   */
  void AppendData(const RunData& rundata);
  
  /**
   * @brief Overloaded << operator for output to stream
   * 
   * @param os 
   * @param run_data 
   * @return std::ostream& 
   */
  friend std::ostream& operator<<(std::ostream& os, const RunData& run_data);

private:
  /**
   * @brief Get header to add...returns input header if not yet used
   * Generates and returns input header with number appended if header already used
   * 
   * @param in_header 
   * @return std::string 
   */
  std::string GetHeaderToAdd(const std::string& in_header) const;

  /**
   * @brief Data stored as map between header and value corresponding to header that can
   * be of type unsigned int, double, bool, or string
   * 
   */
  std::map<std::string, std::variant<unsigned int, double, bool, std::string>> headers_w_data_;
  
  /**
   * @brief Headers corresponding to run data in order
   * 
   */
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