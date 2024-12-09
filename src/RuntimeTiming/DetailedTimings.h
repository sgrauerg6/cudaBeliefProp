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
 * @file DetailedTimings.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef DETAILEDTIMINGS_H_
#define DETAILEDTIMINGS_H_

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <iterator>
#include <map>
#include <iostream>
#include <chrono>
#include <ranges>
#include "RunEval/RunData.h"
#include "RunEval/RunEvalConstsEnums.h"

/**
 * @brief Class to store timings of one or more segments taken during the run(s)
 * of an implementation or across multiple implementations.
 * Index for timing segments must be enum type.
 * 
 * @tparam T 
 */
template <typename T>
requires std::is_enum_v<T>
class DetailedTimings {
public:
  /**
   * @brief Construct a new DetailedTimings object where
   * each timing segment is initialized
   * 
   * @param timing_segment_names 
   */
  DetailedTimings(const std::unordered_map<T, std::string_view>& timing_segment_names);

  /**
   * @brief Reset all timings
   * 
   */
  void ResetTiming() { segment_timings_.clear(); }

  /**
   * @brief Add instance of DetailedTimings to current DetailedTimings
   * 
   * @param in_detailed_timings 
   */
  void AddToCurrentTimings(const DetailedTimings& in_detailed_timings);

  /**
   * @brief Add timing by segment index
   * 
   * @param timing_segment 
   * @param segment_time 
   */
  void AddTiming(const T timing_segment, const std::chrono::duration<double>& segment_time) {
    segment_timings_[timing_segment].push_back(segment_time);
  }

  /**
   * @brief Get median timing for a specified segment that may have been run multiple times
   * 
   * @param run_segment_index 
   * @return std::chrono::duration<double> 
   */
  std::chrono::duration<double> MedianTiming(const T run_segment_index) const;

  /**
   * @brief Return current timing data as a RunData object for evaluation
   * 
   * @return RunData 
   */
  RunData AsRunData() const;

private:
  std::map<T, std::vector<std::chrono::duration<double>>> segment_timings_;
  const std::unordered_map<T, std::string_view> timing_seg_to_str_;
};

//initialize each timing segment
template <typename T>
requires std::is_enum_v<T>
DetailedTimings<T>::DetailedTimings(
  const std::unordered_map<T, std::string_view>& timing_segment_names) :
  timing_seg_to_str_{timing_segment_names}
{
  std::ranges::transform(timing_seg_to_str_, std::inserter(segment_timings_, segment_timings_.end()),
    [](const auto& segment) -> std::pair<T, std::vector<std::chrono::duration<double>>> {
      return {segment.first, std::vector<std::chrono::duration<double>>()}; });
}

//add instance of DetailedTimings to current DetailedTimings
template <typename T>
requires std::is_enum_v<T>
void DetailedTimings<T>::AddToCurrentTimings(
  const DetailedTimings& in_detailed_timings)
{
  std::ranges::for_each(in_detailed_timings.segment_timings_,
    [this](const auto& current_timing) {
      auto iter = this->segment_timings_.find(current_timing.first);
      if (iter != this->segment_timings_.cend()) {
        iter->second.insert(iter->second.cend(), 
                            current_timing.second.cbegin(),
                            current_timing.second.cend());
      }
      else {
        this->segment_timings_[current_timing.first] = current_timing.second;
      }
    });
}

//get median timing for a specified segment that may have been run multiple times
template <typename T>
requires std::is_enum_v<T>
std::chrono::duration<double> DetailedTimings<T>::MedianTiming(
  const T run_segment_index) const
{
  if (segment_timings_.at(run_segment_index).size() > 0) {
    std::vector<std::chrono::duration<double>> segment_timing_vect_copy(
      segment_timings_.at(run_segment_index));
    //get median timing across runs
    std::ranges::nth_element(
      segment_timing_vect_copy,
      segment_timing_vect_copy.begin() + segment_timing_vect_copy.size() / 2);
    return (segment_timing_vect_copy[segment_timing_vect_copy.size() / 2]);
  }
  else {
    return std::chrono::duration<double>();
  }
}

//return current timing data as a RunData object for output
template <typename T>
requires std::is_enum_v<T>
RunData DetailedTimings<T>::AsRunData() const {
  RunData timings_run_data;
  std::ranges::for_each(segment_timings_,
    [this, &timings_run_data](auto current_timing) {
      //generate header for timing data
      const std::string header = 
        std::string(timing_seg_to_str_.at(current_timing.first)) + " " +
        std::string(run_eval::kMedianOfTestRunsDesc);
      //get median timing across runs
      std::ranges::nth_element(
        current_timing.second,
        current_timing.second.begin() + current_timing.second.size() / 2);
      if (current_timing.second.size() > 0) {
        timings_run_data.AddDataWHeader(
          header,
          current_timing.second[current_timing.second.size() / 2].count());
      }
      else {
        timings_run_data.AddDataWHeader(header, "No timings"); 
      }
    });
  return timings_run_data;
}

#endif /* DETAILEDTIMINGS_H_ */
