/*
 * DetailedTimings.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef DETAILEDTIMINGS_H_
#define DETAILEDTIMINGS_H_

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <iostream>
#include <chrono>
#include <ranges>
#include "RunSettingsEval/RunData.h"

//Class to store timings
//Index for timing segments must be enum type (either scoped via "enum class" or not scoped)
template <typename T>
requires std::is_enum_v<T>
class DetailedTimings {
public:
  //initialize each timing segment
  DetailedTimings(const std::unordered_map<T, std::string>& timingSegments);

  //reset all timings
  void resetTiming() { segmentTimings_.clear(); }

  //add instance of DetailedTimings to current DetailedTimings
  void addToCurrentTimings(const DetailedTimings& inDetailedTimings);

  //add timing by segment index
  void addTiming(const T timingSegment, const std::chrono::duration<double>& segmentTime) {
    segmentTimings_[timingSegment].push_back(segmentTime);
  }

  //get median timing for a specified segment that may have been run multiple times
  std::chrono::duration<double> getMedianTiming(const T runSegmentIndex) const;

  //return current timing data as a RunData object for output
  RunData runData() const;

private:
  std::map<T, std::vector<std::chrono::duration<double>>> segmentTimings_;
  const std::unordered_map<T, std::string> timingSegToStr_;
};

//initialize each timing segment
template <typename T>
requires std::is_enum_v<T>
DetailedTimings<T>::DetailedTimings(const std::unordered_map<T, std::string>& timingSegments) : timingSegToStr_{timingSegments} {
  std::ranges::for_each(timingSegments,
    [this](const auto& segment) {
      this->segmentTimings_[segment.first] = std::vector<std::chrono::duration<double>>(); 
    });
}

//add instance of DetailedTimings to current DetailedTimings
template <typename T>
requires std::is_enum_v<T>
void DetailedTimings<T>::addToCurrentTimings(const DetailedTimings& inDetailedTimings)
{
  std::ranges::for_each(inDetailedTimings.segmentTimings_,
    [this](const auto& currentTiming) {
      auto iter = this->segmentTimings_.find(currentTiming.first);
      if (iter != this->segmentTimings_.end()) {
        iter->second.insert(iter->second.end(), currentTiming.second.begin(), currentTiming.second.end());
      }
      else {
        this->segmentTimings_[currentTiming.first] = currentTiming.second;
      }
    });
}

//get median timing for a specified segment that may have been run multiple times
template <typename T>
requires std::is_enum_v<T>
std::chrono::duration<double> DetailedTimings<T>::getMedianTiming(const T runSegmentIndex) const {
  if (segmentTimings_.at(runSegmentIndex).size() > 0) {
    std::vector<std::chrono::duration<double>> segmentTimingVectCopy(segmentTimings_.at(runSegmentIndex));
    //get median timing across runs
    std::ranges::nth_element(segmentTimingVectCopy, segmentTimingVectCopy.begin() + segmentTimingVectCopy.size() / 2);
    return (segmentTimingVectCopy[segmentTimingVectCopy.size() / 2]);
  }
  else {
    return std::chrono::duration<double>();
  }
}

//return current timing data as a RunData object for output
template <typename T>
requires std::is_enum_v<T>
RunData DetailedTimings<T>::runData() const {
  RunData timingsRunData;
  std::ranges::for_each(segmentTimings_,
    [this, &timingsRunData](auto currentTiming) {
      //get median timing across runs
      std::ranges::nth_element(currentTiming.second, currentTiming.second.begin() + currentTiming.second.size() / 2);
      std::string headerStart = timingSegToStr_.at(currentTiming.first);
      if (currentTiming.second.size() > 0) {
        timingsRunData.addDataWHeader(headerStart + " (" + std::to_string(currentTiming.second.size()) +
          " timings)", std::to_string(currentTiming.second[currentTiming.second.size() / 2].count()));
      }
      else {
        timingsRunData.addDataWHeader(headerStart + " (No timings) ", "No timings"); 
      }
    });
  return timingsRunData;
}

#endif /* DETAILEDTIMINGS_H_ */
