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
#include "RunSettingsEval/RunData.h"

//Class to store timings
//Index for timing segments must be enum type (either scoped via "enum class" or not scoped)
template <typename T>
requires std::is_enum_v<T>
class DetailedTimings {
public:

  //initialize each timing segment
  DetailedTimings(const std::unordered_map<T, std::string>& timingSegments) : numToString(timingSegments) {
    std::for_each(timingSegments.begin(), timingSegments.end(),
      [this](const auto& segment) {
        this->timings[segment.first] = std::vector<double>(); 
      });
  }

  void resetTiming() {
    timings.clear();
  }

  void addToCurrentTimings(const DetailedTimings& inDetailedTimings)
  {
    std::for_each(inDetailedTimings.timings.begin(), inDetailedTimings.timings.end(),
      [this](const auto& currentTiming) {
        auto iter = this->timings.find(currentTiming.first);
        if (iter != this->timings.end()) {
          iter->second.insert(iter->second.end(), currentTiming.second.begin(), currentTiming.second.end());
        }
        else {
          this->timings[currentTiming.first] = currentTiming.second;
        }
      });
  }

  //add timing by segment index
  void addTiming(const T timingIndex, const double segmentTime) {
    timings[timingIndex].push_back(segmentTime);
  }

  void printCurrentTimings() const {
    std::cout << *this;
  }

  double getMedianTiming(const T runSegmentIndex) const {
    if (timings.at(runSegmentIndex).size() > 0) {
      std::vector<double> segmentTimingVectCopy(timings.at(runSegmentIndex));
      std::sort(segmentTimingVectCopy.begin(), segmentTimingVectCopy.end());
      return (segmentTimingVectCopy[segmentTimingVectCopy.size() / 2]);
    }
    else {
      return 0.0;
    }
  }

  friend std::ostream& operator<<(std::ostream& os,
    const DetailedTimings& inTimingObj)
  {
    os << "Median Timings\n";
    std::for_each(inTimingObj.timings.begin(), inTimingObj.timings.end(),
      [&os, &inTimingObj](auto currentTiming) {
        std::sort(currentTiming.second.begin(), currentTiming.second.end());
        os << inTimingObj.numToString.at(currentTiming.first);
        if (currentTiming.second.size() > 0) {
          os << " (" << currentTiming.second.size() << " timings) : " <<
                currentTiming.second[currentTiming.second.size() / 2] << std::endl;
        }
        else {
          os << " (No timings) : No timings" << std::endl;
        }
      });
    return os;
  }

  RunData runData() const {
    RunData timingsRunData;
    std::for_each(timings.begin(), timings.end(),
      [this, &timingsRunData](auto currentTiming) {
        std::sort(currentTiming.second.begin(), currentTiming.second.end());
        std::string headerStart = numToString.at(currentTiming.first);
        if (currentTiming.second.size() > 0) {
          timingsRunData.addDataWHeader(headerStart + " (" + std::to_string(currentTiming.second.size()) +
            " timings)", std::to_string(currentTiming.second[currentTiming.second.size() / 2]));
        }
        else {
          timingsRunData.addDataWHeader(headerStart + " (No timings) ", "No timings"); 
        }
      });
    return timingsRunData;
  }

private:
  std::map<T, std::vector<double>> timings;
  std::unordered_map<T, std::string> numToString;
};

#endif /* DETAILEDTIMINGS_H_ */
