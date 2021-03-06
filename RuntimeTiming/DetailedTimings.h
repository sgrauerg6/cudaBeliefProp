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
#include "DetailedTimingBPConsts.h"

//Class to store timings
//Not that there is currently no check that the input segment index for timing is valid; currently is assumed
//that only valid segment numbers (added in the constructor) will be used when adding timings and retrieving median timing
//If enum is used as T and every enum value is mapped with a string in the constructor parameter input, then no issue with invalid index
template <class T>
class DetailedTimings {
public:

	//initialize each timing segment
	DetailedTimings(const std::unordered_map<T, std::string>& timingSegments = timingNames_BP) : numToString(timingSegments) {
		std::for_each(timingSegments.begin(), timingSegments.end(), [this](const auto& segment) { this->timings[segment.first] = std::vector<double>(); });
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
	//required that all timing segment numbers/names be initialized from constructor
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
		std::for_each(inTimingObj.timings.begin(), inTimingObj.timings.end(), [&os, &inTimingObj](auto currentTiming) {
			std::sort(currentTiming.second.begin(), currentTiming.second.end());
			os << inTimingObj.numToString.at(currentTiming.first);
			if (currentTiming.second.size() > 0) {
				os << " (" << currentTiming.second.size() << " timings) : " << currentTiming.second[currentTiming.second.size() / 2] << std::endl; }
			else {
				os << " (No timings) : No timings" << std::endl; }
			});
		return os;
	}

private:
	std::map<T, std::vector<double>> timings;
	std::unordered_map<T, std::string> numToString;
};

#endif /* DETAILEDTIMINGS_H_ */
