/*
 * DetailedTimings.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef DETAILEDTIMINGS_H_
#define DETAILEDTIMINGS_H_

#include <vector>
#include <stdio.h>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <iostream>

class DetailedTimings {
public:

	//initialize each timing segment
	DetailedTimings(const std::unordered_map<unsigned int, std::string>& timingSegments)
	{
		for_each(timingSegments.begin(), timingSegments.end(), [this](const auto& segment) { this->timings[segment] = std::vector<double>(); });
	}

	void resetTiming()
	{
		timings.clear();
	}

	void addToCurrentTimings(const DetailedTimings& inDetailedTimings)
	{
		std::for_each(inDetailedTimings.timings.begin(),
				inDetailedTimings.timings.end(),
				[this](const auto& currentTiming) {
					auto iter = this->timings.find(currentTiming.first);
					if (iter != this->timings.end())
					{
						iter->second.insert(iter->second.end(), currentTiming.second.begin(), currentTiming.second.end());
					}
					else
					{
						this->timings[currentTiming.first] = currentTiming.second;
					}
				});
	}

	//add timing by segment number
	//required that all timing segment numbers/names be initialized from constructor
	void addTiming(const unsigned int numTiming, double segmentTime)
	{
		for (auto& timingMapElement : timings) {
					if (timingMapElement.first.first == numTiming) {
						timingMapElement.second.push_back(segmentTime);
					}
		}
	}

	void sortCurrentTimings() {
		std::for_each(timings.begin(), timings.end(),
				[](auto& currentTiming) {std::sort(currentTiming.second.begin(), currentTiming.second.end());});
	}

	void printCurrentTimings() const {
		std::cout << *this;
	}

	double getMedianTiming(const unsigned int runSegmentNum) const {
		//find first element with input runSegmentNum
		for (auto timingMapElement : timings) {
			if (timingMapElement.first.first == runSegmentNum)
			{
				if (timingMapElement.second.size() > 0)
				{
					std::sort(timingMapElement.second.begin(), timingMapElement.second.end());
					return (timingMapElement.second[timingMapElement.second.size() / 2]);
				}
				else
				{
					return 0.0;
				}
			}
		}

		return 0.0;
	}

	friend std::ostream& operator<<(std::ostream& os,
			const DetailedTimings& timing);

private:
	std::map<std::pair<unsigned int, std::string>, std::vector<double>> timings;
};

#endif /* DETAILEDTIMINGS_H_ */
