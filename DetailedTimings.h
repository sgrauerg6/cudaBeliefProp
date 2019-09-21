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

//Class to store timings
//Not that there is currently no check that the input segment number for timing is valid; currently is assumed
//that only valid segment numbers (added in the constructor) will be used when adding timings and retrieving median timing
class DetailedTimings {
public:

	//initialize each timing segment
	DetailedTimings(const std::unordered_map<unsigned int, std::string>& timingSegments) : numToString(timingSegments)
	{
		for_each(timingSegments.begin(), timingSegments.end(), [this](const auto& segment) { this->timings[segment.first] = std::vector<double>(); });
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
		timings[numTiming].push_back(segmentTime);
	}

	void printCurrentTimings() const {
		std::cout << *this;
	}

	double getMedianTiming(const unsigned int runSegmentNum) const
	{
		if (timings.at(runSegmentNum).size() > 0)
		{
			std::vector<double> segmentTimingVectCopy(timings.at(runSegmentNum));
			std::sort(segmentTimingVectCopy.begin(), segmentTimingVectCopy.end());
			return (segmentTimingVectCopy[segmentTimingVectCopy.size() / 2]);
		}
		else
		{
			return 0.0;
		}

		return 0.0;
	}

	friend std::ostream& operator<<(std::ostream& os,
			const DetailedTimings& timing);

private:
	std::map<unsigned int, std::vector<double>> timings;
	std::unordered_map<unsigned int, std::string> numToString;
};

#endif /* DETAILEDTIMINGS_H_ */
