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
#include <map>
#include <iostream>

class DetailedTimings {
public:

		std::map<std::pair<unsigned int, std::string>, std::vector<double>> timings;

		int totNumTimings = 0;

		DetailedTimings();
			virtual ~DetailedTimings();

			void addToCurrentTimings(DetailedTimings& inDetailedTimings)
			{
				std::for_each(inDetailedTimings.timings.begin(), inDetailedTimings.timings.end(), [this](const auto& currentTiming) {
					auto iter = this->timings.find(currentTiming.first);
									if (iter != this->timings.end())
									{
										iter->second.insert(iter->second.end(), currentTiming.second.begin(), currentTiming.second.end());
									}
									else
									{
										this->timings[currentTiming.first] = currentTiming.second;
									}
				} );
			}

			void addTiming(std::pair<std::pair<unsigned int, std::string>, double> inTiming)
			{
				auto iter = timings.find(inTiming.first);
				if (iter != timings.end())
				{
					iter->second.push_back(inTiming.second);
				}
				else
				{
					timings[inTiming.first] = std::vector<double>(1, inTiming.second);
				}
			}

			void sortCurrentTimings()
			{
				std::for_each(timings.begin(), timings.end(), [](auto& currentTiming) { std::sort(currentTiming.second.begin(), currentTiming.second.end()); });
			}

			void printCurrentTimings() const
			{
				printf("Median Timings\n");
				(timings.size() > 0) ?	printf("Number of timings: %d\n", timings.begin()->second.size()) : printf("No timings recorded\n");
				//sort current timings, then print
				std::for_each(timings.begin(), timings.end(), [](auto currentTiming) {
					std::sort(currentTiming.second.begin(), currentTiming.second.end());
					printf("%s : %f\n", currentTiming.first.second.c_str(), currentTiming.second[currentTiming.second.size() / 2]); });
			}

			double getMedianTiming(const unsigned int runSegmentNum) const
			{
				//find first element with input runSegmentNum
				for (const auto& timingMapElement : timings)
				{
					if (timingMapElement.first.first == runSegmentNum)
					{
						return getMedianTiming(timingMapElement.first);
					}
				}

				return 0.0;
			}

			double getMedianTiming(const std::string& runSegmentString) const
			{
				//find first element with input runSegmentString
				for (const auto& timingMapElement : timings)
				{
					if (timingMapElement.first.second == runSegmentString)
					{
						return getMedianTiming(timingMapElement.first);
					}
				}

				return 0.0;
			}

			double getMedianTiming(std::pair<unsigned int, std::string> runSegment) const
			{
				auto iter = timings.find(runSegment);
				if (iter != timings.end())
				{
					std::vector<double> timingVectorCopy(iter->second);
					std::sort(timingVectorCopy.begin(), timingVectorCopy.end());
					return (timingVectorCopy[timingVectorCopy.size() / 2]);
				}
				else
				{
					return 0.0;
				}
			}

		friend std::ostream& operator<<(std::ostream& os, const DetailedTimings& timing);
};

#endif /* DETAILEDTIMINGS_H_ */
