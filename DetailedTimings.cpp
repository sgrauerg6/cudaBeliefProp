/*
 * DetailedTimings.cpp
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#include "DetailedTimings.h"

std::ostream& operator<<(std::ostream& os, const DetailedTimings& inTimingObj)
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
