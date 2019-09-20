/*
 * DetailedTimings.cpp
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#include "DetailedTimings.h"

DetailedTimings::DetailedTimings() {
	// TODO Auto-generated constructor stub

}

DetailedTimings::~DetailedTimings() {
	// TODO Auto-generated destructor stub
}

std::ostream& operator<<(std::ostream& os, const DetailedTimings& inTimingObj)
{
	os << "Median Timings\n";
	std::for_each(inTimingObj.timings.begin(), inTimingObj.timings.end(), [&os](auto currentTiming) {
		std::sort(currentTiming.second.begin(), currentTiming.second.end());
		os << currentTiming.first.second << ": " << currentTiming.second[currentTiming.second.size() / 2] << std::endl; });
	return os;
}
