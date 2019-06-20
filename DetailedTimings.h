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

class DetailedTimings {
public:
	DetailedTimings();
	virtual ~DetailedTimings();

	int totNumTimings = 0;

	virtual void addTimings(DetailedTimings* timingsToAdd)
	{

	}

	virtual void SortTimings()
	{

	}

	virtual void PrintMedianTimings()
	{

	}

	virtual void PrintMedianTimingsToFile(FILE* pFile) {

	}
};

#endif /* DETAILEDTIMINGS_H_ */
