/*
 * RunData.h
 *
 *  Created on: May 16, 2023
 *      Author: scott
 * 
 *  Class to store headers with data corresponding to current program run.
 */

#ifndef RUN_DATA_H
#define RUN_DATA_H

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <iostream>

class RunData {
public:
    void addDataWHeader(const std::string& header, const std::string& data) {
        headersWData[header] = data;
        if (std::find(headersInOrder.begin(), headersInOrder.end(), header) == headersInOrder.end()) {
            headersInOrder.push_back(header);
        }
    }

    const std::vector<std::string>& getHeadersInOrder() const { return headersInOrder; }
    const std::map<std::string, std::string>& getAllData() const { return headersWData; }
    const std::string getData(const std::string& header) const { return headersWData.at(header); }
    void appendData(const RunData& inRunData) { 
        const auto& inHeaders = inRunData.getHeadersInOrder();
        const auto& inData = inRunData.getAllData();
        headersInOrder.insert(headersInOrder.end(), inHeaders.begin(), inHeaders.end());
        for (const auto& dataWHeader : inData) {
            headersWData[dataWHeader.first] = dataWHeader.second;
        }
    }

private:
    //data stored as mapping between header and data value corresponding to headers
    std::map<std::string, std::string> headersWData;
    
    //headers ordered from first header added to last header added
    std::vector<std::string> headersInOrder;
};

#endif //RUN_DATA_H