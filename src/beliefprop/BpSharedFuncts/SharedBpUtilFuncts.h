/*
Copyright (C) 2024 Scott Grauer-Gray

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/**
 * @file SharedBpUtilFuncts.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef SHARED_BP_UTIL_FUNCTS_H_
#define SHARED_BP_UTIL_FUNCTS_H_

#include "RunSettingsParams/RunSettings.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunImp/UtilityFuncts.h"
#include "BpRunProcessing/BpConstsEnumsAliases.h"
#include "BpRunProcessing/BpRunUtils.h"
#include "BpRunProcessing/BpLevel.h"
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include "SharedBpUtilFuncts.h"

namespace beliefprop {

/**
 * @brief Checks if the current point is within the image bounds
 * Assumed that input x/y vals are above zero since their unsigned int so no need for >= 0 check
 * 
 * @param x_val 
 * @param y_val 
 * @param width 
 * @param height 
 *  
 */
ARCHITECTURE_ADDITION inline bool WithinImageBounds(
  unsigned int x_val, unsigned int y_val,
  unsigned int width, unsigned int height)
{
  return ((x_val < width) && (y_val < height));
}

/**
 * @brief Retrieve the current 1-D index value of the given point at the given
 * disparity in the data cost and message data
 * 
 * @param x_val 
 * @param y_val 
 * @param width 
 * @param height 
 * @param current_disparity 
 * @param total_num_disp_vals 
 * @param offset_data 
 *  
 */
ARCHITECTURE_ADDITION inline unsigned int RetrieveIndexInDataAndMessage(unsigned int x_val, unsigned int y_val,
  unsigned int width, unsigned int height, unsigned int current_disparity, unsigned int total_num_disp_vals,
  unsigned int offset_data = 0u)
{
  if constexpr (beliefprop::kOptimizedIndexingSetting) {
    //indexing is performed in such a way so that the memory accesses as coalesced as much as possible
    return (y_val * width * total_num_disp_vals + width * current_disparity + x_val) + offset_data;
  }
  else {
    return ((y_val * width + x_val) * total_num_disp_vals + current_disparity);
  }
}

}

#endif //SHARED_BP_UTIL_FUNCTS_H_