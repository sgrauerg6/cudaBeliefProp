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
 * @file BpImageConstraints.h
 * @author Scott Grauer-Gray
 * @brief Define constraint for data type in belief propagation processing
 * related to image processing
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef BP_IMAGE_CONSTRAINTS_H_
#define BP_IMAGE_CONSTRAINTS_H_

#include <type_traits>

//constraint for data type when smoothing images
template <typename T>
concept BpImData_t = std::is_same_v<T, float> || std::is_same_v<T, unsigned int>;

#endif //BP_IMAGE_CONSTRAINTS_H_