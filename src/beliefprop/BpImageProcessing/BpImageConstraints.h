/*
 * BpImageConstraints.h
 *
 * Scott Grauer-Gray
 * November 20, 2024
 *
 * Define constraints for data type in belief propagation processing
 * related to image processing
*/

#ifndef BP_IMAGE_CONSTRAINTS_H_
#define BP_IMAGE_CONSTRAINTS_H_

#include <type_traits>

//constraint for data type when smoothing images
template <typename T>
concept BpImData_t = std::is_same_v<T, float> || std::is_same_v<T, unsigned int>;

#endif //BP_IMAGE_CONSTRAINTS_H_