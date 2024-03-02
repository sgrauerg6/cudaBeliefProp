//BpTypeConstraints.h
//
//Define constraints for data type in belief propagation processing

#ifndef BP_TYPE_CONSTRAINTS_H_
#define BP_TYPE_CONSTRAINTS_H_

//constraint for data type when smoothing images
template <typename T>
concept BpImData_t = std::is_same_v<T, float> || std::is_same_v<T, unsigned int>;

#endif //BP_TYPE_CONSTRAINTS_H_