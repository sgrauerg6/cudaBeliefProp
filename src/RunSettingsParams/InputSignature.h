/*
 * InputSignature.h
 *
 *  Created on: Nov 22, 2024
 *      Author: scott
 * 
 *  Class for defining input signature for evaluation run that consists of evaluation set number,
 *  data type, and whether or not to use templated loop iteration count.
 */

#ifndef INPUT_SIGNATURE_H_
#define INPUT_SIGNATURE_H_

#include <string_view>
#include <string>
#include <ostream>
#include <limits>
#include <optional>
#include "RunEval/RunEvalConstsEnums.h"

/**
 * @brief Class defines input signature for evaluation run that contains evaluation set
 * number, data type, and whether or not to use templated loop iteration count
 * 
 */
class InputSignature {
public:
  /**
   * @brief Constructor to generate evaluation input signature from string array with
   * strings corresponding to each part
   * 
   * @param in_sig_strings 
   */
  InputSignature(const std::array<std::string_view, 3>& in_sig_strings);

  /**
   * @brief Constructor to generate evaluation input signature from parameters
   * corresponding to each part
   * 
   * @param data_type_size 
   * @param eval_set_num 
   * @param use_templated_loop_iters 
   */
  InputSignature(
    std::optional<unsigned int> data_type_size,
    std::optional<unsigned int> eval_set_num,
    std::optional<bool> use_templated_loop_iters);

  /**
   * @brief Less than operator for comparing evaluation input signatures
   * so they can be ordered
   * Operator needed to support ordering since input signature
   * is used as std::map key and also for evaluation output order
   * If any InputSignature member is "no value" that property is considered
   * "any" and is ignored in the comparison
   * 
   * @param rhs 
   * @return true 
   * @return false 
   */
  bool operator<(const InputSignature& rhs) const;

  /**
   * @brief Equality operator for comparing evaluation input signatures
   *
   * 
   * @param rhs 
   * @return true 
   * @return false 
   */
  bool operator==(const InputSignature& rhs) const;

  /**
   * @brief Alternate "equal" operator where an attribute is considered "equal"
   * in cases where one side is "ANY" for the attribute as indicated
   * by "no value" for std::optional object
   * 
   * @param rhs 
   * @return true 
   * @return false 
   */
  bool EqualsUsingAny(const InputSignature& rhs) const;

  std::string DataTypeStr() const {
    if (!(data_type_size_.has_value())) {
      return "ANY";
    }
    if (data_type_size_.value() == 2) {
      return "HALF";
    }
    else if (data_type_size_.value() == 4) {
      return "FLOAT";
    }
    else if (data_type_size_.value() == 8) {
      return "DOUBLE";
    }
    return "UNKNOWN";
  }

  std::string EvalSetNumStr() const {
    if (!(eval_set_num_.has_value())) {
      return "ANY";
    }
    else {
      return std::to_string(eval_set_num_.value());
    }
  }

  unsigned int EvalSetNum() const {
    if (!(eval_set_num_.has_value())) {
      return std::numeric_limits<unsigned int>::max();
    }
    else {
      return eval_set_num_.value();
    }
  }

  std::string UseTemplatedLoopItersStr() const {
    if (!(use_templated_loop_iters_.has_value())) {
      return "BOTH";
    }
    return ((!(use_templated_loop_iters_.value())) ?
      std::string(run_eval::kBoolValFalseTrueDispStr[0]) :
      std::string(run_eval::kBoolValFalseTrueDispStr[1]));
  }
  
  /**
   * @brief Remove templated loop iter setting and change it to "any"
   * 
   */
  void RemoveTemplatedLoopIterSetting() {
    use_templated_loop_iters_.reset();
  }

  /**
   * @brief Remove data type setting and change it to "any"
   * 
   */
  void RemoveDatatypeSetting() {
    data_type_size_.reset();
  }

  std::optional<bool> TemplatedLoopIters() const {
    return use_templated_loop_iters_;
  }

  /**
   * @brief Overloaded << operator to write InputSignature object to stream
   * 
   * @param os 
   * @param eval_input_sig 
   * @return std::ostream& 
   */
  friend std::ostream& operator<<(std::ostream& os, const InputSignature& eval_input_sig);

private:
  std::optional<unsigned int> data_type_size_;
  std::optional<unsigned int> eval_set_num_;
  std::optional<bool> use_templated_loop_iters_;
};

//overloaded << operator to write InputSignature object to stream
inline std::ostream& operator<<(std::ostream& os, const InputSignature& eval_input_sig) {
  os << eval_input_sig.DataTypeStr() << " "  << eval_input_sig.EvalSetNumStr()
     << " " << eval_input_sig.UseTemplatedLoopItersStr();
  return os;
}

#endif //INPUT_SIGNATURE_H_