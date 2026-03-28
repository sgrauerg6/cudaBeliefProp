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
 * @file InputSignature.h
 * @author Scott Grauer-Gray
 * @brief Declares class for defining input signature for evaluation run that
 * consists of evaluation set number, data type, and whether or not to use
 * templated loop iteration count.
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef INPUT_SIGNATURE_H_
#define INPUT_SIGNATURE_H_

#include <string_view>
#include <string>
#include <ostream>
#include <limits>
#include <optional>
#include <map>
#include <variant>
#include "RunEval/RunEvalConsts.h"
#include "RunEval/RunEvalEnumsStructs.h"

/**
 * @brief Class defines input signature for evaluation run that contains
 * evaluation set number, data type, and whether or not to use templated
 * loop iteration count
 */
class InputSignature {
public:
  /**
   * @brief Constructor to generate evaluation input signature from string array with
   * strings corresponding to each part
   * 
   * @param in_sig_strings 
   */
  explicit InputSignature(
    const std::array<std::string_view, 3>& in_sig_strings,
    const std::vector<std::pair<std::string, std::string>>& additional_sig_key_vals = {});

  /**
   * @brief Constructor to generate evaluation input signature from parameters
   * corresponding to each part
   * 
   * @param data_type_size 
   * @param eval_set_num 
   * @param use_templated_loop_iters 
   */
  explicit InputSignature(
    std::optional<size_t> data_type_size,
    std::optional<size_t> eval_set_num,
    std::optional<bool> use_templated_loop_iters,
    const std::vector<std::pair<std::string, std::string>>& additional_sig_key_vals = {});

  /**
   * @brief Less than operator for comparing evaluation input signatures
   * so they can be ordered<br>
   * Operator needed to support ordering since input signature
   * is used as std::map key and also for evaluation output order<br>
   * If any InputSignature member is "no value" that property is considered
   * "any" and is ignored in the comparison
   * 
   * @param rhs 
   * @return true if current input signature is less than rhs input signature,
   * false otherwise
   */
  bool operator<(const InputSignature& rhs) const;

  /**
   * @brief Equality operator for comparing evaluation input signatures
   * 
   * @param rhs 
   * @return true if current input signature is same as rhs input signature,
   * false otherwise
   */
  bool operator==(const InputSignature& rhs) const;

  /**
   * @brief Alternate "equal" operator where an attribute is considered "equal"
   * in cases where one side is "ANY" for the attribute as indicated
   * by "no value" for std::optional object
   * 
   * @param rhs 
   * @return true if current input signature is same as rhs input signature
   * where an attribute is considered "equal" in cases where one side is "ANY"
   * for the attribute
   */
  bool EqualsUsingAny(const InputSignature& rhs) const;

  std::string DataTypeStr() const {
    if (!data_type_size_) {
      return "ANY";
    }
    if (*data_type_size_ == 2) {
      return "HALF";
    }
    else if (*data_type_size_ == 4) {
      return "FLOAT";
    }
    else if (*data_type_size_ == 8) {
      return "DOUBLE";
    }
    return "UNKNOWN";
  }

  std::string EvalSetNumStr() const {
    if (!eval_set_num_) {
      return "ANY";
    }
    else {
      return std::to_string(*eval_set_num_);
    }
  }

  size_t EvalSetNum() const {
    if (!eval_set_num_) {
      return std::numeric_limits<size_t>::max();
    }
    else {
      return *eval_set_num_;
    }
  }

  std::string UseTemplatedLoopItersStr() const {
    if (!use_templated_loop_iters_) {
      return "BOTH";
    }
    return ((!(*use_templated_loop_iters_)) ?
      std::string(run_eval::kBoolValFalseTrueDispStr[0]) :
      std::string(run_eval::kBoolValFalseTrueDispStr[1]));
  }
  
  /**
   * @brief Remove templated loop iter setting and change it to "any"
   */
  void RemoveTemplatedLoopIterSetting() {
    use_templated_loop_iters_.reset();
  }

  /**
   * @brief Remove data type setting and change it to "any"
   */
  void RemoveDatatypeSetting() {
    data_type_size_.reset();
  }

  std::optional<bool> TemplatedLoopIters() const {
    return use_templated_loop_iters_;
  }

  /**
   * @brief Remove other characteristic setting and change it to "any"
   */
  void RemoveOtherCharSetting(const std::string& char_name) {
    other_characteristics_.at(char_name).reset();
  }

  void AddOtherCharacteristic(const std::string& char_name) {
    other_characteristics_.insert({char_name, {}});
  }

  void AddOtherCharacteristic(const std::string& char_name, int val) {
    other_characteristics_.insert({char_name, val});
  }

  void AddOtherCharacteristic(const std::string& char_name, bool val) {
    other_characteristics_.insert({char_name, val});
  }

  void AddOtherCharacteristic(const std::string& char_name, const std::string& val) {
    other_characteristics_.insert({char_name, val});    
  }

  /**
   * @brief Overloaded << operator to write InputSignature object to stream
   * 
   * @param os
   * @param eval_input_sig
   * @return Output stream with InputSignature object written to it
   */
  friend std::ostream& operator<<(
    std::ostream& os,
    const InputSignature& eval_input_sig);

private:
  std::optional<size_t> data_type_size_;
  std::optional<size_t> eval_set_num_;
  std::optional<bool> use_templated_loop_iters_;
  std::map<std::string, std::optional<std::variant<int, bool, std::string>>> other_characteristics_;
};

//overloaded << operator to write InputSignature object to stream
inline std::ostream& operator<<(
  std::ostream& os,
  const InputSignature& eval_input_sig)
{
  os << eval_input_sig.DataTypeStr() << " "  << eval_input_sig.EvalSetNumStr()
     << " " << eval_input_sig.UseTemplatedLoopItersStr();
  if (eval_input_sig.other_characteristics_.size() > 0) {
    for (const auto& [characteristic_name, characteristic_value] :
         eval_input_sig.other_characteristics_) {
      os << characteristic_name;
      if (characteristic_value) {
        std::visit([&os](auto&& arg) {
          os << arg << std::endl;
        }, *characteristic_value);
      }
    }
  }
  return os;
}

#endif //INPUT_SIGNATURE_H_