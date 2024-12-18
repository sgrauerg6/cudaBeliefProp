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
 * @file Float16Test.cpp
 * @author Scott Grauer-Gray
 * @brief File with main() function to test _Float16 datatype
 * 
 * @copyright Copyright (c) 2024
 */

#include <iostream>
#include <x86intrin.h>

/**
 * @brief main() function to test _Float16 data type
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char** argv)
{
//#define TEST_FLOAT61 
#ifdef TEST_FLOAT61
  _Float16 x1{1.0};
  _Float16 x2{3.0};
  _Float16 x3{5.0};
  _Float16 x4{7.0};

  _Float16 x_sum = x1 + x2 + x3 + x4;
  std::cout << "Sum: " << x_sum << std::endl;

  _Float16 x_mul = x1 * x3;
  std::cout << "Mul: " << x_mul << std::endl;

  _Float16 x_div = x4 / x3;
  std::cout << "Div: " << x_div << std::endl;

  _Float16 x_sub = x4 - x2;
  std::cout << "Sub: " << x_sub << std::endl;
#else
  std::cout << "No float16 test" << std::endl;
#endif //TEST_FLOAT61

  return 0;
}
