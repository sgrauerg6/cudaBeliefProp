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
 * @file SmoothImage.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#include "SmoothImage.h"

//create a Gaussian filter from a sigma value
std::vector<float> SmoothImage::MakeFilter(float sigma) const
{
  const float sigma_use{std::max(sigma, 0.01f)};
  const unsigned int size_filter{
    (unsigned int)std::ceil(sigma_use * kWidthSigma1) + 1u};
  std::vector<float> mask(size_filter);
  for (unsigned int i = 0; i < size_filter; i++) {
    mask[i] = std::exp(-0.5*((i/sigma_use) * (i/sigma_use)));
  }
  NormalizeFilter(mask);

  return mask;
}

//normalize filter mask so it integrates to one
void SmoothImage::NormalizeFilter(std::vector<float>& filter) const
{
  float sum{0.0};
  for (unsigned int i = 1; i < filter.size(); i++) {
    sum += std::abs(filter[i]);
  }
  sum = 2*sum + std::abs(filter[0]);
  for (unsigned int i = 0; i < filter.size(); i++) {
    filter[i] /= sum;
  }
}
