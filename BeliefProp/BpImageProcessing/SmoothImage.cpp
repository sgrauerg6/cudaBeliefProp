/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

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

#include "SmoothImage.h"

//create a Gaussian filter from a sigma value
std::pair<std::unique_ptr<float[]>, unsigned int> SmoothImage::MakeFilter(float sigma)
{
  const float sigma_use{std::max(sigma, 0.01f)};
  const unsigned int size_filter{(unsigned int)std::ceil(sigma_use * kWidthSigma1) + 1u};
  std::unique_ptr<float[]> mask = std::make_unique<float[]>(size_filter);
  for (unsigned int i = 0; i < size_filter; i++) {
    mask[i] = std::exp(-0.5*((i/sigma_use) * (i/sigma_use)));
  }
  NormalizeFilter(mask, size_filter);

  return {std::move(mask), size_filter};
}

//normalize filter mask so it integrates to one
void SmoothImage::NormalizeFilter(const std::unique_ptr<float[]>& filter, unsigned int size_filter)
{
  float sum{0.0f};
  for (unsigned int i = 1; i < size_filter; i++) {
    sum += std::abs(filter[i]);
  }
  sum = 2*sum + std::abs(filter[0]);
  for (unsigned int i = 0; i < size_filter; i++) {
    filter[i] /= sum;
  }
}
