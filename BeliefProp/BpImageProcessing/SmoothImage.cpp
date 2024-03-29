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
std::pair<std::unique_ptr<float[]>, unsigned int> SmoothImage::makeFilter(const float sigma)
{
  const float sigmaUse{std::max(sigma, 0.01f)};
  const unsigned int sizeFilter{(unsigned int)std::ceil(sigmaUse * WIDTH_SIGMA_1) + 1u};
  std::unique_ptr<float[]> mask = std::make_unique<float[]>(sizeFilter);
  for (unsigned int i = 0; i < sizeFilter; i++) {
    mask[i] = std::exp(-0.5*((i/sigmaUse) * (i/sigmaUse)));
  }
  normalizeFilter(mask, sizeFilter);

  return {std::move(mask), sizeFilter};
}

//normalize filter mask so it integrates to one
void SmoothImage::normalizeFilter(const std::unique_ptr<float[]>& filter, const unsigned int sizeFilter)
{
  float sum{0.0f};
  for (unsigned int i = 1; i < sizeFilter; i++) {
    sum += std::abs(filter[i]);
  }
  sum = 2*sum + std::abs(filter[0]);
  for (unsigned int i = 0; i < sizeFilter; i++) {
    filter[i] /= sum;
  }
}
