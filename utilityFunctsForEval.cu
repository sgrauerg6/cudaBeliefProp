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

//Defines utility functions used for a variety of evaluation functions

#include "utilityFunctsForEvalHeader.cuh"

//retrieve the 1-D index into the "Stereo"/"pixel" image given the current x and y in the image and the width and height of the Stereo
__host__ int retrieveIndexStereoOrPixelImage(int xVal, int yVal, int widthSpace, int heightSpace)
{
	return (yVal*widthSpace + xVal);
}

//return true if pixel if within the bounds of the given space
__host__ bool pixelWithinBounds(int xVal, int yVal, int widthSpace, int heightSpace)
{
	return ((xVal >= 0) && (xVal < widthSpace) && (yVal >= 0) && (yVal < heightSpace));
}

